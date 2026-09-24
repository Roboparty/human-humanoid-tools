"""Isolated GVHMR video-to-motion runtime.

Linux runs the user's installed GVHMR Python environment. Windows keeps the
optional Docker runtime. Neither backend imports GVHMR into the hhtools process.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

GVHMR_ROOT_ENV = "HHTOOLS_GVHMR_ROOT"
GVHMR_IMAGE_ENV = "HHTOOLS_GVHMR_IMAGE"
GVHMR_BODY_MODELS_ENV = "HHTOOLS_GVHMR_BODY_MODELS"
GVHMR_PYTHON_ENV = "HHTOOLS_GVHMR_PYTHON"
GVHMR_TIMEOUT_ENV = "HHTOOLS_GVHMR_TIMEOUT_SECONDS"

DEFAULT_IMAGE = "hhtools-gvhmr:cu128"
DEFAULT_TIMEOUT_SECONDS = 2 * 60 * 60

_PUBLIC_CHECKPOINTS = {
    "GVHMR": Path("gvhmr/gvhmr_siga24_release.ckpt"),
    "HMR2": Path("hmr2/epoch=10-step=25000.ckpt"),
    "ViTPose": Path("vitpose/vitpose-h-multi-coco.pth"),
    "YOLOv8": Path("yolo/yolov8x.pt"),
}
_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"})
_PROGRESS_RE = re.compile(r"^HHTOOLS_PROGRESS\s+([0-9.]+)\s+(.*)$")
_RESULT_PREFIX = "HHTOOLS_RESULT "


def _posix_container_identity() -> tuple[int, int] | None:
    """Return the host identity used for writable bind mounts on Linux."""

    if os.name != "posix" or not hasattr(os, "getuid") or not hasattr(os, "getgid"):
        return None
    return os.getuid(), os.getgid()


def _docker_isolation_args(*, home: str) -> list[str]:
    """Build the security and host-user options shared by GVHMR containers."""

    arguments = [
        "--gpus",
        "all",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
    ]
    identity = _posix_container_identity()
    if identity is not None:
        uid, gid = identity
        arguments.extend(["--user", f"{uid}:{gid}", "--env", f"HOME={home}"])
    return arguments


@dataclass(frozen=True)
class GvhmrConfig:
    root: Path
    body_models_root: Path
    image: str = DEFAULT_IMAGE
    docker: str = "docker"
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    cuda_visible_devices: str | None = None
    runtime: Literal["local", "docker"] = "docker"
    python_executable: Path | None = None

    @classmethod
    def from_environment(cls) -> GvhmrConfig:
        root = _default_root()
        runtime: Literal["local", "docker"] = (
            "local" if sys.platform.startswith("linux") else "docker"
        )
        default_body_models = root / "inputs" / "checkpoints" / "body_models"
        body_models_value = os.environ.get(GVHMR_BODY_MODELS_ENV, default_body_models)
        body_models = Path(body_models_value).expanduser()
        raw_timeout = os.environ.get(GVHMR_TIMEOUT_ENV, str(DEFAULT_TIMEOUT_SECONDS))
        try:
            timeout = max(60, int(raw_timeout))
        except ValueError:
            timeout = DEFAULT_TIMEOUT_SECONDS
        return cls(
            root=root,
            body_models_root=body_models,
            image=os.environ.get(GVHMR_IMAGE_ENV, DEFAULT_IMAGE),
            docker=shutil.which("docker") or "docker",
            timeout_seconds=timeout,
            cuda_visible_devices=(os.environ.get("CUDA_VISIBLE_DEVICES") or None),
            runtime=runtime,
            python_executable=_default_python(root) if runtime == "local" else None,
        )


def _default_root() -> Path:
    override = os.environ.get(GVHMR_ROOT_ENV)
    if override:
        return Path(override).expanduser()
    windows_default = Path("C:/GVHMR")
    if windows_default.is_dir():
        return windows_default
    return Path.home() / "GVHMR"


def _default_python(root: Path) -> Path | None:
    override = os.environ.get(GVHMR_PYTHON_ENV)
    if override:
        return Path(override).expanduser()

    home = Path.home()
    candidates = (
        root / ".venv" / "bin" / "python",
        root / "venv" / "bin" / "python",
        home / ".conda" / "envs" / "gvhmr" / "bin" / "python",
        home / "anaconda3" / "envs" / "gvhmr" / "bin" / "python",
        home / "miniconda3" / "envs" / "gvhmr" / "bin" / "python",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _local_environment(config: GvhmrConfig) -> dict[str, str]:
    """Build an environment owned by the external GVHMR interpreter."""

    environment = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    if config.python_executable is not None:
        current_path = environment.get("PATH", "")
        environment["PATH"] = os.pathsep.join(
            part
            for part in (str(Path(_local_python_command(config)).parent), current_path)
            if part
        )
    if config.cuda_visible_devices:
        environment["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    return environment


def _local_python_command(config: GvhmrConfig) -> str:
    """Return an absolute interpreter path without dereferencing a venv symlink."""

    python = config.python_executable
    if python is None:
        raise RuntimeError(f"{GVHMR_PYTHON_ENV} is not configured")
    # ``Path.resolve()`` follows ``.venv/bin/python`` to the base uv/CPython
    # executable and therefore bypasses that environment's site-packages.
    return os.path.abspath(os.fspath(python.expanduser()))


def _run_probe(
    args: list[str],
    *,
    timeout: float = 10.0,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        return False, str(err)
    output = (completed.stdout or completed.stderr or "").strip()
    return completed.returncode == 0, output


def gvhmr_status(
    config: GvhmrConfig | None = None,
    *,
    probe_runtime: bool = True,
) -> dict[str, Any]:
    """Return actionable readiness without importing GVHMR in this process.

    ``probe_runtime=False`` limits the check to configuration and filesystem
    facts, avoiding external Python, CUDA, and Docker subprocesses.
    """

    cfg = config or GvhmrConfig.from_environment()
    checkpoint_root = cfg.root / "inputs" / "checkpoints"
    checks: dict[str, bool] = {
        "official_repo": (cfg.root / "tools" / "demo" / "demo.py").is_file(),
    }
    missing: list[str] = []
    if not checks["official_repo"]:
        missing.append(f"GVHMR official checkout: {cfg.root}")

    for label, relative in _PUBLIC_CHECKPOINTS.items():
        checkpoint_path = checkpoint_root / relative
        available = checkpoint_path.is_file()
        checks[f"checkpoint_{label.lower()}"] = available
        if not available:
            missing.append(f"{label} checkpoint: {checkpoint_path}")

    smplx = cfg.body_models_root / "smplx" / "SMPLX_NEUTRAL.npz"
    checks["smplx_neutral"] = smplx.is_file()
    if not checks["smplx_neutral"]:
        missing.append(
            "licensed SMPL-X neutral model: "
            f"{smplx} (download after accepting the official SMPL-X license)"
        )

    if cfg.runtime == "local":
        python = cfg.python_executable
        python_command = _local_python_command(cfg) if python is not None else None
        local_environment = _local_environment(cfg)
        checks["python_executable"] = (
            python_command is not None and Path(python_command).is_file()
        )
        checks["ffmpeg"] = shutil.which("ffmpeg", path=local_environment.get("PATH")) is not None
        if not checks["ffmpeg"]:
            missing.append("ffmpeg executable in the GVHMR environment PATH")
        if not checks["python_executable"]:
            missing.append(
                "GVHMR Python executable: set "
                f"{GVHMR_PYTHON_ENV} to the installed environment's Python"
            )
        environment_ready = False
        cuda_ready = False
        if probe_runtime and checks["python_executable"] and checks["official_repo"]:
            probe = (
                "import json, cv2, hydra, hmr4d, pytorch3d, torch; "
                "print('HHTOOLS_GVHMR_PROBE ' + "
                "json.dumps({'cuda': bool(torch.cuda.is_available())}))"
            )
            environment_ready, output = _run_probe(
                [python_command, "-c", probe],
                timeout=30,
                cwd=cfg.root,
                env=local_environment,
            )
            if environment_ready:
                try:
                    payload = next(
                        line.removeprefix("HHTOOLS_GVHMR_PROBE ")
                        for line in reversed(output.splitlines())
                        if line.startswith("HHTOOLS_GVHMR_PROBE ")
                    )
                    cuda_ready = bool(json.loads(payload)["cuda"])
                except (KeyError, StopIteration, TypeError, ValueError):
                    environment_ready = False
        if probe_runtime:
            checks["python_environment"] = environment_ready
            checks["cuda"] = cuda_ready
            if checks["python_executable"] and not environment_ready:
                missing.append("GVHMR Python cannot import the installed inference dependencies")
            if environment_ready and not cuda_ready:
                missing.append("CUDA is not available in the GVHMR Python environment")
    else:
        checks["docker_cli"] = shutil.which(cfg.docker) is not None or Path(cfg.docker).is_file()
        if not probe_runtime and not checks["docker_cli"]:
            missing.append("Docker executable for the GVHMR runtime")
        if probe_runtime:
            docker_ready = False
            image_ready = False
            if checks["docker_cli"]:
                docker_ready, _ = _run_probe(
                    [cfg.docker, "version", "--format", "{{.Server.Version}}"],
                )
                if docker_ready:
                    image_ready, _ = _run_probe(
                        [cfg.docker, "image", "inspect", cfg.image, "--format", "{{.Id}}"],
                    )
            checks["docker_engine"] = docker_ready
            checks["runtime_image"] = image_ready
            if not docker_ready:
                missing.append("running Docker engine")
            elif not image_ready:
                missing.append(f"GVHMR runtime image: {cfg.image}")

    return {
        "ready": all(checks.values()),
        "checks": checks,
        "missing": missing,
        "root": str(cfg.root),
        "body_models_root": str(cfg.body_models_root),
        "image": cfg.image,
        "python": (
            _local_python_command(cfg)
            if cfg.runtime == "local" and cfg.python_executable is not None
            else None
        ),
        "runtime": cfg.runtime,
        "cuda_visible_devices": cfg.cuda_visible_devices,
        "uses_official_weights": True,
        "supports_custom_weights": False,
        "training_enabled": False,
    }


def ensure_gvhmr_ready(config: GvhmrConfig | None = None) -> GvhmrConfig:
    cfg = config or GvhmrConfig.from_environment()
    status = gvhmr_status(cfg)
    if not status["ready"]:
        details = "\n- ".join(status["missing"])
        raise RuntimeError(f"GVHMR is not ready:\n- {details}")
    return cfg


def _container_path(host_path: Path, mount_root: Path, container_root: str) -> str:
    relative = host_path.resolve().relative_to(mount_root.resolve())
    suffix = "/".join(relative.parts)
    return f"{container_root}/{suffix}" if suffix else container_root


def _host_result_path(job_root: Path, container_path: str) -> Path:
    """Resolve a worker result without allowing traversal or output symlinks."""

    container_result = PurePosixPath(container_path)
    try:
        relative = container_result.relative_to(PurePosixPath("/work/output"))
    except ValueError as err:
        raise RuntimeError("GVHMR published a result outside /work/output") from err
    if relative.name != "hmr4d_results.pt" or ".." in relative.parts:
        raise RuntimeError("GVHMR published an invalid result path")
    work_root = job_root.resolve()
    raw_output_root = work_root / "output"
    if raw_output_root.is_symlink():
        raise RuntimeError("GVHMR job output directory must not be a symlink")
    output_root = raw_output_root.resolve()
    try:
        output_root.relative_to(work_root)
    except ValueError as err:
        raise RuntimeError("GVHMR output directory resolves outside the job") from err
    result = (output_root / Path(*relative.parts)).resolve()
    try:
        result.relative_to(output_root)
    except ValueError as err:
        raise RuntimeError("GVHMR result resolves outside the job output directory") from err
    return result


def _local_result_path(job_root: Path, published_path: str) -> Path:
    """Resolve a native worker result inside this job's output directory."""

    work_root = job_root.resolve()
    raw_output_root = work_root / "output"
    if raw_output_root.is_symlink():
        raise RuntimeError("GVHMR job output directory must not be a symlink")
    output_root = raw_output_root.resolve()
    try:
        output_root.relative_to(work_root)
    except ValueError as err:
        raise RuntimeError("GVHMR output directory resolves outside the job") from err

    published = Path(published_path)
    result = published.resolve() if published.is_absolute() else (output_root / published).resolve()
    if result.name != "hmr4d_results.pt":
        raise RuntimeError("GVHMR published an invalid result path")
    try:
        result.relative_to(output_root)
    except ValueError as err:
        raise RuntimeError("GVHMR result resolves outside the job output directory") from err
    return result


def _build_docker_gvhmr_command(
    config: GvhmrConfig,
    *,
    video_path: Path,
    job_root: Path,
    static_cam: bool = True,
    f_mm: int | None = None,
) -> list[str]:
    """Build the Docker argv for one isolated inference job."""

    root = config.root.resolve()
    body_models = config.body_models_root.resolve()
    video = video_path.resolve()
    work = job_root.resolve()
    video.relative_to(work)
    if video.suffix.lower() not in _VIDEO_SUFFIXES:
        raise ValueError(f"unsupported video extension: {video.suffix or '<none>'}")
    output = work / "output"
    output.mkdir(parents=True, exist_ok=True)
    isolation_args = _docker_isolation_args(home="/work/.container-home")
    if "--user" in isolation_args:
        (work / ".container-home").mkdir(exist_ok=True)
    command = [
        config.docker,
        "run",
        "--rm",
        *isolation_args,
        "--name",
        f"hhtools-gvhmr-{uuid.uuid4().hex[:10]}",
        "--mount",
        f"type=bind,source={root},target=/workspace/gvhmr,readonly",
        "--mount",
        f"type=bind,source={work},target=/work",
    ]
    if config.cuda_visible_devices:
        command.extend(["--env", f"CUDA_VISIBLE_DEVICES={config.cuda_visible_devices}"])
    default_body_models = (root / "inputs" / "checkpoints" / "body_models").resolve()
    if body_models != default_body_models:
        command.extend(
            [
                "--mount",
                (
                    "type=bind,"
                    f"source={body_models},"
                    "target=/workspace/gvhmr/inputs/checkpoints/body_models,readonly"
                ),
            ]
        )
    command.extend(
        [
            config.image,
            "--video",
            _container_path(video, work, "/work"),
            "--output-root",
            "/work/output",
        ]
    )
    if static_cam:
        command.append("--static-cam")
    if f_mm is not None:
        if f_mm <= 0:
            raise ValueError("f_mm must be positive")
        command.extend(["--f-mm", str(f_mm)])
    return command


def _build_local_gvhmr_command(
    config: GvhmrConfig,
    *,
    video_path: Path,
    job_root: Path,
    static_cam: bool = True,
    f_mm: int | None = None,
) -> list[str]:
    python_command = _local_python_command(config)
    video = video_path.resolve()
    work = job_root.resolve()
    video.relative_to(work)
    if video.suffix.lower() not in _VIDEO_SUFFIXES:
        raise ValueError(f"unsupported video extension: {video.suffix or '<none>'}")
    output = work / "output"
    output.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("gvhmr_worker.py").resolve()
    if not worker.is_file():
        raise RuntimeError(f"GVHMR worker is missing: {worker}")
    command = [
        python_command,
        str(worker),
        "--video",
        str(video),
        "--output-root",
        str(output),
        "--body-models-root",
        str(config.body_models_root.resolve()),
    ]
    if static_cam:
        command.append("--static-cam")
    if f_mm is not None:
        if f_mm <= 0:
            raise ValueError("f_mm must be positive")
        command.extend(["--f-mm", str(f_mm)])
    return command


def build_gvhmr_command(
    config: GvhmrConfig,
    *,
    video_path: Path,
    job_root: Path,
    static_cam: bool = True,
    f_mm: int | None = None,
) -> list[str]:
    """Build the argv for the platform-selected isolated GVHMR runtime."""

    builder = (
        _build_local_gvhmr_command if config.runtime == "local" else _build_docker_gvhmr_command
    )
    return builder(
        config,
        video_path=video_path,
        job_root=job_root,
        static_cam=static_cam,
        f_mm=f_mm,
    )


def _stop_runtime_process(
    process: subprocess.Popen[str],
    config: GvhmrConfig,
    command: list[str],
) -> None:
    if process.poll() is None:
        if config.runtime == "local" and os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
    process.wait(timeout=30)
    if config.runtime != "docker":
        return
    try:
        container_name = command[command.index("--name") + 1]
        _run_probe([config.docker, "rm", "--force", container_name], timeout=30)
    except (ValueError, IndexError):
        pass


def run_gvhmr(
    video_path: Path,
    job_root: Path,
    *,
    static_cam: bool = True,
    f_mm: int | None = None,
    config: GvhmrConfig | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> Path:
    """Run GVHMR and return the generated ``hmr4d_results.pt`` path."""

    cfg = ensure_gvhmr_ready(config)
    command = build_gvhmr_command(
        cfg,
        video_path=video_path,
        job_root=job_root,
        static_cam=static_cam,
        f_mm=f_mm,
    )
    local = cfg.runtime == "local"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cfg.root if local else None,
        env=_local_environment(cfg) if local else None,
        start_new_session=local and os.name == "posix",
    )
    published_result_path: str | None = None
    recent: list[str] = []
    output_lines: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        try:
            for raw_line in process.stdout:
                output_lines.put(raw_line)
        finally:
            output_lines.put(None)

    reader = threading.Thread(target=read_output, name="gvhmr-output", daemon=True)
    reader.start()
    deadline = time.monotonic() + cfg.timeout_seconds
    timed_out = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                raw = output_lines.get(timeout=min(0.5, remaining))
            except queue.Empty:
                if process.poll() is not None and not reader.is_alive():
                    break
                continue
            if raw is None:
                break
            line = raw.strip()
            if line:
                recent.append(line)
                recent = recent[-30:]
            match = _PROGRESS_RE.match(line)
            if match and progress is not None:
                progress(float(match.group(1)), match.group(2))
            elif line.startswith(_RESULT_PREFIX):
                payload = json.loads(line[len(_RESULT_PREFIX) :])
                published_result_path = str(payload["result_path"])
        if timed_out:
            raise TimeoutError(f"GVHMR exceeded the {cfg.timeout_seconds}-second inference timeout")
        return_code = process.wait(timeout=max(1.0, deadline - time.monotonic()))
    except BaseException:
        _stop_runtime_process(process, cfg, command)
        raise
    if return_code != 0:
        diagnostic = "\n".join(recent[-12:])
        raise RuntimeError(
            f"GVHMR {cfg.runtime} runtime exited with code {return_code}.\n{diagnostic}"
        )
    if not published_result_path:
        raise RuntimeError("GVHMR completed without publishing a result path")
    result = (
        _local_result_path(job_root, published_result_path)
        if local
        else _host_result_path(job_root, published_result_path)
    )
    if not result.is_file():
        raise RuntimeError(f"GVHMR result was not found on the host: {result}")
    if progress is not None:
        progress(1.0, "GVHMR motion ready")
    return result


__all__ = [
    "DEFAULT_IMAGE",
    "GVHMR_BODY_MODELS_ENV",
    "GVHMR_IMAGE_ENV",
    "GVHMR_PYTHON_ENV",
    "GVHMR_ROOT_ENV",
    "GvhmrConfig",
    "build_gvhmr_command",
    "ensure_gvhmr_ready",
    "gvhmr_status",
    "run_gvhmr",
]
