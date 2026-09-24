from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from hhtools.integrations import gvhmr, gvhmr_worker


def _runtime_tree(root: Path, body_models_root: Path) -> None:
    (root / "tools" / "demo").mkdir(parents=True)
    (root / "tools" / "demo" / "demo.py").touch()
    for relative in gvhmr._PUBLIC_CHECKPOINTS.values():  # noqa: SLF001
        checkpoint = root / "inputs" / "checkpoints" / relative
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()
    smplx = body_models_root / "smplx" / "SMPLX_NEUTRAL.npz"
    smplx.parent.mkdir(parents=True, exist_ok=True)
    smplx.touch()


def test_status_requires_licensed_smplx_model(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "GVHMR"
    body_models = tmp_path / "body-models"
    _runtime_tree(root, body_models)
    (body_models / "smplx" / "SMPLX_NEUTRAL.npz").unlink()
    monkeypatch.setattr(gvhmr.shutil, "which", lambda _command: "docker")
    monkeypatch.setattr(gvhmr, "_run_probe", lambda *_args, **_kwargs: (True, "ok"))

    status = gvhmr.gvhmr_status(gvhmr.GvhmrConfig(root=root, body_models_root=body_models))

    assert status["ready"] is False
    assert status["checks"]["smplx_neutral"] is False
    assert any("SMPL-X" in item for item in status["missing"])
    assert status["uses_official_weights"] is True
    assert status["supports_custom_weights"] is False
    assert status["training_enabled"] is False


def test_command_uses_isolated_gpu_container_and_no_training(tmp_path: Path) -> None:
    root = tmp_path / "GVHMR"
    body_models = tmp_path / "licensed-models"
    _runtime_tree(root, body_models)
    job_root = tmp_path / "job"
    job_root.mkdir()
    video = job_root / "source clip.mp4"
    video.touch()
    config = gvhmr.GvhmrConfig(
        root=root,
        body_models_root=body_models,
        image="hhtools-gvhmr:test",
        docker="docker",
        cuda_visible_devices="2",
    )

    command = gvhmr.build_gvhmr_command(
        config,
        video_path=video,
        job_root=job_root,
        static_cam=True,
    )

    assert command[:3] == ["docker", "run", "--rm"]
    assert command[command.index("--gpus") + 1] == "all"
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert command[-4:] == [
        "/work/source clip.mp4",
        "--output-root",
        "/work/output",
        "--static-cam",
    ]
    assert "hhtools-gvhmr:test" in command
    assert "CUDA_VISIBLE_DEVICES=2" in command
    assert not any("train" in argument.lower() for argument in command)
    assert any("target=/workspace/gvhmr,readonly" in argument for argument in command)
    assert any(
        "target=/workspace/gvhmr/inputs/checkpoints/body_models,readonly" in argument
        for argument in command
    )


def test_command_maps_the_posix_user_and_uses_a_writable_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "GVHMR"
    body_models = tmp_path / "body-models"
    _runtime_tree(root, body_models)
    job_root = tmp_path / "job"
    job_root.mkdir()
    video = job_root / "source.mp4"
    video.touch()
    monkeypatch.setattr(gvhmr, "_posix_container_identity", lambda: (1234, 5678))

    command = gvhmr.build_gvhmr_command(
        gvhmr.GvhmrConfig(root=root, body_models_root=body_models),
        video_path=video,
        job_root=job_root,
    )

    assert command[command.index("--user") + 1] == "1234:5678"
    assert command[command.index("--env") + 1] == "HOME=/work/.container-home"
    assert (job_root / ".container-home").is_dir()


def test_command_rejects_video_outside_job_root(tmp_path: Path) -> None:
    root = tmp_path / "GVHMR"
    body_models = tmp_path / "body-models"
    _runtime_tree(root, body_models)
    job_root = tmp_path / "job"
    job_root.mkdir()
    video = tmp_path / "outside.mp4"
    video.touch()

    with pytest.raises(ValueError):
        gvhmr.build_gvhmr_command(
            gvhmr.GvhmrConfig(root=root, body_models_root=body_models),
            video_path=video,
            job_root=job_root,
        )


def test_command_rejects_non_video_extension(tmp_path: Path) -> None:
    root = tmp_path / "GVHMR"
    body_models = tmp_path / "body-models"
    _runtime_tree(root, body_models)
    job_root = tmp_path / "job"
    job_root.mkdir()
    source = job_root / "payload.txt"
    source.touch()

    with pytest.raises(ValueError, match="unsupported video extension"):
        gvhmr.build_gvhmr_command(
            gvhmr.GvhmrConfig(root=root, body_models_root=body_models),
            video_path=source,
            job_root=job_root,
        )


def test_linux_environment_selects_the_installed_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "GVHMR"
    python = tmp_path / "gvhmr-env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(gvhmr.sys, "platform", "linux")
    monkeypatch.setenv(gvhmr.GVHMR_ROOT_ENV, str(root))
    monkeypatch.setenv(gvhmr.GVHMR_PYTHON_ENV, str(python))
    body_models = tmp_path / "bundled-body-models"
    monkeypatch.setenv(gvhmr.GVHMR_BODY_MODELS_ENV, str(body_models))

    config = gvhmr.GvhmrConfig.from_environment()

    assert config.runtime == "local"
    assert config.python_executable == python
    assert config.body_models_root == body_models


def test_windows_environment_keeps_the_docker_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gvhmr.sys, "platform", "win32")
    monkeypatch.setenv(gvhmr.GVHMR_ROOT_ENV, str(tmp_path / "GVHMR"))
    monkeypatch.setenv(gvhmr.GVHMR_PYTHON_ENV, str(tmp_path / "ignored-python"))
    monkeypatch.setattr(gvhmr.shutil, "which", lambda _command: "docker")

    config = gvhmr.GvhmrConfig.from_environment()

    assert config.runtime == "docker"
    assert config.python_executable is None
    assert config.docker == "docker"


def test_local_status_probes_python_and_cuda(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "GVHMR"
    body_models = root / "inputs" / "checkpoints" / "body_models"
    _runtime_tree(root, body_models)
    python = tmp_path / "gvhmr-env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(
        gvhmr,
        "_run_probe",
        lambda *_args, **_kwargs: (True, 'HHTOOLS_GVHMR_PROBE {"cuda": true}'),
    )
    monkeypatch.setattr(gvhmr.shutil, "which", lambda *_args, **_kwargs: "/usr/bin/ffmpeg")

    status = gvhmr.gvhmr_status(
        gvhmr.GvhmrConfig(
            root=root,
            body_models_root=body_models,
            runtime="local",
            python_executable=python,
        )
    )

    assert status["ready"] is True
    assert status["runtime"] == "local"
    assert status["python"] == str(python)
    assert status["checks"]["python_environment"] is True
    assert status["checks"]["cuda"] is True
    assert "docker_engine" not in status["checks"]


def test_local_command_preserves_virtual_environment_python_symlink(tmp_path: Path) -> None:
    root = tmp_path / "GVHMR"
    body_models = root / "inputs" / "checkpoints" / "body_models"
    _runtime_tree(root, body_models)
    base_python = tmp_path / "python-build" / "bin" / "python3.10"
    base_python.parent.mkdir(parents=True)
    base_python.touch()
    python = tmp_path / "gvhmr env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    try:
        python.symlink_to(base_python)
    except OSError:
        pytest.skip("file symlinks are not available on this host")
    job_root = tmp_path / "job"
    job_root.mkdir()
    video = job_root / "source clip.mp4"
    video.touch()

    command = gvhmr.build_gvhmr_command(
        gvhmr.GvhmrConfig(
            root=root,
            body_models_root=body_models,
            runtime="local",
            python_executable=python,
        ),
        video_path=video,
        job_root=job_root,
        static_cam=True,
        f_mm=35,
    )

    assert command[0] == os.path.abspath(python)
    assert command[0] != str(base_python.resolve())
    assert command[1].endswith("hhtools/integrations/gvhmr_worker.py")
    assert command[command.index("--video") + 1] == str(video.resolve())
    assert command[command.index("--output-root") + 1] == str(job_root / "output")
    assert command[command.index("--body-models-root") + 1] == str(body_models.resolve())
    assert command[-3:] == ["--static-cam", "--f-mm", "35"]


def test_worker_overlays_external_body_models_without_modifying_checkout(tmp_path: Path) -> None:
    root = tmp_path / "GVHMR"
    body_models = tmp_path / "bundled-body-models"
    _runtime_tree(root, body_models)
    (root / "hmr4d").mkdir()
    (root / "inputs" / "demo").mkdir()
    overlay = tmp_path / "project-overlay"

    project_root = gvhmr_worker._prepare_project_overlay(  # noqa: SLF001
        root,
        body_models,
        overlay,
    )

    overlaid_models = project_root / "inputs" / "checkpoints" / "body_models"
    assert overlaid_models.is_symlink()
    assert (overlaid_models / "smplx" / "SMPLX_NEUTRAL.npz").resolve() == (
        body_models / "smplx" / "SMPLX_NEUTRAL.npz"
    ).resolve()
    assert (project_root / "hmr4d").resolve() == (root / "hmr4d").resolve()
    overlaid_checkpoint = (
        project_root / "inputs" / "checkpoints" / "gvhmr" / "gvhmr_siga24_release.ckpt"
    )
    source_checkpoint = (
        root / "inputs" / "checkpoints" / "gvhmr" / "gvhmr_siga24_release.ckpt"
    )
    assert overlaid_checkpoint.resolve() == source_checkpoint.resolve()
    assert not (root / "inputs" / "checkpoints" / "body_models").exists()


def test_local_environment_drops_the_hhtools_python_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = tmp_path / "gvhmr-env" / "bin" / "python"
    monkeypatch.setenv("VIRTUAL_ENV", "/hhtools/.venv")
    monkeypatch.setenv("PYTHONHOME", "/hhtools/python")
    monkeypatch.setenv("PYTHONPATH", "/hhtools/source")
    config = gvhmr.GvhmrConfig(
        root=tmp_path / "GVHMR",
        body_models_root=tmp_path / "body-models",
        runtime="local",
        python_executable=python,
    )

    environment = gvhmr._local_environment(config)  # noqa: SLF001

    assert "VIRTUAL_ENV" not in environment
    assert "PYTHONHOME" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["PATH"].split(gvhmr.os.pathsep)[0] == str(python.parent)


def test_local_runtime_executes_the_external_process_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "GVHMR"
    body_models = root / "inputs" / "checkpoints" / "body_models"
    _runtime_tree(root, body_models)
    python = tmp_path / "gvhmr-env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f"""#!{sys.executable}
import json
import sys
from pathlib import Path

if '-c' in sys.argv:
    print('HHTOOLS_GVHMR_PROBE {{"cuda": true}}')
else:
    output = Path(sys.argv[sys.argv.index('--output-root') + 1]) / 'source'
    output.mkdir(parents=True, exist_ok=True)
    result = output / 'hmr4d_results.pt'
    result.write_bytes(b'motion')
    print('HHTOOLS_PROGRESS 0.5 running official checkpoint', flush=True)
    print('HHTOOLS_RESULT ' + json.dumps({{'result_path': str(result)}}), flush=True)
""",
        encoding="utf-8",
    )
    python.chmod(0o755)
    job_root = tmp_path / "job"
    job_root.mkdir()
    video = job_root / "source.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(gvhmr.shutil, "which", lambda *_args, **_kwargs: "/usr/bin/ffmpeg")
    progress: list[tuple[float, str]] = []

    result = gvhmr.run_gvhmr(
        video,
        job_root,
        config=gvhmr.GvhmrConfig(
            root=root,
            body_models_root=body_models,
            runtime="local",
            python_executable=python,
            timeout_seconds=10,
        ),
        progress=lambda value, message: progress.append((value, message)),
    )

    assert result == (job_root / "output" / "source" / "hmr4d_results.pt").resolve()
    assert result.read_bytes() == b"motion"
    assert progress[0] == (0.5, "running official checkpoint")
    assert progress[-1] == (1.0, "GVHMR motion ready")


def test_result_path_is_confined_to_the_job_output(tmp_path: Path) -> None:
    job_root = tmp_path / "job"

    result = gvhmr._host_result_path(  # noqa: SLF001
        job_root,
        "/work/output/video/hmr4d_results.pt",
    )

    assert result == (job_root / "output" / "video" / "hmr4d_results.pt").resolve()


def test_local_result_path_is_confined_to_the_job_output(tmp_path: Path) -> None:
    job_root = tmp_path / "job"
    result = job_root / "output" / "video" / "hmr4d_results.pt"

    resolved = gvhmr._local_result_path(job_root, str(result))  # noqa: SLF001

    assert resolved == result.resolve()
    with pytest.raises(RuntimeError):
        gvhmr._local_result_path(  # noqa: SLF001
            job_root,
            str(tmp_path / "outside" / "hmr4d_results.pt"),
        )


@pytest.mark.parametrize(
    "published",
    [
        "/work/output/../../outside/hmr4d_results.pt",
        "/work/not-output/hmr4d_results.pt",
        "/work/output/video/arbitrary.pt",
    ],
)
def test_result_path_rejects_worker_traversal(tmp_path: Path, published: str) -> None:
    with pytest.raises(RuntimeError):
        gvhmr._host_result_path(tmp_path / "job", published)  # noqa: SLF001


def test_result_path_rejects_symlinked_output_root(tmp_path: Path) -> None:
    job_root = tmp_path / "job"
    outside = tmp_path / "outside"
    job_root.mkdir()
    outside.mkdir()
    try:
        (job_root / "output").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are not available on this host")

    with pytest.raises(RuntimeError, match="must not be a symlink"):
        gvhmr._host_result_path(  # noqa: SLF001
            job_root,
            "/work/output/video/hmr4d_results.pt",
        )
