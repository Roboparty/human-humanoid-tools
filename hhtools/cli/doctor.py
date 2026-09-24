"""Side-effect-free readiness checks for the local HHTools installation."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import typer

from hhtools._version import __version__


class Requirement(StrEnum):
    WEB = "web"
    ROBOT = "robot"
    RETARGET = "retarget"
    MCP = "mcp"
    BODYMODELS = "bodymodels"
    GVHMR = "gvhmr"


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ready: bool
    summary: str
    details: dict[str, Any]


@dataclass(frozen=True)
class DoctorReport:
    schema_version: Literal["1.0"]
    kind: Literal["hhtools_doctor"]
    hhtools_version: str
    base_ready: bool
    requested: list[str]
    ready: bool
    checks: list[dict[str, Any]]


app = typer.Typer(help="Check local runtime readiness without running jobs.")

_BASE_MODULES = (
    "numpy",
    "scipy",
    "typer",
    "rich",
    "yaml",
    "pydantic",
    "trimesh",
    "pandas",
    "pyarrow",
    "sklearn",
    "tqdm",
    "platformdirs",
)
_MCP_RUNTIME_MODULES = (
    "mcp",
    "fastapi",
    "uvicorn",
    "multipart",
    "yourdfpy",
    "mujoco",
    "smplx",
    "torch",
    "chumpy",
    "pygltflib",
    "osqp",
    "warp",
    "newton",
)


def _available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _availability(modules: tuple[str, ...]) -> dict[str, bool]:
    return {module: _available(module) for module in modules}


def _check_base() -> DoctorCheck:
    modules = _availability(_BASE_MODULES)
    python_ready = sys.version_info >= (3, 12)
    ready = python_ready and all(modules.values())
    return DoctorCheck(
        "base",
        ready,
        "Python and required package imports are available"
        if ready
        else "base runtime is incomplete",
        {
            "python": ".".join(str(part) for part in sys.version_info[:3]),
            "python_minimum": "3.12",
            "python_ready": python_ready,
            "modules": modules,
        },
    )


def _check_web() -> DoctorCheck:
    modules = _availability(("fastapi", "uvicorn", "multipart"))
    missing = [module for module, available in modules.items() if not available]
    return DoctorCheck(
        "web",
        not missing,
        "Web runtime imports are available" if not missing else "Web runtime imports are missing",
        {"modules": modules, "missing": missing},
    )


def _check_robot() -> DoctorCheck:
    try:
        from hhtools.robot.registry import list_presets_readonly

        presets = list_presets_readonly()
        ready_count = sum(
            bool(preset.has_urdf and preset.ik_map and preset.dof_order) for preset in presets
        )
        ready = ready_count > 0
        details = {"catalog_count": len(presets), "ready_count": ready_count}
        summary = f"{ready_count} of {len(presets)} robot presets are ready"
    except Exception as exception:  # noqa: BLE001 - diagnostics must continue
        ready = False
        details = {"catalog_count": 0, "ready_count": 0, "error": type(exception).__name__}
        summary = "robot catalog could not be read"
    return DoctorCheck("robot", ready, summary, details)


def _check_retarget() -> DoctorCheck:
    modules = _availability(("newton", "warp", "mujoco", "yourdfpy"))
    ready = all(modules.values())
    return DoctorCheck(
        "retarget",
        ready,
        "Newton and Warp runtime imports are available"
        if ready
        else "retarget imports are missing",
        {"modules": modules},
    )


def _check_mcp() -> DoctorCheck:
    modules = _availability(_MCP_RUNTIME_MODULES)
    entries = _availability(("hhtools.mcp.runtime", "hhtools.mcp.server"))
    ready = all(modules.values()) and all(entries.values())
    return DoctorCheck(
        "mcp",
        ready,
        "MCP SDK, entry point, and runtime imports are available"
        if ready
        else "MCP runtime imports are missing",
        {"modules": modules, "entries": entries},
    )


def _check_bodymodels() -> DoctorCheck:
    from hhtools.bodymodels.paths import check_body_models

    families = check_body_models()
    ready = bool(families) and all(families.values())
    return DoctorCheck(
        "bodymodels",
        ready,
        "all SMPL-family weights are available" if ready else "SMPL-family weights are missing",
        {"families": families},
    )


def _check_gvhmr() -> DoctorCheck:
    root = Path(os.environ.get("HHTOOLS_GVHMR_ROOT", Path.home() / "GVHMR")).expanduser()
    body_models = Path(
        os.environ.get(
            "HHTOOLS_GVHMR_BODY_MODELS",
            root / "inputs" / "checkpoints" / "body_models",
        )
    ).expanduser()
    checkpoint_root = root / "inputs" / "checkpoints"
    checks = {
        "official_repo": (root / "tools" / "demo" / "demo.py").is_file(),
        "checkpoint_gvhmr": (checkpoint_root / "gvhmr/gvhmr_siga24_release.ckpt").is_file(),
        "checkpoint_hmr2": (checkpoint_root / "hmr2/epoch=10-step=25000.ckpt").is_file(),
        "checkpoint_vitpose": (checkpoint_root / "vitpose/vitpose-h-multi-coco.pth").is_file(),
        "checkpoint_yolov8": (checkpoint_root / "yolo/yolov8x.pt").is_file(),
        "smplx_neutral": (body_models / "smplx" / "SMPLX_NEUTRAL.npz").is_file(),
    }
    runtime = "local" if sys.platform.startswith("linux") else "docker"
    if runtime == "local":
        configured_python = os.environ.get("HHTOOLS_GVHMR_PYTHON")
        python_candidates = (
            Path(configured_python).expanduser() if configured_python else None,
            root / ".venv" / "bin" / "python",
            root / "venv" / "bin" / "python",
            Path.home() / ".conda" / "envs" / "gvhmr" / "bin" / "python",
            Path.home() / "anaconda3" / "envs" / "gvhmr" / "bin" / "python",
            Path.home() / "miniconda3" / "envs" / "gvhmr" / "bin" / "python",
        )
        python = next(
            (
                candidate
                for candidate in python_candidates
                if candidate is not None and candidate.is_file()
            ),
            None,
        )
        checks["python_executable"] = python is not None
        runtime_path = os.pathsep.join(
            part
            for part in (
                str(python.parent) if python is not None else "",
                os.environ.get("PATH", ""),
            )
            if part
        )
        checks["ffmpeg"] = shutil.which("ffmpeg", path=runtime_path) is not None
    else:
        checks["docker_cli"] = shutil.which("docker") is not None
    missing = [name for name, available in checks.items() if not available]
    ready = not missing
    return DoctorCheck(
        "gvhmr",
        ready,
        "GVHMR configuration is ready" if ready else "GVHMR configuration is incomplete",
        {
            "probe_scope": "configuration_only",
            "runtime": runtime,
            "checks": checks,
            "missing": missing,
        },
    )


def build_report(required: list[Requirement]) -> DoctorReport:
    checks = [
        _check_base(),
        _check_web(),
        _check_robot(),
        _check_retarget(),
        _check_mcp(),
        _check_bodymodels(),
        _check_gvhmr(),
    ]
    requested = sorted({item.value for item in required})
    readiness = {check.name: check.ready for check in checks}
    base_ready = readiness["base"]
    ready = base_ready and all(readiness[name] for name in requested)
    return DoctorReport(
        schema_version="1.0",
        kind="hhtools_doctor",
        hhtools_version=__version__,
        base_ready=base_ready,
        requested=requested,
        ready=ready,
        checks=[asdict(check) for check in checks],
    )


def _human_output(report: DoctorReport) -> str:
    requested = set(report.requested)
    lines = [f"hhtools doctor {report.hhtools_version}"]
    for check in report.checks:
        required = check["name"] == "base" or check["name"] in requested
        status = "OK" if check["ready"] else ("FAIL" if required else "unavailable")
        role = "required" if required else "optional"
        lines.append(f"[{status}] {check['name']} ({role}): {check['summary']}")
    lines.append("Result: ready" if report.ready else "Result: required runtime is not ready")
    return "\n".join(lines)


@app.callback(invoke_without_command=True)
def doctor(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Write one stable JSON document."),
    require: list[Requirement] = typer.Option(
        [],
        "--require",
        help="Require an optional group; repeat for multiple groups.",
    ),
) -> None:
    """Inspect local readiness without network access, jobs, or solver initialization."""

    if ctx.invoked_subcommand is not None:
        return
    report = build_report(require)
    if json_output:
        typer.echo(json.dumps(asdict(report), ensure_ascii=False, separators=(",", ":")))
    else:
        typer.echo(_human_output(report))
    if not report.ready:
        raise typer.Exit(code=1)


__all__ = ["DoctorCheck", "DoctorReport", "Requirement", "app", "build_report"]
