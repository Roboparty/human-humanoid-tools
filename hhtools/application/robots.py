"""Robot-library policy, IK prewarming, and scaffold reference preservation."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from .state import Job, SessionState

_log = logging.getLogger(__name__)

# Product-curated models remain read-only even when a distribution provisions
# them below the same per-user root as uploaded models. Keep this list aligned
# with the front-end Robot Library catalog.
_BUILTIN_ROBOT_PRESET_NAMES = frozenset(
    {
        "g1_29dof",
        "roboto_origin",
        "agibot_x2_ultra",
        "asimov_1",
        "fourier_gr2",
        "berkeley_humanoid_lite",
    }
)


def _is_builtin_robot_preset(name: str) -> bool:
    return str(name).strip().lower() in _BUILTIN_ROBOT_PRESET_NAMES


def _start_robot_prewarm(state: SessionState, model: Any, name: str) -> None:
    """Background-compile Warp IK kernels after a robot loads."""

    def _run() -> None:
        try:
            _require_newton_package()
            from hhtools.retarget.newton_basic._warp_config import configure as configure_warp_cache
            from hhtools.retarget.newton_basic.pipeline import NewtonBasicPipeline

            configure_warp_cache()
            NewtonBasicPipeline.prewarm_for_robot(model)
        except Exception:  # noqa: BLE001 - optional GPU / missing newton
            _log.debug("background IK prewarm failed for %r", name, exc_info=True)

    prev = state.robot_prewarm_threads.get(name)
    if isinstance(prev, threading.Thread) and prev.is_alive():
        return
    thread = threading.Thread(
        target=_run,
        name=f"hhtools-web-prewarm-{name}",
        daemon=True,
    )
    state.robot_prewarm_threads[name] = thread
    thread.start()


def _join_robot_prewarm(state: SessionState, robot_name: str, job: Job | None) -> None:
    """Wait for background prewarm before the first retarget solve."""
    thread = state.robot_prewarm_threads.get(robot_name)
    if not isinstance(thread, threading.Thread) or not thread.is_alive():
        return
    if job is not None:
        job.progress = max(job.progress, 0.03)
        job.message = "正在预热 IK 内核（新机器人首次 retarget 较慢，请稍候）…"
    thread.join(timeout=180.0)


def _require_newton_package() -> None:
    """Raise a clear error when the optional NVIDIA ``newton`` wheel is missing."""
    try:
        import newton  # noqa: F401
    except ModuleNotFoundError as err:
        raise ValueError(
            "未安装 newton（Newton IK 依赖）。请先安装 retarget 额外依赖：\n"
            "  uv sync --extra web --extra retarget\n"
            "并按 NVIDIA / SOMA-Retargeter 文档安装 newton 包；"
            "仅预览 AMASS/parc_ms 动作不需要 newton，但 Retarget 与部分缩放预览需要。"
        ) from err


def _read_yaml_retarget_references(drop: Path) -> dict | None:
    """Extract ``retarget.references`` from a robot dir's yaml (pre-rebuild)."""
    import yaml

    for yp in sorted(drop.glob("*.yaml")):
        if yp.name.startswith("retarget_calibration_"):
            continue
        try:
            data = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            continue
        refs = (data.get("retarget") or {}).get("references")
        if isinstance(refs, dict) and refs:
            return refs
    return None


def _merge_retarget_references(yaml_path: str | Path | None, refs: dict) -> None:
    """Re-attach preserved ``retarget.references`` onto a freshly scaffolded yaml."""
    import yaml

    if not yaml_path or not refs:
        return
    p = Path(yaml_path)
    if not p.is_file():
        return
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    rt = data.get("retarget")
    if not isinstance(rt, dict):
        rt = {}
        data["retarget"] = rt
    existing = rt.get("references")
    if not isinstance(existing, dict):
        existing = {}
    existing.update(refs)
    rt["references"] = existing
    p.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
