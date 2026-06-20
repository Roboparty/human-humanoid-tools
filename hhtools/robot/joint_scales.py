"""Per-canonical joint scale inference from URDF geometry."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from hhtools.robot.loader import URDFRobotModel

_log = logging.getLogger(__name__)

__all__ = [
    "infer_joint_scales_for_scaffold",
    "infer_joint_scales_from_urdf",
    "joint_scale_baselines_for_preset",
    "robot_dir_has_calibration",
    "scale_cache_key",
    "scale_context_for_preset",
    "scales_for_robot_yaml_from_derived",
    "sync_joint_scale_multipliers_to_robot_yaml",
]

# Relative tolerance: yaml matching calibration within this band is a no-op.
_SCALE_MATCH_EPS: float = 1e-3

_CALIBRATION_REF_ORDER: tuple[str, ...] = (
    "smplx", "smpl", "lafan_bvh", "soma_bvh", "xsens_mocap", "gvhmr",
)

# Per-preset scale context — avoids reloading URDF on every IK objective weight.
_scale_context_cache: dict[tuple[object, ...], tuple[dict[str, float], dict[str, float]]] = {}


def _newest_calibration_mtime(robot_dir: Path) -> float:
    from hhtools.retarget.calibration.calibration import resolve_calibration_file

    newest = 0.0
    for ref in _CALIBRATION_REF_ORDER:
        cal_path = resolve_calibration_file(robot_dir, ref)
        if cal_path is not None and cal_path.is_file():
            try:
                newest = max(newest, cal_path.stat().st_mtime)
            except OSError:
                pass
    return newest


def _scale_context_cache_key(preset) -> tuple[object, ...]:
    yaml_path = preset.meta.get("yaml_path")
    y_mtime = 0.0
    if yaml_path:
        yp = Path(yaml_path)
        if yp.is_file():
            try:
                y_mtime = yp.stat().st_mtime
            except OSError:
                pass
    urdf_mtime = 0.0
    if preset.urdf_path is not None and preset.urdf_path.is_file():
        try:
            urdf_mtime = preset.urdf_path.stat().st_mtime
        except OSError:
            pass
    cal_mtime = _newest_calibration_mtime(Path(preset.root_dir))
    return (
        preset.name,
        y_mtime,
        urdf_mtime,
        cal_mtime,
        tuple(sorted(preset.ik_map.items())),
        tuple(str(p) for p in preset.mesh_search_paths),
    )


def scale_cache_key(preset) -> tuple[object, ...]:
    """Cache key for yaml / calibration / URDF scale context."""

    return _scale_context_cache_key(preset)


def scale_context_for_preset(
    preset,
    robot_model: "URDFRobotModel | None" = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return ``(calibration_baselines, zero_pose_scales)`` with process cache."""

    key = _scale_context_cache_key(preset)
    cached = _scale_context_cache.get(key)
    if cached is not None:
        return cached

    baselines = joint_scale_baselines_for_preset(preset, robot_model)
    zero_pose: dict[str, float] = {}
    if preset.urdf_path is not None and preset.ik_map:
        zero_pose = infer_joint_scales_from_urdf(
            preset.urdf_path,
            dict(preset.ik_map),
            preset_name=preset.name,
            mesh_search_paths=preset.mesh_search_paths,
        )
    _scale_context_cache[key] = (baselines, zero_pose)
    return baselines, zero_pose


def infer_joint_scales_from_urdf(
    urdf_path: Path | str,
    ik_map: dict[str, str],
    *,
    preset_name: str = "scaffold",
    mesh_search_paths: Iterable[Path | str] = (),
) -> dict[str, float]:
    """Per-canonical scales from URDF FK at zero pose (same as calibration).

    Uses :func:`~hhtools.retarget.calibration.calibration.derive_calibration_params`
    with an empty ``calibrated_joint_q`` so scaffolded ``robot.yaml`` values
    match what retarget would derive before the user edits them.
    """

    if not ik_map:
        return {}

    from hhtools.robot.base import RobotPreset
    from hhtools.robot.loader import load_robot
    from hhtools.retarget.calibration.calibration import (
        RobotRetargetCalibration,
        derive_calibration_params,
    )

    urdf_path = Path(urdf_path).resolve()
    mesh_paths = tuple(
        Path(p).resolve()
        for p in mesh_search_paths
        if Path(p).is_dir()
    )

    preset = RobotPreset(
        name=preset_name,
        display_name=preset_name,
        root_dir=urdf_path.parent,
        urdf_path=urdf_path,
        mesh_search_paths=mesh_paths,
        ik_map=dict(ik_map),
    )

    try:
        model = load_robot(preset, compile_mjcf=False)
        cal = RobotRetargetCalibration(
            robot=preset_name,
            reference="smplx",  # type: ignore[arg-type]
            calibrated_joint_q={},
        )
        derived = derive_calibration_params(cal, model)
    except Exception as exc:
        _log.debug("infer_joint_scales_from_urdf failed for %s: %s", urdf_path, exc)
        return {canonical: 1.0 for canonical in ik_map}

    out: dict[str, float] = {}
    for canonical in ik_map:
        scale = float(derived.scales.get(canonical, 1.0))
        out[canonical] = round(scale, 4)
    return out


def infer_joint_scales_for_scaffold(
    robot_dir: Path | str,
    urdf_path: Path | str,
    ik_map: dict[str, str],
    *,
    preset_name: str = "scaffold",
    mesh_search_paths: Iterable[Path | str] = (),
) -> dict[str, float]:
    """Calibration-derived ``joint_scale_multipliers`` for re-scaffolded ``robot.yaml``.

    Call only when ``robot_dir`` already has a calibration file (see
    :func:`robot_dir_has_calibration`).
    """

    robot_dir = Path(robot_dir).resolve()
    if not ik_map:
        return {}

    from hhtools.robot.base import RobotPreset

    urdf_path = Path(urdf_path).resolve()
    mesh_paths = tuple(
        Path(p).resolve()
        for p in mesh_search_paths
        if Path(p).is_dir()
    )
    preset = RobotPreset(
        name=preset_name,
        display_name=preset_name,
        root_dir=robot_dir,
        urdf_path=urdf_path,
        mesh_search_paths=mesh_paths,
        ik_map=dict(ik_map),
    )

    baselines = joint_scale_baselines_for_preset(preset)
    return {
        canonical: round(float(baselines.get(canonical, 1.0)), 4)
        for canonical in ik_map
    }


def active_joint_scale_overrides(
    yaml_scales: dict[str, float],
    baseline_scales: dict[str, float],
    *,
    zero_pose_scales: dict[str, float] | None = None,
) -> dict[str, float]:
    """Keep only yaml entries that intentionally differ from calibration."""

    if not yaml_scales or not baseline_scales:
        return {}

    active: dict[str, float] = {}
    for canonical, yaml_val in yaml_scales.items():
        base = baseline_scales.get(canonical)
        if base is None or float(base) <= 1e-6:
            continue
        if zero_pose_scales is not None:
            zp = zero_pose_scales.get(canonical)
            if zp is not None and float(zp) > 1e-6:
                zp_rel = abs(float(yaml_val) - float(zp)) / float(zp)
                if zp_rel <= _SCALE_MATCH_EPS:
                    # Stale scaffold zero-pose default — user has not edited.
                    continue
        rel = abs(float(yaml_val) - float(base)) / float(base)
        if rel <= _SCALE_MATCH_EPS:
            continue
        active[canonical] = float(yaml_val)
    return active


def robot_dir_has_calibration(robot_dir: Path | str) -> bool:
    """True when ``robot_dir`` contains a retarget calibration yaml."""

    from hhtools.retarget.calibration.calibration import resolve_calibration_file

    robot_dir = Path(robot_dir)
    for ref in _CALIBRATION_REF_ORDER:
        cal_path = resolve_calibration_file(robot_dir, ref)
        if cal_path is not None and cal_path.is_file():
            return True
    return False


def scales_for_robot_yaml_from_derived(
    derived_scales: dict[str, float],
    ik_map: dict[str, str],
) -> dict[str, float]:
    """Format calibration ``derived.scales`` for ``robot.yaml``."""

    return {
        canonical: round(float(derived_scales[canonical]), 4)
        for canonical in ik_map
        if canonical in derived_scales
    }


def sync_joint_scale_multipliers_to_robot_yaml(
    yaml_path: str | Path,
    derived_scales: dict[str, float],
    ik_map: dict[str, str],
) -> None:
    """Write calibration scales into ``retarget.joint_scale_multipliers``."""

    from hhtools.robot.yaml_io import update_robot_yaml_joint_scale_multipliers

    scales = scales_for_robot_yaml_from_derived(derived_scales, ik_map)
    if scales:
        update_robot_yaml_joint_scale_multipliers(yaml_path, scales)


def joint_scale_baselines_for_preset(
    preset,
    robot_model: "URDFRobotModel | None" = None,
) -> dict[str, float]:
    """Calibration- or URDF-derived scales without ``robot.yaml`` overrides."""

    from hhtools.retarget.calibration.calibration import (
        derive_calibration_params,
        load_calibration,
        resolve_calibration_file,
    )

    model = robot_model
    if model is None:
        from hhtools.robot.loader import load_robot

        model = load_robot(preset, compile_mjcf=False)

    for ref in _CALIBRATION_REF_ORDER:
        cal_path = resolve_calibration_file(preset.root_dir, ref)
        if cal_path is None or not cal_path.is_file():
            continue
        try:
            cal = load_calibration(cal_path)
            if cal.robot and cal.robot != preset.name:
                continue
            return {
                str(k): float(v)
                for k, v in derive_calibration_params(cal, model).scales.items()
            }
        except Exception:
            continue

    if preset.urdf_path is not None and preset.ik_map:
        return infer_joint_scales_from_urdf(
            preset.urdf_path,
            dict(preset.ik_map),
            preset_name=preset.name,
            mesh_search_paths=preset.mesh_search_paths,
        )
    return {}
