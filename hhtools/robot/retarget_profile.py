"""Per-robot retarget defaults from ``robot.yaml``'s ``retarget:`` block."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from hhtools.retarget.newton_basic.config import (
    FeetStabilizerConfig,
    ScalerConfig,
    load_scaler_config,
)

if TYPE_CHECKING:
    from hhtools.core.motion import Motion
    from hhtools.retarget.calibration.calibration import RobotRetargetCalibration
    from hhtools.robot.base import RobotPreset
    from hhtools.robot.loader import URDFRobotModel

# Matches :data:`hhtools.retarget.calibration.calibration._CANONICAL_HUMAN_HEIGHT_M`.
_DEFAULT_HUMAN_HEIGHT_BY_REFERENCE: dict[str, float] = {
    "smpl": 1.65,
    "smplx": 1.65,
    "gvhmr": 1.65,
    "soma_bvh": 1.65,
    "lafan_bvh": 1.65,
    "glb": 1.65,
}

# Pre-IK feet / body-ground defaults mirroring soma lafan_to_rp1_scaler_config.json.
# Applied when robot.yaml has no explicit ``retarget.feet_stabilizer`` block.
_REFERENCE_FEET_DEFAULTS: dict[str, dict[str, Any]] = {
    "lafan_bvh": {
        "apply_feet_stabilizer": True,
        "feet_stabilizer": {
            "ground_contact_z": 0.045,
            "foot_planting_velocity_threshold": 0.005,
            "foot_planting_height_margin": 0.02,
            "min_lateral_separation": 0.1,
            "left_foot_name": "LeftFoot",
            "right_foot_name": "RightFoot",
            "left_toe_name": "LeftToe",
            "right_toe_name": "RightToe",
            "hips_name": "Hips",
            "enable_body_ground_clearance": True,
            "body_ground_clearance": 0.025,
            "body_ground_probe_joints": [
                "Head", "Neck", "Spine2",
                "LeftLeg", "RightLeg",
                "LeftForeArm", "RightForeArm",
                "LeftHand", "RightHand",
            ],
            "body_ground_probe_below_meters": {"Head": 0.11},
            "body_ground_lift_max_rate": 0.015,
            "body_ground_snap_on_penetration": True,
            "hand_ground_contact_z": 0.02,
            "chest_name": "Spine2",
            "arm_chains": [
                {
                    "shoulder": "LeftArm",
                    "chain": ["LeftForeArm", "LeftHand"],
                },
                {
                    "shoulder": "RightArm",
                    "chain": ["RightForeArm", "RightHand"],
                },
            ],
        },
        "ground_collision_weight": 10.0,
    },
    "soma_bvh": {
        "apply_feet_stabilizer": True,
        "feet_stabilizer": {
            "left_foot_name": "LeftFoot",
            "right_foot_name": "RightFoot",
            "hips_name": "Hips",
        },
    },
}


def _reference_defaults(reference: str) -> dict[str, Any]:
    return dict(_REFERENCE_FEET_DEFAULTS.get(reference, {}))


def _retarget_block(preset: "RobotPreset") -> dict[str, Any]:
    block = preset.meta.get("retarget")
    return dict(block) if isinstance(block, dict) else {}


def _reference_block(preset: "RobotPreset", reference: str) -> dict[str, Any]:
    block = _retarget_block(preset)
    refs = block.get("references")
    if isinstance(refs, dict):
        ref_cfg = refs.get(reference)
        if isinstance(ref_cfg, dict):
            return dict(ref_cfg)
    return {}


def _workspace_robots_root() -> Path | None:
    """``configs/robots/`` in the source tree, if present."""

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs" / "robots"
        if candidate.is_dir():
            return candidate
    return None


def _workspace_robot_dir(preset_name: str) -> Path | None:
    """``configs/robots/<name>/`` in the source tree, if present."""

    root = _workspace_robots_root()
    if root is None:
        return None
    candidate = root / preset_name
    return candidate if candidate.is_dir() else None


def _scaler_search_roots(preset: "RobotPreset") -> list[Path]:
    """Preset dir first, then same-named workspace bundle (user upload shadowing)."""

    roots: list[Path] = [preset.root_dir.resolve()]
    ws = _workspace_robot_dir(preset.name)
    if ws is not None:
        resolved = ws.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def _scaler_rel_candidates(
    preset: "RobotPreset",
    reference: str,
) -> list[str]:
    """Scaler yaml filenames declared in ``robot.yaml`` for ``reference``."""

    rels: list[str] = []
    user_rel = _reference_block(preset, reference).get("scaler_config")
    if user_rel:
        rels.append(str(user_rel))

    ws = _workspace_robot_dir(preset.name)
    if ws is not None and ws.resolve() != preset.root_dir.resolve():
        yaml_path = ws / "robot.yaml"
        if yaml_path.is_file():
            try:
                with yaml_path.open("r", encoding="utf-8") as fp:
                    data = yaml.safe_load(fp) or {}
                refs = (data.get("retarget") or {}).get("references") or {}
                ref_cfg = refs.get(reference) or {}
                ws_rel = ref_cfg.get("scaler_config")
                if ws_rel and str(ws_rel) not in rels:
                    rels.append(str(ws_rel))
            except Exception:  # noqa: BLE001 — optional metadata
                pass
    return rels


def bundled_scaler_path(preset: "RobotPreset", reference: str) -> Path | None:
    """Return a preset-local scaler yaml when ``robot.yaml`` declares one.

    Scaler YAML is optional and must be referenced explicitly via
    ``retarget.references.<reference>.scaler_config``.  All robots otherwise
    derive scaler parameters from Web / CLI calibration
    (``retarget_calibration_<reference>.yaml``).
    """

    for root in _scaler_search_roots(preset):
        for rel in _scaler_rel_candidates(preset, reference):
            path = (root / rel).resolve()
            if path.is_file():
                return path
    return None


def default_human_height(
    preset: "RobotPreset",
    reference: str,
    *,
    fallback: float = 1.7,
) -> float:
    """Default source-human height when the request omits one.

    Prefer an optional per-robot bundled scaler's ``human_height_assumption``,
    else a reference-family canonical stature (1.65 m for SMPL / SOMA / LAFAN /
    GLB), else ``fallback``.
    """

    bundled = bundled_scaler_path(preset, reference)
    if bundled is not None:
        try:
            cfg = load_scaler_config(bundled)
        except Exception:  # noqa: BLE001 - fall back to a sane constant
            pass
        else:
            h = float(getattr(cfg, "human_height_assumption", 0.0) or 0.0)
            if h > 0.1:
                return h

    from hhtools.retarget.calibration.calibration import normalize_calibration_reference

    ref = normalize_calibration_reference(reference)
    if ref in _DEFAULT_HUMAN_HEIGHT_BY_REFERENCE:
        return _DEFAULT_HUMAN_HEIGHT_BY_REFERENCE[ref]
    return float(fallback)


def resolve_retarget_scaler_config(
    preset: "RobotPreset",
    reference: str,
    *,
    calibration: "RobotRetargetCalibration | None",
    model: "URDFRobotModel",
    motion: "Motion",
    human_height: float,
) -> ScalerConfig:
    """Prefer calibration-derived scaler; fall back to optional bundled yaml."""

    if calibration is not None and model is not None:
        from hhtools.retarget.calibration import build_scaler_config_from_calibration

        return build_scaler_config_from_calibration(
            calibration, model, motion, human_height=human_height,
        )

    bundled = bundled_scaler_path(preset, reference)
    if bundled is not None:
        cfg = load_scaler_config(bundled)
        if motion is not None:
            from hhtools.retarget.newton_basic.scaler import (
                adapt_scaler_config_for_hierarchy,
            )

            return adapt_scaler_config_for_hierarchy(cfg, motion.hierarchy)
        return cfg

    raise ValueError(
        f"robot {preset.name!r} has no bundled scaler for reference "
        f"{reference!r} and no calibration file"
    )


def _feet_stabilizer_key_explicit(
    preset: "RobotPreset",
    reference: str,
    key: str,
) -> float | None:
    """Return a feet-stabilizer value only when set on the robot yaml (not defaults)."""
    block = _retarget_block(preset)
    ref_cfg = _reference_block(preset, reference)
    for src in (ref_cfg.get("feet_stabilizer"), block.get("feet_stabilizer")):
        if isinstance(src, dict) and key in src:
            return float(src[key])
    return None


def _resolve_min_lateral_separation(
    preset: "RobotPreset",
    reference: str,
    feet_raw: dict[str, Any],
    *,
    model: "URDFRobotModel | None" = None,
) -> float:
    """Pick ``min_lateral_separation`` from yaml and/or foot mesh geometry."""
    merged = float(feet_raw.get("min_lateral_separation", 0.0))
    explicit = _feet_stabilizer_key_explicit(preset, reference, "min_lateral_separation")

    inferred: float | None = None
    if model is not None:
        from hhtools.robot.foot_geometry import estimate_min_lateral_foot_separation

        inferred = estimate_min_lateral_foot_separation(model)
    elif preset.urdf_path is not None and preset.urdf_path.is_file():
        try:
            from hhtools.robot.foot_geometry import estimate_min_lateral_foot_separation
            from hhtools.robot.loader import load_robot

            inferred = estimate_min_lateral_foot_separation(
                load_robot(preset, compile_mjcf=False),
            )
        except Exception:
            inferred = None

    if inferred is not None and inferred > 0.0:
        if explicit is not None:
            return max(explicit, inferred)
        return max(merged, inferred)
    return merged if explicit is None else float(explicit)


def _arm_chain_max_reach_explicit(
    preset: "RobotPreset",
    reference: str,
    shoulder: str,
) -> float | None:
    """Return ``max_reach`` only when authored on the robot yaml arm chain."""
    block = _retarget_block(preset)
    ref_cfg = _reference_block(preset, reference)
    for src in (ref_cfg.get("feet_stabilizer"), block.get("feet_stabilizer")):
        if not isinstance(src, dict):
            continue
        for entry in src.get("arm_chains") or ():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("shoulder", "")) != shoulder:
                continue
            if "max_reach" in entry:
                return float(entry["max_reach"])
    return None


def _resolve_arm_chain_max_reach(
    preset: "RobotPreset",
    reference: str,
    shoulder: str,
    feet_raw: dict[str, Any],
    *,
    model: "URDFRobotModel | None" = None,
) -> float:
    """Pick ``max_reach`` from robot FK and/or yaml."""
    merged = 0.0
    for entry in feet_raw.get("arm_chains") or ():
        if isinstance(entry, dict) and str(entry.get("shoulder", "")) == shoulder:
            merged = float(entry.get("max_reach", 0.0) or 0.0)
            break

    inferred: float | None = None
    if model is not None:
        from hhtools.robot.arm_geometry import (
            estimate_shoulder_to_wrist_reach,
            infer_side_from_shoulder_name,
        )

        side = infer_side_from_shoulder_name(shoulder)
        if side is not None:
            inferred = estimate_shoulder_to_wrist_reach(model, side=side)

    explicit = _arm_chain_max_reach_explicit(preset, reference, shoulder)

    if inferred is not None and inferred > 0.0:
        if explicit is not None:
            return max(float(explicit), inferred)
        return inferred
    if explicit is not None:
        return float(explicit)
    if merged > 0.0:
        return merged
    return 0.50


def build_feet_stabilizer_config(
    preset: "RobotPreset",
    reference: str,
    *,
    model: "URDFRobotModel | None" = None,
) -> FeetStabilizerConfig | None:
    """Feet stabilizer knobs from ``retarget.feet`` / per-reference overrides."""

    block = _retarget_block(preset)
    ref_cfg = _reference_block(preset, reference)
    ref_defaults = _reference_defaults(reference)

    feet_raw: dict[str, Any] = {}
    for src in (
        ref_defaults.get("feet_stabilizer"),
        block.get("feet_stabilizer"),
        ref_cfg.get("feet_stabilizer"),
    ):
        if isinstance(src, dict):
            feet_raw.update(src)

    if not feet_raw and not ref_defaults.get("feet_stabilizer"):
        return None

    probe_raw = feet_raw.get("body_ground_probe_joints") or ()
    probe_below_raw = feet_raw.get("body_ground_probe_below_meters") or {}
    probe_below = {
        str(k): float(v) for k, v in probe_below_raw.items()
        if isinstance(probe_below_raw, dict)
    }

    from hhtools.retarget.newton_basic.config import ArmChainConfig

    arm_chains: list[ArmChainConfig] = []
    for entry in feet_raw.get("arm_chains") or ():
        if not isinstance(entry, dict):
            continue
        shoulder = str(entry.get("shoulder", ""))
        chain_raw = entry.get("chain") or ()
        if not shoulder or not chain_raw:
            continue
        max_reach = _resolve_arm_chain_max_reach(
            preset, reference, shoulder, feet_raw, model=model,
        )
        if max_reach > 0.0:
            arm_chains.append(
                ArmChainConfig(
                    shoulder=shoulder,
                    chain=tuple(str(c) for c in chain_raw),
                    max_reach=max_reach,
                )
            )

    return FeetStabilizerConfig(
        up_axis=str(feet_raw.get("up_axis", preset.up_axis)),  # type: ignore[arg-type]
        forward_axis=str(feet_raw.get("forward_axis", preset.forward_axis)),  # type: ignore[arg-type]
        ground_contact_z=float(feet_raw.get("ground_contact_z", 0.0)),
        min_foot_clearance=float(feet_raw.get("min_foot_clearance", 0.0)),
        max_ground_correction=float(feet_raw.get("max_ground_correction", 0.05)),
        ground_uprightness_range=float(feet_raw.get("ground_uprightness_range", 0.30)),
        foot_planting_velocity_threshold=float(
            feet_raw.get("foot_planting_velocity_threshold", 0.0)
        ),
        foot_planting_height_margin=float(
            feet_raw.get("foot_planting_height_margin", 0.02)
        ),
        foot_planting_release_frames=int(
            feet_raw.get("foot_planting_release_frames", 3)
        ),
        min_lateral_separation=_resolve_min_lateral_separation(
            preset, reference, feet_raw, model=model,
        ),
        smoothing_max_rate=float(feet_raw.get("smoothing_max_rate", 0.008)),
        left_foot_name=str(feet_raw.get("left_foot_name", "left_ankle")),
        right_foot_name=str(feet_raw.get("right_foot_name", "right_ankle")),
        left_toe_name=feet_raw.get("left_toe_name"),
        right_toe_name=feet_raw.get("right_toe_name"),
        hips_name=str(feet_raw.get("hips_name", "hips")),
        enable_body_ground_clearance=bool(
            feet_raw.get("enable_body_ground_clearance", False)
        ),
        body_ground_plane_z=float(feet_raw.get("body_ground_plane_z", 0.0)),
        body_ground_clearance=float(feet_raw.get("body_ground_clearance", 0.025)),
        body_ground_probe_joints=tuple(str(j) for j in probe_raw),
        body_ground_probe_below_meters=probe_below,
        body_ground_default_probe_below=float(
            feet_raw.get("body_ground_default_probe_below", 0.0)
        ),
        body_ground_lift_max_rate=float(feet_raw.get("body_ground_lift_max_rate", 0.015)),
        body_ground_snap_on_penetration=bool(
            feet_raw.get("body_ground_snap_on_penetration", True)
        ),
        hand_ground_contact_z=float(feet_raw.get("hand_ground_contact_z", 0.0)),
        chest_name=str(feet_raw.get("chest_name", "Spine2")),
        arm_chains=tuple(arm_chains),
    )


def _resolve_ground_collision_bodies(
    preset: "RobotPreset",
    reference: str,
    ground_weight: float,
) -> tuple[dict, ...]:
    """Explicit yaml bodies override; otherwise derive from ``ik_map``."""

    block = _retarget_block(preset)
    ref_cfg = _reference_block(preset, reference)

    for src in (ref_cfg, block):
        if "ground_collision_bodies" in src:
            raw = src["ground_collision_bodies"]
            if isinstance(raw, list):
                return tuple(dict(b) for b in raw)
            return ()

    ref_defaults = _reference_defaults(reference)
    if "ground_collision_bodies" in ref_defaults:
        raw = ref_defaults["ground_collision_bodies"]
        if isinstance(raw, list):
            return tuple(dict(b) for b in raw)

    if ground_weight <= 0.0 or not preset.ik_map or not preset.has_urdf:
        return ()

    from hhtools.retarget.newton_basic.ground_collision_bodies import (
        build_ground_collision_bodies_from_ik_map,
    )

    assert preset.urdf_path is not None
    built = build_ground_collision_bodies_from_ik_map(
        preset.ik_map, preset.urdf_path,
    )
    return tuple(dict(b) for b in built)


def build_pipeline_config_for_preset(
    preset: "RobotPreset",
    reference: str,
    *,
    ik_iterations: int,
):
    """Merge ``retarget:`` defaults into :class:`PipelineConfig`."""

    from hhtools.retarget.newton_basic.pipeline import PipelineConfig

    block = _retarget_block(preset)
    ref_cfg = _reference_block(preset, reference)

    def _pick(key: str, default: Any) -> Any:
        if key in ref_cfg:
            return ref_cfg[key]
        if key in block:
            return block[key]
        ref_defaults = _reference_defaults(reference)
        if key in ref_defaults:
            return ref_defaults[key]
        return default

    ground_weight = float(_pick("ground_collision_weight", 0.0))
    ground_bodies = _resolve_ground_collision_bodies(
        preset, reference, ground_weight,
    )

    return PipelineConfig(
        ik_iterations=int(ik_iterations),
        joint_limit_weight=float(_pick("joint_limit_weight", 10.0)),
        smooth_joint_filter_weight=float(_pick("smooth_joint_filter_weight", 5.5)),
        # Per-frame velocity rate limiter.  Newton's per-frame IK only couples
        # adjacent frames through warm-starting (there is no temporal-coherence
        # objective), so near redundant/singular poses — falls, get-ups, even
        # ordinary walking — the LM solver can hop to a different null-space
        # branch and produce a single-frame joint "teleport".  An 8 rad/s joint
        # / 6 rad/s root cap (matching soma-retargeter's lafan_to_rp1 config)
        # clamps those teleports while leaving genuine fast motion intact.
        max_joint_velocity=float(_pick("max_joint_velocity", 8.0)),
        max_root_angular_velocity=float(_pick("max_root_angular_velocity", 6.0)),
        num_initialization_frames=int(_pick("num_initialization_frames", 0)),
        num_stabilization_frames=int(_pick("num_stabilization_frames", 0)),
        apply_feet_stabilizer=bool(_pick("apply_feet_stabilizer", False)),
        ground_collision_weight=ground_weight,
        ground_collision_z=float(_pick("ground_collision_z", 0.0)),
        ground_collision_bodies=ground_bodies,
        ground_collision_dynamic_boost=bool(
            _pick("ground_collision_dynamic_boost", True)
        ),
    )
