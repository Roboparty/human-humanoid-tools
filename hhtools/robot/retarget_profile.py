"""Per-robot retarget defaults from ``robot.yaml``'s ``retarget:`` block."""

from __future__ import annotations

import logging
from dataclasses import replace
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
    from numpy.typing import NDArray

_log = logging.getLogger(__name__)

# Matches :data:`hhtools.retarget.calibration.calibration._CANONICAL_HUMAN_HEIGHT_M`.
_DEFAULT_HUMAN_HEIGHT_BY_REFERENCE: dict[str, float] = {
    "smpl": 1.65,
    "smplx": 1.65,
    "gvhmr": 1.65,
    "soma_bvh": 1.65,
    "lafan_bvh": 1.65,
    "mocap_bvh": 1.65,
    "xsens_mocap": 1.65,
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
            "min_lateral_separation": 0.0,
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
    "mocap_bvh": {
        "apply_feet_stabilizer": True,
        "feet_stabilizer": {
            "ground_contact_z": 0.045,
            "foot_planting_velocity_threshold": 0.005,
            "foot_planting_height_margin": 0.02,
            "min_lateral_separation": 0.0,
            "left_foot_name": "LeftFoot",
            "right_foot_name": "RightFoot",
            "left_toe_name": "LeftToeBase",
            "right_toe_name": "RightToeBase",
            "hips_name": "Hips",
            "enable_body_ground_clearance": True,
            "body_ground_clearance": 0.025,
            "body_ground_probe_joints": [
                "Head", "Neck", "Spine3",
                "LeftLeg", "RightLeg",
                "LeftForeArm", "RightForeArm",
                "LeftHand", "RightHand",
            ],
            "body_ground_lift_max_rate": 0.015,
            "body_ground_snap_on_penetration": True,
            "hand_ground_contact_z": 0.02,
            "chest_name": "Spine3",
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
    "xsens_mocap": {
        "apply_feet_stabilizer": True,
        "feet_stabilizer": {
            "ground_contact_z": 0.045,
            "foot_planting_velocity_threshold": 0.005,
            "foot_planting_height_margin": 0.02,
            "min_lateral_separation": 0.0,
            "left_foot_name": "LeftAnkle",
            "right_foot_name": "RightAnkle",
            "left_toe_name": "LeftToe",
            "right_toe_name": "RightToe",
            "hips_name": "Hips",
            "enable_body_ground_clearance": True,
            "body_ground_clearance": 0.025,
            "body_ground_probe_joints": [
                "Head", "Neck", "Chest4",
                "LeftKnee", "RightKnee",
                "LeftElbow", "RightElbow",
                "LeftWrist", "RightWrist",
            ],
            "body_ground_lift_max_rate": 0.015,
            "body_ground_snap_on_penetration": True,
            "chest_name": "Chest4",
            "arm_chains": [
                {
                    "shoulder": "LeftShoulder",
                    "chain": ["LeftElbow", "LeftWrist"],
                },
                {
                    "shoulder": "RightShoulder",
                    "chain": ["RightElbow", "RightWrist"],
                },
            ],
        },
        "ground_collision_weight": 10.0,
        # High-fps Xsens gait oscillates solved ankle height; anti-float root
        # pumping reads as vertical bobbing.  Keep the ground-penetration lift.
        "foot_clamp_anti_float": False,
    },
}


def _reference_defaults(reference: str) -> dict[str, Any]:
    return dict(_REFERENCE_FEET_DEFAULTS.get(reference, {}))


# Canonical slots that receive lateral IK narrowing when yaml scales drop.
_UPPER_BODY_CANONICAL = frozenset({
    "chest",
    "spine",
    "neck",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_collar",
    "right_collar",
})

_narrowing_ratios_cache: dict[tuple[object, ...], dict[str, float]] = {}


def _retarget_block(preset: "RobotPreset") -> dict[str, Any]:
    block = preset.meta.get("retarget")
    return dict(block) if isinstance(block, dict) else {}


def _reload_retarget_block(preset: "RobotPreset") -> dict[str, Any]:
    """Read ``retarget:`` from disk so yaml edits apply without restarting."""

    yaml_path = preset.meta.get("yaml_path")
    if yaml_path:
        try:
            with Path(yaml_path).open("r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
            block = data.get("retarget")
            if isinstance(block, dict):
                return dict(block)
        except OSError:
            pass
    return _retarget_block(preset)


def joint_scale_overrides_from_preset(
    preset: "RobotPreset",
) -> dict[str, float]:
    """Read ``retarget.joint_scale_multipliers`` from ``robot.yaml``.

    Values are absolute per-canonical joint scales (same units as calibration
    ``derived.scales``).  Only intentional manual tweaks belong here — values
    that match the current or any sibling calibration are ignored at retarget
    time (see :func:`~hhtools.robot.joint_scales.active_joint_scale_overrides`).
    """

    block = _reload_retarget_block(preset)
    raw = block.get("joint_scale_multipliers")
    if not isinstance(raw, dict):
        return {}
    return {str(k): float(v) for k, v in raw.items()}


def _yaml_active_scale_edits(
    preset: "RobotPreset",
    robot_model: "URDFRobotModel | None" = None,
    *,
    calibration: "RobotRetargetCalibration | None" = None,
    motion: "Motion | None" = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return ``(active yaml edits, calibration / URDF baselines)``."""

    yaml_scales = joint_scale_overrides_from_preset(preset)
    if not yaml_scales:
        return {}, {}

    from hhtools.robot.joint_scales import (
        active_joint_scale_overrides,
        all_calibration_scales_for_preset,
        scale_context_for_preset,
    )

    if calibration is not None and robot_model is not None:
        from hhtools.retarget.calibration.calibration import (
            derive_calibration_params,
            normalize_calibration_reference,
        )

        ref = normalize_calibration_reference(str(calibration.reference))
        if ref == "glb" and motion is None:
            baselines, _ = scale_context_for_preset(preset, robot_model)
        else:
            baselines = {
                str(k): float(v)
                for k, v in derive_calibration_params(
                    calibration,
                    robot_model,
                    reference_motion=motion,
                ).scales.items()
            }
    else:
        baselines, _ = scale_context_for_preset(preset, robot_model)
    zero_pose: dict[str, float] = {}
    if robot_model is not None:
        _, zero_pose = scale_context_for_preset(preset, robot_model)
    siblings = all_calibration_scales_for_preset(preset, robot_model)
    active = active_joint_scale_overrides(
        yaml_scales,
        baselines,
        zero_pose_scales=zero_pose,
        sibling_calibration_scales=siblings,
    )
    return active, baselines


def _apply_joint_scale_overrides_to_config(
    cfg: ScalerConfig,
    overrides: dict[str, float],
    *,
    motion: "Motion | None" = None,
) -> ScalerConfig:
    if not overrides:
        return cfg

    from hhtools.retarget.newton_basic.human_aliases import auto_source_to_canonical

    if motion is not None:
        joint_names = motion.hierarchy.bone_names
    else:
        joint_names = tuple(cfg.joint_scales.keys())
    src2can = auto_source_to_canonical(joint_names)
    new_scales = dict(cfg.joint_scales)
    for src in new_scales:
        canonical = src2can.get(src, src)
        override = overrides.get(canonical)
        if override is None:
            override = overrides.get(src)
        if override is not None:
            new_scales[src] = float(override)
    return replace(cfg, joint_scales=new_scales)


def joint_scale_narrowing_ratios(
    preset: "RobotPreset",
    *,
    robot_model: "URDFRobotModel | None" = None,
) -> dict[str, float]:
    """Lateral IK / roll narrowing factors: yaml scale ÷ baseline scale."""

    from hhtools.robot.joint_scales import scale_cache_key

    key = scale_cache_key(preset)
    cached = _narrowing_ratios_cache.get(key)
    if cached is not None:
        return dict(cached)

    active, baselines = _yaml_active_scale_edits(preset, robot_model)
    ratios: dict[str, float] = {}
    for canonical, yaml_val in active.items():
        if canonical not in _UPPER_BODY_CANONICAL:
            continue
        base = baselines.get(canonical)
        if base is not None and float(base) > 1e-6:
            ratio = float(yaml_val) / float(base)
            if abs(ratio - 1.0) > 1e-6:
                ratios[canonical] = ratio
    _narrowing_ratios_cache[key] = ratios
    return dict(ratios)


def _active_joint_scale_overrides_for_model(
    preset: "RobotPreset",
    robot_model: "URDFRobotModel | None" = None,
    *,
    calibration: "RobotRetargetCalibration | None" = None,
    motion: "Motion | None" = None,
) -> dict[str, float]:
    """Yaml scale tweaks that differ from calibration / URDF baseline."""

    active, _ = _yaml_active_scale_edits(
        preset, robot_model, calibration=calibration, motion=motion,
    )
    return dict(active)


def apply_upper_body_lateral_ik_narrowing(
    ik_targets: "NDArray",
    entries,
    preset: "RobotPreset",
    *,
    robot_model: "URDFRobotModel | None" = None,
) -> "NDArray":
    """Pull upper-body IK positions toward the body midplane in the lateral axis.

    Compares active ``retarget.joint_scale_multipliers`` edits against
    calibration / URDF baselines so the solved robot mesh narrows when the
    user intentionally reduced yaml scales.
    """

    import numpy as np

    affected = {
        c: float(r)
        for c, r in joint_scale_narrowing_ratios(
            preset, robot_model=robot_model,
        ).items()
        if float(r) < 1.0 - 1e-6
    }
    if not affected:
        return ik_targets

    name_to_i = {e.canonical_name: i for i, e in enumerate(entries)}
    out = np.asarray(ik_targets, dtype=np.float32).copy()
    n_frames = int(out.shape[0])

    for f in range(n_frames):
        ls_i = name_to_i.get("left_shoulder")
        rs_i = name_to_i.get("right_shoulder")
        lh_i = name_to_i.get("left_hip")
        rh_i = name_to_i.get("right_hip")
        chest_i = name_to_i.get("chest")
        hips_i = name_to_i.get("hips")

        lat_u: np.ndarray | None = None
        if ls_i is not None and rs_i is not None:
            lat = out[f, ls_i, :3] - out[f, rs_i, :3]
            lat[2] = 0.0
            lat_norm = float(np.linalg.norm(lat))
            if lat_norm >= 1e-6:
                lat_u = (lat / lat_norm).astype(np.float32, copy=False)
        elif lh_i is not None and rh_i is not None:
            lat = out[f, lh_i, :3] - out[f, rh_i, :3]
            lat[2] = 0.0
            lat_norm = float(np.linalg.norm(lat))
            if lat_norm >= 1e-6:
                lat_u = (lat / lat_norm).astype(np.float32, copy=False)
        if lat_u is None:
            lat_u = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        if chest_i is not None:
            anchor = out[f, chest_i, :3]
        elif hips_i is not None:
            anchor = out[f, hips_i, :3]
        elif ls_i is not None and rs_i is not None:
            anchor = 0.5 * (out[f, ls_i, :3] + out[f, rs_i, :3])
        else:
            continue

        for canon, mult in affected.items():
            idx = name_to_i.get(canon)
            if idx is None:
                continue
            p = out[f, idx, :3]
            d = p - anchor
            d_lat = float(np.dot(d, lat_u)) * lat_u
            d_rest = d - d_lat
            out[f, idx, :3] = anchor + d_rest + d_lat * np.float32(mult)

    return out


def effective_ik_t_weight(
    canonical_name: str,
    base_weight: float,
    preset: "RobotPreset",
    *,
    robot_model: "URDFRobotModel | None" = None,
) -> float:
    """Raise shoulder/arm tracking when yaml requests lateral narrowing."""

    ratio = joint_scale_narrowing_ratios(
        preset, robot_model=robot_model,
    ).get(canonical_name)
    if ratio is None or float(ratio) >= 1.0:
        return float(base_weight)
    # Shoulder pitch links need a strong pull — Ultron's redundant roll/yaw
    # chain otherwise keeps the solved mesh width unchanged.
    if canonical_name in ("left_shoulder", "right_shoulder"):
        return max(float(base_weight), 8.0) / max(float(ratio), 0.25)
    boost = min(4.0, 1.0 / max(float(ratio), 0.25))
    return float(base_weight) * boost


def _shoulder_side_from_dof(name: str) -> str | None:
    """Best-effort left/right from a DOF name (``left``, ``l_``, ``Left_``, …)."""

    low = name.lower()
    if "left" in low or low.startswith("l_"):
        return "left"
    if "right" in low or low.startswith("r_"):
        return "right"
    return None


def _is_shoulder_roll_dof(name: str) -> bool:
    """True for ``*_shoulder_roll_*`` or abbreviated ``*_shoulder_r_*`` (Ultron)."""

    low = name.lower()
    if "shoulder" not in low:
        return False
    if "roll" in low:
        return True
    return "shoulder_r_" in low or low.endswith("shoulder_r_joint")


def _infer_shoulder_roll_joints(preset: "RobotPreset") -> dict[str, str]:
    """Map canonical shoulders → shoulder-roll DOF names (best-effort)."""

    block = _reload_retarget_block(preset)
    explicit = block.get("shoulder_roll_joints")
    if isinstance(explicit, dict):
        return {str(k): str(v) for k, v in explicit.items()}

    rolls: dict[str, str] = {}
    for dof in preset.dof_order:
        name = str(dof)
        side = _shoulder_side_from_dof(name)
        if side is None or not _is_shoulder_roll_dof(name):
            continue
        canon = f"{side}_shoulder"
        if canon not in rolls:
            rolls[canon] = name
    return rolls


def _shoulder_roll_scale_ratios(
    preset: "RobotPreset",
    robot_model: "URDFRobotModel | None" = None,
) -> dict[str, float]:
    """Yaml shoulder scale ÷ calibration baseline (falls back to URDF narrowing)."""

    if robot_model is None:
        return {
            k: float(v)
            for k, v in joint_scale_narrowing_ratios(
                preset, robot_model=robot_model,
            ).items()
            if k in ("left_shoulder", "right_shoulder")
        }

    from hhtools.retarget.calibration.calibration import (
        load_calibration,
        resolve_calibration_file,
    )
    from hhtools.robot.joint_scales import _CALIBRATION_REF_ORDER

    calibration = None
    for ref in _CALIBRATION_REF_ORDER:
        cal_path = resolve_calibration_file(preset.root_dir, ref)
        if cal_path is None or not cal_path.is_file():
            continue
        try:
            cal = load_calibration(cal_path)
            if cal.robot and cal.robot != preset.name:
                continue
            calibration = cal
            break
        except Exception:
            continue

    if calibration is None:
        return {
            k: float(v)
            for k, v in joint_scale_narrowing_ratios(
                preset, robot_model=robot_model,
            ).items()
            if k in ("left_shoulder", "right_shoulder")
        }

    active, baselines = _yaml_active_scale_edits(
        preset, robot_model, calibration=calibration,
    )
    ratios: dict[str, float] = {}
    narrow_fallback = joint_scale_narrowing_ratios(
        preset, robot_model=robot_model,
    )
    for canon in ("left_shoulder", "right_shoulder"):
        yaml_val = active.get(canon)
        base = baselines.get(canon)
        if yaml_val is not None and base is not None and float(base) > 1e-6:
            ratios[canon] = float(yaml_val) / float(base)
        elif canon in narrow_fallback:
            ratios[canon] = float(narrow_fallback[canon])
    return ratios


# Radians of shoulder-roll abduction per unit (scale_ratio - 1) when yaml
# shoulder scale exceeds calibration (e.g. ratio 1.5 → +0.20 rad ≈ 11°).
_SHOULDER_ROLL_WIDEN_GAIN: float = 0.40
_SHOULDER_ROLL_WIDEN_MAX: float = 0.85


def apply_upper_body_roll_narrowing_post_ik(
    joint_q: "NDArray",
    dof_names: list[str] | tuple[str, ...],
    preset: "RobotPreset",
    *,
    root_coord_count: int,
    robot_model: "URDFRobotModel | None" = None,
) -> "NDArray":
    """Adjust shoulder roll after IK from yaml shoulder scale vs calibration.

    3-DOF shoulder IK often leaves pitch-link separation at the URDF rest
    width; roll abduction is the practical DOF for perceived shoulder span.
    Narrow when yaml scale < baseline (× ratio); abduct when yaml scale >
    baseline (+ gain × (ratio − 1)).
    """

    import numpy as np

    ratios = _shoulder_roll_scale_ratios(preset, robot_model=robot_model)
    rolls = _infer_shoulder_roll_joints(preset)
    if not rolls or not ratios:
        return joint_q

    out = np.asarray(joint_q, dtype=np.float32).copy()
    for canon, roll_joint in rolls.items():
        ratio = ratios.get(canon)
        if ratio is None or abs(float(ratio) - 1.0) <= 1e-3:
            continue
        if roll_joint not in dof_names:
            continue
        col = int(root_coord_count) + int(dof_names.index(roll_joint))
        r = float(ratio)
        if r < 1.0:
            out[:, col] = out[:, col] * np.float32(r)
            continue
        widen = min((r - 1.0) * _SHOULDER_ROLL_WIDEN_GAIN, _SHOULDER_ROLL_WIDEN_MAX)
        sign = 1.0 if "left" in canon else -1.0
        out[:, col] = out[:, col] + np.float32(sign * widen)
    return out


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


def build_scaler_config_for_robot(
    calibration: "RobotRetargetCalibration",
    model: "URDFRobotModel",
    motion: "Motion",
    *,
    human_height: float,
) -> ScalerConfig:
    """Build calibration-derived scaler and apply ``robot.yaml`` scale overrides."""

    from hhtools.retarget.calibration import build_scaler_config_from_calibration

    overrides = _active_joint_scale_overrides_for_model(
        model.preset, robot_model=model, calibration=calibration, motion=motion,
    )
    return build_scaler_config_from_calibration(
        calibration,
        model,
        motion,
        human_height=human_height,
        joint_scale_overrides=overrides or None,
    )


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
        return build_scaler_config_for_robot(
            calibration, model, motion, human_height=human_height,
        )

    bundled = bundled_scaler_path(preset, reference)
    if bundled is not None:
        cfg = load_scaler_config(bundled)
        if motion is not None:
            from hhtools.retarget.newton_basic.scaler import (
                adapt_scaler_config_for_hierarchy,
            )

            cfg = adapt_scaler_config_for_hierarchy(cfg, motion.hierarchy)
        overrides = _active_joint_scale_overrides_for_model(preset, robot_model=model)
        return _apply_joint_scale_overrides_to_config(cfg, overrides, motion=motion)

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
    """Pick ``min_lateral_separation`` from robot yaml only (no mesh auto-infer)."""
    merged = float(feet_raw.get("min_lateral_separation", 0.0))
    explicit = _feet_stabilizer_key_explicit(preset, reference, "min_lateral_separation")
    if explicit is not None:
        return float(explicit)
    return merged


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
        # Default ON (0.02 m): the post-IK clamp it drives only nudges hip
        # abduction on frames where the solved foot *meshes* actually
        # interpenetrate, so it is a no-op for normal gait / wide stances and
        # safe for every robot.  Set ``min_foot_clearance: 0.0`` in robot.yaml
        # to explicitly disable.
        min_foot_clearance=float(feet_raw.get("min_foot_clearance", 0.02)),
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
    foot_clamp_anti_penetration: bool | None = None,
):
    """Merge ``retarget:`` defaults into :class:`PipelineConfig`.

    When ``foot_clamp_anti_penetration`` is set (e.g. from the web UI), it
    overrides ``robot.yaml`` for that retarget run.
    """

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

    anti_pen = (
        bool(foot_clamp_anti_penetration)
        if foot_clamp_anti_penetration is not None
        else bool(_pick("foot_clamp_anti_penetration", True))
    )
    # Web / batch UI "脚穿地修正" off ⇒ raw IK root+DOF only: skip the whole
    # post-IK foot suite (anti-float + lateral mesh).  Those steps exist for
    # contact polish and dominate wall time on large LAFAN-style batches.
    foot_polish = bool(anti_pen)

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
        post_ik_foot_clamps=foot_polish,
        # Always default-on: clip-wide sole↔floor snap is cheap vs mesh clamps
        # and must still run when the UI disables per-frame foot polish.
        clip_floor_snap=bool(_pick("clip_floor_snap", True)),
        foot_clamp_anti_float=(
            bool(_pick("foot_clamp_anti_float", True)) if foot_polish else False
        ),
        foot_clamp_anti_penetration=anti_pen,
        foot_clamp_max_lift_rate=float(_pick("foot_clamp_max_lift_rate", 0.02)),
    )
