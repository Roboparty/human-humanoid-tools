"""Scaled skeleton preview for the Web UI.

The yellow overlay uses **uniform** ``robot_height / human_height`` scaling on the
motion's source topology when the clip has enough bones (>= 10).  The older
``NewtonBasicPipeline.scale_only`` path only exposes IK canonical effectors and
distorts dense rigs (OMOMO, meshmimic, terrain clips).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hhtools.core.anatomy import (
    exclude_joint_from_compact_scaled_preview,
    motion_has_interaction_scene,
)
from hhtools.core.grounding import (
    human_source_floor_z_world,
    retarget_source_floor_z_world,
)
from hhtools.core.motion import Motion
from hhtools.retarget.anatomy import (
    exclude_unmapped_head_neck_from_scaled_preview,
    scaled_overlay_exclude_bone_indices,
)


def resolve_web_scaler_config(
    model,
    motion: Motion,
    reference: str,
    human_height: float,
):
    """Same scaler resolution as :func:`_retarget_single` (bundled > calibration)."""

    from hhtools.retarget.calibration import (
        load_calibration,
        resolve_preset_calibration_file,
    )
    from hhtools.robot.retarget_profile import (
        bundled_scaler_path,
        resolve_retarget_scaler_config,
    )

    preset = model.preset
    cal_path = resolve_preset_calibration_file(preset, reference)
    if cal_path is None and bundled_scaler_path(preset, reference) is None:
        raise ValueError(
            f"robot {preset.name!r} has no bundled scaler or calibration "
            f"for reference {reference!r}"
        )
    calibration = load_calibration(cal_path) if cal_path is not None else None
    return resolve_retarget_scaler_config(
        preset,
        reference,
        calibration=calibration,
        model=model,
        motion=motion,
        human_height=float(human_height),
    )


def _scaler_skeleton_segment_indices(
    joint_names: tuple[str, ...] | list[str],
    hierarchy,
    *,
    ik_map_canonicals: frozenset[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Edges (parent→child) in scaler joint index space."""
    pi = np.asarray(hierarchy.parent_indices, dtype=np.int64)
    hnames = list(hierarchy.bone_names)
    h_idx = {hnames[i]: i for i in range(len(hnames))}
    name_to_i = {n: i for i, n in enumerate(joint_names)}
    src: list[int] = []
    dst: list[int] = []
    for i, n in enumerate(joint_names):
        if n not in h_idx:
            continue
        if str(n).lower().endswith("footmod"):
            continue
        hi = int(h_idx[n])
        p = int(pi[hi])
        anc_sc = None
        while p >= 0:
            pn = hnames[p]
            if pn in name_to_i:
                j = int(name_to_i[pn])
                if j != i:
                    anc_sc = j
                break
            p = int(pi[p])
        if anc_sc is not None:
            src.append(anc_sc)
            dst.append(i)
    fs: list[int] = []
    fd: list[int] = []
    for s, d in zip(src, dst, strict=True):
        ns, nd = joint_names[s], joint_names[d]
        if exclude_joint_from_compact_scaled_preview(ns) or exclude_joint_from_compact_scaled_preview(nd):
            continue
        if ik_map_canonicals and (
            exclude_unmapped_head_neck_from_scaled_preview(ns, ik_map_canonicals=ik_map_canonicals)
            or exclude_unmapped_head_neck_from_scaled_preview(nd, ik_map_canonicals=ik_map_canonicals)
        ):
            continue
        fs.append(s)
        fd.append(d)
    return (
        np.asarray(fs, dtype=np.int32),
        np.asarray(fd, dtype=np.int32),
    )


def _visible_joint_indices(
    joint_names: list[str],
    ik_canons: frozenset[str],
    *,
    include_unmapped_head_neck: bool = False,
) -> np.ndarray:
    idx = []
    for i, n in enumerate(joint_names):
        if exclude_joint_from_compact_scaled_preview(n):
            continue
        if (
            not include_unmapped_head_neck
            and ik_canons
            and exclude_unmapped_head_neck_from_scaled_preview(
                n, ik_map_canonicals=ik_canons,
            )
        ):
            continue
        idx.append(i)
    return np.asarray(idx, dtype=np.int32)


def _scaled_uniform_positions_frame0(
    motion: Motion,
    ratio: float,
    *,
    frame: int = 0,
) -> np.ndarray:
    """Source bone positions after foot-floor subtract + uniform scale (one frame)."""
    z_floor = float(human_source_floor_z_world(motion))
    pos = np.asarray(motion.positions[int(frame)], dtype=np.float64).copy()
    pos[:, 2] = (pos[:, 2] - z_floor) * float(ratio)
    return pos


def _endpoint_bone_indices(
    bone_names: tuple[str, ...] | list[str],
    *,
    kind: str,
) -> np.ndarray:
    """Bone indices for foot or hand endpoint landmarks on the source rig."""
    from hhtools.retarget.newton_basic.human_aliases import auto_source_to_canonical

    src2can = auto_source_to_canonical(tuple(bone_names))
    if kind == "foot":
        want = frozenset({"left_ankle", "right_ankle", "left_foot", "right_foot"})
    elif kind == "hand":
        want = frozenset({"left_wrist", "right_wrist", "left_hand", "right_hand"})
    else:
        raise ValueError(f"unknown endpoint kind {kind!r}")
    idx: list[int] = []
    for i, name in enumerate(bone_names):
        canon = str(src2can.get(name, name)).lower()
        if canon in want:
            idx.append(i)
    return np.asarray(idx, dtype=np.int64)


def _endpoint_overlay_z_correction(
    motion: Motion,
    scaler,
    ratio: float,
    *,
    frame: int = 0,
) -> float:
    """Vertical shift for the yellow overlay after uniform scale.

    :func:`human_source_floor_z_world` already subtracts the clip-wide lowest
    joint, so the global minimum rests on ``z=0``; no extra foot-hub snap.
    """
    _ = (motion, scaler, ratio, frame)
    return 0.0


def _snap_scaled_overlay_positions_to_foot_floor(
    positions: np.ndarray,
    joint_names: tuple[str, ...] | list[str],
) -> np.ndarray:
    """Shift overlay Z so preferred foot/ankle hubs meet the ground grid."""
    from hhtools.core.grounding import foot_floor_z_in_positions

    pos = np.asarray(positions, dtype=np.float32)
    z_floor = float(foot_floor_z_in_positions(pos, tuple(joint_names)))
    if abs(z_floor) <= 1e-6:
        return pos
    out = pos.copy()
    out[:, :, 2] -= np.float32(z_floor)
    return out


def resolve_scaled_overlay_z_correction(
    motion: Motion,
    scaler,
    ratio: float,
) -> float:
    """Vertical shift for the yellow skeleton overlay vs uniform scaling alone.

    Clips with terrain and/or interaction props must **not** apply this correction:
    the overlay must stay co-aligned with the uniformly-scaled scene geometry
    (``scaled_scene`` terrain / objects use the same ``z_min`` + ``ratio`` chain
    as :meth:`InteractionMeshPipeline._build_scaled_source_pose`).
    """
    if motion_has_interaction_scene(motion):
        return 0.0
    return _endpoint_overlay_z_correction(motion, scaler, ratio)


def _scaler_effector_joint_positions(
    motion: Motion,
    scaler,
    joint_names: tuple[str, ...] | list[str],
) -> np.ndarray:
    """World positions from :meth:`HumanToRobotScaler.apply`, source heading.

    The scaler pre-rotates by ``source_body_quat`` for IK; the solved robot
    root is counter-rotated back (see
    :meth:`NewtonBasicPipeline._align_root_to_source_heading`).  Apply the
    same undo here so the yellow overlay faces the same way as the robot mesh.
    """
    from hhtools.retarget.newton_basic.heading_align import (
        align_effector_tensor_to_source_heading,
    )

    eff = scaler.apply(motion)
    name_to_col = {n: i for i, n in enumerate(scaler.joint_names)}
    cols = np.asarray([name_to_col[n] for n in joint_names], dtype=np.int64)
    subset = eff.transforms[:, cols, :]
    aligned = align_effector_tensor_to_source_heading(
        subset,
        source_body_quat=np.asarray(scaler.config.source_body_quat, dtype=np.float32),
    )
    return aligned[:, :, :3].astype(np.float32, copy=False)


def _uniform_scaled_joint_positions(
    motion: Motion,
    scaler_cfg,
    human_height: float,
    joint_names: tuple[str, ...] | list[str],
    *,
    ik_canons: frozenset[str],
    z_correction: float = 0.0,
    robot_model=None,
) -> np.ndarray:
    """Uniform ``robot_height / human_height`` positions for yellow overlay joints.

    Preserves the established preview behavior: dense rigs (holosoma, OMOMO,
    meshmimic, SMPL) keep source bone proportions; per-joint soma-style
    ``scaler.apply`` targets are only exact at calibration rest and distort
    limb lengths on motion frames (elongated arms, puffed torso).

    ``z_correction`` (from :func:`resolve_scaled_overlay_z_correction`) snaps
    foot endpoints to ``z=0`` after uniform scale.  Hands inherit the same
    transform; COM/pelvis is not shifted independently.

    When ``robot_model`` is set, OmniContact / 2-spine Mixamo clips also get
    the same shoulder-pivoted arm reach boost as
    :meth:`InteractionMeshPipeline._build_scaled_source_pose`.
    """
    from hhtools.retarget.calibration.calibration import uniform_overlay_scale_for_motion

    ratio = float(
        uniform_overlay_scale_for_motion(
            scaler_cfg, human_height, motion, ik_map_keys=ik_canons,
        )
    )
    # Flat ground: match Newton ``_floor_normalize_motion`` (foot floor when
    # upright stance exists).  Terrain/object clips keep the all-joint floor
    # so the yellow overlay stays co-aligned with scaled scene geometry.
    if motion_has_interaction_scene(motion):
        z_min = float(human_source_floor_z_world(motion))
    else:
        z_min = float(retarget_source_floor_z_world(motion))
    src_pos = np.asarray(motion.positions, dtype=np.float32).copy()
    src_pos[:, :, 2] -= z_min
    src_pos *= ratio
    if robot_model is not None:
        from hhtools.retarget.interaction_mesh.arm_reach import (
            maybe_boost_arm_reach_positions,
        )

        src_pos = maybe_boost_arm_reach_positions(src_pos, motion, robot_model)
    if abs(z_correction) > 1e-6:
        src_pos[:, :, 2] += np.float32(z_correction)

    hname_to_idx = {n: i for i, n in enumerate(motion.hierarchy.bone_names)}
    parents_h = np.asarray(motion.hierarchy.parent_indices, dtype=np.int64)
    root_arr = np.where(parents_h < 0)[0]
    root_idx = int(root_arr[0]) if root_arr.size > 0 else 0
    mapped = np.asarray(
        [hname_to_idx.get(n, root_idx) for n in joint_names],
        dtype=np.int64,
    )
    return src_pos[:, mapped, :].astype(np.float32, copy=False)


def _uniform_scaled_preview_fallback(
    motion: Motion,
    scaler_cfg,
    human_height: float,
    ik_canons: frozenset[str],
    *,
    max_frames: int = 0,
    z_correction: float = 0.0,
    robot_model=None,
) -> dict[str, Any]:
    """Numpy-only scaled overlay when ``newton`` is not installed."""
    from hhtools.io.scene_serialize import _downsample_indices

    jn = list(motion.hierarchy.bone_names)
    pos = _uniform_scaled_joint_positions(
        motion, scaler_cfg, human_height, jn,
        ik_canons=ik_canons, z_correction=z_correction,
        robot_model=robot_model,
    )
    idx = _downsample_indices(int(pos.shape[0]), max_frames, motion=motion)
    pos = pos[idx]
    parents = motion.hierarchy.parent_indices.tolist()
    return {
        "name": "scaled_targets",
        "bone_names": jn,
        "parent_indices": parents,
        "frame_indices": idx.tolist(),
        "positions": np.round(pos, 4).tolist(),
        "num_frames_total": int(motion.num_frames),
        "framerate": float(motion.framerate),
        "duration": float(motion.duration),
    }


def compute_web_scaled_preview(
    model,
    motion: Motion,
    reference: str,
    human_height: float,
    *,
    max_frames: int = 0,
) -> dict[str, Any]:
    """Build scaled skeleton payload for the browser (dense topology when possible)."""
    from hhtools.io.scene_serialize import _downsample_indices, serialize_scaled_preview
    from hhtools.retarget.newton_basic.scaler import HumanToRobotScaler

    scaler_cfg = resolve_web_scaler_config(
        model, motion, reference, float(human_height),
    )
    from hhtools.robot.ik_map_policy import ik_map_canonicals_for_motion

    ik_canons = ik_map_canonicals_for_motion(
        model.preset.name, model.preset.ik_map, motion,
    )

    if int(motion.num_bones) < 10:
        try:
            from hhtools.retarget.newton_basic import NewtonBasicPipeline, PipelineConfig

            pipeline = NewtonBasicPipeline(
                model,
                scaler_config=scaler_cfg,
                pipeline_config=PipelineConfig(ik_iterations=1),
                human_height=float(human_height),
                configure_warp=False,
            )
            return serialize_scaled_preview(
                pipeline.scale_only(motion),
                max_frames=max_frames,
                ik_map_canonicals=ik_canons,
            )
        except ModuleNotFoundError:
            return _uniform_scaled_preview_fallback(
                motion, scaler_cfg, float(human_height), ik_canons,
                max_frames=max_frames, robot_model=model,
            )

    from hhtools.retarget.calibration.calibration import uniform_overlay_scale_for_motion

    scaler = HumanToRobotScaler(
        motion.hierarchy, scaler_cfg, human_height=float(human_height),
    )
    overlay_ratio = float(
        uniform_overlay_scale_for_motion(
            scaler_cfg, float(human_height), motion, ik_map_keys=ik_canons,
        )
    )
    jn = list(scaler.joint_names)
    seg_s, seg_d = _scaler_skeleton_segment_indices(jn, motion.hierarchy, ik_map_canonicals=ik_canons)
    if int(seg_s.size) == 0:
        try:
            from hhtools.retarget.newton_basic import NewtonBasicPipeline, PipelineConfig

            pipeline = NewtonBasicPipeline(
                model,
                scaler_config=scaler_cfg,
                pipeline_config=PipelineConfig(ik_iterations=1),
                human_height=float(human_height),
                configure_warp=False,
            )
            return serialize_scaled_preview(
                pipeline.scale_only(motion),
                max_frames=max_frames,
                ik_map_canonicals=ik_canons,
            )
        except ModuleNotFoundError:
            return _uniform_scaled_preview_fallback(
                motion, scaler_cfg, float(human_height), ik_canons,
                max_frames=max_frames,
                z_correction=resolve_scaled_overlay_z_correction(
                    motion, scaler, overlay_ratio,
                ),
                robot_model=model,
            )

    # Dense source topology always uses uniform ``robot_height / human_height``
    # scaling on raw source positions.  Per-joint soma-style scaler targets are
    # only exact at calibration rest — on motion frames they distort segment
    # lengths (puffed torso, elongated limbs) and sit at robot pelvis height
    # instead of the grounded blue source skeleton.
    z_correction = resolve_scaled_overlay_z_correction(
        motion, scaler, overlay_ratio,
    )
    pos_m = _uniform_scaled_joint_positions(
        motion, scaler_cfg, float(human_height), jn,
        ik_canons=ik_canons, z_correction=z_correction,
        robot_model=model,
    )
    pos_m = _snap_scaled_overlay_positions_to_foot_floor(pos_m, jn)
    include_head_neck = False

    vis_idx = _visible_joint_indices(
        jn, ik_canons, include_unmapped_head_neck=include_head_neck,
    )
    if vis_idx.size == 0:
        vis_idx = np.arange(len(jn), dtype=np.int32)

    # Remap segments to visible joint subset.
    old_to_new = {int(old): int(new) for new, old in enumerate(vis_idx.tolist())}
    bone_names = [jn[int(i)] for i in vis_idx.tolist()]
    seg_pairs: list[tuple[int, int]] = []
    overlay_exclude = (
        set()
        if motion_has_interaction_scene(motion)
        else scaled_overlay_exclude_bone_indices(motion, ik_canons)
    )
    from hhtools.retarget.newton_basic.human_aliases import auto_source_to_canonical

    src2can = auto_source_to_canonical(tuple(motion.hierarchy.bone_names))
    can_to_hidx: dict[str, int] = {}
    for i, raw in enumerate(motion.hierarchy.bone_names):
        can_to_hidx[str(src2can.get(raw, raw)).lower()] = i
    for s, d in zip(seg_s.tolist(), seg_d.tolist(), strict=True):
        if int(s) not in old_to_new or int(d) not in old_to_new:
            continue
        hd = str(jn[int(d)]).lower()
        hi_d = can_to_hidx.get(hd)
        if hi_d is not None and int(hi_d) in overlay_exclude:
            continue
        seg_pairs.append((old_to_new[int(s)], old_to_new[int(d)]))
    parent_indices = [-1] * len(bone_names)
    for s, d in seg_pairs:
        parent_indices[int(d)] = int(s)

    positions = pos_m[:, vis_idx, :]
    idx = _downsample_indices(int(positions.shape[0]), max_frames, motion=motion)
    positions = positions[idx]

    return {
        "name": "scaled_targets",
        "bone_names": bone_names,
        "parent_indices": parent_indices,
        "frame_indices": idx.tolist(),
        "positions": np.round(positions, 4).tolist(),
        "num_frames_total": int(motion.num_frames),
        "framerate": float(motion.framerate),
        "duration": float(motion.duration),
    }
