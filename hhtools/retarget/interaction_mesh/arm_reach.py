# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Post-height-scale arm reach boost for short-armed Mixamo / OmniContact.

Interaction-mesh uses a single ``robot_height / human_height`` scale so the
yellow overlay and Laplacian targets stay proportionally consistent.  On
OmniContact (and other 2-spine Mixamo optical mocap) that under-scales the
arms relative to many humanoids (e.g. RP1), so the solver matches short
wrist targets.  After the uniform scale we radially lengthen each arm from
the glenohumeral joint (canonical ``*_shoulder``) until the source
shoulder→elbow→wrist chain matches the robot's.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from hhtools.core.motion import Motion
    from hhtools.robot.loader import URDFRobotModel

_log = logging.getLogger(__name__)

__all__ = [
    "boost_arm_reach_from_shoulders",
    "maybe_boost_arm_reach_positions",
    "needs_arm_reach_boost",
    "robot_shoulder_wrist_chain_m",
    "source_arm_chain_length_m",
]

# Only lengthen; never shrink arms that already match or exceed the robot.
_MIN_BOOST = 1.01
_MAX_BOOST = 1.75


def needs_arm_reach_boost(motion: "Motion") -> bool:
    """True for OmniContact / 2-spine Mixamo clips that need arm lengthening."""
    meta = getattr(motion, "meta", None) or {}
    if str(meta.get("dataset") or "") == "omnicontact":
        return True
    from hhtools.retarget.newton_basic.human_aliases import is_two_segment_mixamo_like

    names = getattr(getattr(motion, "hierarchy", None), "bone_names", None)
    return bool(names) and is_two_segment_mixamo_like(names)


def _resolve_robot_arm_joint_q(
    robot: "URDFRobotModel",
    joint_q: dict[str, float] | None,
) -> dict[str, float] | None:
    """Prefer explicit ``joint_q``, else load a retarget calibration T-pose."""
    if joint_q:
        return dict(joint_q)
    try:
        from hhtools.retarget.calibration import (
            load_calibration,
            resolve_preset_calibration_file,
        )

        urdf = getattr(robot.preset, "urdf_path", None)
        if urdf is None:
            return None
        # OmniContact / Mixamo use lafan_bvh; fall through common refs.
        for ref in ("lafan_bvh", "smpl", "soma_bvh", "omnicontact_bvh"):
            path = resolve_preset_calibration_file(robot.preset, ref)
            if path is None:
                continue
            cal = load_calibration(path)
            q = dict(cal.calibrated_joint_q or {})
            if q:
                return q
    except Exception:
        return None
    return None


def robot_shoulder_wrist_chain_m(
    robot: "URDFRobotModel",
    *,
    joint_q: dict[str, float] | None = None,
) -> float | None:
    """Mean left/right shoulder→wrist length on the robot (metres).

    Prefers FK at the calibration T-pose (``joint_q`` or auto-loaded
    ``retarget_calibration_*.yaml``).  Zero-configuration FK is often a bent
    rest pose and underestimates reach; static URDF chain is the last fallback.
    """
    from hhtools.robot.arm_geometry import (
        _fk_link_distance,
        _ik_shoulder_wrist_links,
        _static_chain_length,
    )

    q = _resolve_robot_arm_joint_q(robot, joint_q)
    lengths: list[float] = []
    for side in ("left", "right"):
        links = _ik_shoulder_wrist_links(robot, side)  # type: ignore[arg-type]
        if links is None:
            continue
        shoulder_link, wrist_link = links
        fk = _fk_link_distance(
            robot, shoulder_link, wrist_link, joint_q=q,
        )
        static = _static_chain_length(robot, shoulder_link, wrist_link)
        # Calibrated T-pose FK matches the yellow/robot overlay the user sees.
        # Only fall back to the static fully-extended chain when FK is missing.
        pick = fk if fk is not None and fk > 1e-4 else static
        if pick is not None and pick > 1e-4:
            lengths.append(float(pick))
    if not lengths:
        return None
    return float(sum(lengths) / len(lengths))


def source_arm_chain_length_m(
    positions: NDArray[np.floating],
    bone_names: Iterable[str],
    *,
    frame_sample: int = 64,
) -> float | None:
    """Median shoulder→elbow→wrist chain length over a frame subsample."""
    from hhtools.retarget.newton_basic.human_aliases import auto_source_to_canonical

    names = tuple(bone_names)
    name_to_can = auto_source_to_canonical(names)
    can_to_idx: dict[str, int] = {}
    for i, name in enumerate(names):
        can = name_to_can.get(name)
        if can and can not in can_to_idx:
            can_to_idx[can] = i

    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[0] == 0:
        return None

    f_count = int(pos.shape[0])
    if f_count <= frame_sample:
        frames = np.arange(f_count, dtype=np.int64)
    else:
        frames = np.linspace(0, f_count - 1, num=frame_sample, dtype=np.int64)

    chains: list[float] = []
    for side in ("left", "right"):
        sh = can_to_idx.get(f"{side}_shoulder")
        el = can_to_idx.get(f"{side}_elbow")
        wr = can_to_idx.get(f"{side}_wrist")
        if sh is None or wr is None:
            continue
        for f in frames:
            p_sh = pos[int(f), sh]
            p_wr = pos[int(f), wr]
            if el is not None:
                p_el = pos[int(f), el]
                length = float(
                    np.linalg.norm(p_el - p_sh) + np.linalg.norm(p_wr - p_el)
                )
            else:
                length = float(np.linalg.norm(p_wr - p_sh))
            if length > 1e-4:
                chains.append(length)
    if not chains:
        return None
    return float(np.median(np.asarray(chains, dtype=np.float64)))


def _descendant_indices(parent_indices: NDArray[np.integer], root: int) -> list[int]:
    """Bone indices strictly descending from ``root`` (excludes ``root``)."""
    parents = np.asarray(parent_indices, dtype=np.int64)
    children: dict[int, list[int]] = {}
    for i, p in enumerate(parents.tolist()):
        if int(p) < 0:
            continue
        children.setdefault(int(p), []).append(int(i))
    out: list[int] = []
    stack = list(children.get(int(root), []))
    while stack:
        j = stack.pop()
        out.append(j)
        stack.extend(children.get(j, []))
    return out


def boost_arm_reach_from_shoulders(
    positions: NDArray[np.floating],
    bone_names: Iterable[str],
    parent_indices: Iterable[int] | NDArray[np.integer],
    *,
    target_arm_length: float,
    source_arm_length: float | None = None,
) -> tuple[NDArray[np.float32], float]:
    """Radially lengthen arm chains from each shoulder toward ``target_arm_length``.

    Returns ``(boosted_positions, boost_factor)``.  ``boost_factor`` is ``1.0``
    when no change is applied (already long enough, or missing joints).
    """
    pos = np.asarray(positions, dtype=np.float32).copy()
    names = tuple(bone_names)
    parents = np.asarray(list(parent_indices), dtype=np.int64)
    target = float(target_arm_length)
    if target <= 1e-4 or pos.ndim != 3:
        return pos, 1.0

    src_len = (
        float(source_arm_length)
        if source_arm_length is not None
        else source_arm_chain_length_m(pos, names)
    )
    if src_len is None or src_len <= 1e-4:
        return pos, 1.0

    boost = float(target / src_len)
    if boost < _MIN_BOOST:
        return pos, 1.0
    if boost > _MAX_BOOST:
        _log.warning(
            "Arm reach boost %.3f exceeds cap %.2f; clamping "
            "(source_chain=%.3f m, target=%.3f m)",
            boost, _MAX_BOOST, src_len, target,
        )
        boost = _MAX_BOOST

    from hhtools.retarget.newton_basic.human_aliases import auto_source_to_canonical

    name_to_can = auto_source_to_canonical(names)
    can_to_idx: dict[str, int] = {}
    for i, name in enumerate(names):
        can = name_to_can.get(name)
        if can and can not in can_to_idx:
            can_to_idx[can] = i

    for side in ("left", "right"):
        sh = can_to_idx.get(f"{side}_shoulder")
        if sh is None:
            continue
        desc = _descendant_indices(parents, sh)
        if not desc:
            continue
        shoulder = pos[:, sh, :]
        for j in desc:
            delta = pos[:, j, :] - shoulder
            pos[:, j, :] = shoulder + np.float32(boost) * delta

    _log.info(
        "Arm reach boost ×%.3f (source_chain=%.3f m → target=%.3f m)",
        boost, src_len, target,
    )
    return pos, boost


def maybe_boost_arm_reach_positions(
    positions: NDArray[np.floating],
    motion: "Motion",
    robot: "URDFRobotModel",
    *,
    joint_q: dict[str, float] | None = None,
) -> NDArray[np.float32]:
    """Apply arm reach boost when ``needs_arm_reach_boost(motion)``; else copy."""
    pos = np.asarray(positions, dtype=np.float32)
    if not needs_arm_reach_boost(motion):
        return pos.copy() if pos.flags.writeable else pos.astype(np.float32, copy=True)

    target = robot_shoulder_wrist_chain_m(robot, joint_q=joint_q)
    if target is None:
        return pos.astype(np.float32, copy=True)

    boosted, _boost = boost_arm_reach_from_shoulders(
        pos,
        motion.hierarchy.bone_names,
        motion.hierarchy.parent_indices,
        target_arm_length=target,
    )
    return boosted
