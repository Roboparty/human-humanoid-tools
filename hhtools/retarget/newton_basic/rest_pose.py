"""Unified source rest-pose extraction.

Every retarget pipeline needs to answer a single question before it can
run any scaling math: *what does this source skeleton look like at
rest?*  ``soma-retargeter`` answers it by shipping a dedicated rest BVH
per source rig (``soma_zero_frame0.bvh`` for SOMA, etc.) and loading
that once per batch.  Our pipeline has previously skipped the question,
scaling motions directly against a generic canonical T-pose — which is
why subject-dependent anthropometry (tall vs. short AMASS subjects,
different SOMA skeletons, LAFAN reference offsets) silently produced
wrong bone-length ratios.

This module introduces :class:`SourceRestPose`, a minimal struct that
captures a single "rest" snapshot (positions + quaternions + root)
regardless of the source format, plus a handful of constructors:

* :func:`rest_pose_from_motion` — take frame ``N`` of an already-imported
  :class:`~hhtools.core.motion.Motion`; the common case for AMASS /
  SOMA batches where frame 0 of the clip itself is already near-rest.
* :func:`rest_pose_from_reference` — derive from a
  :class:`~hhtools.retarget.calibration.reference.HumanReferencePose`
  (useful for unit tests and for when the user doesn't have a rest clip
  but wants to drive calibration off the canonical / SMPL-X T-pose).

All downstream math in
:func:`hhtools.retarget.calibration.calibration.build_scaler_config_soma_style`
consumes :class:`SourceRestPose` — the same dataclass whether the rest
comes from a BVH clip or a synthetic reference.  This keeps the
calibration closed-form and source-format-agnostic.

Why a standalone module (and not just a helper on :class:`Motion`)?  The
rest pose is a *pipeline* concept: it's the fixed point that defines
what "zero motion" means for a given subject.  Tying it to ``Motion``
would force a per-frame interpretation, which is wrong when the rest
comes from a separate file or from a parametric body model forward
pass at zero pose.  The dataclass stays deliberately minimal so future
adapters (SMPL-X θ=0 forward pass, BVH ``OFFSETS``-only parse) can slot
in without churning consumer code.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from hhtools.core.math import quaternion as Q
from hhtools.core.coord import rotate_y_up_to_z_up_quaternions

if TYPE_CHECKING:  # pragma: no cover — import only for type checkers
    from hhtools.core.motion import Motion
    from hhtools.retarget.calibration.reference import HumanReferencePose


__all__ = [
    "SourceRestPose",
    "bundled_reference_bvh_path",
    "rest_pose_from_bundled_reference",
    "rest_pose_from_motion",
    "rest_pose_from_motion_bind",
    "rest_pose_from_reference",
]

# Upstream soma-retargeter ships ``soma/soma_zero_frame0.bvh``.  We bundle the
# same file under ``assets/reference_poses/`` so SOMA calibration and scaler
# derivation always use the format's canonical rest pose — not an arbitrary
# clip's frame 0 (which may be mid-stride, crouched, etc.).
_BUNDLED_REFERENCE_BVH: dict[str, str] = {
    "soma_bvh": "soma_zero_frame0.bvh",
    "xsens_mocap": "xsens_mocap_zero_frame0.bvh",
}


def _reference_poses_dir() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "assets" / "reference_poses"
        if candidate.is_dir():
            return candidate
    return None


def bundled_reference_bvh_path(reference: str) -> Path | None:
    """Return the bundled zero-frame BVH for ``soma_bvh`` / ``xsens_mocap``."""

    rel = _BUNDLED_REFERENCE_BVH.get(reference)
    if rel is None:
        return None
    root = _reference_poses_dir()
    if root is None:
        return None
    path = (root / rel).resolve()
    return path if path.is_file() else None


@lru_cache(maxsize=2)
def rest_pose_from_bundled_reference(
    reference: Literal["soma_bvh", "xsens_mocap"],
) -> SourceRestPose:
    """Load a bundled zero-frame BVH as the source rest pose."""

    path = bundled_reference_bvh_path(reference)
    if path is None:
        raise FileNotFoundError(
            f"no bundled zero-frame BVH for reference {reference!r} "
            f"(expected under assets/reference_poses/)"
        )
    from hhtools.io.bvh import load_bvh

    motion = load_bvh(path)
    return rest_pose_from_motion(
        motion,
        frame=0,
        source_tag=f"bundled_reference:{reference}",
    )


@dataclass(frozen=True)
class SourceRestPose:
    """Snapshot of a source skeleton at its rest configuration.

    Every field is expressed in the **same world frame** as
    :class:`~hhtools.core.motion.Motion` (Z-up, xyzw quaternions) so
    callers never have to remember which rig used Y-up etc. — the
    importer side already normalised.

    Attributes
    ----------
    bone_names
        Ordered bone names; indexing matches ``positions`` /
        ``quaternions``.  Same order as the source ``Hierarchy`` this
        rest came from, so indices are reusable without re-lookup.
    parent_names
        Parent of each bone; ``None`` marks the root.  Kept around so
        downstream code can reason about the skeleton's topology
        without re-importing the full hierarchy.
    positions
        ``(B, 3)`` world-space joint positions at rest, metres.
    quaternions
        ``(B, 4)`` xyzw world-space joint quaternions at rest.
    root_name
        Name of the root bone (``bone_names[i]`` with
        ``parent_names[i] is None``).  Retained as a name rather than
        an index so tests and callers can compare against AMASS-style
        ``"pelvis"`` / SOMA-style ``"Hips"`` without re-searching.
    height_m
        Measured total height (``max(z) - min(z)`` across all bones).
        Used by :class:`~hhtools.retarget.newton_basic.config.ScalerConfig`
        as ``human_height_assumption`` so the runtime ratio
        ``human_height / human_height_assumption`` defaults to ``1.0``
        when the subject matches this rest pose — i.e. the scaler
        doesn't double-correct body size.
    source
        Provenance tag (``"motion_frame0"``, ``"reference_pose"``, …).
        Only consumed by logging / diagnostics; the math is the same
        regardless.
    """

    bone_names: tuple[str, ...]
    parent_names: tuple[str | None, ...]
    positions: NDArray
    quaternions: NDArray
    root_name: str
    height_m: float
    source: str = "unknown"

    def index(self, name: str) -> int:
        """Bone index by name, or ``-1`` if not present."""
        try:
            return self.bone_names.index(name)
        except ValueError:
            return -1


def _measure_height(
    positions: NDArray,
    parent_indices: NDArray | None = None,
) -> float:
    """Best-effort skeleton height.

    Primary method: ``max(z) − min(z)`` across all joints.  This works
    well for standing / T-pose configurations.

    For BVH zero-rotation bind poses where all bones extend along their
    local axis (giving near-zero Z-extent), falls back to **chain-length
    height** — sum of segment lengths along the two longest sub-chains
    emanating from the root — which approximates head-to-hips +
    hips-to-ankle regardless of orientation.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.size == 0:
        return 1.7
    z_height = float(pos[:, 2].max() - pos[:, 2].min())

    if z_height > 0.5 or parent_indices is None:
        return max(z_height, 1e-3)

    # Chain-length fallback for degenerate Z-extent.
    pi = np.asarray(parent_indices, dtype=np.int64)
    N = len(pos)
    root_dist = np.zeros(N, dtype=np.float64)
    for j in range(N):
        p = int(pi[j])
        if p < 0:
            continue
        root_dist[j] = root_dist[p] + float(np.linalg.norm(pos[j] - pos[p]))

    root = int(np.where(pi < 0)[0][0]) if (pi < 0).any() else 0

    # Group deepest reach per first-generation subtree off root.
    first_gen: dict[int, float] = {}
    child_first: np.ndarray = np.full(N, -1, dtype=np.int64)
    for j in range(N):
        if j == root:
            continue
        k = j
        while int(pi[k]) != root and int(pi[k]) >= 0:
            k = int(pi[k])
        child_first[j] = k
        first_gen[k] = max(first_gen.get(k, 0.0), root_dist[j])

    depths = sorted(first_gen.values(), reverse=True)
    if len(depths) >= 2:
        chain_h = float(depths[0] + depths[1])
    elif depths:
        chain_h = float(depths[0]) * 2
    else:
        chain_h = z_height

    return max(chain_h, z_height, 0.5)


def rest_pose_from_motion(
    motion: "Motion",
    frame: int = 0,
    *,
    source_tag: str | None = None,
) -> SourceRestPose:
    """Build a :class:`SourceRestPose` from frame ``frame`` of ``motion``.

    The default (``frame=0``) is the most common case — AMASS / SOMA /
    LAFAN clips almost always start in a near-T / near-A pose.  Use a
    different frame only when you know the clip's frame 0 is pose-y
    (e.g. "start-of-dance" captures that skip the neutral pre-roll).

    The returned snapshot copies the selected frame's data so subsequent
    mutations of ``motion`` (e.g. feet stabilisation) don't retroactively
    change what "rest" means.
    """

    F = motion.positions.shape[0]
    if F == 0:
        raise ValueError(
            f"Cannot extract rest pose: motion {motion.name!r} has 0 frames"
        )
    if not 0 <= frame < F:
        raise IndexError(
            f"frame {frame} out of range for motion {motion.name!r} "
            f"(nframes={F})"
        )

    positions = np.asarray(motion.positions[frame], dtype=np.float32).copy()
    quats = Q.normalize(
        np.asarray(motion.quaternions[frame], dtype=np.float32).copy()
    )

    bone_names = tuple(motion.hierarchy.bone_names)
    parent_names = tuple(motion.hierarchy.parent_names)

    root_idx = int(np.where(motion.hierarchy.parent_indices < 0)[0][0]) if (
        (motion.hierarchy.parent_indices < 0).any()
    ) else 0
    root_name = bone_names[root_idx]

    parent_idx = np.asarray(motion.hierarchy.parent_indices, dtype=np.int64)
    return SourceRestPose(
        bone_names=bone_names,
        parent_names=parent_names,
        positions=positions,
        quaternions=quats,
        root_name=root_name,
        height_m=_measure_height(positions, parent_idx),
        source=source_tag or f"motion_frame{frame}",
    )


def _is_position_only_motion(motion: "Motion") -> bool:
    """True when all quaternions in the motion are identity (position-only data).

    Position-only datasets (holosoma, OMOMO) set all quaternions to
    ``(0, 0, 0, 1)`` because their source format carries only world-
    space joint positions.  The bind-reconstruction path (zero local
    rotations → FK) degenerates to frame-0 positions on such data,
    giving a rest that reflects the subject's starting *pose* rather
    than an anatomical T-pose.  Callers use this flag to switch to a
    reference-based T-pose synthesiser instead.
    """
    quat = np.asarray(motion.quaternions, dtype=np.float32)
    identity = np.zeros(4, dtype=np.float32)
    identity[3] = 1.0
    sample_frames = min(motion.num_frames, 5)
    for f_idx in range(sample_frames):
        frame_q = quat[f_idx]
        max_dev = float(np.abs(frame_q - identity[None, :]).max())
        if max_dev > 0.01:
            return False
    return True


def _synthesise_tpose_from_bone_lengths(
    motion: "Motion",
    parent_idx: "NDArray[np.int64]",
) -> "NDArray[np.float32]":
    """Build a T-pose from median bone lengths measured across the motion.

    For position-only datasets the bind-reconstruction path is useless
    (all local rotations are already identity), so we instead measure
    each bone's length across all frames and place joints in a standard
    T-pose layout, preserving the source skeleton's actual proportions.

    The T-pose convention:
    - Root (Hips) at origin
    - Spine chain goes straight up (+Z)
    - Arms extend laterally (±X at shoulder height)
    - Legs extend straight down (-Z)
    """
    from hhtools.retarget.newton_basic.human_aliases import (
        auto_source_to_canonical,
    )

    bone_names = tuple(motion.hierarchy.bone_names)
    N = len(bone_names)
    positions = np.asarray(motion.positions, dtype=np.float32)
    F = positions.shape[0]

    bone_lengths = np.zeros(N, dtype=np.float32)
    for j in range(N):
        p = int(parent_idx[j])
        if p < 0:
            continue
        diffs = positions[:, j, :] - positions[:, p, :]
        lengths = np.linalg.norm(diffs, axis=-1)
        bone_lengths[j] = float(np.median(lengths))

    src2can = auto_source_to_canonical(bone_names)

    # Enforce left-right symmetry: average paired bone lengths so the
    # synthesized T-pose is perfectly symmetric regardless of asymmetric
    # motion content (e.g. climbing with one arm extended).
    _can_to_indices: dict[str, list[int]] = {}
    for j in range(N):
        _can_to_indices.setdefault(src2can.get(bone_names[j], bone_names[j]), []).append(j)

    _LR_SUFFIXES = (
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_collar", "right_collar"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
        ("left_foot", "right_foot"),
    )
    for left_can, right_can in _LR_SUFFIXES:
        l_indices = _can_to_indices.get(left_can, [])
        r_indices = _can_to_indices.get(right_can, [])
        if l_indices and r_indices:
            avg = 0.5 * (bone_lengths[l_indices[0]] + bone_lengths[r_indices[0]])
            for idx in l_indices:
                bone_lengths[idx] = avg
            for idx in r_indices:
                bone_lengths[idx] = avg

    _TPOSE_DIRECTIONS: dict[str, tuple[float, float, float]] = {
        "hips": (0.0, 0.0, 0.0),
        "spine": (0.0, 0.0, 1.0),
        "chest": (0.0, 0.0, 1.0),
        "neck": (0.0, 0.0, 1.0),
        "head": (0.0, 0.0, 1.0),
        "left_collar": (-0.7, 0.0, 0.7),
        "right_collar": (0.7, 0.0, 0.7),
        "left_shoulder": (-1.0, 0.0, 0.0),
        "left_elbow": (-1.0, 0.0, 0.0),
        "left_wrist": (-1.0, 0.0, 0.0),
        "right_shoulder": (1.0, 0.0, 0.0),
        "right_elbow": (1.0, 0.0, 0.0),
        "right_wrist": (1.0, 0.0, 0.0),
        "left_hip": (-0.5, 0.0, -0.87),
        "left_knee": (0.0, 0.0, -1.0),
        "left_ankle": (0.0, 0.0, -1.0),
        "left_foot": (0.0, 1.0, 0.0),
        "right_hip": (0.5, 0.0, -0.87),
        "right_knee": (0.0, 0.0, -1.0),
        "right_ankle": (0.0, 0.0, -1.0),
        "right_foot": (0.0, 1.0, 0.0),
    }

    rest_pos = np.zeros((N, 3), dtype=np.float32)
    placed = np.zeros(N, dtype=bool)

    root_idx = int(np.where(parent_idx < 0)[0][0]) if (parent_idx < 0).any() else 0
    rest_pos[root_idx] = 0.0
    placed[root_idx] = True

    topo_order = []
    visited = set()
    def _visit(idx: int) -> None:
        if idx in visited:
            return
        visited.add(idx)
        p = int(parent_idx[idx])
        if p >= 0 and p not in visited:
            _visit(p)
        topo_order.append(idx)
    for j in range(N):
        _visit(j)

    for j in topo_order:
        p = int(parent_idx[j])
        if p < 0:
            continue
        canonical = src2can.get(bone_names[j], bone_names[j])
        direction = _TPOSE_DIRECTIONS.get(canonical)
        length = bone_lengths[j]

        if direction is not None:
            d = np.asarray(direction, dtype=np.float32)
            d_norm = float(np.linalg.norm(d))
            if d_norm > 1e-6:
                d = d / d_norm
            rest_pos[j] = rest_pos[p] + d * length
        else:
            parent_canonical = src2can.get(bone_names[p], bone_names[p])
            parent_dir = _TPOSE_DIRECTIONS.get(parent_canonical)
            if parent_dir is not None:
                d = np.asarray(parent_dir, dtype=np.float32)
                d_norm = float(np.linalg.norm(d))
                if d_norm > 1e-6:
                    d = d / d_norm
                rest_pos[j] = rest_pos[p] + d * length
            else:
                rest_pos[j] = rest_pos[p] + np.array([0.0, 0.0, length * 0.5], dtype=np.float32)

        placed[j] = True

    zmin = float(rest_pos[:, 2].min())
    if zmin < 0:
        rest_pos[:, 2] -= zmin

    return rest_pos


def rest_pose_from_motion_bind(
    motion: "Motion",
    *,
    source_tag: str | None = None,
) -> SourceRestPose:
    """Synthesise a T-pose rest from the motion's **bind** geometry.

    Unlike :func:`rest_pose_from_motion` — which grabs frame 0 verbatim
    and therefore bakes the subject's *starting pose* (e.g. a crouch or
    a "hands raised to pick up" configuration) into the scaler's notion
    of rest — this helper reconstructs the BVH bind pose: every non-root
    joint's **local** rotation is set to identity, positions are
    propagated by forward kinematics from the per-joint bone offsets,
    and only the loader-applied Y→Z (or similar) up-axis alignment is
    preserved on the root.  The result is an anatomically meaningful
    T-pose that reflects the source skeleton's proportions without any
    subject-specific pose bias.

    This matches what ``soma-retargeter`` does by shipping a dedicated
    zero-frame BVH (``soma/soma_zero_frame0.bvh`` for SOMA clips,
    ``lafan1/lafan1_zero_frame0.bvh`` for LAFAN) — except we derive it
    on-the-fly from the motion itself, which removes the need for a
    paired rest file per dataset.  See
    :func:`hhtools.retarget.calibration.calibration.build_scaler_config_soma_style`
    for why this is critical: the scaler solves ``q_offset[j]`` so
    that ``q_src[j, t_rest] · q_offset[j] == q_rbt_rest[j]``; if
    ``t_rest`` is a pose-y frame, then every frame that differs from
    that pose gets re-interpreted as a delta away from the robot's
    rest, producing visually collapsed starting frames.

    For **position-only** datasets (holosoma, OMOMO) where all
    quaternions are identity, the standard inverse-FK→zero-rotation→FK
    path degenerates to frame-0 positions — which are NOT a T-pose.
    In that case we synthesise a proper T-pose from measured bone
    lengths, placing joints in a canonical T-pose layout that matches
    the proportions the calibration reference was computed against.

    Implementation notes
    --------------------
    * Per-joint **local** offsets + rotations are recovered from frame 0
      via inverse-FK.  Bone offsets are frame-invariant by BVH spec so
      any frame would work; frame 0 keeps the operation deterministic.
    * We then set every non-root local quaternion to identity and
      re-run forward kinematics.  The root's local quaternion is kept
      because the BVH loader pre-multiplies the global Y→Z up-axis
      conversion into it — clearing it would re-orient the entire
      skeleton onto the source's original up axis (typically +Y)
      instead of the hhtools convention (+Z).
    * The root's world translation is zeroed on the horizontal plane
      and reset so the skeleton's lowest joint sits at ``z = 0``, which
      keeps :attr:`SourceRestPose.height_m` meaningful without a
      subsequent ground-align step.
    * **Position-only fallback**: when all quaternions are identity
      (detected by :func:`_is_position_only_motion`), bone lengths are
      measured from frame-0 positions and joints are placed in a
      standard T-pose layout, so the scaler sees arm displacements
      that match the SMPL T-pose calibration reference.

    Parameters
    ----------
    motion
        A loaded :class:`~hhtools.core.motion.Motion`; must have at
        least one frame.
    source_tag
        Optional provenance override; defaults to
        ``"motion_bind_from:<name>"``.
    """

    F = motion.positions.shape[0]
    if F == 0:
        raise ValueError(
            f"Cannot build bind rest pose: motion {motion.name!r} has 0 frames"
        )

    hierarchy = motion.hierarchy
    N = hierarchy.num_bones
    world_pos = np.asarray(motion.positions[0], dtype=np.float32)
    world_quat = Q.normalize(
        np.asarray(motion.quaternions[0], dtype=np.float32)
    )
    parent_idx = np.asarray(hierarchy.parent_indices, dtype=np.int64)

    # ---- Position-only fallback ------------------------------------------
    # For datasets with all-identity quaternions (holosoma, OMOMO), the
    # standard bind reconstruction gives back frame-0 positions unchanged,
    # which is typically NOT a T-pose.  Synthesise one from bone lengths.
    if _is_position_only_motion(motion):
        rest_pos = _synthesise_tpose_from_bone_lengths(motion, parent_idx)
        rest_quat = np.zeros((N, 4), dtype=np.float32)
        rest_quat[:, 3] = 1.0

        bone_names = tuple(hierarchy.bone_names)
        parent_names = tuple(hierarchy.parent_names)
        root_idx_arr = np.where(parent_idx < 0)[0]
        root_idx = int(root_idx_arr[0]) if root_idx_arr.size > 0 else 0
        root_name = bone_names[root_idx]

        height = _measure_height(rest_pos, parent_idx)

        # For position-only data the frame-0 Z-extent is a more reliable
        # height measurement (character is standing) than the synthesized
        # T-pose (which uses hardcoded direction vectors).  However the
        # frame-0 extent must be measured as the subject's actual stature
        # (head-to-foot), not head-to-world-origin.  Use the per-frame
        # pelvis-to-extremity span to avoid being thrown off by a non-zero
        # ground-plane offset.
        _all_pos = np.asarray(motion.positions, dtype=np.float32)
        _sample_n = min(motion.num_frames, 10)
        _sample_z_extents = []
        for _fi in range(_sample_n):
            _fz = _all_pos[_fi, :, 2]
            _sample_z_extents.append(float(_fz.max() - _fz.min()))
        frame0_height = float(np.median(_sample_z_extents)) if _sample_z_extents else 0.0

        if frame0_height > 0.5:
            height = max(height, frame0_height)

        return SourceRestPose(
            bone_names=bone_names,
            parent_names=parent_names,
            positions=rest_pos,
            quaternions=rest_quat,
            root_name=root_name,
            height_m=height,
            source=source_tag or f"motion_bind_tpose_synth:{motion.name}",
        )

    # ---- Inverse-FK: recover parent-local positions + quaternions. -------
    # Frame-0 world FK obeys
    #   world_pos[j]  = world_pos[parent]  + rotate(world_quat[parent], local_pos[j])
    #   world_quat[j] = world_quat[parent] ·  local_quat[j]
    # so:
    #   local_pos[j]  = rotate(conj(world_quat[parent]), world_pos[j] − world_pos[parent])
    #   local_quat[j] = conj(world_quat[parent]) · world_quat[j]
    # The root's "parent" is the world; local == world for the root.
    local_pos = np.zeros((N, 3), dtype=np.float32)
    local_quat = np.zeros((N, 4), dtype=np.float32)
    local_quat[:, 3] = 1.0
    root_idx_arr = np.where(parent_idx < 0)[0]
    root_idx = int(root_idx_arr[0]) if root_idx_arr.size > 0 else 0
    for j in range(N):
        parent = int(parent_idx[j])
        if parent < 0:
            local_pos[j] = world_pos[j]
            local_quat[j] = world_quat[j]
            continue
        diff = (world_pos[j] - world_pos[parent])[None, :].astype(np.float32)
        q_parent = world_quat[parent][None, :]
        local_pos[j] = Q.rotate(Q.conjugate(q_parent), diff)[0]
        local_quat[j] = Q.multiply(
            Q.conjugate(q_parent), world_quat[j][None, :]
        )[0]
    local_quat = Q.normalize(local_quat)

    # ---- Zero every non-root local rotation; keep root's when needed. ----
    # Non-root local quats at identity collapse the per-subject pose to a
    # canonical T-pose (arms stretched out, legs straight).
    #
    # BVH-style loaders may bake an up-axis conversion into the root; keep
    # that so the synthesized rest pose stays upright.  HMR4D/GVHMR clips
    # are special: their root quaternion carries the video-estimated global
    # body orientation, not a rest-frame axis fix-up.  Keeping that tilt
    # makes the synthesized bind pose lean sideways, shrinking the measured
    # Z-height and corrupting both the yellow preview scale and IK targets.
    #
    # Other SMPL-family clips keep the historical behaviour so arbitrary
    # frame-0 root rotations are absorbed into q_offset by calibration.
    bind_local_quat = np.zeros_like(local_quat)
    bind_local_quat[:, 3] = 1.0  # identity by default
    try:
        from hhtools.retarget.newton_basic.human_aliases import is_smpl_like

        smpl_like = is_smpl_like(hierarchy.bone_names)
    except Exception:
        smpl_like = False
    source_format = str(getattr(motion, "source_format", "") or "").lower()
    dataset = str(getattr(motion, "meta", {}).get("dataset", "")).lower()
    hmr_global_orient = smpl_like and (
        "hmr4d" in source_format
        or "gvhmr" in source_format
        or dataset in {"hmr4d", "gvhmr", "kungfu_athlete"}
    )
    if hmr_global_orient:
        if motion.up_axis == "Z":
            bind_local_quat[root_idx] = rotate_y_up_to_z_up_quaternions(
                bind_local_quat[root_idx][None, :]
            )[0]
    else:
        bind_local_quat[root_idx] = local_quat[root_idx]

    # ---- Zero the root's horizontal translation so rest lives at origin. --
    # This keeps ``height_m`` meaningful (no offset accumulated from an
    # arbitrary clip start position) and matches the soma rest-BVH
    # convention where the root is at (0, 0, 0) at bind.
    bind_local_pos = local_pos.copy()
    bind_local_pos[root_idx] = 0.0

    # ---- Forward kinematics with the bind locals. ------------------------
    rest_pos = np.zeros((N, 3), dtype=np.float32)
    rest_quat = np.zeros((N, 4), dtype=np.float32)
    rest_quat[:, 3] = 1.0
    for j in range(N):
        parent = int(parent_idx[j])
        if parent < 0:
            rest_quat[j] = bind_local_quat[j]
            rest_pos[j] = bind_local_pos[j]
            continue
        rest_quat[j] = Q.multiply(
            rest_quat[parent][None, :], bind_local_quat[j][None, :]
        )[0]
        rest_pos[j] = rest_pos[parent] + Q.rotate(
            rest_quat[parent][None, :], bind_local_pos[j][None, :]
        )[0]
    rest_quat = Q.normalize(rest_quat)

    # ---- Ground-align: lift so min(z) == 0, matching the BVH-rest
    # convention used by soma's bundled zero-frame BVH (the subject
    # stands on the floor at rest).  Without this the SMPL-X style root
    # offset in the BVH hierarchy would leave feet below z = 0.
    zmin = float(rest_pos[:, 2].min())
    if zmin < 0:
        rest_pos[:, 2] -= zmin

    bone_names = tuple(hierarchy.bone_names)
    parent_names = tuple(hierarchy.parent_names)
    root_name = bone_names[root_idx]

    # For BVH zero-rotation bind poses the skeleton extends along bone
    # axes → Z-extent can be much smaller than true stature.  Pass the
    # hierarchy so _measure_height can fall back to chain-length.
    height = _measure_height(rest_pos, parent_idx)

    # Extra guard: use frame-0 Z-extent (character is normally standing)
    # when the bind-pose height is still unreasonably small.
    frame0_height = _measure_height(world_pos)
    if frame0_height > height:
        height = frame0_height

    return SourceRestPose(
        bone_names=bone_names,
        parent_names=parent_names,
        positions=rest_pos,
        quaternions=rest_quat,
        root_name=root_name,
        height_m=height,
        source=source_tag or f"motion_bind_from:{motion.name}",
    )


def rest_pose_from_reference(ref: "HumanReferencePose") -> SourceRestPose:
    """Build a :class:`SourceRestPose` from a calibration reference pose.

    Useful when:

    * writing unit tests that want deterministic input (``canonical_human``
      is hard-coded, SMPL-X at betas = 0 is reproducible);
    * the user hasn't authored a rest clip for a new source dataset and
      just wants to fall back to the canonical T-pose while developing
      the pipeline.

    The reference pose's positions are *hips-relative* by convention;
    we lift them into world-space by adding a nominal pelvis-at-origin
    offset — identical to how the calibration-side derivation consumes
    them — so :func:`build_scaler_config_soma_style` sees the same
    numbers whether the rest came from a motion or a reference.
    """

    positions = np.asarray(ref.positions, dtype=np.float32).copy()
    quats = Q.normalize(np.asarray(ref.quaternions, dtype=np.float32).copy())
    parent_names = tuple(p if p else None for p in ref.parent_names)

    return SourceRestPose(
        bone_names=tuple(ref.joint_names),
        parent_names=parent_names,
        positions=positions,
        quaternions=quats,
        root_name=ref.root_joint,
        height_m=float(ref.height_m),
        source=f"reference_pose:{ref.name}",
    )


def estimate_skeleton_height(motion: "Motion") -> float:
    """Estimate the source skeleton's height from its joint positions.

    Returns the *median* per-frame Z-extent (``max(z) - min(z)``) over a
    sample of frames.  This gives a stable measure of how tall the
    skeleton representation is in the motion data — which is the value
    that should be plugged into the "Subject height" field for correct
    uniform scaling.

    Falls back to 1.7 if the motion has no frames.
    """
    pos = np.asarray(motion.positions, dtype=np.float32)
    F = pos.shape[0]
    if F == 0:
        return 1.7
    sample_n = min(F, 30)
    z_exts = np.empty(sample_n, dtype=np.float64)
    for i in range(sample_n):
        z_exts[i] = float(pos[i, :, 2].max() - pos[i, :, 2].min())
    height = float(np.median(z_exts))
    return max(height, 0.3)
