"""Static and motion-derived human reference poses for retarget calibration.

``HumanReferencePose`` snapshots describe a single known human skeleton layout
(hips-relative positions + world quaternions) for a given source naming
convention — SMPL, SOMA BVH, LAFAN / Mixamo, SMPL-X, or a GLB clip's frame 0.  :mod:`hhtools.retarget.calibration.calibration` consumes these structs
when deriving per-robot scale / offset parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

from hhtools.bodymodels.layout import (
    SMPL_JOINT_NAMES,
    SMPLX_JOINT_NAMES,
    SMPLX_PARENTS,
)
from hhtools.core.coord import (
    rotate_y_up_to_z_up_positions,
    rotate_y_up_to_z_up_quaternions,
)
from hhtools.core.math import quaternion as Q
from hhtools.retarget.newton_basic.human_aliases import (
    MIXAMO_CMU_TO_CANONICAL,
    SMPL_BODY_TO_CANONICAL,
    SOMA_BVH_TO_CANONICAL,
    XSENS_MOCAP_TO_CANONICAL,
    auto_source_to_canonical,
)

if TYPE_CHECKING:
    from hhtools.core.motion import Motion

ReferenceName = Literal[
    "smplx", "smpl", "gvhmr", "soma_bvh", "lafan_bvh", "xsens_mocap", "glb",
]


_LEGACY_REFERENCE_ALIASES: dict[str, str] = {
    "canonical_human": "smpl",
    "mixamo_bvh": "lafan_bvh",
    "fbx": "lafan_bvh",
}

# SMPL / Mixamo-style 17-joint subset (native SMPL names) used for ``smpl`` calibration.
# Topological order: ``spine3`` parents ``spine1`` (``spine2`` omitted from the list).
_SMPL_REF_JOINTS: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "spine3",
    "neck",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

_SMPL_REF_PARENT: dict[str, str | None] = {
    "pelvis": None,
    "left_hip": "pelvis",
    "right_hip": "pelvis",
    "spine1": "pelvis",
    "left_knee": "left_hip",
    "right_knee": "right_hip",
    "left_ankle": "left_knee",
    "right_ankle": "right_knee",
    "spine3": "spine1",
    "neck": "spine3",
    "head": "neck",
    "left_shoulder": "neck",
    "right_shoulder": "neck",
    "left_elbow": "left_shoulder",
    "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow",
    "right_wrist": "right_elbow",
}

# Indices into the 24-joint SMPL layout matching ``_SMPL_REF_JOINTS`` order.
_SMPL_REF_FROM_LAYOUT: tuple[int, ...] = tuple(
    SMPL_JOINT_NAMES.index(n) for n in _SMPL_REF_JOINTS
)


@dataclass(frozen=True)
class HumanReferencePose:
    """Hips-root human layout for calibration (native names + canonical map)."""

    name: str
    root_joint: str
    joint_names: tuple[str, ...]
    parent_names: tuple[str | None, ...]
    positions: NDArray[np.float32]
    quaternions: NDArray[np.float32]
    height_m: float
    source_to_canonical: dict[str, str] = field(default_factory=dict)
    fallback: bool = False

    def __post_init__(self) -> None:
        pos = np.asarray(self.positions, dtype=np.float32)
        quat = np.asarray(self.quaternions, dtype=np.float32)
        j = len(self.joint_names)
        if pos.shape != (j, 3):
            raise ValueError(f"positions shape {pos.shape} != ({j}, 3)")
        if quat.shape != (j, 4):
            raise ValueError(f"quaternions shape {quat.shape} != ({j}, 4)")
        if len(self.parent_names) != j:
            raise ValueError("parent_names length must match joint_names")


def _measure_height(
    positions: NDArray[np.float32],
    parent_indices: NDArray[np.int64] | None = None,
) -> float:
    """``max(z)−min(z)`` with chain-length fallback (see ``rest_pose``)."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.size == 0:
        return 1.7
    z_height = float(pos[:, 2].max() - pos[:, 2].min())
    if z_height > 0.5 or parent_indices is None:
        return max(z_height, 1e-3)
    pi = np.asarray(parent_indices, dtype=np.int64)
    n = len(pos)
    root_dist = np.zeros(n, dtype=np.float64)
    for j in range(n):
        p = int(pi[j])
        if p < 0:
            continue
        root_dist[j] = root_dist[p] + float(np.linalg.norm(pos[j] - pos[p]))
    root = int(np.where(pi < 0)[0][0]) if (pi < 0).any() else 0
    first_gen: dict[int, float] = {}
    for j in range(n):
        if j == root:
            continue
        k = j
        while int(pi[k]) != root and int(pi[k]) >= 0:
            k = int(pi[k])
        first_gen[k] = max(first_gen.get(k, 0.0), root_dist[j])
    depths = sorted(first_gen.values(), reverse=True)
    if len(depths) >= 2:
        chain_h = float(depths[0] + depths[1])
    elif depths:
        chain_h = float(depths[0]) * 2
    else:
        chain_h = z_height
    return max(chain_h, z_height, 0.5)


def _quantize_local_rotation(q: NDArray[np.float32]) -> NDArray[np.float32]:
    """Snap a *local* quaternion to the nearest π/2 rotation about its axis."""
    q = Q.normalize(np.asarray(q, dtype=np.float32).reshape(1, 4))[0]
    aa = Q.to_axis_angle(q.reshape(1, 4))[0]
    ang = float(np.linalg.norm(aa))
    if ang < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = aa / ang
    step = 0.5 * math.pi
    new_ang = round(ang / step) * step
    if abs(new_ang) < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return Q.from_axis_angle((axis * new_ang).reshape(1, 3))[0]


def _parent_in_subset(
    full_names: tuple[str, ...],
    full_parents: tuple[int, ...],
    joint: str,
    subset: frozenset[str],
) -> str | None:
    j = full_names.index(joint)
    p = int(full_parents[j])
    while p >= 0:
        pname = full_names[p]
        if pname in subset:
            return pname
        p = int(full_parents[p])
    return None


def _try_smpl_forward(
    family: Literal["smpl", "smplx"],
) -> tuple[NDArray[np.float32], NDArray[np.float32], tuple[str, ...], tuple[int, ...]] | None:
    """Return (positions, quats, joint_names, parents) in model layout, or ``None``."""
    from hhtools.bodymodels.engine import SmplxEngine
    from hhtools.bodymodels.params import SmplMotionParams

    for gender in ("neutral", "male", "female"):
        try:
            engine = SmplxEngine(family, gender=cast("Literal['neutral','male','female']", gender))
        except (FileNotFoundError, OSError, ImportError, RuntimeError, ValueError):
            continue
        try:
            if family == "smpl":
                body = np.zeros((1, 69), dtype=np.float32)
            else:
                body = np.zeros((1, 63), dtype=np.float32)
            params = SmplMotionParams(
                surface_model=family,
                root_orient=np.zeros((1, 3), dtype=np.float32),
                body_pose=body,
                betas=np.zeros(10, dtype=np.float32),
                trans=np.zeros((1, 3), dtype=np.float32),
                gender=cast("Literal['neutral','male','female']", gender),
            )
            result = engine.forward(params, return_mesh=False)
        except Exception:
            continue
        names = tuple(engine.layout.joint_names)
        parents = engine.layout.parents
        j0 = result.joints[0].astype(np.float32)
        q0 = result.quaternions_global[0].astype(np.float32)
        # MPI SMPL-family joints are authored in a Y-up frame; calibration /
        # viewer assume Z-up (feet low on Z, ``ReferenceSkeletonRenderer`` ground
        # snap uses ``min(z)``). Without this rigid rotation the T-pose lies in the
        # horizontal plane and reads as "on the floor".
        j0 = rotate_y_up_to_z_up_positions(j0)
        q0 = rotate_y_up_to_z_up_quaternions(q0)
        j0 = j0 - j0[0:1]
        return j0, q0, names, parents
    return None


def _smpl17_from_layout(
    joints: NDArray[np.float32],
    quats: NDArray[np.float32],
) -> tuple[
    tuple[str, ...],
    tuple[str | None, ...],
    NDArray[np.float32],
    NDArray[np.float32],
    dict[str, str],
]:
    subset = frozenset(_SMPL_REF_JOINTS)
    idxs = _SMPL_REF_FROM_LAYOUT
    names = _SMPL_REF_JOINTS
    parents: list[str | None] = []
    for n in names:
        parents.append(_SMPL_REF_PARENT[n])
    pos = joints[list(idxs)].copy()
    q = quats[list(idxs)].copy()
    pos -= pos[0:1]
    src2can = {n: SMPL_BODY_TO_CANONICAL[n] for n in names if n in SMPL_BODY_TO_CANONICAL}
    return names, tuple(parents), pos, q, src2can


def _canonical_smpl17_positions() -> NDArray[np.float32]:
    """Fallback Z-up hips-rooted T-pose when SMPL weights are unavailable."""
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-0.095, 0.0, -0.02],
            [0.095, 0.0, -0.02],
            [0.0, 0.0, 0.11],
            [-0.095, 0.0, -0.46],
            [0.095, 0.0, -0.46],
            [-0.095, 0.0, -0.86],
            [0.095, 0.0, -0.86],
            [0.0, 0.0, 0.22],
            [0.0, 0.0, 0.38],
            [0.0, 0.0, 0.62],
            [-0.17, 0.0, 0.38],
            [0.17, 0.0, 0.38],
            [-0.38, 0.0, 0.38],
            [0.38, 0.0, 0.38],
            [-0.58, 0.0, 0.38],
            [0.58, 0.0, 0.38],
        ],
        dtype=np.float32,
    )


def _canonical_soma17_positions() -> NDArray[np.float32]:
    """SOMA-style arms-down rest (distinct elbows from SMPL T-pose).

    Row order matches :data:`_SOMA_REF_JOINTS` (LeftArm … Head), **not**
    the LAFAN/Mixamo spine→neck→arms layout used by
    :func:`_canonical_lafan17_positions`.
    """
    return np.asarray(
        [
            [0.0, 0.0, 0.0],  # Hips
            [-0.085, 0.02, -0.02],  # LeftLeg
            [0.085, 0.02, -0.02],  # RightLeg
            [0.0, 0.0, 0.10],  # Spine1
            [-0.085, 0.02, -0.44],  # LeftShin
            [0.085, 0.02, -0.44],  # RightShin
            [-0.085, 0.02, -0.82],  # LeftFoot
            [0.085, 0.02, -0.82],  # RightFoot
            [0.0, 0.0, 0.20],  # Chest
            [-0.12, 0.05, 0.22],  # LeftArm
            [0.12, 0.05, 0.22],  # RightArm
            [-0.14, 0.06, -0.05],  # LeftForeArm
            [0.14, 0.06, -0.05],  # RightForeArm
            [-0.13, 0.04, -0.28],  # LeftHand
            [0.13, 0.04, -0.28],  # RightHand
            [0.0, 0.0, 0.36],  # Neck1
            [0.0, 0.0, 0.58],  # Head
        ],
        dtype=np.float32,
    )


def _canonical_lafan17_positions() -> NDArray[np.float32]:
    """Mixamo / LAFAN-style near–T-pose (order matches ``_LAFAN_REF_JOINTS``)."""
    return np.asarray(
        [
            [0.0, 0.0, 0.0],  # Hips
            [-0.10, 0.0, -0.02],  # LeftUpLeg
            [0.10, 0.0, -0.02],  # RightUpLeg
            [0.0, 0.0, 0.10],  # Spine
            [-0.10, 0.0, -0.44],  # LeftLeg
            [0.10, 0.0, -0.44],  # RightLeg
            [0.0, 0.0, 0.24],  # Spine2
            [-0.10, 0.0, -0.84],  # LeftFoot
            [0.10, 0.0, -0.84],  # RightFoot
            [0.0, 0.0, 0.38],  # Neck
            [-0.16, 0.0, 0.38],  # LeftArm
            [0.16, 0.0, 0.38],  # RightArm
            [0.0, 0.0, 0.58],  # Head
            [-0.38, 0.0, 0.38],  # LeftForeArm
            [0.38, 0.0, 0.38],  # RightForeArm
            [-0.58, 0.0, 0.38],  # LeftHand
            [0.58, 0.0, 0.38],  # RightHand
        ],
        dtype=np.float32,
    )


_SOMA_REF_JOINTS: tuple[str, ...] = (
    "Hips",
    "LeftLeg",
    "RightLeg",
    "Spine1",
    "LeftShin",
    "RightShin",
    "LeftFoot",
    "RightFoot",
    "Chest",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
    "Neck1",
    "Head",
)

_SOMA_REF_PARENT: dict[str, str | None] = {
    "Hips": None,
    "LeftLeg": "Hips",
    "RightLeg": "Hips",
    "Spine1": "Hips",
    "LeftShin": "LeftLeg",
    "RightShin": "RightLeg",
    "LeftFoot": "LeftShin",
    "RightFoot": "RightShin",
    "Chest": "Spine1",
    "LeftArm": "Chest",
    "RightArm": "Chest",
    "LeftForeArm": "LeftArm",
    "RightForeArm": "RightArm",
    "LeftHand": "LeftForeArm",
    "RightHand": "RightForeArm",
    "Neck1": "Chest",
    "Head": "Neck1",
}

_LAFAN_REF_JOINTS: tuple[str, ...] = (
    "Hips",
    "LeftUpLeg",
    "RightUpLeg",
    "Spine",
    "LeftLeg",
    "RightLeg",
    "Spine2",
    "LeftFoot",
    "RightFoot",
    "Neck",
    "LeftArm",
    "RightArm",
    "Head",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
)

_LAFAN_REF_PARENT: dict[str, str | None] = {
    "Hips": None,
    "LeftUpLeg": "Hips",
    "RightUpLeg": "Hips",
    "Spine": "Hips",
    "LeftLeg": "LeftUpLeg",
    "RightLeg": "RightUpLeg",
    "Spine2": "Spine",
    "LeftFoot": "LeftLeg",
    "RightFoot": "RightLeg",
    "Neck": "Spine2",
    "LeftArm": "Spine2",
    "RightArm": "Spine2",
    "Head": "Neck",
    "LeftForeArm": "LeftArm",
    "RightForeArm": "RightArm",
    "LeftHand": "LeftForeArm",
    "RightHand": "RightForeArm",
}

_XSENS_REF_JOINTS: tuple[str, ...] = (
    "Hips",
    "LeftHip",
    "RightHip",
    "Chest",
    "LeftKnee",
    "RightKnee",
    "Chest4",
    "LeftAnkle",
    "RightAnkle",
    "Neck",
    "LeftShoulder",
    "RightShoulder",
    "Head",
    "LeftElbow",
    "RightElbow",
    "LeftWrist",
    "RightWrist",
)

_XSENS_REF_PARENT: dict[str, str | None] = {
    "Hips": None,
    "LeftHip": "Hips",
    "RightHip": "Hips",
    "Chest": "Hips",
    "LeftKnee": "LeftHip",
    "RightKnee": "RightHip",
    "Chest4": "Chest",
    "LeftAnkle": "LeftKnee",
    "RightAnkle": "RightKnee",
    "Neck": "Chest4",
    "LeftShoulder": "Chest4",
    "RightShoulder": "Chest4",
    "Head": "Neck",
    "LeftElbow": "LeftShoulder",
    "RightElbow": "RightShoulder",
    "LeftWrist": "LeftElbow",
    "RightWrist": "RightElbow",
}


def _identity_quats(n: int) -> NDArray[np.float32]:
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 3] = 1.0
    return q


def _smplx_fallback_pose() -> HumanReferencePose:
    """SMPL-X topology with canonical proxy geometry when MPI weights are absent."""
    names = SMPLX_JOINT_NAMES
    parents_i = SMPLX_PARENTS
    subset = frozenset(names)
    parent_names = tuple(
        _parent_in_subset(names, parents_i, n, subset) for n in names
    )
    pos = np.zeros((len(names), 3), dtype=np.float32)
    smpl_pos = _canonical_smpl17_positions()
    for i, n in enumerate(_SMPL_REF_JOINTS):
        if n in names:
            pos[names.index(n)] = smpl_pos[i]
    # Jaw / eyes near head
    head_i = names.index("head")
    for extra in ("jaw", "left_eye_smplhf", "right_eye_smplhf"):
        if extra in names:
            pos[names.index(extra)] = pos[head_i] + np.array([0.0, 0.0, 0.03], dtype=np.float32)
    lw_i = names.index("left_wrist")
    rw_i = names.index("right_wrist")
    for i, n in enumerate(names):
        if i <= 21:
            continue
        if n in ("jaw", "left_eye_smplhf", "right_eye_smplhf"):
            continue
        if n.startswith("left_"):
            pos[i] = pos[lw_i]
        elif n.startswith("right_"):
            pos[i] = pos[rw_i]
    pos -= pos[0:1]
    q = _identity_quats(len(names))
    src2can = {k: v for k, v in SMPL_BODY_TO_CANONICAL.items() if k in names}
    pi = np.asarray(parents_i, dtype=np.int64)
    h = _measure_height(pos, pi)
    return HumanReferencePose(
        name="smplx",
        root_joint="pelvis",
        joint_names=names,
        parent_names=parent_names,
        positions=pos,
        quaternions=q,
        source_to_canonical=src2can,
        height_m=float(h),
        fallback=True,
    )


def _build_static_soma() -> HumanReferencePose:
    """SOMA calibration reference — extracted from bundled ``soma_zero_frame0.bvh``.

    The old hard-coded proxy (:func:`_canonical_soma17_positions`) mirrored
    left/right incorrectly and disagreed with the scaler's rest pose by up
    to ~0.6 m on the hands, which locked retargeted arms at the calibrated
    hang-down pose regardless of clip motion.
    """
    from_bvh = _soma17_from_zero_bvh()
    if from_bvh is not None:
        return from_bvh

    names = _SOMA_REF_JOINTS
    parents = tuple(_SOMA_REF_PARENT[n] for n in names)
    pos = _canonical_soma17_positions()
    q = _identity_quats(len(names))
    src2can = {k: v for k, v in SOMA_BVH_TO_CANONICAL.items() if k in names}
    pi = np.array(
        [-1 if p is None else names.index(p) for p in parents],
        dtype=np.int64,
    )
    h = _measure_height(pos, pi)
    return HumanReferencePose(
        name="soma_bvh",
        root_joint="Hips",
        joint_names=names,
        parent_names=parents,
        positions=pos,
        quaternions=q,
        source_to_canonical=src2can,
        height_m=float(h),
        fallback=True,
    )


@lru_cache(maxsize=1)
def _soma17_from_zero_bvh() -> HumanReferencePose | None:
    """17-joint subset of ``assets/reference_poses/soma_zero_frame0.bvh`` frame 0."""

    try:
        from hhtools.io.bvh import load_bvh
        from hhtools.retarget.newton_basic.rest_pose import bundled_reference_bvh_path
    except ImportError:
        return None

    path = bundled_reference_bvh_path("soma_bvh")
    if path is None:
        return None

    motion = load_bvh(path)
    if motion.num_frames < 1:
        return None

    bone_names = tuple(motion.hierarchy.bone_names)
    name2i = {n: i for i, n in enumerate(bone_names)}
    if "Hips" not in name2i:
        return None

    names = _SOMA_REF_JOINTS
    missing = [n for n in names if n not in name2i]
    if missing:
        return None

    hips_i = name2i["Hips"]
    frame_pos = np.asarray(motion.positions[0], dtype=np.float32)
    frame_quat = Q.normalize(np.asarray(motion.quaternions[0], dtype=np.float32))
    anchor = frame_pos[hips_i].copy()

    pos = np.stack([frame_pos[name2i[n]] - anchor for n in names], axis=0)
    quat = np.stack([frame_quat[name2i[n]] for n in names], axis=0)
    parents = tuple(_SOMA_REF_PARENT[n] for n in names)
    src2can = {k: v for k, v in SOMA_BVH_TO_CANONICAL.items() if k in names}
    pi = np.array(
        [-1 if p is None else names.index(p) for p in parents],
        dtype=np.int64,
    )
    h = _measure_height(pos, pi)
    return HumanReferencePose(
        name="soma_bvh",
        root_joint="Hips",
        joint_names=names,
        parent_names=parents,
        positions=pos.astype(np.float32),
        quaternions=quat.astype(np.float32),
        source_to_canonical=src2can,
        height_m=float(h),
        fallback=False,
    )


def _build_static_lafan() -> HumanReferencePose:
    names = _LAFAN_REF_JOINTS
    parents = tuple(_LAFAN_REF_PARENT[n] for n in names)
    pos = _canonical_lafan17_positions()
    q = _identity_quats(len(names))
    src2can = {k: v for k, v in MIXAMO_CMU_TO_CANONICAL.items() if k in names}
    pi = np.array(
        [-1 if p is None else names.index(p) for p in parents],
        dtype=np.int64,
    )
    h = _measure_height(pos, pi)
    return HumanReferencePose(
        name="lafan_bvh",
        root_joint="Hips",
        joint_names=names,
        parent_names=parents,
        positions=pos,
        quaternions=q,
        source_to_canonical=src2can,
        height_m=float(h),
        fallback=False,
    )


def _build_static_xsens_mocap() -> HumanReferencePose:
    """Xsens calibration reference — extracted from bundled stand-pose BVH."""
    from_bvh = _xsens17_from_zero_bvh()
    if from_bvh is not None:
        return from_bvh

    names = _XSENS_REF_JOINTS
    parents = tuple(_XSENS_REF_PARENT[n] for n in names)
    pos = _canonical_lafan17_positions()
    q = _identity_quats(len(names))
    src2can = {k: v for k, v in XSENS_MOCAP_TO_CANONICAL.items() if k in names}
    pi = np.array(
        [-1 if p is None else names.index(p) for p in parents],
        dtype=np.int64,
    )
    h = _measure_height(pos, pi)
    return HumanReferencePose(
        name="xsens_mocap",
        root_joint="Hips",
        joint_names=names,
        parent_names=parents,
        positions=pos,
        quaternions=q,
        source_to_canonical=src2can,
        height_m=float(h),
        fallback=True,
    )


@lru_cache(maxsize=1)
def _xsens17_from_zero_bvh() -> HumanReferencePose | None:
    """17-joint subset of ``assets/reference_poses/xsens_mocap_zero_frame0.bvh``."""

    try:
        from hhtools.io.bvh import load_bvh
        from hhtools.retarget.newton_basic.rest_pose import bundled_reference_bvh_path
    except ImportError:
        return None

    path = bundled_reference_bvh_path("xsens_mocap")
    if path is None:
        return None

    motion = load_bvh(path)
    if motion.num_frames < 1:
        return None

    bone_names = tuple(motion.hierarchy.bone_names)
    name2i = {n: i for i, n in enumerate(bone_names)}
    if "Hips" not in name2i:
        return None

    names = _XSENS_REF_JOINTS
    missing = [n for n in names if n not in name2i]
    if missing:
        return None

    hips_i = name2i["Hips"]
    frame_pos = np.asarray(motion.positions[0], dtype=np.float32)
    frame_quat = Q.normalize(np.asarray(motion.quaternions[0], dtype=np.float32))
    anchor = frame_pos[hips_i].copy()

    pos = np.stack([frame_pos[name2i[n]] - anchor for n in names], axis=0)
    quat = np.stack([frame_quat[name2i[n]] for n in names], axis=0)
    parents = tuple(_XSENS_REF_PARENT[n] for n in names)
    src2can = {k: v for k, v in XSENS_MOCAP_TO_CANONICAL.items() if k in names}
    pi = np.array(
        [-1 if p is None else names.index(p) for p in parents],
        dtype=np.int64,
    )
    h = _measure_height(pos, pi)
    return HumanReferencePose(
        name="xsens_mocap",
        root_joint="Hips",
        joint_names=names,
        parent_names=parents,
        positions=pos.astype(np.float32),
        quaternions=quat.astype(np.float32),
        source_to_canonical=src2can,
        height_m=float(h),
        fallback=False,
    )


def _load_smpl(*, allow_engine: bool) -> HumanReferencePose:
    if allow_engine:
        got = _try_smpl_forward("smpl")
        if got is not None:
            joints, quats, _, _ = got
            names, parents, pos, q, src2can = _smpl17_from_layout(joints, quats)
            pi = np.array(
                [-1 if p is None else names.index(cast(str, p)) for p in parents],
                dtype=np.int64,
            )
            h = _measure_height(pos, pi)
            return HumanReferencePose(
                name="smpl",
                root_joint="pelvis",
                joint_names=names,
                parent_names=parents,
                positions=pos,
                quaternions=q,
                source_to_canonical=src2can,
                height_m=float(h),
                fallback=False,
            )
    names = _SMPL_REF_JOINTS
    parents = tuple(_SMPL_REF_PARENT[n] for n in names)
    pos = _canonical_smpl17_positions()
    q = _identity_quats(len(names))
    src2can = {n: SMPL_BODY_TO_CANONICAL[n] for n in names}
    pi = np.array(
        [-1 if p is None else names.index(cast(str, p)) for p in parents],
        dtype=np.int64,
    )
    h = _measure_height(pos, pi)
    return HumanReferencePose(
        name="smpl",
        root_joint="pelvis",
        joint_names=names,
        parent_names=parents,
        positions=pos,
        quaternions=q,
        source_to_canonical=src2can,
        height_m=float(h),
        fallback=True,
    )


def _load_smplx() -> HumanReferencePose:
    got = _try_smpl_forward("smplx")
    if got is not None:
        joints, quats, names, parents_i = got
        subset = frozenset(names)
        parent_names = tuple(
            _parent_in_subset(names, parents_i, n, subset) for n in names
        )
        pos = joints.copy()
        pos -= pos[0:1]
        q = quats.copy()
        src2can = {k: v for k, v in SMPL_BODY_TO_CANONICAL.items() if k in names}
        pi = np.asarray(parents_i, dtype=np.int64)
        h = _measure_height(pos, pi)
        return HumanReferencePose(
            name="smplx",
            root_joint="pelvis",
            joint_names=names,
            parent_names=parent_names,
            positions=pos.astype(np.float32),
            quaternions=q.astype(np.float32),
            source_to_canonical=src2can,
            height_m=float(h),
            fallback=False,
        )
    return _smplx_fallback_pose()


def _load_gvhmr() -> HumanReferencePose:
    return replace(_load_smpl(allow_engine=True), name="gvhmr")


def list_reference_names() -> tuple[str, ...]:
    return (
        "smplx", "smpl", "gvhmr", "soma_bvh", "lafan_bvh", "xsens_mocap", "glb",
    )


def load_reference_pose(name: str) -> HumanReferencePose:
    """Load a built-in static reference.

    ``glb`` still requires :func:`build_motion_reference` — it uses the loaded
    clip's frame 0.
    """
    key = _LEGACY_REFERENCE_ALIASES.get(name, name)
    if key == "glb":
        raise ValueError(
            "reference 'glb' requires a loaded motion — use "
            "build_motion_reference(motion, 'glb') instead."
        )
    if key == "smpl":
        return _load_smpl(allow_engine=True)
    if key == "smplx":
        return _load_smplx()
    if key == "gvhmr":
        return _load_gvhmr()
    if key == "soma_bvh":
        return _build_static_soma()
    if key == "lafan_bvh":
        return _build_static_lafan()
    if key == "xsens_mocap":
        return _build_static_xsens_mocap()
    raise ValueError(f"Unknown reference name: {name!r}")


# Names that indicate a virtual wrapper node above the real pelvis/hips.
# Mirrors :data:`hhtools.viewer.anatomy._VIRTUAL_ROOT_NAMES` but kept local
# so ``reference.py`` stays independent of viewer internals.
_VIRTUAL_ROOT_NAMES: frozenset[str] = frozenset({
    "root", "reference", "world", "armature", "origin", "root_body",
    "scene_root", "body_world", "worldroot", "world_root", "rig",
    "skeleton", "skeleton_root", "body", "character", "main", "rootnode",
})


def _is_virtual_root_name(name: str) -> bool:
    token = name.strip().lower()
    for sep in (":", "|", "/"):
        if sep in token:
            token = token.split(sep)[-1]
    return token in _VIRTUAL_ROOT_NAMES


def _skip_virtual_roots(
    root_i: int,
    bone_names: tuple[str, ...],
    parent_indices: NDArray,
) -> int:
    """Walk down from *root_i* past known virtual wrapper nodes.

    Returns the index of the first descendant whose name is **not** in
    :data:`_VIRTUAL_ROOT_NAMES`.  If no such descendant exists (degenerate
    skeleton), returns *root_i* unchanged.
    """
    cur = root_i
    while _is_virtual_root_name(bone_names[cur]):
        children = [
            j for j in range(len(bone_names))
            if int(parent_indices[j]) == cur
        ]
        if not children:
            break
        cur = children[0]
    return cur


def build_motion_reference(motion: "Motion", name: Literal["glb"] = "glb") -> HumanReferencePose:
    """Frame-0 skeleton from ``motion``, root-anchored at the origin.

    When the hierarchy root is a known virtual wrapper node (``body_world``,
    ``Armature``, etc.) the ``root_joint`` is set to the first non-virtual
    descendant — typically the anatomical pelvis (``Hips``).  Using the
    wrapper as the reference root produces catastrophically degenerate
    calibration scales (all 1.0) because downstream code treats it as the
    anatomical anchor.
    """
    if name != "glb":
        raise ValueError("build_motion_reference only supports name='glb'")
    if motion.num_frames < 1:
        raise ValueError("motion has no frames")
    hier = motion.hierarchy
    bone_names = tuple(hier.bone_names)
    parents = tuple(hier.parent_names)
    pi = np.asarray(hier.parent_indices, dtype=np.int64)
    root_i = int(np.where(pi < 0)[0][0]) if (pi < 0).any() else 0

    effective_root_i = _skip_virtual_roots(root_i, bone_names, pi)
    effective_root_name = bone_names[effective_root_i]

    pos = np.asarray(motion.positions[0], dtype=np.float32).copy()
    q = Q.normalize(np.asarray(motion.quaternions[0], dtype=np.float32).copy())
    anchor = pos[effective_root_i].copy()
    pos -= anchor
    auto = auto_source_to_canonical(bone_names)
    h = _measure_height(pos, pi)
    return HumanReferencePose(
        name=name,
        root_joint=effective_root_name,
        joint_names=bone_names,
        parent_names=parents,
        positions=pos,
        quaternions=q,
        source_to_canonical=auto,
        height_m=float(h),
        fallback=False,
    )


def reference_pose_from_motion_frame0_quantized(
    motion: "Motion",
    *,
    display_name: str = "glb",
) -> HumanReferencePose:
    """Frame-0 pose with per-bone local rotations snapped to π/2, then FK (Z-up)."""
    if motion.num_frames < 1:
        raise ValueError("motion has no frames")
    hier = motion.hierarchy
    n = hier.num_bones
    parent_idx = np.asarray(hier.parent_indices, dtype=np.int64)
    root_idx_arr = np.where(parent_idx < 0)[0]
    root_idx = int(root_idx_arr[0]) if root_idx_arr.size > 0 else 0
    world_pos = np.asarray(motion.positions[0], dtype=np.float32)
    world_quat = Q.normalize(np.asarray(motion.quaternions[0], dtype=np.float32).copy())

    local_pos = np.zeros((n, 3), dtype=np.float32)
    local_quat = np.zeros((n, 4), dtype=np.float32)
    local_quat[:, 3] = 1.0
    for j in range(n):
        parent = int(parent_idx[j])
        if parent < 0:
            local_pos[j] = world_pos[j]
            local_quat[j] = world_quat[j]
            continue
        diff = (world_pos[j] - world_pos[parent])[None, :].astype(np.float32)
        q_parent = world_quat[parent][None, :]
        local_pos[j] = Q.rotate(Q.conjugate(q_parent), diff)[0]
        local_quat[j] = Q.multiply(Q.conjugate(q_parent), world_quat[j][None, :])[0]
    local_quat = Q.normalize(local_quat)

    q_snap = local_quat.copy()
    for j in range(n):
        if j == root_idx:
            continue
        q_snap[j] = _quantize_local_rotation(local_quat[j])

    rest_pos = np.zeros((n, 3), dtype=np.float32)
    rest_quat = np.zeros((n, 4), dtype=np.float32)
    rest_quat[:, 3] = 1.0
    for j in range(n):
        parent = int(parent_idx[j])
        if parent < 0:
            rest_quat[j] = q_snap[j]
            rest_pos[j] = local_pos[j]
            continue
        rest_quat[j] = Q.multiply(rest_quat[parent][None, :], q_snap[j][None, :])[0]
        rest_pos[j] = rest_pos[parent] + Q.rotate(
            rest_quat[parent][None, :], local_pos[j][None, :]
        )[0]
    rest_quat = Q.normalize(rest_quat)

    bone_names = tuple(hier.bone_names)
    parents = tuple(hier.parent_names)
    root_name = bone_names[root_idx]
    anchor = rest_pos[root_idx].copy()
    rest_pos = rest_pos - anchor

    auto = auto_source_to_canonical(bone_names)
    h = _measure_height(rest_pos, parent_idx)
    return HumanReferencePose(
        name=display_name,
        root_joint=root_name,
        joint_names=bone_names,
        parent_names=parents,
        positions=rest_pos,
        quaternions=rest_quat,
        source_to_canonical=auto,
        height_m=float(h),
        fallback=False,
    )


__all__ = [
    "HumanReferencePose",
    "ReferenceName",
    "build_motion_reference",
    "list_reference_names",
    "load_reference_pose",
    "reference_pose_from_motion_frame0_quantized",
]
