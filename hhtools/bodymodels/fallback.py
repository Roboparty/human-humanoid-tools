"""Weight-free kinematic fallback for SMPL-family motion parameters.

The licensed SMPL assets are needed for a skinned surface, but they are not
needed to expose a useful skeleton to the viewer or to the retargeter.  This
module keeps that small fallback independent from calibration and web code:
it uses the canonical joint topology plus a neutral proxy rest pose, then
applies the source local axis-angle rotations with ordinary forward
kinematics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hhtools.bodymodels.layout import BodyModelLayout, layout_for
from hhtools.bodymodels.params import SmplMotionParams
from hhtools.core.hierarchy import Hierarchy
from hhtools.core.math import quaternion as Q
from hhtools.core.motion import Motion

ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class FallbackForwardResult:
    """Forward-kinematics output with the same fields used by ``SmplxEngine``."""

    joints: NDArray[np.float32]
    quaternions_global: NDArray[np.float32]
    layout: BodyModelLayout

    @property
    def vertices(self) -> None:
        return None

    @property
    def faces(self) -> None:
        return None


# Neutral proxy in SMPL's native Y-up model coordinates. ``root_orient`` maps
# this model frame into each dataset's world frame; pre-rotating the rest pose
# to ``params.up_axis`` would apply that conversion twice. Omitted hand/face
# joints are attached to their parent below.
_SMPL_PROXY_POSITIONS_Y = np.asarray(
    [
        [0.0, 0.0, 0.00],
        [0.095, -0.02, 0.0],
        [-0.095, -0.02, 0.0],
        [0.0, 0.11, 0.0],
        [0.095, -0.46, 0.0],
        [-0.095, -0.46, 0.0],
        [0.0, 0.165, 0.0],
        [0.095, -0.86, 0.0],
        [-0.095, -0.86, 0.0],
        [0.0, 0.22, 0.0],
        [0.095, -0.86, 0.0],
        [-0.095, -0.86, 0.0],
        [0.0, 0.38, 0.0],
        [0.0, 0.38, 0.0],
        [0.0, 0.38, 0.0],
        [0.0, 0.62, 0.0],
        [0.17, 0.38, 0.0],
        [-0.17, 0.38, 0.0],
        [0.38, 0.38, 0.0],
        [-0.38, 0.38, 0.0],
        [0.58, 0.38, 0.0],
        [-0.58, 0.38, 0.0],
        [0.58, 0.38, 0.0],
        [-0.58, 0.38, 0.0],
    ],
    dtype=np.float32,
)


def _proxy_positions(layout: BodyModelLayout) -> NDArray[np.float32]:
    """Return a proxy rest pose in SMPL's native model coordinate frame."""
    positions = np.zeros((layout.num_joints, 3), dtype=np.float32)
    if layout.family == "smpl":
        positions[:] = _SMPL_PROXY_POSITIONS_Y
    else:
        positions[:22] = _SMPL_PROXY_POSITIONS_Y[:22]
        for index, name in enumerate(layout.joint_names[22:], start=22):
            if layout.family == "smplx" and name in {
                "jaw",
                "left_eye_smplhf",
                "right_eye_smplhf",
            }:
                positions[index] = positions[15] + np.array(
                    [0.0, 0.03, 0.0], dtype=np.float32
                )
            elif name.startswith("left_"):
                positions[index] = positions[20]
            elif name.startswith("right_"):
                positions[index] = positions[21]

    return positions


def _axis_angle_array(
    value: NDArray | None,
    frames: int,
    width: int,
) -> NDArray[np.float32]:
    """Normalise an optional per-frame axis-angle block and zero-pad it."""
    out = np.zeros((frames, width), dtype=np.float32)
    if value is None or width == 0:
        return out
    raw = np.asarray(value, dtype=np.float32)
    if raw.ndim == 1:
        if frames == 1:
            raw = raw[None, :]
        else:
            raw = np.broadcast_to(raw, (frames, raw.shape[0]))
    if raw.ndim != 2 or raw.shape[0] != frames:
        raise ValueError(
            f"axis-angle block must have {frames} frames; got shape {raw.shape}"
        )
    copy_width = min(width, raw.shape[1])
    out[:, :copy_width] = raw[:, :copy_width]
    return out


def _assign_axis_angles(
    local: NDArray[np.float32],
    start: int,
    value: NDArray | None,
    count: int,
) -> None:
    """Insert up to ``count`` 3-vector rotations into a local quaternion block."""
    if start >= local.shape[1] or count <= 0:
        return
    frames = local.shape[0]
    raw = _axis_angle_array(value, frames, count * 3).reshape(frames, count, 3)
    end = min(start + count, local.shape[1])
    local[:, start:end] = Q.from_axis_angle(raw[:, : end - start])


def forward_without_weights(
    params: SmplMotionParams,
    *,
    progress_callback: ProgressCallback | None = None,
) -> FallbackForwardResult:
    """Compute approximate global joints/quaternions using NumPy-only FK."""
    layout = layout_for(params.surface_model)
    frames = params.num_frames
    joints = layout.num_joints
    rest = _proxy_positions(layout)
    offsets = np.zeros_like(rest)
    parents = np.asarray(layout.parents, dtype=np.int64)
    for index, parent in enumerate(parents):
        if parent >= 0:
            offsets[index] = rest[index] - rest[parent]

    local = np.broadcast_to(Q.identity(joints), (frames, joints, 4)).copy()
    local[:, 0] = Q.from_axis_angle(
        _axis_angle_array(params.root_orient, frames, 3).reshape(frames, 3)
    )
    body_count = 23 if layout.family == "smpl" else 21
    _assign_axis_angles(local, 1, params.body_pose, body_count)

    if layout.family == "smplh":
        _assign_axis_angles(local, 22, params.hand_pose_left, 15)
        _assign_axis_angles(local, 37, params.hand_pose_right, 15)
    elif layout.family == "smplx":
        _assign_axis_angles(local, 22, params.jaw_pose, 1)
        _assign_axis_angles(local, 23, params.leye_pose, 1)
        _assign_axis_angles(local, 24, params.reye_pose, 1)
        _assign_axis_angles(local, 25, params.hand_pose_left, 15)
        _assign_axis_angles(local, 40, params.hand_pose_right, 15)

    global_quats = np.empty_like(local)
    global_positions = np.empty((frames, joints, 3), dtype=np.float32)
    for index, parent in enumerate(parents):
        if parent < 0:
            global_quats[:, index] = local[:, index]
            global_positions[:, index] = np.asarray(params.trans, dtype=np.float32)
            continue
        global_quats[:, index] = Q.multiply(
            global_quats[:, parent], local[:, index]
        )
        global_positions[:, index] = global_positions[:, parent] + Q.rotate(
            global_quats[:, parent], offsets[index]
        )

    global_quats = Q.normalize(global_quats).astype(np.float32)
    if progress_callback is not None:
        progress_callback(1.0, "骨架代理完成")
    return FallbackForwardResult(
        joints=global_positions,
        quaternions_global=global_quats,
        layout=layout,
    )


def motion_from_fallback(
    params: SmplMotionParams,
    *,
    name: str,
    source_format: str | None = None,
    reason: BaseException | str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Motion:
    """Wrap weight-free FK output as a regular :class:`Motion`."""
    if progress_callback is not None:
        progress_callback(0.0, "SMPL 权重不可用，构建骨架代理…")
    result = forward_without_weights(params, progress_callback=progress_callback)
    meta = dict(params.meta or {})
    meta.update(
        {
            "surface_model": params.surface_model,
            "gender": params.gender,
            "joint_layout": params.surface_model,
            "body_model_fallback": True,
            "body_model_fallback_reason": str(
                reason or "SMPL weights unavailable; kinematic proxy skeleton"
            ),
            "baked_mesh_unavailable": True,
        }
    )
    return Motion(
        name=name,
        hierarchy=Hierarchy.from_parent_indices(
            list(result.layout.joint_names), result.layout.parents
        ),
        positions=result.joints,
        quaternions=result.quaternions_global,
        framerate=params.framerate,
        up_axis=params.up_axis,
        source_format=source_format or params.surface_model,
        meta=meta,
    )


__all__ = [
    "FallbackForwardResult",
    "forward_without_weights",
    "motion_from_fallback",
]
