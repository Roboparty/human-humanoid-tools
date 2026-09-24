# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""R2R yellow overlay keeps scaled source feet; the robot ankles must match them."""

from __future__ import annotations

import numpy as np
import pytest

from hhtools.core.grounding import SOURCE_FLOOR_META_KEY
from hhtools.core.hierarchy import Hierarchy
from hhtools.core.motion import Motion
from hhtools.retarget.retarget_result import RetargetedMotion
from hhtools.robot.foot_geometry import lowest_ankle_z, quat_xyzw_to_rotmat
from hhtools.web.output.serialize import (
    _scaled_overlay_foot_z,
    serialize_robot_trajectory,
)


@pytest.fixture(scope="module")
def g1_rp1(grounding_robot_pair):
    return grounding_robot_pair


def _r2r_motion(*, ankle_z: float = 0.06, sole_z: float = 0.0) -> Motion:
    names = [
        "hips", "spine", "chest", "neck", "head",
        "left_shoulder", "left_elbow", "left_wrist",
        "right_shoulder", "right_elbow", "right_wrist",
        "left_hip", "left_knee", "left_ankle",
        "right_hip", "right_knee", "right_ankle",
    ]
    parents = [-1, 0, 1, 2, 3, 2, 5, 6, 2, 8, 9, 0, 11, 12, 0, 14, 15]
    parent_names = [None] + [names[p] for p in parents[1:]]
    hier = Hierarchy(
        bone_names=names, parent_indices=parents, parent_names=parent_names,
    )
    pos = np.zeros((4, len(names), 3), dtype=np.float32)
    pos[:, names.index("hips"), 2] = 0.9
    pos[:, names.index("left_ankle"), 2] = ankle_z
    pos[:, names.index("right_ankle"), 2] = ankle_z
    pos[:, names.index("left_wrist"), 2] = 0.8
    pos[-1, names.index("left_wrist"), 2] = -0.08
    quat = np.zeros((4, len(names), 4), dtype=np.float32)
    quat[..., 3] = 1.0
    return Motion(
        name="r2r_like",
        hierarchy=hier,
        positions=pos,
        quaternions=quat,
        framerate=50.0,
        meta={
            "robot_to_robot_source": "g1",
            SOURCE_FLOOR_META_KEY: sole_z,
        },
    )


def _standing_retarget(model, *, root_z: float = 0.75, frames: int = 4) -> RetargetedMotion:
    dof_names = list(model.dof_names())
    root = np.zeros((frames, 7), dtype=np.float32)
    root[:, 2] = root_z
    root[:, 6] = 1.0
    dof = np.zeros((frames, len(dof_names)), dtype=np.float32)
    return RetargetedMotion(
        name="fake",
        joint_q=np.concatenate([root, dof], axis=1),
        sample_rate=50.0,
        dof_names=tuple(dof_names),
        root_coord_count=7,
    )


def _playback_ankle_world_z(model, traj: dict) -> float:
    frame = traj["frames"][0]
    root = np.asarray(frame["root"], dtype=np.float64)
    lift = float(frame.get("mesh_z_lift") or 0.0)
    ik_map = dict(model.preset.ik_map) if model.preset.ik_map else {}
    model.apply_configuration(model.zero_configuration())
    ankle = lowest_ankle_z(model, ik_map, quat_xyzw_to_rotmat(root[3:7]))
    assert ankle is not None
    return float(root[2] + lift + ankle)


def test_r2r_yellow_keeps_scaled_source_feet(g1_rp1):
    from hhtools.web.server.preview_runtime import _compute_r2r_scaled_preview

    src, tgt = g1_rp1
    motion = _r2r_motion(ankle_z=0.06, sole_z=0.0)
    calib = {
        j.name: 0.0 for j in tgt.actuated_joints if j.joint_type != "fixed"
    }
    scaled = _compute_r2r_scaled_preview(src, tgt, motion, calib)
    yellow_foot = _scaled_overlay_foot_z(scaled, 0)
    assert yellow_foot is not None
    # Declared sole plane is z=0; yellow ankles must keep the sole thickness
    # instead of being snapped onto the ground grid.
    assert yellow_foot > 0.02, f"yellow feet should keep scaled sole thickness, got {yellow_foot}"
    wrist_i = motion.hierarchy.bone_names.index("left_wrist")
    wrist_z = float(np.asarray(scaled["positions"])[-1, wrist_i, 2])
    assert yellow_foot > wrist_z


def test_r2r_playback_aligns_robot_ankles_to_yellow_feet(g1_rp1):
    from hhtools.retarget.robot_to_robot import align_retargeted_ankles_to_scaled_source
    from hhtools.web.server.preview_runtime import _compute_r2r_scaled_preview

    src, tgt = g1_rp1
    motion = _r2r_motion(ankle_z=0.06, sole_z=0.0)
    calib = {
        j.name: 0.0 for j in tgt.actuated_joints if j.joint_type != "fixed"
    }
    scaled = _compute_r2r_scaled_preview(src, tgt, motion, calib)
    yellow_foot = _scaled_overlay_foot_z(scaled, 0)
    assert yellow_foot is not None

    ret = _standing_retarget(tgt)
    sole_traj = serialize_robot_trajectory(
        tgt, ret, scaled_preview=scaled, ground_follow=False, yellow_align="sole",
    )
    ankle_traj = serialize_robot_trajectory(
        tgt, ret, scaled_preview=scaled, ground_follow=False, yellow_align="ankle",
    )
    sole_aligned = _playback_ankle_world_z(tgt, sole_traj)
    ankle_aligned = _playback_ankle_world_z(tgt, ankle_traj)
    # Planting the mesh sole on the yellow ankle floats the robot above it.
    assert sole_aligned > yellow_foot + 0.015
    assert abs(ankle_aligned - yellow_foot) < 0.012

    shifted = align_retargeted_ankles_to_scaled_source(
        tgt, src, motion, ret, calib,
    )
    aligned_traj = serialize_robot_trajectory(
        tgt, shifted, scaled_preview=scaled, ground_follow=False, yellow_align="ankle",
    )
    assert abs(_playback_ankle_world_z(tgt, aligned_traj) - yellow_foot) < 0.012
