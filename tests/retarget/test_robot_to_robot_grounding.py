# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""R2R source clips must ground on the source robot's soles, not its ankles."""

from __future__ import annotations

import numpy as np
import pytest

from hhtools.core.grounding import (
    SOURCE_FLOOR_META_KEY,
    foot_floor_z_in_positions,
    retarget_source_floor_z_world,
)
from hhtools.retarget.robot_to_robot import source_trajectory_to_motion


@pytest.fixture(scope="module")
def standing_g1(grounding_robot_pair):
    """Self-contained robot whose soles rest exactly on ``z=0``."""
    model = grounding_robot_pair[0]

    from hhtools.robot.standing_height import _trimesh_scene_z_bounds

    model.apply_configuration(model.zero_configuration())
    bounds = _trimesh_scene_z_bounds(model.trimesh_scene(collision=False))
    assert bounds is not None

    dof_names = tuple(model.dof_names())
    joint_q = np.zeros((4, 7 + len(dof_names)), dtype=np.float32)
    joint_q[:, 0] = np.linspace(0.0, 0.3, 4)  # translate so it is not a single pose
    joint_q[:, 2] = -bounds[0]
    joint_q[:, 6] = 1.0
    return model, joint_q, dof_names


def test_source_motion_declares_sole_contact_plane(standing_g1):
    model, joint_q, dof_names = standing_g1
    motion = source_trajectory_to_motion(model, joint_q, dof_names, framerate=50.0)

    declared = motion.meta.get(SOURCE_FLOOR_META_KEY)
    assert declared is not None, "R2R source motion must declare its contact plane"
    assert abs(float(declared)) < 1e-3, (
        f"soles were placed on z=0, so the declared plane should be ~0, got {declared}"
    )

    # The canonical skeleton ends at the ankles, which sit a foot thickness above
    # that plane.  Grounding on them is precisely the bug this guards against.
    ankle_floor = float(
        foot_floor_z_in_positions(motion.positions, tuple(motion.hierarchy.bone_names))
    )
    assert ankle_floor - float(declared) > 0.02, (
        "expected the ankle joints to sit clearly above the sole plane; "
        f"ankle floor={ankle_floor}, declared plane={declared}"
    )

    assert retarget_source_floor_z_world(motion) == pytest.approx(
        float(declared), abs=1e-6,
    ), "the declared plane must win over the ankle-minimum heuristic"


def test_declared_plane_survives_a_raised_source_clip(standing_g1):
    """A source hovering 25 cm up keeps that gap instead of being snapped down."""
    model, joint_q, dof_names = standing_g1
    raised = joint_q.copy()
    raised[:, 2] += 0.25

    motion = source_trajectory_to_motion(model, raised, dof_names, framerate=50.0)
    declared = motion.meta.get(SOURCE_FLOOR_META_KEY)
    assert declared is not None
    assert float(declared) == pytest.approx(0.25, abs=1e-3)


def test_r2r_defaults_disable_anti_float_root_pump() -> None:
    from hhtools.robot.retarget_profile import _reference_defaults

    r2r = _reference_defaults("robot_g1")
    human = _reference_defaults("lafan_bvh")
    assert r2r.get("foot_clamp_anti_float") is False
    assert human.get("foot_clamp_anti_float") is not False
