"""Regression coverage for IsaacLab training-convention NPZ uploads."""

from __future__ import annotations

import numpy as np
import pytest

from hhtools.retarget.robot_to_robot import load_source_trajectory
from hhtools.web.r2r_upload_resolve import _is_robot_export_trajectory


def _write_isaaclab_training_npz(path, *, base_quat_w=None):
    np.savez(
        path,
        framerate=np.array(50.0, dtype=np.float64),
        joint_names=np.array(["left_hip_joint", "right_hip_joint"]),
        joint_pos=np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32),
        base_pos_w=np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=np.float32),
        # IsaacLab's training convention stores world quaternions as wxyz.
        base_quat_w=np.asarray(
            base_quat_w
            if base_quat_w is not None
            else [[0.9, 0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )


def test_r2r_accepts_and_loads_isaaclab_training_convention_npz(tmp_path):
    path = tmp_path / "parkour_motion_retargetted.npz"
    _write_isaaclab_training_npz(path)

    assert _is_robot_export_trajectory(path)

    trajectory = load_source_trajectory(path)

    np.testing.assert_allclose(
        trajectory.joint_q,
        np.array(
            [
                [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9, 0.1, -0.2],
                [1.5, 2.5, 3.5, 0.0, 0.0, 0.0, 1.0, 0.3, -0.4],
            ],
            dtype=np.float32,
        ),
    )
    assert trajectory.dof_names == ("left_hip_joint", "right_hip_joint")
    assert trajectory.framerate == 50.0
    assert trajectory.meta["source_format"] == "isaaclab_training_convention_npz"


def test_r2r_rejects_malformed_isaaclab_training_convention_npz(tmp_path):
    path = tmp_path / "bad_reference.npz"
    _write_isaaclab_training_npz(path, base_quat_w=[[1.0, 0.0, 0.0, 0.0]])

    assert not _is_robot_export_trajectory(path)
    with pytest.raises(ValueError, match="invalid IsaacLab training-convention shapes"):
        load_source_trajectory(path)
