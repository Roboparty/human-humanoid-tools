from __future__ import annotations

import numpy as np

from hhtools.application.retarget import _retarget_single
from hhtools.retarget.calibration import RobotRetargetCalibration, save_calibration


def test_application_interaction_mesh_runs_one_self_contained_terrain_frame(
    grounding_robot_pair,
    synthetic_terrain_motion,
) -> None:
    model = grounding_robot_pair[1]
    calibration_path = model.preset.root_dir / "retarget_calibration_smpl.yaml"
    save_calibration(
        RobotRetargetCalibration(
            robot=model.preset.name,
            reference="smpl",
            calibrated_joint_q={name: 0.0 for name in model.dof_names()},
            notes="self-contained Interaction-Mesh smoke fixture",
        ),
        calibration_path,
    )

    result = _retarget_single(
        model,
        model.preset.name,
        synthetic_terrain_motion,
        "smpl",
        "interaction_mesh",
        24,
        1.7,
        1,
        None,
        preset=model.preset,
        foot_clamp_anti_penetration=False,
    )

    assert result.num_frames == 1
    assert np.isfinite(result.joint_q).all()
    assert result.meta["retarget_backend"] == "interaction_mesh"
    assert result.meta["execution_provenance"]["runtime"] == "mujoco"
    assert result.meta["execution_provenance"]["device_kind"] == "cpu"
