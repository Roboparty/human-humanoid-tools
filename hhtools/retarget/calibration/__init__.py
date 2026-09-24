"""Robot↔human retargeting calibration.

Establishes a one-time, per-robot alignment between:

* a known, reference-frame human pose (e.g. ``smpl`` T-pose,
  ``soma_bvh`` arms-down rest, or ``lafan_bvh`` near-T-pose), and
* the same robot posed via manually dialled or deterministically proposed
  actuated-joint angles with the floating base at identity.

When the user or a validated GPT-vision workflow confirms the poses, a yaml file is written
as ``retarget_calibration_<reference>.yaml`` (one calibration per robot **and**
per reference format). Writable source trees retain a sibling file next to the
URDF; packaged read-only presets use a per-user override and fall back to their
bundled calibration. At retarget time, per-canonical-joint scales + orientation
offsets are re-derived in closed form from the stored joint-angle configuration
so that the source motion's frame 0 lines up exactly with the robot's calibrated
pose. Subsequent frames flow through the scaler as "relative rotation from
motion-frame-0 composed with the calibrated orientation offset".

Compared to the ad-hoc first-frame heuristic this replaces, calibration
is explicit, inspectable (``git diff`` the YAML), and — once done per
robot — amortises across every source motion retargeted to that robot.
"""

from __future__ import annotations

from hhtools.retarget.calibration.assistant import (
    CalibrationAssessment,
    assess_calibration_pose,
    propose_calibration_pose,
    render_calibration_preview_png,
)
from hhtools.retarget.calibration.calibration import (
    RobotRetargetCalibration,
    build_scaler_config_from_calibration,
    build_scaler_config_soma_style,
    calibration_path_for,
    derive_calibration_params,
    effective_retarget_human_height,
    load_calibration,
    normalize_calibration_reference,
    repair_apose_calibration_for_straight_t_reference,
    resolve_calibration_file,
    resolve_preset_calibration_file,
    save_calibration,
    save_calibration_for_preset,
)
from hhtools.retarget.calibration.reference import (
    HumanReferencePose,
    ReferenceName,
    build_motion_reference,
    list_reference_names,
    load_reference_pose,
    reference_pose_from_motion_frame0_quantized,
)

__all__ = [
    "HumanReferencePose",
    "CalibrationAssessment",
    "ReferenceName",
    "RobotRetargetCalibration",
    "build_motion_reference",
    "assess_calibration_pose",
    "build_scaler_config_from_calibration",
    "build_scaler_config_soma_style",
    "calibration_path_for",
    "derive_calibration_params",
    "effective_retarget_human_height",
    "list_reference_names",
    "load_calibration",
    "load_reference_pose",
    "reference_pose_from_motion_frame0_quantized",
    "propose_calibration_pose",
    "normalize_calibration_reference",
    "repair_apose_calibration_for_straight_t_reference",
    "resolve_calibration_file",
    "resolve_preset_calibration_file",
    "save_calibration",
    "save_calibration_for_preset",
    "render_calibration_preview_png",
]
