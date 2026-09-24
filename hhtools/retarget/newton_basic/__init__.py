"""Newton-based IK retargeting pipeline — Stage 1 (pure NumPy).

This package is an Apache-2.0 re-implementation of the core blocks from
soma-retargeter (https://github.com/NVIDIA/SOMA-Retargeter), rewritten
against the hhtools :class:`~hhtools.core.motion.Motion` /
:class:`~hhtools.core.hierarchy.Hierarchy` / :class:`~hhtools.robot.base
.RobotModel` contracts.  See the project-root ``NOTICE`` for full
attribution.

Stage-1 scope (this file):

* :class:`ScalerConfig` + :class:`HumanToRobotScaler` — scale a source human
  motion into per-effector world-space targets.
* :class:`FeetStabilizerConfig` + :class:`FeetStabilizer` — rate-limited
  ground-contact / foot-planting / lateral-separation constraints applied to
  those targets *before* IK.
* :class:`JointLimitClamper` — numpy DOF-limit clip consumed at every frame.

All three modules are pure python + numpy; no Newton, Warp, or GPU.  They
are intended to be independently unit-testable and reusable from both the
future IK pipeline (stage 2) and downstream analytics (``analytics/`` uses
the same clamper to score DOF limit violations, for example).
"""

from __future__ import annotations

from hhtools.retarget.newton_basic.config import (
    FeetStabilizerConfig,
    ScalerConfig,
    load_feet_stabilizer_config,
    load_scaler_config,
)
from hhtools.retarget.newton_basic.feet_stabilizer import (
    FeetStabilizer,
    StabilizationStats,
)
from hhtools.retarget.newton_basic.joint_limit_clamper import JointLimitClamper
from hhtools.retarget.newton_basic.rest_pose import (
    SourceRestPose,
    rest_pose_from_motion,
    rest_pose_from_motion_bind,
    rest_pose_from_reference,
)
from hhtools.retarget.newton_basic.scaler import (
    HumanToRobotScaler,
    ScaledEffectors,
)


def _stage2_lazy_imports():  # pragma: no cover — re-exports for public API
    # Stage-2 modules depend on warp / newton / yourdfpy.  We import them
    # lazily so stage-1 users can ``import hhtools.retarget.newton_basic``
    # without paying the ~1 s warp initialisation cost.
    from hhtools.retarget.newton_basic.pipeline import (
        NewtonBasicPipeline,
        PipelineConfig,
        RetargetedMotion,
        ScaledMotionPreview,
    )
    from hhtools.retarget.newton_basic.robot_model import (
        IKMapping,
        IKMappingEntry,
        NewtonRobotContext,
        build_newton_model,
        resolve_ik_map,
    )
    return {
        "NewtonBasicPipeline": NewtonBasicPipeline,
        "PipelineConfig": PipelineConfig,
        "RetargetedMotion": RetargetedMotion,
        "ScaledMotionPreview": ScaledMotionPreview,
        "IKMapping": IKMapping,
        "IKMappingEntry": IKMappingEntry,
        "NewtonRobotContext": NewtonRobotContext,
        "build_newton_model": build_newton_model,
        "resolve_ik_map": resolve_ik_map,
    }


def __getattr__(name):
    if name in {
        "NewtonBasicPipeline", "PipelineConfig", "RetargetedMotion",
        "ScaledMotionPreview",
        "IKMapping", "IKMappingEntry", "NewtonRobotContext",
        "build_newton_model", "resolve_ik_map",
    }:
        return _stage2_lazy_imports()[name]
    raise AttributeError(name)


__all__ = [
    "FeetStabilizer",
    "FeetStabilizerConfig",
    "HumanToRobotScaler",
    "IKMapping",
    "IKMappingEntry",
    "JointLimitClamper",
    "NewtonBasicPipeline",
    "NewtonRobotContext",
    "PipelineConfig",
    "RetargetedMotion",
    "ScaledEffectors",
    "ScaledMotionPreview",
    "ScalerConfig",
    "SourceRestPose",
    "StabilizationStats",
    "build_newton_model",
    "load_feet_stabilizer_config",
    "load_scaler_config",
    "rest_pose_from_motion",
    "rest_pose_from_motion_bind",
    "rest_pose_from_reference",
    "resolve_ik_map",
]
