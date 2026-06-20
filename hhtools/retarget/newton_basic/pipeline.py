# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions adapted from soma-retargeter (Apache-2.0).
# See https://github.com/NVlabs/SOMA-Retargeter and the project root NOTICE.
"""Newton-based IK retargeting pipeline (stage 2).

This is the hhtools port of ``soma_retargeter.pipelines.newton_pipeline`` —
the top-level orchestration that threads a human motion through the scaler,
feet stabilizer, Newton IK solver, joint limit clamper, and finally CSV
export.  The implementation walks the same flow as upstream but is re-cut to
consume hhtools' :class:`~hhtools.core.motion.Motion` /
:class:`~hhtools.robot.base.RobotModel` contracts, so there's no BVH / USD
dependency on the input side.

Flow per input motion:

    Motion --(scaler)--> (F, M, 7) effector targets
           --(feet stabilizer)--> (F, M, 7) constrained targets
           --(per-frame Newton IK)--> (F, joint_coord_count) joint_q
           --(joint limit clamper)--> same, hard-clipped to URDF limits
           --(velocity rate limiter)--> smoothed joint_q
           --> RetargetedMotion { joint_q, sample_rate, dof_names, ... }

We intentionally run one IK problem per frame (sequential) for clarity — the
upstream multi-env batching is reserved for batch export and is used when
``num_envs > 1`` on the :class:`NewtonRobotContext`.

**CUDA vs CPU:** when Warp's current device is CUDA and
:attr:`PipelineConfig.ik_use_cuda_graph` is ``True`` (default), the IK
``solver.step`` is recorded once with ``wp.ScopedCapture`` (same pattern as
soma-retargeter's ``NewtonPipeline`` when dynamic collision / temporal
objectives are off).  Each frame then updates objective targets from Python and
replays the graph with ``wp.capture_launch``.  On CPU, non-CUDA devices, or if
capture fails, the pipeline falls back to per-frame ``solver.step`` calls.
Disable the graph explicitly via ``PipelineConfig(ik_use_cuda_graph=False)``
when debugging.  Other mitigations for very large clips: lower
``ik_iterations`` in the viewer or cap ``max_frames`` for previews.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

import newton
import newton.ik as ik
import warp as wp

from hhtools.core.motion import Motion
from hhtools.robot.loader import URDFRobotModel
from hhtools.retarget.newton_basic._warp_config import configure as configure_warp_cache
from hhtools.retarget.newton_basic.config import (
    FeetStabilizerConfig,
    ScalerConfig,
)
from hhtools.retarget.newton_basic.feet_stabilizer import FeetStabilizer
from hhtools.retarget.newton_basic.heading_align import align_effector_tensor_to_source_heading
from hhtools.retarget.newton_basic.human_aliases import (
    effectors_to_canonical_table,
    is_smpl_like,
)
from hhtools.retarget.newton_basic.joint_limit_clamper import JointLimitClamper
from hhtools.retarget.newton_basic.robot_model import (
    IKMapping,
    IKMappingEntry,
    NewtonRobotContext,
    build_newton_model,
)
from hhtools.retarget.newton_basic.scaler import HumanToRobotScaler

from hhtools.core.math import quaternion as Q


__all__ = [
    "NewtonBasicPipeline",
    "PipelineConfig",
    "RetargetedMotion",
    "ScaledMotionPreview",
    "align_effector_tensor_to_source_heading",
    "is_newton_ik_prewarmed",
    "prewarm_newton_ik_for_robot",
]


_log = logging.getLogger(__name__)

# Warp / CUDA context is not safe across threads.  The lock serialises
# *concurrent* GPU sessions (background prewarm vs retarget worker), **not**
# clips inside :meth:`NewtonBasicPipeline.run_batch` — batch retarget builds
# one multi-env model (``num_envs=N``) and runs a single ``solver.step`` per
# frame for all N clips in parallel inside one lock acquisition.
_WARP_IK_LOCK = threading.Lock()
_cuda_graph_disabled_process = False


def _default_ik_use_cuda_graph() -> bool:
    raw = os.environ.get("HHTOOLS_IK_USE_CUDA_GRAPH", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Position-only motion helpers
# ---------------------------------------------------------------------------


def _is_position_only(motion: Motion) -> bool:
    """True when the source motion carries no meaningful quaternion data.

    Holosoma and similar position-only mocap rigs fill ``quaternions`` with
    xyzw identity ``[0, 0, 0, 1]``.  Detecting this lets the pipeline
    avoid using the (constant) quaternion targets that would otherwise pin
    the robot to a fixed heading.
    """
    q = np.asarray(motion.quaternions, dtype=np.float32)
    identity = np.zeros(4, dtype=np.float32)
    identity[3] = 1.0
    return float(np.abs(q - identity[None, None, :]).max()) < 0.05


def _compute_perframe_yaw(
    ik_targets: NDArray,
    entries,
) -> NDArray | None:
    """Derive per-frame Z-axis yaw quaternion from the shoulder/hip axis.

    Returns ``(F, 4)`` xyzw quaternions or ``None`` when the required
    landmarks cannot be found in *entries*.

    Uses both shoulder and hip axes when available, averaging them for
    robustness against arm-swing noise.
    """
    shoulder_l = shoulder_r = hip_l = hip_r = None
    for i, e in enumerate(entries):
        if e.canonical_name == "left_shoulder":
            shoulder_l = i
        elif e.canonical_name == "right_shoulder":
            shoulder_r = i
        elif e.canonical_name == "left_hip":
            hip_l = i
        elif e.canonical_name == "right_hip":
            hip_r = i

    pairs: list[tuple[int, int]] = []
    if shoulder_l is not None and shoulder_r is not None:
        pairs.append((shoulder_l, shoulder_r))
    if hip_l is not None and hip_r is not None:
        pairs.append((hip_l, hip_r))
    if not pairs:
        return None

    F = ik_targets.shape[0]
    up = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    all_angles: list[NDArray] = []

    for l_idx, r_idx in pairs:
        p_l = ik_targets[:, l_idx, :3].copy()
        p_r = ik_targets[:, r_idx, :3].copy()
        lr = p_l - p_r
        lr[:, 2] = 0.0
        fwd = np.cross(lr, np.broadcast_to(up, (F, 3)))
        fwd[:, 2] = 0.0
        fwd_norm = np.linalg.norm(fwd, axis=1, keepdims=True)
        fwd = fwd / np.maximum(fwd_norm, 1e-6)
        all_angles.append(np.arctan2(fwd[:, 1], fwd[:, 0]))

    if len(all_angles) == 2:
        # Average shoulder and hip yaw via unit-vector mean to handle
        # the ±π wrap-around correctly.
        cx = np.cos(all_angles[0]) + np.cos(all_angles[1])
        sy = np.sin(all_angles[0]) + np.sin(all_angles[1])
        angles = np.arctan2(sy, cx).astype(np.float32)
    else:
        angles = all_angles[0].astype(np.float32)

    # Temporal smoothing: 3-frame causal moving average to reduce
    # high-frequency jitter from position noise while preserving
    # intentional heading changes.
    if F >= 3:
        padded = np.pad(angles, 1, mode="edge")
        kernel = np.ones(3, dtype=np.float32) / 3.0
        angles = np.convolve(padded, kernel, mode="valid").astype(np.float32)

    yaw_q = np.zeros((F, 4), dtype=np.float32)
    yaw_q[:, 2] = np.sin(angles / 2)  # z (xyzw)
    yaw_q[:, 3] = np.cos(angles / 2)  # w
    return yaw_q


def _warp_device_is_cuda() -> bool:
    try:
        return bool(wp.get_device().is_cuda)
    except Exception:  # pragma: no cover — no Warp context yet
        return False


def _cuda_graph_enabled(config: PipelineConfig) -> bool:
    if _cuda_graph_disabled_process:
        return False
    return bool(config.ik_use_cuda_graph and _warp_device_is_cuda())


def _disable_cuda_graph_globally(reason: str) -> None:
    global _cuda_graph_disabled_process
    if not _cuda_graph_disabled_process:
        _log.warning(
            "Disabling IK CUDA graphs for this process (%s). "
            "Set HHTOOLS_IK_USE_CUDA_GRAPH=0 to silence capture attempts.",
            reason,
        )
        _cuda_graph_disabled_process = True
    try:
        wp.synchronize()
    except Exception:  # pragma: no cover — driver edge cases
        pass


def _try_capture_ik_step(step_fn: Callable[[], None]) -> object | None:
    try:
        with wp.ScopedCapture() as cap:
            step_fn()
        return cap.graph
    except Exception as err:  # pragma: no cover — driver / Warp edge cases
        _log.warning("IK CUDA graph capture failed, using eager steps: %s", err)
        _disable_cuda_graph_globally(str(err))
        return None


def _run_ik_step(
    *,
    ik_graph: object | None,
    step_fn: Callable[[], None],
) -> object | None:
    """Replay a captured CUDA graph or fall back to an eager ``step_fn``."""
    if ik_graph is None:
        step_fn()
        return None
    try:
        wp.capture_launch(ik_graph)
        return ik_graph
    except Exception as err:  # pragma: no cover — stale / poisoned graphs
        _log.warning("IK CUDA graph replay failed, using eager steps: %s", err)
        _disable_cuda_graph_globally(str(err))
        step_fn()
        return None


def _warp_ik_guard():
    return _WARP_IK_LOCK if _warp_device_is_cuda() else nullcontext()


# ---------------------------------------------------------------------------
# Auto-endpoint offset helpers (toe / hand-end)
# ---------------------------------------------------------------------------


def _compute_endpoint_offset(
    mj_model,
    link_name: str,
    endpoint_type: str,
) -> tuple[float, float, float] | None:
    """Derive a toe-tip or hand-end local offset from MuJoCo collision geometry.

    Re-uses the collision analysis from
    :func:`~hhtools.retarget.interaction_mesh.contact_points.build_contact_mpc_points`
    but returns a single ``(x, y, z)`` offset in the link's body frame.

    * ``endpoint_type="toe"``:  most forward (+X) sole-level point.
    * ``endpoint_type="hand_end"``:  farthest collision point from the link
      origin (maximises reach).

    Returns ``None`` when the link has no usable collision geometry.
    """
    from hhtools.retarget.interaction_mesh.contact_points import (
        _body_collision_offsets,
        _pick_extreme,
    )

    pts = _body_collision_offsets(mj_model, link_name)
    if pts.shape[0] == 0:
        return None

    if endpoint_type == "toe":
        z_low = float(np.min(pts[:, 2]))
        sole = pts[pts[:, 2] <= z_low + 0.03]
        use = sole if sole.shape[0] >= 2 else pts
        tip = _pick_extreme(use, 0, 1.0)
    else:
        tip = pts[int(np.argmax(np.linalg.norm(pts, axis=1)))].copy()

    off = (float(tip[0]), float(tip[1]), float(tip[2]))
    if np.linalg.norm(tip) < 1e-4:
        return None
    return off


# --------------------------------------------------------------------------- configs


@dataclass
class PipelineConfig:
    """Knobs for :class:`NewtonBasicPipeline`.

    Attributes:
        ik_iterations: Newton IK Levenberg-Marquardt iterations per frame.
            Defaults mirror the upstream G1 config; lower values (~8) trade
            accuracy for speed in preview modes.
        joint_limit_weight: Weight of the hard-margin
            :class:`newton.ik.IKObjectiveJointLimit` added to the solver.
        smooth_joint_filter_weight: Weight of the soft
            :class:`IKSmoothJointFilter` (midpoint pull).  Set to 0 to
            disable.  Default ``5.5`` matches upstream soma-retargeter
            RP1 / G1 configs — the joint-filter regularises the
            redundant yaw / outer-gimbal DOFs that hhtools' shoulder and
            hip ik_map targets (roll_link, the middle of the gimbal)
            leave in the null space.  With weight 0 the solver picks any
            pose in that null space and non-rest clips stabilise at
            wildly rotated postures (e.g. RP1 hip_yaw pegged at 43° on
            a stand-still pickup clip).
        smooth_joint_filter_masks: Optional override for the per-link
            mask dict used by :class:`IKSmoothJointFilter`.  When
            ``None``, the pipeline falls back to
            ``robot.preset.smooth_joint_filter_masks`` from the robot
            yaml (set on RP1 & G1 to match the soma mask schedule).
            When an empty dict is explicitly passed, every joint coord
            is regularised uniformly (weight=1.0 on all coords).
        max_joint_velocity: If > 0, rate-limit non-root DOFs between
            successive frames (radians / second).  Keeps CSVs physically
            playable on a real robot.
        max_root_angular_velocity: Rate-limit for the floating-base root's
            4-component quaternion tail (radians / second).  Uses the same
            cap as ``max_joint_velocity`` when unset.
        apply_feet_stabilizer: Whether to apply :class:`FeetStabilizer`
            constraints to the scaled effectors before IK.  Disabled by
            default because the stage-1 defaults don't configure foot-plant
            thresholds — users that want ground-contact enforcement should
            pass a fully-populated :class:`FeetStabilizerConfig`.
        ik_use_cuda_graph: When ``True`` and the Warp device is CUDA, record
            one IK ``step`` CUDA graph (soma-style) and replay it each frame
            after ``set_target_*``.  Ignored on CPU; capture errors log a
            warning and fall back to eager ``step``.
    """

    ik_iterations: int = 24
    joint_limit_weight: float = 10.0
    smooth_joint_filter_weight: float = 5.5
    smooth_joint_filter_masks: dict[str, float] | None = None
    max_joint_velocity: float = 0.0
    max_root_angular_velocity: float = 0.0
    apply_feet_stabilizer: bool = False
    # Warm-up frames prepended to the IK target sequence.  Every prepended
    # frame is a copy of the clip's **frame 0** effector target, so the
    # solver (which warm-starts from its previous joint_q every frame)
    # gets ``num_initialization_frames`` extra passes on a static pose
    # before the real motion begins.  This is the pattern soma-retargeter
    # uses to hide the "first frame looks weird because IK hasn't
    # converged yet" artefact: by the time the solver hits motion
    # frame 0 for real, it's already had N passes to settle into a
    # steady-state configuration close to the target.  Trimmed from
    # :class:`RetargetedMotion` output so the user sees exactly the
    # input duration back — the initialization is purely internal.
    #
    # Default 0 keeps the behaviour identical to previous hhtools
    # releases; set to ~16–32 for production retargets (soma uses 20).
    num_initialization_frames: int = 0
    # Stabilization frames appended after the last real motion frame,
    # same mechanism as ``num_initialization_frames``.  Useful when the
    # clip ends on a sharp pose change (jump landings, stops) and the
    # solver needs a few more passes to converge the final pose.
    # Trimmed from output like the init frames.
    num_stabilization_frames: int = 0
    # Escape-hatch knob: when ``True``, every ``IKObjectiveRotation``
    # *except* the one attached to the pelvis (canonical ``hips``) is
    # constructed with weight 0 so the solver only chases position
    # targets for the rest of the body; the pelvis is additionally
    # swapped for a world-Z-yaw-only target when
    # :attr:`pelvis_yaw_only_rotation_target` is also set.  Default
    # ``False`` — the soma-style scaler (see
    # :func:`hhtools.retarget.calibration.calibration.build_scaler_config_soma_style`)
    # emits body-frame-aligned quaternion targets (``q_out = q_source
    # * q_offset``) so full rotation objectives are the right default
    # again.  The knob is retained for two reasons:
    #  1. users running legacy ``scale_anchor="origin"`` configs hand-
    #     authored before the soma-style builder emitted ``"root"`` may
    #     still want to disable rotation objectives if those configs
    #     carry only positional offsets.
    #  2. A/B experiments comparing position-only vs. full objectives
    #     on hard clips (e.g. cartwheels where rotation overshoot is
    #     worse than position-only drift).
    disable_rotation_objectives: bool = False
    # Companion to :attr:`disable_rotation_objectives`.  When both are
    # ``True``, the pelvis rotation target is replaced with the world-
    # Z yaw extracted from the source root quaternion, keeping the
    # robot upright and only tracking heading.  Default ``False`` —
    # disabled together with the parent knob under the soma-style
    # builder.
    pelvis_yaw_only_rotation_target: bool = False
    # Ground-plane capsule collision (soma ``IKObjectiveGroundCollision``).
    ground_collision_weight: float = 0.0
    ground_collision_z: float = 0.0
    ground_collision_bodies: tuple[dict, ...] = ()
    ground_collision_dynamic_boost: bool = True
    # Post-IK ``_clamp_solved_foot_heights`` anti-float: when ``False``, only
    # lift the root if solved ankles penetrate the ground plane (soma-style
    # ``_clamp_foot_positions`` lift path).  Xsens mocap clips at high fps
    # oscillate solved ankle height during gait; the rate-limited anti-float
    # correction then pumps root Z and looks like vertical bobbing.
    foot_clamp_anti_float: bool = True
    ik_use_cuda_graph: bool = field(default_factory=_default_ik_use_cuda_graph)


# --------------------------------------------------------------------------- output


@dataclass(frozen=True)
class ScaledMotionPreview:
    """Pre-IK effector preview: the scaler's world-space target trajectory.

    Attributes:
        joint_names: Canonical joint names, in the same order as
            :attr:`NewtonRobotContext.ik_mapping.entries` — so frame-i slice
            ``transforms[i, k, :]`` corresponds to the *k*-th entry of the
            robot's ik_map.
        transforms: ``(F, M, 7)`` array of ``(x, y, z, qx, qy, qz, qw)``
            effector targets in world space, after scaler + (optional) feet
            stabilizer + canonical rename.  ``NaN`` columns flag ik_map
            entries that the scaler didn't populate (mismatch between
            ``scaler.joint_scales`` and ``robot.ik_map``).
        source_seg_src / source_seg_dst / source_transforms: optional dense
            skeleton preview — same heading alignment as ``transforms``, but
            one row per :class:`~hhtools.retarget.newton_basic.scaler.HumanToRobotScaler`
            joint in ``source_joint_names`` order, wired with motion-hierarchy
            parent indices so the yellow overlay matches high-DOF mocap rigs.
        source_bead_indices: optional index list for which ``source_transforms`` rows
            get joint beads in the yellow overlay (fingers / toes culled for clarity).
    """

    joint_names: tuple[str, ...]
    transforms: NDArray
    source_joint_names: tuple[str, ...] | None = None
    source_seg_src: NDArray | None = None
    source_seg_dst: NDArray | None = None
    source_transforms: NDArray | None = None
    source_bead_indices: NDArray | None = None

    @property
    def num_frames(self) -> int:
        return int(self.transforms.shape[0])


from hhtools.retarget.retarget_result import RetargetedMotion  # noqa: F401 — re-export


# --------------------------------------------------------------------------- pipeline


class NewtonBasicPipeline:
    """End-to-end retargeting from ``Motion`` to robot ``joint_q``.

    Construction is eager — we build the Newton model and IK solver up-front
    so subsequent :meth:`run` calls reuse the compiled kernels.  Not
    thread-safe: call ``run`` from a single worker (the UI's preview thread
    is fine).

    Args:
        robot: Loaded :class:`URDFRobotModel` for the target robot.
        scaler_config: Stage-1 :class:`ScalerConfig`; its ``joint_scales``
            keys must match the ``ik_map`` keys on ``robot.yaml`` (or at
            least be a superset — extra scaler joints are ignored).
        feet_stabilizer_config: Optional :class:`FeetStabilizerConfig`.  If
            omitted the pipeline still runs, just without ground-contact
            enforcement.
        pipeline_config: Pipeline knobs; uses upstream G1 defaults when
            omitted.
        human_height: Pelvis-to-head height (metres) of the source subject.
            Mirrors ``soma-retargeter``'s ``human_height`` argument and
            drives the height-ratio correction inside
            :class:`HumanToRobotScaler`.
        configure_warp: Whether to auto-configure Warp's kernel cache
            location on first construction — set ``False`` in tests that
            manage Warp themselves.
    """

    def __init__(
        self,
        robot: URDFRobotModel,
        *,
        scaler_config: ScalerConfig,
        feet_stabilizer_config: FeetStabilizerConfig | None = None,
        pipeline_config: PipelineConfig | None = None,
        human_height: float = 1.7,
        configure_warp: bool = True,
        source_to_canonical: dict[str, str] | None = None,
    ) -> None:
        if configure_warp:
            configure_warp_cache()

        self.robot = robot
        self.scaler_config = scaler_config
        self.feet_stabilizer_config = feet_stabilizer_config
        self.config = pipeline_config or PipelineConfig()
        self.human_height = float(human_height)
        # Optional explicit source-joint → canonical rename.  When ``None``
        # we auto-detect SMPL-family rigs at ``run`` time.
        self._source_to_canonical_override = source_to_canonical

        self.ctx: NewtonRobotContext = build_newton_model(
            robot, num_envs=1
        )
        if self.ctx.ik_mapping is None or not self.ctx.ik_mapping.entries:
            raise ValueError(
                f"robot preset {robot.preset.name!r} has no usable ik_map. "
                f"warnings: {self.ctx.mapping_warnings}"
            )
        if robot.preset.urdf_path is not None and robot.preset.ik_map:
            from hhtools.robot.kinematics import require_valid_ik_map

            require_valid_ik_map(
                robot.preset.urdf_path,
                dict(robot.preset.ik_map),
                robot_name=robot.preset.name,
            )
        self.ik_mapping: IKMapping = self.ctx.ik_mapping

        # Auto-add endpoint objectives (toe, hand-end) so the IK solver
        # tracks the robot's physical extremities, preventing foot
        # penetration and improving hand reach accuracy.
        self._endpoint_entries: tuple[IKMappingEntry, ...] = ()
        _extra = self._build_endpoint_entries()
        if _extra:
            self._endpoint_entries = tuple(_extra)
            self.ik_mapping = IKMapping(
                entries=self.ik_mapping.entries + self._endpoint_entries,
            )
            _log.info(
                "Auto-augmented IK with %d endpoint objectives: %s",
                len(_extra),
                [e.canonical_name for e in _extra],
            )

        # Cache the scaler lazily — we can't build it until we see a source
        # motion (scaler needs the hierarchy).  See :meth:`_build_scaler`.
        self._scaler_by_hierarchy: dict[int, HumanToRobotScaler] = {}
        self._feet_stab_by_hierarchy: dict[int, FeetStabilizer] = {}

        self.clamper = JointLimitClamper.from_robot(robot)
        self._ndof_actuated = len(robot.actuated_joints)

    @classmethod
    def prewarm_for_robot(
        cls,
        robot: URDFRobotModel,
        *,
        pipeline_config: PipelineConfig | None = None,
        ik_iterations: int = 1,
        force: bool = False,
    ) -> bool:
        """JIT-compile Warp IK kernels (+ CUDA graph) for ``robot``.

        Safe to call from a background thread after the robot URDF loads.
        Subsequent :class:`NewtonBasicPipeline` instances for the same preset
        reuse the on-disk Warp cache and pay only solver setup cost.

        Returns ``True`` when a dummy solve ran, ``False`` when skipped.
        """
        return prewarm_newton_ik_for_robot(
            robot,
            pipeline_config=pipeline_config,
            ik_iterations=ik_iterations,
            force=force,
        )

    def _effectors_to_canonical_map(
        self,
        scaler: HumanToRobotScaler,
        targets: NDArray,
    ) -> dict[str, NDArray]:
        """Single choke-point: scaler rows → canonical keys (``run`` / batch / preview)."""

        return effectors_to_canonical_table(
            scaler.joint_names,
            targets,
            source_to_canonical=self._source_to_canonical_override,
        )

    # ---------------------------------------------------------------- public API

    def scale_only(self, motion: Motion) -> "ScaledMotionPreview":
        """Run stages 1-3 (scaler → stabilizer → canonical rename) only.

        Returns the world-space effector trajectory aligned with the robot's
        ik_map, *before* IK.  Useful for previewing "what the robot's
        effectors would chase" without paying for the Newton solve — the UI
        uses this to draw a scaled-human skeleton next to the robot so the
        user can eyeball the target envelope.
        """
        motion = self._ensure_z_up(motion)
        if motion.num_frames == 0:
            return ScaledMotionPreview(
                joint_names=tuple(e.canonical_name for e in self.ik_mapping.entries),
                transforms=np.zeros(
                    (0, len(self.ik_mapping.entries), 7), dtype=np.float32
                ),
            )
        scaler = self._build_scaler(motion)
        scaled = scaler.apply(motion)
        targets = scaled.transforms  # (F, M_scaler, 7)
        if self.config.apply_feet_stabilizer and self.feet_stabilizer_config is not None:
            stab = self._build_feet_stabilizer(scaler.joint_names)
            targets = stab.apply(targets)

        canonical_to_target = self._effectors_to_canonical_map(scaler, targets)
        canonical_to_target = self._augment_canonical_targets(
            motion, scaler, targets, canonical_to_target,
        )

        # Preserve ik_map order so downstream users can index alongside
        # ``ik_mapping.entries``.  Missing canonical joints (ik_map references
        # something the scaler didn't produce) become NaN — the previewer can
        # skip or highlight those instead of crashing.
        names_list = [e.canonical_name for e in self.ik_mapping.entries]
        F = targets.shape[0]
        out = np.full((F, len(names_list), 7), np.nan, dtype=np.float32)
        for i, canon in enumerate(names_list):
            tgt = canonical_to_target.get(canon)
            if tgt is not None:
                out[:, i, :] = tgt

        names = tuple(names_list)
        out = self._align_preview_to_source_heading(out)
        return ScaledMotionPreview(joint_names=names, transforms=out)

    @staticmethod
    def _ensure_z_up(motion: Motion) -> Motion:
        """Convert to Z-up if needed; log a warning on mismatch."""
        if motion.up_axis == "Z":
            return motion
        from hhtools.core.coord import to_up_axis
        _log.warning(
            "Motion %r has up_axis=%r; converting to Z-up before retarget.",
            motion.name, motion.up_axis,
        )
        return to_up_axis(motion, "Z")

    @staticmethod
    def _floor_normalize_motion(motion: Motion) -> Motion:
        """Subtract the clip foot-floor so IK targets match the yellow overlay.

        Interaction-mesh retarget already does this; Newton basic relied on
        calibration ``root_z_offset`` alone, which leaves prone frames below
        ``z=0`` when the clip's lowest foot contact is not at frame 0.
        """
        from dataclasses import replace

        from hhtools.core.grounding import human_source_floor_z_world

        z_min = float(human_source_floor_z_world(motion))
        if abs(z_min) < 1e-6:
            return motion
        pos = np.asarray(motion.positions, dtype=np.float32).copy()
        pos[:, :, 2] -= np.float32(z_min)
        return replace(motion, positions=pos)

    def run(
        self,
        motion: Motion,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> RetargetedMotion:
        """Retarget a single ``Motion`` to the target robot.

        Args:
            motion: Source human motion.
            progress_callback: Optional ``fn(frame_done, frame_total)`` invoked
                after each per-frame IK solve.  Exceptions raised by the
                callback are swallowed so a buggy UI doesn't abort the solve.
        """
        motion = self._ensure_z_up(motion)
        motion = self._floor_normalize_motion(motion)
        if motion.num_frames == 0:
            return RetargetedMotion(
                name=motion.name,
                joint_q=np.zeros((0, self.ctx.joint_coord_count), dtype=np.float32),
                sample_rate=motion.framerate,
                dof_names=self.robot.dof_names(),
                meta={"robot": self.robot.preset.name},
            )

        # 1. Scale source motion into effector targets.
        scaler = self._build_scaler(motion)
        scaled = scaler.apply(motion)

        # 2. Optional feet stabilizer.
        targets = scaled.transforms  # (F, M, 7)
        hand_ground_drop: np.ndarray | None = None
        if self.config.apply_feet_stabilizer and self.feet_stabilizer_config is not None:
            stab = self._build_feet_stabilizer(scaler.joint_names)
            targets = stab.apply(targets)
            if stab.hand_ground_drop_per_frame.size:
                hand_ground_drop = stab.hand_ground_drop_per_frame.astype(
                    np.float32, copy=False,
                )

        # 3. Translate scaler-native joint names (often SMPL-X: pelvis /
        # spine1 / …) to the canonical hhtools names that robot.yaml ik_map
        # is authored against (hips / spine / …).
        canonical_to_target = self._effectors_to_canonical_map(scaler, targets)
        canonical_to_target = self._augment_canonical_targets(
            motion, scaler, targets, canonical_to_target,
        )

        missing = [
            e.canonical_name
            for e in self.ik_mapping.entries
            if e.canonical_name not in canonical_to_target
        ]
        if missing:
            raise KeyError(
                f"robot ik_map references canonical joints not produced by "
                f"the scaler: {missing}. Source joint names: "
                f"{list(scaler.joint_names)}. Consider passing "
                f"source_to_canonical={{...}} to NewtonBasicPipeline or "
                f"editing robot.yaml."
            )
        # (F, M_mapped, 7) aligned with ik_mapping.entries order.
        ik_targets = np.stack(
            [canonical_to_target[e.canonical_name] for e in self.ik_mapping.entries],
            axis=1,
        ).astype(np.float32, copy=False)

        from hhtools.robot.retarget_profile import apply_upper_body_lateral_ik_narrowing

        ik_targets = apply_upper_body_lateral_ik_narrowing(
            ik_targets, self.ik_mapping.entries, self.robot.preset,
            robot_model=self.robot,
        )

        # 3a. Prepend / append warm-up frames so the IK solver can settle
        # before the real motion starts (and after it ends).  See
        # :attr:`PipelineConfig.num_initialization_frames` for the
        # rationale.  We track the padding explicitly so we can trim
        # output to the original duration at the end.
        n_init = max(0, int(self.config.num_initialization_frames))
        n_stab = max(0, int(self.config.num_stabilization_frames))
        if n_init > 0 or n_stab > 0:
            init_block = np.repeat(
                ik_targets[:1], n_init, axis=0
            ) if n_init > 0 else ik_targets[:0]
            stab_block = np.repeat(
                ik_targets[-1:], n_stab, axis=0
            ) if n_stab > 0 else ik_targets[:0]
            ik_targets = np.concatenate(
                [init_block, ik_targets, stab_block], axis=0
            )

        # 3b. Optional escape hatch for legacy / experimental configs:
        # replace the pelvis rotation target with a world-Z yaw-only
        # quaternion so the robot stays upright while tracking subject
        # heading.  Off by default; only runs when both
        # ``disable_rotation_objectives`` and
        # ``pelvis_yaw_only_rotation_target`` are set on the
        # PipelineConfig.  Non-pelvis rotation weights are zeroed inside
        # ``_solve_ik_sequence`` via the same flag.
        if (
            self.config.disable_rotation_objectives
            and self.config.pelvis_yaw_only_rotation_target
        ):
            # Local import keeps the pipeline module importable even if
            # the calibration package isn't installed (e.g. headless
            # test harnesses that stub it out).
            from hhtools.retarget.calibration.calibration import _extract_yaw_quat

            pelvis_idx = next(
                (
                    i
                    for i, e in enumerate(self.ik_mapping.entries)
                    if e.canonical_name in ("hips", "pelvis")
                ),
                None,
            )
            if pelvis_idx is not None:
                fwd_body = (
                    np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    if is_smpl_like(scaler.joint_names)
                    else np.array([1.0, 0.0, 0.0], dtype=np.float32)
                )
                pelvis_q_src = ik_targets[:, pelvis_idx, 3:7].copy()
                yaw_q = np.zeros_like(pelvis_q_src)
                for f in range(pelvis_q_src.shape[0]):
                    yaw_q[f] = _extract_yaw_quat(pelvis_q_src[f], fwd_body)
                ik_targets[:, pelvis_idx, 3:7] = yaw_q

        # 3c. Position-only source data (e.g. holosoma): all source
        # quaternions are identity, so the scaler produced constant
        # rotation targets that would pin the robot to its initial
        # heading.  Replace the pelvis rotation target with a per-frame
        # yaw derived from the shoulder/hip axis and zero non-pelvis
        # rotation weights so the solver is guided by positions alone.
        _pos_only = _is_position_only(motion)
        if _pos_only:
            _log.info(
                "Position-only source detected — computing per-frame "
                "heading from shoulder/hip positions."
            )
            pelvis_idx = next(
                (
                    i
                    for i, e in enumerate(self.ik_mapping.entries)
                    if e.canonical_name in ("hips", "pelvis")
                ),
                None,
            )
            yaw_q = _compute_perframe_yaw(ik_targets, self.ik_mapping.entries)
            if pelvis_idx is not None and yaw_q is not None:
                ik_targets[:, pelvis_idx, 3:7] = yaw_q

        # 4. Solve IK per frame (including any prepended init / appended
        # stab frames; see step 3a).
        drop_for_ik = hand_ground_drop
        if drop_for_ik is not None and (n_init > 0 or n_stab > 0):
            padded = np.zeros(ik_targets.shape[0], dtype=np.float32)
            n_drop = min(int(drop_for_ik.shape[0]), int(ik_targets.shape[0]) - n_init - n_stab)
            if n_drop > 0:
                padded[n_init : n_init + n_drop] = drop_for_ik[:n_drop]
            drop_for_ik = padded

        joint_q_all = self._solve_ik_sequence(
            ik_targets,
            progress_callback=progress_callback,
            force_disable_rotation_objectives=_pos_only,
            hand_ground_drop=drop_for_ik,
        )

        # 4b. Trim warm-up frames so the user sees exactly the original
        # motion duration back.  The solver state carries over between
        # frames so the first "real" output frame benefits from n_init
        # prior IK iterations on the same target.
        if n_init > 0 or n_stab > 0:
            end = joint_q_all.shape[0] - n_stab if n_stab > 0 else joint_q_all.shape[0]
            joint_q_all = joint_q_all[n_init:end]

        # 5. Hard-clip DOFs to URDF limits.
        # Our clamper works on the actuated-DOF slice (past the 7-coord root).
        root7 = joint_q_all[:, : self.ctx.root_coord_count]
        dof = joint_q_all[:, self.ctx.root_coord_count :]
        # Newton may add extra coords past the declared actuated count for
        # mimic / constrained joints; we only clamp the first N columns
        # that correspond to the robot's actuated joints.
        n_clamp = min(dof.shape[1], self._ndof_actuated)
        if n_clamp > 0:
            dof[:, :n_clamp] = self.clamper.apply(dof[:, :n_clamp])

        # 6. Optional per-frame velocity rate limiter.
        if self.config.max_joint_velocity > 0.0:
            joint_q_all = self._rate_limit(
                root7=root7, dof=dof, framerate=motion.framerate,
            )
        else:
            joint_q_all = np.concatenate([root7, dof], axis=1)

        # 7. Truncate to the CSV-exportable width (root + actuated).
        csv_width = self.ctx.root_coord_count + self._ndof_actuated
        joint_q_out = joint_q_all[:, :csv_width].astype(np.float32, copy=False)

        # 8. Align output heading with source: inverse-rotate the root so
        #    the robot moves in the same direction as the source skeleton.
        joint_q_out = self._align_root_to_source_heading(joint_q_out)

        # 9. Root displacement is already scaled inside HumanToRobotScaler so
        #    IK targets and final floating-base motion stay consistent.
        joint_q_out = self._rescale_root_displacement(joint_q_out)
        joint_q_out = self._clamp_solved_foot_heights(joint_q_out)
        joint_q_out = self._clamp_solved_foot_lateral(joint_q_out)

        from hhtools.robot.retarget_profile import apply_upper_body_roll_narrowing_post_ik

        joint_q_out = apply_upper_body_roll_narrowing_post_ik(
            joint_q_out,
            self.robot.dof_names(),
            self.robot.preset,
            root_coord_count=self.ctx.root_coord_count,
            robot_model=self.robot,
        )

        return RetargetedMotion(
            name=motion.name,
            joint_q=joint_q_out,
            sample_rate=motion.framerate,
            dof_names=self.robot.dof_names(),
            root_coord_count=self.ctx.root_coord_count,
            meta={
                "robot": self.robot.preset.name,
                "num_mapped_joints": len(self.ik_mapping.entries),
                "used_mjcf": self.ctx.used_mjcf,
                "ik_iterations": self.config.ik_iterations,
                "human_height": self.human_height,
            },
        )

    def run_many(self, motions: Iterable[Motion]) -> list[RetargetedMotion]:
        """Convenience wrapper around :meth:`run` for a batch of motions.

        .. deprecated::
            Prefer :meth:`run_batch` which solves all motions in a single
            GPU-parallel Newton step (multi-env layout) instead of looping
            sequentially.  ``run_many`` is kept for backward compatibility.
        """
        return [self.run(m) for m in motions]

    # ---------------------------------------------------------------- GPU-parallel batch

    def run_batch(
        self,
        motions: list[Motion],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[RetargetedMotion]:
        """Retarget multiple motions in GPU-parallel (multi-env Newton).

        All *N* motions are solved simultaneously: a single ``solver.step()``
        per frame advances all *N* environments at once, matching
        ``soma-retargeter``'s multi-env layout.  Throughput scales roughly
        linearly in *N* for moderate batch sizes (≤ 16 on most GPUs).

        The GPU device lock (see module-level ``_WARP_IK_LOCK``) is held once
        for the entire batched IK pass — it does **not** force clip-by-clip
        serial solves.  Only overlapping sessions (e.g. prewarm + retarget on
        different threads) are serialised.

        Shorter motions are padded to the longest clip's frame count (last
        frame repeated) so the solver processes a rectangular buffer;
        padding is stripped before output.

        Args:
            motions: One or more source :class:`~hhtools.core.motion.Motion`
                objects.  Different hierarchies within the same batch are
                fine — each motion gets its own scaler.
            progress_callback: ``fn(frame_done, max_frame)`` called once per
                frame across *all* envs.  The callback receives the global
                frame index (0-based), not per-motion counts.

        Returns:
            One :class:`RetargetedMotion` per input motion, in the same
            order.  Empty motions (0 frames) yield zero-row results.
        """
        if not motions:
            return []
        motions = [self._ensure_z_up(m) for m in motions]
        if len(motions) == 1:
            return [self.run(motions[0], progress_callback=progress_callback)]

        entries = self.ik_mapping.entries
        M = len(entries)
        n_init = max(0, int(self.config.num_initialization_frames))
        n_stab = max(0, int(self.config.num_stabilization_frames))

        # --- 1. Pre-compute IK targets for each motion (CPU) ----------------
        all_targets: list[NDArray] = []
        real_frame_counts: list[int] = []  # original clip length (no padding)
        for m in motions:
            if m.num_frames == 0:
                all_targets.append(
                    np.zeros((0, M, 7), dtype=np.float32)
                )
                real_frame_counts.append(0)
                continue
            ik_tgt = self._build_ik_targets(m)
            all_targets.append(ik_tgt)
            real_frame_counts.append(m.num_frames)

        # Separate empty motions from non-empty ones.
        active: list[tuple[int, NDArray]] = [
            (i, t) for i, t in enumerate(all_targets) if t.shape[0] > 0
        ]
        if not active:
            return [
                self._empty_result(m) for m in motions
            ]

        active_idx = [i for i, _ in active]
        active_tgt = [t for _, t in active]
        N = len(active_tgt)
        max_frames = max(t.shape[0] for t in active_tgt)

        # --- 2. Pad to rectangular (N, max_frames, M, 7) -------------------
        padded = np.zeros((N, max_frames, M, 7), dtype=np.float32)
        padded_counts: list[int] = []
        for j, t in enumerate(active_tgt):
            F = t.shape[0]
            padded[j, :F] = t
            if F < max_frames:
                padded[j, F:] = t[-1:]
            padded_counts.append(F)

        # --- 3. Batched solve (GPU): single-robot model, N parallel problems
        # Newton's ``IKSolver`` takes a SINGLE-robot model plus ``n_problems=N``
        # and factorises one ``dof × dof`` Cholesky tile *per problem* in
        # parallel, so per-block shared memory is independent of N (this is
        # exactly soma-retargeter's layout, which batches ~100 clips at once).
        # Feeding an N-robot *merged* model here instead would make
        # ``model.joint_coord_count == N · dof`` and each Cholesky tile
        # ``(N · dof)²`` — blowing the per-block shared-memory budget and
        # forcing the clip-by-clip fallback.  We therefore reuse ``self.ctx``
        # (built with ``num_envs=1``) and let the solver replicate it N ways.
        any_pos_only = any(_is_position_only(motions[i]) for i in active_idx)
        per_env_jq = self._solve_ik_batch(
            self.ctx, padded, padded_counts,
            progress_callback=progress_callback,
            force_disable_rotation_objectives=any_pos_only,
        )

        # --- 4. Post-process per motion (CPU) -------------------------------
        results: list[RetargetedMotion | None] = [None] * len(motions)
        for j, orig_idx in enumerate(active_idx):
            m = motions[orig_idx]
            jq = per_env_jq[j]

            # Trim warm-up / stabilization padding.
            real_f = real_frame_counts[orig_idx]
            if n_init > 0 or n_stab > 0:
                jq = jq[n_init : n_init + real_f]
            else:
                jq = jq[:real_f]

            results[orig_idx] = self._postprocess_joint_q(jq, m)

        # Fill empty-motion slots.
        for i, m in enumerate(motions):
            if results[i] is None:
                results[i] = self._empty_result(m)

        return results  # type: ignore[return-value]

    # ---------------------------------------------------------------- internals

    def _empty_result(self, m: Motion) -> RetargetedMotion:
        return RetargetedMotion(
            name=m.name,
            joint_q=np.zeros(
                (0, self.ctx.joint_coord_count), dtype=np.float32
            ),
            sample_rate=m.framerate,
            dof_names=self.robot.dof_names(),
            meta={"robot": self.robot.preset.name},
        )

    def _build_ik_targets(self, motion: Motion) -> NDArray:
        """Scale a single motion and return padded IK targets ``(F', M, 7)``.

        Applies the scaler, optional feet stabiliser, canonical rename, and
        warm-up / stabilisation padding.  The returned array's frame count
        includes ``num_initialization_frames + real_frames +
        num_stabilization_frames``.

        For position-only source data the pelvis rotation column is
        replaced with per-frame yaw derived from shoulder/hip positions
        (same logic as :meth:`run`, step 3c).
        """
        entries = self.ik_mapping.entries
        scaler = self._build_scaler(motion)
        scaled = scaler.apply(motion)
        targets = scaled.transforms  # (F, M_scaler, 7)

        if self.config.apply_feet_stabilizer and self.feet_stabilizer_config is not None:
            stab = self._build_feet_stabilizer(scaler.joint_names)
            targets = stab.apply(targets)

        canonical_to_target = self._effectors_to_canonical_map(scaler, targets)
        canonical_to_target = self._augment_canonical_targets(
            motion, scaler, targets, canonical_to_target,
        )

        missing = [
            e.canonical_name
            for e in entries
            if e.canonical_name not in canonical_to_target
        ]
        if missing:
            raise KeyError(
                f"robot ik_map references canonical joints not produced by "
                f"the scaler: {missing}. Source joint names: "
                f"{list(scaler.joint_names)}."
            )

        ik_targets = np.stack(
            [canonical_to_target[e.canonical_name] for e in entries],
            axis=1,
        ).astype(np.float32, copy=False)

        from hhtools.robot.retarget_profile import apply_upper_body_lateral_ik_narrowing

        ik_targets = apply_upper_body_lateral_ik_narrowing(
            ik_targets, entries, self.robot.preset,
            robot_model=self.robot,
        )

        # Warm-up / stabilisation padding.
        n_init = max(0, int(self.config.num_initialization_frames))
        n_stab = max(0, int(self.config.num_stabilization_frames))
        if n_init > 0 or n_stab > 0:
            init_block = (
                np.repeat(ik_targets[:1], n_init, axis=0) if n_init > 0
                else ik_targets[:0]
            )
            stab_block = (
                np.repeat(ik_targets[-1:], n_stab, axis=0) if n_stab > 0
                else ik_targets[:0]
            )
            ik_targets = np.concatenate(
                [init_block, ik_targets, stab_block], axis=0
            )

        # Position-only heading injection (batch path).
        if _is_position_only(motion):
            pelvis_idx = next(
                (i for i, e in enumerate(entries) if e.canonical_name in ("hips", "pelvis")),
                None,
            )
            yaw_q = _compute_perframe_yaw(ik_targets, entries)
            if pelvis_idx is not None and yaw_q is not None:
                ik_targets[:, pelvis_idx, 3:7] = yaw_q

        return ik_targets

    def _augment_canonical_targets(
        self,
        motion: Motion,
        scaler: HumanToRobotScaler,
        targets: NDArray,
        canonical_to_target: dict[str, NDArray],
    ) -> dict[str, NDArray]:
        """Add toe / hand-end targets for auto-detected endpoint objectives.

        * **Toe** (``left_foot`` / ``right_foot``):  Most motion formats
          already produce this canonical via the scaler; when absent (reduced
          skeletons like parc_ms), we fall back to the ankle target so the
          objective degrades gracefully to a redundant ankle constraint.
        * **Hand-end** (``left_hand_end`` / ``right_hand_end``):  Always
          synthesized by extrapolating from the wrist along the elbow→wrist
          direction by the robot's physical hand length (from URDF geometry).
        """
        if not self._endpoint_entries:
            return canonical_to_target

        for entry in self._endpoint_entries:
            canon = entry.canonical_name
            if canon in canonical_to_target:
                continue

            if canon in ("left_foot", "right_foot"):
                side = "left" if canon.startswith("left") else "right"
                fallback = canonical_to_target.get(f"{side}_ankle")
                if fallback is not None:
                    canonical_to_target[canon] = fallback.copy()

            elif canon in ("left_hand_end", "right_hand_end"):
                side = "left" if canon.startswith("left") else "right"
                wrist_tgt = canonical_to_target.get(f"{side}_wrist")
                if wrist_tgt is None:
                    continue
                hand_end_tgt = wrist_tgt.copy()
                try:
                    from hhtools.viewer.anatomy import scaled_hand_tip_positions_world

                    tips = scaled_hand_tip_positions_world(motion, scaler, side)
                except Exception:
                    tips = None
                if tips is not None and tips.shape[0] == wrist_tgt.shape[0]:
                    hand_end_tgt[:, :3] = tips
                else:
                    elbow_tgt = canonical_to_target.get(f"{side}_elbow")
                    if elbow_tgt is not None:
                        direction = wrist_tgt[:, :3] - elbow_tgt[:, :3]
                        norms = np.linalg.norm(direction, axis=1, keepdims=True)
                        direction = direction / np.maximum(norms, 1e-6)
                        hand_len = float(np.linalg.norm(entry.t_offset))
                        hand_end_tgt[:, :3] = wrist_tgt[:, :3] + direction * hand_len
                canonical_to_target[canon] = hand_end_tgt

        return canonical_to_target

    # ---- endpoint entry construction -----------------------------------------

    _ENDPOINT_SPECS: list[tuple[str, str, str]] = [
        # (parent_canonical, new_canonical, endpoint_type)
        ("left_ankle",  "left_foot",       "toe"),
        ("right_ankle", "right_foot",      "toe"),
        ("left_wrist",  "left_hand_end",   "hand_end"),
        ("right_wrist", "right_hand_end",  "hand_end"),
    ]

    def _build_endpoint_entries(self) -> list[IKMappingEntry]:
        """Auto-detect toe / hand-end offsets and create supplementary IK entries.

        For each ankle/wrist entry in the existing ``ik_map``:

        1. If the URDF has a child link *and* it appears in the Newton body
           list, target that child link directly (zero offset).
        2. Otherwise derive a local offset from MuJoCo collision geometry
           (same analysis as :func:`build_contact_mpc_points`) and attach it
           to the parent (ankle/wrist) link.
        3. Skip silently when no usable geometry is available.
        """
        existing = {e.canonical_name for e in self.ik_mapping.entries}
        mj_model = self.robot.mujoco_model
        extra: list[IKMappingEntry] = []
        body_set = set(self.ctx.body_labels)

        for parent_canon, new_canon, etype in self._ENDPOINT_SPECS:
            if new_canon in existing:
                continue
            parent = next(
                (e for e in self.ik_mapping.entries if e.canonical_name == parent_canon),
                None,
            )
            if parent is None:
                continue

            child_link = self._find_urdf_child_link(parent.t_body_link)
            if child_link and child_link in body_set:
                child_idx = self.ctx.body_labels.index(child_link)
                extra.append(IKMappingEntry(
                    canonical_name=new_canon,
                    t_body_link=child_link,
                    r_body_link=child_link,
                    t_body_index=child_idx,
                    r_body_index=child_idx,
                    t_weight=1.0 if etype == "toe" else 0.5,
                    r_weight=0.0,
                ))
                continue

            if mj_model is None:
                continue
            offset = _compute_endpoint_offset(mj_model, parent.t_body_link, etype)
            if offset is None:
                continue
            extra.append(IKMappingEntry(
                canonical_name=new_canon,
                t_body_link=parent.t_body_link,
                r_body_link=parent.r_body_link,
                t_body_index=parent.t_body_index,
                r_body_index=parent.r_body_index,
                t_weight=1.0 if etype == "toe" else 0.5,
                r_weight=0.0,
                t_offset=offset,
            ))

        return extra

    def _find_urdf_child_link(self, link_name: str) -> str | None:
        """Return the first child link of *link_name* in the URDF, or ``None``."""
        link_info = next((lk for lk in self.robot.links if lk.name == link_name), None)
        if link_info is None or not link_info.child_joint_names:
            return None
        for jname in link_info.child_joint_names:
            joint = next((j for j in self.robot.joints if j.name == jname), None)
            if joint is not None:
                return joint.child_link
        return None

    def _postprocess_joint_q(
        self,
        joint_q_all: NDArray,
        motion: Motion,
    ) -> RetargetedMotion:
        """Clamp, rate-limit, align and wrap a raw ``(F, coord)`` array."""
        root7 = joint_q_all[:, : self.ctx.root_coord_count]
        dof = joint_q_all[:, self.ctx.root_coord_count :]
        n_clamp = min(dof.shape[1], self._ndof_actuated)
        if n_clamp > 0:
            dof[:, :n_clamp] = self.clamper.apply(dof[:, :n_clamp])

        if self.config.max_joint_velocity > 0.0:
            joint_q_all = self._rate_limit(
                root7=root7, dof=dof, framerate=motion.framerate,
            )
        else:
            joint_q_all = np.concatenate([root7, dof], axis=1)

        csv_width = self.ctx.root_coord_count + self._ndof_actuated
        jq_out = joint_q_all[:, :csv_width].astype(np.float32, copy=False)
        jq_out = self._align_root_to_source_heading(jq_out)
        jq_out = self._rescale_root_displacement(jq_out)
        jq_out = self._clamp_solved_foot_heights(jq_out)
        jq_out = self._clamp_solved_foot_lateral(jq_out)

        from hhtools.robot.retarget_profile import apply_upper_body_roll_narrowing_post_ik

        jq_out = apply_upper_body_roll_narrowing_post_ik(
            jq_out,
            self.robot.dof_names(),
            self.robot.preset,
            root_coord_count=self.ctx.root_coord_count,
            robot_model=self.robot,
        )

        return RetargetedMotion(
            name=motion.name,
            joint_q=jq_out,
            sample_rate=motion.framerate,
            dof_names=self.robot.dof_names(),
            root_coord_count=self.ctx.root_coord_count,
            meta={
                "robot": self.robot.preset.name,
                "num_mapped_joints": len(self.ik_mapping.entries),
                "used_mjcf": self.ctx.used_mjcf,
                "ik_iterations": self.config.ik_iterations,
                "human_height": self.human_height,
            },
        )

    def _inverse_body_quat(self) -> NDArray | None:
        """Return ``conj(source_body_quat)`` or ``None`` if it's identity."""
        sbq = np.asarray(self.scaler_config.source_body_quat, dtype=np.float32)
        if np.allclose(sbq, [0, 0, 0, 1], atol=1e-7):
            return None
        return Q.conjugate(sbq[None, :])[0]

    def _align_root_to_source_heading(self, joint_q: NDArray) -> NDArray:
        """Counter-rotate the floating-base root so it faces the source heading.

        The IK solver runs in the robot's canonical frame (+X forward),
        but downstream consumers (viewer, CSV export) expect the robot to
        move in the same direction as the original human motion.  Applying
        ``conj(source_body_quat)`` to the root position and quaternion
        undoes the heading alignment that the scaler applied, while
        keeping actuated-joint angles unchanged (they live in joint-local
        frames and are unaffected by a global yaw rotation).
        """
        inv_q = self._inverse_body_quat()
        if inv_q is None:
            return joint_q
        F = joint_q.shape[0]
        q_bc = np.broadcast_to(inv_q[None, :], (F, 4))
        out = joint_q.copy()
        out[:, :3] = Q.rotate(q_bc, joint_q[:, :3]).astype(np.float32)
        out[:, 3:7] = Q.multiply(q_bc, joint_q[:, 3:7]).astype(np.float32)
        return out

    def _align_preview_to_source_heading(self, transforms: NDArray) -> NDArray:
        """Counter-rotate preview positions and quaternions to source heading.

        Same idea as :meth:`_align_root_to_source_heading` but for the
        ``(F, M, 7)`` scaled-effector tensor used by the preview skeleton.
        """
        return align_effector_tensor_to_source_heading(
            transforms,
            source_body_quat=np.asarray(self.scaler_config.source_body_quat, dtype=np.float32),
        )

    def _rescale_root_displacement(self, joint_q: NDArray) -> NDArray:
        """Compatibility no-op for root displacement scaling.

        World trajectory scaling now happens in
        :class:`HumanToRobotScaler`, before IK, so the yellow preview,
        effector targets, and final robot root all share the same shortened
        displacement.  Kept as a separate method because older call sites and
        tests refer to this stage in the pipeline.
        """
        return joint_q

    def _clamp_solved_foot_heights(self, joint_q: NDArray) -> NDArray:
        """Post-IK safety clamp mirroring soma ``_clamp_foot_positions``.

        Immediately lifts the floating base when solved ankle links sit below
        the ground plane.  Anti-floating corrections use an uprightness blend
        and per-frame rate limit so cartwheels are not snapped downward.
        """
        from hhtools.web.serialize import _lowest_ankle_z, _quat_xyzw_to_rotmat

        ik_map = dict(self.robot.preset.ik_map) if self.robot.preset.ik_map else {}
        if not ik_map:
            return joint_q

        _FOOT_COLLISION_OFFSET = 0.05
        _GROUND_CLEARANCE = 0.01
        _MIN_ANKLE_Z = _FOOT_COLLISION_OFFSET + _GROUND_CLEARANCE
        _FLOAT_TOL = 0.015
        _UPRIGHT_BLEND_RANGE = 0.30
        _MAX_FLOAT_CORRECTION = 0.05
        _MAX_CORRECTION_RATE = 0.008

        out = joint_q.astype(np.float32, copy=True)
        dof_names = self.robot.dof_names()
        n_dof = min(out.shape[1] - self.ctx.root_coord_count, len(dof_names))
        if n_dof <= 0:
            return out

        rest_cfg = {dof_names[i]: 0.0 for i in range(n_dof)}
        self.robot.apply_configuration(rest_cfg)
        root_rot0 = _quat_xyzw_to_rotmat(out[0, 3:7])
        rest_ankle = _lowest_ankle_z(self.robot, ik_map, root_rot0)
        rest_foot_z = float(rest_ankle) if rest_ankle is not None else 0.0

        prev_float_correction = 0.0
        for f in range(out.shape[0]):
            cfg = {
                dof_names[i]: float(out[f, self.ctx.root_coord_count + i])
                for i in range(n_dof)
            }
            self.robot.apply_configuration(cfg)
            ankle_z = _lowest_ankle_z(
                self.robot, ik_map, _quat_xyzw_to_rotmat(out[f, 3:7]),
            )
            if ankle_z is None:
                continue
            root_z = float(out[f, 2])
            world_ankle_z = root_z + float(ankle_z)

            if world_ankle_z < _MIN_ANKLE_Z:
                out[f, 2] += np.float32(_MIN_ANKLE_Z - world_ankle_z)
                prev_float_correction = 0.0
                continue

            target = 0.0
            if (
                self.config.foot_clamp_anti_float
                and float(ankle_z) > rest_foot_z + _FLOAT_TOL
            ):
                uprightness = max(
                    0.0,
                    min(1.0, (root_z - world_ankle_z) / _UPRIGHT_BLEND_RANGE),
                )
                target = min(
                    (float(ankle_z) - rest_foot_z - _FLOAT_TOL) * uprightness,
                    _MAX_FLOAT_CORRECTION,
                )
            delta = target - prev_float_correction
            delta = max(-_MAX_CORRECTION_RATE, min(_MAX_CORRECTION_RATE, delta))
            correction = prev_float_correction + delta
            prev_float_correction = correction
            if correction > 0.001:
                out[f, 2] -= np.float32(correction)
        return out

    def _clamp_solved_foot_lateral(self, joint_q: NDArray) -> NDArray:
        """Post-IK hip abduction spread when solved foot meshes overlap."""
        min_clearance = self._resolved_foot_lateral_clearance_m()
        if min_clearance <= 0.0 or joint_q.shape[0] == 0:
            return joint_q

        from hhtools.robot.foot_geometry import clamp_joint_q_foot_lateral_clearance

        feet_cfg = self.feet_stabilizer_config
        ankle_prefilter = 0.0
        if feet_cfg is not None and float(feet_cfg.min_lateral_separation) > 0.0:
            min_lat = float(feet_cfg.min_lateral_separation)
            ankle_prefilter = min_lat + max(0.03, min_lat * 0.3)

        dof_names = self.robot.dof_names()
        out = joint_q.astype(np.float32, copy=True)
        for f in range(out.shape[0]):
            out[f] = clamp_joint_q_foot_lateral_clearance(
                self.robot,
                out[f],
                dof_names,
                root_coord_count=self.ctx.root_coord_count,
                min_clearance_m=min_clearance,
                ankle_prefilter_m=ankle_prefilter if ankle_prefilter > 0.0 else None,
            )
        return out

    #: Default inner foot-mesh clearance (m) applied to every robot when its
    #: config does not specify one.  The post-IK clamp this drives is purely
    #: mesh-gated (a no-op unless the solved foot meshes actually
    #: interpenetrate), so a small positive default is safe universally and
    #: protects narrow-hip / wide-foot robots out of the box.
    _DEFAULT_FOOT_LATERAL_CLEARANCE_M: float = 0.02

    def _resolved_foot_lateral_clearance_m(self) -> float:
        """Target inner foot-mesh clearance from feet stabilizer / robot geometry."""
        feet_cfg = self.feet_stabilizer_config
        if feet_cfg is None:
            # No feet config at all → fall back to the universal default so
            # the mesh-aware clamp still protects every robot.
            return float(self._DEFAULT_FOOT_LATERAL_CLEARANCE_M)
        # An explicit value (including 0.0 to disable) is always respected.
        return float(feet_cfg.min_foot_clearance)

    def _build_scaler(self, motion: Motion) -> HumanToRobotScaler:
        key = id(motion.hierarchy)
        cached = self._scaler_by_hierarchy.get(key)
        if cached is not None:
            return cached
        scaler = HumanToRobotScaler(
            motion.hierarchy,
            self.scaler_config,
            human_height=self.human_height,
        )
        self._scaler_by_hierarchy[key] = scaler
        return scaler

    def _build_ground_collision_objective(self, model) -> ik.IKObjective | None:
        weight = float(self.config.ground_collision_weight)
        bodies = list(self.config.ground_collision_bodies or ())
        if weight <= 0.0 or not bodies:
            return None
        from hhtools.retarget.newton_basic.ik_collision_objective import (
            IKObjectiveGroundCollision,
            resolve_ground_collision_bodies,
        )

        resolved = resolve_ground_collision_bodies(self.ctx.body_labels, bodies)
        if not resolved:
            _log.warning(
                "ground_collision_bodies configured but none match robot links %s",
                self.ctx.body_labels[:8],
            )
            return None
        return IKObjectiveGroundCollision(
            model,
            body_labels=self.ctx.body_labels,
            ground_bodies=resolved,
            weight=weight,
            ground_z=float(self.config.ground_collision_z),
        )

    def _build_feet_stabilizer(
        self, joint_names: tuple[str, ...]
    ) -> FeetStabilizer:
        key = hash(joint_names)
        cached = self._feet_stab_by_hierarchy.get(key)
        if cached is not None:
            return cached
        assert self.feet_stabilizer_config is not None
        stab = FeetStabilizer(
            self.feet_stabilizer_config, joint_names=joint_names,
        )
        self._feet_stab_by_hierarchy[key] = stab
        return stab

    # ---- core Newton IK solve --------------------------------------------------

    def _solve_ik_sequence(
        self,
        ik_targets: NDArray,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        force_disable_rotation_objectives: bool = False,
        hand_ground_drop: NDArray | None = None,
    ) -> NDArray:
        """Run the IK solver frame-by-frame and return ``(F, joint_coord_count)``.

        The IK state persists between frames (warm start from the previous
        solution), which both speeds things up and avoids root-pose
        discontinuities.

        When *force_disable_rotation_objectives* is ``True``, non-pelvis
        rotation objectives are zeroed regardless of the pipeline config.
        Used for position-only source data where the quaternion targets are
        meaningless constants.
        """
        with _warp_ik_guard():
            return self._solve_ik_sequence_inner(
                ik_targets,
                progress_callback=progress_callback,
                force_disable_rotation_objectives=force_disable_rotation_objectives,
                hand_ground_drop=hand_ground_drop,
            )

    def _solve_ik_sequence_inner(
        self,
        ik_targets: NDArray,
        *,
        progress_callback: Callable[[int, int], None] | None,
        force_disable_rotation_objectives: bool,
        hand_ground_drop: NDArray | None = None,
    ) -> NDArray:
        num_frames = ik_targets.shape[0]
        num_mapped = ik_targets.shape[1]

        model = self.ctx.model

        # Build per-objective ``target`` warp arrays; we mutate them in place
        # every frame via ``set_target_position`` / ``set_target_rotation``.
        entries = self.ik_mapping.entries
        pos_targets_np = np.zeros((num_mapped, 3), dtype=np.float32)
        rot_targets_np = np.zeros((num_mapped, 4), dtype=np.float32)
        rot_targets_np[:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        position_objectives: list[ik.IKObjectivePosition] = []
        from hhtools.robot.retarget_profile import effective_ik_t_weight

        preset = self.robot.preset
        for i, entry in enumerate(entries):
            pos_wp = wp.array(pos_targets_np[i : i + 1], dtype=wp.vec3)
            position_objectives.append(
                ik.IKObjectivePosition(
                    link_index=entry.t_body_index,
                    link_offset=wp.vec3(*entry.t_offset),
                    target_positions=pos_wp,
                    weight=effective_ik_t_weight(
                        entry.canonical_name, entry.t_weight, preset,
                        robot_model=self.robot,
                    ),
                )
            )

        _disable_rot = (
            force_disable_rotation_objectives
            or self.config.disable_rotation_objectives
        )
        rotation_objectives: list[ik.IKObjectiveRotation] = []
        for i, entry in enumerate(entries):
            rot_wp = wp.array(rot_targets_np[i : i + 1], dtype=wp.vec4)
            if _disable_rot:
                is_pelvis = entry.canonical_name in ("hips", "pelvis")
                # Boost pelvis rotation weight for position-only data where
                # heading tracking depends solely on this objective.
                weight = max(entry.r_weight, 5.0) if is_pelvis else 0.0
            else:
                weight = entry.r_weight
            rotation_objectives.append(
                ik.IKObjectiveRotation(
                    link_index=entry.r_body_index,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=rot_wp,
                    weight=weight,
                )
            )

        objectives: list[ik.IKObjective] = [*position_objectives, *rotation_objectives]

        if self.config.joint_limit_weight > 0.0:
            objectives.append(
                ik.IKObjectiveJointLimit(
                    joint_limit_lower=model.joint_limit_lower,
                    joint_limit_upper=model.joint_limit_upper,
                    weight=self.config.joint_limit_weight,
                )
            )

        if self.config.smooth_joint_filter_weight > 0.0:
            # Deferred import so the verbose warp kernels only compile when
            # this optional objective is actually enabled.
            from hhtools.retarget.newton_basic.ik_objectives import IKSmoothJointFilter

            coord_masks = self._build_smooth_joint_filter_coord_masks(model)
            objectives.append(
                IKSmoothJointFilter(
                    joint_limit_lower=model.joint_limit_lower,
                    joint_limit_upper=model.joint_limit_upper,
                    weight=self.config.smooth_joint_filter_weight,
                    coord_masks=coord_masks,
                )
            )

        ground_collision_obj = self._build_ground_collision_objective(model)
        if ground_collision_obj is not None:
            objectives.append(ground_collision_obj)

        use_dynamic_ground = (
            ground_collision_obj is not None
            and self.config.ground_collision_dynamic_boost
        )
        _STANDING_ROOT_Z = 0.70
        _GROUND_ROOT_Z = 0.25
        _COLLISION_BOOST = 2.0
        pelvis_entry = next(
            (e for e in entries if e.canonical_name in ("hips", "pelvis")),
            None,
        )
        pelvis_i = entries.index(pelvis_entry) if pelvis_entry is not None else -1
        hips_default_weight = (
            float(entries[pelvis_i].t_weight) if pelvis_i >= 0 else 10.0
        )
        _HIPS_REDUCED_WEIGHT = 3.0

        solver = ik.IKSolver(
            model=model,
            n_problems=1,
            objectives=objectives,
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        joint_q = wp.empty(shape=(1, model.joint_coord_count), dtype=wp.float32)
        wp.copy(joint_q, model.joint_q)

        # First frame: seed the root at the scaled pelvis target.  The Newton
        # default is the URDF zero-pose which often has the pelvis at origin;
        # this prevents an aggressive first-frame root drift that otherwise
        # shows up as a single-frame "jump" in the CSV.
        if pelvis_entry is not None:
            jq = joint_q.numpy().copy()
            jq[0, 0:3] = ik_targets[0, pelvis_i, 0:3]
            jq[0, 3:7] = ik_targets[0, pelvis_i, 3:7]
            joint_q.assign(jq)

        out = np.empty((num_frames, model.joint_coord_count), dtype=np.float32)
        solver.reset()

        if progress_callback is not None:
            try:
                progress_callback(0, num_frames)
            except Exception:  # pragma: no cover
                pass

        ik_graph: object | None = None
        if num_frames > 0 and _cuda_graph_enabled(self.config):

            def _ik_step_once() -> None:
                solver.step(
                    joint_q, joint_q, iterations=self.config.ik_iterations,
                )

            ik_graph = _try_capture_ik_step(_ik_step_once)

        for frame in range(num_frames):
            if use_dynamic_ground and pelvis_i >= 0:
                root_z = float(ik_targets[frame, pelvis_i, 2])
                groundedness = max(
                    0.0,
                    min(
                        1.0,
                        (_STANDING_ROOT_Z - root_z)
                        / max(_STANDING_ROOT_Z - _GROUND_ROOT_Z, 0.01),
                    ),
                )
                boost = 1.0 + _COLLISION_BOOST * groundedness
                ground_collision_obj.set_weight(
                    float(self.config.ground_collision_weight) * boost
                )

            if pelvis_i >= 0 and hand_ground_drop is not None and frame < len(hand_ground_drop):
                drop = float(hand_ground_drop[frame])
                if drop > 0.01:
                    blend = min(1.0, drop / 0.10)
                    position_objectives[pelvis_i].weight = (
                        hips_default_weight * (1.0 - blend)
                        + _HIPS_REDUCED_WEIGHT * blend
                    )
                else:
                    position_objectives[pelvis_i].weight = hips_default_weight

            for i in range(num_mapped):
                position_objectives[i].set_target_position(
                    0, wp.vec3(*ik_targets[frame, i, 0:3])
                )
                rotation_objectives[i].set_target_rotation(
                    0, wp.quat(*ik_targets[frame, i, 3:7])
                )

            def _ik_step_once() -> None:
                solver.step(
                    joint_q, joint_q, iterations=self.config.ik_iterations,
                )

            ik_graph = _run_ik_step(ik_graph=ik_graph, step_fn=_ik_step_once)
            out[frame] = joint_q.numpy()[0]
            if progress_callback is not None:
                try:
                    progress_callback(frame + 1, num_frames)
                except Exception:  # pragma: no cover — UI hooks shouldn't break IK
                    pass

        return out

    # ---- multi-env batch solve ------------------------------------------------

    def _solve_ik_batch(
        self,
        ctx: NewtonRobotContext,
        padded_targets: NDArray,
        frame_counts: list[int],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        force_disable_rotation_objectives: bool = False,
    ) -> list[NDArray]:
        """Multi-env IK: one ``solver.step()`` per frame, *N* envs in parallel.

        Args:
            ctx: Newton context built with ``num_envs=1``.  The IK solver
                replicates this single-robot model into ``n_problems=N``
                independent problems internally; we must **not** pass a merged
                N-robot model or the per-problem Cholesky tile becomes
                ``(N · dof)²`` and exceeds the per-block shared-memory budget.
            padded_targets: ``(N, max_frames, M, 7)`` rectangular target buffer.
            frame_counts: Per-env real frame count (used to trim output).
            progress_callback: ``fn(frame_done, max_frames)``.
            force_disable_rotation_objectives: Zero non-pelvis rotation weights.

        Returns:
            List of ``(F_i, joint_coord_count)`` arrays, one per env.
        """
        with _warp_ik_guard():
            return self._solve_ik_batch_inner(
                ctx,
                padded_targets,
                frame_counts,
                progress_callback=progress_callback,
                force_disable_rotation_objectives=force_disable_rotation_objectives,
            )

    def _solve_ik_batch_inner(
        self,
        ctx: NewtonRobotContext,
        padded_targets: NDArray,
        frame_counts: list[int],
        *,
        progress_callback: Callable[[int, int], None] | None,
        force_disable_rotation_objectives: bool,
    ) -> list[NDArray]:
        N, max_frames, M, _ = padded_targets.shape
        model = ctx.model
        entries = self.ik_mapping.entries

        # --- Objectives (one array per objective, sized for N problems) ------
        position_objectives: list[ik.IKObjectivePosition] = []
        preset = self.robot.preset
        from hhtools.robot.retarget_profile import effective_ik_t_weight

        for entry in entries:
            pos_wp = wp.zeros(shape=N, dtype=wp.vec3)
            position_objectives.append(
                ik.IKObjectivePosition(
                    link_index=entry.t_body_index,
                    link_offset=wp.vec3(*entry.t_offset),
                    target_positions=pos_wp,
                    weight=effective_ik_t_weight(
                        entry.canonical_name, entry.t_weight, preset,
                        robot_model=self.robot,
                    ),
                )
            )

        _disable_rot = (
            force_disable_rotation_objectives
            or self.config.disable_rotation_objectives
        )
        rotation_objectives: list[ik.IKObjectiveRotation] = []
        for entry in entries:
            # Match single-env ``_solve_ik_sequence_inner``: Newton's
            # ``_update_rotation_target`` kernel expects ``vec4f``, not ``quatf``.
            rot_wp = wp.zeros(shape=N, dtype=wp.vec4)
            if _disable_rot:
                is_pelvis = entry.canonical_name in ("hips", "pelvis")
                weight = max(entry.r_weight, 5.0) if is_pelvis else 0.0
            else:
                weight = entry.r_weight
            rotation_objectives.append(
                ik.IKObjectiveRotation(
                    link_index=entry.r_body_index,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=rot_wp,
                    weight=weight,
                )
            )

        objectives: list[ik.IKObjective] = [
            *position_objectives, *rotation_objectives,
        ]
        if self.config.joint_limit_weight > 0.0:
            objectives.append(
                ik.IKObjectiveJointLimit(
                    joint_limit_lower=model.joint_limit_lower,
                    joint_limit_upper=model.joint_limit_upper,
                    weight=self.config.joint_limit_weight,
                )
            )
        if self.config.smooth_joint_filter_weight > 0.0:
            from hhtools.retarget.newton_basic.ik_objectives import (
                IKSmoothJointFilter,
            )
            coord_masks = self._build_smooth_joint_filter_coord_masks(model)
            objectives.append(
                IKSmoothJointFilter(
                    joint_limit_lower=model.joint_limit_lower,
                    joint_limit_upper=model.joint_limit_upper,
                    weight=self.config.smooth_joint_filter_weight,
                    coord_masks=coord_masks,
                )
            )

        solver = ik.IKSolver(
            model=model,
            n_problems=N,
            objectives=objectives,
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        joint_q = wp.empty(
            shape=(N, model.joint_coord_count), dtype=wp.float32,
        )
        # ``model`` is the single-robot context (``num_envs=1``), so its
        # ``joint_q`` holds one rest pose; broadcast it to all N problems.
        rest_np = model.joint_q.numpy().reshape(-1)
        joint_q.assign(
            np.broadcast_to(rest_np, (N, rest_np.shape[0])).astype(
                np.float32, copy=True
            )
        )

        # Seed pelvis per env from frame-0 targets.
        pelvis_entry = next(
            (e for e in entries if e.canonical_name in ("hips", "pelvis")),
            None,
        )
        if pelvis_entry is not None:
            jq_np = joint_q.numpy().copy()
            pidx = entries.index(pelvis_entry)
            for env in range(N):
                jq_np[env, 0:3] = padded_targets[env, 0, pidx, 0:3]
                jq_np[env, 3:7] = padded_targets[env, 0, pidx, 3:7]
            joint_q.assign(jq_np)

        out = np.empty(
            (N, max_frames, model.joint_coord_count), dtype=np.float32,
        )
        solver.reset()

        ik_graph: object | None = None
        if max_frames > 0 and _cuda_graph_enabled(self.config):

            def _ik_batch_step_once() -> None:
                solver.step(
                    joint_q, joint_q, iterations=self.config.ik_iterations,
                )

            ik_graph = _try_capture_ik_step(_ik_batch_step_once)

        for frame in range(max_frames):
            for env in range(N):
                for i in range(M):
                    position_objectives[i].set_target_position(
                        env, wp.vec3(*padded_targets[env, frame, i, 0:3]),
                    )
                    rotation_objectives[i].set_target_rotation(
                        env, wp.quat(*padded_targets[env, frame, i, 3:7]),
                    )

            def _ik_batch_step_once() -> None:
                solver.step(
                    joint_q, joint_q, iterations=self.config.ik_iterations,
                )

            ik_graph = _run_ik_step(ik_graph=ik_graph, step_fn=_ik_batch_step_once)
            out[:, frame, :] = joint_q.numpy()

            if progress_callback is not None:
                try:
                    progress_callback(frame + 1, max_frames)
                except Exception:
                    pass

        return [out[env, :frame_counts[env], :] for env in range(N)]

    # ---- smooth-joint-filter mask assembly ------------------------------------

    def _build_smooth_joint_filter_coord_masks(self, model) -> np.ndarray | None:
        """Expand the per-link mask dict into a per-coord mask array.

        Resolution order:

        1. :attr:`PipelineConfig.smooth_joint_filter_masks` if set
           (callers take full control; pipeline does NOT fall back to
           the preset when an empty dict is provided — empty means "no
           per-link preferences, regularise every coord uniformly").
        2. ``robot.preset.smooth_joint_filter_masks`` otherwise.
        3. ``None`` (uniform weight=1.0 on every coord) when neither
           source names any link.

        Newton's ``IKSmoothJointFilter`` expects ``(joint_coord_count,)``
        float32 values in ``[0, 1]``.  We look up each link name in
        ``model.body_label``, read its driving joint's ``joint_q_start``
        / ``joint_dof_dim`` from the model, and fill those coord slots
        with the mask value.  Coords not covered by any named link keep
        the default ``0.0`` (pass-through — the smoother skips them).

        This mirrors soma-retargeter's
        ``newton_utils.create_joint_coord_masks`` so mask values copied
        from the upstream config apply with identical semantics.
        """

        masks = self.config.smooth_joint_filter_masks
        if masks is None:
            masks = dict(self.ctx.preset.smooth_joint_filter_masks or {})
        if not masks and self.ctx.preset.urdf_path is not None and self.ctx.preset.ik_map:
            from hhtools.robot.kinematics import infer_smooth_joint_filter_masks

            masks = infer_smooth_joint_filter_masks(
                self.ctx.preset.urdf_path, dict(self.ctx.preset.ik_map),
            )
        if not masks:
            return None  # uniform weight — let IKSmoothJointFilter default
        try:
            joint_q_start = model.joint_q_start.numpy()
            joint_dof_dim = model.joint_dof_dim.numpy()
        except AttributeError:
            return None
        body_to_idx = {
            name: i for i, name in enumerate(self.ctx.body_labels)
        }
        out = np.zeros(model.joint_coord_count, dtype=np.float32)
        missing: list[str] = []
        for link_name, mask_value in masks.items():
            idx = body_to_idx.get(link_name)
            if idx is None:
                missing.append(link_name)
                continue
            start = int(joint_q_start[idx])
            dim = int(joint_dof_dim[idx][1])
            if start < 0 or dim <= 0:
                # ``start == -1`` / ``dim == 0`` means this body has no
                # driving joint (fixed / welded) — skip silently.
                continue
            out[start : start + dim] = float(mask_value)
        if missing:
            _log.warning(
                "smooth_joint_filter_masks references unknown links on "
                "robot %r (ignored): %s. Available link names: %s...",
                self.ctx.preset.name, missing, list(body_to_idx)[:8],
            )
        return out

    # ---- rate limiter ---------------------------------------------------------

    def _rate_limit(
        self, *, root7: NDArray, dof: NDArray, framerate: float,
    ) -> NDArray:
        """Clip per-frame joint deltas to ``max_joint_velocity``.

        We only rate-limit the rotational part of the root (quat tail) and
        the actuated DOFs; position deltas are left alone because a walking
        subject legitimately moves a few metres per second and a position
        clip would gate real locomotion to zero.  This mirrors upstream
        behaviour.
        """
        dt = 1.0 / max(framerate, 1.0)
        max_dq = self.config.max_joint_velocity * dt
        max_dq_root = (
            self.config.max_root_angular_velocity * dt
            if self.config.max_root_angular_velocity > 0.0
            else max_dq
        )

        out = np.empty((root7.shape[0], root7.shape[1] + dof.shape[1]), dtype=np.float32)
        out[0, : root7.shape[1]] = root7[0]
        out[0, root7.shape[1] :] = dof[0]

        for frame in range(1, root7.shape[0]):
            prev = out[frame - 1]
            # Position columns (0..3): unclipped.
            out[frame, 0:3] = root7[frame, 0:3]
            # Root quat (3..7): clip delta.
            dq_root = np.clip(root7[frame, 3:7] - prev[3:7], -max_dq_root, max_dq_root)
            out[frame, 3:7] = prev[3:7] + dq_root
            # DOFs (7..): clip delta.
            dq = np.clip(dof[frame] - prev[7:], -max_dq, max_dq)
            out[frame, 7:] = prev[7:] + dq
        return out


# --------------------------------------------------------------------------- prewarm

_WARMED_ROBOT_PRESETS: set[str] = set()


def is_newton_ik_prewarmed(preset_name: str) -> bool:
    """True after :func:`prewarm_newton_ik_for_robot` succeeded for this preset."""
    return preset_name in _WARMED_ROBOT_PRESETS


def prewarm_newton_ik_for_robot(
    robot: URDFRobotModel,
    *,
    pipeline_config: PipelineConfig | None = None,
    ik_iterations: int = 1,
    force: bool = False,
) -> bool:
    """Run one dummy IK step so Warp kernels are compiled before the user clicks Retarget.

    The first :class:`NewtonBasicPipeline` solve in a fresh process triggers
    Warp JIT (often several seconds) plus an optional CUDA-graph capture.
    Calling this after the robot loads moves that cost off the critical path
    of the first retarget click.
    """
    preset_name = robot.preset.name
    if not force and preset_name in _WARMED_ROBOT_PRESETS:
        return False

    from hhtools.retarget.newton_basic.config import ScalerConfig

    stub_cfg = ScalerConfig(
        joint_scales={"hips": 1.0},
        root_joint="hips",
        model_height=1.3,
        human_height_assumption=1.7,
    )
    if pipeline_config is None:
        from hhtools.robot.retarget_profile import build_pipeline_config_for_preset

        pcfg = build_pipeline_config_for_preset(
            robot.preset,
            "lafan_bvh",
            ik_iterations=max(1, int(ik_iterations)),
        )
    else:
        pcfg = pipeline_config
    # Prewarm only JIT-compiles kernels — CUDA graph capture here races the
    # retarget worker on the same device and can poison subsequent launches.
    pcfg = replace(pcfg, ik_use_cuda_graph=False)
    try:
        pipe = NewtonBasicPipeline(
            robot,
            scaler_config=stub_cfg,
            pipeline_config=pcfg,
            human_height=1.7,
            configure_warp=True,
        )
        m = len(pipe.ik_mapping.entries)
        targets = np.zeros((1, m, 7), dtype=np.float32)
        targets[..., 6] = 1.0
        pipe._solve_ik_sequence(targets)
    except Exception as err:  # noqa: BLE001 — GPU optional in some builds
        _log.warning(
            "Newton IK prewarm failed for robot preset %r: %s",
            preset_name,
            err,
        )
        return False

    _WARMED_ROBOT_PRESETS.add(preset_name)
    _log.info("Newton IK prewarmed for robot preset %r", preset_name)
    return True
