"""Human-to-robot effector-target scaler (pure NumPy, CPU).

Stage-1 port of ``soma_retargeter.robotics.human_to_robot_scaler`` that
matches upstream output (within float32 tolerance) *without* any Newton or
Warp dependency.  It consumes hhtools' global-pose :class:`Motion`
representation (where ``positions`` / ``quaternions`` are already in world
space) instead of the upstream BVH-local-transform + Warp-FK path, so we save
an FK pass and stay on pure NumPy.

Attribution:
  Portions of the scaling formula and offset semantics are adapted from
  soma-retargeter (Apache-2.0).
  https://github.com/NVIDIA/SOMA-Retargeter
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.

The upstream scaler also ships a long chain of ``_enforce_*`` post-processors
(ground contact, foot planting, lateral separation, yaw bias, body-ground
clearance, correction smoothing …) on top of the raw scale.  We split those
into :class:`hhtools.retarget.newton_basic.feet_stabilizer.FeetStabilizer` so
the "scale effectors" concern and the "clean up targets before IK" concern
stay testable in isolation.

LAFAN1 / GMR foot handling matches ``soma_retargeter.assets.lafan1_foot_mod``:
when ``ScalerConfig.lafan_foot_mod_use_toe_orientation`` is ``True`` and the
source has LAFAN1-style ``*Foot``/``*Toe`` naming, mapped ``*Foot`` **rotation**
targets use the source toe joint's **global** quaternion (positions still come
from the ankle).  ``False`` / ``None`` leave the foot quaternion unchanged,
like upstream ``use_toe_orientation=False``.

Notation used throughout:
  - ``F``: number of frames
  - ``M``: number of mapped joints (= ``len(config.joint_scales)``)
  - ``7``: effector-transform layout  ``(x, y, z, qx, qy, qz, qw)``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hhtools.core.math import quaternion as Q
from hhtools.core.motion import Motion
from hhtools.retarget.newton_basic.config import ScalerConfig


__all__ = [
    "HumanToRobotScaler",
    "ScaledEffectors",
    "adapt_scaler_config_for_hierarchy",
    "resolve_source_bone_name",
]

# Source-bone synonym chains (SOMA / LAFAN / Mixamo naming variants).
_SOURCE_BONE_ALIASES: dict[str, tuple[str, ...]] = {
    "LeftToe": ("LeftToe", "LeftToeBase", "LeftToeEnd"),
    "RightToe": ("RightToe", "RightToeBase", "RightToeEnd"),
    "LeftToeBase": ("LeftToeBase", "LeftToe", "LeftToeEnd"),
    "RightToeBase": ("RightToeBase", "RightToe", "RightToeEnd"),
    "LeftToeEnd": ("LeftToeEnd", "LeftToeBase", "LeftToe"),
    "RightToeEnd": ("RightToeEnd", "RightToeBase", "RightToe"),
    "Hips": ("Hips", "Root", "pelvis"),
    "Root": ("Root", "Hips", "pelvis"),
}


def resolve_source_bone_name(
    bone_names: tuple[str, ...] | list[str],
    name: str,
) -> str | None:
    """Map a scaler bone name to the first matching name in ``bone_names``."""

    bones = {str(x) for x in bone_names}
    if name in bones:
        return name
    for cand in _SOURCE_BONE_ALIASES.get(name, (name,)):
        if cand in bones:
            return cand
    return None


def adapt_scaler_config_for_hierarchy(
    config: ScalerConfig,
    hierarchy,
) -> ScalerConfig:
    """Drop / rename scaler entries that use alternate source bone spellings.

    Upstream soma scalers list both ``LeftToe`` and ``LeftToeBase``; many
    SOMA BVHs only ship ``LeftToeBase`` + ``LeftToeEnd``.  Without remapping,
    :class:`HumanToRobotScaler` raises on the missing ``LeftToe`` entry even
    though the toe chain is present under a different name.
    """

    bones = tuple(hierarchy.bone_names)
    new_scales: dict[str, float] = {}
    new_offsets: dict[str, tuple[tuple[float, float, float],
                                 tuple[float, float, float, float]]] = {}
    seen: set[str] = set()

    for name in config.joint_names():
        resolved = resolve_source_bone_name(bones, name)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        new_scales[resolved] = float(config.joint_scales[name])
        entry = config.joint_offsets.get(name)
        if entry is not None:
            new_offsets[resolved] = entry

    if not new_scales:
        raise ValueError(
            "ScalerConfig has no joint_scales entries resolvable against the "
            f"source hierarchy ({list(bones)[:8]}…)"
        )

    root = resolve_source_bone_name(bones, config.root_joint)
    if root is None:
        raise KeyError(
            f"root_joint {config.root_joint!r} not found in source hierarchy"
        )

    return ScalerConfig(
        human_height_assumption=config.human_height_assumption,
        model_height=config.model_height,
        root_trajectory_scale=config.root_trajectory_scale,
        joint_scales=new_scales,
        joint_offsets=new_offsets,
        root_joint=root,
        scale_mode=config.scale_mode,
        up_axis=config.up_axis,
        scale_anchor=config.scale_anchor,
        root_z_offset=config.root_z_offset,
        robot_pelvis_height=config.robot_pelvis_height,
        source_body_quat=config.source_body_quat,
        lafan_foot_mod_use_toe_orientation=config.lafan_foot_mod_use_toe_orientation,
    )


def _is_lafan1_foot_skeleton(bone_names: tuple[str, ...] | list[str]) -> bool:
    """Heuristic: LAFAN1 / GMR naming with ankle + toe chains on both legs."""

    b = {str(x) for x in bone_names}
    required = (
        "LeftFoot",
        "RightFoot",
        "LeftUpLeg",
        "RightUpLeg",
    )
    if not all(x in b for x in required):
        return False
    return _toe_bone_name(bone_names, "Left") is not None and _toe_bone_name(
        bone_names, "Right"
    ) is not None


def _toe_bone_name(bone_names: tuple[str, ...] | list[str], side: str) -> str | None:
    b = {str(x) for x in bone_names}
    if side == "Left":
        for cand in ("LeftToe", "LeftToeEnd", "LeftToeBase"):
            if cand in b:
                return cand
    else:
        for cand in ("RightToe", "RightToeEnd", "RightToeBase"):
            if cand in b:
                return cand
    return None


@dataclass(frozen=True)
class ScaledEffectors:
    """Scaler output bundle.

    Attributes:
        joint_names: Ordered canonical joint names (``len == M``).  Matches
            ``config.joint_scales`` insertion order.
        transforms: ``(F, M, 7)`` array of ``(pos, quat_xyzw)`` per joint.
        raw_world_positions: ``(F, M, 3)`` pre-scale world positions for the
            same subset — useful for diagnostics and for the feet stabilizer
            which wants the unscaled trajectory as a "what the source motion
            actually did" reference.
    """

    joint_names: tuple[str, ...]
    transforms: NDArray
    raw_world_positions: NDArray


class HumanToRobotScaler:
    """Scale source human motions into robot-frame effector targets.

    Lifecycle:

        >>> scaler = HumanToRobotScaler(hierarchy, config, human_height=1.72)
        >>> result = scaler.apply(motion)
        >>> result.transforms.shape  # (F, M, 7)

    The scaler is *stateless* across ``apply`` calls — each input motion gets
    its own output; nothing about the scaler is mutated.  This makes it safe
    to share a single instance between the UI preview thread and the batch
    export worker.
    """

    def __init__(
        self,
        hierarchy,  # hhtools.core.Hierarchy — type-annotated w/o import cycle
        config: ScalerConfig,
        *,
        human_height: float,
        trajectory_human_height: float | None = None,
    ) -> None:
        if human_height <= 0.0:
            raise ValueError(f"human_height must be positive; got {human_height}")
        if not config.joint_scales:
            raise ValueError(
                "ScalerConfig.joint_scales is empty — nothing to map. "
                "Populate it with canonical human joint names."
            )

        config = adapt_scaler_config_for_hierarchy(config, hierarchy)
        self._config = config
        self._human_height = float(human_height)
        self._hierarchy = hierarchy

        self._up_axis_idx = _AXIS_TO_IDX[config.up_axis]

        # Pre-rotation: align source body heading with robot heading.
        sbq = np.asarray(config.source_body_quat, dtype=np.float32)
        if np.allclose(sbq, [0, 0, 0, 1], atol=1e-7):
            self._source_body_quat: NDArray | None = None
        else:
            self._source_body_quat = Q.normalize(sbq[None, :])[0]

        # Height ratio matches soma:
        #     joint_scales[name] = config.joint_scales[name] * human_height / human_height_assumption
        ratio = human_height / config.human_height_assumption
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise ValueError(
                f"Non-finite / non-positive scale ratio ({ratio}) derived from "
                f"human_height={human_height} and "
                f"human_height_assumption={config.human_height_assumption}"
            )

        joint_names = config.joint_names()
        self._mapped_joint_names: tuple[str, ...] = tuple(joint_names)
        self._mapped_indices = np.array(
            [hierarchy.index(name) for name in joint_names], dtype=np.int32
        )
        missing = [
            name for name, idx in zip(joint_names, self._mapped_indices.tolist())
            if idx < 0
        ]
        if missing:
            raise KeyError(
                "ScalerConfig references canonical joints not found in the "
                f"source hierarchy: {missing}. Available: {hierarchy.bone_names}"
            )

        self._scales = np.array(
            [config.joint_scales[name] * ratio for name in joint_names],
            dtype=np.float32,
        )

        # Offsets default to identity for joints not explicitly listed.
        offsets_t = np.zeros((len(joint_names), 3), dtype=np.float32)
        offsets_q = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (len(joint_names), 1),
        )
        for i, name in enumerate(joint_names):
            entry = config.joint_offsets.get(name)
            if entry is None:
                continue
            t, q = entry
            offsets_t[i] = np.asarray(t, dtype=np.float32)
            q_arr = np.asarray(q, dtype=np.float32)
            offsets_q[i] = Q.normalize(q_arr[None, :])[0]

        self._offsets_t = offsets_t
        self._offsets_q = offsets_q

        # Root index within the *mapped* subset.  We scale geocentrically
        # around this joint, matching soma's use of mapped_joint_indices[0].
        try:
            self._root_mapped_idx = joint_names.index(config.root_joint)
        except ValueError as err:
            raise KeyError(
                f"root_joint {config.root_joint!r} missing from joint_scales: "
                f"{joint_names}"
            ) from err

        # The builder computed root_z_offset assuming height_ratio == 1.
        # At runtime, root positions are multiplied by (scale_root * ratio)
        # instead of bare scale_root.  Adjust the vertical shift so the
        # scaled pelvis still lands at the robot's pelvis height.
        #
        #   effective = h_robot * (1 - ratio) + offset_orig * ratio
        #
        # Derived from: new_offset = h_robot - src_root_z * scale_root * ratio
        #   where src_root_z * scale_root = h_robot - offset_orig.
        if config.robot_pelvis_height is not None and ratio != 1.0:
            h_rbt = config.robot_pelvis_height
            self._effective_root_z_offset = float(
                h_rbt * (1.0 - ratio) + config.root_z_offset * ratio
            )
        else:
            self._effective_root_z_offset = config.root_z_offset

        # The root joint's anatomical scale is intentionally 1.0 in saved
        # calibrations, but mimic clips still need their *world trajectory*
        # scaled to the target robot size.  Apply that trajectory scale here
        # (before IK) so the yellow preview, IK targets, and final robot root
        # all agree; doing it only as a post-IK root correction makes the feet
        # solve against over-long human strides and then slide when the root is
        # shortened afterwards.
        #
        # ``model_height`` is the robot's mesh stature, so the default scale
        # assumes the robot's leg-to-stature proportions resemble a human's.
        # ``root_trajectory_scale`` overrides it for robots where they don't
        # (headless / short-torso models measure a small stature, which would
        # shorten the root stride well below the leg scale and force the
        # stance foot to skate).
        override = config.root_trajectory_scale
        model_h = float(config.model_height)
        if override is not None and np.isfinite(override) and override > 0.0:
            self._trajectory_scale = float(override)
        elif np.isfinite(model_h) and model_h > 0.01:
            traj_h = float(
                trajectory_human_height
                if trajectory_human_height is not None
                else self._human_height
            )
            self._trajectory_scale = model_h / max(1e-3, traj_h)
        else:
            self._trajectory_scale = 1.0

    # ---------------------------------------------------------------- properties

    @property
    def config(self) -> ScalerConfig:
        return self._config

    @property
    def human_height(self) -> float:
        return self._human_height

    @property
    def trajectory_scale(self) -> float:
        """World-space uniform scale applied to root trajectory (``model_h / h``)."""
        return float(self._trajectory_scale)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Ordered canonical joints produced by :meth:`apply`."""
        return self._mapped_joint_names

    @property
    def num_mapped_joints(self) -> int:
        return len(self._mapped_joint_names)

    @property
    def joint_scales(self) -> NDArray:
        """Per-joint scale factors *after* the height ratio correction."""
        return self._scales

    @property
    def joint_offsets_t(self) -> NDArray:
        return self._offsets_t

    @property
    def joint_offsets_q(self) -> NDArray:
        return self._offsets_q

    # --------------------------------------------------------------------- apply

    def apply(self, motion: Motion) -> ScaledEffectors:
        """Compute scaled effector transforms for every frame in ``motion``.

        The hierarchy baked into this scaler must match ``motion.hierarchy``
        (same ``bone_names`` order).  We assert this explicitly so a silent
        name-swap upstream (e.g. a simplified rig reorder) can't produce
        garbage targets.
        """
        if motion.hierarchy.bone_names != self._hierarchy.bone_names:
            raise ValueError(
                "motion.hierarchy does not match the hierarchy this scaler "
                "was constructed with; rebuild the scaler for this source."
            )

        # Gather the per-frame sub-trajectories for the mapped joints only.
        # positions: (F, M, 3), quats: (F, M, 4)
        pos_mapped = motion.positions[:, self._mapped_indices, :]
        quat_mapped = motion.quaternions[:, self._mapped_indices, :].copy()
        quat_mapped = self._inject_lafan_foot_mod_quaternions(
            motion, quat_mapped,
        )

        scaled_pos, scaled_quat = self._apply_np(pos_mapped, quat_mapped)

        transforms = np.concatenate(
            [scaled_pos.astype(np.float32, copy=False),
             scaled_quat.astype(np.float32, copy=False)],
            axis=-1,
        )  # (F, M, 7)

        return ScaledEffectors(
            joint_names=self._mapped_joint_names,
            transforms=transforms,
            raw_world_positions=pos_mapped.astype(np.float32, copy=False),
        )

    def apply_single(
        self, positions: NDArray, quaternions: NDArray
    ) -> ScaledEffectors:
        """Single-frame variant for the preview / REPL use case.

        Args:
            positions: ``(num_bones, 3)`` global positions for every bone in
                the hierarchy (not just the mapped subset).
            quaternions: ``(num_bones, 4)`` xyzw global quaternions.
        """
        p = np.asarray(positions, dtype=np.float32)
        q = np.asarray(quaternions, dtype=np.float32)
        if p.ndim == 2:
            p = p[None, ...]
            q = q[None, ...]

        sub_p = p[:, self._mapped_indices, :]
        sub_q = q[:, self._mapped_indices, :].copy()
        sub_q = self._inject_lafan_foot_mod_quaternions_from_full(p, q, sub_q)
        scaled_pos, scaled_quat = self._apply_np(sub_p, sub_q)
        transforms = np.concatenate([scaled_pos, scaled_quat], axis=-1)[0]
        return ScaledEffectors(
            joint_names=self._mapped_joint_names,
            transforms=transforms[None, ...].astype(np.float32, copy=False),
            raw_world_positions=sub_p.astype(np.float32, copy=False),
        )

    def scale_world_points_about_root(
        self, motion: Motion, points: NDArray[np.floating]
    ) -> NDArray[np.float32]:
        """Apply the same root-anchor scaling as :meth:`apply` to arbitrary world points.

        Used by interaction-mesh retargeting so rigid objects / terrain sample
        points shrink together with the human when ``human_height`` differs
        from the calibration assumption.

        Args:
            motion: Same motion passed to :meth:`apply` (defines per-frame root).
            points: ``(F, K, 3)`` or ``(K, 3)`` world positions. **Do not** pass a
                per-frame single point as ``(F, 3)`` — that is interpreted as
                ``K == F`` static markers; use ``(F, 1, 3)`` instead.
                ``F`` must match ``motion.num_frames`` when 3D.

        Returns:
            ``(F, K, 3)`` float32 scaled positions with the same global Z shift
            as mapped joint targets.
        """
        if motion.hierarchy.bone_names != self._hierarchy.bone_names:
            raise ValueError(
                "motion.hierarchy does not match the hierarchy this scaler "
                "was constructed with; rebuild the scaler for this source."
            )

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 2:
            pts = np.broadcast_to(pts[None, ...], (motion.num_frames, pts.shape[0], 3))
        if pts.ndim != 3 or pts.shape[2] != 3:
            raise ValueError(f"points must be (F, K, 3) or (K, 3); got {pts.shape}")
        if pts.shape[0] != motion.num_frames:
            raise ValueError(
                f"points frames {pts.shape[0]} != motion.num_frames {motion.num_frames}"
            )

        try:
            root_hier = self._hierarchy.index(self._config.root_joint)
        except KeyError as err:
            raise KeyError(
                f"root_joint {self._config.root_joint!r} not in hierarchy"
            ) from err

        root_world = motion.positions[:, root_hier, :].astype(np.float32, copy=False)
        F, K, _ = pts.shape

        # Body-frame alignment (same as _apply_np on mapped positions).
        sbq = self._source_body_quat
        if sbq is None:
            root_alg = root_world
            pts_alg = pts
        else:
            q_bc = np.broadcast_to(sbq[None, :], (F, 4))
            root_alg = Q.rotate(q_bc, root_world)
            n = F * K
            q_rep = np.broadcast_to(sbq[None, :], (n, 4))
            pts_alg = Q.rotate(q_rep, pts.reshape(n, 3)).reshape(F, K, 3)

        eff = self.apply(motion)
        try:
            j_root = self._mapped_joint_names.index(self._config.root_joint)
        except ValueError as err:
            raise KeyError(
                f"root_joint {self._config.root_joint!r} must appear in scaler "
                "joint_scales / mapped names"
            ) from err

        scaled_root = eff.transforms[:, j_root, :3].astype(np.float32, copy=False)
        scale_vec = self._build_scale_vec()
        s_root = scale_vec[self._root_mapped_idx : self._root_mapped_idx + 1, :].astype(
            np.float32, copy=False
        )

        disp = (pts_alg - root_alg[:, None, :]) * s_root[:, None, :]
        out = disp + scaled_root[:, None, :]

        if self._effective_root_z_offset != 0.0:
            out = out.copy()
            out[:, :, self._up_axis_idx] = (
                out[:, :, self._up_axis_idx] + np.float32(self._effective_root_z_offset)
            )
        return out.astype(np.float32, copy=False)

    # ---------------------------------------------------------------- internals

    def _lafan_foot_mod_enabled(self) -> bool:
        if self._config.lafan_foot_mod_use_toe_orientation is not True:
            return False
        return _is_lafan1_foot_skeleton(self._hierarchy.bone_names)

    def _inject_lafan_foot_mod_quaternions(
        self,
        motion: Motion,
        quat_mapped: NDArray,
    ) -> NDArray:
        """GMR / soma LAFAN1 foot mod: optional toe global quat on ankle row."""

        return self._inject_lafan_foot_mod_quaternions_from_full(
            motion.positions, motion.quaternions, quat_mapped,
        )

    def _inject_lafan_foot_mod_quaternions_from_full(
        self,
        p_full: NDArray,
        q_full: NDArray,
        quat_mapped: NDArray,
    ) -> NDArray:
        if not self._lafan_foot_mod_enabled():
            return quat_mapped
        bones = self._hierarchy.bone_names
        names = self._mapped_joint_names
        out = quat_mapped
        q_all = np.asarray(q_full, dtype=np.float32)
        for side, foot in (("Left", "LeftFoot"), ("Right", "RightFoot")):
            toe = _toe_bone_name(bones, side)
            if toe is None or foot not in names:
                continue
            try:
                col = names.index(foot)
                toe_i = bones.index(toe)
            except ValueError:
                continue
            out[:, col, :] = Q.normalize(q_all[:, toe_i, :].astype(np.float32, copy=False))
        return out

    def _apply_np(self, pos_mapped: NDArray, quat_mapped: NDArray):
        """Core numpy kernel; outputs (F, M, 3), (F, M, 4).

        Two position-scaling semantics are supported, selected by
        ``ScalerConfig.scale_anchor``.  Both reduce to the same answer
        at rest when the matching ``t_offset`` was computed from the
        builder; they differ on motion frames.

        - ``"root"`` (soma-compatible, *the* correct mode, default for
          every config emitted by
          :func:`hhtools.retarget.calibration.calibration.build_scaler_config_soma_style`):
          each non-root joint's *displacement from the source root* is
          scaled by its own scale[j], then lifted by the *root's* scaled
          world position.  This matches soma-retargeter's
          ``wp_compute_scaled_effectors`` kernel line-for-line::

              scaled_root_t = root_t * scale[root]           # once per frame
              geo_scaled    = (p - root_t) * scale[j]        # per joint
              t_out[j]      = geo_scaled + scaled_root_t + rotate(q_out, t_offset[j])

          In particular the root position is scaled *uniformly* (by the
          root's own scale) for every non-root joint's target — the
          semantics the upstream pipeline was calibrated against.
        - ``"origin"`` (legacy, geometrically biased on motion frames):
          every joint's *world* position is scaled by its own scale[j],
          so ``t_out[j] = p_src[j] * scale[j] + rotate(q_out[j],
          t_offset[j])``.  This matches ``"root"`` at rest (by
          construction of the builder's ``t_offset``) but picks up a
          per-frame bias ``(scale[j] - scale[root]) · (p_src_root[t] -
          Δq ⊙ p_src_root_rest)`` that amplifies shoulder/hip motion on
          anything that isn't a static T-pose.  Kept for backward
          compatibility with previously-saved configs; new configs
          should always emit ``"root"``.
        """
        if pos_mapped.shape[:2] != quat_mapped.shape[:2]:
            raise ValueError(
                f"pos_mapped / quat_mapped leading shape mismatch: "
                f"{pos_mapped.shape} vs {quat_mapped.shape}"
            )

        # --- Body-frame heading alignment (yaw pre-rotation) ----------
        # When the source skeleton and robot disagree on which world
        # direction is "forward" (e.g. source -Y vs robot +X after a
        # Y-up→Z-up BVH import), source_body_quat corrects the heading
        # so displacements fed into the scale/offset math point the
        # same way the robot expects.  Applied to BOTH positions (to
        # fix translation heading) and quaternions (so q_offset still
        # lands on q_rbt at rest).
        pos_mapped, quat_mapped = self._align_body_frame(
            pos_mapped, quat_mapped
        )

        root_t = pos_mapped[:, self._root_mapped_idx : self._root_mapped_idx + 1, :]
        # (F, 1, 3)

        scale_vec = self._build_scale_vec()  # (M, 3)

        if self._config.scale_anchor == "root":
            # Anchor every joint at the root's scaled position (shared).
            root_scale = scale_vec[self._root_mapped_idx : self._root_mapped_idx + 1, :]  # (1, 3)
            scaled_root_t = np.broadcast_to(
                root_t * root_scale[None, :, :], (root_t.shape[0], self.num_mapped_joints, 3)
            )
        else:
            # Legacy / soma-compatible: each joint's scale multiplies the root
            # position too — expanded below as ``geocentric_scaled +
            # scaled_root_t`` so both branches share the rotation path.
            scaled_root_t = root_t * scale_vec[None, :, :]

        # geocentric_scaled = (p - root_t) * per_joint_scale  ; (F, M, 3)
        geocentric_scaled = (pos_mapped - root_t) * scale_vec[None, :, :]

        # q_out = q * offset.q  ; (F, M, 4)
        F = quat_mapped.shape[0]
        offsets_q = np.broadcast_to(
            self._offsets_q[None, :, :], (F, self.num_mapped_joints, 4)
        )
        q_out = Q.multiply(quat_mapped.reshape(-1, 4), offsets_q.reshape(-1, 4))
        q_out = Q.normalize(q_out).reshape(F, self.num_mapped_joints, 4)
        q_out = Q.ensure_continuous(q_out)

        # rotated_offset = quat_rotate(q_out, offset.p) ; (F, M, 3)
        offsets_t = np.broadcast_to(
            self._offsets_t[None, :, :], (F, self.num_mapped_joints, 3)
        )
        rotated_offset = Q.rotate(
            q_out.reshape(-1, 4), offsets_t.reshape(-1, 3)
        ).reshape(F, self.num_mapped_joints, 3)

        t_out = geocentric_scaled + scaled_root_t + rotated_offset

        # Scale global root trajectory displacement about frame 0 so shorter
        # robots take proportionally shorter world-space strides.  The local
        # body pose (joint displacements from the root) is already handled by
        # per-joint scales above; this is only the clip-level translation.
        traj_scale = float(self._trajectory_scale)
        current_root_scale = scale_vec[
            self._root_mapped_idx : self._root_mapped_idx + 1, :
        ].astype(np.float32, copy=False)
        desired_root_scale = np.float32(traj_scale)
        traj_correction = desired_root_scale - current_root_scale
        if np.any(np.abs(traj_correction) > np.float32(1e-6)):
            root_disp = root_t - root_t[0:1, :, :]
            t_out = t_out + root_disp * traj_correction[None, :, :]

        # Apply the calibration-derived vertical shift to *every* mapped
        # joint target (not just the root).  Shifting only the root would
        # leave child-limb targets at their original world Z while the
        # root moved — the IK would then distort the robot to straddle
        # both elevations.  Rigid body-wide shift keeps relative limb
        # geometry intact and just slides the whole targeted pose up or
        # down until the feet land on the ground.
        if self._effective_root_z_offset != 0.0:
            t_out = t_out.copy()
            t_out[:, :, self._up_axis_idx] = (
                t_out[:, :, self._up_axis_idx] + np.float32(self._effective_root_z_offset)
            )

        return t_out.astype(np.float32, copy=False), q_out.astype(np.float32, copy=False)

    def _align_body_frame(
        self, pos: NDArray, quat: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Pre-rotate source positions/quaternions by ``source_body_quat``.

        Returns the inputs unchanged when source_body_quat is identity.
        """
        sbq = self._source_body_quat
        if sbq is None:
            return pos, quat
        F, M = pos.shape[:2]
        n = F * M
        q_bc = np.broadcast_to(sbq[None, :], (n, 4))
        pos_out = Q.rotate(q_bc, pos.reshape(n, 3)).reshape(F, M, 3)
        quat_out = Q.multiply(q_bc, quat.reshape(n, 4)).reshape(F, M, 4)
        return pos_out.astype(np.float32, copy=False), quat_out.astype(np.float32, copy=False)

    def _build_scale_vec(self) -> NDArray:
        """(M, 3) per-joint scale broadcast to xyz depending on scale_mode."""
        scales = self._scales.astype(np.float32, copy=False)
        if self._config.scale_mode == "uniform":
            return np.broadcast_to(scales[:, None], (scales.size, 3)).copy()
        # height mode — only the up-axis is scaled, the other two are 1.0
        out = np.ones((scales.size, 3), dtype=np.float32)
        out[:, self._up_axis_idx] = scales
        return out


_AXIS_TO_IDX = {"X": 0, "Y": 1, "Z": 2}
