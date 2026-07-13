# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Interaction-mesh retarget pipeline: scaler + Laplacian MPC/SQP (MuJoCo).

Architecture follows holosoma's InteractionMeshRetargeter:

1. **Uniform scaling** — all human joints are scaled by a single
   ``robot_height / human_height`` ratio, preserving natural human
   proportions rather than distorting per-limb like the Newton scaler.
2. **Mapped joints only** — the Laplacian target mesh uses exactly the
   IK-mapped joints (e.g. 14 for RP1), ensuring a 1:1 correspondence
   with robot bodies.  Holosoma's ``JOINTS_MAPPING`` serves the same
   role as ``preset.ik_map``.
3. **Explicit body correspondence** — robot body names for the SQP are
   derived from ``ik_map`` values (link names), not from MuJoCo body
   traversal order.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hhtools.core.math import quaternion as Q
from hhtools.core.grounding import (
    human_source_floor_z_world,
    terrain_heightfield_z_offset_world,
    use_split_terrain_grounding,
)
from hhtools.core.motion import Motion
from hhtools.retarget.calibration import (
    load_calibration,
)
from hhtools.robot.retarget_profile import build_scaler_config_for_robot
from hhtools.retarget.interaction_mesh.config import InteractionMeshPipelineConfig
from hhtools.retarget.interaction_mesh.motion_bridge import (
    ScaledMotionScene,
    scale_motion_and_objects,
)
from hhtools.retarget.interaction_mesh.mujoco_jacobians import pack_joint_q_csv
from hhtools.retarget.interaction_mesh.mujoco_scene import MujocoScene, require_mujoco_model
from hhtools.retarget.newton_basic.human_aliases import auto_source_to_canonical
from hhtools.retarget.newton_basic.scaler import HumanToRobotScaler
from hhtools.robot.loader import URDFRobotModel

_log = logging.getLogger(__name__)

# Upper bound on frames sampled for the post-MPC alignment-quality diagnostic
# (logging only).  Bounds a second full-clip ``mj_forward`` pass on long
# terrain clips without changing the exported trajectory.
_ALIGNMENT_DIAG_MAX_FRAMES = 256


@dataclass
class InteractionMeshPipeline:
    """Laplacian interaction-mesh backend: precompute + RTI MPC + SQP (no cvxpy)."""

    robot: URDFRobotModel
    scaler: HumanToRobotScaler
    cfg: InteractionMeshPipelineConfig

    @classmethod
    def from_calibration(
        cls,
        robot: URDFRobotModel,
        motion: Motion,
        calibration_path: str,
        *,
        human_height: float = 1.7,
        cfg: InteractionMeshPipelineConfig | None = None,
    ) -> InteractionMeshPipeline:
        if robot.preset.urdf_path is not None and robot.preset.ik_map:
            from hhtools.robot.kinematics import require_valid_ik_map

            require_valid_ik_map(
                robot.preset.urdf_path,
                dict(robot.preset.ik_map),
                robot_name=robot.preset.name,
            )
        cal = load_calibration(calibration_path)
        scaler_cfg = build_scaler_config_for_robot(
            cal, robot, motion, human_height=human_height,
        )
        scaler = HumanToRobotScaler(
            motion.hierarchy, scaler_cfg, human_height=human_height,
        )
        return cls(robot=robot, scaler=scaler, cfg=cfg or InteractionMeshPipelineConfig())

    def scale_scene(self, motion: Motion) -> ScaledMotionScene:
        """Human + ``motion.objects`` in the same scaled world frame as mimic."""
        return scale_motion_and_objects(motion, self.scaler, motion.objects)

    def _align_root_to_source_heading(self, joint_q: np.ndarray) -> np.ndarray:
        """Identity pass-through — kept for call-site compatibility.

        Historically this method counter-rotated the floating base by
        ``conj(source_body_quat)`` to undo a matching ``sbq`` pre-rotation
        applied to the SQP's source-pose targets in
        :meth:`_build_scaled_source_pose`.  After the heightfield-aware
        rewrite, both the source positions **and** the terrain
        heightfield stay in the source world frame (a heightfield is an
        axis-aligned grid that cannot be rotated by an arbitrary
        ``sbq`` yaw without lossy resampling).  The SQP therefore
        already solves for ``joint_q`` in the source frame, and a
        post-rotation here would mis-rotate the output by ``sbq``
        again — exactly the symptom observed in ``parc_ms``
        (sbq ≈ +134°), where the retargeted base trajectory ended up
        rotated ~134° from the yellow scaled-skeleton trajectory.

        Removing the rotation completely would break callers that
        depend on the method existing; we keep the no-op so the call
        site stays semantically clear.
        """
        return joint_q

    def _clamp_solved_foot_lateral(
        self,
        joint_q: np.ndarray,
        *,
        framerate: float = 30.0,
    ) -> np.ndarray:
        """Post-MPC hip abduction when solved foot meshes overlap / ankles cross.

        Mirrors :meth:`NewtonBasicPipeline._clamp_solved_foot_lateral`.  MPC's
        position cost cannot disambiguate left/right laterality when ankle
        targets nearly coincide (``hip_yaw`` is nearly in the nullspace), so
        the trajectory can settle into an X-legged pose; this mesh-gated
        clamp abducts the hips until the feet clear.

        Skips inverted frames (root +Z pointing downward).  On upright frames
        the correction is **rate-limited** so opening the upright gate cannot
        dump the full abduction budget in a single stand-up frame.
        """
        min_clearance = float(self.cfg.min_foot_clearance_m)
        if min_clearance <= 0.0 or joint_q.size == 0:
            return joint_q

        from hhtools.core.math import quaternion as Q
        from hhtools.robot.foot_geometry import clamp_joint_q_foot_lateral_clearance

        dof_names = self.robot.dof_names()
        out = joint_q.astype(np.float32, copy=True)
        n_fixed = 0
        max_abd = float(self.cfg.foot_lateral_max_abduction_rad)
        n_iter = int(self.cfg.foot_lateral_max_iterations)
        dt = 1.0 / max(float(framerate), 1.0)
        max_step = max(0.0, float(self.cfg.foot_lateral_max_step_rad_s) * dt)
        for f in range(out.shape[0]):
            # Root up-vector z: skip while rolling / inverted.
            if out.shape[1] >= 7:
                up = Q.to_matrix(out[f, 3:7][None, :])[0] @ np.array(
                    [0.0, 0.0, 1.0], dtype=np.float64
                )
                if float(up[2]) < 0.35:
                    continue
            before = out[f].copy()
            desired = clamp_joint_q_foot_lateral_clearance(
                self.robot,
                out[f],
                dof_names,
                root_coord_count=7,
                min_clearance_m=min_clearance,
                max_abduction_rad=max_abd,
                max_iterations=n_iter,
            )
            if max_step > 0.0:
                delta = np.asarray(desired, dtype=np.float32) - before
                # Only the actuated part may move (root stays).
                if before.size > 7:
                    delta[:7] = 0.0
                    delta[7:] = np.clip(delta[7:], -max_step, max_step)
                out[f] = before + delta
            else:
                out[f] = desired
            if not np.allclose(before, out[f], atol=1e-6):
                n_fixed += 1
        if n_fixed:
            _log.info(
                "foot-lateral clamp: adjusted hip abduction on %d/%d frames "
                "(min_clearance=%.3fm, max_abd=%.2frad, max_step=%.3frad)",
                n_fixed,
                out.shape[0],
                min_clearance,
                max_abd,
                max_step,
            )
        return out

    def _stabilize_joints_after_mpc(
        self,
        joint_q: np.ndarray,
        motion: Motion,
        *,
        framerate: float,
    ) -> np.ndarray:
        """Optional post-MPC rate-limit (off by default).

        Prefer ``lock_root_orientation_to_source`` in the SQP — that removes
        roll→stand orientation basin hops at the solver.  This path only
        papers over residual jumps; enable via
        ``stabilize_joints_after_mpc=True`` for emergency clips.
        """
        if joint_q.ndim != 2 or joint_q.shape[0] < 2 or joint_q.shape[1] < 7:
            return joint_q

        from hhtools.core.math import quaternion as Q

        dt = 1.0 / max(float(framerate), 1.0)
        floor_root = float(self.cfg.max_root_angular_velocity)
        mult = float(self.cfg.root_angular_velocity_source_multiplier)
        max_yaw = float(self.cfg.max_yaw_joint_velocity) * dt
        max_leg = float(self.cfg.max_leg_joint_velocity) * dt
        teleport = float(np.deg2rad(self.cfg.leg_joint_teleport_deg))

        out = joint_q.astype(np.float32, copy=True)
        out[:, 3:7] = Q.ensure_continuous(out[:, 3:7])
        raw = joint_q.astype(np.float32, copy=True)

        src_speed: np.ndarray | None = None
        try:
            pelvis_i = self._pelvis_source_index(motion)
            if pelvis_i >= 0 and motion.quaternions.shape[0] == out.shape[0]:
                qs = Q.ensure_continuous(
                    Q.normalize(np.asarray(motion.quaternions[:, pelvis_i], dtype=np.float32))
                )
                dots = np.abs(np.sum(qs[1:] * qs[:-1], axis=1)).clip(0.0, 1.0)
                src_speed = (2.0 * np.arccos(dots)) / dt
        except Exception:
            src_speed = None

        dof_names = self.robot.dof_names()
        _leg_keys = ("hip", "knee", "ankle", "thigh", "calf", "leg", "shank")
        _yaw_keys = ("yaw", "waist")
        yaw_cols: list[int] = []
        leg_cols: list[int] = []
        for i, n in enumerate(dof_names):
            col = 7 + i
            if col >= out.shape[1]:
                break
            nl = str(n).lower()
            is_yaw = any(k in nl for k in _yaw_keys)
            is_leg = any(k in nl for k in _leg_keys)
            if is_yaw:
                yaw_cols.append(col)
            elif is_leg:
                leg_cols.append(col)

        n_root_clamped = 0
        n_hinge_clamped = 0
        n_hinge_held = 0

        def _limit_hinge(f: int, c: int, max_step: float) -> None:
            nonlocal n_hinge_clamped, n_hinge_held
            if max_step <= 0.0:
                return
            prev = float(out[f - 1, c])
            target = float(raw[f, c])
            d = target - prev
            d = (d + np.pi) % (2.0 * np.pi) - np.pi
            if abs(d) > teleport:
                # Isolated spike (next frame returns near prev) → hold.
                # Sustained stand-up pose change → rate-limit toward target.
                nxt = float(raw[min(f + 1, out.shape[0] - 1), c])
                d_next = nxt - prev
                d_next = (d_next + np.pi) % (2.0 * np.pi) - np.pi
                if abs(d_next) < 0.5 * teleport:
                    out[f, c] = prev
                    n_hinge_held += 1
                    return
            if abs(d) > max_step:
                out[f, c] = prev + float(np.clip(d, -max_step, max_step))
                n_hinge_clamped += 1
            else:
                out[f, c] = prev + d

        for f in range(1, out.shape[0]):
            prev_q = out[f - 1, 3:7]
            cur_q = Q.normalize(out[f, 3:7])
            cap = floor_root
            if src_speed is not None and mult > 0.0:
                cap = max(floor_root, float(src_speed[f - 1]) * mult)
            max_ang = cap * dt
            dot = float(np.dot(prev_q, cur_q))
            target_q = -cur_q if dot < 0.0 else cur_q
            ang = 2.0 * float(np.arccos(min(abs(dot), 1.0)))
            if max_ang > 0.0 and ang > max_ang and ang > 1e-6:
                out[f, 3:7] = Q.slerp(prev_q, target_q, max_ang / ang)
                n_root_clamped += 1
            else:
                out[f, 3:7] = Q.normalize(target_q)

            for c in yaw_cols:
                _limit_hinge(f, c, max_yaw if max_yaw > 0.0 else max_leg)
            for c in leg_cols:
                _limit_hinge(f, c, max_leg)

        if n_root_clamped or n_hinge_clamped or n_hinge_held:
            _log.info(
                "joint stabilize: root_clamp=%d hinge_clamp=%d hinge_hold=%d "
                "(root_floor=%.2f rad/s, leg=%.2f rad/s, teleport=%.0f°)",
                n_root_clamped,
                n_hinge_clamped,
                n_hinge_held,
                floor_root,
                float(self.cfg.max_leg_joint_velocity),
                float(self.cfg.leg_joint_teleport_deg),
            )
        return out

    # Back-compat alias for anything that imported the old name.
    def _stabilize_yaw_after_mpc(self, joint_q, motion, *, framerate: float):
        return self._stabilize_joints_after_mpc(
            joint_q, motion, framerate=framerate,
        )

    def _enforce_mapped_ankle_laterality(
        self,
        mapped_pos: np.ndarray,
        robot_points: list,
    ) -> np.ndarray:
        """Keep left/right ankle (and foot) targets on the correct sides.

        During ground rolls the source ankles can nearly coincide for a few
        frames; position-only MPC then freely swaps laterality and the
        subsequent stand-up locks into an X-leg.  When hip landmarks imply a
        clear left/right axis **and the body is upright**, push ankle/foot
        targets apart along that axis to at least
        ``min_target_foot_separation_m``.

        Inverted / rolling frames are skipped: the hip lateral axis flips in
        the XY projection while upside-down, and enforcing laterality there
        teaches the QP a 180° waist-yaw jump on stand-up.
        """
        min_sep = float(self.cfg.min_target_foot_separation_m)
        if min_sep <= 0.0 or mapped_pos.size == 0:
            return mapped_pos

        def _find(*keys: str) -> int | None:
            for key in keys:
                for i, pt in enumerate(robot_points):
                    sem = str(getattr(pt, "semantic", "")).lower()
                    if key == sem or sem.endswith(key) or key in sem.split(":"):
                        return i
            for key in keys:
                for i, pt in enumerate(robot_points):
                    if key in str(getattr(pt, "semantic", "")).lower():
                        return i
            return None

        li = _find("left_ankle", "left_foot")
        ri = _find("right_ankle", "right_foot")
        l_hip = _find("left_hip", "left_thigh")
        r_hip = _find("right_hip", "right_thigh")
        pelvis = _find("pelvis", "hips", "torso")
        foot_pairs: list[tuple[int, int]] = []
        if li is not None and ri is not None:
            foot_pairs.append((li, ri))
        l_foot = _find("left_foot")
        r_foot = _find("right_foot")
        if (
            l_foot is not None
            and r_foot is not None
            and (l_foot, r_foot) not in foot_pairs
            and (l_foot != li or r_foot != ri)
        ):
            foot_pairs.append((l_foot, r_foot))
        if not foot_pairs:
            return mapped_pos

        upright_min = float(self.cfg.laterality_upright_min_m)
        out = np.asarray(mapped_pos, dtype=np.float32, copy=True)
        prev_axis: np.ndarray | None = None
        for f in range(out.shape[0]):
            # Upright gate: hip/pelvis clearly above the feet.
            foot_zs = [float(out[f, a, 2]) for a, _ in foot_pairs] + [
                float(out[f, b, 2]) for _, b in foot_pairs
            ]
            hip_zs: list[float] = []
            if l_hip is not None:
                hip_zs.append(float(out[f, l_hip, 2]))
            if r_hip is not None:
                hip_zs.append(float(out[f, r_hip, 2]))
            if pelvis is not None:
                hip_zs.append(float(out[f, pelvis, 2]))
            if hip_zs and foot_zs:
                if min(hip_zs) - min(foot_zs) < upright_min:
                    prev_axis = None
                    continue

            if l_hip is not None and r_hip is not None:
                axis = out[f, l_hip, :2] - out[f, r_hip, :2]
            else:
                axis = out[f, foot_pairs[0][0], :2] - out[f, foot_pairs[0][1], :2]
            n = float(np.linalg.norm(axis))
            if n < 1e-4:
                continue
            axis = axis / n
            # Keep the lateral axis in one hemisphere across upright frames.
            if prev_axis is not None and float(np.dot(axis, prev_axis)) < 0.0:
                axis = -axis
            prev_axis = axis.copy()

            for li_i, ri_i in foot_pairs:
                mid = 0.5 * (out[f, li_i, :2] + out[f, ri_i, :2])
                sep = float(np.dot(out[f, li_i, :2] - out[f, ri_i, :2], axis))
                if sep >= min_sep:
                    continue
                half = 0.5 * min_sep
                out[f, li_i, 0] = mid[0] + half * axis[0]
                out[f, li_i, 1] = mid[1] + half * axis[1]
                out[f, ri_i, 0] = mid[0] - half * axis[0]
                out[f, ri_i, 1] = mid[1] - half * axis[1]
        return out

    def _resolve_mapped_joints(
        self, motion: Motion,
    ) -> tuple[list[int], list[str], list[str]]:
        """Resolve IK-mapped source bone indices and corresponding robot link names.

        Returns ``(source_indices, robot_link_names)`` where each entry i
        maps human joint ``source_indices[i]`` to robot body
        ``robot_link_names[i]``.  This gives the 1:1 vertex correspondence
        the Laplacian optimization requires.
        """
        ik_map = self.robot.preset.ik_map
        src2can = auto_source_to_canonical(motion.hierarchy.bone_names)
        can2src: dict[str, str] = {}
        for src_name, can_name in src2can.items():
            can2src.setdefault(can_name, src_name)

        bone_index = {n: i for i, n in enumerate(motion.hierarchy.bone_names)}
        source_indices: list[int] = []
        robot_links: list[str] = []
        canonical_names: list[str] = []

        for canonical, link_name in ik_map.items():
            src_name = can2src.get(canonical)
            if src_name is None or src_name not in bone_index:
                continue
            source_indices.append(bone_index[src_name])
            robot_links.append(link_name)
            canonical_names.append(canonical)

        if not source_indices:
            raise ValueError(
                "No IK-mapped joints could be resolved between the source "
                f"skeleton ({len(motion.hierarchy.bone_names)} bones) and "
                f"robot ik_map ({len(ik_map)} entries). Check calibration "
                "and source→canonical alias mapping."
            )
        _log.info(
            "Interaction mesh: %d mapped joints for Laplacian "
            "(source skeleton has %d bones, ik_map has %d entries)",
            len(source_indices), len(motion.hierarchy.bone_names), len(ik_map),
        )
        return source_indices, robot_links, canonical_names

    @staticmethod
    def _validate_body_names_against_mujoco(
        mj_model, robot_links: list[str], robot: URDFRobotModel,
    ) -> list[str]:
        """Remap ik_map URDF link names to bodies present in the compiled model.

        MuJoCo drops the URDF root (→ ``floating_base`` after freejoint injection)
        and merges fixed-joint children (``torso_link``, sensor shells,
        ``*_end_effector_link``, …).  Walk the URDF parent chain so any humanoid
        works without hand-editing ``ik_map`` per robot naming convention.
        """
        from hhtools.robot.kinematics import resolve_urdf_links_to_mujoco_bodies

        urdf_path = robot.preset.urdf_path
        if urdf_path is None or not urdf_path.is_file():
            raise ValueError(
                f"robot {robot.preset.name!r} has no URDF for MuJoCo body resolution"
            )
        try:
            return resolve_urdf_links_to_mujoco_bodies(
                urdf_path,
                mj_model,
                robot_links,
                urdf_base=robot.base_link,
            )
        except ValueError as err:
            raise ValueError(
                f"{err}  Check ik_map / URDF topology for robot "
                f"{robot.preset.name!r}."
            ) from err

    def _build_mapped_scaled_positions(
        self,
        motion: Motion,
        source_indices: list[int],
    ) -> tuple[NDArray[np.float32], float, float]:
        """Extract mapped joint positions and apply uniform scaling (holosoma-style).

        Steps (matching holosoma's ``preprocess_human_joints``):
        1. Pre-rotate by ``source_body_quat`` to align heading.
        2. Floor-normalise (subtract ``z_min``): ``z_min`` is the foot-floor from
           :func:`~hhtools.core.grounding.human_source_floor_z_world` (ankle / foot
           hubs, excluding ``*FootMod``).  Legacy ``min(all joint Z)`` lifted clips
           whose toe joints dip below the foot floor — e.g. holosoma ``parkour_*`` —
           ~5 cm above the yellow scaled overlay and the viewer grid.  Heightfield
           ``z_offset`` follows :func:`~hhtools.core.grounding.terrain_heightfield_z_offset_world`
           (split datasets can differ from ``z_min``; see
           :func:`~hhtools.core.grounding.use_split_terrain_grounding`).  Laplacian
           terrain anchors add ``(z_min - z_terrain) · smpl_scale`` to Z only in the
           split case.
        3. Uniform scale by ``robot_height / human_height``.

        Returns ``(mapped_positions, z_min, smpl_scale)`` where
        ``mapped_positions`` is ``(F, K, 3)`` float32.
        """
        scaled_pos, _scaled_quat, z_min, smpl_scale = self._build_scaled_source_pose(motion)
        idx = np.array(source_indices, dtype=np.int32)
        mapped = scaled_pos[:, idx, :].copy()  # (F, K, 3)
        return mapped, z_min, smpl_scale

    def _build_scaled_source_pose(
        self,
        motion: Motion,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float, float]:
        """Source positions/quaternions in the solver frame.

        Applies a uniform 3-axis scale
        ``smpl_scale = robot_height / human_height`` plus a Z ground-snap.
        For all clips ``z_min`` is the foot-floor from
        :func:`~hhtools.core.grounding.human_source_floor_z_world`.  The heightfield's
        ``TerrainHeightfield.scaled(..., z_offset=...)`` uses
        :func:`~hhtools.core.grounding.terrain_heightfield_z_offset_world`; Laplacian
        terrain anchors add ``(z_min - z_terrain) · smpl_scale`` to Z only when that
        offset differs from ``z_min`` (split case).

        We deliberately **do not** apply ``source_body_quat`` here.
        ``source_body_quat`` is a yaw rotation that aligns the
        source actor's "forward" with the robot's URDF-declared
        forward axis (e.g. SMPL faces -Y, RoboParty RP1 faces +X).
        For an interaction-mesh / heightfield pipeline a yaw on the
        positions would also need to be applied to the terrain
        heightfield — but the heightfield is an axis-aligned
        cell grid that cannot be rotated by an arbitrary yaw
        without a lossy resample.  Rotating only the human
        positions silently creates a frame mismatch where the
        Laplacian / position-cost pulls the robot one way and the
        hard non-penetration constraints (still in the unrotated
        terrain frame) pull it the other way; observed in
        ``parc_ms`` (sbq ≈ 134°) as the robot tracking the source
        with **inverted X** because the heightfield is in the
        unrotated frame and the OSQP solver kept the robot above
        the terrain rather than where the rotated source target
        was.

        The fix is to leave both human and terrain in the source
        world frame and rely on the robot's free joint quaternion
        to absorb whatever yaw the actor has — the SQP can rotate
        the floating base freely and the Laplacian does not care
        about absolute heading.
        """
        all_pos = np.asarray(motion.positions, dtype=np.float32).copy()
        all_quat = np.asarray(motion.quaternions, dtype=np.float32).copy()

        z_min = float(human_source_floor_z_world(motion))
        robot_height = float(self.scaler.config.model_height)
        smpl_scale = float(self.scaler.trajectory_scale)
        human_height = float(self.scaler.human_height)
        if human_height < 0.1:
            human_height = 1.7

        all_pos[:, :, 2] -= z_min
        all_pos *= smpl_scale

        _log.debug(
            "Source pose scale: × %.4f, Z shifted by %.4f "
            "(robot_h=%.3f / human_h=%.3f); source_body_quat NOT applied — "
            "kept in source world frame to match heightfield",
            smpl_scale, z_min, robot_height, human_height,
        )
        return all_pos.astype(np.float32), all_quat.astype(np.float32), z_min, smpl_scale

    def precompute_laplacian_targets(
        self,
        motion: Motion,
        *,
        mj_model=None,
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        """Per-frame Laplacian targets using only IK-mapped joints.

        Returns ``(targets, robot_links, z_min, smpl_scale)`` so callers
        can reuse the scaling parameters (e.g. for terrain height maps).
        """
        from hhtools.retarget.interaction_mesh.contact_points import build_contact_mpc_points
        from hhtools.retarget.interaction_mesh.mpc_loop import (
            RobotMpcPoint,
            precompute_target_laplacians,
        )

        source_indices, robot_links, canonical_names = self._resolve_mapped_joints(motion)
        if mj_model is not None:
            robot_links = self._validate_body_names_against_mujoco(
                mj_model, robot_links, self.robot,
            )
            # On grasping clips (interaction objects present) prioritise the
            # wrist / hand-tip effector so a hand-less robot's wrist collision
            # tip actually reaches the contact — the user's "首先要尽量接触到".
            grasping = bool(getattr(motion, "objects", None))
            hand_w = float(self.cfg.hand_contact_weight) if grasping else 1.0
            robot_points = build_contact_mpc_points(
                mj_model, robot_links, source_indices, canonical_names,
                hand_effector_weight=hand_w,
            )
        else:
            robot_points = [
                RobotMpcPoint(
                    body_name=link,
                    local_offset=np.zeros(3, dtype=np.float64),
                    semantic=canon,
                    source_index=src_idx,
                )
                for link, src_idx, canon in zip(
                    robot_links, source_indices, canonical_names, strict=True,
                )
            ]

        leg_w = float(self.cfg.leg_effector_weight)
        if leg_w != 1.0:
            import dataclasses

            _leg_keys = ("ankle", "knee", "hip", "foot", "pelvis")
            robot_points = [
                dataclasses.replace(
                    pt,
                    weight=float(getattr(pt, "weight", 1.0)) * leg_w,
                )
                if any(k in str(getattr(pt, "semantic", "")).lower() for k in _leg_keys)
                else pt
                for pt in robot_points
            ]

        mapped_pos, z_min, smpl_scale = self._build_contact_scaled_positions(
            motion, robot_points,
        )
        mapped_pos = self._enforce_mapped_ankle_laterality(mapped_pos, robot_points)

        obj_positions, object_points = self._build_scaled_object_points(
            motion, z_min, smpl_scale,
        )

        # Inject terrain heightfield surface samples as additional
        # anchor vertices.  Holosoma builds the Delaunay tetrahedral
        # mesh out of *both* human joints and terrain points so the
        # Laplacian δ for each joint encodes its position relative to
        # the static terrain — that's the only mechanism in this
        # backend that anchors global root motion.  Without these
        # anchors the cost is translation-invariant and the robot
        # essentially stops walking even though its yellow source
        # skeleton sweeps several metres.
        terrain_pts = self._build_terrain_anchor_points(
            motion, z_min, smpl_scale, num_frames=int(mapped_pos.shape[0]),
        )
        if terrain_pts is not None:
            terrain_centroid = np.tile(
                terrain_pts[0].mean(axis=0)[None, :],
                (int(mapped_pos.shape[0]), 1),
            ).astype(np.float32)
            obj_positions = list(obj_positions) + [terrain_centroid]
            object_points = (list(object_points) if object_points else []) + [terrain_pts]

        scene = ScaledMotionScene(
            human_positions=mapped_pos,
            object_positions=obj_positions,
            object_uniform_scales=[smpl_scale] * len(obj_positions),
            object_points=object_points,
        )

        targets = precompute_target_laplacians(
            scene,
            object_extents=None,
            object_samples=self.cfg.object_surface_samples,
            max_human_vertices=None,
            progress_callback=progress_callback,
        )

        # Attach continuous source pelvis quaternions on **every** frame.
        # The SQP locks the FREE-joint orientation to these values so the
        # floating base cannot flip into another orientation basin during
        # ground rolls (the root cause of stand-up leg / yaw teleports).
        # Use the **raw** source quaternion (no ``source_body_quat``)
        # because :meth:`_build_scaled_source_pose` and the heightfield
        # transform also keep the source frame.
        try:
            from hhtools.core.math import quaternion as Q

            pelvis_idx = self._pelvis_source_index(motion)
            F = len(targets)
            if (
                pelvis_idx >= 0
                and motion.quaternions.shape[0] >= F
                and F > 0
            ):
                qs = Q.ensure_continuous(
                    Q.normalize(
                        np.asarray(
                            motion.quaternions[:F, pelvis_idx],
                            dtype=np.float64,
                        )
                    )
                )
                for f in range(F):
                    pq = qs[f]
                    if float(np.linalg.norm(pq)) > 1e-9:
                        targets[f].source_root_quat_xyzw = (
                            float(pq[0]),
                            float(pq[1]),
                            float(pq[2]),
                            float(pq[3]),
                        )
        except Exception:
            pass

        return targets, robot_links, z_min, smpl_scale, robot_points

    def _build_contact_scaled_positions(
        self,
        motion: Motion,
        robot_points,
    ) -> tuple[NDArray[np.float32], float, float]:
        scaled_pos, scaled_quat, z_min, smpl_scale = self._build_scaled_source_pose(motion)
        F = int(scaled_pos.shape[0])
        out = np.zeros((F, len(robot_points), 3), dtype=np.float32)
        src2can = auto_source_to_canonical(motion.hierarchy.bone_names)
        canonical_idx = self._canonical_source_indices(motion, src2can)
        for i, pt in enumerate(robot_points):
            src_idx = int(getattr(pt, "source_index", -1))
            if src_idx < 0 or src_idx >= int(scaled_pos.shape[1]):
                continue
            anchor = scaled_pos[:, src_idx, :]
            off = np.asarray(pt.local_offset, dtype=np.float32).reshape(3)
            if np.linalg.norm(off) > 1e-8:
                semantic = str(getattr(pt, "semantic", "")).lower()
                anchor = anchor + self._contact_offset_world(
                    semantic,
                    off,
                    scaled_pos,
                    scaled_quat,
                    src_idx,
                    canonical_idx,
                )
            out[:, i, :] = anchor
        return out, z_min, smpl_scale

    @staticmethod
    def _normalise_vectors(v: NDArray[np.float32], fallback: NDArray[np.float32]) -> NDArray[np.float32]:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        fb = np.broadcast_to(fallback.reshape(1, 3), v.shape).astype(np.float32)
        return np.where(n > 1e-6, v / np.maximum(n, 1e-6), fb).astype(np.float32)

    @staticmethod
    def _enforce_directional_continuity(
        fwd: NDArray[np.float32],
        body_heading: NDArray[np.float32],
        max_step_deg: float = 90.0,
    ) -> NDArray[np.float32]:
        """Clamp single-frame angular outliers in a unit direction sequence.

        Walks the sequence forward; whenever ``fwd[t]`` differs from
        ``fwd[t-1]`` by more than ``max_step_deg`` we replace ``fwd[t]``
        by ``body_heading[t]``.  This is a last-line guard against
        residual per-frame flicker on noisy rigs — by the time we reach
        it the temporal box filter has already removed the bulk of the
        oscillation, but a single 180° flip surviving the filter would
        still be visually disturbing.  Using ``body_heading[t]`` as the
        replacement keeps the substitute aligned with the rotating
        body, not frozen on the previous frame.
        """
        out = np.asarray(fwd, dtype=np.float32).copy()
        body = np.asarray(body_heading, dtype=np.float32)
        if out.shape != body.shape:
            body = np.broadcast_to(body.reshape(-1, 3), out.shape)
        threshold = float(np.cos(np.deg2rad(max_step_deg)))
        for t in range(1, out.shape[0]):
            d = float(out[t] @ out[t - 1])
            if d < threshold:
                out[t] = body[t]
        return out

    @staticmethod
    def _temporal_box_filter(
        v: NDArray[np.float32],
        window: int = 5,
    ) -> NDArray[np.float32]:
        """Replicate-edge box filter applied along axis 0.

        Used to denoise the per-frame XY projection of ``ankle → toe``
        before normalising it.  Without this step, the BVH-style rigs
        (LAFAN, Mixamo) produce 180° flips between adjacent frames
        because the toe joint is mostly *below* the ankle (the foot
        descriptor is dominated by ``-Z``) and the leftover XY component
        is small and noise-driven.  A 5-frame ~150ms smoother removes
        that flicker while preserving the slower rotational motion the
        user actually wants to see.
        """
        v_arr = np.asarray(v, dtype=np.float32)
        F = int(v_arr.shape[0])
        if F < 2 or window <= 1:
            return v_arr.copy()
        w = int(window)
        pad = w // 2
        idx = np.arange(F)
        out = np.zeros_like(v_arr)
        for k in range(-pad, pad + 1):
            shifted = np.clip(idx + k, 0, F - 1)
            out += v_arr[shifted]
        return (out / float(w)).astype(np.float32)

    @staticmethod
    def _normalise_vectors_per_frame(
        v: NDArray[np.float32],
        fallback: NDArray[np.float32],
        *,
        ref_norm: NDArray[np.float32] | None = None,
        ratio_threshold: float = 0.0,
    ) -> NDArray[np.float32]:
        """Normalise ``v`` per row, falling back to ``fallback[t]`` per frame.

        Unlike :meth:`_normalise_vectors`, the fallback is taken from the
        same frame as the degenerate ``v[t]`` instead of broadcasting a
        single reference vector across all frames.  That matters when the
        body rotates while a contact joint is briefly degenerate (e.g. a
        lifted foot during a 180° pivot): the fallback then tracks the
        rotating body heading rather than freezing on frame 0.

        When ``ref_norm`` and ``ratio_threshold`` are both supplied the
        fallback is also activated whenever ``norm(v[t]) < threshold *
        ref_norm[t]``.  This catches the "near-vertical foot" regime —
        the ankle → toe vector is mostly along ``-z`` (foot pointing
        down, en pointe, mid-kick) and its XY projection is dominated by
        noise that flips sign frame-to-frame.  Without this guard, the
        BVH-style LAFAN / Mixamo skeletons produce sporadic 180° jumps
        in the displayed toe direction even though the source motion is
        smooth.
        """
        v_arr = np.asarray(v, dtype=np.float32)
        fb_arr = np.asarray(fallback, dtype=np.float32)
        if fb_arr.ndim == 1:
            fb_arr = np.broadcast_to(fb_arr.reshape(1, 3), v_arr.shape)
        n = np.linalg.norm(v_arr, axis=1, keepdims=True)
        good = n > 1e-6
        if ref_norm is not None and ratio_threshold > 0.0:
            ref = np.asarray(ref_norm, dtype=np.float32).reshape(-1, 1)
            good = good & (n > float(ratio_threshold) * ref)
        return np.where(good, v_arr / np.maximum(n, 1e-6), fb_arr).astype(np.float32)

    @staticmethod
    def _global_anatomical_sign(
        v: NDArray[np.float32],
        ref: NDArray[np.float32],
    ) -> float:
        """Decide ONE global sign so ``v`` agrees with ``ref`` on average.

        Earlier versions used per-frame ``_align_direction_to_reference`` +
        ``_make_direction_continuous`` to keep contact-frame axes from
        flipping.  That combination is fragile when a single frame falls
        back to a frozen reference (e.g. ankle → foot becomes degenerate
        while the foot is lifted): the per-frame flip propagates through
        :meth:`_make_direction_continuous` and locks every subsequent
        frame to the wrong half-space, producing the visible "ankle bones
        don't rotate, then jump 180° once the body finishes turning"
        artefact reported on meshmimic / intermimic 180°-pivot clips.

        We instead trust the per-frame anatomical direction (it rotates
        smoothly with the body in any well-formed source motion) and only
        decide the **global** sign once, based on the median dot product
        with the per-frame body forward.  Outliers (a few lifted-foot
        frames) cannot pollute the decision, and there is no propagation
        step that can lock the trajectory to a stale heading.
        """
        v_arr = np.asarray(v, dtype=np.float32)
        r_arr = np.asarray(ref, dtype=np.float32)
        if v_arr.shape != r_arr.shape:
            r_arr = np.broadcast_to(r_arr.reshape(1, 3), v_arr.shape)
        dots = np.sum(v_arr * r_arr, axis=1)
        median_dot = float(np.median(dots)) if dots.size > 0 else 1.0
        return -1.0 if median_dot < 0.0 else 1.0

    @staticmethod
    def _pelvis_source_index(motion: Motion) -> int:
        """Return the bone index whose canonical role is ``pelvis``.

        Looks first for an exact ``pelvis`` name match (the SMPL
        convention), falls back to the alias map, and finally to bone
        index 0 — the canonical root of every supported source rig.
        """
        names = [str(n).lower() for n in motion.hierarchy.bone_names]
        for i, n in enumerate(names):
            if n == "pelvis":
                return i
        try:
            from hhtools.retarget.newton_basic.human_aliases import (
                auto_source_to_canonical,
            )
            src2can = auto_source_to_canonical(motion.hierarchy.bone_names)
            for i, n in enumerate(motion.hierarchy.bone_names):
                if src2can.get(n, "").lower() == "pelvis":
                    return i
        except Exception:
            pass
        return 0 if len(names) > 0 else -1

    @staticmethod
    def _canonical_source_indices(motion: Motion, src2can: dict[str, str]) -> dict[str, int]:
        """Choose one source bone index for each canonical semantic.

        Prefer explicit distal markers (toe / hand) when they exist, because
        contact virtual points need the end-effector direction rather than only
        the coarse ankle/wrist anchor.
        """
        names = list(motion.hierarchy.bone_names)
        result: dict[str, int] = {}
        priority_words = {
            "left_foot": ("toeend", "toe_end", "toebase", "toe", "footmod", "foot"),
            "right_foot": ("toeend", "toe_end", "toebase", "toe", "footmod", "foot"),
            "left_wrist": ("hand", "wrist"),
            "right_wrist": ("hand", "wrist"),
        }
        for i, name in enumerate(names):
            canon = src2can.get(name, name)
            if canon not in result:
                result[canon] = i

        for canon, words in priority_words.items():
            best: tuple[int, int] | None = None
            for i, name in enumerate(names):
                if src2can.get(name, name) != canon:
                    continue
                nn = name.lower()
                rank = next((r for r, w in enumerate(words) if w in nn), len(words))
                if best is None or rank < best[0]:
                    best = (rank, i)
            if best is not None:
                result[canon] = best[1]
        return result

    def _body_heading_forward(
        self,
        scaled_pos: NDArray[np.float32],
        canonical_idx: dict[str, int],
    ) -> NDArray[np.float32]:
        """Per-frame body forward direction from hip-lateral × world-up.

        We deliberately use **only** the hip / shoulder lateral cross
        product here, with no velocity-based blending.  The heading must
        be smooth across in-place rotations (where root velocity is zero
        or noisy) so it can serve as a per-frame fallback for
        :meth:`_contact_offset_world` when the ankle → foot vector is
        degenerate (foot lifted directly above the ankle).  The earlier
        velocity-blended version introduced threshold-induced
        discontinuities that the contact-offset code then propagated
        through :meth:`_make_direction_continuous`, locking the heel /
        toe direction to the wrong half-space for the rest of the clip.

        The convention matches
        :func:`hhtools.retarget.calibration.calibration._forward_from_shoulder_axis`
        — ``cross(left − right, world_up)`` — so the calibration's
        ``source_body_quat`` and this runtime fallback agree.
        """
        F = int(scaled_pos.shape[0])
        fallback = np.broadcast_to(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (F, 3),
        ).copy()

        left_i = canonical_idx.get("left_hip")
        right_i = canonical_idx.get("right_hip")
        if left_i is None or right_i is None:
            left_i = canonical_idx.get("left_shoulder")
            right_i = canonical_idx.get("right_shoulder")
        if left_i is None or right_i is None:
            return fallback

        lateral = scaled_pos[:, left_i, :] - scaled_pos[:, right_i, :]
        lateral[:, 2] = 0.0
        up = np.broadcast_to(
            np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (F, 3),
        )
        fwd_lat = np.cross(lateral, up).astype(np.float32)
        return self._normalise_vectors(fwd_lat, fallback[0])

    def _contact_offset_world(
        self,
        semantic: str,
        local_offset: NDArray[np.float32],
        scaled_pos: NDArray[np.float32],
        scaled_quat: NDArray[np.float32],
        anchor_idx: int,
        canonical_idx: dict[str, int],
    ) -> NDArray[np.float32]:
        F = int(scaled_pos.shape[0])
        up = np.broadcast_to(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (F, 3))

        if "ankle:toe" in semantic or "ankle:heel" in semantic:
            side = "left" if "left_" in semantic else "right"
            foot_i = canonical_idx.get(f"{side}_foot")
            body_heading = self._body_heading_forward(scaled_pos, canonical_idx)
            if foot_i is not None and foot_i != anchor_idx:
                raw3d = scaled_pos[:, foot_i, :] - scaled_pos[:, anchor_idx, :]
                xy = raw3d.copy()
                xy[:, 2] = 0.0
                # Two-stage denoise of the foot's anatomical forward
                # direction:
                #   1. small temporal box filter — kills per-frame XY
                #      noise on rigs where the toe is mostly below the
                #      ankle (BVH/LAFAN/Mixamo).  Without this, two
                #      adjacent frames can flip the XY sign (≈180°)
                #      even though the underlying motion is smooth.
                #   2. per-frame fallback to ``body_heading[t]`` when
                #      the smoothed XY is still effectively degenerate
                #      (foot lifted, en pointe).  The fallback is
                #      *per-frame* so that a degenerate frame in the
                #      middle of a body rotation tracks the rotating
                #      heading rather than freezing on frame 0.
                xy_smooth = self._temporal_box_filter(xy, window=5)
                xy_norm = np.linalg.norm(xy_smooth, axis=1)
                raw_norm = np.linalg.norm(raw3d, axis=1)
                fwd = self._normalise_vectors_per_frame(
                    xy_smooth, body_heading,
                    ref_norm=raw_norm, ratio_threshold=0.2,
                )
                # Belt-and-braces continuity guard: if a single frame
                # still flipped past 90° relative to its neighbours
                # (e.g. on a rig where the foot direction genuinely
                # oscillates between ground contacts), nudge it back to
                # the running average so the visualization stays stable.
                fwd = self._enforce_directional_continuity(fwd, body_heading)
            else:
                fwd = body_heading
            sign = self._global_anatomical_sign(fwd, body_heading)
            if sign < 0.0:
                fwd = -fwd
            side_axis = np.cross(up, fwd).astype(np.float32)
            side_axis = self._normalise_vectors(
                side_axis, np.array([0.0, 1.0, 0.0], dtype=np.float32),
            )
            return (
                fwd * float(local_offset[0])
                + side_axis * float(local_offset[1])
                + up * float(local_offset[2])
            ).astype(np.float32)

        if "wrist:hand_tip" in semantic:
            side = "left" if "left_" in semantic else "right"
            hand_i = canonical_idx.get(f"{side}_hand", canonical_idx.get(f"{side}_wrist"))
            elbow_i = canonical_idx.get(f"{side}_elbow")
            body_heading = self._body_heading_forward(scaled_pos, canonical_idx)
            if hand_i is not None and hand_i != anchor_idx:
                # The robot has no hand link: the wrist link's collision
                # geometry terminates where a hand would be, and that
                # collision tip is the de-facto end-effector (the robot
                # ``wrist:hand_tip`` point already sits at the far end of the
                # last wrist collision geom).  When the *source* rig carries a
                # distinct hand joint past the wrist (OMOMO ``l_hand``/``r_hand``,
                # grasping a chair/box), drive that robot tip to the human hand
                # joint itself — i.e. target = anchor(wrist) + (hand − wrist) =
                # hand — so the robot's physical hand-end reaches the actual
                # contact point instead of stopping a collision-tip-length
                # short at the wrist (the "still only a yellow skeleton touches
                # the chair" artefact).
                return (
                    scaled_pos[:, hand_i, :] - scaled_pos[:, anchor_idx, :]
                ).astype(np.float32)
            if elbow_i is not None:
                raw = scaled_pos[:, anchor_idx, :] - scaled_pos[:, elbow_i, :]
            else:
                raw = body_heading
            # No distinct hand joint (e.g. plain SMPL wrists are the tip):
            # extend along the forearm by the robot collision-tip length.
            # Same per-frame fallback logic as the foot path: when the arm
            # is fully extended along the body's vertical axis the XY
            # projection collapses, and we want to track the body heading
            # at *that* frame rather than freeze on frame 0.
            fwd = self._normalise_vectors_per_frame(raw, body_heading)
            return (fwd * float(np.linalg.norm(local_offset))).astype(np.float32)

        q = scaled_quat[:, anchor_idx, :]
        off_bc = np.broadcast_to(local_offset[None, :], (F, 3))
        return Q.rotate(q, off_bc).astype(np.float32)

    def _build_scaled_object_points(
        self,
        motion: Motion,
        z_min: float,
        smpl_scale: float,
    ) -> tuple[list[NDArray[np.float32]], list[NDArray[np.float32]] | None]:
        """Scale interaction object trajectories into the solver frame.

        Same contact-preserving transform as
        :meth:`_build_scaled_source_pose`: object centre XY × ``smpl_scale``,
        Z shifted by ``z_min`` (no Z scaling).  Box extents follow the
        same convention so the surface sampling lands on the scaled
        prop surface.  Terrain lives on :attr:`Motion.terrain` and is
        handled separately by :meth:`_build_terrain_anchor_points`;
        this helper deals only with the prop/box/mop entries on
        :attr:`Motion.objects`.
        """
        from hhtools.retarget.interaction_mesh.mpc_loop import sample_axis_aligned_box

        if not motion.objects:
            return [], None

        # Like ``_build_scaled_source_pose``, we keep objects in the
        # source world frame (no ``source_body_quat`` rotation) so they
        # stay aligned with the un-rotated terrain heightfield.  See
        # the docstring of ``_build_scaled_source_pose`` for the
        # parc_ms inverted-X bug this avoids.
        obj_positions: list[NDArray[np.float32]] = []
        obj_points: list[NDArray[np.float32]] = []
        for ob in motion.objects:
            op = ob.positions.copy().astype(np.float32)
            oq = ob.quaternions.copy().astype(np.float32)
            op[:, 2] -= z_min
            op *= smpl_scale
            obj_positions.append(op)

            ext = np.asarray(ob.extents, dtype=np.float32).reshape(3)
            ext = ext * float(ob.scale) * float(smpl_scale)
            local = sample_axis_aligned_box(self.cfg.object_surface_samples, ext)
            F = int(op.shape[0])
            pts = np.zeros((F, local.shape[0], 3), dtype=np.float32)
            for f in range(F):
                q_bc = np.broadcast_to(oq[f][None, :], (local.shape[0], 4))
                pts[f] = Q.rotate(q_bc, local).astype(np.float32) + op[f][None, :]
            obj_points.append(pts)

        return obj_positions, (obj_points if obj_points else None)

    def _build_terrain_anchor_points(
        self,
        motion: Motion,
        z_min: float,
        smpl_scale: float,
        num_frames: int,
    ) -> NDArray[np.float32] | None:
        """Subsample the heightfield surface and emit a ``(F, K, 3)`` anchor.

        This is the architectural counterpart of holosoma's
        ``object_points_local_demo`` (see
        ``holosoma/.../interaction_mesh_retargeter.py:_precompute_target_laplacians``).
        Holosoma stacks human joints **and** terrain surface samples
        into a single Delaunay tetrahedral mesh; the resulting Laplacian
        coordinates of every human joint are taken with respect to its
        terrain-point neighbours, which are stationary in world space.
        That neighbourhood structure is what gives the cost function a
        signal for **global** translation — without it, the Laplacian is
        translation-invariant and the floating base just smoothness-
        damps to a near-stationary trajectory regardless of how fast
        the source human is walking.

        Sampling strategy (per user spec — "地形采样点应该尽量以高
        度图变化比较大的位置作为采样点更可靠"):

        1. Score every cell by its local heightfield variation —
           specifically the gradient magnitude
           ``|∇h| = sqrt((∂h/∂x)² + (∂h/∂y)²)`` computed via
           central differences.  Edges of obstacles, ramp lips,
           stair treads etc. score high; flat ground scores ~0.
        2. Always include a low-density **uniform** stratum so the
           Delaunay tetrahedralisation has enough flat-ground
           anchors to triangulate the actor's footprint cleanly.
        3. Top up the remainder with cells drawn from the
           gradient-weighted distribution (probability ∝
           ``α + |∇h|²``).  ``α`` keeps the weight non-zero on
           flat areas so the sampler never collapses to a single
           hot edge if the rest of the field happens to be flat.

        The XY layout already sits in the **scaled** robot frame
        because we use :meth:`TerrainHeightfield.scaled` with
        ``z_offset=z_terrain`` (see :func:`~hhtools.core.grounding.terrain_heightfield_z_offset_world`)
        plus a Z shift ``(z_min - z_terrain) · smpl_scale`` so anchor heights
        match the human joint frame (``z_min`` is the foot floor passed in).
        """
        terrain = motion.terrain
        if terrain is None:
            return None
        K = max(8, int(self.cfg.terrain_surface_samples))

        z_terrain = float(terrain_heightfield_z_offset_world(motion, z_min))
        scaled_terrain = terrain.scaled(smpl_scale, z_offset=z_terrain)
        nx, ny = int(scaled_terrain.shape[0]), int(scaled_terrain.shape[1])
        if nx < 2 or ny < 2:
            return None

        hf = scaled_terrain.hf.astype(np.float32, copy=False)

        # --- Stratum A: uniform grid (covers the whole footprint) ---
        # Roughly half of the budget so we always have a regular
        # backbone for Delaunay even when the heightfield is mostly
        # flat (e.g. a single pit in an otherwise flat parkour map).
        K_uniform = max(8, K // 2)
        stride = max(1, int(np.floor(np.sqrt((nx * ny) / float(K_uniform)))))
        ix_u = np.arange(0, nx, stride, dtype=np.int32)
        iy_u = np.arange(0, ny, stride, dtype=np.int32)
        gx_u, gy_u = np.meshgrid(ix_u, iy_u, indexing="ij")
        uni_idx = np.stack([gx_u.ravel(), gy_u.ravel()], axis=1)

        # --- Stratum B: gradient-weighted samples (oversamples edges) ---
        # ``np.gradient`` returns (gx, gy) with central differences in
        # the interior and one-sided at borders — perfectly fine for a
        # weighting score.  Squared magnitude amplifies sharp edges
        # without going to infinity at any single cell.
        K_grad = max(0, K - uni_idx.shape[0])
        grad_idx: NDArray[np.int32] | None = None
        if K_grad > 0:
            gh_x, gh_y = np.gradient(hf)
            grad_sq = (gh_x.astype(np.float32) ** 2 + gh_y.astype(np.float32) ** 2)
            # ``alpha`` keeps flat cells non-zero — without it, a clip
            # whose terrain is mostly flat with one cliff would
            # collapse every sample onto that cliff and Delaunay would
            # produce a near-degenerate hull.
            alpha = max(1e-6, float(grad_sq.mean()) * 0.05)
            w = (grad_sq + alpha).reshape(-1)
            w = w / float(w.sum())
            rng = np.random.default_rng(seed=int(nx) * 7919 + int(ny))
            flat_pick = rng.choice(
                w.size, size=int(K_grad), replace=(K_grad > w.size), p=w,
            )
            grad_idx = np.stack(
                [flat_pick // ny, flat_pick % ny], axis=1,
            ).astype(np.int32)

        # Merge uniform + gradient samples and dedupe.
        if grad_idx is not None and grad_idx.size > 0:
            all_idx = np.vstack([uni_idx.astype(np.int32), grad_idx])
        else:
            all_idx = uni_idx.astype(np.int32)
        # Dedupe via Cantor pairing on (gx, gy) — keeps insertion order
        # so the uniform backbone wins ties over gradient resamples.
        keys = all_idx[:, 0].astype(np.int64) * (ny + 1) + all_idx[:, 1].astype(np.int64)
        _, uniq_first = np.unique(keys, return_index=True)
        uniq_first.sort()
        all_idx = all_idx[uniq_first]
        gx, gy = all_idx[:, 0], all_idx[:, 1]

        dx = float(scaled_terrain.dx)
        x0, y0 = float(scaled_terrain.min_point[0]), float(scaled_terrain.min_point[1])
        xs = x0 + gx.astype(np.float32) * dx
        ys = y0 + gy.astype(np.float32) * dx
        zs = hf[gx, gy].astype(np.float32)

        pts_static = np.stack([xs, ys, zs], axis=1).astype(np.float32)
        dz_align = float(z_min - z_terrain) * float(smpl_scale)
        if abs(dz_align) > 1e-9:
            pts_static = pts_static.copy()
            pts_static[:, 2] += np.float32(dz_align)
        n_uni = int(uni_idx.shape[0])
        n_grad = int(pts_static.shape[0] - n_uni)
        _log.debug(
            "Terrain anchors: %d cells (%d uniform + %d curvature-weighted) "
            "from %dx%d heightfield",
            int(pts_static.shape[0]), n_uni, max(n_grad, 0), nx, ny,
        )
        pts_traj = np.broadcast_to(
            pts_static[None, :, :], (int(num_frames), pts_static.shape[0], 3)
        ).copy()
        return pts_traj

    def _build_hard_collision_scene(
        self,
        motion: Motion,
        base_model,
        z_min: float,
        smpl_scale: float,
        *,
        base_xml: str = "",
    ):
        """Compile a MuJoCo collision model with the terrain as ``<hfield>``.

        Returns ``(collision_model, collision_data, tmp_files)``.  When
        ``motion.terrain is None`` we still return a model with the
        ground plane only (callers that don't have terrain still benefit
        from hard ground non-penetration).  Returns
        ``(None, None, [])`` on compile failure so callers can fall back
        to the soft penalty.

        The terrain heightfield is transported into the **robot frame**
        before being handed to MuJoCo:

            terrain_robot = motion.terrain.scaled(
                smpl_scale,
                z_offset=float(terrain_heightfield_z_offset_world(motion, z_min)),
            )

        which matches the heightfield passed to MuJoCo (foot-floor ``z_min`` for
        the skeleton; :func:`~hhtools.core.grounding.terrain_heightfield_z_offset_world`
        for ``z_offset`` on the grid).
        """
        import mujoco

        from hhtools.retarget.interaction_mesh.collision import (
            build_collision_model_with_hfield,
        )

        urdf_path = getattr(self.robot.preset, "urdf_path", None)
        if urdf_path is None:
            _log.warning(
                "robot has no preset.urdf_path; falling back to soft "
                "collision penalty (no hard non-penetration)",
            )
            return None, None, []
        urdf_dir = Path(urdf_path).parent

        terrain_robot = None
        if motion.terrain is not None:
            terrain_robot = motion.terrain.scaled(
                smpl_scale,
                z_offset=float(terrain_heightfield_z_offset_world(motion, z_min)),
            )

        try:
            coll_model, tmp_files = build_collision_model_with_hfield(
                base_model,
                urdf_dir,
                terrain_robot,
                base_xml=base_xml,
                add_ground=True,
                ground_z=0.0,
            )
        except Exception as exc:
            _log.error(
                "build_collision_model_with_hfield failed: %s; "
                "falling back to soft penalty",
                exc,
            )
            return None, None, []

        coll_data = mujoco.MjData(coll_model)
        return coll_model, coll_data, tmp_files

    def run(
        self,
        motion: Motion,
        *,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ):
        """SQP + RTI MPC on MuJoCo; returns :class:`~hhtools.retarget.newton_basic.pipeline.RetargetedMotion`."""
        from hhtools.retarget.interaction_mesh.collision import cleanup_terrain_files
        from hhtools.retarget.interaction_mesh.mpc_loop import (
            _arm_actuated_qpos_indices,
            _leg_actuated_qpos_indices,
            causal_smooth_actuated_qpos,
            iterate_mpc_rti,
        )
        from hhtools.retarget.retarget_result import RetargetedMotion

        scene = MujocoScene.from_robot(self.robot)
        mj = scene.model

        def _pre_cb(cur: int, tot: int) -> None:
            if progress_callback is not None:
                progress_callback("precompute", cur, tot)

        targets, robot_links, z_min, smpl_scale, robot_points = self.precompute_laplacian_targets(
            motion, mj_model=mj, progress_callback=_pre_cb,
        )

        def _mpc_cb(cur: int, tot: int) -> None:
            if progress_callback is not None:
                progress_callback("mpc", cur, tot)

        # ------------------------------------------------------------------
        # Collision strategy.  See :class:`InteractionMeshPipelineConfig`.
        #
        # The terrain is compiled as a MuJoCo ``<hfield>`` and the SQP
        # picks up ``mj_geomDistance``-based hard inequality constraints
        # in OSQP.  This is the only path: soft penalties on a
        # translation-invariant Laplacian cost cause the floating base
        # to drop / teleport, which is the failure mode this refactor
        # exists to fix.  When ``enable_collision=False`` (or the
        # collision model fails to compile) the SQP runs without
        # terrain awareness — fine for flat-ground clips.
        # ------------------------------------------------------------------
        coll_model = None
        coll_data = None
        tmp_terrain_files: list[Path] = []
        used_collision_mode = "none"

        if self.cfg.enable_collision:
            coll_model, coll_data, tmp_terrain_files = self._build_hard_collision_scene(
                motion, mj, z_min, smpl_scale,
                base_xml=getattr(scene, "mjcf_xml", "") or "",
            )
            if coll_model is not None:
                used_collision_mode = "hard_mj"
            else:
                _log.info(
                    "hard non-penetration model unavailable; "
                    "running without terrain-aware collision",
                )
                used_collision_mode = "none"

        try:
            traj = iterate_mpc_rti(
                mj,
                scene.data,
                targets,
                robot_body_names=robot_links,
                robot_points=robot_points,
                laplacian_weight=self.cfg.laplacian_weight,
                step_size=self.cfg.sqp_step_size,
                smooth_weight=self.cfg.smooth_weight,
                mpc_horizon=self.cfg.mpc_horizon,
                sqp_inner_iters=self.cfg.sqp_inner_iters,
                sqp_inner_iters_frame0=self.cfg.sqp_inner_iters_frame0,
                mpc_window_sqp_iters=self.cfg.mpc_window_sqp_iters,
                mpc_window_warm_start=self.cfg.mpc_window_warm_start,
                mpc_collision_commit_only=self.cfg.mpc_collision_commit_only,
                position_weight=self.cfg.position_weight,
                home_pose_weight=self.cfg.home_pose_weight,
                activate_foot_sticking=self.cfg.activate_foot_sticking,
                foot_sticking_tolerance=self.cfg.foot_sticking_tolerance,
                foot_sticking_velocity_threshold=self.cfg.foot_sticking_velocity_threshold,
                foot_sticking_release_hysteresis=self.cfg.foot_sticking_release_hysteresis,
                leg_smooth_weight=self.cfg.leg_smooth_weight,
                leg_sqp_step_scale=self.cfg.leg_sqp_step_scale,
                collision_model=coll_model,
                collision_data=coll_data,
                collision_threshold=self.cfg.collision_threshold,
                penetration_tolerance=self.cfg.penetration_tolerance,
                collision_fd_epsilon=self.cfg.collision_fd_epsilon,
                base_step_size=self.cfg.sqp_base_step_size,
                lock_root_orientation_to_source=self.cfg.lock_root_orientation_to_source,
                progress_callback=_mpc_cb,
            )
        finally:
            if tmp_terrain_files:
                cleanup_terrain_files(tmp_terrain_files)

        if self.cfg.post_smooth_leg_joints and traj.shape[0] > 1:
            leg_idx = _leg_actuated_qpos_indices(mj)
            traj = causal_smooth_actuated_qpos(
                traj,
                leg_idx,
                beta=float(self.cfg.post_smooth_leg_beta),
            )
        if self.cfg.post_smooth_arm_joints and traj.shape[0] > 1:
            arm_idx = _arm_actuated_qpos_indices(mj)
            traj = causal_smooth_actuated_qpos(
                traj,
                arm_idx,
                beta=float(self.cfg.post_smooth_arm_beta),
            )

        dof_names = self.robot.dof_names()
        rows = [pack_joint_q_csv(mj, dof_names, traj[f]) for f in range(traj.shape[0])]
        joint_q = np.stack(rows, axis=0).astype(np.float32, copy=False)
        joint_q = self._align_root_to_source_heading(joint_q)
        fps = float(motion.framerate)
        joint_q = self._clamp_solved_foot_lateral(joint_q, framerate=fps)
        if self.cfg.stabilize_joints_after_mpc:
            joint_q = self._stabilize_joints_after_mpc(
                joint_q, motion, framerate=fps,
            )
        if (
            self.cfg.post_smooth_leg_joints_after_stabilize
            and joint_q.shape[0] > 1
        ):
            # Causal low-pass on packed leg columns (after clamp+rate-limit).
            beta = float(self.cfg.post_smooth_leg_beta_after)
            if beta > 0.0:
                _leg_keys = ("hip", "knee", "ankle", "thigh", "calf", "leg", "shank")
                leg_cols = [
                    7 + i
                    for i, n in enumerate(dof_names)
                    if 7 + i < joint_q.shape[1]
                    and any(k in str(n).lower() for k in _leg_keys)
                ]
                if leg_cols:
                    b = float(np.clip(beta, 0.0, 0.95))
                    out_s = joint_q.astype(np.float32, copy=True)
                    for t in range(1, out_s.shape[0]):
                        for c in leg_cols:
                            out_s[t, c] = (1.0 - b) * out_s[t, c] + b * out_s[t - 1, c]
                    joint_q = out_s

        # ---- Diagnostic: per-clip alignment quality vs. yellow-skeleton ----
        # Measures the residual between every retargeted robot
        # contact-point world position and its scaled-source target
        # (i.e. the yellow-overlay landmark it was supposed to hit).
        # This is the same quantity the user complains about when
        # they say "the robot doesn't track the yellow skeleton" —
        # logging it explicitly turns "feels off" into "12 cm at the
        # left wrist", which is far easier to debug.  Cheap (one
        # ``mj_forward`` per frame in the pre-existing collision
        # model) and only runs once per clip, so the cost is
        # negligible relative to the SQP itself.
        align_summary = self._summarise_alignment_quality(
            traj, targets, robot_points,
            mj_model=coll_model if coll_model is not None else mj,
        )

        # ---- Z-snap to terrain ----------------------------------
        # The SQP only enforces ``foot ≥ terrain`` via hard non-
        # penetration; nothing pulls the foot **down** to actually
        # touch the heightfield, so a clip whose source actor was
        # genuinely airborne for some frames may end the entire
        # retargeted trajectory floating a few centimetres above
        # the heightfield even at frames where contact was clearly
        # intended.  Per the user spec, the lowest foot point over
        # the **whole** clip should sit exactly on the heightfield
        # (gap = 0); anything above that "floor offset" is removed
        # in a single global Z translation that preserves all
        # relative motion (jumps, stairs, terrain following).
        #
        # We re-use the SAME collision model that the SQP just
        # solved against — so foot geoms and the heightfield
        # exist in a single MuJoCo scene at the right scale — but
        # measure the gap **geometrically** (foot OBB lowest Z vs
        # bilinear-interp terrain Z at that XY) rather than via
        # ``mj_geomDistance``.  The latter returns ambiguous deep-
        # penetration depths for any foot pose that intersects the
        # hfield solid (a known MuJoCo limitation for non-convex
        # primitives), which would falsely report the entire clip
        # as "in penetration" and disable the snap.  The OBB-Z vs
        # bilinear-Z formulation is what the user wrote in their
        # spec ("足端最低点和跑酷地形之间的距离") and is robust
        # against constraint slack at SQP termination.
        floor_offset = 0.0
        signed_min_gap = float("nan")
        if coll_model is not None and motion.terrain is not None:
            terrain_robot = motion.terrain.scaled(
                float(smpl_scale),
                z_offset=float(terrain_heightfield_z_offset_world(motion, z_min)),
            )
            signed_min_gap = self._terrain_signed_min_gap_geom(
                coll_model, coll_data, joint_q, terrain_robot,
            )
            floor_offset = max(0.0, signed_min_gap)
        if floor_offset > 1e-4:
            joint_q = joint_q.copy()
            joint_q[:, 2] -= np.float32(floor_offset)
            _log.info(
                "Z-snap: lifted ground contact by %.4fm so the lowest "
                "foot point during the trajectory touches the terrain",
                floor_offset,
            )
        elif np.isfinite(signed_min_gap) and signed_min_gap < -1e-4:
            _log.warning(
                "Z-snap: lowest foot point is %.4fm BELOW terrain — "
                "leaving trajectory unchanged.  This means the SQP's "
                "non-penetration constraint left residual penetration "
                "at some frame; consider raising "
                "``collision_threshold`` or tightening "
                "``penetration_tolerance``.",
                signed_min_gap,
            )

        _meta_r = {
                "retarget_backend": "interaction_mesh",
                "mpc_horizon": self.cfg.mpc_horizon,
                "laplacian_weight": self.cfg.laplacian_weight,
                "sqp_step_size": self.cfg.sqp_step_size,
                "sqp_inner_iters": self.cfg.sqp_inner_iters,
                "smooth_weight": self.cfg.smooth_weight,
                "enable_collision": self.cfg.enable_collision,
                "collision_mode_used": used_collision_mode,
                "collision_threshold": self.cfg.collision_threshold,
                "penetration_tolerance": self.cfg.penetration_tolerance,
                "sqp_base_step_size": self.cfg.sqp_base_step_size,
                "contact_points": len(robot_points),
                "smpl_scale": float(smpl_scale),
                "source_z_min": float(z_min),
                "source_terrain_z_offset": (
                    float(terrain_heightfield_z_offset_world(motion, z_min))
                    if motion.terrain is not None
                    else float("nan")
                ),
                "terrain_floor_offset": float(floor_offset),
                "terrain_signed_min_gap": float(signed_min_gap),
                "position_weight": float(self.cfg.position_weight),
                "alignment_mean_m": align_summary["mean_m"],
                "alignment_max_m": align_summary["max_m"],
                "alignment_pelvis_m": align_summary["pelvis_m"],
                "alignment_wrist_m": align_summary["wrist_m"],
                "alignment_ankle_m": align_summary["ankle_m"],
        }

        from hhtools.robot.retarget_profile import apply_upper_body_roll_narrowing_post_ik

        joint_q = apply_upper_body_roll_narrowing_post_ik(
            joint_q,
            dof_names,
            self.robot.preset,
            root_coord_count=7,
            robot_model=self.robot,
        )

        return RetargetedMotion(
            name=motion.name or "interaction_mesh",
            joint_q=joint_q,
            sample_rate=float(motion.framerate),
            dof_names=dof_names,
            root_coord_count=7,
            meta=_meta_r,
        )

    @staticmethod
    def _summarise_alignment_quality(
        traj: NDArray[np.float64],
        targets,
        robot_points,
        *,
        mj_model,
    ) -> dict[str, float]:
        """Mean/max world-space residual between robot points and targets.

        Diagnostic only — the SQP has already converged, so this just
        replays each frame's qpos through MuJoCo's forward kinematics
        and compares ``robot_points[i]`` world position against
        ``targets[f].source_vertices[i]`` (which is the **same**
        scaled-source landmark that the yellow overlay renders for
        that joint).  Logged at ``info`` level so the user can see
        "is the robot tracking the yellow skeleton" directly without
        running an external script.
        """
        import mujoco

        if not targets or traj.shape[0] == 0:
            return {
                "mean_m": float("nan"), "max_m": float("nan"),
                "pelvis_m": float("nan"), "wrist_m": float("nan"),
                "ankle_m": float("nan"),
            }
        nh = int(targets[0].n_human_vertices)
        F = int(traj.shape[0])
        # Diagnostic only — replaying ``mj_forward`` for *every* frame of a long
        # terrain clip (thousands of frames) is a second full FK pass purely to
        # log alignment numbers.  Sample an evenly-spaced subset: the
        # mean/max/region stats are statistically indistinguishable while the
        # cost stays bounded regardless of clip length.
        if F > _ALIGNMENT_DIAG_MAX_FRAMES:
            sample_idx = np.linspace(
                0, F - 1, _ALIGNMENT_DIAG_MAX_FRAMES,
            ).round().astype(np.int64)
            sample_idx = np.unique(sample_idx)
        else:
            sample_idx = np.arange(F, dtype=np.int64)
        body_ids: list[int] = []
        for pt in robot_points[:nh]:
            bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, pt.body_name)
            body_ids.append(int(bid))
        d = mujoco.MjData(mj_model)
        all_err = np.zeros((len(sample_idx), nh), dtype=np.float64)
        for row_i, f in enumerate(sample_idx):
            f = int(f)
            target_pos = np.asarray(
                targets[f].source_vertices[:nh], dtype=np.float64,
            )
            d.qpos[:mj_model.nq] = traj[f, :mj_model.nq]
            mujoco.mj_forward(mj_model, d)
            for i, bid in enumerate(body_ids):
                if bid < 0:
                    continue
                R = d.xmat[bid].reshape(3, 3)
                off = np.asarray(robot_points[i].local_offset, dtype=np.float64).reshape(3)
                p_w = d.xpos[bid].astype(np.float64) + R @ off
                all_err[row_i, i] = float(np.linalg.norm(p_w - target_pos[i]))
        # Per-semantic subsets so we can spot which body region drifts.
        wrist_idx = [i for i, p in enumerate(robot_points[:nh]) if "wrist" in str(p.semantic).lower()]
        ankle_idx = [i for i, p in enumerate(robot_points[:nh]) if "ankle" in str(p.semantic).lower()]
        out = {
            "mean_m": float(all_err.mean()),
            "max_m": float(all_err.max()),
            "pelvis_m": float(all_err[:, 0].mean()) if nh > 0 else float("nan"),
            "wrist_m": float(all_err[:, wrist_idx].mean()) if wrist_idx else float("nan"),
            "ankle_m": float(all_err[:, ankle_idx].mean()) if ankle_idx else float("nan"),
        }
        _log.info(
            "Yellow-skeleton alignment: mean %.3fm  max %.3fm  "
            "(pelvis %.3fm, wrist %.3fm, ankle %.3fm) — "
            "sampled %d/%d frames, n_points=%d",
            out["mean_m"], out["max_m"],
            out["pelvis_m"], out["wrist_m"], out["ankle_m"],
            len(sample_idx), F, nh,
        )
        return out

    @staticmethod
    def _terrain_signed_min_gap_geom(
        coll_model,
        coll_data,
        joint_q: NDArray[np.float32],
        terrain_robot: TerrainHeightfield,
    ) -> float:
        """Signed minimum foot-OBB-bottom ↔ terrain-elevation gap.

        For every frame and every foot **body** (URDF link whose
        name contains ``"foot"``, ``"ankle"`` or ``"toe"``) we:

        1. Compute the body's world XY from ``data.xpos``.
        2. Query the heightfield's bilinearly-interpolated elevation
           at that XY — this is "the terrain the foot is standing on".
        3. Compute the lowest world-Z point reachable by any
           collidable geom rigidly attached to that body.  We use
           the OBB closed-form ``z_min = xpos.z − Σ |R[2,i]| · h_i``
           where ``R`` is ``geom_xmat`` and ``h`` is the local AABB
           half-extents from ``geom_aabb`` — exact for boxes/spheres
           and for mesh geoms whose AABB is precomputed by MuJoCo.
        4. Track the minimum of ``z_min − terrain_z`` over the
           whole clip.

        Querying terrain at the *body* XY (rather than at every OBB
        corner XY) is what the user's "foot point vs terrain"
        requirement captures: we don't want the lowest corner to
        sample a different stair next to the foot's standing
        location, which would falsely report metres of "penetration"
        because the OBB corner happens to sit over a tall box.

        Returns the signed minimum gap; negative values mean some
        frame already shows the foot sole sitting below the
        heightfield (constraint slack at SQP termination) and the
        caller is responsible for clamping to ``≥ 0`` before using
        it as a Z-snap offset.

        Frames whose foot XY falls outside the heightfield extent
        are skipped — the source clip likely has the actor stepping
        off the hfield's footprint and the gap there isn't well
        defined.
        """
        import mujoco

        # Group collidable geoms by foot body so we can reduce
        # "lowest world-Z" per body × frame.
        foot_body_to_geoms: dict[int, list[int]] = {}
        for g in range(coll_model.ngeom):
            if int(coll_model.geom_contype[g]) == 0 and int(coll_model.geom_conaffinity[g]) == 0:
                continue
            bid = int(coll_model.geom_bodyid[g])
            bname = (mujoco.mj_id2name(coll_model, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
            if any(tag in bname for tag in ("foot", "ankle", "toe")):
                foot_body_to_geoms.setdefault(bid, []).append(g)
        if not foot_body_to_geoms:
            return 0.0

        actuated_qadr: list[int] = []
        for j in range(coll_model.njnt):
            jt = int(coll_model.jnt_type[j])
            if jt in (int(mujoco.mjtJoint.mjJNT_HINGE),
                      int(mujoco.mjtJoint.mjJNT_SLIDE)):
                actuated_qadr.append(int(coll_model.jnt_qposadr[j]))

        nx, ny = terrain_robot.hf.shape
        min_x = float(terrain_robot.min_point[0])
        min_y = float(terrain_robot.min_point[1])
        dx = float(terrain_robot.dx)
        ext_x = (nx - 1) * dx
        ext_y = (ny - 1) * dx
        hf = terrain_robot.hf.astype(np.float64, copy=False)

        def hf_at(x: float, y: float) -> float | None:
            if x < min_x or x > min_x + ext_x or y < min_y or y > min_y + ext_y:
                return None
            fx = (x - min_x) / dx
            fy = (y - min_y) / dx
            i0 = int(np.clip(int(fx), 0, nx - 2))
            j0 = int(np.clip(int(fy), 0, ny - 2))
            tx = fx - i0
            ty = fy - j0
            h00 = float(hf[i0, j0]); h10 = float(hf[i0 + 1, j0])
            h01 = float(hf[i0, j0 + 1]); h11 = float(hf[i0 + 1, j0 + 1])
            return ((1 - tx) * (1 - ty) * h00 + tx * (1 - ty) * h10
                    + (1 - tx) * ty * h01 + tx * ty * h11)

        F = int(joint_q.shape[0])
        min_gap = float("inf")
        for f in range(F):
            row = joint_q[f]
            coll_data.qpos[:3] = row[:3]
            coll_data.qpos[3] = float(row[6])
            coll_data.qpos[4] = float(row[3])
            coll_data.qpos[5] = float(row[4])
            coll_data.qpos[6] = float(row[5])
            for k, qadr in enumerate(actuated_qadr):
                coll_data.qpos[qadr] = float(row[7 + k])
            mujoco.mj_forward(coll_model, coll_data)
            for bid, geoms in foot_body_to_geoms.items():
                # Body XY (= the foot's standing location).
                bx, by, _ = coll_data.xpos[bid]
                h_terrain = hf_at(float(bx), float(by))
                if h_terrain is None:
                    continue
                # Lowest reachable world-Z over all collidable geoms
                # rigidly attached to this foot body.  Closed-form
                # OBB minimum-Z: z_c − Σ |R[2,i]| · h_i.  Includes
                # the AABB centre offset ``ab[:3]`` projected onto
                # world Z so spheres / capsules whose local centres
                # are non-zero don't drift up.
                z_lowest = float("inf")
                for g in geoms:
                    cw_z = float(coll_data.geom_xpos[g, 2])
                    R = np.asarray(
                        coll_data.geom_xmat[g], dtype=np.float64,
                    ).reshape(3, 3)
                    ab = np.asarray(
                        coll_model.geom_aabb[g], dtype=np.float64,
                    )
                    centre_z = float(R[2] @ ab[:3])
                    half_z = float(np.abs(R[2]) @ ab[3:])
                    z_geom = cw_z + centre_z - half_z
                    if z_geom < z_lowest:
                        z_lowest = z_geom
                if not np.isfinite(z_lowest):
                    continue
                gap = z_lowest - h_terrain
                if gap < min_gap:
                    min_gap = gap
        if not np.isfinite(min_gap):
            return float("nan")
        return float(min_gap)


__all__ = ["InteractionMeshPipeline", "InteractionMeshPipelineConfig"]
