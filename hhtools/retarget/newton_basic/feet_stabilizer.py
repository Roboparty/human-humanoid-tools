"""Pre-IK effector-target constraints (pure NumPy, CPU).

Upstream soma-retargeter splits its "feet stabilization" concern across two
modules:

* ``soma_retargeter.robotics.human_to_robot_scaler`` owns the *pre-IK*
  constraint chain (``_enforce_ground_contact``, ``_enforce_foot_planting``,
  ``_enforce_min_lateral_separation``, ``_smooth_corrections`` …) that fixes
  effector targets *before* they're fed to IK.
* ``soma_retargeter.pipelines.feet_stabilizer`` runs a *post-IK* Warp
  two-bone-IK solve on the robot rig to snap ankles to those targets.

Stage-1 of the hhtools port only ships the *pre-IK* half here — purely numpy,
no Newton/Warp required.  The two-bone-IK solve will land in a follow-up
stage that introduces the actual IK solver.  Splitting these responsibilities
keeps the constraint logic trivially unit-testable (synthetic effector
trajectories in, constrained trajectories out) while we wait on the solver.

Attribution:
  Portions of the constraint formulas are adapted from soma-retargeter
  (Apache-2.0).
  https://github.com/NVlabs/SOMA-Retargeter
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hhtools.retarget.newton_basic.config import FeetStabilizerConfig


__all__ = ["FeetStabilizer", "StabilizationStats"]


_AXIS_TO_IDX = {"X": 0, "Y": 1, "Z": 2}


@dataclass(frozen=True)
class StabilizationStats:
    """Diagnostics so callers can surface per-clip tuning signal.

    Attributes:
        ground_corrections: ``(F,)`` float array — per-frame metres the
            effector block was lifted or pushed to satisfy ``ground_contact_z``.
        planted_frames: ``(num_foot_pairs, F)`` bool array — ``True`` on
            frames that were detected as "foot planted" and locked.
        smoothed_frames: Count of frames that had at least one correction
            clamped by the smoothing rate limit.
    """

    ground_corrections: NDArray
    planted_frames: NDArray
    smoothed_frames: int


class FeetStabilizer:
    """Apply foot-planting / ground-contact / lateral-separation / smoothing
    constraints to an effector trajectory.

    The input/output layout matches :class:`~hhtools.retarget.newton_basic
    .scaler.ScaledEffectors.transforms`: ``(F, M, 7)`` with
    ``(x, y, z, qx, qy, qz, qw)`` per mapped joint.  Quaternions are passed
    through untouched (we only constrain positional targets at this stage).
    """

    def __init__(
        self,
        config: FeetStabilizerConfig,
        *,
        joint_names: tuple[str, ...],
    ) -> None:
        self._config = config
        self._joint_names = tuple(joint_names)
        self._up = _AXIS_TO_IDX[config.up_axis]
        # Lateral / forward axes are whatever remains after up
        axes = {0, 1, 2} - {self._up}
        self._fwd = _AXIS_TO_IDX[config.forward_axis]
        if self._fwd not in axes:
            # forward == up is invalid — fall back to first non-up axis so the
            # stabilizer is still usable when misconfigured.
            self._fwd = sorted(axes)[0]
        self._lat = sorted(axes - {self._fwd})[0]

        self._foot_indices = self._resolve_foot_indices()
        self._foot_toe_pairs = self._resolve_foot_toe_pairs()
        self._hips_idx = self._resolve(
            "hips_name",
            fallbacks=("Hips", "hips", "pelvis"),
        )
        self._lateral_pairs = self._resolve_lateral_pairs()
        self._body_probe_indices, self._body_probe_below = self._resolve_body_probes()
        self._body_ground_lift_smoothed = 0.0
        self._hand_chain_indices = self._resolve_hand_chains()
        self._arm_chains = self._resolve_arm_chains()
        self._arm_reach_map = self._build_arm_reach_map()
        self._chest_idx = self._resolve_joint(
            self._config.chest_name,
            fallbacks=("Spine2", "chest", "spine", "Spine1"),
        )
        self.hand_ground_drop_per_frame = np.zeros(0, dtype=np.float32)

    # --------------------------------------------------------------- properties

    @property
    def config(self) -> FeetStabilizerConfig:
        return self._config

    @property
    def up_axis_idx(self) -> int:
        return self._up

    @property
    def foot_indices(self) -> tuple[int, ...]:
        """Indices (into the mapped-joint list) of the left / right foot."""
        return self._foot_indices

    # --------------------------------------------------------------- apply

    def apply(
        self, effectors: NDArray, *, return_stats: bool = False
    ) -> NDArray | tuple[NDArray, StabilizationStats]:
        """Run the full constraint chain on ``(F, M, 7)`` effectors.

        Returns a *new* array.  The original upstream returns an in-place
        mutated buffer; we copy up-front because modern numpy pipelines lean
        on immutability for cache friendliness.

        Order of operations mirrors soma's ``_postprocess_scaled_effectors_batched``:

        1. Foot planting       — lock horizontal positions when a foot is still.
        2. Min lateral sep     — push L/R pairs apart.
        3. Ground contact      — lift blocks that float in upright poses.
        4. Hand ground contact — lower torso so hands can reach the floor.
        5. Smoothing           — rate-limit the deltas we just added.
        6. Body ground clearance — lift probes that still penetrate.
        """
        arr = np.asarray(effectors, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[-1] != 7:
            raise ValueError(
                f"effectors must be (F, M, 7); got {arr.shape}"
            )
        if arr.shape[1] != len(self._joint_names):
            raise ValueError(
                f"effectors M={arr.shape[1]} does not match "
                f"len(joint_names)={len(self._joint_names)}"
            )

        pre_constraints_pos = arr[..., 0:3].copy()
        out = arr.copy()

        planted = self._apply_foot_planting(out)
        self._apply_min_lateral_separation(out)
        ground_corr = self._apply_ground_contact(out)
        self._apply_hand_ground_contact(out)
        smoothed_frames = self._smooth_corrections(out, pre_constraints_pos)
        self._apply_body_ground_clearance(out)

        if not return_stats:
            return out

        stats = StabilizationStats(
            ground_corrections=ground_corr.astype(np.float32, copy=False),
            planted_frames=planted,
            smoothed_frames=int(smoothed_frames),
        )
        return out, stats

    # --------------------------------------------------------------- planting

    def _apply_foot_planting(self, effectors: NDArray) -> NDArray:
        """Horizontal lock when a foot's horizontal velocity stays below threshold.

        Returns an ``(num_pairs, F)`` bool mask of planted frames (for stats).
        """
        cfg = self._config
        pairs = self._foot_toe_pairs
        if cfg.foot_planting_velocity_threshold <= 0.0 or not pairs:
            return np.zeros((len(pairs), effectors.shape[0]), dtype=bool)

        F = effectors.shape[0]
        if F < 3:
            return np.zeros((len(pairs), F), dtype=bool)

        vel_thresh = cfg.foot_planting_velocity_threshold
        height_limit = cfg.ground_contact_z + cfg.foot_planting_height_margin
        release_n = max(1, cfg.foot_planting_release_frames)
        h_axes = [self._fwd, self._lat]
        up = self._up

        planted_all = np.zeros((len(pairs), F), dtype=bool)

        for p_idx, (fi, ti) in enumerate(pairs):
            horiz = effectors[:, fi, h_axes].copy()
            h = effectors[:, fi, up]

            vel_h = np.zeros(F)
            vel_h[1:] = np.linalg.norm(np.diff(horiz, axis=0), axis=1)
            planted = (vel_h < vel_thresh) & (h < height_limit)
            planted_all[p_idx] = planted

            lock_pos = None
            toe_delta = np.zeros(2) if ti is not None else None

            for f in range(F):
                if planted[f]:
                    if lock_pos is None:
                        lock_pos = horiz[f].copy()
                        if ti is not None:
                            toe_delta = effectors[f, ti, h_axes] - horiz[f]
                    effectors[f, fi, h_axes[0]] = lock_pos[0]
                    effectors[f, fi, h_axes[1]] = lock_pos[1]
                    if ti is not None:
                        effectors[f, ti, h_axes[0]] = lock_pos[0] + toe_delta[0]
                        effectors[f, ti, h_axes[1]] = lock_pos[1] + toe_delta[1]
                else:
                    if lock_pos is not None:
                        for k in range(release_n):
                            target_f = f + k
                            if target_f >= F:
                                break
                            blend = (k + 1) / (release_n + 1)
                            orig = horiz[target_f]
                            blended = lock_pos * (1.0 - blend) + orig * blend
                            effectors[target_f, fi, h_axes[0]] = blended[0]
                            effectors[target_f, fi, h_axes[1]] = blended[1]
                            if ti is not None:
                                effectors[target_f, ti, h_axes[0]] = blended[0] + toe_delta[0]
                                effectors[target_f, ti, h_axes[1]] = blended[1] + toe_delta[1]
                        lock_pos = None

        return planted_all

    # --------------------------------------------------------- lateral separation

    def _apply_min_lateral_separation(self, effectors: NDArray) -> None:
        cfg = self._config
        if cfg.min_lateral_separation <= 0.0 or not self._lateral_pairs:
            return
        la = self._lat
        min_dist = cfg.min_lateral_separation

        for f in range(effectors.shape[0]):
            for li, ri in self._lateral_pairs:
                ll = effectors[f, li, la]
                rl = effectors[f, ri, la]
                gap = ll - rl
                if gap < min_dist:
                    mid = (ll + rl) * 0.5
                    half = min_dist * 0.5
                    effectors[f, li, la] = mid + half
                    effectors[f, ri, la] = mid - half

    # --------------------------------------------------------- ground contact

    def _apply_ground_contact(self, effectors: NDArray) -> NDArray:
        """Lift uprightly-posed effector blocks so feet don't float.

        Returns ``(F,)`` the per-frame correction magnitude (0 when skipped).
        """
        cfg = self._config
        corrections = np.zeros(effectors.shape[0], dtype=np.float32)
        if cfg.ground_contact_z <= 0.0 or not self._foot_indices:
            return corrections

        ref_h = cfg.ground_contact_z
        max_correction = cfg.max_ground_correction
        blend_range = cfg.ground_uprightness_range
        up = self._up
        hips_idx = self._hips_idx

        for f in range(effectors.shape[0]):
            min_foot_h = min(effectors[f, idx, up] for idx in self._foot_indices)

            uprightness = 1.0
            if hips_idx >= 0:
                uprightness = float(np.clip(
                    (effectors[f, hips_idx, up] - min_foot_h) / max(blend_range, 1e-6),
                    0.0, 1.0,
                ))
            if uprightness < 0.01:
                continue

            excess = min_foot_h - ref_h
            if excess > 0.002:
                correction = min(excess, max_correction) * uprightness
                effectors[f, :, up] -= correction
                corrections[f] = -correction

        return corrections

    # --------------------------------------------------------- hand ground contact

    def _apply_hand_ground_contact(self, effectors: NDArray) -> None:
        """Pull torso/arms down so hand targets can touch the ground plane.

        Port of soma ``_enforce_hand_ground_contact``.  Populates
        :attr:`hand_ground_drop_per_frame` for dynamic Hips IK weight reduction.
        """
        cfg = self._config
        if cfg.hand_ground_contact_z <= 0.0 or not self._hand_chain_indices:
            self.hand_ground_drop_per_frame = np.zeros(effectors.shape[0], dtype=np.float32)
            return

        hand_ref = float(cfg.hand_ground_contact_z)
        approach_threshold = 0.20
        sh_pull_ratio = 0.70
        up = self._up
        h_axes = [self._fwd, self._lat]
        spine_idx = self._chest_idx

        self.hand_ground_drop_per_frame = np.zeros(effectors.shape[0], dtype=np.float32)

        for f in range(effectors.shape[0]):
            max_torso_drop = 0.0
            for hips_idx, sh_idx, fa_idx, h_idx in self._hand_chain_indices:
                hand_h = float(effectors[f, h_idx, up])
                hips_h = float(effectors[f, hips_idx, up])
                if hand_h >= approach_threshold or hand_h >= hips_h:
                    continue
                rough_deficit = hand_h - hand_ref
                if rough_deficit <= 0.01:
                    continue

                sh_h_orig = float(effectors[f, sh_idx, up])
                arm_vert = sh_h_orig - hand_h
                if arm_vert < 0.05:
                    continue

                sh_shift = rough_deficit * sh_pull_ratio
                effectors[f, sh_idx, up] -= np.float32(sh_shift)

                target_h = hand_ref
                if h_idx in self._arm_reach_map:
                    arm_sh_idx, max_reach = self._arm_reach_map[h_idx]
                    sh_pos = effectors[f, arm_sh_idx, 0:3].copy()
                    h_horiz = effectors[f, h_idx, h_axes]
                    sh_horiz = sh_pos[h_axes]
                    horiz_dist_sq = float(np.sum((h_horiz - sh_horiz) ** 2))
                    vert_budget_sq = max_reach ** 2 - horiz_dist_sq
                    if vert_budget_sq > 0.0:
                        min_h = float(sh_pos[up]) - math.sqrt(vert_budget_sq)
                        target_h = max(hand_ref, min_h)
                    else:
                        effectors[f, sh_idx, up] += np.float32(sh_shift)
                        continue

                actual_deficit = hand_h - target_h
                if actual_deficit <= 0.01:
                    effectors[f, sh_idx, up] += np.float32(sh_shift)
                    continue

                effectors[f, h_idx, up] = np.float32(target_h)
                fa_h = float(effectors[f, fa_idx, up])
                fa_ratio = (fa_h - hand_h) / arm_vert if arm_vert > 1e-6 else 0.0
                effectors[f, fa_idx, up] -= np.float32(actual_deficit * (1.0 - fa_ratio))

                needed_torso_drop = max(0.0, target_h - hand_ref + 0.02)
                if needed_torso_drop > max_torso_drop:
                    max_torso_drop = needed_torso_drop

            if max_torso_drop > 0.0:
                hips_idx = self._hand_chain_indices[0][0]
                effectors[f, hips_idx, up] -= np.float32(max_torso_drop * 0.8)
                if spine_idx >= 0:
                    effectors[f, spine_idx, up] -= np.float32(max_torso_drop * 0.5)
                self.hand_ground_drop_per_frame[f] = np.float32(max_torso_drop)

    # --------------------------------------------------------- body ground clearance

    def _apply_body_ground_clearance(self, effectors: NDArray) -> None:
        """Lift effectors when probe joints penetrate the ground plane.

        Adapted from soma-retargeter ``_enforce_body_ground_clearance``.  Runs
        after smoothing so foot-planting blends are not rate-limited twice.
        """
        cfg = self._config
        if not cfg.enable_body_ground_clearance or not self._body_probe_indices:
            return

        up = self._up
        floor = float(cfg.body_ground_plane_z) + float(cfg.body_ground_clearance)
        max_rate = float(cfg.body_ground_lift_max_rate)
        snap = bool(cfg.body_ground_snap_on_penetration)
        probes = self._body_probe_indices
        belows = self._body_probe_below

        for f in range(effectors.shape[0]):
            min_h = min(
                effectors[f, idx, up] - b for idx, b in zip(probes, belows)
            )
            lift_inst = max(0.0, floor - min_h)
            if snap and lift_inst > 1e-7:
                self._body_ground_lift_smoothed = lift_inst
            else:
                delta = lift_inst - self._body_ground_lift_smoothed
                if max_rate > 0.0:
                    if delta > max_rate:
                        delta = max_rate
                    elif delta < -max_rate:
                        delta = -max_rate
                self._body_ground_lift_smoothed += delta
            if abs(self._body_ground_lift_smoothed) > 1e-10:
                effectors[f, :, up] += self._body_ground_lift_smoothed

    # --------------------------------------------------------- smoothing

    def _smooth_corrections(
        self, constrained: NDArray, unconstrained: NDArray,
    ) -> int:
        """Rate-limit per-effector position corrections across frames.

        Args:
            constrained: ``(F, M, 7)`` effectors we just constrained.  Only
                ``[..., 0:3]`` is mutated; quats stay untouched.
            unconstrained: ``(F, M, 3)`` pre-constraint positions.

        Returns:
            Number of frames that had at least one DOF clamped.
        """
        max_rate = self._config.smoothing_max_rate
        if max_rate <= 0.0 or constrained.shape[0] < 2:
            return 0

        F = constrained.shape[0]
        corrections = constrained[:, :, 0:3] - unconstrained
        clamped_frames = 0

        for f in range(1, F):
            delta = corrections[f] - corrections[f - 1]
            magnitudes = np.linalg.norm(delta, axis=1, keepdims=True)
            exceeded = magnitudes > max_rate
            if not np.any(exceeded):
                continue
            clamped_frames += 1
            scale = np.where(
                exceeded, max_rate / np.maximum(magnitudes, 1e-8), 1.0,
            )
            corrections[f] = corrections[f - 1] + delta * scale

        constrained[:, :, 0:3] = unconstrained + corrections
        return clamped_frames

    # --------------------------------------------------------- name resolution

    def _resolve(self, attr: str, *, fallbacks: tuple[str, ...] = ()) -> int:
        name = getattr(self._config, attr, None)
        return self._resolve_joint(name, fallbacks=fallbacks)

    def _resolve_joint(self, name: str | None, *, fallbacks: tuple[str, ...] = ()) -> int:
        candidates: list[str] = []
        if name:
            candidates.append(str(name))
        candidates.extend(fallbacks)
        lower_map = {n.lower(): i for i, n in enumerate(self._joint_names)}
        for cand in candidates:
            if not cand:
                continue
            try:
                return self._joint_names.index(cand)
            except ValueError:
                pass
            idx = lower_map.get(str(cand).lower())
            if idx is not None:
                return idx
        return -1

    def _resolve_foot_indices(self) -> tuple[int, ...]:
        out = []
        for attr, fallbacks in (
            ("left_foot_name", ("LeftFoot", "left_foot", "LeftLeg")),
            ("right_foot_name", ("RightFoot", "right_foot", "RightLeg")),
        ):
            idx = self._resolve(attr, fallbacks=fallbacks)
            if idx >= 0:
                out.append(idx)
        return tuple(out)

    def _resolve_foot_toe_pairs(self) -> tuple[tuple[int, int | None], ...]:
        pairs = []
        for foot_attr, toe_attr, foot_fb, toe_fb in (
            ("left_foot_name", "left_toe_name", ("LeftFoot",), ("LeftToe", "LeftFootMod")),
            ("right_foot_name", "right_toe_name", ("RightFoot",), ("RightToe", "RightFootMod")),
        ):
            foot_idx = self._resolve(foot_attr, fallbacks=foot_fb)
            if foot_idx < 0:
                continue
            toe_name = getattr(self._config, toe_attr, None)
            toe_idx = self._resolve_joint(toe_name, fallbacks=toe_fb)
            pairs.append((foot_idx, toe_idx if toe_idx >= 0 else None))
        return tuple(pairs)

    def _resolve_body_probes(self) -> tuple[tuple[int, ...], tuple[float, ...]]:
        cfg = self._config
        if not cfg.enable_body_ground_clearance:
            return (), ()
        indices: list[int] = []
        belows: list[float] = []
        default_below = float(cfg.body_ground_default_probe_below)
        below_map = dict(cfg.body_ground_probe_below_meters or {})
        for jname in cfg.body_ground_probe_joints:
            idx = self._resolve_joint(jname)
            if idx < 0:
                continue
            indices.append(idx)
            belows.append(float(below_map.get(jname, default_below)))
        return tuple(indices), tuple(belows)

    def _resolve_hand_chains(self) -> tuple[tuple[int, int, int, int], ...]:
        """Per-side (hips, shoulder, forearm, hand) effector indices."""
        if self._hips_idx < 0:
            return ()
        chains: list[tuple[int, int, int, int]] = []
        for side in ("Left", "Right"):
            shoulder = self._resolve_joint(
                f"{side}Arm",
                fallbacks=(f"{side.lower()}_shoulder", f"{side}Shoulder"),
            )
            forearm = self._resolve_joint(
                f"{side}ForeArm",
                fallbacks=(f"{side.lower()}_elbow", f"{side}Elbow"),
            )
            hand = self._resolve_joint(
                f"{side}Hand",
                fallbacks=(f"{side.lower()}_wrist", f"{side}Wrist"),
            )
            if shoulder >= 0 and forearm >= 0 and hand >= 0:
                chains.append((self._hips_idx, shoulder, forearm, hand))
        return tuple(chains)

    def _resolve_arm_chains(self) -> tuple[tuple[int, tuple[int, ...], float], ...]:
        out: list[tuple[int, tuple[int, ...], float]] = []
        for spec in self._config.arm_chains:
            sh_idx = self._resolve_joint(spec.shoulder)
            if sh_idx < 0:
                continue
            chain_idx: list[int] = []
            for name in spec.chain:
                ji = self._resolve_joint(name)
                if ji >= 0:
                    chain_idx.append(ji)
            if chain_idx:
                out.append((sh_idx, tuple(chain_idx), float(spec.max_reach)))
        return tuple(out)

    def _build_arm_reach_map(self) -> dict[int, tuple[int, float]]:
        reach: dict[int, tuple[int, float]] = {}
        for sh_idx, chain_idx, max_reach in self._arm_chains:
            if chain_idx:
                reach[chain_idx[-1]] = (sh_idx, max_reach)
        return reach

    def _resolve_lateral_pairs(self) -> tuple[tuple[int, int], ...]:
        pairs: list[tuple[int, int]] = []
        for left, right in self._config.lateral_pairs:
            li = self._joint_names.index(left) if left in self._joint_names else -1
            ri = self._joint_names.index(right) if right in self._joint_names else -1
            if li >= 0 and ri >= 0:
                pairs.append((li, ri))
        if not pairs and len(self._foot_indices) >= 2:
            pairs.append((self._foot_indices[0], self._foot_indices[1]))
        return tuple(pairs)
