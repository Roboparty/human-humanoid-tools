"""Per-DOF joint-limit clamper (pure NumPy, CPU).

Stage-1 port of ``soma_retargeter.pipelines.joint_limit_clamper`` rewritten
against the hhtools :class:`~hhtools.robot.base.RobotModel` contract.  No
Warp / Newton dependency — we consume the plain ``JointInfo.limit_lower /
limit_upper`` fields that :mod:`hhtools.robot.loader` already exposes so the
module is trivially unit-testable on CI machines without a GPU.

Original implementation: soma-retargeter, licensed under Apache-2.0.
  https://github.com/NVIDIA/SOMA-Retargeter
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.

Key differences from the original:

* Works on ``np.ndarray`` batches of shape ``(D,)``, ``(F, D)`` or
  ``(F, E, D)`` — the original only accepted Warp 2-D arrays.
* Joints with ``limit_lower is None`` *or* ``limit_upper is None`` are treated
  as *unlimited* (continuous revolute / unset prismatic).  The original
  assumed every DOF had a finite limit on both sides.
* No multi-DOF joint handling.  hhtools CSV schema emits one scalar per
  actuated joint — if we ever support 3-DOF ball joints we'll revisit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hhtools.robot.base import RobotModel


__all__ = ["JointLimitClamper"]


@dataclass(frozen=True)
class _Limits:
    """Pre-extracted ``(D,)`` limit arrays + a validity mask.

    ``mask[d] == True`` means DOF ``d`` has a *finite* clamp to apply; unclamped
    axes are left untouched so continuous joints don't get pinned at ``np.inf``.
    """

    lower: NDArray
    upper: NDArray
    mask: NDArray


class JointLimitClamper:
    """Clamp joint configurations to the URDF-declared DOF limits.

    Example:

        >>> clamper = JointLimitClamper.from_robot(robot)
        >>> q = np.random.uniform(-5.0, 5.0, size=(num_frames, robot.num_dofs))
        >>> q_safe = clamper.apply(q)  # new array; input unchanged
    """

    def __init__(self, lower: NDArray, upper: NDArray) -> None:
        lower = np.asarray(lower, dtype=np.float32)
        upper = np.asarray(upper, dtype=np.float32)
        if lower.shape != upper.shape:
            raise ValueError(
                f"lower / upper shape mismatch: {lower.shape} vs {upper.shape}"
            )
        if lower.ndim != 1:
            raise ValueError(f"limits must be 1-D (D,); got shape {lower.shape}")

        mask_low = np.isfinite(lower)
        mask_high = np.isfinite(upper)
        mask = mask_low & mask_high
        if np.any(mask & (lower > upper)):
            bad = int(np.argmax(mask & (lower > upper)))
            raise ValueError(
                f"lower > upper at DOF {bad}: {float(lower[bad])} > {float(upper[bad])}"
            )

        self._limits = _Limits(
            lower=lower.astype(np.float32, copy=False),
            upper=upper.astype(np.float32, copy=False),
            mask=mask.astype(np.bool_, copy=False),
        )

    # --------------------------------------------------------------- constructors

    @classmethod
    def from_robot(cls, robot: RobotModel) -> "JointLimitClamper":
        """Build a clamper from a loaded :class:`RobotModel`.

        Joints whose URDF omits ``<limit>`` (``None`` on either side) stay
        unclamped — which matches how MuJoCo treats ``range`` being absent.
        """
        joints = robot.actuated_joints
        n = len(joints)
        lower = np.full(n, -np.inf, dtype=np.float32)
        upper = np.full(n, +np.inf, dtype=np.float32)
        for i, j in enumerate(joints):
            if j.limit_lower is not None:
                lower[i] = float(j.limit_lower)
            if j.limit_upper is not None:
                upper[i] = float(j.limit_upper)
        return cls(lower, upper)

    # ------------------------------------------------------------------ queries

    @property
    def num_dofs(self) -> int:
        return int(self._limits.lower.shape[0])

    @property
    def lower(self) -> NDArray:
        return self._limits.lower

    @property
    def upper(self) -> NDArray:
        return self._limits.upper

    @property
    def finite_mask(self) -> NDArray:
        """Boolean ``(D,)`` array — ``True`` where both limits are finite."""
        return self._limits.mask

    # ------------------------------------------------------------------ apply

    def apply(self, joint_q: NDArray, *, in_place: bool = False) -> NDArray:
        """Clamp ``joint_q`` along its last axis.

        Args:
            joint_q: Array with last dim equal to ``num_dofs``.  Any leading
                shape is allowed (e.g. ``(D,)``, ``(F, D)``, ``(E, F, D)``).
            in_place: Whether to mutate ``joint_q`` directly (returns the same
                object) or allocate a copy (default — matches the original
                ``soma`` semantics of returning the clamped array).

        Returns:
            Clamped array with the same shape / dtype as ``joint_q``.

        Raises:
            ValueError: If the last-axis dimension doesn't match ``num_dofs``.
        """
        arr = np.asarray(joint_q)
        if arr.shape[-1] != self.num_dofs:
            raise ValueError(
                f"joint_q last-axis size {arr.shape[-1]} does not match "
                f"num_dofs={self.num_dofs}"
            )

        out = arr if in_place else arr.copy()
        # np.clip broadcasts across leading dims; unlimited DOFs keep -inf/+inf
        # which leaves their values untouched.
        np.clip(out, self._limits.lower, self._limits.upper, out=out)
        return out

    # ------------------------------------------------------------------ dunder

    def __repr__(self) -> str:  # pragma: no cover — trivial
        finite = int(self._limits.mask.sum())
        return (
            f"<JointLimitClamper dofs={self.num_dofs} "
            f"finite={finite}/{self.num_dofs}>"
        )
