# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared output data class for all retarget backends (Newton, interaction mesh)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["IKTargetTrace", "RetargetedMotion"]


@dataclass(frozen=True)
class IKTargetTrace:
    """Position targets the IK solver actually tracked, one row per output frame.

    ``positions[f, i]`` is the world-space target for the point
    ``link_offsets[i]`` expressed in ``t_body_links[i]``'s frame, in the same
    heading-aligned, floor-referenced world frame as
    :attr:`RetargetedMotion.joint_q`.  Result
    diagnostics compare robot links against these instead of the uniformly
    scaled yellow overlay, whose limb proportions and link-origin placement
    differ from the per-joint scaled IK targets.
    """

    canonical_names: tuple[str, ...]
    t_body_links: tuple[str, ...]
    link_offsets: NDArray
    positions: NDArray


@dataclass
class RetargetedMotion:
    """Output bundle for a single retargeted clip.

    Layout of :attr:`joint_q`: ``(F, root_coord_count + actuated_dof_count)``.
    The first 7 columns are ``(tx, ty, tz, qx, qy, qz, qw)`` — Newton's
    floating-base root joint — matching the hhtools CSV schema.

    :attr:`ik_targets` is set by backends that solve against explicit point
    targets (Newton IK) and must be dropped when ``joint_q`` is resampled.
    """

    name: str
    joint_q: NDArray
    sample_rate: float
    dof_names: tuple[str, ...]
    root_coord_count: int = 7
    meta: dict = field(default_factory=dict)
    ik_targets: IKTargetTrace | None = None

    @property
    def num_frames(self) -> int:
        return int(self.joint_q.shape[0])

    @property
    def root_trajectory(self) -> NDArray:
        """``(F, 7)`` slice: floating-base root (xyz + xyzw quat)."""
        return self.joint_q[:, : self.root_coord_count]

    @property
    def dof_trajectory(self) -> NDArray:
        """``(F, D)`` slice: actuated DOFs (post root)."""
        return self.joint_q[:, self.root_coord_count :]
