# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Clip-wide foot floor snap for retargeted trajectories.

After Newton IK or Interaction-Mesh MPC, the floating base can sit a few
centimetres above ``z = 0`` even when the scaled source looks planted: IK
tracks ankle frames, not sole meshes, and foot geometry differs from the
human.  :func:`snap_joint_q_clip_floor` applies one global root-Z translation
so the **lowest foot sole among upright frames** (or all frames if none) sits
on ``ground_z`` while preserving relative jumps and steps.

Lie→stand clips must ignore prone-frame sole penetration when choosing the
snap reference — otherwise lifting those frames floats the standing feet.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from hhtools.robot.foot_geometry import quat_xyzw_to_rotmat

if TYPE_CHECKING:
    from hhtools.robot.loader import URDFRobotModel

_log = logging.getLogger(__name__)

# Pelvis/root above sole (m) to count as upright for snap reference selection.
_UPRIGHT_ROOT_ABOVE_SOLE_M = 0.25

__all__ = [
    "measure_clip_min_foot_world_z",
    "snap_joint_q_clip_floor",
]


def _root_transform(root_xyzw: NDArray) -> NDArray[np.float64]:
    root = np.asarray(root_xyzw, dtype=np.float64).reshape(-1)
    T = np.eye(4, dtype=np.float64)
    if root.size < 7:
        return T
    T[:3, 3] = root[:3]
    T[:3, :3] = quat_xyzw_to_rotmat(root[3:7])
    return T


def _foot_link_parts(robot: "URDFRobotModel") -> list[tuple[str, tuple]]:
    from hhtools.robot.foot_geometry import (
        _foot_contact_links,
        _foot_mesh_node_parts,
    )

    left, right = _foot_contact_links(robot)
    out: list[tuple[str, tuple]] = []
    for link in (left, right):
        if not link:
            continue
        parts = _foot_mesh_node_parts(robot, link)
        if parts:
            out.append((link, parts))
    return out


def _frame_min_foot_world_z(
    robot: "URDFRobotModel",
    root7: NDArray,
    *,
    foot_parts: list[tuple[str, tuple]],
) -> float | None:
    """Lowest foot sole world-Z for the current robot configuration."""
    from hhtools.robot.foot_geometry import _cached_geom_vertices

    scene = robot.urdf.scene
    Tw = _root_transform(root7)
    zs: list[float] = []

    for link, parts in foot_parts:
        for node, geom_name in parts:
            v = _cached_geom_vertices(robot, geom_name)
            if v is None:
                continue
            mat = scene.graph.get(node)[0]
            ones = np.ones((v.shape[0], 1), dtype=np.float64)
            w = (Tw @ mat @ np.c_[v, ones].T).T[:, :3]
            zs.append(float(w[:, 2].min()))
        if not parts:
            try:
                Tl, _ = scene.graph.get(link)
            except Exception:
                try:
                    Tl = scene.graph.get(frame_to=link)[0]
                except Exception:
                    continue
            zs.append(float((Tw @ Tl)[2, 3]))

    return min(zs) if zs else None


def measure_clip_min_foot_world_z(
    robot: "URDFRobotModel",
    joint_q: NDArray,
    *,
    root_coord_count: int = 7,
    upright_only: bool = False,
    upright_root_above_sole_m: float = _UPRIGHT_ROOT_ABOVE_SOLE_M,
) -> float | None:
    """Minimum foot-sole world Z over frames of ``joint_q``.

    When ``upright_only`` is true, only frames whose root sits at least
    ``upright_root_above_sole_m`` above the sole are considered.  If no such
    frame exists, falls back to the full-clip minimum.
    """
    q = np.asarray(joint_q, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] < root_coord_count:
        return None

    foot_parts = _foot_link_parts(robot)
    if not foot_parts:
        _log.warning(
            "clip floor snap: robot %r has no foot meshes/links; skip measure",
            getattr(getattr(robot, "preset", None), "name", "?"),
        )
        return None

    dof_names = robot.dof_names()
    n_dof = min(len(dof_names), q.shape[1] - root_coord_count)
    saved = robot.zero_configuration()
    min_all = float("inf")
    min_upright = float("inf")
    try:
        for f in range(q.shape[0]):
            if n_dof > 0:
                cfg = {
                    dof_names[i]: float(q[f, root_coord_count + i])
                    for i in range(n_dof)
                }
                robot.apply_configuration(cfg)
            else:
                robot.apply_configuration(saved)
            z = _frame_min_foot_world_z(
                robot, q[f, :root_coord_count], foot_parts=foot_parts,
            )
            if z is None:
                continue
            if z < min_all:
                min_all = z
            root_z = float(q[f, 2])
            if root_z - float(z) >= float(upright_root_above_sole_m):
                if z < min_upright:
                    min_upright = z
    finally:
        robot.apply_configuration(saved)

    if upright_only and np.isfinite(min_upright):
        return float(min_upright)
    if np.isfinite(min_all):
        return float(min_all)
    return None


def snap_joint_q_clip_floor(
    robot: "URDFRobotModel",
    joint_q: NDArray,
    *,
    root_coord_count: int = 7,
    ground_z: float = 0.0,
    z_index: int = 2,
) -> tuple[NDArray, float]:
    """Translate root Z so the reference foot sole sits on ``ground_z``.

    Prefers the lowest sole among **upright** frames (root clearly above the
    feet).  Fully prone clips fall back to the all-frame minimum.  Relative
    motion (jumps, steps) is preserved — only a constant root-Z shift.

    Returns ``(joint_q_out, delta_z)`` where ``delta_z`` was subtracted from the
    root height column (``out[:, z_index] = in[:, z_index] - delta_z``).
    """
    q = np.asarray(joint_q)
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] <= z_index:
        return q, 0.0

    min_z = measure_clip_min_foot_world_z(
        robot, q, root_coord_count=root_coord_count, upright_only=True,
    )
    if min_z is None:
        return q, 0.0

    delta = float(min_z) - float(ground_z)
    if abs(delta) < 1e-4:
        return q, 0.0

    out = q.astype(np.float32, copy=True)
    out[:, z_index] = out[:, z_index] - np.float32(delta)
    _log.info(
        "clip floor snap: Δz=%+.4fm so upright min foot sole (was %.4fm) "
        "sits on z=%.4fm",
        delta,
        min_z,
        ground_z,
    )
    return out, delta
