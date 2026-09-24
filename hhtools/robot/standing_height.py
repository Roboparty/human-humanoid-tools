"""Standing height from a robot URDF's visual / collision geometry.

The yellow skeleton overlay and :class:`~hhtools.retarget.newton_basic.config.ScalerConfig`
share one definition of robot stature: the vertical span of every loaded
visual mesh, collision mesh, and (when available) MuJoCo collision geom at
the pose being calibrated / previewed. This matches the Web Stage robot after
ground alignment (``max_z − min_z`` over rendered meshes, feet on ``z = 0``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hhtools.robot.loader import URDFRobotModel

_log = logging.getLogger(__name__)

__all__ = ["estimate_robot_standing_height"]


def _trimesh_scene_z_bounds(scene) -> tuple[float, float] | None:
    """Return ``(min_z, max_z)`` over all mesh vertices in a trimesh scene."""
    import trimesh

    min_z: float | None = None
    max_z: float | None = None
    for _node in scene.graph.nodes_geometry:
        mat, geom_name = scene.graph[_node]
        if geom_name is None:
            continue
        geom = scene.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh) or geom.is_empty:
            continue
        v = np.asarray(geom.vertices, dtype=np.float64)
        z = (
            mat[2, 0] * v[:, 0]
            + mat[2, 1] * v[:, 1]
            + mat[2, 2] * v[:, 2]
            + mat[2, 3]
        )
        zmin = float(z.min())
        zmax = float(z.max())
        min_z = zmin if min_z is None else min(min_z, zmin)
        max_z = zmax if max_z is None else max(max_z, zmax)
    if min_z is None or max_z is None:
        return None
    return min_z, max_z


def _set_mujoco_joint_q(mj_model, mj_data, joint_q: dict[str, float]) -> None:
    """Write hinge / slide joint values into ``mj_data.qpos``."""
    import mujoco

    for jidx in range(int(mj_model.njnt)):
        jtype = int(mj_model.jnt_type[jidx])
        if jtype not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            continue
        name = mj_model.joint(jidx).name
        if name not in joint_q:
            continue
        adr = int(mj_model.jnt_qposadr[jidx])
        mj_data.qpos[adr] = float(joint_q[name])


def _mujoco_geom_z_bounds(mj_model, mj_data) -> tuple[float, float] | None:
    """Approximate world ``z`` bounds from MuJoCo collision geoms."""
    import mujoco

    mujoco.mj_forward(mj_model, mj_data)
    min_z: float | None = None
    max_z: float | None = None
    for gid in range(int(mj_model.ngeom)):
        pos = mj_data.geom_xpos[gid]
        r = float(mj_model.geom_rbound[gid])
        if not np.isfinite(r) or r <= 0.0:
            continue
        zc = float(pos[2])
        zlo = zc - r
        zhi = zc + r
        min_z = zlo if min_z is None else min(min_z, zlo)
        max_z = zhi if max_z is None else max(max_z, zhi)
    if min_z is None or max_z is None:
        return None
    return min_z, max_z


def estimate_robot_standing_height(
    model: "URDFRobotModel",
    joint_q: dict[str, float] | None = None,
) -> float:
    """Ground-to-apex standing height (metres) from URDF visual + collision meshes.

    At ``joint_q`` (defaults to the all-zero T-pose), union the axis-aligned
    ``z`` bounds of:

    * every **visual** mesh in the yourdfpy scene (what the viewer draws);
    * every **collision** mesh when a collision scene graph is available.

    MuJoCo **geom** bounding-sphere bounds are used only as a *fallback* when
    no mesh geometry is available: bounding spheres add their radius past the
    real mesh extent and over-estimate stature by 10-15%, which would inflate
    the yellow overlay's ``model_height / human_height`` uniform scale.

    Returns ``max_z − min_z`` over that union, scaled by
    ``preset.length_scale``. This is the same vertical span used by the Web
    Stage after lifting the lowest visual vertex to ``z = 0``.
    """
    q = dict(joint_q) if joint_q is not None else model.zero_configuration()
    saved_q = model.zero_configuration()
    # Exact mesh-vertex bounds (visual / collision) are the source of truth.
    # MuJoCo geom bounds use per-geom *bounding spheres*, which add the sphere
    # radius above the head and below the feet (over-estimating stature by
    # 10-15%).  An inflated ``model_height`` enlarges the yellow overlay's
    # ``model_height / human_height`` uniform scale and floats the scaled
    # shoulders up to the robot's head.  Only fall back to MuJoCo bounds when
    # no mesh geometry is available (primitive-only / mesh-load failure).
    mesh_bounds: list[tuple[float, float]] = []
    mj_bounds: list[tuple[float, float]] = []

    try:
        model.apply_configuration(q)

        try:
            b = _trimesh_scene_z_bounds(model.trimesh_scene(collision=False))
            if b is not None:
                mesh_bounds.append(b)
        except Exception as exc:
            _log.debug("visual mesh z-bounds failed for %r: %s", model.preset.name, exc)

        try:
            coll_scene = model.urdf.collision_scene
            if coll_scene is not None:
                b = _trimesh_scene_z_bounds(coll_scene)
                if b is not None:
                    mesh_bounds.append(b)
        except Exception as exc:
            _log.debug("collision mesh z-bounds failed for %r: %s", model.preset.name, exc)

        if not mesh_bounds:
            try:
                from hhtools.retarget.interaction_mesh.mujoco_scene import (
                    require_mujoco_model,
                )

                mj_model = require_mujoco_model(model)
                import mujoco

                mj_data = mujoco.MjData(mj_model)
                _set_mujoco_joint_q(mj_model, mj_data, q)
                b = _mujoco_geom_z_bounds(mj_model, mj_data)
                if b is not None:
                    mj_bounds.append(b)
            except Exception as exc:
                _log.debug("MuJoCo geom z-bounds failed for %r: %s", model.preset.name, exc)

    finally:
        model.apply_configuration(saved_q)

    bounds = mesh_bounds if mesh_bounds else mj_bounds
    if not bounds:
        _log.warning(
            "No URDF / MuJoCo geometry for robot %r — falling back to 1.3 m",
            model.preset.name,
        )
        return max(1e-3, 1.3 * float(model.preset.length_scale))

    min_z = min(b[0] for b in bounds)
    max_z = max(b[1] for b in bounds)
    span = max(1e-3, max_z - min_z)
    scale = float(model.preset.length_scale)
    if abs(scale - 1.0) > 1e-9:
        span *= scale
    return span
