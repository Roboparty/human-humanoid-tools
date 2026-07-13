# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""OBJ → :class:`hhtools.core.scene.TerrainHeightfield` rasterisation.

Pure rasterisation pipeline.  Loads a terrain ``*.obj`` mesh, optionally
applies the rigid + uniform-scale transform chain used by the rest of the
retargeter (object_pos / object_quat → world_rotation → z_offset → scale),
and rasterises the result onto a regular 2-D grid suitable for storage in
PARC's ``MSTerrainData`` schema.

The resulting :class:`TerrainHeightfield` is the canonical terrain record
throughout the hhtools pipeline.  Its data layout (``hf[ix, iy]`` with
``min_point`` as world XY origin and isotropic ``dx``) matches PARC's
``SubTerrain.from_ms_terrain_data`` 1:1 so produced terrain pickles are
PARC-trainable without further conversion.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hhtools.core.scene import TerrainHeightfield

_log = logging.getLogger(__name__)


def _rasterise_triangles_to_grid(
    verts: NDArray[np.floating],
    faces: NDArray[np.integer],
    *,
    xmin: float,
    ymin: float,
    nx: int,
    ny: int,
    dx: float,
    fill_value: float,
) -> NDArray[np.float32]:
    """Maximum-z triangle rasterisation onto a regular grid.

    For each triangle, walks the AABB cells, evaluates barycentric
    coordinates and updates ``grid[ix, iy]`` to the maximum of its current
    value and the per-cell triangle-plane height.  Cells not touched by any
    triangle keep ``fill_value`` — typically the global ``z_min`` of the
    mesh, so the heightfield extrapolates beyond the OBJ footprint as flat
    "ground floor" rather than as a bottomless pit.
    """
    grid = np.full((nx, ny), fill_value, dtype=np.float64)
    inv_dx = 1.0 / max(dx, 1e-9)
    for fi in range(len(faces)):
        tri = verts[faces[fi]]
        txmin = float(tri[:, 0].min())
        txmax = float(tri[:, 0].max())
        tymin = float(tri[:, 1].min())
        tymax = float(tri[:, 1].max())

        ix0 = max(int((txmin - xmin) * inv_dx), 0)
        ix1 = min(int((txmax - xmin) * inv_dx) + 1, nx - 1)
        iy0 = max(int((tymin - ymin) * inv_dx), 0)
        iy1 = min(int((tymax - ymin) * inv_dx) + 1, ny - 1)

        v0, v1, v2 = tri[0], tri[1], tri[2]
        e1 = v1 - v0
        e2 = v2 - v0
        n_cross = float(e1[0] * e2[1] - e1[1] * e2[0])
        degen = abs(n_cross) < 1e-12

        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                px = xmin + ix * dx
                py = ymin + iy * dx
                if degen:
                    z_val = float(tri[:, 2].max())
                else:
                    dpx = px - v0[0]
                    dpy = py - v0[1]
                    u = (dpx * e2[1] - dpy * e2[0]) / n_cross
                    v = (e1[0] * dpy - e1[1] * dpx) / n_cross
                    if u >= -0.01 and v >= -0.01 and (u + v) <= 1.01:
                        z_val = float(v0[2] + u * e1[2] + v * e2[2])
                    else:
                        continue
                if z_val > grid[ix, iy]:
                    grid[ix, iy] = z_val

    return grid.astype(np.float32, copy=False)


def obj_to_heightfield(
    obj_path: str | Path,
    *,
    dx: float = 0.05,
    padding: float = 0.5,
    object_position: NDArray[np.floating] | None = None,
    object_quat_xyzw: NDArray[np.floating] | None = None,
    mesh_scale: float = 1.0,
    rotation_quat_xyzw: NDArray[np.floating] | None = None,
    z_offset: float = 0.0,
    scale: float = 1.0,
    z_buffer: float = 3.0,
    fill_value: float | None = None,
) -> TerrainHeightfield:
    """Rasterise a terrain OBJ into a :class:`TerrainHeightfield`.

    The transform chain matches the rest of the retargeter so the
    resulting grid lives in the **source-motion world frame**:

        v' = scale * (R_global * (R_obj * v * mesh_scale + obj_pos) - [0,0,z_offset])

    This is the source-frame heightfield; downstream consumers
    (MPC-SQP collision, PARC export for retargeted clips) call
    :meth:`TerrainHeightfield.scaled` with the appropriate
    ``smpl_scale`` to get a robot-frame copy.

    Parameters
    ----------
    obj_path
        Path to the ``*_terrain.obj`` (or ``terrain.obj``) file.
    dx
        Cell size in metres.  Default ``0.05`` is fine enough to resolve
        a robot foot vs. a stair edge while still cheap to store.
    padding
        Extra metres of flat ``fill_value`` ground added on each side of
        the mesh's XY bbox.
    object_position, object_quat_xyzw, mesh_scale
        Pose taken from a single :class:`SceneObject` frame
        (``positions[0]`` / ``quaternions[0]`` / ``scale``).
    rotation_quat_xyzw
        Optional additional global rotation (e.g. ``source_body_quat``
        applied during scaling).
    z_offset, scale
        ``z_world = (z_local - z_offset) * scale``; ``xy_world *= scale``.
    z_buffer
        Vertical buffer used to seed ``hf_maxmin`` per-cell bounds.
    fill_value
        Height for cells outside any triangle's AABB; default is the
        global ``z_min`` of the rasterised vertices.
    """
    import trimesh
    from scipy.spatial.transform import Rotation

    src = Path(obj_path)
    if not src.is_file():
        raise FileNotFoundError(f"terrain OBJ not found: {src}")

    mesh = trimesh.load(str(src), force="mesh", process=False)
    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError(f"empty terrain mesh: {src}")

    verts = np.asarray(mesh.vertices, dtype=np.float64) * float(mesh_scale)
    if object_quat_xyzw is not None:
        q_obj = np.asarray(object_quat_xyzw, dtype=np.float64)
        if not np.allclose(q_obj, [0, 0, 0, 1], atol=1e-7):
            verts = Rotation.from_quat(q_obj).apply(verts)
    if object_position is not None:
        verts = verts + np.asarray(object_position, dtype=np.float64).reshape(1, 3)

    if rotation_quat_xyzw is not None:
        q = np.asarray(rotation_quat_xyzw, dtype=np.float64)
        if not np.allclose(q, [0, 0, 0, 1], atol=1e-7):
            verts = Rotation.from_quat(q).apply(verts)
    verts[:, 2] -= float(z_offset)
    verts *= float(scale)

    faces = np.asarray(mesh.faces, dtype=np.intp)

    xmin = float(verts[:, 0].min()) - float(padding)
    ymin = float(verts[:, 1].min()) - float(padding)
    xmax = float(verts[:, 0].max()) + float(padding)
    ymax = float(verts[:, 1].max()) + float(padding)
    z_min_g = float(verts[:, 2].min())
    z_max_g = float(verts[:, 2].max())

    nx = max(int(np.ceil((xmax - xmin) / dx)) + 1, 2)
    ny = max(int(np.ceil((ymax - ymin) / dx)) + 1, 2)
    nx = min(nx, 4096)
    ny = min(ny, 4096)

    fill = float(z_min_g) if fill_value is None else float(fill_value)
    hf = _rasterise_triangles_to_grid(
        verts, faces,
        xmin=xmin, ymin=ymin, nx=nx, ny=ny, dx=float(dx), fill_value=fill,
    )

    hf_maxmin = np.zeros((nx, ny, 2), dtype=np.float32)
    hf_maxmin[..., 0] = z_max_g + float(z_buffer)
    hf_maxmin[..., 1] = z_min_g - float(z_buffer)

    _log.info(
        "OBJ → heightfield (%s): %dx%d grid, dx=%.3fm, "
        "z-range [%.3f, %.3f], min_point=(%.3f, %.3f)",
        src.name, nx, ny, float(dx),
        float(hf.min()), float(hf.max()), xmin, ymin,
    )

    return TerrainHeightfield(
        hf=hf,
        hf_maxmin=hf_maxmin,
        min_point=np.array([xmin, ymin], dtype=np.float32),
        dx=float(dx),
    )


def posed_meshes_to_heightfield(
    meshes: list[
        tuple[
            str | Path,
            NDArray[np.floating] | None,
            NDArray[np.floating] | None,
            float,
        ]
    ],
    *,
    dx: float = 0.05,
    padding: float = 0.5,
    z_buffer: float = 3.0,
    fill_value: float | None = None,
) -> TerrainHeightfield:
    """Rasterise one or more posed OBJs onto a single max-z heightfield.

    Each entry is ``(obj_path, object_position, object_quat_xyzw, mesh_scale)``.
    Transforms match :func:`obj_to_heightfield` so multi-obstacle parkour clips
    land in the same world frame as a single-mesh bake.
    """
    import trimesh
    from scipy.spatial.transform import Rotation

    if not meshes:
        raise ValueError("posed_meshes_to_heightfield requires at least one mesh")

    all_verts: list[NDArray[np.floating]] = []
    all_faces: list[NDArray[np.integer]] = []
    vert_offset = 0
    for obj_path, object_position, object_quat_xyzw, mesh_scale in meshes:
        src = Path(obj_path)
        if not src.is_file():
            raise FileNotFoundError(f"terrain OBJ not found: {src}")
        mesh = trimesh.load(str(src), force="mesh", process=False)
        if mesh is None or len(mesh.vertices) == 0:
            raise ValueError(f"empty terrain mesh: {src}")
        verts = np.asarray(mesh.vertices, dtype=np.float64) * float(mesh_scale)
        if object_quat_xyzw is not None:
            q_obj = np.asarray(object_quat_xyzw, dtype=np.float64)
            if not np.allclose(q_obj, [0, 0, 0, 1], atol=1e-7):
                verts = Rotation.from_quat(q_obj).apply(verts)
        if object_position is not None:
            verts = verts + np.asarray(object_position, dtype=np.float64).reshape(1, 3)
        faces = np.asarray(mesh.faces, dtype=np.intp) + vert_offset
        all_verts.append(verts)
        all_faces.append(faces)
        vert_offset += int(verts.shape[0])

    verts_cat = np.concatenate(all_verts, axis=0)
    faces_cat = np.concatenate(all_faces, axis=0)

    xmin = float(verts_cat[:, 0].min()) - float(padding)
    ymin = float(verts_cat[:, 1].min()) - float(padding)
    xmax = float(verts_cat[:, 0].max()) + float(padding)
    ymax = float(verts_cat[:, 1].max()) + float(padding)
    z_min_g = float(verts_cat[:, 2].min())
    z_max_g = float(verts_cat[:, 2].max())

    nx = max(int(np.ceil((xmax - xmin) / dx)) + 1, 2)
    ny = max(int(np.ceil((ymax - ymin) / dx)) + 1, 2)
    nx = min(nx, 4096)
    ny = min(ny, 4096)

    fill = float(z_min_g) if fill_value is None else float(fill_value)
    hf = _rasterise_triangles_to_grid(
        verts_cat,
        faces_cat,
        xmin=xmin,
        ymin=ymin,
        nx=nx,
        ny=ny,
        dx=float(dx),
        fill_value=fill,
    )

    # Thin planks (balance beams ~15 cm) often underfill the XY AABB because
    # slanted / sparse triangles miss neighbouring cells.  A 1-cell max
    # dilate restores roughly the convex footprint without changing tall
    # obstacle heights (max of existing values).
    if dx > 0.0:
        from scipy.ndimage import maximum_filter

        hf = maximum_filter(hf, size=3)

    hf_maxmin = np.zeros((nx, ny, 2), dtype=np.float32)
    hf_maxmin[..., 0] = z_max_g + float(z_buffer)
    hf_maxmin[..., 1] = z_min_g - float(z_buffer)

    _log.info(
        "posed meshes → heightfield (%d mesh(es)): %dx%d grid, dx=%.3fm, "
        "z-range [%.3f, %.3f], min_point=(%.3f, %.3f)",
        len(meshes),
        nx,
        ny,
        float(dx),
        float(hf.min()),
        float(hf.max()),
        xmin,
        ymin,
    )

    return TerrainHeightfield(
        hf=hf,
        hf_maxmin=hf_maxmin,
        min_point=np.array([xmin, ymin], dtype=np.float32),
        dx=float(dx),
    )


__all__ = [
    "obj_to_heightfield",
    "posed_meshes_to_heightfield",
]
