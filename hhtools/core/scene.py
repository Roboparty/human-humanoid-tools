"""Scene objects carried alongside a :class:`Motion`.

Some datasets (OMOMO, BEHAVE, GRAB ...) pair a human with one or more physical objects whose
6-DoF trajectory is also recorded. ``SceneObject`` is the minimal per-object record we carry
through the pipeline: a name, per-frame translation, per-frame rotation (xyzw), a coarse
cuboid "extent" used as a placeholder when no real mesh is available, and optionally a path
to a triangle mesh plus a uniform scale to apply to that mesh at render time.

This is deliberately a *thin* record — it carries just enough for the viewer to render a box
(or a mesh when ``mesh_path`` is set) and for analytics code to read object kinematics.
Heavy mesh IO lives outside the Motion object; the :class:`ObjectsRenderer` loads the mesh
lazily via ``trimesh`` when the handle is first materialised.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class SceneObject:
    """One rigid object's 6-DoF trajectory plus optional mesh metadata.

    Attributes:
        name: Dataset-specific identifier (e.g. ``"largebox"``, ``"mop"``). Used by the viewer
            and analytics for labelling.
        positions: ``(num_frames, 3)`` float32 world-frame translation per frame, in metres.
        quaternions: ``(num_frames, 4)`` float32 xyzw rotations per frame, unit norm.
        extents: ``(3,)`` float32 full-side cuboid dimensions (width, depth, height) in metres.
            Used as a placeholder box when a real mesh is unavailable.
        mesh_path: Optional path (absolute, resolved at load time) to a triangle mesh
            (OBJ / STL / PLY). Empty string means "no mesh, use ``extents`` placeholder".
        scale: Uniform scale factor applied to the mesh's vertices at render time. OMOMO
            captures a per-frame ``obj_scale`` which is effectively constant within a clip;
            we collapse it to a single float here. Defaults to 1.0 so callers without a
            dataset-provided scale (e.g. BEHAVE) get the mesh's native units.
        opacity: Optional per-object render opacity in [0, 1]. ``None`` means "fall back to
            the viewer's global object opacity" (0.55 today, keeping OMOMO's see-through
            behaviour by default). Dataset adapters producing **large, non-occluding scene
            geometry** — terrain meshes, floors, static environments — should set this
            explicitly to ``1.0`` so the ground plane stops looking like frosted glass.
        color: Optional per-object RGB colour tuple ``(r, g, b)`` in 0-255. ``None`` means
            "use the viewer's default object colour". Let adapters pick a semantically
            meaningful hue for large scene geometry (e.g. a neutral slate grey for rock
            terrain) without having to plumb through the global palette.
    """

    name: str
    positions: NDArray
    quaternions: NDArray
    extents: NDArray = field(default_factory=lambda: np.array([0.3, 0.3, 0.3], dtype=np.float32))
    mesh_path: str = ""
    scale: float = 1.0
    opacity: float | None = None
    color: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float32)
        self.quaternions = np.asarray(self.quaternions, dtype=np.float32)
        self.extents = np.asarray(self.extents, dtype=np.float32).reshape(3)
        self.scale = float(self.scale)
        if self.opacity is not None:
            self.opacity = float(min(1.0, max(0.0, self.opacity)))
        if self.color is not None:
            # Accept any iterable of three ints, clip to 0-255 for Viser safety.
            rgb = tuple(int(max(0, min(255, int(c)))) for c in self.color)
            if len(rgb) != 3:
                raise ValueError(
                    f"SceneObject.color must be an (r, g, b) tuple; got {self.color!r}"
                )
            self.color = rgb  # type: ignore[assignment]

        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(
                f"SceneObject.positions must be (num_frames, 3); got {self.positions.shape}"
            )
        if self.quaternions.ndim != 2 or self.quaternions.shape[1] != 4:
            raise ValueError(
                f"SceneObject.quaternions must be (num_frames, 4); got {self.quaternions.shape}"
            )
        if self.positions.shape[0] != self.quaternions.shape[0]:
            raise ValueError(
                "SceneObject.positions and quaternions must agree on frame count: "
                f"{self.positions.shape[0]} vs {self.quaternions.shape[0]}"
            )
        # Defensive quaternion renormalisation to keep the viewer from ending up with
        # drifting rotations after many per-frame updates.
        n = np.linalg.norm(self.quaternions, axis=-1, keepdims=True)
        n = np.where(n < 1e-8, 1.0, n)
        self.quaternions = (self.quaternions / n).astype(np.float32)

    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[0])


@dataclass
class TerrainHeightfield:
    """Regular 2-D heightfield terrain — the canonical static-environment record.

    This is the **single source of truth** for terrain in the hhtools pipeline.
    All three downstream consumers (viser viewer, MuJoCo MPC-SQP collision,
    PARC training export) read from the same ``hf`` array, so what the user
    sees on screen is exactly what the optimizer feels and what gets shipped
    for training.

    The on-disk schema is mirrored from PARC's ``MSTerrainData``
    (`PARC/PARC/util/file_io.py`) — :meth:`to_ms_terrain_data_dict` and
    :meth:`from_ms_terrain_data_dict` round-trip without loss, so any sidecar
    ``.pkl`` produced by the hhtools converter can be loaded directly by
    PARC training scripts.

    Attributes:
        hf: ``(nx, ny)`` float32 grid of terrain z values, in metres, in the
            current motion's world frame.  ``hf[ix, iy]`` is the z value at
            world XY ``min_point + (ix, iy) * dx``.
        hf_maxmin: ``(nx, ny, 2)`` float32 array of (max_z, min_z) bounds
            per cell; used by PARC for height-augmentation during training.
            For freshly-converted terrains we fill globals (max_z + buf,
            min_z − buf); ``compute_hf_extra_vals`` collapses to (z, z) on
            cells the character actually visits.
        min_point: ``(2,)`` float32 world (x_min, y_min) — lower-left corner
            of cell (0, 0).
        dx: float — cell size in metres (assumed isotropic).
    """

    hf: NDArray
    hf_maxmin: NDArray
    min_point: NDArray
    dx: float

    def __post_init__(self) -> None:
        self.hf = np.asarray(self.hf, dtype=np.float32)
        if self.hf.ndim != 2:
            raise ValueError(
                f"TerrainHeightfield.hf must be 2-D (nx, ny); got {self.hf.shape}"
            )
        nx, ny = self.hf.shape
        self.hf_maxmin = np.asarray(self.hf_maxmin, dtype=np.float32)
        if self.hf_maxmin.shape != (nx, ny, 2):
            raise ValueError(
                f"TerrainHeightfield.hf_maxmin must be (nx, ny, 2); got "
                f"{self.hf_maxmin.shape} (expected {(nx, ny, 2)})"
            )
        self.min_point = np.asarray(self.min_point, dtype=np.float32).reshape(2)
        self.dx = float(self.dx)
        if not (self.dx > 0.0):
            raise ValueError(f"TerrainHeightfield.dx must be > 0; got {self.dx}")

    # ----------------------------------------------------------------- shape

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.hf.shape)  # type: ignore[return-value]

    @property
    def nx(self) -> int:
        return int(self.hf.shape[0])

    @property
    def ny(self) -> int:
        return int(self.hf.shape[1])

    @property
    def x_max(self) -> float:
        """World x of the highest-index cell centre."""
        return float(self.min_point[0]) + (self.nx - 1) * self.dx

    @property
    def y_max(self) -> float:
        return float(self.min_point[1]) + (self.ny - 1) * self.dx

    # ----------------------------------------------------------------- transforms

    def scaled(
        self,
        scale: float,
        z_offset: float = 0.0,
        *,
        z_scale: float | None = None,
    ) -> "TerrainHeightfield":
        """Return a copy in a frame scaled by ``(scale, scale, z_scale)``.

        Applies the chain used by the rest of the retargeter:
        ``xy_world = xy_local * scale`` and
        ``z_world = (z_local − z_offset) · z_scale``.  ``z_scale``
        defaults to ``scale`` (uniform 3-axis scaling) when not
        provided, which is the convention older callers expect.

        Pipelines that want **contact-preserving** retargeting pass
        ``z_scale=1.0`` so the heightfield's vertical extent is only
        shifted (by ``z_offset``) and never compressed — this keeps
        the foot-to-terrain offsets in the source units.  XY still
        shrinks by ``scale`` so the robot's reachable workspace
        matches the source's stride / reach.

        ``hf_maxmin`` follows the same chain; the ``z_buffer``
        semantics are preserved because both bounds shift by the
        same affine transform.
        """
        s = float(scale)
        sz = float(z_scale) if z_scale is not None else s
        z = float(z_offset)
        new_hf = (self.hf - z) * sz
        new_maxmin = (self.hf_maxmin - z) * sz
        new_min = self.min_point.astype(np.float32) * s
        new_dx = self.dx * s
        return TerrainHeightfield(
            hf=new_hf.astype(np.float32, copy=False),
            hf_maxmin=new_maxmin.astype(np.float32, copy=False),
            min_point=new_min,
            dx=new_dx,
        )

    def shifted(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
    ) -> "TerrainHeightfield":
        """Return a copy translated by ``(dx, dy, dz)`` in world coordinates.

        Used by viewer anatomy helpers (``center_motion_root_xy``,
        ``snap_motion_to_ground``) so the terrain follows whatever rigid
        translation the skeleton receives; otherwise the foot would float
        above (or sink into) a static heightfield.
        """
        new_min = self.min_point.astype(np.float32) + np.array([dx, dy], dtype=np.float32)
        new_hf = self.hf + np.float32(dz)
        new_maxmin = self.hf_maxmin + np.float32(dz)
        return TerrainHeightfield(
            hf=new_hf.astype(np.float32, copy=False),
            hf_maxmin=new_maxmin.astype(np.float32, copy=False),
            min_point=new_min,
            dx=self.dx,
        )

    def expanded_to_xy(
        self,
        xy_min: NDArray[np.floating],
        xy_max: NDArray[np.floating],
        *,
        padding: float = 0.0,
        fill_value: float | None = None,
    ) -> "TerrainHeightfield":
        """Pad the grid so it covers ``[xy_min - pad, xy_max + pad]``.

        New cells are filled with ``fill_value`` (default: current ``min(hf)``)
        so the actor never walks off the triangulated mesh on long parkour runs.
        """
        pad = float(padding)
        want_min = np.asarray(xy_min, dtype=np.float64).reshape(2) - pad
        want_max = np.asarray(xy_max, dtype=np.float64).reshape(2) + pad
        dx = float(self.dx)
        cur_min = self.min_point.astype(np.float64)
        cur_max = np.array([self.x_max, self.y_max], dtype=np.float64)
        new_min = np.minimum(cur_min, want_min)
        new_max = np.maximum(cur_max, want_max)
        if np.allclose(new_min, cur_min) and np.allclose(new_max, cur_max):
            return self

        nx = max(int(np.ceil((new_max[0] - new_min[0]) / dx)) + 1, 2)
        ny = max(int(np.ceil((new_max[1] - new_min[1]) / dx)) + 1, 2)
        nx = min(nx, 4096)
        ny = min(ny, 4096)
        fill = float(np.min(self.hf) if fill_value is None else fill_value)
        hf = np.full((nx, ny), fill, dtype=np.float32)
        # Paste original grid at its world-aligned offset.
        ox = int(round((cur_min[0] - new_min[0]) / dx))
        oy = int(round((cur_min[1] - new_min[1]) / dx))
        x1 = min(ox + self.nx, nx)
        y1 = min(oy + self.ny, ny)
        sx0 = max(0, -ox)
        sy0 = max(0, -oy)
        ox = max(0, ox)
        oy = max(0, oy)
        hf[ox:x1, oy:y1] = self.hf[sx0 : sx0 + (x1 - ox), sy0 : sy0 + (y1 - oy)]

        zmax = float(np.max(self.hf_maxmin[..., 0]))
        zmin = float(np.min(self.hf_maxmin[..., 1]))
        hf_mm = np.zeros((nx, ny, 2), dtype=np.float32)
        hf_mm[..., 0] = zmax
        hf_mm[..., 1] = zmin
        return TerrainHeightfield(
            hf=hf,
            hf_maxmin=hf_mm,
            min_point=new_min.astype(np.float32),
            dx=dx,
        )

    # ----------------------------------------------------------------- queries

    def height_at(self, x: float, y: float) -> float:
        """Bilinear-interpolated terrain height at world (x, y).

        Out-of-bounds queries return the nearest-cell height.  Used by the
        soft-penalty fallback path; the hard-NP path goes through MuJoCo's
        ``mj_geomDistance`` against the ``<hfield>`` geom instead.
        """
        fx = (float(x) - float(self.min_point[0])) / self.dx
        fy = (float(y) - float(self.min_point[1])) / self.dx
        nx, ny = self.shape
        ix = int(np.clip(int(np.floor(fx)), 0, nx - 2))
        iy = int(np.clip(int(np.floor(fy)), 0, ny - 2))
        ax = float(np.clip(fx - ix, 0.0, 1.0))
        ay = float(np.clip(fy - iy, 0.0, 1.0))
        g = self.hf
        return float(
            g[ix, iy] * (1 - ax) * (1 - ay)
            + g[ix + 1, iy] * ax * (1 - ay)
            + g[ix, iy + 1] * (1 - ax) * ay
            + g[ix + 1, iy + 1] * ax * ay
        )

    def triangulate(self) -> tuple[NDArray, NDArray]:
        """Return ``(vertices, faces)`` triangulating the heightfield.

        Each grid cell becomes two triangles, sharing the diagonal from
        ``(ix, iy)`` to ``(ix+1, iy+1)``.  Used by the viser renderer.

        Vertices are produced in row-major (ix, iy) order so
        ``vertex_index = ix * ny + iy``; faces use that linearisation.
        """
        nx, ny = self.shape
        xs = self.min_point[0] + np.arange(nx, dtype=np.float32) * np.float32(self.dx)
        ys = self.min_point[1] + np.arange(ny, dtype=np.float32) * np.float32(self.dx)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        verts = np.stack([gx, gy, self.hf.astype(np.float32, copy=False)], axis=-1).reshape(
            -1, 3
        )

        faces: list[NDArray] = []
        # vertex(ix, iy) = ix * ny + iy
        ix = np.arange(nx - 1, dtype=np.int32)
        iy = np.arange(ny - 1, dtype=np.int32)
        IX, IY = np.meshgrid(ix, iy, indexing="ij")
        v00 = (IX * ny + IY).reshape(-1)
        v01 = (IX * ny + IY + 1).reshape(-1)
        v10 = ((IX + 1) * ny + IY).reshape(-1)
        v11 = ((IX + 1) * ny + IY + 1).reshape(-1)
        # Two triangles per cell, with consistent winding (CCW from +Z).
        f1 = np.stack([v00, v10, v11], axis=-1)
        f2 = np.stack([v00, v11, v01], axis=-1)
        faces_arr = np.concatenate([f1, f2], axis=0).astype(np.int32)
        return verts.astype(np.float32, copy=False), faces_arr

    # ----------------------------------------------------------------- IO

    def to_ms_terrain_data_dict(self) -> dict:
        """Pack into the on-disk dict consumed by PARC's ``MSTerrainData``.

        Mirror of :class:`PARC.util.file_io.MSTerrainData`.  Each key is
        directly ``pickle.dumps``-able; PARC's
        ``SubTerrain.from_ms_terrain_data`` consumes exactly these keys.
        """
        return {
            "hf": self.hf.astype(np.float32, copy=False),
            "hf_maxmin": self.hf_maxmin.astype(np.float32, copy=False),
            "min_point": self.min_point.astype(np.float32, copy=False).reshape(2),
            "dx": float(self.dx),
        }

    @classmethod
    def from_ms_terrain_data_dict(cls, d: Mapping) -> "TerrainHeightfield":
        """Build from PARC's on-disk ``MSTerrainData`` dict.

        Accepts both the raw dict form (what
        ``pickle.loads(container[TERRAIN_DATA_KEY])`` returns) and the
        ``MSTerrainData`` dataclass once converted via ``asdict``.
        """
        hf = np.asarray(d["hf"], dtype=np.float32)
        nx, ny = hf.shape
        if "hf_maxmin" in d and d["hf_maxmin"] is not None:
            hf_maxmin = np.asarray(d["hf_maxmin"], dtype=np.float32)
        else:
            hf_maxmin = np.zeros((nx, ny, 2), dtype=np.float32)
            hf_maxmin[..., 0] = float(hf.max())
            hf_maxmin[..., 1] = float(hf.min())
        min_point = np.asarray(d["min_point"], dtype=np.float32).reshape(2)
        return cls(
            hf=hf,
            hf_maxmin=hf_maxmin,
            min_point=min_point,
            dx=float(d["dx"]),
        )


__all__ = ["SceneObject", "TerrainHeightfield"]
