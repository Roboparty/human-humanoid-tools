"""The unified internal motion representation.

A ``Motion`` stores per-frame global bone positions (``(num_frames, num_bones, 3)``) and global
bone orientations as xyzw quaternions (``(num_frames, num_bones, 4)``), together with a bone
hierarchy, a framerate, an up-axis hint, and freeform metadata.

The public NPZ schema consumed and produced by :mod:`hhtools.io.npz` mirrors these fields
1:1 so that converting between in-memory and on-disk forms is zero-cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from hhtools.core.hierarchy import Hierarchy
from hhtools.core.math import quaternion as Q
from hhtools.core.scene import SceneObject, TerrainHeightfield

UpAxis = Literal["X", "Y", "Z"]
SourceFormat = Literal["bvh", "glb", "npz", "smpl", "smplh", "smplx", "csv", "unknown"]


@dataclass
class Motion:
    """Unified motion sequence: hierarchy + per-frame global positions & quaternions.

    Attributes:
        name: Human-readable sequence name (used as file basename by exporters).
        hierarchy: Bone topology.
        positions: ``(num_frames, num_bones, 3)`` float32 array of global joint positions.
        quaternions: ``(num_frames, num_bones, 4)`` float32 array of xyzw global joint orientations.
        framerate: Sampling rate in frames per second.
        up_axis: Up-axis of the stored positions ("X" | "Y" | "Z"). The canonical internal up axis
            is ``"Z"``; IO importers rotate incoming data to this convention.
        source_format: Original file format for provenance.
        meta: Free-form extra metadata (units, source path, text labels, camera, etc.).
        objects: Optional list of :class:`SceneObject` records (props/tools the subject interacts
            with). Their per-frame 6-DoF trajectories are carried alongside the skeleton so that
            viewers and analytics can render and reason about them without re-opening the source
            dataset.  **Terrain is no longer carried here** — see :attr:`terrain`.
        terrain: Optional :class:`TerrainHeightfield` describing static environment geometry
            (stairs, platforms, slopes).  When set, this is the single source of truth for
            terrain throughout the pipeline: the viser viewer renders it as a triangulated
            surface, the MPC-SQP retargeter compiles it as a MuJoCo ``<hfield>`` for hard
            non-penetration constraints, and PARC export ships the raw ``hf`` grid for training.
            ``None`` means the clip has no terrain (e.g. flat-ground AMASS clips).
    """

    name: str
    hierarchy: Hierarchy
    positions: NDArray
    quaternions: NDArray
    framerate: float
    up_axis: UpAxis = "Z"
    source_format: SourceFormat = "unknown"
    meta: dict = field(default_factory=dict)
    objects: list[SceneObject] = field(default_factory=list)
    terrain: TerrainHeightfield | None = None

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float32)
        self.quaternions = np.asarray(self.quaternions, dtype=np.float32)
        n_bones = self.hierarchy.num_bones
        if self.positions.ndim != 3 or self.positions.shape[1:] != (n_bones, 3):
            raise ValueError(
                f"positions shape {self.positions.shape} must be (num_frames, {n_bones}, 3)"
            )
        if self.quaternions.ndim != 3 or self.quaternions.shape[1:] != (n_bones, 4):
            raise ValueError(
                f"quaternions shape {self.quaternions.shape} must be (num_frames, {n_bones}, 4)"
            )
        if self.positions.shape[0] != self.quaternions.shape[0]:
            raise ValueError(
                f"positions and quaternions disagree on frame count: "
                f"{self.positions.shape[0]} vs {self.quaternions.shape[0]}"
            )
        if self.framerate <= 0:
            raise ValueError(f"framerate must be positive; got {self.framerate}")
        # Renormalise quaternions to reduce drift across pipeline stages.
        self.quaternions = Q.normalize(self.quaternions).astype(np.float32)

    # ----------------------------------------------------------------- properties

    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_bones(self) -> int:
        return self.hierarchy.num_bones

    @property
    def bone_names(self) -> list[str]:
        return self.hierarchy.bone_names

    @property
    def duration(self) -> float:
        if self.num_frames == 0:
            return 0.0
        return (self.num_frames - 1) / self.framerate

    @property
    def delta_time(self) -> float:
        return 1.0 / self.framerate

    # ----------------------------------------------------------------- queries

    def bone_indices(self, names_or_indices: list[str | int] | None = None) -> list[int]:
        if names_or_indices is None:
            return list(range(self.num_bones))
        result: list[int] = []
        for item in names_or_indices:
            if isinstance(item, int):
                result.append(item)
            else:
                idx = self.hierarchy.index(item)
                if idx < 0:
                    raise KeyError(f"Bone {item!r} not found")
                result.append(idx)
        return result

    def bone_positions(
        self, timestamps: NDArray | None = None, bones: list[str | int] | None = None
    ) -> NDArray:
        """Return global positions sampled at ``timestamps`` (defaults to every stored frame)."""
        indices = self.bone_indices(bones)
        if timestamps is None:
            return self.positions[:, indices, :]
        frames = self._frame_indices(timestamps)
        return self.positions[frames][:, indices, :]

    def bone_quaternions(
        self, timestamps: NDArray | None = None, bones: list[str | int] | None = None
    ) -> NDArray:
        """Return global xyzw quaternions sampled at ``timestamps``."""
        indices = self.bone_indices(bones)
        if timestamps is None:
            return self.quaternions[:, indices, :]
        frames = self._frame_indices(timestamps)
        return self.quaternions[frames][:, indices, :]

    def bone_velocities(
        self, bones: list[str | int] | None = None, clip: bool = True
    ) -> NDArray:
        """Finite-difference velocities of bone positions (m/s).

        The returned array has shape ``(num_frames, num_bones, 3)``; the first frame copies the
        second-frame estimate so output length matches ``positions``.
        """
        pos = self.bone_positions(bones=bones)
        if pos.shape[0] < 2:
            return np.zeros_like(pos)
        dt = self.delta_time
        vel = np.empty_like(pos)
        vel[1:] = (pos[1:] - pos[:-1]) / dt
        vel[0] = vel[1]
        if clip:
            # Guard against absurd spikes from degenerate frames.
            vel = np.clip(vel, -1e4, 1e4)
        return vel

    # ----------------------------------------------------------------- helpers

    def _frame_indices(self, timestamps: NDArray) -> NDArray:
        t = np.asarray(timestamps, dtype=np.float32)
        idx = np.clip(np.round(t * self.framerate).astype(np.int64), 0, self.num_frames - 1)
        return idx

    def with_meta(self, **extra) -> Motion:
        """Return a shallow copy with extra metadata merged into ``meta``."""
        return Motion(
            name=self.name,
            hierarchy=self.hierarchy,
            positions=self.positions,
            quaternions=self.quaternions,
            framerate=self.framerate,
            up_axis=self.up_axis,
            source_format=self.source_format,
            meta={**self.meta, **extra},
            objects=list(self.objects),
            terrain=self.terrain,
        )

    def debug(self) -> str:
        return (
            f"Motion(name={self.name!r}, frames={self.num_frames}, bones={self.num_bones}, "
            f"fps={self.framerate:.2f}, duration={self.duration:.2f}s, up={self.up_axis}, "
            f"source={self.source_format})"
        )
