"""Generate a capsule mesh that follows a skeleton over time.

We don't need a full skinned mesh for BVH visualisation — a per-bone capsule is enough to convey
mass distribution and is what :mod:`hhtools.viewer` renders by default.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bone_line_segments(positions: NDArray, parent_indices: NDArray) -> tuple[NDArray, NDArray]:
    """Return ``(segment_starts, segment_ends)`` of shape ``(num_frames, num_bones-roots, 3)``.

    Useful for drawing the skeleton as lines in 3D viewers.
    """
    positions = np.asarray(positions, dtype=np.float32)
    parent_indices = np.asarray(parent_indices, dtype=np.int32)
    valid = parent_indices >= 0
    children_idx = np.where(valid)[0]
    parents_idx = parent_indices[valid]
    starts = positions[:, parents_idx]
    ends = positions[:, children_idx]
    return starts, ends


def build_capsule_skeleton_mesh(
    positions: NDArray,
    parent_indices: NDArray,
    radius: float = 0.05,
    segments: int = 6,
) -> dict:
    """Return a lightweight capsule-per-bone representation.

    For the viewer's purposes we don't actually need a tessellated mesh up front — we return
    segment metadata so a renderer can draw capsules using its native primitives (the Web Stage offers
    scene.add_segments, mesh, and cylinder, all of which are cheap to update per frame).
    """
    starts, ends = bone_line_segments(positions, parent_indices)
    return {
        "segments": {
            "starts": starts,
            "ends": ends,
        },
        "radius": float(radius),
        "num_bones": int(positions.shape[1]),
        "num_frames": int(positions.shape[0]),
        "capsule_resolution": int(segments),
    }


__all__ = ["bone_line_segments", "build_capsule_skeleton_mesh"]
