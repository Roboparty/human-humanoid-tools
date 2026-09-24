"""Linear-blend skinning math and the :class:`SkinnedMesh` data container.

A :class:`Motion` only holds joint global transforms (position + quaternion per bone per
frame).  To render an actual *body* — the visual mesh attached to that skeleton — we need
three more pieces of information from the source file (GLB, SMPL pickle, etc.):

* the **rest-pose vertices** in model-space (``vertices_rest``),
* the **triangle topology** (``triangles``),
* and per-vertex **skin attachment**: which bones influence each vertex (``skin_indices``)
  with what weights (``skin_weights``), plus the per-bone **inverse bind matrices**
  (``inverse_bind``) that map model-space points into each bone's local rest frame.

This module bundles all of that into a :class:`SkinnedMesh` and provides
:func:`lbs_deform`, which performs the standard linear-blend skinning equation:

.. math::
    v^{world}_v = \\sum_k w_{v,k} \\; G_{j_{v,k}}(t) \\; B^{-1}_{j_{v,k}} \\; v^{model}_v

where ``G[j]`` is the joint's current global transform (built per-frame from a Motion)
and ``B^{-1}[j]`` is the inverse-bind matrix.

Why the math is structured the way it is:

* We build per-joint skinning matrices ``S[j] = G[j] @ B^{-1}[j]`` once per frame, then
  gather/blend per-vertex.  The alternative (directly evaluating the sum vertex-by-vertex
  with `G` and `B^{-1}` separately) is mathematically equivalent but does V × K extra
  4×4 multiplies — wasteful when K small (4) but V is large (10K–50K).

* ``inverse_bind`` is stored as authored (in the source file's coordinate system).  The
  Motion's joint global transforms ``G`` are in whatever up-axis the Motion currently
  carries (typically Z after :func:`hhtools.core.coord.to_up_axis` runs).  As long as
  ``v_rest`` and ``inverse_bind`` came from the same authoring frame, ``S = G @ B^{-1}``
  produces world points in *Motion's current frame*, no extra conversion needed —
  conjugation by the up-axis rotation cancels naturally between ``G`` and ``B^{-1}``
  when both come from the same source.  See ``hhtools.io.glb`` for a worked derivation.

The API stays NumPy-only (no torch dependency) so the viewer can use it on CPU at 60 fps
for typical character meshes.  At ~50 K vertices and 159 joints the dominant cost is the
per-vertex 4×4 matmul, which vectorised einsum handles in a few ms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hhtools.core.math import quaternion as Q


@dataclass
class SkinnedMesh:
    """Static skin data: rest geometry + per-vertex bone attachment.

    Indexing convention: ``skin_indices`` holds *Motion bone indices* — that is, indices
    into the same hierarchy as ``Motion.positions[frame]``.  This means importers must
    reorder data when they drop or remap joints (e.g. when using ``joint_names=`` to
    pick a subset of glTF skin joints).  The renderer treats this as the source of
    truth and never has to know about the original glTF skin layout.

    Attributes:
        vertices_rest: ``(V, 3)`` float32 model-space rest positions.
        triangles: ``(T, 3)`` int32 vertex-index triangles.
        skin_indices: ``(V, K)`` int32 bone indices per vertex (0 ≤ idx < num_bones).
            Vertices with fewer than K influences should pad with index 0 and weight 0.
        skin_weights: ``(V, K)`` float32 weights, expected to sum to ≤ 1 along axis -1
            (we re-normalise on construction so callers don't have to).
        inverse_bind: ``(J, 4, 4)`` float32 inverse-bind matrices.  ``J`` must equal
            the Motion's bone count; element ``j`` is the inverse of the bone's
            global transform at the rest pose, in the source file's coordinate frame.
        normals_rest: Optional ``(V, 3)`` float32 per-vertex rest normals.  When
            present, the renderer can either ignore them (flat-shaded) or transform
            them per-frame via the upper-3×3 of the skinning matrices.
    """

    vertices_rest: NDArray
    triangles: NDArray
    skin_indices: NDArray
    skin_weights: NDArray
    inverse_bind: NDArray
    normals_rest: NDArray | None = None

    def __post_init__(self) -> None:
        self.vertices_rest = np.ascontiguousarray(self.vertices_rest, dtype=np.float32)
        self.triangles = np.ascontiguousarray(self.triangles, dtype=np.int32)
        self.skin_indices = np.ascontiguousarray(self.skin_indices, dtype=np.int32)
        self.skin_weights = np.ascontiguousarray(self.skin_weights, dtype=np.float32)
        self.inverse_bind = np.ascontiguousarray(self.inverse_bind, dtype=np.float32)
        if self.normals_rest is not None:
            self.normals_rest = np.ascontiguousarray(self.normals_rest, dtype=np.float32)

        if self.vertices_rest.ndim != 2 or self.vertices_rest.shape[1] != 3:
            raise ValueError(
                f"vertices_rest must be (V, 3); got {self.vertices_rest.shape}"
            )
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(f"triangles must be (T, 3); got {self.triangles.shape}")
        if self.skin_indices.shape != self.skin_weights.shape:
            raise ValueError(
                "skin_indices and skin_weights shapes must match: "
                f"{self.skin_indices.shape} vs {self.skin_weights.shape}"
            )
        if self.skin_indices.ndim != 2:
            raise ValueError(f"skin_indices must be (V, K); got {self.skin_indices.shape}")
        if self.skin_indices.shape[0] != self.vertices_rest.shape[0]:
            raise ValueError(
                "skin_indices vertex count disagrees with vertices_rest: "
                f"{self.skin_indices.shape[0]} vs {self.vertices_rest.shape[0]}"
            )
        if self.inverse_bind.ndim != 3 or self.inverse_bind.shape[1:] != (4, 4):
            raise ValueError(
                f"inverse_bind must be (J, 4, 4); got {self.inverse_bind.shape}"
            )

        # Re-normalise weights row-wise so a stray (0.5, 0.4, 0.0, 0.0) doesn't shrink
        # vertices. Rows that are all-zero (rare; usually a non-skinned vertex incorrectly
        # included) are left as zero — the LBS would otherwise produce NaN.
        row_sum = self.skin_weights.sum(axis=1, keepdims=True)
        safe = np.where(row_sum > 1e-8, row_sum, 1.0)
        self.skin_weights = (self.skin_weights / safe).astype(np.float32)

        # Clamp out-of-range bone indices for defensive behaviour. Out-of-range usually
        # means the mesh expects more bones than the Motion exposes (e.g. user provided a
        # joint_names whitelist smaller than the source skin); clamping to 0 ensures we
        # render *something* rather than raising during the per-frame loop.
        max_bone = int(self.inverse_bind.shape[0]) - 1
        if max_bone >= 0:
            self.skin_indices = np.clip(self.skin_indices, 0, max_bone).astype(np.int32)

    @property
    def num_vertices(self) -> int:
        return int(self.vertices_rest.shape[0])

    @property
    def num_triangles(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.inverse_bind.shape[0])

    @property
    def influences_per_vertex(self) -> int:
        return int(self.skin_indices.shape[1])


# ---------------------------------------------------------------------- LBS math


def joint_global_matrices(positions: NDArray, quaternions: NDArray) -> NDArray:
    """Build per-joint 4×4 global transforms from Motion.positions + quaternions.

    Args:
        positions: ``(J, 3)`` float — global joint translations (one frame).
        quaternions: ``(J, 4)`` float xyzw — global joint orientations (one frame).

    Returns:
        ``(J, 4, 4)`` float32 stack of homogeneous transforms.
    """
    quat = Q.normalize(np.asarray(quaternions, dtype=np.float32))
    R = Q.to_matrix(quat)
    out = np.zeros((R.shape[0], 4, 4), dtype=np.float32)
    out[:, :3, :3] = R
    out[:, :3, 3] = np.asarray(positions, dtype=np.float32)
    out[:, 3, 3] = 1.0
    return out


def lbs_deform(
    mesh: SkinnedMesh,
    positions: NDArray,
    quaternions: NDArray,
) -> NDArray:
    """Deform ``mesh.vertices_rest`` for one frame using linear-blend skinning.

    Args:
        mesh: Static skinning data.
        positions: ``(J, 3)`` joint global translations for the target frame.
        quaternions: ``(J, 4)`` xyzw joint global orientations for the target frame.

    Returns:
        ``(V, 3)`` float32 deformed vertex positions in the same coordinate frame as
        the input ``positions``.

    Raises:
        ValueError: when joint count of ``positions`` disagrees with ``mesh.inverse_bind``.

    Notes:
        Uses the ``S = G @ B^{-1}`` per-joint precomputation to avoid redoing the
        same joint matmul per-vertex.  The final per-vertex blend is a single
        ``einsum("vk,vkij->vij", weights, S[indices])`` followed by a homogeneous
        matvec — which handles 50 K vertices × 4 influences in a few ms on CPU.
    """
    pos = np.asarray(positions, dtype=np.float32)
    quat = np.asarray(quaternions, dtype=np.float32)
    if pos.shape[0] != mesh.inverse_bind.shape[0]:
        raise ValueError(
            "Joint count mismatch: motion has "
            f"{pos.shape[0]} joints but SkinnedMesh.inverse_bind has "
            f"{mesh.inverse_bind.shape[0]}. Make sure the same joint set was used to "
            "build the SkinnedMesh and to drive the Motion."
        )

    G = joint_global_matrices(pos, quat)  # (J, 4, 4)
    S = np.matmul(G, mesh.inverse_bind)  # (J, 4, 4) per-joint skinning matrix

    # Gather per-influence skinning matrices, blend with weights.
    S_per_inf = S[mesh.skin_indices]  # (V, K, 4, 4)
    M = np.einsum("vk,vkij->vij", mesh.skin_weights, S_per_inf)  # (V, 4, 4)

    # Homogeneous mat-vec: (V, 4, 4) @ (V, 4) -> (V, 4)
    v_homo = np.empty((mesh.num_vertices, 4), dtype=np.float32)
    v_homo[:, :3] = mesh.vertices_rest
    v_homo[:, 3] = 1.0
    v_world = np.einsum("vij,vj->vi", M, v_homo)
    return v_world[:, :3].astype(np.float32, copy=False)


def lbs_deform_normals(
    mesh: SkinnedMesh,
    positions: NDArray,
    quaternions: NDArray,
) -> NDArray | None:
    """Deform ``mesh.normals_rest`` using only the rotation part of the skinning matrices.

    Returns ``None`` when ``mesh.normals_rest`` is unset.  Strictly speaking, normals
    should use the inverse-transpose of the upper 3×3, but for the rigid (rotation +
    translation) skinning we get from joint poses the inverse-transpose equals the matrix
    itself, so a plain rotation suffices.  Re-normalised on output.
    """
    if mesh.normals_rest is None:
        return None
    pos = np.asarray(positions, dtype=np.float32)
    quat = np.asarray(quaternions, dtype=np.float32)
    G = joint_global_matrices(pos, quat)
    S = np.matmul(G, mesh.inverse_bind)
    S3 = S[:, :3, :3]  # (J, 3, 3) rotation part only
    S_per_inf = S3[mesh.skin_indices]  # (V, K, 3, 3)
    M = np.einsum("vk,vkij->vij", mesh.skin_weights, S_per_inf)
    n_world = np.einsum("vij,vj->vi", M, mesh.normals_rest)
    norms = np.linalg.norm(n_world, axis=-1, keepdims=True)
    n_world = n_world / np.where(norms > 1e-8, norms, 1.0)
    return n_world.astype(np.float32, copy=False)


@dataclass
class BakedMesh:
    """Pre-computed per-frame vertex positions for a deforming surface.

    Unlike :class:`SkinnedMesh` (which carries rest geometry + skin attachment and is
    deformed *on the fly* via LBS), ``BakedMesh`` stores the already-deformed vertex
    cloud for every frame.  This is the format we use for SMPL / SMPL-H / SMPL-X
    visualisation because:

    * The SMPL forward pass already gives us exact vertices including pose-dependent
      corrective blendshapes (``posedirs``) which pure LBS cannot reproduce.  Baking
      keeps that fidelity at ~0% runtime cost.
    * The memory footprint is modest for typical clips: SMPL (V=6890) at 300 frames
      ≈ 25 MB float32; SMPL-X (V=10475) at 300 frames ≈ 38 MB — well within what
      the browser renderer handles in a single mesh update per frame.

    Attributes:
        vertices: ``(T, V, 3)`` float32 deformed positions, one per frame.
        triangles: ``(F, 3)`` int32 face topology (shared across all frames).
        normals: Optional ``(T, V, 3)`` float32 per-frame vertex normals.  When
            absent the renderer computes smooth-shading defaults.
    """

    vertices: NDArray
    triangles: NDArray
    normals: NDArray | None = None

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float32)
        self.triangles = np.ascontiguousarray(self.triangles, dtype=np.int32)
        if self.normals is not None:
            self.normals = np.ascontiguousarray(self.normals, dtype=np.float32)
        if self.vertices.ndim != 3 or self.vertices.shape[2] != 3:
            raise ValueError(f"vertices must be (T, V, 3); got {self.vertices.shape}")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(f"triangles must be (F, 3); got {self.triangles.shape}")
        if self.normals is not None and self.normals.shape != self.vertices.shape:
            raise ValueError(
                f"normals shape {self.normals.shape} must match vertices {self.vertices.shape}"
            )

    @property
    def num_frames(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[1])

    @property
    def num_triangles(self) -> int:
        return int(self.triangles.shape[0])

    def frame(self, index: int) -> NDArray:
        """Return the ``(V, 3)`` vertex positions for ``index`` (clamped into range)."""
        idx = max(0, min(index, self.num_frames - 1))
        return self.vertices[idx]


__all__ = [
    "BakedMesh",
    "SkinnedMesh",
    "joint_global_matrices",
    "lbs_deform",
    "lbs_deform_normals",
]
