"""Human mesh utilities: capsule mesh (for bare skeletons), skinned mesh, SMPL surface mesh.

Implementations are added incrementally through milestones M7 and beyond. The module currently
exposes a capsule mesh generator that is sufficient for the Web Stage to render any imported
BVH with a playful placeholder body.
"""

from __future__ import annotations

from hhtools.human.capsule_mesh import build_capsule_skeleton_mesh

__all__ = ["build_capsule_skeleton_mesh"]
