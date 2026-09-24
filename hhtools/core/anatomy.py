"""Skeleton-anatomy helpers used by viewer renderers.

The helpers here reason about a motion's bone hierarchy and produce per-bone rendering hints:

* :func:`detect_virtual_root` flags skeletons whose bone 0 is a placeholder node such as
  ``Root`` / ``Reference`` / ``Armature``; these nodes clutter visualisation because they sit
  at the world origin and draw a long "fake spine" segment up to the real pelvis.
* :func:`degenerate_auxiliary_bone_indices` drops mocap sole markers (e.g. holosoma
  ``*FootMod``) when they are **coincident** with the parent foot — the shipped
  ``parkour_*.npy`` files often duplicate the foot position, which makes line renderers
  emit spurious long strokes if drawn as a bone.
* :func:`compute_bone_radii` derives a per-bone capsule radius that respects both bone length
  (short bones = thin capsules) and bone name (finger / toe / eye / jaw joints = always thin).
* :func:`snap_motion_to_ground` returns a translated copy of a :class:`Motion` whose lowest z
  is 0 so the feet rest on the ground grid.  For ``20260429_mocap`` clips with a heightfield
  (see :func:`hhtools.core.grounding.use_split_terrain_grounding`) the terrain mesh uses a
  separate vertical shift so ``min(hf)`` also meets the margin — the legacy single ``dz``
  for both skeleton and HF would leave deep heightfields half-buried when feet sit higher
  than ``min(hf)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hhtools.core.grounding import (
    human_source_floor_z_world,
    parc_ms_shares_human_terrain_z,
    use_split_terrain_grounding,
)
from hhtools.core.human_anatomy import (
    SMALL_BONE_KEYWORDS as _SMALL_BONE_KEYWORDS,
)
from hhtools.core.human_anatomy import (
    compact_skeleton_exclude_indices,
    degenerate_auxiliary_bone_indices,
    dense_rig_viz_exclude_indices,
    detect_virtual_root,
    exclude_joint_from_compact_scaled_preview,
    hand_foot_subtree_exclude_indices,
)
from hhtools.core.motion import Motion
from hhtools.core.scene import SceneObject


def _shifted_object(
    obj: SceneObject, *, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0
) -> SceneObject:
    """Return a copy of ``obj`` with its per-frame translation shifted by (dx, dy, dz)."""
    new_pos = obj.positions.copy()
    new_pos[..., 0] += dx
    new_pos[..., 1] += dy
    new_pos[..., 2] += dz
    return SceneObject(
        name=obj.name,
        positions=new_pos,
        quaternions=obj.quaternions.copy(),
        extents=obj.extents.copy(),
        mesh_path=obj.mesh_path,
        scale=obj.scale,
        opacity=obj.opacity,
        color=obj.color,
    )


def _translate_mesh_meta(
    meta: dict, *, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0
) -> dict:
    """Return a shallow-copied ``meta`` with any attached ``BakedMesh`` translated by (dx, dy, dz).

    ``BakedMesh`` stores absolute world-space vertex positions per frame — unlike
    :class:`~hhtools.core.skinning.SkinnedMesh` where LBS multiplies the (possibly shifted)
    joint transforms ``G`` against a fixed rest pose, so any translation on joints
    propagates naturally to the deformed surface.  For baked caches we have to copy
    the same shift onto the stored vertices explicitly, otherwise toggling
    ``center_xy`` or ``snap_ground`` pulls the skeleton back to the origin while the
    baked SMPL body stays several metres away (the visible misalignment reported in
    the AMASS viewer).
    """
    from hhtools.core.skinning import BakedMesh  # lazy; keeps core out of the import cycle

    new_meta = dict(meta)
    baked = new_meta.get("baked_mesh")
    if isinstance(baked, BakedMesh):
        offset = np.array([dx, dy, dz], dtype=np.float32)
        if np.any(offset != 0.0):
            new_meta["baked_mesh"] = BakedMesh(
                vertices=baked.vertices + offset,
                triangles=baked.triangles,
                normals=baked.normals,  # directions, unaffected by pure translation
            )
    return new_meta


_LIMB_CANONICAL_CHAINS: tuple[tuple[str, ...], ...] = (
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_knee", "left_ankle", "left_foot"),
    ("right_hip", "right_knee", "right_ankle", "right_foot"),
)


def deepest_mapped_canonicals(ik_map_canonicals: frozenset[str]) -> frozenset[str]:
    """Deepest IK-mapped joint per limb chain (wrist/ankle when toe/hand are unmapped)."""
    terminals: set[str] = set()
    for chain in _LIMB_CANONICAL_CHAINS:
        for canon in reversed(chain):
            if canon in ik_map_canonicals:
                terminals.add(canon)
                break
    return frozenset(terminals)


def motion_has_interaction_scene(motion: Motion) -> bool:
    """True when the clip carries terrain and/or rigid props (OMOMO / parc_ms)."""
    if motion.terrain is not None:
        return True
    for ob in motion.objects:
        if ob.mesh_path:
            return True
        ext = np.asarray(ob.extents, dtype=np.float64).reshape(-1)
        if ext.size >= 3 and float(np.max(np.abs(ext))) > 1e-6:
            return True
    return False


def scaler_compact_bead_row_indices(
    scaler_joint_names: tuple[str, ...], motion: Motion,
) -> NDArray[np.int32]:
    """Scalar joint rows to draw as beads in the yellow scaler overlay (drops dense hands).

    Maps scaler joint names back to source motion bone indices via the hierarchy and
    applies the same exclusion set as :func:`dense_rig_viz_exclude_indices`.
    """

    ex = dense_rig_viz_exclude_indices(motion)
    h = motion.hierarchy
    rows: list[int] = []
    for i, name in enumerate(scaler_joint_names):
        bi = h.index(name)
        if bi >= 0 and bi in ex:
            continue
        rows.append(i)
    return np.asarray(rows, dtype=np.int32)


def compute_bone_radii(
    bone_names: list[str],
    parent_indices: NDArray,
    positions_frame0: NDArray,
    *,
    base_body: float = 0.04,
    base_small: float = 0.012,
    length_ratio: float = 0.30,
    min_radius: float = 0.004,
) -> NDArray:
    """Return an ``(J,)`` float32 array of per-bone capsule radii.

    A bone is considered "small" (fingers / toes / eyes / jaw) if any of a small set of
    keywords appears in its name. Small bones get at most ``base_small`` (~1.2 cm); regular
    body bones get at most ``base_body`` (~4 cm). For both, the final radius is capped at
    ``length_ratio`` of the bone length so that very short bones cannot bloat into blobs.
    """
    J = len(bone_names)
    radii = np.empty(J, dtype=np.float32)
    for i in range(J):
        parent = int(parent_indices[i])
        if parent >= 0:
            length = float(np.linalg.norm(positions_frame0[i] - positions_frame0[parent]))
        else:
            length = 0.0
        name_low = bone_names[i].lower()
        is_small = any(kw in name_low for kw in _SMALL_BONE_KEYWORDS)
        base = base_small if is_small else base_body
        radii[i] = max(min_radius, min(base, length * length_ratio))
    return radii


def center_motion_root_xy(motion: Motion) -> Motion:
    """Return a copy of ``motion`` translated so frame-0 root XY sits at the world origin.

    Captures such as LAFAN are recorded in a mocap stage with the subject several meters away
    from the origin; when dropped into our viewer the character starts off-grid, which is
    jarring. We shift all frames by a constant (-root_xy_at_frame_0) so the subject's starting
    pose is centred. Vertical (Z) motion and all relative displacements are preserved.
    """
    pos = np.asarray(motion.positions, dtype=np.float32)
    if pos.size == 0:
        return motion
    root_xy = pos[0, 0, :2].copy()
    if np.linalg.norm(root_xy) < 1e-4:
        return motion
    shifted = pos.copy()
    shifted[..., 0] -= root_xy[0]
    shifted[..., 1] -= root_xy[1]
    new_objects = [_shifted_object(obj, dx=-root_xy[0], dy=-root_xy[1]) for obj in motion.objects]
    new_meta = _translate_mesh_meta(motion.meta, dx=-float(root_xy[0]), dy=-float(root_xy[1]))
    new_meta["root_xy_offset"] = (-float(root_xy[0]), -float(root_xy[1]))
    new_terrain = (
        motion.terrain.shifted(dx=-float(root_xy[0]), dy=-float(root_xy[1]))
        if motion.terrain is not None
        else None
    )
    return Motion(
        name=motion.name,
        hierarchy=motion.hierarchy,
        positions=shifted,
        quaternions=motion.quaternions,
        framerate=motion.framerate,
        up_axis=motion.up_axis,
        source_format=motion.source_format,
        meta=new_meta,
        objects=new_objects,
        terrain=new_terrain,
    )


def snap_motion_to_ground(motion: Motion, *, margin: float = 0.0) -> Motion:
    """Return a copy of ``motion`` translated so the actor meets the ground grid on Z.

    SMPL-based datasets (PHUMA, Motion-X) often emit frames centred around the pelvis,
    leaving feet at negative z. This helper shifts every frame by a constant so the lowest
    joint rests at ``margin`` on the up axis.  A small positive ``margin`` matches the
    spirit of soma-retargeter's ground-contact offsets and keeps thick capsule bones from
    visually intersecting the ground grid.

    When the hierarchy names at least two plausible foot / ankle / toe bones, the
    vertical shift is computed from **those joints only** (minimum Z across all frames).
    Otherwise the legacy rule applies: minimum Z over every joint — matching older AMASS
    clips that only expose body joints without explicit ``*Foot`` markers.

    For ``meshmimic/20260429_mocap`` clips with :attr:`~hhtools.core.motion.Motion.terrain`,
    the skeleton/objects use a foot-based shift while the heightfield uses a shift derived
    from ``min(terrain.hf)`` so the triangulated mesh is not left half-under the grid when
    the raster extends below the feet (matches interaction-mesh ``z_offset`` split).
    """
    pos = np.asarray(motion.positions, dtype=np.float32)
    if pos.size == 0:
        return motion

    terr = motion.terrain
    # Authored ``parc_ms`` clips keep feet and terrain in one world frame, so both
    # must share the human-floor shift (:func:`parc_ms_shares_human_terrain_z`).
    # ``20260429_mocap`` re-exports reuse the parc_ms pkl layout but need a
    # separate ``min(hf)`` shift — otherwise obstacle bottoms (Maya floor) sink
    # below the viewer grid by the foot-marker height (~10 cm).
    if (
        use_split_terrain_grounding(motion)
        and terr is not None
        and not parc_ms_shares_human_terrain_z(motion)
    ):
        z_human = float(human_source_floor_z_world(motion))
        z_hf_min = float(np.min(terr.hf))
        off_h = float(margin) - z_human
        off_t = float(margin) - z_hf_min
        if max(abs(off_h), abs(off_t)) < 1e-4:
            return motion
        shifted = pos.copy()
        shifted[..., 2] += np.float32(off_h)
        new_objects = [_shifted_object(obj, dz=off_h) for obj in motion.objects]
        new_meta = _translate_mesh_meta(motion.meta, dz=off_h)
        new_meta["ground_offset_z"] = float(off_h)
        new_meta["terrain_ground_offset_z"] = float(off_t)
        new_terrain = terr.shifted(dz=off_t)
        return Motion(
            name=motion.name,
            hierarchy=motion.hierarchy,
            positions=shifted,
            quaternions=motion.quaternions,
            framerate=motion.framerate,
            up_axis=motion.up_axis,
            source_format=motion.source_format,
            meta=new_meta,
            objects=new_objects,
            terrain=new_terrain,
        )

    z_min = human_source_floor_z_world(motion)
    offset = margin - z_min
    if abs(offset) < 1e-4:
        return motion
    shifted = pos.copy()
    shifted[..., 2] += offset
    new_objects = [_shifted_object(obj, dz=float(offset)) for obj in motion.objects]
    new_meta = _translate_mesh_meta(motion.meta, dz=float(offset))
    new_meta["ground_offset_z"] = float(offset)
    new_terrain = (
        motion.terrain.shifted(dz=float(offset)) if motion.terrain is not None else None
    )
    return Motion(
        name=motion.name,
        hierarchy=motion.hierarchy,
        positions=shifted,
        quaternions=motion.quaternions,
        framerate=motion.framerate,
        up_axis=motion.up_axis,
        source_format=motion.source_format,
        meta=new_meta,
        objects=new_objects,
        terrain=new_terrain,
    )


__all__ = [
    "center_motion_root_xy",
    "compact_skeleton_exclude_indices",
    "compute_bone_radii",
    "dense_rig_viz_exclude_indices",
    "degenerate_auxiliary_bone_indices",
    "detect_virtual_root",
    "exclude_joint_from_compact_scaled_preview",
    "deepest_mapped_canonicals",
    "hand_foot_subtree_exclude_indices",
    "motion_has_interaction_scene",
    "scaler_compact_bead_row_indices",
    "snap_motion_to_ground",
]
