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

_VIRTUAL_ROOT_NAMES: frozenset[str] = frozenset(
    {
        "root",
        "reference",
        "world",
        "armature",
        "origin",
        "root_body",
        "scene_root",
        # Common glTF / FBX / Maya authoring wrappers that sit one level above the
        # real pelvis.  E.g. Cranberry rig uses "body_world" as a parent of "b_root",
        # which itself is a real joint at the hips.  Without this entry the viewer
        # draws a tall fake capsule from the world origin up to the character's hips.
        "body_world",
        "worldroot",
        "world_root",
        "rig",
        "skeleton",
        "skeleton_root",
        "body",
        "character",
        "main",
        "rootnode",
    }
)

_SMALL_BONE_KEYWORDS: tuple[str, ...] = (
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
    "finger",
    "toe",
    "eye",
    "jaw",
    "eyeball",
    "end",
    "tip",
    "headend",
)

_EXTRA_COMPACT_EXCLUDE: tuple[str, ...] = (
    "phalange",
    "metacarpal",
    "carpal",
    "metatarsal",
    "tarsal",
    "handtip",
    "footmod",
)


def exclude_unmapped_head_neck_from_scaled_preview(
    name: str,
    *,
    ik_map_canonicals: frozenset[str],
) -> bool:
    """True when ``name`` is head/neck but the robot ``ik_map`` has no ``head``.

    Headless humanoids (RP1, …) only track up to ``chest``.  Drawing the
    source head chain in the yellow overlay at uniform scale makes the figure
    look taller than the robot mesh even after stature normalisation.
    """
    if "head" in ik_map_canonicals:
        return False
    from hhtools.retarget.newton_basic.human_aliases import auto_source_to_canonical

    canon = auto_source_to_canonical((str(name),)).get(str(name), str(name))
    # ``auto_source_to_canonical`` needs the full rig to detect Mixamo / LAFAN
    # naming; a single-bone tuple stays identity (``Head`` → ``Head``).  Compare
    # case-insensitively so TitleCase BVH joints still hide on headless robots.
    return str(canon).lower() in ("head", "neck")


def exclude_joint_from_compact_scaled_preview(name: str) -> bool:
    """True if this joint should not participate in yellow scaler skeleton segments.

    Dense SMPL-H / meshmimic rigs carry full finger / toe chains; drawing every edge
    clutters the overlay and makes limb proportions hard to read.  Uses the same
    keyword heuristics as :func:`compute_bone_radii` plus a few anatomy tokens.
    """

    n = str(name).lower()
    if any(kw in n for kw in _SMALL_BONE_KEYWORDS):
        return True
    if any(kw in n for kw in _EXTRA_COMPACT_EXCLUDE):
        return True
    return False


def compact_skeleton_exclude_indices(motion: Motion) -> set[int]:
    """Bone indices to omit from line/capsule skeleton for dense rigs (fingers, toes, …).

    Mirrors :func:`exclude_joint_from_compact_scaled_preview` but returns indices for
    :class:`~hhtools.viewer.renderers.skeleton.SkeletonRenderer` /
    :class:`~hhtools.viewer.renderers.capsules.CapsuleMeshRenderer` ``exclude_bones``.
    """

    return {
        i
        for i, name in enumerate(motion.hierarchy.bone_names)
        if exclude_joint_from_compact_scaled_preview(name)
    }


def _strip_bone_basename(name: str) -> str:
    """Drop Mixamo / Blender namespace prefixes from the last path segment."""
    t = str(name)
    for sep in ("|", ":", "/"):
        if sep in t:
            t = t.split(sep)[-1]
    return t.strip()


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


def scaled_hand_tip_positions_world(
    motion: Motion,
    scaler: object,
    side: str,
) -> NDArray[np.float32] | None:
    """Per-frame scaled-world positions of the farthest hand descendant past the wrist.

    Used to drive ``left_hand_end`` / ``right_hand_end`` IK targets so robots without
    finger links still reach the same contact point as the yellow overlay (OMOMO chair
    touches, etc.).
    """
    from hhtools.retarget.newton_basic.human_aliases import auto_source_to_canonical

    names = list(motion.hierarchy.bone_names)
    parents = np.asarray(motion.hierarchy.parent_indices, dtype=np.int64)
    src2can = auto_source_to_canonical(tuple(names))

    wrist_i: int | None = None
    for i, raw in enumerate(names):
        if str(src2can.get(raw, raw)).lower() == f"{side}_wrist":
            wrist_i = i
            break
    if wrist_i is None:
        return None

    children: list[list[int]] = [[] for _ in names]
    for i, p in enumerate(parents):
        if int(p) >= 0:
            children[int(p)].append(i)

    def subtree(root: int) -> list[int]:
        stack = [root]
        out: list[int] = []
        while stack:
            j = stack.pop()
            out.append(j)
            stack.extend(children[j])
        return out

    tips = [j for j in subtree(wrist_i) if j != wrist_i]
    if not tips:
        tips = [wrist_i]

    scaled = scaler.scale_world_points_about_root(
        motion, motion.positions.astype(np.float32, copy=False),
    )
    wrist_pos = scaled[:, wrist_i, :]
    tip_pos = scaled[:, tips, :]
    dists = np.linalg.norm(tip_pos - wrist_pos[:, None, :], axis=2)
    best = np.argmax(dists, axis=1)
    return tip_pos[np.arange(scaled.shape[0]), best, :].astype(np.float32, copy=False)


def scaled_overlay_exclude_bone_indices(
    motion: Motion,
    ik_map_canonicals: frozenset[str],
) -> set[int]:
    """Human bone indices to omit from the yellow scaled overlay past the robot's IK tips.

    When a headless / handless robot only maps through ``left_wrist`` (no finger
    targets), segments to downstream ``*Hand`` / toe bones make the overlay reach
    past the physical mesh (the RP1 wrist link ends before the SMPL hand joint).
    """
    if not ik_map_canonicals:
        return set()
    from hhtools.retarget.newton_basic.human_aliases import auto_source_to_canonical

    terminals = deepest_mapped_canonicals(ik_map_canonicals)
    if not terminals:
        return set()

    names = tuple(motion.hierarchy.bone_names)
    src2can = auto_source_to_canonical(names)
    terminal_src: set[int] = set()
    for i, raw in enumerate(names):
        canon = str(src2can.get(raw, raw)).lower()
        if canon in terminals:
            terminal_src.add(i)

    parents = np.asarray(motion.hierarchy.parent_indices, dtype=np.int64)
    exclude: set[int] = set()
    n = len(names)
    for i in range(n):
        j = i
        while j >= 0:
            if j in terminal_src and j != i:
                exclude.add(i)
                break
            j = int(parents[j])
    exclude |= hand_foot_subtree_exclude_indices(motion)
    return exclude


def hand_foot_subtree_exclude_indices(motion: Motion) -> set[int]:
    """Exclude bones strictly below wrist / ankle hubs (hand and foot descendants).

    Keyword heuristics from :func:`compact_skeleton_exclude_indices` miss some DCC
    naming schemes; walking the hierarchy from each ``*Hand`` / ``*Foot`` hub catches
    finger / toe chains regardless of child bone labels.
    """

    names = motion.hierarchy.bone_names
    parents = np.asarray(motion.hierarchy.parent_indices, dtype=np.int64)
    hubs: set[int] = set()
    for i, raw in enumerate(names):
        bn = _strip_bone_basename(raw).lower()
        if bn.endswith("hand") and "forearm" not in bn:
            hubs.add(i)
        elif bn.endswith("foot") and not bn.endswith("footmod"):
            hubs.add(i)
        elif bn.endswith("feet"):
            hubs.add(i)
    out: set[int] = set()
    n = len(names)
    for i in range(n):
        j = i
        while j >= 0:
            if j in hubs and j != i:
                out.add(i)
                break
            j = int(parents[j])
    return out


def dense_rig_viz_exclude_indices(motion: Motion) -> set[int]:
    """Full body-only skeleton mask: keywords + hand/foot subtree for compact viz."""
    return compact_skeleton_exclude_indices(motion) | hand_foot_subtree_exclude_indices(
        motion,
    )


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


def detect_virtual_root(bone_names: list[str]) -> bool:
    """Return True when bone 0 is a known placeholder root node (not a real pelvis/hip)."""
    if not bone_names:
        return False
    token = bone_names[0].strip().lower()
    # Strip common prefixes like "Armature|" or "Skeleton:".
    for sep in (":", "|", "/"):
        if sep in token:
            token = token.split(sep)[-1]
    return token in _VIRTUAL_ROOT_NAMES


def degenerate_auxiliary_bone_indices(
    motion: Motion,
    frame: int = 0,
    *,
    eps: float = 2e-3,
) -> set[int]:
    """Return child-bone indices whose parent segment is ~zero-length (skip in line viz).

    meshmimic/holosoma ``*FootMod`` markers are often stored **coincident** with the
    parent ``*Foot`` joint in ``joint_positions.npy`` (sole triangle collapses to a
    point in some releases).  Line renderers may turn degenerate segments into
    spurious long strokes; excluding these indices matches the Laplacian intent
    (auxiliary contact geometry, not an extra limb bone).
    """

    names = tuple(motion.hierarchy.bone_names)
    parents = np.asarray(motion.hierarchy.parent_indices, dtype=np.int64)
    f = int(np.clip(frame, 0, max(0, motion.num_frames - 1)))
    pos = np.asarray(motion.positions[f], dtype=np.float64)
    out: set[int] = set()
    for i, name in enumerate(names):
        p = int(parents[i])
        if p < 0:
            continue
        n_low = name.lower()
        if not n_low.endswith("footmod"):
            continue
        d = float(np.linalg.norm(pos[i] - pos[p]))
        if d <= float(eps):
            out.add(i)
    return out


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
    "exclude_unmapped_head_neck_from_scaled_preview",
    "deepest_mapped_canonicals",
    "hand_foot_subtree_exclude_indices",
    "motion_has_interaction_scene",
    "scaled_hand_tip_positions_world",
    "scaled_overlay_exclude_bone_indices",
    "scaler_compact_bead_row_indices",
    "snap_motion_to_ground",
]
