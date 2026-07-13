# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Convert ``20260429_mocap`` clip folders into meshmimic / parc_ms lean layout.

Source layout (per take folder)::

    Take_*_Skeleton0.bvh          # human (23-bone Spine3)
    Take_*_*_rig*.bvh / PaoKu_*   # static obstacle rigid tracks (ROOT-only)
    obj/*.obj                     # Maya cm meshes

Output (same as ``assets/motions/meshmimic/parc_ms``)::

    <clip>/<clip>.pkl            # PARC MSFileData: motion_data + terrain_data
    <clip>/<clip>_terrain.obj    # triangulated heightfield (when terrain present)

The 23-bone Spine3 mocap is remapped onto the canonical PARC 15-body rig so
:class:`~hhtools.io.datasets.parc_ms.ParcMsAdapter` can load the clip.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hhtools.core.grounding import preferred_floor_contact_bone_indices
from hhtools.core.math import quaternion as Q
from hhtools.core.motion import Motion
from hhtools.core.scene import TerrainHeightfield
from hhtools.io.bvh import load_bvh
from hhtools.io.bvh_detect import bvh_hierarchy_has_joints
from hhtools.io.parc_export import save_parc_pkl, world_quaternions_to_local
from hhtools.io.parc_import import heightfield_to_wavefront_obj
from hhtools.io.parc_ms_skeleton import (
    PARC_MS_BONE_NAMES,
    default_parc_ms_skeleton_bundle,
    parc_ms_parent_indices,
)
from hhtools.retarget.interaction_mesh.heightfield import posed_meshes_to_heightfield

_log = logging.getLogger(__name__)

_FOLDER_LABEL = "20260429_mocap"
_OBJ_MESH_SCALE_CM_TO_M = 0.01
_BAKE_ORIGIN_EPS = 1e-3
_FOOT_CLEARANCE_WARN_M = 0.10

# PARC body → preferred Spine3 / Mixamo source names (first hit wins).
_PARC_FROM_MOCAP: dict[str, tuple[str, ...]] = {
    "pelvis": ("Hips", "hip", "Pelvis"),
    "torso": ("Spine3", "Spine2", "Spine1", "Spine", "Chest"),
    "head": ("Head",),
    "right_upper_arm": ("RightArm", "RightUpperArm"),
    "right_lower_arm": ("RightForeArm", "RightLowerArm"),
    "right_hand": ("RightHand",),
    "left_upper_arm": ("LeftArm", "LeftUpperArm"),
    "left_lower_arm": ("LeftForeArm", "LeftLowerArm"),
    "left_hand": ("LeftHand",),
    "right_thigh": ("RightUpLeg", "RightThigh"),
    "right_shin": ("RightLeg", "RightShin"),
    "right_foot": ("RightFoot",),
    "left_thigh": ("LeftUpLeg", "LeftThigh"),
    "left_shin": ("LeftLeg", "LeftShin"),
    "left_foot": ("LeftFoot",),
}


@dataclass(frozen=True)
class RigObjPair:
    """One obstacle rigid BVH paired with its local OBJ mesh."""

    bvh_path: Path
    obj_path: Path


@dataclass
class ConvertResult:
    """Outcome of converting one source take folder."""

    clip_name: str
    out_dir: Path | None
    ok: bool
    error: str | None = None
    warnings: list[str] | None = None
    has_terrain: bool = False
    bake_frame: int | None = None
    n_obstacles: int = 0


def _norm_key(text: str) -> str:
    """Lowercase alphanumeric key for rig↔obj matching (``paoku_11`` → ``paoku11``)."""
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _rig_match_key(bvh_path: Path) -> str | None:
    """Derive the obstacle stem key from a ``Take_*_*_rig*.bvh`` / ``PaoKu`` name."""
    stem = bvh_path.stem
    stem = re.sub(r"^take_\d+_", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_rig(?:_\d+)?$", "", stem, flags=re.IGNORECASE)
    key = _norm_key(stem)
    return key or None


def _obj_match_key(obj_path: Path) -> str:
    return _norm_key(obj_path.stem)


def find_skeleton_bvh(clip_dir: Path) -> Path | None:
    """Return the human ``*Skeleton*.bvh`` under ``clip_dir``, if any."""
    clip_dir = Path(clip_dir)
    candidates = sorted(clip_dir.glob("*Skeleton*.bvh")) + sorted(
        clip_dir.glob("*skeleton*.bvh")
    )
    for p in candidates:
        try:
            if bvh_hierarchy_has_joints(p):
                return p
        except OSError:
            continue
    for p in sorted(clip_dir.glob("*.bvh")):
        try:
            if bvh_hierarchy_has_joints(p):
                return p
        except OSError:
            continue
    return None


def match_rig_to_obj(clip_dir: Path) -> list[RigObjPair]:
    """Pair ROOT-only obstacle BVHs with ``obj/*.obj`` by normalised stem.

    Preference rules:
    * Keep every uniquely matched obstacle (individuals, ``*zuhe*``, and
      dedicated ``*_s`` instances).  Combo courses place ``PaoKu_06_s`` /
      ``PaoKu_06_zuhe`` at different world poses than ``paoku_06``; dropping
      ``*_s`` whenever any non-``_s`` exists punched a hole under the actor
      around mid-combo (~9 s on PKO015 takes).
    * An unused ``foo_s.obj`` sidecar (no matching ``*_s`` BVH) is simply
      never paired — there is no blanket drop of paired ``*_s`` rigs.
    * Allow unique prefix fuzzy matches (``paoku02_02_rig`` ↔ ``paoku_02_02_01.obj``).
    * When the folder has exactly one OBJ, bind unmatched rigs to it.
    """
    clip_dir = Path(clip_dir)
    obj_dir = clip_dir / "obj"
    if not obj_dir.is_dir():
        return []

    objs = [p for p in sorted(obj_dir.glob("*.obj")) if p.is_file()]
    if not objs:
        return []
    obj_by_key: dict[str, Path] = {_obj_match_key(obj): obj for obj in objs}

    def _resolve_obj(key: str) -> Path | None:
        if key in obj_by_key:
            return obj_by_key[key]
        prefix_hits = [p for k, p in obj_by_key.items() if k.startswith(key) and k != key]
        if len(prefix_hits) == 1:
            return prefix_hits[0]
        ext_hits = [p for k, p in obj_by_key.items() if key.startswith(k) and k != key]
        if len(ext_hits) == 1:
            return ext_hits[0]
        return None

    pairs: list[RigObjPair] = []
    unmatched_rigs: list[Path] = []
    for bvh in sorted(clip_dir.glob("*.bvh")):
        if not bvh.is_file():
            continue
        try:
            if bvh_hierarchy_has_joints(bvh):
                continue
        except OSError:
            continue
        key = _rig_match_key(bvh)
        if not key or key in {"skeleton0", "skeleton"}:
            continue
        obj = _resolve_obj(key)
        if obj is None:
            unmatched_rigs.append(bvh)
            continue
        pairs.append(RigObjPair(bvh_path=bvh, obj_path=obj))

    if unmatched_rigs and len(objs) == 1:
        only = objs[0]
        for bvh in unmatched_rigs:
            pairs.append(RigObjPair(bvh_path=bvh, obj_path=only))
        unmatched_rigs = []
    for bvh in unmatched_rigs:
        _log.warning(
            "no OBJ for rig %s (key=%s) in %s",
            bvh.name,
            _rig_match_key(bvh),
            clip_dir,
        )

    return pairs


def detect_bake_frame(
    root_positions: NDArray[np.floating],
    *,
    eps: float = _BAKE_ORIGIN_EPS,
) -> int:
    """First frame whose root XY leaves the origin (Maya bind pose is frame 0)."""
    pos = np.asarray(root_positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[0] == 0:
        return 0
    for t in range(int(pos.shape[0])):
        if float(np.linalg.norm(pos[t, :2])) > float(eps):
            return int(t)
    return int(min(1, pos.shape[0] - 1))


def trim_leading_origin_root_frames(
    motion_dict: dict[str, Any],
    *,
    eps: float = _BAKE_ORIGIN_EPS,
) -> int:
    """Drop leading Maya bind-pose frames where root XY sits at the origin.

    MotionBuilder / Maya exports often park the character at world origin on
    frame 0, then teleport to the real start on frame 1.  Leaving that frame
    in the clip makes retarget crawl from the origin under the SQP trust
    region — a multi-metre "flight" into the reference trajectory.
    Returns the number of frames removed.
    """
    root = np.asarray(motion_dict["root_pos"], dtype=np.float64)
    if root.ndim != 2 or root.shape[0] == 0:
        return 0
    t0 = detect_bake_frame(root, eps=eps)
    if t0 <= 0:
        return 0
    if t0 >= int(root.shape[0]):
        return 0
    for key in ("root_pos", "root_rot", "joint_rot", "body_contacts"):
        arr = motion_dict.get(key)
        if arr is None:
            continue
        motion_dict[key] = np.asarray(arr)[t0:]
    return int(t0)

def _sample_heightfield_xy_bounds(hf: TerrainHeightfield) -> tuple[float, float, float, float]:
    nx, ny = hf.shape
    x0 = float(hf.min_point[0])
    y0 = float(hf.min_point[1])
    x1 = x0 + (nx - 1) * float(hf.dx)
    y1 = y0 + (ny - 1) * float(hf.dx)
    return x0, y0, x1, y1


def validate_human_terrain_alignment(
    motion: Motion,
    *,
    bake_frame: int,
    clearance_warn_m: float = _FOOT_CLEARANCE_WARN_M,
) -> list[str]:
    """Return soft warnings when human and heightfield look misaligned."""
    warnings: list[str] = []
    terr = motion.terrain
    if terr is None:
        return warnings

    tx0, ty0, tx1, ty1 = _sample_heightfield_xy_bounds(terr)
    root_xy = np.asarray(motion.positions[:, 0, :2], dtype=np.float64)
    inside_root = (
        (root_xy[:, 0] >= tx0)
        & (root_xy[:, 0] <= tx1)
        & (root_xy[:, 1] >= ty0)
        & (root_xy[:, 1] <= ty1)
    )
    if not bool(np.any(inside_root)):
        f = int(np.clip(bake_frame, 0, motion.num_frames - 1))
        pos = np.asarray(motion.positions[f], dtype=np.float64)
        hx0, hy0 = float(pos[:, 0].min()), float(pos[:, 1].min())
        hx1, hy1 = float(pos[:, 0].max()), float(pos[:, 1].max())
        warnings.append(
            f"human root never enters terrain XY footprint "
            f"(bake human=[{hx0:.2f},{hy0:.2f}]-[{hx1:.2f},{hy1:.2f}] "
            f"terrain=[{tx0:.2f},{ty0:.2f}]-[{tx1:.2f},{ty1:.2f}])"
        )

    foot_i = preferred_floor_contact_bone_indices(tuple(motion.hierarchy.bone_names))
    if foot_i.size == 0:
        return warnings

    n = motion.num_frames
    sample_frames = sorted(
        {
            int(np.clip(bake_frame, 0, n - 1)),
            0,
            n // 4,
            n // 2,
            (3 * n) // 4,
            max(0, n - 1),
            *np.flatnonzero(inside_root)[:: max(1, int(inside_root.sum()) // 8)].tolist(),
        }
    )
    clearances: list[float] = []
    for t in sample_frames:
        for bi in foot_i.tolist():
            xy = motion.positions[t, int(bi), :2]
            if float(xy[0]) < tx0 or float(xy[0]) > tx1 or float(xy[1]) < ty0 or float(xy[1]) > ty1:
                continue
            hz = terr.height_at(float(xy[0]), float(xy[1]))
            clearances.append(float(motion.positions[t, int(bi), 2]) - hz)

    if clearances:
        med = float(np.median(np.abs(clearances)))
        if med > float(clearance_warn_m):
            warnings.append(
                f"median |foot_z - terrain_z| = {med:.3f} m "
                f"(threshold {clearance_warn_m:.3f} m)"
            )
    return warnings


def _maya_y_up_to_z_up_quat() -> NDArray[np.float32]:
    """Fixed Rx(+90°) that maps Maya Y-up mesh verts into hhtools Z-up."""
    # Avoid importing scipy at module import time in tests that only match paths.
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("xyz", [90.0, 0.0, 0.0], degrees=True).as_quat().astype(
        np.float32
    )


def _count_samples_in_posed_footprint(
    vertices_m: NDArray[np.floating],
    position: NDArray[np.floating],
    quat_xyzw: NDArray[np.floating],
    sample_xy: NDArray[np.floating],
) -> int:
    """How many ``sample_xy`` points fall inside the posed mesh's XY convex hull."""
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    if sample_xy.size == 0 or vertices_m.size == 0:
        return 0
    posed = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).apply(
        np.asarray(vertices_m, dtype=np.float64)
    ) + np.asarray(position, dtype=np.float64).reshape(1, 3)
    pts = posed[:, :2]
    if pts.shape[0] < 3:
        return 0
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 0
    poly = pts[hull.vertices]
    samples = np.asarray(sample_xy, dtype=np.float64).reshape(-1, 2)
    # Ray casting (even-odd) against the convex hull polygon.
    x, y = samples[:, 0], samples[:, 1]
    n = poly.shape[0]
    inside = np.zeros(samples.shape[0], dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
        )
        inside ^= intersect
        j = i
    return int(inside.sum())


def _select_obstacle_bake_quat(
    obj_path: Path,
    position: NDArray[np.floating],
    rig_quat_xyzw: NDArray[np.floating],
    *,
    mesh_scale: float,
    sample_xy: NDArray[np.floating] | None,
) -> NDArray[np.float32]:
    """Choose full rig quat vs upright Rx(+90°) so the mesh sits under the actor.

    Some Maya obstacle locators carry a large heading while the exported OBJ is
    already elongated along the course.  Re-applying that yaw swings the
    footprint off the human path (``paoku_07and08`` on combo runs).  Other
    assets (``paoku_06``) need the yaw.  When foot samples are available, pick
    the orientation that covers more of them; otherwise keep the rig quat.
    """
    upright = _maya_y_up_to_z_up_quat()
    rig_q = np.asarray(rig_quat_xyzw, dtype=np.float32).reshape(4)
    if sample_xy is None or np.asarray(sample_xy).size == 0:
        return rig_q

    try:
        import trimesh
    except ImportError:
        return rig_q

    mesh = trimesh.load(str(obj_path), force="mesh", process=False)
    if mesh is None or len(mesh.vertices) == 0:
        return rig_q
    verts = np.asarray(mesh.vertices, dtype=np.float64) * float(mesh_scale)
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    samples = np.asarray(sample_xy, dtype=np.float64).reshape(-1, 2)

    # Restrict scoring to samples near the locator so distant frames don't dominate.
    near = np.linalg.norm(samples - pos[:2], axis=1) < 2.5
    if not bool(np.any(near)):
        return rig_q
    samples = samples[near]

    n_rig = _count_samples_in_posed_footprint(verts, pos, rig_q, samples)
    n_up = _count_samples_in_posed_footprint(verts, pos, upright, samples)
    if n_up > n_rig:
        _log.info(
            "obstacle %s: using upright bake quat (foot hits %d > rig %d)",
            obj_path.name,
            n_up,
            n_rig,
        )
        return upright
    return rig_q


def build_terrain_from_pairs(
    pairs: list[RigObjPair],
    *,
    bake_frame: int | None = None,
    dx: float = 0.05,
    padding: float = 0.5,
    mesh_scale: float = _OBJ_MESH_SCALE_CM_TO_M,
    human_foot_xy: NDArray[np.floating] | None = None,
) -> tuple[TerrainHeightfield, int]:
    """Load rig poses, bake OBJs at ``bake_frame``, return heightfield + frame used.

    When ``human_foot_xy`` (N,2) is provided, each obstacle's bake rotation is
    chosen between the rig world quat and an upright Rx(+90°) so the mesh
    footprint stays under the actor (see :func:`_select_obstacle_bake_quat`).
    """
    if not pairs:
        raise ValueError("no rig/obj pairs to bake")

    loaded: list[tuple[RigObjPair, Motion]] = []
    for pair in pairs:
        rig = load_bvh(pair.bvh_path, unit="cm", target_up_axis="Z")
        loaded.append((pair, rig))

    if bake_frame is None:
        bake = 0
        for _, rig in loaded:
            bake = max(bake, detect_bake_frame(rig.positions[:, 0, :]))
    else:
        bake = int(bake_frame)

    mesh_specs: list[
        tuple[Path, NDArray[np.floating], NDArray[np.floating], float]
    ] = []
    for pair, rig in loaded:
        f = int(np.clip(bake, 0, rig.num_frames - 1))
        pos = np.asarray(rig.positions[f, 0, :], dtype=np.float32)
        quat = _select_obstacle_bake_quat(
            pair.obj_path,
            pos,
            np.asarray(rig.quaternions[f, 0, :], dtype=np.float32),
            mesh_scale=float(mesh_scale),
            sample_xy=human_foot_xy,
        )
        mesh_specs.append((pair.obj_path, pos, quat, float(mesh_scale)))

    terrain = posed_meshes_to_heightfield(
        mesh_specs, dx=float(dx), padding=float(padding)
    )
    if human_foot_xy is not None:
        samples = np.asarray(human_foot_xy, dtype=np.float64).reshape(-1, 2)
        if samples.size:
            terrain = terrain.expanded_to_xy(
                samples.min(axis=0),
                samples.max(axis=0),
                padding=max(float(padding), 1.0),
            )
    return terrain, int(bake)


def _resolve_source_bone_index(
    bone_names: list[str],
    aliases: tuple[str, ...],
    *,
    fallback: int | None = None,
) -> int:
    lower = {n.lower(): i for i, n in enumerate(bone_names)}
    for alias in aliases:
        i = lower.get(alias.lower())
        if i is not None:
            return int(i)
    if fallback is not None:
        return int(fallback)
    raise KeyError(f"none of {aliases} found in {bone_names}")


def _kabsch_quat_xyzw(
    rest: NDArray[np.floating],
    obs: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Rigid rotation (xyzw) aligning rest row-vectors onto obs (same shape ``(N,3)``)."""
    r = np.asarray(rest, dtype=np.float64)
    o = np.asarray(obs, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != 3 or o.shape != r.shape:
        raise ValueError(f"rest/obs must be (N,3); got {r.shape} / {o.shape}")
    if r.shape[0] == 0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if r.shape[0] == 1:
        return _from_to_quat_xyzw(r[0], o[0])
    h = r.T @ o
    u, _s, vt = np.linalg.svd(h)
    d = float(np.linalg.det(vt.T @ u.T))
    rm = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return Q.from_matrix(rm).astype(np.float64, copy=False)


def _from_to_quat_xyzw(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Shortest-arc quaternion rotating unit vector ``a`` onto ``b`` (xyzw)."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    a = a / na
    b = b / nb
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float64)
    v = np.cross(a, b)
    q = np.array([v[0], v[1], v[2], 1.0 + c], dtype=np.float64)
    return q / float(np.linalg.norm(q))


def _parc_world_quats_from_positions(
    mapped_pos: NDArray[np.floating],
    *,
    local_translation: NDArray[np.floating] | None = None,
    parent_indices: NDArray[np.integer] | None = None,
) -> NDArray[np.float32]:
    """Build PARC world quats so FK bone offsets best match mapped joint positions.

    Maya / Mixamo joint frames use local **+Y** as the bone axis, while PARC FK
    places each child with ``R_parent @ local_translation[child]`` (spine along
    **+Z**, arms along **±Y**, etc.).  Copying source world quaternions into
    ``root_rot`` therefore lays the character on its side.  Solving orientations
    from positions avoids that axis mismatch (and absorbs proportion differences
    as residual joint error).
    """
    pos = np.asarray(mapped_pos, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[1:] != (len(PARC_MS_BONE_NAMES), 3):
        raise ValueError(
            f"mapped_pos must be (T, {len(PARC_MS_BONE_NAMES)}, 3); got {pos.shape}"
        )
    t_count = int(pos.shape[0])
    parents = (
        np.asarray(parent_indices, dtype=np.int32)
        if parent_indices is not None
        else parc_ms_parent_indices()
    )
    if local_translation is None:
        local_trans = np.asarray(
            default_parc_ms_skeleton_bundle()[2], dtype=np.float64
        )
    else:
        local_trans = np.asarray(local_translation, dtype=np.float64)
    children: list[list[int]] = [[] for _ in range(len(PARC_MS_BONE_NAMES))]
    for j in range(1, len(PARC_MS_BONE_NAMES)):
        children[int(parents[j])].append(j)

    world = np.zeros((t_count, len(PARC_MS_BONE_NAMES), 4), dtype=np.float64)
    world[:, :, 3] = 1.0
    for j in range(len(PARC_MS_BONE_NAMES)):
        ch = children[j]
        if not ch:
            # Leaf orientation does not affect FK positions; keep identity local.
            p = int(parents[j])
            if p >= 0:
                world[:, j, :] = world[:, p, :]
            continue
        rest = np.stack([local_trans[c] for c in ch], axis=0)
        mask = np.linalg.norm(rest, axis=1) > 1e-8
        if not bool(np.any(mask)):
            p = int(parents[j])
            if p >= 0:
                world[:, j, :] = world[:, p, :]
            continue
        rest_m = rest[mask]
        for f in range(t_count):
            obs = np.stack([pos[f, c] - pos[f, j] for c in ch], axis=0)[mask]
            world[f, j, :] = _kabsch_quat_xyzw(rest_m, obs)
        # Flip sign for temporal continuity (q and -q are the same rotation).
        for f in range(1, t_count):
            if float(np.dot(world[f, j], world[f - 1, j])) < 0.0:
                world[f, j] *= -1.0
    return world.astype(np.float32, copy=False)


def _finalize_parc_ms_motion_dict(motion_dict: dict[str, Any]) -> dict[str, Any]:
    """Trim bind-pose lead-in and return the motion_data dict ready to save."""
    n_trim = trim_leading_origin_root_frames(motion_dict)
    if n_trim:
        _log.info(
            "trimmed %d leading Maya bind-pose frame(s) (root was at origin)",
            n_trim,
        )
    return motion_dict


def mocap_motion_to_parc_ms_dict(
    motion: Motion,
    *,
    loop_mode: str = "CLAMP",
) -> dict[str, Any]:
    """Remap a Spine3 / Mixamo mocap :class:`Motion` onto PARC MS ``motion_data``.

    Returns a dict with ``root_pos (T,3)``, ``root_rot (T,4)``, ``joint_rot (T,14,4)``,
    ``body_contacts (T,15)``, ``fps``, ``loop_mode`` — the same schema as
    ``assets/motions/meshmimic/parc_ms/*.pkl``.

    Orientations are recovered from mapped joint **positions** (Kabsch on PARC
    rest bone offsets) rather than copying Maya world quaternions, which use a
    different bone-axis convention and would leave the character lying down.

    Leading Maya bind-pose frames (root XY at the origin) are dropped so
    retarget does not crawl from world origin into the real trajectory.
    """
    names = list(motion.hierarchy.bone_names)
    t_count = int(motion.num_frames)
    src_pos = np.asarray(motion.positions, dtype=np.float32)

    pelvis_i = _resolve_source_bone_index(names, _PARC_FROM_MOCAP["pelvis"])
    mapped = np.zeros((t_count, len(PARC_MS_BONE_NAMES), 3), dtype=np.float32)
    for pi, parc_name in enumerate(PARC_MS_BONE_NAMES):
        si = _resolve_source_bone_index(
            names,
            _PARC_FROM_MOCAP[parc_name],
            fallback=pelvis_i,
        )
        mapped[:, pi, :] = src_pos[:, si, :]

    parents = parc_ms_parent_indices()
    world = _parc_world_quats_from_positions(mapped, parent_indices=parents)
    local = world_quaternions_to_local(parents, world).astype(np.float32, copy=False)
    joint_rot = local[:, 1:, :].copy()
    body_contacts = np.zeros((t_count, len(PARC_MS_BONE_NAMES)), dtype=np.float32)

    out = {
        "root_pos": mapped[:, 0, :].copy(),
        "root_rot": world[:, 0, :].copy(),
        "joint_rot": joint_rot,
        "body_contacts": body_contacts,
        "fps": int(round(float(motion.framerate))),
        "loop_mode": str(loop_mode),
    }
    return _finalize_parc_ms_motion_dict(out)

def convert_clip(
    src_dir: str | Path,
    out_root: str | Path,
    *,
    clip_name: str | None = None,
    overwrite: bool = False,
    dx: float = 0.05,
    validate: bool = True,
) -> ConvertResult:
    """Convert one ``20260429_mocap`` take folder into a parc_ms-style clip directory."""
    src_dir = Path(src_dir).resolve()
    out_root = Path(out_root)
    name = clip_name or src_dir.name
    out_dir = out_root / name

    try:
        skel = find_skeleton_bvh(src_dir)
        if skel is None:
            raise FileNotFoundError(f"no Skeleton BVH in {src_dir}")

        human = load_bvh(skel, unit="cm", target_up_axis="Z")
        pairs = match_rig_to_obj(src_dir)

        terrain: TerrainHeightfield | None = None
        bake_frame: int | None = None
        if pairs:
            foot_i = preferred_floor_contact_bone_indices(
                tuple(human.hierarchy.bone_names)
            )
            foot_xy = None
            if foot_i.size:
                # Subsample feet along the clip for obstacle-orientation scoring.
                step = max(1, int(human.num_frames) // 400)
                foot_xy = np.asarray(
                    human.positions[::step][:, foot_i, :2], dtype=np.float64
                ).reshape(-1, 2)
            terrain, bake_frame = build_terrain_from_pairs(
                pairs, dx=dx, human_foot_xy=foot_xy
            )

        check_motion = Motion(
            name=name,
            hierarchy=human.hierarchy,
            positions=human.positions,
            quaternions=human.quaternions,
            framerate=human.framerate,
            up_axis=human.up_axis,
            source_format=human.source_format,
            meta={
                "dataset": "parc_ms",
                "library_folder_label": _FOLDER_LABEL,
                "split_terrain_grounding": True,
                "mocap_source_take_dir": str(src_dir),
                "source_skeleton_bvh": skel.name,
                **(
                    {"terrain_heightfield_frame": int(bake_frame)}
                    if bake_frame is not None
                    else {}
                ),
            },
            objects=[],
            terrain=terrain,
        )

        warnings: list[str] = []
        if validate and terrain is not None and bake_frame is not None:
            warnings = validate_human_terrain_alignment(
                check_motion, bake_frame=bake_frame
            )

        motion_dict = mocap_motion_to_parc_ms_dict(human)

        if out_dir.exists():
            if not overwrite:
                raise FileExistsError(f"output exists (pass overwrite=True): {out_dir}")
            for pattern in (f"{name}.npz", f"{name}.pkl", f"{name}_terrain.obj"):
                old = out_dir / pattern
                if old.is_file() or old.is_symlink():
                    old.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)

        save_parc_pkl(
            out_dir / f"{name}.pkl",
            motion_data=motion_dict,
            terrain_data=terrain,
            misc_data={
                "library_folder_label": _FOLDER_LABEL,
                "mocap_source_take_dir": str(src_dir),
                "source_skeleton_bvh": skel.name,
                "terrain_heightfield_frame": (
                    int(bake_frame) if bake_frame is not None else None
                ),
                "obstacle_pairs": [
                    {"bvh": p.bvh_path.name, "obj": p.obj_path.name} for p in pairs
                ],
            },
        )
        if terrain is not None:
            heightfield_to_wavefront_obj(terrain, out_dir / f"{name}_terrain.obj")

        for w in warnings:
            _log.warning("%s: %s", name, w)

        return ConvertResult(
            clip_name=name,
            out_dir=out_dir,
            ok=True,
            warnings=warnings or None,
            has_terrain=terrain is not None,
            bake_frame=bake_frame,
            n_obstacles=len(pairs),
        )
    except Exception as exc:
        return ConvertResult(
            clip_name=name,
            out_dir=None,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def list_source_clips(src_root: str | Path) -> list[Path]:
    """Return take folders under ``src_root`` that contain a skeleton BVH."""
    src_root = Path(src_root)
    clips: list[Path] = []
    for child in sorted(src_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if find_skeleton_bvh(child) is not None:
            clips.append(child)
    return clips


__all__ = [
    "ConvertResult",
    "RigObjPair",
    "build_terrain_from_pairs",
    "convert_clip",
    "detect_bake_frame",
    "find_skeleton_bvh",
    "list_source_clips",
    "match_rig_to_obj",
    "mocap_motion_to_parc_ms_dict",
    "trim_leading_origin_root_frames",
    "validate_human_terrain_alignment",
]
