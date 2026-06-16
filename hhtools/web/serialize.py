"""Serialise hhtools data structures into browser-friendly JSON / GLB.

The front-end (three.js) needs:

* **Skeletons**: bone names, parent indices, and per-frame global positions
  (+ optionally quaternions).  We downsample long clips so the payload stays
  light and animation stays smooth in the browser.
* **Scene objects** (OMOMO props, terrain): per-frame transforms + a GLB blob
  for the actual geometry, fetched lazily.
* **Robots**: a static GLB of all link visual meshes posed at the link-local
  frame, plus per-link parent/transform metadata so the browser can pose the
  robot per frame from a DOF trajectory.

Everything here is import-light (numpy only) except the GLB helpers, which
import ``trimesh`` lazily.
"""

from __future__ import annotations

import base64
import gzip
from pathlib import Path
from typing import Any

import numpy as np

from hhtools.core.motion import Motion

# Playback frame cap for browser payloads.  ``0`` = send every source frame
# (no downsampling).  Set to e.g. 600 to bound JSON size on slow links.
_MAX_PLAYBACK_FRAMES = 0
# Baked SMPL mesh frames (independent of skeleton when > 0).  ``0`` = all frames.
_MAX_MESH_FRAMES = 0
# Compressed mesh size limit; ``0`` = no limit.
_MAX_MESH_GZIP_BYTES = 0


def skeleton_exclude_joint_indices(motion: Motion) -> list[int]:
    """Bone indices to skip in stick/capsule viz (matches Viser defaults)."""
    from hhtools.viewer.anatomy import (
        degenerate_auxiliary_bone_indices,
        detect_virtual_root,
    )

    ex: set[int] = set(degenerate_auxiliary_bone_indices(motion))
    if detect_virtual_root(list(motion.hierarchy.bone_names)):
        ex.add(0)
    return sorted(ex)


def _motion_salient_frame_indices(
    motion: Motion,
    *,
    min_turn_deg: float = 35.0,
) -> np.ndarray:
    """Source frames to preserve when downsampling (sharp heading changes).

    Long clips (e.g. LAFAN折返跑) are capped to ``_MAX_PLAYBACK_FRAMES`` for
    transport.  Uniform ``linspace`` drops most turn frames; the browser then
    linearly blends between sparse keys and the runner *glides* through corners.
    Keeping salient frames makes preview faithful without sending every frame.
    """
    n = int(motion.num_frames)
    if n < 3:
        return np.zeros(0, dtype=np.int64)

    hip_idx = 0
    for i, name in enumerate(motion.hierarchy.bone_names):
        low = str(name).lower()
        if low in ("hips", "pelvis", "root") or low.endswith("hips"):
            hip_idx = i
            break

    xy = np.asarray(motion.positions[:, hip_idx, :2], dtype=np.float64)
    dxy = np.diff(xy, axis=0)
    speed = np.linalg.norm(dxy, axis=1)
    if not np.any(speed > 1e-5):
        return np.zeros(0, dtype=np.int64)

    angles = np.arctan2(dxy[:, 1], dxy[:, 0])
    dangle = np.abs(np.diff(angles))
    dangle = np.minimum(dangle, 2.0 * np.pi - dangle)
    thresh = np.radians(float(min_turn_deg))
    # +1: index into positions/quaternions at the frame *after* the heading jump.
    turns = np.where(dangle > thresh)[0] + 1
    return turns.astype(np.int64, copy=False)


def _downsample_indices(
    num_frames: int,
    max_frames: int,
    *,
    motion: Motion | None = None,
) -> np.ndarray:
    if max_frames <= 0 or num_frames <= max_frames:
        return np.arange(num_frames, dtype=np.int64)

    must: set[int] = {0, num_frames - 1}
    if motion is not None:
        for f in _motion_salient_frame_indices(motion):
            fi = int(f)
            if 0 <= fi < num_frames:
                must.add(fi)
                if fi > 0:
                    must.add(fi - 1)
                if fi + 1 < num_frames:
                    must.add(fi + 1)

    must_sorted = sorted(must)
    if len(must_sorted) >= max_frames:
        pick = np.linspace(0, len(must_sorted) - 1, max_frames).round().astype(np.int64)
        return np.asarray([must_sorted[int(i)] for i in pick], dtype=np.int64)

    room = max_frames - len(must_sorted)
    uniform = np.linspace(0, num_frames - 1, room + 2).round().astype(np.int64)[1:-1]
    merged = sorted(must | {int(x) for x in uniform})
    return np.asarray(merged[:max_frames], dtype=np.int64)


def _playback_fields(
    *,
    num_frames_total: int,
    playback_frames: int,
    framerate: float,
    full_duration: float,
) -> dict[str, Any]:
    """Timeline metadata for browser playback after optional downsampling.

    ``playback_duration`` is ALWAYS the full clip duration, never the
    downsampled-frame-count duration.  The browser spreads ``playback_frames``
    evenly across this duration (it interpolates between them), so a clip
    downsampled from e.g. 3000 → 600 frames must still report the full
    ~100 s timeline.  Reporting ``(playback_frames - 1) / framerate`` instead
    made downsampled clips (especially long LAFAN sequences) play several
    times too fast.
    """
    out: dict[str, Any] = {"playback_frames": int(playback_frames)}
    out["playback_duration"] = float(full_duration)
    return out


def serialize_motion(
    motion: Motion,
    *,
    max_frames: int = _MAX_PLAYBACK_FRAMES,
    include_quaternions: bool = False,
    progress_callback=None,
) -> dict[str, Any]:
    """Serialise a :class:`Motion` skeleton + animation for the browser.

    Coordinates are kept in hhtools' native frame (Z-up, metres).  The
    front-end applies a Z-up→Y-up rotation for three.js.
    """
    idx = _downsample_indices(motion.num_frames, max_frames, motion=motion)
    positions = np.asarray(motion.positions, dtype=np.float32)[idx]  # (F, J, 3)
    parents = np.asarray(motion.hierarchy.parent_indices, dtype=np.int32).tolist()

    playback_frames = int(len(idx))
    payload: dict[str, Any] = {
        "name": motion.name,
        "source_format": motion.source_format,
        "up_axis": motion.up_axis,
        "framerate": float(motion.framerate),
        "num_frames_total": int(motion.num_frames),
        "duration": float(motion.duration),
        "bone_names": list(motion.bone_names),
        "parent_indices": parents,
        "exclude_joint_indices": skeleton_exclude_joint_indices(motion),
        "frame_indices": idx.tolist(),
        # (F, J, 3) → nested lists; float32 keeps it compact enough.
        "positions": np.round(positions, 4).tolist(),
        "objects": [_serialize_object_meta(o, idx) for o in motion.objects],
        "has_terrain": motion.terrain is not None,
        "meta": _safe_meta(motion.meta),
        **_playback_fields(
            num_frames_total=motion.num_frames,
            playback_frames=playback_frames,
            framerate=motion.framerate,
            full_duration=motion.duration,
        ),
    }
    if include_quaternions:
        quats = np.asarray(motion.quaternions, dtype=np.float32)[idx]
        payload["quaternions"] = np.round(quats, 5).tolist()
    if motion.terrain is not None:
        payload["terrain"] = _serialize_terrain(motion.terrain)
    if progress_callback is not None:
        progress_callback(0.55, "序列化骨架…")
    payload["body_mesh"] = _serialize_body_mesh(
        motion, idx, progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(1.0, "序列化完成")
    return payload


def _serialize_body_mesh(
    motion: Motion,
    idx: np.ndarray,
    *,
    progress_callback=None,
) -> dict[str, Any]:
    """SMPL/GLB skinned surface for the browser (gzip float32 vertex cache).

    Returns ``{"available": false}`` when the motion has no mesh attachment or
    the compressed payload would be too large for a single JSON response.
    """
    from hhtools.core.skinning import BakedMesh

    meta = motion.meta if isinstance(motion.meta, dict) else {}
    baked = meta.get("baked_mesh")
    if not isinstance(baked, BakedMesh):
        reason = meta.get("baked_mesh_error") or meta.get("baked_mesh_unavailable")
        if reason:
            return {"available": False, "reason": str(reason)}
        return {"available": False, "reason": "no skinned mesh on this clip"}

    if _MAX_MESH_FRAMES <= 0:
        pick = np.arange(baked.num_frames, dtype=np.int64)
    else:
        n_mesh = min(_MAX_MESH_FRAMES, len(idx))
        pick = np.linspace(0, baked.num_frames - 1, n_mesh).round().astype(np.int64)
    verts = np.asarray(baked.vertices, dtype=np.float32)[pick]

    if progress_callback is not None:
        progress_callback(
            0.72,
            f"压缩身体网格 {verts.shape[0]} 帧 × {baked.num_vertices} 顶点…",
        )
    raw = np.ascontiguousarray(verts).tobytes()
    gz = gzip.compress(raw, compresslevel=6)
    if _MAX_MESH_GZIP_BYTES > 0 and len(gz) > _MAX_MESH_GZIP_BYTES:
        return {
            "available": False,
            "reason": "mesh too large — use Viser UI or a shorter clip",
        }
    return {
        "available": True,
        "type": "baked",
        "num_verts": int(baked.num_vertices),
        "num_frames": int(verts.shape[0]),
        "triangles": np.asarray(baked.triangles, dtype=np.int32).tolist(),
        "vertices_gz_b64": base64.b64encode(gz).decode("ascii"),
    }


def _safe_meta(meta: dict) -> dict[str, Any]:
    """Keep only JSON-serialisable scalar/list metadata."""
    out: dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[str(k)] = v
        elif isinstance(v, (list, tuple)) and all(
            isinstance(x, (str, int, float, bool)) for x in v
        ):
            out[str(k)] = list(v)
    return out


def _serialize_object_meta(obj, idx: np.ndarray) -> dict[str, Any]:
    pos = np.asarray(obj.positions, dtype=np.float32)[idx]
    quat = np.asarray(obj.quaternions, dtype=np.float32)[idx]
    return {
        "name": obj.name,
        "extents": np.asarray(obj.extents, dtype=np.float32).tolist(),
        "has_mesh": bool(obj.mesh_path),
        "scale": float(obj.scale),
        "color": list(obj.color) if obj.color else None,
        "opacity": obj.opacity,
        "positions": np.round(pos, 4).tolist(),
        "quaternions": np.round(quat, 5).tolist(),
    }


def _serialize_terrain(terrain) -> dict[str, Any]:
    """Triangulated terrain mesh for three.js (same topology as Viser).

    Uses :meth:`TerrainHeightfield.triangulate` on a downsampled grid so the
    browser does not have to guess PlaneGeometry vertex ordering.
    """
    from hhtools.core.scene import TerrainHeightfield

    hf = np.asarray(terrain.hf, dtype=np.float32)
    dx = float(terrain.dx)
    min_point = np.asarray(terrain.min_point, dtype=np.float32)
    hf_mm = np.asarray(terrain.hf_maxmin, dtype=np.float32)
    nx0, ny0 = hf.shape[0], hf.shape[1]
    # Keep the grid dense: striding a step terrain skips the sharp risers and
    # makes stairs look like ramps.  256 keeps ~65k cells which the browser
    # handles fine, and preserves vertical edges.
    max_dim = 256
    rstep = max(1, int(np.ceil(nx0 / max_dim)))
    cstep = max(1, int(np.ceil(ny0 / max_dim)))
    th = TerrainHeightfield(
        hf=hf[::rstep, ::cstep],
        hf_maxmin=hf_mm[::rstep, ::cstep, :],
        min_point=min_point,
        dx=dx * float(rstep),
    )
    verts, faces = th.triangulate()
    return {
        "vertices": np.round(verts, 4).tolist(),
        "faces": faces.astype(np.int32).tolist(),
    }


# Canonical human topology for scaled-effector preview (matches Viser yellow overlay).
_CANONICAL_PARENT: dict[str, str | None] = {
    "hips": None,
    "spine": "hips",
    "chest": "spine",
    "neck": "chest",
    "head": "neck",
    "left_shoulder": "chest",
    "left_elbow": "left_shoulder",
    "left_wrist": "left_elbow",
    "right_shoulder": "chest",
    "right_elbow": "right_shoulder",
    "right_wrist": "right_elbow",
    "left_hip": "hips",
    "left_knee": "left_hip",
    "left_ankle": "left_knee",
    "left_foot": "left_ankle",
    "right_hip": "hips",
    "right_knee": "right_hip",
    "right_ankle": "right_knee",
    "right_foot": "right_ankle",
}


def _prune_canonical_names(
    names: list[str],
    ik_map_canonicals: frozenset[str],
) -> list[str]:
    """Drop canonical joints below the deepest mapped IK target on each limb."""
    from hhtools.viewer.anatomy import deepest_mapped_canonicals

    terminals = deepest_mapped_canonicals(ik_map_canonicals)
    if not terminals:
        return names

    children: dict[str, list[str]] = {}
    for child, parent in _CANONICAL_PARENT.items():
        if parent is not None:
            children.setdefault(parent, []).append(child)

    drop: set[str] = set()

    def _mark_descendants(node: str) -> None:
        for ch in children.get(node, ()):
            drop.add(ch)
            _mark_descendants(ch)

    for term in terminals:
        _mark_descendants(term)

    return [n for n in names if n not in drop]


def serialize_motion_skeleton_preview(
    motion: Motion,
    *,
    max_frames: int = _MAX_PLAYBACK_FRAMES,
) -> dict[str, Any]:
    """Lightweight stick-figure payload from a :class:`Motion` (no mesh / env)."""
    idx = _downsample_indices(motion.num_frames, max_frames, motion=motion)
    positions = np.asarray(motion.positions, dtype=np.float32)[idx]
    parents = np.asarray(motion.hierarchy.parent_indices, dtype=np.int32).tolist()
    playback_frames = int(len(idx))
    full_duration = float(motion.duration)
    return {
        "name": motion.name,
        "bone_names": list(motion.bone_names),
        "parent_indices": parents,
        "frame_indices": idx.tolist(),
        "positions": np.round(positions, 4).tolist(),
        "num_frames_total": int(motion.num_frames),
        "framerate": float(motion.framerate),
        "duration": full_duration,
        **_playback_fields(
            num_frames_total=motion.num_frames,
            playback_frames=playback_frames,
            framerate=motion.framerate,
            full_duration=full_duration,
        ),
    }


def serialize_scaled_preview(
    preview,
    *,
    max_frames: int = _MAX_PLAYBACK_FRAMES,
    ik_map_canonicals: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Serialise :class:`~hhtools.retarget.newton_basic.ScaledMotionPreview` for the browser."""
    idx = _downsample_indices(preview.num_frames, max_frames)
    names = list(preview.joint_names)
    if ik_map_canonicals:
        names = _prune_canonical_names(names, ik_map_canonicals)
    name_to_i = {n: i for i, n in enumerate(names)}
    parents: list[int] = []
    for n in names:
        p = _CANONICAL_PARENT.get(n)
        parents.append(-1 if p is None or p not in name_to_i else name_to_i[p])

    tr = np.asarray(preview.transforms, dtype=np.float32)[idx]  # (F, M, 7)
    positions = tr[:, :, :3]
    return {
        "name": "scaled_targets",
        "bone_names": names,
        "parent_indices": parents,
        "frame_indices": idx.tolist(),
        "positions": np.round(positions, 4).tolist(),
        "num_frames_total": int(preview.num_frames),
        "framerate": 30.0,
        "duration": float(preview.num_frames) / 30.0,
    }


# --------------------------------------------------------------------------- GLB helpers


def object_mesh_glb(obj, *, scale: float | None = None) -> bytes | None:
    """Export a SceneObject's mesh to GLB bytes, or ``None`` if unavailable.

    Centres on the mesh geometric centroid then applies ``scale`` (override or
    ``obj.scale``), matching :func:`hhtools.viewer.renderers.objects._load_mesh_arrays`
    and OMOMO ``positions`` semantics.  Pass an explicit ``scale`` for robot-frame
    scaled props (``obj.scale * uniform_ratio``).
    """
    if not obj.mesh_path:
        return None
    try:
        import trimesh

        loaded = trimesh.load(obj.mesh_path, force="mesh", process=False)
        verts = np.asarray(getattr(loaded, "vertices", np.zeros((0, 3))), dtype=np.float64)
        faces = np.asarray(getattr(loaded, "faces", np.zeros((0, 3), dtype=np.int64)), dtype=np.int64)
        if verts.size == 0 or faces.size == 0:
            return None
        centroid = verts.mean(axis=0)
        eff = float(scale) if scale is not None else float(obj.scale or 1.0)
        verts = ((verts - centroid) * eff).astype(np.float32)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        return mesh.export(file_type="glb")
    except Exception:
        return None


# --------------------------------------------------------------------------- robot


def _mesh_to_link_payload(model) -> dict[str, str]:
    """Map trimesh / GLB geometry node names to URDF link names for ray picking.

    SolidWorks and other exporters name GLB nodes after mesh files
    (``Left_hip_pitch.STL``) rather than link names (``left_hip_pitch_link``).
    Calibration drag in the web UI needs the authoritative URDF mapping.
    """
    out: dict[str, str] = {}
    link_map = getattr(model.urdf, "link_map", None) or {}
    for link_name, link in link_map.items():
        out.setdefault(str(link_name), str(link_name))
        for attr in ("visuals", "collisions"):
            for item in getattr(link, attr, None) or []:
                mesh = getattr(getattr(item, "geometry", None), "mesh", None)
                fname = getattr(mesh, "filename", None) if mesh is not None else None
                if not fname:
                    continue
                path = Path(str(fname))
                for key in (path.name, path.stem):
                    if key:
                        out.setdefault(key, str(link_name))
    return out


def serialize_robot(model, *, name: str) -> dict[str, Any]:
    """Serialise a URDFRobotModel: link metadata + a GLB of the zero-pose scene.

    GLB mesh nodes are usually named after link frames, but some URDF exporters
    (SolidWorks, Onshape, …) emit geometry nodes named after mesh files; see
    ``mesh_to_link`` for the authoritative node→link map used by calibration
    picking in the browser.
    """
    from hhtools.robot.dof_schema import header_columns  # noqa: F401 - validate import

    actuated = [j.name for j in model.actuated_joints]
    links = [link.name for link in model.links]

    # Per-link world transforms at zero configuration (Z-up, metres).
    model.apply_configuration(model.zero_configuration())
    link_transforms: dict[str, list[float]] = {}
    for link in links:
        try:
            T = model.urdf.get_transform(link)
            link_transforms[link] = np.asarray(T, dtype=np.float32).flatten().tolist()
        except Exception:
            link_transforms[link] = np.eye(4, dtype=np.float32).flatten().tolist()

    glb_b64: str | None = None
    ground_offset_z = 0.0
    try:
        scene = model.trimesh_scene()
        glb_bytes = scene.export(file_type="glb")
        glb_b64 = base64.b64encode(glb_bytes).decode("ascii")
        ground_offset_z = _ground_offset_z(scene)
    except Exception:
        glb_b64 = None

    ik_map = dict(model.preset.ik_map) if model.preset.ik_map else {}
    return {
        "name": name,
        "display_name": model.preset.display_name,
        "base_link": model.base_link,
        "links": links,
        "actuated_joints": actuated,
        "num_dof": len(actuated),
        "ik_map": ik_map,
        "ik_prewarmed": False,
        "link_transforms_zero": link_transforms,
        "mesh_to_link": _mesh_to_link_payload(model),
        "glb_base64": glb_b64,
        # Vertical lift so the lowest mesh vertex sits on z=0 at the zero pose
        # (URDFs often place the base at the pelvis, leaving feet below ground).
        "ground_offset_z": ground_offset_z,
    }


def _ground_offset_z(scene) -> float:
    """Lift (metres) so the scene's lowest vertex rests on the ground plane."""
    min_z = _scene_min_mesh_z(scene)
    return max(0.0, -min_z) if min_z is not None else 0.0


def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _resolve_ik_link(ik_map: dict[str, Any], canonical: str) -> str | None:
    spec = ik_map.get(canonical)
    if spec is None:
        return None
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        link = spec.get("link") or spec.get("body")
        return str(link) if link is not None else None
    return str(spec)


def _scene_min_mesh_z(scene, root_rot: np.ndarray | None = None) -> float | None:
    """Lowest mesh vertex z with optional floating-base rotation (no translation)."""
    try:
        import trimesh
    except Exception:
        return None
    R = np.eye(3, dtype=np.float64) if root_rot is None else np.asarray(root_rot, dtype=np.float64)
    T_root = np.eye(4, dtype=np.float64)
    T_root[:3, :3] = R
    min_z: float | None = None
    for node_name in scene.graph.nodes_geometry:
        mat, geom_name = scene.graph[node_name]
        if geom_name is None:
            continue
        geom = scene.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh) or geom.is_empty:
            continue
        mat_world = T_root @ np.asarray(mat, dtype=np.float64)
        v = np.asarray(geom.vertices, dtype=np.float64)
        z = (
            mat_world[2, 0] * v[:, 0]
            + mat_world[2, 1] * v[:, 1]
            + mat_world[2, 2] * v[:, 2]
            + mat_world[2, 3]
        ).min()
        if min_z is None or z < min_z:
            min_z = float(z)
    return min_z


def _lowest_ankle_z(model, ik_map: dict[str, Any], root_rot: np.ndarray) -> float | None:
    """Lowest ankle link origin z with floating-base rotation (no translation)."""
    T_root = np.eye(4, dtype=np.float64)
    T_root[:3, :3] = np.asarray(root_rot, dtype=np.float64)
    ankle_zs: list[float] = []
    for canonical in ("left_ankle", "right_ankle"):
        link = _resolve_ik_link(ik_map, canonical)
        if not link:
            continue
        try:
            T_link = np.asarray(model.urdf.get_transform(link), dtype=np.float64)
        except Exception:
            continue
        ankle_zs.append(float((T_root @ T_link)[2, 3]))
    return min(ankle_zs) if ankle_zs else None


def _sole_depth_reference(model, ik_map: dict[str, Any]) -> float | None:
    """Ankle→sole distance at the zero pose (metres, positive when sole is below ankle)."""
    model.apply_configuration(model.zero_configuration())
    scene = model.trimesh_scene()
    ankle_z = _lowest_ankle_z(model, ik_map, np.eye(3, dtype=np.float64))
    min_mesh_z = _scene_min_mesh_z(scene)
    if ankle_z is None or min_mesh_z is None:
        return None
    return float(ankle_z - min_mesh_z)


def _mesh_playback_z_lift(
    model,
    cfg: dict[str, float],
    root_xyzw: np.ndarray,
    *,
    sole_depth_ref: float | None,
    ik_map: dict[str, Any],
    yellow_foot_z: float | None = None,
) -> float:
    """Rigid Z lift on the browser ``group`` during trajectory playback.

    When ``yellow_foot_z`` is supplied (from the scaled overlay), lift the mesh
    so its lowest vertex matches that height — the yellow skeleton uses uniform
    scaling while IK uses per-joint scale, so scheme-A ankle→sole alone leaves
    a standing gap.  Because ``yellow_foot_z`` follows the clip, jumps stay in
    the air (no per-frame snap to ``z=0``).

    Without overlay data, fall back to scheme A (ankle→sole at ``cfg`` only).
    """
    model.apply_configuration(cfg)
    scene = model.trimesh_scene()
    root_rot = _quat_xyzw_to_rotmat(root_xyzw[3:7])
    min_mesh_z = _scene_min_mesh_z(scene, root_rot)
    if min_mesh_z is None:
        return 0.0
    if yellow_foot_z is not None:
        return float(yellow_foot_z - float(root_xyzw[2]) - min_mesh_z)
    if sole_depth_ref is None:
        # Robots without ankle links in ik_map: rest the lowest mesh vertex on z=0.
        return float(-float(root_xyzw[2]) - min_mesh_z)
    ankle_z = _lowest_ankle_z(model, ik_map, root_rot)
    if ankle_z is None:
        return 0.0
    expected_min_mesh = float(ankle_z - sole_depth_ref)
    return float(expected_min_mesh - min_mesh_z)


def _scaled_overlay_foot_z(scaled_preview: dict[str, Any], playback_i: int) -> float | None:
    """Lowest foot/ankle Z on the yellow scaled overlay at playback frame ``playback_i``."""

    def _norm(name: str) -> str:
        return str(name).lower().replace("_", "").replace(" ", "")

    names = scaled_preview.get("bone_names") or []
    positions = scaled_preview.get("positions") or []
    if playback_i < 0 or playback_i >= len(positions):
        return None
    pos = positions[playback_i]
    name_to_i = {_norm(n): i for i, n in enumerate(names)}
    zs: list[float] = []
    for key in (
        "leftankle",
        "rightankle",
        "leftfoot",
        "rightfoot",
        "leftleg",
        "rightleg",
    ):
        idx = name_to_i.get(key)
        if idx is not None and idx < len(pos):
            zs.append(float(pos[idx][2]))
    return min(zs) if zs else None


def serialize_robot_trajectory(
    model,
    retargeted,
    *,
    scaled_preview: dict[str, Any] | None = None,
    max_frames: int = _MAX_PLAYBACK_FRAMES,
    ground_follow: bool = False,
):
    """Per-frame root transform + DOF values + per-link world transforms.

    ``retargeted`` is a :class:`RetargetedMotion`.  We compute link world
    transforms per frame with yourdfpy FK so the browser only needs to set
    each link group's matrix.

    ``ground_follow`` selects the vertical-grounding scheme for ``mesh_z_lift``:

    * ``False`` (default): a **single constant** lift, computed once, is reused
      for every frame.  The mesh sole rests on the ground at the standing pose
      and the clip's own root-Z then carries the body up/down — so tumbling and
      backflips do **not** snap upward.  Prone / crawl penetration is handled
      in the IK pipeline (body-ground clearance + post-solve foot clamp), not
      by per-frame mesh lifting here.
    * ``True``: the per-frame foot-follow correction (mesh sole tracks the
      yellow overlay foot) used for climbing / terrain so the robot stays glued
      to a rising surface.  Costs one ``trimesh_scene`` rebuild per frame.
    """
    root = np.asarray(retargeted.root_trajectory, dtype=np.float32)  # (F, 7) xyz+xyzw
    dof = np.asarray(retargeted.dof_trajectory, dtype=np.float32)  # (F, D)
    num_frames = root.shape[0]
    idx = _downsample_indices(num_frames, max_frames)
    links = [link.name for link in model.links]

    # Map retargeted dof_names → model actuated joints (order may differ).
    ret_dof_names = list(retargeted.dof_names)
    ik_map = dict(model.preset.ik_map) if model.preset.ik_map else {}
    sole_depth_ref = _sole_depth_reference(model, ik_map)

    # When foot-follow is off, compute the grounding lift ONCE at the first
    # played frame (scheme A, no overlay tracking) and reuse it for all frames.
    const_lift = 0.0
    if not ground_follow and len(idx) > 0:
        f0 = int(idx[0])
        cfg0 = {ret_dof_names[i]: float(dof[f0, i]) for i in range(len(ret_dof_names))}
        root0 = root[f0]
        lift0 = _mesh_playback_z_lift(
            model,
            cfg0,
            root0,
            sole_depth_ref=sole_depth_ref,
            ik_map=ik_map,
            yellow_foot_z=None,
        )
        # Exported robot trajectories carry an absolute floating-base root z.
        # Shift the constant group lift so the lowest mesh vertex rests on z=0
        # at the first played frame (scheme A + root-height correction).
        model.apply_configuration(cfg0)
        root_rot0 = _quat_xyzw_to_rotmat(root0[3:7])
        min_mesh_z0 = _scene_min_mesh_z(model.trimesh_scene(), root_rot0)
        if min_mesh_z0 is not None:
            world_min = float(root0[2]) + lift0 + min_mesh_z0
            const_lift = float(lift0 - world_min)
        else:
            const_lift = lift0

    frames: list[dict[str, Any]] = []
    for pi, f in enumerate(idx):
        cfg = {ret_dof_names[i]: float(dof[f, i]) for i in range(len(ret_dof_names))}
        model.apply_configuration(cfg)
        link_T: dict[str, list[float]] = {}
        for link in links:
            try:
                T = np.asarray(model.urdf.get_transform(link), dtype=np.float32)
            except Exception:
                T = np.eye(4, dtype=np.float32)
            link_T[link] = T.flatten().tolist()
        if ground_follow:
            yellow_foot = (
                _scaled_overlay_foot_z(scaled_preview, pi)
                if scaled_preview is not None
                else None
            )
            mesh_z_lift = _mesh_playback_z_lift(
                model,
                cfg,
                root[f],
                sole_depth_ref=sole_depth_ref,
                ik_map=ik_map,
                yellow_foot_z=yellow_foot,
            )
        else:
            mesh_z_lift = const_lift
        frames.append(
            {
                "root": np.round(root[f], 5).tolist(),
                "links": link_T,
                "mesh_z_lift": round(mesh_z_lift, 5),
            }
        )

    # Do **not** post-shift root z here.  Playback adds per-frame ``mesh_z_lift``
    # on the browser ``group`` (mesh sole ↔ yellow overlay foot, or scheme A).

    framerate = float(getattr(retargeted, "sample_rate", 30.0))
    playback_frames = int(len(idx))
    full_duration = max(0.1, (num_frames - 1) / framerate) if num_frames > 1 else 0.1
    return {
        "framerate": framerate,
        "num_frames_total": int(num_frames),
        "duration": full_duration,
        "frame_indices": idx.tolist(),
        "frames": frames,
        **_playback_fields(
            num_frames_total=num_frames,
            playback_frames=playback_frames,
            framerate=framerate,
            full_duration=full_duration,
        ),
    }


def resample_joint_q(
    joint_q: np.ndarray,
    src_fps: float,
    dst_fps: float,
    *,
    root_coord_count: int = 7,
) -> np.ndarray:
    """Resample a ``(F, 7+D)`` robot trajectory from ``src_fps`` to ``dst_fps``.

    Linear interpolation for translation + DOFs; quaternion columns
    (``root_coord_count-4 : root_coord_count``, xyzw) are slerp-interpolated
    via normalised lerp (good enough for dense sampling) so the root rotation
    stays unit-norm.
    """
    q = np.asarray(joint_q, dtype=np.float64)
    f = q.shape[0]
    if f < 2 or src_fps <= 0 or dst_fps <= 0 or abs(src_fps - dst_fps) < 1e-6:
        return np.asarray(joint_q, dtype=np.float32)
    duration = (f - 1) / src_fps
    n_out = max(2, int(round(duration * dst_fps)) + 1)
    src_t = np.arange(f) / src_fps
    dst_t = np.linspace(0.0, duration, n_out)
    out = np.empty((n_out, q.shape[1]), dtype=np.float64)
    for c in range(q.shape[1]):
        out[:, c] = np.interp(dst_t, src_t, q[:, c])
    # Renormalise quaternion block (xyzw lives in the last 4 root columns).
    qs = root_coord_count - 4
    if qs >= 0 and root_coord_count <= q.shape[1]:
        quat = out[:, qs:root_coord_count]
        norm = np.linalg.norm(quat, axis=1, keepdims=True)
        norm[norm < 1e-8] = 1.0
        out[:, qs:root_coord_count] = quat / norm
    return out.astype(np.float32)
