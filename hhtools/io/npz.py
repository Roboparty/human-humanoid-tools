"""Unified internal NPZ schema.

The on-disk layout is a single ``.npz`` archive with the following keys:

- ``schema_version``: string ``"1"`` — bumped on breaking changes.
- ``name``: scalar string. Human-readable motion name (used as the default export stem).
- ``framerate``: scalar float64. Samples per second.
- ``up_axis``: scalar string (``"X" | "Y" | "Z"``). Up axis of the stored ``positions``.
- ``source_format``: scalar string. Original format (``"bvh" | "glb" | "smpl" | ...``).
- ``bone_names``: array of length ``num_bones`` (``U128``). Bone names.
- ``parent_indices``: int32 array of length ``num_bones``. ``-1`` marks the root.
- ``positions``: float32 array of shape ``(num_frames, num_bones, 3)``.
- ``quaternions``: float32 array of shape ``(num_frames, num_bones, 4)`` stored as ``[x, y, z, w]``.
- ``meta_json``: JSON-encoded scalar string with extra free-form metadata.

Optional scene-object arrays (present only when ``motion.objects`` is non-empty):

- ``objects_names``: array of length ``num_objects`` (``U128``). Object identifiers.
- ``objects_positions``: float32 array of shape ``(num_frames, num_objects, 3)``.
- ``objects_quaternions``: float32 array of shape ``(num_frames, num_objects, 4)`` (xyzw).
- ``objects_extents``: float32 array of shape ``(num_objects, 3)``. Placeholder cuboid W/D/H.
- ``objects_mesh_paths``: array of length ``num_objects`` (``U512``). Empty string = no mesh.
  Relative paths resolve against the NPZ's own directory so a per-clip bundle
  (``<clip>/<clip>.npz`` + ``<clip>/<mesh>.obj``) stays self-contained after relocation.
- ``objects_scales``: float32 array of shape ``(num_objects,)``. Uniform mesh scale factor.

Optional ``meta_json`` keys consumed by terrain loaders:

- ``terrain_heightfield_frame`` (int): 0-based frame index used when rasterising the
  terrain mesh into a static :class:`~hhtools.core.scene.TerrainHeightfield` (default 0).
  MoCap exports often leave frame 0 at the origin while props are placed from frame 1.

Rendering hints (``SceneObject.opacity`` / ``color``) are **not** persisted: the NPZ
is purely a trajectory + topology artifact. Visual policy (terrain slate-gray, prop
orange, …) is applied at render time in :class:`hhtools.viewer.renderers.objects.ObjectsRenderer`
based on the object's ``name`` so saved bundles still render consistently without
having to carry viz metadata in the data file.

A few optional side arrays (``root_translation``, ``skinned_mesh_vertices`` ...) may be stored by
higher-level exporters; they are ignored by the loader unless explicitly requested.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hhtools.core.hierarchy import Hierarchy
from hhtools.core.motion import Motion
from hhtools.core.scene import SceneObject, TerrainHeightfield

SCHEMA_VERSION = "1"


def _is_terrain_mesh_name(name: str, mesh_path: str) -> bool:
    """Heuristic: does this object_* entry describe static terrain?

    True iff either the object name is ``"terrain"`` (matches the
    holosoma adapter convention) or the referenced mesh basename
    matches ``*_terrain.obj`` / ``terrain.obj`` (parc_ms convention).
    Everything else (boxes, balls, mops, …) stays a regular
    :class:`SceneObject` for the existing prop-tracking pipeline.
    """
    if name and name.strip().lower() == "terrain":
        return True
    if mesh_path:
        stem = Path(mesh_path).name.lower()
        return stem == "terrain.obj" or stem.endswith("_terrain.obj")
    return False


def save_npz(motion: Motion, path: str | Path, *, compressed: bool = True) -> None:
    """Serialise a :class:`Motion` to the unified NPZ schema.

    Heightfield terrain (``motion.terrain``) is **not** packed into the NPZ
    itself — it would round-trip awkwardly through ``np.savez``'s flat
    key/value layout, and it shares its on-disk format with the PARC
    training rig.  Instead, when present, we write a sidecar
    ``<stem>.pkl`` next to the NPZ via :func:`save_parc_pkl`; the matching
    :func:`_resolve_terrain_for_npz` discovers it on load.

    This makes save/load a closed round-trip even for caches that live
    far from the original clip directory (e.g. ``EphemeralCache``'s
    ``/tmp/hhtools_cache_*`` directory), which previously dropped
    terrain because the source-tree sidecar wasn't reachable from the
    cache path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bone_names = np.array(motion.hierarchy.bone_names, dtype="U128")
    # Strip non-serialisable meta entries before JSON-dumping.  Today that covers the
    # ``skinned_mesh`` attachment produced by ``load_glb(with_mesh=True)`` — a
    # :class:`~hhtools.core.skinning.SkinnedMesh` dataclass of numpy arrays that would
    # explode the default ``_json_default`` handler.  We replace it with a marker dict
    # holding only lightweight summary counts so a round-tripped NPZ can at least tell
    # the user "this clip used to have a mesh with N vertices; re-open the source file
    # to get it back".
    clean_meta = _strip_unserialisable_meta(dict(motion.meta or {}))
    payload: dict[str, np.ndarray] = {
        "schema_version": np.array(SCHEMA_VERSION),
        "name": np.array(str(motion.name)),
        "framerate": np.array(float(motion.framerate), dtype=np.float64),
        "up_axis": np.array(motion.up_axis),
        "source_format": np.array(motion.source_format),
        "bone_names": bone_names,
        "parent_indices": np.asarray(motion.hierarchy.parent_indices, dtype=np.int32),
        "positions": np.asarray(motion.positions, dtype=np.float32),
        "quaternions": np.asarray(motion.quaternions, dtype=np.float32),
        "meta_json": np.array(
            json.dumps(clean_meta, ensure_ascii=False, default=_json_default)
        ),
    }

    # Only write object arrays when at least one is present — keeps the schema forward
    # compatible with pre-existing NPZs that predate the scene-objects extension.
    if motion.objects:
        num_frames = motion.num_frames
        for obj in motion.objects:
            if obj.num_frames != num_frames:
                raise ValueError(
                    f"SceneObject {obj.name!r} has {obj.num_frames} frames but motion has "
                    f"{num_frames}; mismatched scene-object trajectories cannot be serialised."
                )
        payload["objects_names"] = np.array(
            [obj.name for obj in motion.objects], dtype="U128"
        )
        payload["objects_positions"] = np.stack(
            [obj.positions for obj in motion.objects], axis=1
        ).astype(np.float32)  # (T, N_obj, 3)
        payload["objects_quaternions"] = np.stack(
            [obj.quaternions for obj in motion.objects], axis=1
        ).astype(np.float32)  # (T, N_obj, 4)
        payload["objects_extents"] = np.stack(
            [obj.extents for obj in motion.objects], axis=0
        ).astype(np.float32)  # (N_obj, 3)
        payload["objects_mesh_paths"] = np.array(
            [obj.mesh_path for obj in motion.objects], dtype="U512"
        )
        payload["objects_scales"] = np.asarray(
            [obj.scale for obj in motion.objects], dtype=np.float32
        )

    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)

    # Persist terrain heightfield as a sidecar PARC pkl so cache-based
    # round-trips (and viewer→retarget hand-off) don't lose it.  We
    # consciously overwrite an existing sidecar — the in-memory motion is
    # the new source of truth at this point.  Failures are non-fatal: a
    # read-only filesystem would otherwise crash an otherwise-successful
    # NPZ save.
    if motion.terrain is not None:
        from hhtools.io.parc_export import save_parc_pkl

        sidecar = path.with_suffix(".pkl")
        try:
            save_parc_pkl(
                sidecar,
                motion_data=None,
                terrain_data=motion.terrain,
                misc_data=None,
            )
        except OSError:
            pass


def load_npz(
    path: str | Path,
    *,
    progress_callback: Any = None,  # noqa: ANN401 — ProgressCallback, lazy import
) -> Motion:
    """Deserialise a unified NPZ motion file."""
    from hhtools.io.loader_progress import report_progress

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    report_progress(progress_callback, 0.0, f"读取 NPZ {path.name}…")
    with np.load(path, allow_pickle=False) as data:
        schema = str(data["schema_version"].item()) if "schema_version" in data.files else "0"
        if schema not in {"1"}:
            raise ValueError(
                f"Unsupported NPZ schema version {schema!r} in {path}; expected {SCHEMA_VERSION}"
            )
        name = str(data["name"].item())
        framerate = float(data["framerate"].item())
        up_axis = str(data["up_axis"].item())
        source_format = (
            str(data["source_format"].item()) if "source_format" in data.files else "npz"
        )
        bone_names = [str(x) for x in data["bone_names"].tolist()]
        parent_indices = np.asarray(data["parent_indices"], dtype=np.int32)
        positions = np.asarray(data["positions"], dtype=np.float32)
        quaternions = np.asarray(data["quaternions"], dtype=np.float32)
        meta = {}
        if "meta_json" in data.files:
            try:
                meta = json.loads(str(data["meta_json"].item()))
            except json.JSONDecodeError:
                meta = {}

        # Optional scene objects — all arrays must be present together. If any are missing
        # we treat the file as a pure skeletal motion (backwards compatibility).
        objects: list[SceneObject] = []
        # Terrain objects are split off from objects_* and routed to a
        # heightfield (Motion.terrain) instead of a SceneObject; we
        # collect the rasterisation params here, then build the
        # heightfield after the np.load context closes.
        terrain_obj_pose: dict | None = None
        if {
            "objects_names",
            "objects_positions",
            "objects_quaternions",
            "objects_extents",
        }.issubset(set(data.files)):
            names = [str(x) for x in data["objects_names"].tolist()]
            positions_arr = np.asarray(data["objects_positions"], dtype=np.float32)
            quaternions_arr = np.asarray(data["objects_quaternions"], dtype=np.float32)
            extents_arr = np.asarray(data["objects_extents"], dtype=np.float32)
            raw_mesh_paths = (
                [str(x) for x in data["objects_mesh_paths"].tolist()]
                if "objects_mesh_paths" in data.files
                else ["" for _ in names]
            )
            # Relative mesh paths resolve **against the NPZ's own directory** so a
            # per-clip bundle (``.../parkour_1/parkour_1.npz`` + ``.../parkour_1/
            # terrain.obj``) is self-contained and survives relocation. Absolute
            # paths are passed through unchanged for backwards compatibility with
            # NPZs minted from raw source trees (the ephemeral cache still writes
            # absolute paths there).
            npz_dir = path.parent
            mesh_paths: list[str] = []
            for raw in raw_mesh_paths:
                if not raw:
                    mesh_paths.append("")
                    continue
                p = Path(raw)
                if p.is_absolute():
                    mesh_paths.append(raw)
                else:
                    mesh_paths.append(str((npz_dir / raw).resolve()))
            scales_arr = (
                np.asarray(data["objects_scales"], dtype=np.float32)
                if "objects_scales" in data.files
                else np.ones(len(names), dtype=np.float32)
            )
            for i, obj_name in enumerate(names):
                obj_mesh = mesh_paths[i] if i < len(mesh_paths) else ""
                if _is_terrain_mesh_name(obj_name, obj_mesh) and obj_mesh:
                    # Capture pose / scale for one-shot rasterisation
                    # below; we only ever materialise the FIRST terrain
                    # entry — meshmimic clips have a single static
                    # terrain by convention.
                    if terrain_obj_pose is None:
                        # Static heightfield rasterisation uses a single pose.  When Maya
                        # leaves frame 0 at the origin and the real placement starts on a
                        # later frame (common for ``*_rig.bvh`` exports), ``meta_json`` may
                        # carry ``terrain_heightfield_frame`` (0-based) to pick that frame.
                        hf_t = 0
                        try:
                            raw = int(meta.get("terrain_heightfield_frame", 0))
                        except (TypeError, ValueError):
                            raw = 0
                        n_frames = int(positions_arr.shape[0])
                        hf_t = max(0, min(raw, n_frames - 1))
                        terrain_obj_pose = {
                            "mesh_path": obj_mesh,
                            "position": positions_arr[hf_t, i, :].astype(np.float32),
                            "quaternion": quaternions_arr[hf_t, i, :].astype(np.float32),
                            "scale": float(scales_arr[i]) if i < len(scales_arr) else 1.0,
                        }
                    continue
                # ``opacity`` / ``color`` are rendering hints resolved at render time
                # (see ObjectsRenderer's name → style fallback) — the NPZ deliberately
                # does not store them, so loads always produce SceneObjects with None
                # overrides and the viewer decides the look.
                objects.append(
                    SceneObject(
                        name=obj_name,
                        positions=positions_arr[:, i, :],
                        quaternions=quaternions_arr[:, i, :],
                        extents=extents_arr[i],
                        mesh_path=obj_mesh,
                        scale=float(scales_arr[i]) if i < len(scales_arr) else 1.0,
                    )
                )

    terrain = _resolve_terrain_for_npz(path, name, terrain_obj_pose)

    hierarchy = Hierarchy.from_parent_indices(bone_names, parent_indices)

    motion_out = Motion(
        name=name,
        hierarchy=hierarchy,
        positions=positions,
        quaternions=quaternions,
        framerate=framerate,
        up_axis=up_axis,  # type: ignore[arg-type]
        source_format=source_format,  # type: ignore[arg-type]
        meta=meta,
        objects=objects,
        terrain=terrain,
    )
    report_progress(progress_callback, 1.0, f"NPZ 加载完成 ({path.name})")
    return motion_out


def _resolve_terrain_for_npz(
    path: Path, npz_name: str, terrain_obj_pose: dict | None
) -> TerrainHeightfield | None:
    """Resolve a clip's terrain heightfield once.

    1. Look for a sidecar PARC ``.pkl`` next to the NPZ — either
       ``<stem>.pkl`` (clip-named, matches the build_terrain script
       output) or ``<motion-name>.pkl`` for compatibility with PARC's
       direct-source convention.  If present, decode it.
    2. If absent and the NPZ carried a terrain object entry, rasterise
       the OBJ once via :func:`obj_to_heightfield` and persist the
       result as a sidecar so future loads hit case 1.
    3. Otherwise return ``None``.
    """
    from hhtools.io.parc_export import load_parc_pkl_terrain, save_parc_pkl
    from hhtools.retarget.interaction_mesh.heightfield import obj_to_heightfield

    candidates = [
        path.with_suffix(".pkl"),
        path.parent / f"{npz_name}.pkl",
    ]
    for cand in candidates:
        if cand.is_file():
            terrain = load_parc_pkl_terrain(cand)
            if terrain is not None:
                return terrain

    if terrain_obj_pose is None:
        return None

    obj_path = Path(terrain_obj_pose["mesh_path"])
    if not obj_path.is_file():
        return None

    terrain = obj_to_heightfield(
        obj_path,
        dx=0.05,
        padding=0.5,
        object_position=terrain_obj_pose["position"],
        object_quat_xyzw=terrain_obj_pose["quaternion"],
        mesh_scale=terrain_obj_pose["scale"],
    )

    sidecar = candidates[0]
    try:
        save_parc_pkl(sidecar, motion_data=None, terrain_data=terrain, misc_data=None)
    except OSError:
        pass
    return terrain


def _json_default(obj):  # type: ignore[no-untyped-def]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serialisable")


def _strip_unserialisable_meta(meta: dict) -> dict:
    """Remove meta entries that JSON can't encode, replacing them with summaries.

    Currently handles :class:`hhtools.core.skinning.SkinnedMesh` (attached by
    ``load_glb(with_mesh=True)``).  We replace the mesh with a ``{"_kind": ..., ...}``
    marker so analytics / reloading code can still tell the NPZ came from a file that
    *had* a mesh, without forcing the NPZ schema to carry the vertex buffer itself
    (which would bloat the cached files for every GLB clip by ~0.5–5 MB).

    Extend this function when new non-JSON-friendly attachments land on Motion.meta.
    """
    from hhtools.core.skinning import BakedMesh, SkinnedMesh  # lazy; avoid circulars

    out = {}
    for key, value in meta.items():
        if isinstance(value, SkinnedMesh):
            out[key] = {
                "_kind": "SkinnedMesh.stripped",
                "num_vertices": int(value.num_vertices),
                "num_triangles": int(value.num_triangles),
                "num_joints": int(value.num_joints),
                "influences_per_vertex": int(value.influences_per_vertex),
                "note": (
                    "SkinnedMesh dropped at NPZ save time; re-open the authored GLB "
                    "with ``load_glb(with_mesh=True)`` to get the mesh back."
                ),
            }
        elif isinstance(value, BakedMesh):
            out[key] = {
                "_kind": "BakedMesh.stripped",
                "num_frames": int(value.num_frames),
                "num_vertices": int(value.num_vertices),
                "num_triangles": int(value.num_triangles),
                "note": (
                    "BakedMesh dropped at NPZ save time; re-run the SMPL forward pass "
                    "via the dataset adapter with ``with_mesh=True`` to get vertices back."
                ),
            }
        else:
            out[key] = value
    return out


__all__ = ["SCHEMA_VERSION", "load_npz", "save_npz"]
