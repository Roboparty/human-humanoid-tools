# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""R2R export bundles — mirror :mod:`hhtools.io.export_bundle` for robot-export clips.

Source clips already carry robot-frame terrain / object sidecars from the
human→robot export step.  R2R batch re-targets the robot trajectory onto a new
robot and re-scales scene assets with the same uniform ratio used for the yellow
overlay / IK scaler.
"""

from __future__ import annotations

import logging
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from hhtools.io.export_bundle import (
    _bake_export_joint_q,
    _robot_pkl_blob,
    _save_object_track_csv,
    ensure_export_path,
    resolve_clip_export_dir,
    sanitize_export_stem,
)

_log = logging.getLogger(__name__)


def _r2r_scaled_source_foot_z(source_motion, ratio: float) -> float | None:
    """Lowest source ankle/foot Z after the same uniform scale as the yellow overlay."""
    from hhtools.io.scene_serialize import _scaled_overlay_foot_z

    hierarchy = getattr(source_motion, "hierarchy", None)
    names = list(getattr(hierarchy, "bone_names", []) or [])
    positions = np.asarray(getattr(source_motion, "positions", None), dtype=np.float64)
    if not names or positions.ndim != 3 or positions.shape[0] == 0:
        return None
    from hhtools.core.grounding import retarget_source_floor_z_world

    z_min = float(retarget_source_floor_z_world(source_motion))
    scaled = np.asarray(positions[:1], dtype=np.float32).copy()
    scaled[:, :, 2] -= np.float32(z_min)
    scaled *= float(ratio)
    return _scaled_overlay_foot_z(
        {"bone_names": names, "positions": scaled.tolist()},
        0,
    )


def resolve_r2r_source_clip_dir(entry: dict[str, Any]) -> Path | None:
    """Return the folder that holds the source robot traj + scene sidecars."""
    clip_dir = str(entry.get("clip_dir") or "").strip()
    if clip_dir:
        path = Path(clip_dir).expanduser().resolve()
        if path.is_dir():
            return path

    source_path = str(entry.get("source_path") or "").strip()
    if source_path:
        path = Path(source_path).expanduser().resolve().parent
        if path.is_dir():
            return path

    upload_drop = str(entry.get("upload_drop") or "").strip()
    sequence_id = str(entry.get("sequence_id") or "").strip()
    if upload_drop and sequence_id:
        path = (Path(upload_drop).expanduser().resolve() / sequence_id).parent
        if path.is_dir():
            return path
    return None


def clip_has_export_scene(clip_dir: Path, *, stem: str, profile: str = "") -> bool:
    clip_dir = Path(clip_dir)
    prof = (profile or "").strip().lower()
    if (
        prof == "meshmimic" or any(clip_dir.glob("*_terrain.obj"))
    ) and (
        (clip_dir / f"{stem}_terrain.obj").is_file() or any(
            clip_dir.glob("*_terrain.obj")
        )
    ):
        return True
    # Web robot-export: ``object_*.csv`` + plain ``.obj`` (not only OMOMO
    # ``*_cleaned_simplified.obj``).
    if any(clip_dir.glob("object_*.csv")):
        return True
    if any(clip_dir.glob("*_cleaned_simplified.obj")):
        return True
    if prof == "intermimic":
        return any(
            p.is_file() and "_terrain" not in p.name.lower()
            for p in clip_dir.glob("*.obj")
        )
    return False


def _terrain_src_path(clip_dir: Path, stem: str) -> Path | None:
    clip_dir = Path(clip_dir)
    cand = clip_dir / f"{stem}_terrain.obj"
    if cand.is_file():
        return cand
    hits = sorted(clip_dir.glob("*_terrain.obj"))
    return hits[0] if hits else None


def _export_scaled_terrain_obj(src: Path, dst: Path, ratio: float) -> bool:
    try:
        import trimesh

        loaded = trimesh.load(str(src), force="mesh", process=False)
        verts = np.asarray(loaded.vertices, dtype=np.float64)
        if verts.size == 0:
            return False
        loaded.vertices = (verts * float(ratio)).astype(np.float32)
        dst.parent.mkdir(parents=True, exist_ok=True)
        loaded.export(str(dst))
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("r2r terrain rescale failed %s → %s: %s", src, dst, exc)
        return False


def _export_scaled_object_mesh(src: Path, dst: Path, ratio: float) -> bool:
    try:
        import trimesh

        loaded = trimesh.load(str(src), force="mesh", process=False)
        verts = np.asarray(loaded.vertices, dtype=np.float64)
        faces = np.asarray(getattr(loaded, "faces", np.zeros((0, 3))), dtype=np.int64)
        if verts.size == 0 or faces.size == 0:
            return False
        centroid = verts.mean(axis=0)
        if float(np.max(np.abs(centroid))) > 1e-2:
            verts = (verts - centroid) * float(ratio)
        else:
            verts = verts * float(ratio)
        mesh = trimesh.Trimesh(vertices=verts.astype(np.float32), faces=faces, process=False)
        dst.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(dst))
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("r2r object mesh rescale failed %s → %s: %s", src, dst, exc)
        return False


def _write_r2r_object_tracks(
    clip_dir: Path,
    source_clip_dir: Path,
    *,
    ratio: float,
    sample_rate: float,
    num_frames: int,
    fmt: str,
    csv_header: bool,
) -> list[str]:
    from hhtools.io.r2r_scene import _align_track_frames, _load_object_track_csv

    written: list[str] = []
    for i, src_csv in enumerate(sorted(source_clip_dir.glob("object_*.csv"))):
        ob = _load_object_track_csv(src_csv)
        if ob is None:
            continue
        positions = np.asarray(ob["positions"], dtype=np.float32) * np.float32(ratio)
        quat_xyzw = np.asarray(ob["quaternions"], dtype=np.float32)
        positions, quat_xyzw = _align_track_frames(positions, quat_xyzw, int(num_frames))
        extents = np.asarray(ob["extents"], dtype=np.float32) * np.float32(ratio)
        quat_wxyz = np.empty_like(quat_xyzw)
        quat_wxyz[..., 0] = quat_xyzw[..., 3]
        quat_wxyz[..., 1:] = quat_xyzw[..., :3]
        safe = "".join(
            (c if c.isalnum() or c in "._-" else "_" for c in str(ob["name"])),
        )
        ext = "csv" if (fmt or "csv").lower() == "csv" else "pkl"
        out_path = clip_dir / f"object_{i}_{safe}.{ext}"
        blob = {
            "name": str(ob["name"]),
            "positions": positions,
            "quaternions": quat_wxyz,
            "extents": extents,
            "mesh_filename": Path(str(ob.get("mesh_path") or "")).name,
            "sample_rate": float(sample_rate),
        }
        if ext == "csv":
            _save_object_track_csv(out_path, blob, include_header=csv_header)
        else:
            with out_path.open("wb") as fp:
                pickle.dump(blob, fp)
        written.append(out_path.name)
    return written


def _copy_r2r_scene_meshes(
    clip_dir: Path,
    source_clip_dir: Path,
    stem: str,
    *,
    ratio: float,
) -> list[str]:
    copied: list[str] = []
    terrain_src = _terrain_src_path(source_clip_dir, stem)
    if terrain_src is not None:
        dst = clip_dir / f"{stem}_terrain.obj"
        if _export_scaled_terrain_obj(terrain_src, dst, ratio):
            copied.append(dst.name)

    mesh_srcs: list[Path] = []
    seen_mesh: set[str] = set()
    for src in sorted(source_clip_dir.glob("*_cleaned_simplified.obj")):
        key = str(src.resolve())
        if key not in seen_mesh:
            seen_mesh.add(key)
            mesh_srcs.append(src)
    from hhtools.io.r2r_scene import resolve_object_mesh_path

    for csv_path in sorted(source_clip_dir.glob("object_*.csv")):
        mesh = resolve_object_mesh_path(csv_path)
        if mesh is None or not mesh.is_file():
            continue
        key = str(mesh.resolve())
        if key in seen_mesh:
            continue
        seen_mesh.add(key)
        mesh_srcs.append(mesh)

    for src in mesh_srcs:
        dst = clip_dir / src.name
        if _export_scaled_object_mesh(src, dst, ratio):
            copied.append(dst.name)
        elif src.is_file():
            try:
                shutil.copy2(src, dst)
                copied.append(dst.name)
            except OSError as exc:
                _log.warning("object mesh copy failed %s: %s", src, exc)
    return copied


def write_r2r_export_bundle(
    retargeted: Any,
    target_model,
    source_motion,
    *,
    scene_scale_ratio: float,
    entry: dict[str, Any],
    out_root: str | Path,
    stem: str,
    fps: float | None,
    fmt: str,
    resample_fn,
    csv_header: bool = True,
    yellow_foot_z: float | None = None,
    pack_scene: bool = True,
    t_start: float | None = None,
    t_end: float | None = None,
) -> Path:
    """Write robot + rescaled scene sidecars; zip when terrain/objects present.

    When ``pack_scene`` is False (offline batch), keep an uncompressed folder
    instead of a ``.zip``. File contents match the Web export either way.

    ``t_start`` / ``t_end`` (seconds on the retargeted timeline) keep a sub-clip.
    """
    import dataclasses

    from hhtools.io.export_bundle import apply_export_time_window

    retargeted, source_motion = apply_export_time_window(
        retargeted, source_motion, t_start=t_start, t_end=t_end,
    )

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stem = sanitize_export_stem(stem)
    fmt = (fmt or "csv").lower()

    source_path = entry.get("source_path")
    source_clip_dir = resolve_r2r_source_clip_dir(entry) or out_root
    profile = str(entry.get("upload_profile") or "")
    has_scene = bool(entry.get("has_scene")) or clip_has_export_scene(
        source_clip_dir, stem=stem, profile=profile,
    )

    joint_q, sample_rate = resample_fn(retargeted, fps)
    ratio = float(scene_scale_ratio)

    clip_dir = resolve_clip_export_dir(
        out_root, stem, source_path, has_scene=has_scene,
    )
    if clip_dir.exists() and clip_dir.is_dir() and has_scene:
        shutil.rmtree(clip_dir, ignore_errors=True)
    clip_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(getattr(retargeted, "meta", {}) or {})
    meta["retarget_backend"] = "r2r"
    meta["r2r_scale_ratio"] = f"{ratio:.6f}"
    has_terrain = (
        getattr(source_motion, "terrain", None) is not None
        or _terrain_src_path(Path(source_clip_dir), stem) is not None
    )
    if yellow_foot_z is None and not has_terrain:
        yellow_foot_z = _r2r_scaled_source_foot_z(source_motion, ratio)

    joint_q, playback_lift = _bake_export_joint_q(
        target_model,
        retargeted,
        joint_q,
        source_motion,
        meta,
        yellow_foot_z=yellow_foot_z,
        preserve_absolute_z=has_terrain,
        yellow_align="ankle",
    )
    if abs(playback_lift) > 1e-12:
        meta["playback_mesh_z_lift"] = f"{playback_lift:.6f}"
    ret2 = dataclasses.replace(
        retargeted, joint_q=joint_q, sample_rate=sample_rate, meta=meta,
    )

    if fmt == "pkl":
        pkl_path = ensure_export_path(out_root, clip_dir / f"{stem}.pkl")
        with pkl_path.open("wb") as fp:
            pickle.dump(
                {
                    "hhtools_export": "r2r_v1",
                    "format": "pkl",
                    "retarget_backend": "r2r",
                    "robot": _robot_pkl_blob(ret2, joint_q, sample_rate, meta),
                    "r2r_scale_ratio": ratio,
                },
                fp,
            )
    else:
        from hhtools.io.robot_csv import save_robot_csv

        trajectory_path = ensure_export_path(out_root, clip_dir / f"{stem}.csv")
        save_robot_csv(
            trajectory_path,
            robot=target_model,
            joint_q=joint_q,
            sample_rate=sample_rate,
            meta=meta,
            include_header=csv_header,
        )

    object_tracks: list[str] = []
    mesh_names: list[str] = []
    if has_scene:
        object_tracks = _write_r2r_object_tracks(
            clip_dir,
            source_clip_dir,
            ratio=ratio,
            sample_rate=sample_rate,
            num_frames=int(joint_q.shape[0]),
            fmt=fmt,
            csv_header=csv_header,
        )
        mesh_names = _copy_r2r_scene_meshes(
            clip_dir, source_clip_dir, stem, ratio=ratio,
        )

    if not has_scene:
        return ensure_export_path(
            out_root,
            clip_dir / (f"{stem}.pkl" if fmt == "pkl" else f"{stem}.csv"),
        )

    if not pack_scene:
        _log.info(
            "r2r export folder %s (ratio=%.4f, meshes=%s, object_tracks=%s)",
            clip_dir,
            ratio,
            mesh_names,
            object_tracks,
        )
        return clip_dir

    # See ``write_retarget_export_bundle`` — avoid zipping a directory into
    # a ``.zip`` path that lives inside that same directory (R2R batch uploads
    # where ``out_root.name == stem``).
    from hhtools.io.export_bundle import zip_directory

    zip_path = ensure_export_path(out_root, zip_directory(clip_dir, stem))
    shutil.rmtree(clip_dir, ignore_errors=True)
    _log.info(
        "r2r export bundle %s (ratio=%.4f, meshes=%s, object_tracks=%s)",
        zip_path.name,
        ratio,
        mesh_names,
        object_tracks,
    )
    return zip_path
