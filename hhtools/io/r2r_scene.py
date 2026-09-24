# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Load terrain / interaction-object scenes from hhtools robot-export folders."""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

TerrainLoader = Callable[[Path], Any]


def _parse_comment_meta(path: Path) -> dict[str, str]:
    """Parse ``# key: value`` comment headers from text robot trajectory files.

    Binary exports (``.pkl`` / ``.npz``) carry metadata inside the archive, not
    as UTF-8 comment lines — skip them instead of failing decode.
    """
    meta: dict[str, str] = {}
    if path.suffix.lower() not in (".csv", ".txt"):
        return meta
    try:
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                s = line.strip()
                if not s.startswith("#"):
                    break
                body = s.lstrip("#").strip()
                if ":" in body:
                    k, _, v = body.partition(":")
                    meta[k.strip()] = v.strip()
    except (OSError, UnicodeDecodeError):
        pass
    return meta


def _mesh_bbox_extents(mesh_path: Path) -> np.ndarray:
    """Full-side bounding-box dimensions of a mesh, or zeros if unreadable."""
    try:
        import trimesh

        loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
        verts = np.asarray(getattr(loaded, "vertices", np.zeros((0, 3))), dtype=np.float64)
        if verts.size == 0:
            return np.zeros(3, dtype=np.float32)
        return (verts.max(axis=0) - verts.min(axis=0)).astype(np.float32)
    except Exception:  # noqa: BLE001
        return np.zeros(3, dtype=np.float32)


def _row_looks_numeric(cells: list[str]) -> bool:
    if len(cells) < 8:
        return False
    try:
        float(cells[0])
        float(cells[7])
        return True
    except ValueError:
        return False


def _looks_like_object_header(cells: list[str]) -> bool:
    if not cells:
        return False
    first = cells[0].strip().lower()
    if first == "time":
        return True
    return any(c.strip().lower() in ("pos_x", "quat_x", "quat_w") for c in cells)


def _object_display_name_from_stem(stem: str) -> str:
    """``object_0_Box_H_1`` → ``Box_H_1``; otherwise the stem itself."""
    name = str(stem)
    if name.startswith("object_") and name.count("_") >= 2:
        return name.split("_", 2)[2]
    return name


def resolve_object_mesh_path(csv_path: Path, meta: dict[str, str] | None = None) -> Path | None:
    """Locate the object ``.obj`` next to an ``object_*.csv`` track.

    Mesh names are **not** restricted to ``*_cleaned_simplified.obj``.  Web
    exports commonly ship ``Box_H_1.obj`` beside ``object_0_Box_H_1.csv``.
    """
    folder = Path(csv_path).parent
    meta = meta or {}
    mesh_name = str(meta.get("mesh_filename") or "").strip()
    if mesh_name:
        cand = folder / mesh_name
        if cand.is_file():
            return cand

    display = _object_display_name_from_stem(Path(csv_path).stem)
    preferred = [
        folder / f"{display}.obj",
        folder / f"{display}_cleaned_simplified.obj",
        folder / f"{Path(csv_path).stem}.obj",
    ]
    for cand in preferred:
        if cand.is_file():
            return cand

    for cand in sorted(folder.glob("*_cleaned_simplified.obj")):
        return cand

    objs = [
        p for p in sorted(folder.glob("*.obj"))
        if "_terrain" not in p.name.lower()
    ]
    if len(objs) == 1:
        return objs[0]
    display_l = display.lower()
    for p in objs:
        stem_l = p.stem.lower()
        if display_l and (display_l in stem_l or stem_l in display_l):
            return p
    return None


def _load_object_track_csv(path: Path) -> dict[str, Any] | None:
    """Parse ``object_<i>_<name>.csv`` (robot-frame pose; geometry from mesh).

    Supports both headered exports and headerless ``time,pos,quat`` rows (the
    default when ``csv_header=0``).  ``ext_*`` columns are optional.
    """
    meta = _parse_comment_meta(path)
    header: list[str] | None = None
    rows: list[list[str]] = []
    with path.open(encoding="utf-8") as fp:
        reader = csv.reader(fp)
        for raw in reader:
            if not raw:
                continue
            if raw[0].startswith("#"):
                continue
            if header is None:
                if _looks_like_object_header(raw):
                    header = [c.strip() for c in raw]
                    continue
                if _row_looks_numeric(raw):
                    header = []
                    rows.append(list(raw))
                    continue
                header = [c.strip() for c in raw]
                continue
            rows.append(list(raw))
    if header is None or not rows:
        return None
    arr = np.asarray(rows, dtype=np.float64)
    col = {name: i for i, name in enumerate(header)} if header else {}

    def _cols(names: tuple[str, ...], fallback: tuple[int, ...]) -> list[int]:
        if header and all(n in col for n in names):
            return [col[n] for n in names]
        return list(fallback)

    t_i = col.get("time", 0)
    pos_i = _cols(("pos_x", "pos_y", "pos_z"), (1, 2, 3))
    quat_i = _cols(("quat_x", "quat_y", "quat_z", "quat_w"), (4, 5, 6, 7))
    times = arr[:, t_i]
    positions = arr[:, pos_i].astype(np.float32)
    quats = arr[:, quat_i].astype(np.float32)  # xyzw

    if times.shape[0] > 1:
        fps = float(1.0 / max(times[1] - times[0], 1e-6))
    else:
        fps = float(meta.get("sample_rate", 30.0) or 30.0)
    mesh_path = resolve_object_mesh_path(path, meta)

    if all(n in col for n in ("ext_x", "ext_y", "ext_z")):
        extents = arr[0, [col["ext_x"], col["ext_y"], col["ext_z"]]].astype(np.float32)
    elif mesh_path is not None:
        extents = _mesh_bbox_extents(mesh_path)
    else:
        extents = np.zeros(3, dtype=np.float32)

    name = meta.get("object") or path.stem
    if name.startswith("object_") and name.count("_") >= 2:
        name = name.split("_", 2)[2]
    return {
        "name": str(name),
        "positions": positions,
        "quaternions": quats,
        "extents": extents,
        "mesh_path": str(mesh_path) if mesh_path and mesh_path.is_file() else "",
        "framerate": fps,
        "scale": 1.0,
    }


def _downsample_scene_frames(num_frames: int, max_frames: int = 600) -> np.ndarray:
    from hhtools.io.scene_serialize import _downsample_indices

    return _downsample_indices(num_frames, max_frames)


def _serialize_object_for_web(
    ob: dict[str, Any],
    idx: np.ndarray,
    *,
    source_index: int,
) -> dict[str, Any]:
    pos = np.asarray(ob["positions"], dtype=np.float32)[idx]
    quat = np.asarray(ob["quaternions"], dtype=np.float32)[idx]
    return {
        "name": ob["name"],
        "extents": np.asarray(ob["extents"], dtype=np.float32).tolist(),
        "has_mesh": bool(ob.get("mesh_path")),
        "mesh_file": Path(str(ob.get("mesh_path") or "")).name,
        "scale": float(ob.get("scale", 1.0)),
        "color": [106, 159, 212],
        "opacity": 0.82,
        "positions": np.round(pos, 4).tolist(),
        "quaternions": np.round(quat, 5).tolist(),
        "source_index": source_index,
    }


def load_r2r_clip_scene(
    clip_dir: Path,
    *,
    profile: str,
    robot_path: Path,
    num_frames: int,
    framerate: float,
    terrain_loader: TerrainLoader,
) -> dict[str, Any] | None:
    """Build a ``scaled_scene`` payload from an hhtools robot-export clip folder.

    Terrain / object meshes and tracks are already in the **source robot**
    retarget frame (as written by :mod:`hhtools.io.export_bundle`).
    """
    from hhtools.io.scene_serialize import _serialize_terrain

    clip_dir = Path(clip_dir).resolve()
    prof = (profile or "mimic").strip().lower()
    payload: dict[str, Any] = {
        "scale_ratio": 1.0,
        "objects": [],
        "terrain": None,
        "framerate": float(framerate),
        "clip_dir": str(clip_dir),
    }

    stem = robot_path.stem
    terrain_obj = clip_dir / f"{stem}_terrain.obj"
    if not terrain_obj.is_file():
        hits = sorted(clip_dir.glob("*_terrain.obj"))
        terrain_obj = hits[0] if hits else None

    if prof in ("meshmimic", "auto") and terrain_obj is not None:
        try:
            hf = terrain_loader(terrain_obj)
            payload["terrain"] = _serialize_terrain(hf)
        except Exception as err:  # noqa: BLE001
            _log.warning("r2r terrain load failed for %s: %s", terrain_obj, err)

    if prof in ("intermimic", "auto"):
        obj_paths = sorted(clip_dir.glob("object_*.csv"))
        for i, op in enumerate(obj_paths):
            ob = _load_object_track_csv(op)
            if ob is None:
                continue
            n = int(ob["positions"].shape[0])
            ob_idx = _downsample_scene_frames(n)
            payload["objects"].append(
                _serialize_object_for_web(ob, ob_idx, source_index=i),
            )

    if payload["terrain"] is None and not payload["objects"]:
        return None
    return payload


def compute_r2r_target_scaled_scene(
    motion,
    *,
    scale_ratio: float,
    terrain_loader: TerrainLoader,
    clip_dir: Path,
    profile: str,
    robot_path: Path,
    num_frames: int,
    framerate: float,
) -> dict[str, Any] | None:
    """Scale source-robot terrain/objects for target-robot web preview.

    Uses the same uniform ``ratio`` as the yellow overlay and
    :func:`~hhtools.io.r2r_export_bundle.write_r2r_export_bundle`.
    """

    from hhtools.io.scene_serialize import _serialize_terrain

    scene_motion = attach_r2r_clip_scene_to_motion(
        motion,
        clip_dir,
        profile=profile,
        robot_path=robot_path,
        terrain_loader=terrain_loader,
    )
    if scene_motion.terrain is None and not scene_motion.objects:
        return None

    ratio = float(scale_ratio)
    idx = _downsample_scene_frames(int(num_frames))
    payload: dict[str, Any] = {
        "scale_ratio": round(ratio, 5),
        "objects": [],
        "terrain": None,
        "framerate": float(framerate),
        "clip_dir": str(Path(clip_dir).resolve()),
    }

    for i, ob in enumerate(scene_motion.objects):
        pos, quat = _align_track_frames(
            np.asarray(ob.positions, dtype=np.float32) * np.float32(ratio),
            np.asarray(ob.quaternions, dtype=np.float32),
            int(num_frames),
        )
        scaled = {
            "name": ob.name,
            "positions": pos,
            "quaternions": quat,
            "extents": np.asarray(ob.extents, dtype=np.float32) * np.float32(ratio),
            "mesh_path": ob.mesh_path,
            "scale": float(ob.scale) * ratio,
        }
        payload["objects"].append(
            _serialize_object_for_web(scaled, idx, source_index=i),
        )

    if scene_motion.terrain is not None:
        payload["terrain"] = _serialize_terrain(scene_motion.terrain.scaled(ratio))

    return payload


def _align_track_frames(
    positions: np.ndarray,
    quaternions: np.ndarray,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim or hold-last-frame so object tracks match ``motion.num_frames``."""
    n = int(positions.shape[0])
    if n == num_frames:
        return positions, quaternions
    if n > num_frames:
        return positions[:num_frames], quaternions[:num_frames]
    if n == 0:
        return (
            np.zeros((num_frames, 3), dtype=np.float32),
            np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (num_frames, 1)),
        )
    last_p = positions[-1:]
    last_q = quaternions[-1:]
    pad = num_frames - n
    return (
        np.concatenate([positions, np.repeat(last_p, pad, axis=0)], axis=0),
        np.concatenate([quaternions, np.repeat(last_q, pad, axis=0)], axis=0),
    )


def attach_r2r_clip_scene_to_motion(
    motion,
    clip_dir: Path,
    *,
    profile: str,
    robot_path: Path,
    terrain_loader: TerrainLoader,
):
    """Attach full-resolution terrain / objects for Interaction-Mesh retarget.

    Web preview uses downsampled ``load_r2r_clip_scene``; the MPC backend needs
    per-frame tracks aligned to ``motion.num_frames``.
    """
    from dataclasses import replace

    from hhtools.core.scene import SceneObject

    clip_dir = Path(clip_dir).resolve()
    robot_path = Path(robot_path).resolve()
    prof = (profile or "mimic").strip().lower()
    num_frames = int(motion.num_frames)
    terrain = None
    objects: list[SceneObject] = []

    stem = robot_path.stem
    terrain_obj = clip_dir / f"{stem}_terrain.obj"
    if not terrain_obj.is_file():
        hits = sorted(clip_dir.glob("*_terrain.obj"))
        terrain_obj = hits[0] if hits else None

    if prof in ("meshmimic", "auto") and terrain_obj is not None:
        try:
            terrain = terrain_loader(terrain_obj)
        except Exception as err:  # noqa: BLE001
            _log.warning("r2r terrain attach failed for %s: %s", terrain_obj, err)

    if prof in ("intermimic", "auto"):
        for op in sorted(clip_dir.glob("object_*.csv")):
            ob = _load_object_track_csv(op)
            if ob is None:
                continue
            pos, quat = _align_track_frames(
                np.asarray(ob["positions"], dtype=np.float32),
                np.asarray(ob["quaternions"], dtype=np.float32),
                num_frames,
            )
            mesh_path = str(ob.get("mesh_path") or "")
            objects.append(
                SceneObject(
                    name=str(ob["name"]),
                    positions=pos,
                    quaternions=quat,
                    extents=np.asarray(ob["extents"], dtype=np.float32),
                    mesh_path=mesh_path,
                    scale=float(ob.get("scale", 1.0)),
                ),
            )

    if terrain is None and not objects:
        return motion
    return replace(motion, terrain=terrain, objects=objects)
