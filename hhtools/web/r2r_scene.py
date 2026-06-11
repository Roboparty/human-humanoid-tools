# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Load terrain / interaction-object scenes from hhtools robot-export folders."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)


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


def _load_object_track_csv(path: Path) -> dict[str, Any] | None:
    """Parse ``object_<i>_<name>.csv`` (robot-frame pose; geometry from mesh).

    ``ext_*`` columns are optional — older exports recorded a cuboid extent, but
    current exports omit it and rely on the sidecar ``.obj`` mesh.  Columns are
    matched by header name so both layouts load.
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
                header = [c.strip() for c in raw]
                continue
            rows.append(list(raw))
    if header is None or not rows:
        return None
    arr = np.asarray(rows, dtype=np.float64)
    col = {name: i for i, name in enumerate(header)}

    def _cols(names: tuple[str, ...], fallback: tuple[int, ...]) -> list[int]:
        if all(n in col for n in names):
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
    mesh_name = meta.get("mesh_filename", "")
    mesh_path = path.parent / mesh_name if mesh_name else None
    if mesh_path is not None and not mesh_path.is_file():
        mesh_path = None
    if mesh_path is None:
        for cand in path.parent.glob("*_cleaned_simplified.obj"):
            mesh_path = cand
            break

    if all(n in col for n in ("ext_x", "ext_y", "ext_z")):
        extents = arr[0, [col["ext_x"], col["ext_y"], col["ext_z"]]].astype(np.float32)
    elif mesh_path is not None:
        extents = _mesh_bbox_extents(mesh_path)
    else:
        extents = np.zeros(3, dtype=np.float32)

    name = meta.get("object") or path.stem
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
    from hhtools.web.serialize import _downsample_indices

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
) -> dict[str, Any] | None:
    """Build a ``scaled_scene`` payload from an hhtools robot-export clip folder.

    Terrain / object meshes and tracks are already in the **source robot**
    retarget frame (as written by :mod:`hhtools.web.export_bundle`).
    """
    from hhtools.web.serialize import _serialize_terrain

    clip_dir = Path(clip_dir).resolve()
    prof = (profile or "mimic").strip().lower()
    idx = _downsample_scene_frames(int(num_frames))
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
            from hhtools.retarget.interaction_mesh.heightfield import obj_to_heightfield

            hf = obj_to_heightfield(terrain_obj)
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
            from hhtools.retarget.interaction_mesh.heightfield import obj_to_heightfield

            terrain = obj_to_heightfield(terrain_obj)
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
