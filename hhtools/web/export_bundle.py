# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Browser-facing retarget export bundles (OMOMO / parc_ms style folders).

When a clip has terrain or interaction objects, exports are packaged as::

    <stem>/
        <stem>.csv   OR   <stem>.pkl   # robot trajectory
        object_<i>_<name>.csv|.pkl     # per interaction object (robot-scaled frame)
        <stem>_terrain.obj             # terrain mesh in robot scale
        <object_mesh>.obj              # centred mesh vertices scaled to robot frame

Interaction-object tracks are the same scaled robot-frame 6-DoF trajectories the
interaction-mesh SQP used as anchors (``smpl_scale`` + foot-floor ``z_offset``).
Terrain OBJ uses ``source_terrain_z_offset`` (split grounding) when stamped on
``retargeted.meta``; object meshes are centred + scaled by ``obj.scale * smpl_scale``.

The caller receives a ``.zip`` path suitable for ``FileResponse`` download
into the user's default browser save folder (never written under ``assets/``).
"""

from __future__ import annotations

import csv
import logging
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)


def motion_has_scene(motion) -> bool:
    return bool(getattr(motion, "terrain", None) is not None or getattr(motion, "objects", None))


def resolve_clip_export_dir(
    out_root: str | Path,
    stem: str,
    source_path: str | Path | None = None,
    *,
    has_scene: bool = False,
) -> Path:
    """Directory for one clip's export files, mirroring the source tree.

    Flat sources (``AMASS/clip.npz``) write into ``out_root/clip.csv``.
    Folder clips (``OMOMO/clip/clip.pkl``) write into ``out_root/clip/``.
    Upload drops whose ``export_subdir`` already ends at the clip folder do not
    gain an extra ``clip/clip/`` nesting level.
    """
    out_root = Path(out_root)
    if source_path is not None:
        parent = Path(source_path).resolve().parent
        if parent.name == stem:
            if out_root.name == stem:
                return out_root
            return out_root / stem
        if has_scene:
            return out_root / stem
        return out_root
    return out_root / stem if has_scene else out_root


OBJECT_CSV_HEADER = (
    "time",
    "pos_x",
    "pos_y",
    "pos_z",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w",
)


def _resolve_export_scene_params(meta: dict, source_motion) -> tuple[float, float, float]:
    """Return ``(smpl_scale, z_offset_skeleton, z_offset_terrain)`` for export."""
    from hhtools.core.grounding import terrain_heightfield_z_offset_world

    smpl_scale = float(meta.get("smpl_scale", 1.0))
    z_offset = float(meta.get("source_z_min", 0.0))
    terrain = getattr(source_motion, "terrain", None)
    if terrain is None:
        return smpl_scale, z_offset, z_offset

    z_terrain_raw = meta.get("source_terrain_z_offset")
    if z_terrain_raw is not None and np.isfinite(float(z_terrain_raw)):
        z_terrain = float(z_terrain_raw)
    else:
        z_terrain = float(terrain_heightfield_z_offset_world(source_motion, z_offset))
    return smpl_scale, z_offset, z_terrain


def _yellow_foot_z_from_motion_meta(source_motion, meta: dict) -> float | None:
    """Lowest ankle/foot Z in the retarget robot frame (``smpl_scale`` + floor snap).

    Used when no browser scaled-preview is available (batch export) so the
    baked root_z still tracks terrain/object height the same way playback does.
    """
    if source_motion is None:
        return None
    try:
        smpl_scale = float(meta.get("smpl_scale", 1.0))
        z_min = float(meta.get("source_z_min", 0.0))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(smpl_scale) or smpl_scale <= 0.0:
        return None
    hierarchy = getattr(source_motion, "hierarchy", None)
    names = list(getattr(hierarchy, "bone_names", []) or [])
    positions = np.asarray(getattr(source_motion, "positions", None), dtype=np.float64)
    if not names or positions.ndim != 3 or positions.shape[0] == 0:
        return None

    def _norm(name: str) -> str:
        return str(name).lower().replace("_", "").replace(" ", "")

    name_to_i = {_norm(n): i for i, n in enumerate(names)}
    zs: list[float] = []
    pos0 = positions[0]
    for key in (
        "leftankle",
        "rightankle",
        "leftfoot",
        "rightfoot",
        "leftleg",
        "rightleg",
    ):
        idx = name_to_i.get(key)
        if idx is not None and idx < pos0.shape[0]:
            zs.append(float((pos0[idx, 2] - z_min) * smpl_scale))
    return min(zs) if zs else None


def _bake_export_joint_q(
    model,
    retargeted,
    joint_q: np.ndarray,
    source_motion,
    meta: dict,
    *,
    scaled_preview: dict | None = None,
    yellow_foot_z: float | None = None,
    preserve_absolute_z: bool | None = None,
) -> tuple[np.ndarray, float]:
    """Bake viewer ``mesh_z_lift`` into robot ``root_z`` for CSV/PKL export."""
    from hhtools.web.serialize import bake_playback_mesh_z_lift_into_joint_q

    if model is None or not hasattr(model, "trimesh_scene"):
        return np.asarray(joint_q, dtype=np.float32), 0.0

    if preserve_absolute_z is None:
        preserve_absolute_z = getattr(source_motion, "terrain", None) is not None
    yellow = yellow_foot_z
    if yellow is None and scaled_preview is None:
        has_scene = motion_has_scene(source_motion)
        if has_scene or preserve_absolute_z:
            yellow = _yellow_foot_z_from_motion_meta(source_motion, meta)

    return bake_playback_mesh_z_lift_into_joint_q(
        model,
        retargeted,
        joint_q,
        scaled_preview=scaled_preview,
        yellow_foot_z=yellow,
        preserve_absolute_z=bool(preserve_absolute_z),
    )


def bake_export_root_z(
    model,
    retargeted,
    joint_q: np.ndarray | None = None,
    source_motion=None,
    *,
    scaled_preview: dict | None = None,
    yellow_foot_z: float | None = None,
    preserve_absolute_z: bool | None = None,
) -> tuple[np.ndarray, float]:
    """Public helper: bake viewer mesh Z lift into ``joint_q`` for any export path."""
    q = joint_q if joint_q is not None else np.asarray(retargeted.joint_q, dtype=np.float32)
    meta = dict(getattr(retargeted, "meta", {}) or {})
    return _bake_export_joint_q(
        model,
        retargeted,
        q,
        source_motion,
        meta,
        scaled_preview=scaled_preview,
        yellow_foot_z=yellow_foot_z,
        preserve_absolute_z=preserve_absolute_z,
    )


def _scaled_terrain(source_motion, smpl_scale: float, z_terrain: float):
    terrain = getattr(source_motion, "terrain", None)
    if terrain is None:
        return None
    try:
        return terrain.scaled(float(smpl_scale), z_offset=float(z_terrain))
    except Exception:
        return terrain


def _robot_pkl_blob(retargeted, joint_q: np.ndarray, sample_rate: float, meta: dict) -> dict[str, object]:
    joint_q_wxyz = np.empty_like(joint_q)
    joint_q_wxyz[:, :3] = joint_q[:, :3]
    joint_q_wxyz[:, 3] = joint_q[:, 6]
    joint_q_wxyz[:, 4] = joint_q[:, 3]
    joint_q_wxyz[:, 5] = joint_q[:, 4]
    joint_q_wxyz[:, 6] = joint_q[:, 5]
    if joint_q.shape[1] > 7:
        joint_q_wxyz[:, 7:] = joint_q[:, 7:]

    dof_all = list(getattr(retargeted, "dof_names", []) or [])
    nq_act = joint_q.shape[1] - 7
    actuated_dof_names = dof_all[-nq_act:] if len(dof_all) >= nq_act else dof_all

    return {
        "joint_q": joint_q_wxyz,
        "dof_names": actuated_dof_names,
        "sample_rate": float(sample_rate),
        "name": str(getattr(retargeted, "name", "retargeted")),
        "root_quat_format": "wxyz",
        "smpl_scale": float(meta.get("smpl_scale", 1.0)),
        "z_offset": float(meta.get("source_z_min", 0.0)),
        "meta": {k: str(v) for k, v in meta.items()},
    }


def _object_track_blob(
    ob,
    retargeted,
    *,
    smpl_scale: float,
    z_offset: float,
) -> dict[str, object]:
    """One interaction object's trajectory in the retarget / robot frame."""
    op = np.asarray(ob.positions, dtype=np.float32).copy()
    op[:, 2] -= float(z_offset)
    op *= float(smpl_scale)
    oq_xyzw = np.asarray(ob.quaternions, dtype=np.float32)
    oq_wxyz = np.empty_like(oq_xyzw)
    oq_wxyz[..., 0] = oq_xyzw[..., 3]
    oq_wxyz[..., 1:] = oq_xyzw[..., :3]
    mesh_name = Path(str(getattr(ob, "mesh_path", "") or "")).name
    # Cuboid extents follow the same mesh scaling as the exported OBJ
    # (``ob.scale * smpl_scale``); they are dimensions, not positions, so the
    # ``z_offset`` grounding shift does not apply.
    extents = (
        np.asarray(getattr(ob, "extents", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(3)
        * float(getattr(ob, "scale", 1.0))
        * float(smpl_scale)
    )
    return {
        "name": str(ob.name),
        "positions": op,
        "quaternions": oq_wxyz,
        "extents": extents.astype(np.float32),
        "mesh_filename": mesh_name,
        "mesh_path": str(getattr(ob, "mesh_path", "") or ""),
        "sample_rate": float(retargeted.sample_rate),
        "quat_format": "wxyz",
        "frame": "retarget_robot",
    }


def _object_track_blobs(
    retargeted,
    source_motion,
    *,
    smpl_scale: float,
    z_offset: float,
) -> list[dict[str, object]]:
    return [
        _object_track_blob(ob, retargeted, smpl_scale=smpl_scale, z_offset=z_offset)
        for ob in (getattr(source_motion, "objects", []) or [])
    ]


def _save_object_track_csv(
    path: Path,
    blob: dict[str, object],
    *,
    meta: dict | None = None,
    include_header: bool = True,
) -> Path:
    """Write one interaction-object trajectory as CSV (robot frame, xyzw quat)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.asarray(blob["positions"], dtype=np.float64)
    quats_wxyz = np.asarray(blob["quaternions"], dtype=np.float64)
    sample_rate = float(blob["sample_rate"])
    num_frames = int(positions.shape[0])
    times = np.arange(num_frames, dtype=np.float64) / max(sample_rate, 1.0)

    # ``ext_*`` cuboid dimensions are intentionally not written: consumers read
    # the object geometry from the sidecar ``.obj`` mesh instead.
    header_meta = {
        "object": str(blob["name"]),
        "sample_rate": f"{sample_rate:.6f}",
        "quat_format": "xyzw",
        "frame": "retarget_robot",
        "mesh_filename": str(blob.get("mesh_filename", "")),
    }
    header_meta.update(meta or {})

    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if include_header:
            for key in sorted(header_meta):
                fp.write(f"# {key}: {header_meta[key]}\n")
            writer.writerow(OBJECT_CSV_HEADER)
        for frame in range(num_frames):
            q = quats_wxyz[frame]
            writer.writerow([
                f"{times[frame]:.6f}",
                f"{positions[frame, 0]:.6f}",
                f"{positions[frame, 1]:.6f}",
                f"{positions[frame, 2]:.6f}",
                f"{q[1]:.6f}",
                f"{q[2]:.6f}",
                f"{q[3]:.6f}",
                f"{q[0]:.6f}",
            ])
    return path


def _write_object_tracks(
    clip_dir: Path,
    retargeted,
    source_motion,
    *,
    smpl_scale: float,
    z_offset: float,
    fmt: str,
    csv_header: bool = True,
) -> list[str]:
    """Write ``object_<i>_<name>.{csv|pkl}`` sidecars in the robot frame."""
    written: list[str] = []
    use_csv = (fmt or "csv").lower() == "csv"
    for idx, ob in enumerate(getattr(source_motion, "objects", []) or []):
        blob = _object_track_blob(
            ob, retargeted, smpl_scale=smpl_scale, z_offset=z_offset,
        )
        safe_name = "".join(
            (c if c.isalnum() or c in "._-" else "_" for c in str(ob.name)),
        )
        ext = "csv" if use_csv else "pkl"
        obj_path = clip_dir / f"object_{idx}_{safe_name}.{ext}"
        if use_csv:
            _save_object_track_csv(obj_path, blob, include_header=csv_header)
        else:
            with open(obj_path, "wb") as f:
                pickle.dump(blob, f)
        written.append(obj_path.name)
        _log.info(
            "object track %s %s (frames=%d)",
            ext,
            obj_path.name,
            int(np.asarray(blob["positions"]).shape[0]),
        )
    return written


def _export_scaled_object_mesh(ob, dst: Path, mesh_scale: float) -> bool:
    """Centre mesh on centroid and scale vertices to the robot frame."""
    raw = str(getattr(ob, "mesh_path", "") or "").strip()
    if not raw:
        return False
    src = Path(raw)
    if not src.is_file():
        return False
    try:
        import trimesh

        loaded = trimesh.load(str(src), force="mesh", process=False)
        verts = np.asarray(getattr(loaded, "vertices", np.zeros((0, 3))), dtype=np.float64)
        faces = np.asarray(getattr(loaded, "faces", np.zeros((0, 3))), dtype=np.int64)
        if verts.size == 0 or faces.size == 0:
            return False
        centroid = verts.mean(axis=0)
        verts = (verts - centroid) * float(mesh_scale)
        mesh = trimesh.Trimesh(
            vertices=verts.astype(np.float32),
            faces=faces,
            process=False,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(dst))
        return True
    except Exception as exc:
        _log.warning("scaled object mesh export failed %s → %s: %s", src, dst, exc)
        return False


def _copy_scene_meshes(
    clip_dir: Path,
    source_motion,
    stem: str,
    *,
    smpl_scale: float,
    z_terrain: float,
) -> list[str]:
    """Export robot-scaled terrain OBJ + interaction object meshes into ``clip_dir``."""
    from hhtools.io.parc_import import heightfield_to_wavefront_obj

    copied: list[str] = []
    terrain_robot = _scaled_terrain(source_motion, smpl_scale, z_terrain)
    if terrain_robot is not None:
        obj_path = clip_dir / f"{stem}_terrain.obj"
        try:
            heightfield_to_wavefront_obj(terrain_robot, obj_path)
            copied.append(obj_path.name)
        except OSError as exc:
            _log.warning("terrain OBJ export failed: %s", exc)

    for ob in getattr(source_motion, "objects", []) or []:
        raw = str(getattr(ob, "mesh_path", "") or "").strip()
        if not raw:
            continue
        src = Path(raw)
        dst = clip_dir / src.name
        mesh_scale = float(getattr(ob, "scale", 1.0)) * float(smpl_scale)
        if _export_scaled_object_mesh(ob, dst, mesh_scale):
            copied.append(dst.name)
        elif src.is_file() and dst.resolve() != src.resolve():
            try:
                shutil.copy2(src, dst)
                copied.append(dst.name)
                _log.warning(
                    "fell back to unscaled mesh copy for %s (trimesh export failed)",
                    src.name,
                )
            except OSError as exc:
                _log.warning("could not copy mesh %s → %s: %s", src, dst, exc)
    return copied


def identity_resample(retargeted: Any, fps: float | None):
    """No-op resample helper for CLI / batch exporters (keeps native sample rate)."""
    src = float(getattr(retargeted, "sample_rate", 30.0))
    jq = np.asarray(retargeted.joint_q, dtype=np.float32)
    if fps is None or fps <= 0 or abs(float(fps) - src) < 1e-6:
        return jq, src
    from hhtools.web.serialize import resample_joint_q

    rc = int(getattr(retargeted, "root_coord_count", 7))
    return resample_joint_q(jq, src, float(fps), root_coord_count=rc), float(fps)


def write_retarget_export_bundle(
    retargeted: Any,
    model,
    source_motion,
    out_root: str | Path,
    *,
    stem: str,
    fps: float | None,
    fmt: str,
    backend: str,
    resample_fn,
    csv_header: bool = True,
    source_path: str | Path | None = None,
    scaled_preview: dict | None = None,
    yellow_foot_z: float | None = None,
    pack_scene: bool = True,
) -> Path:
    """Write a clip bundle and return the path to a ``.zip`` (or bare file if no scene).

    ``resample_fn`` is ``_resample_retargeted`` from :mod:`hhtools.web.server` to
    avoid a circular import at module load time.

    Robot ``root_z`` is baked with the same constant ``mesh_z_lift`` the browser
    applies during playback so external sims match ground / terrain / objects.

    When ``pack_scene`` is False (batch / offline), scene clips keep an uncompressed
    folder (``<stem>/<stem>.csv`` + terrain/object sidecars) instead of a ``.zip``.
    File contents match the Web export either way.
    """
    import dataclasses

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    fmt = (fmt or "csv").lower()
    has_scene = motion_has_scene(source_motion)

    joint_q, sample_rate = resample_fn(retargeted, fps)
    meta = dict(getattr(retargeted, "meta", {}) or {})
    joint_q, playback_lift = _bake_export_joint_q(
        model,
        retargeted,
        joint_q,
        source_motion,
        meta,
        scaled_preview=scaled_preview,
        yellow_foot_z=yellow_foot_z,
    )
    if abs(playback_lift) > 1e-12:
        meta["playback_mesh_z_lift"] = f"{playback_lift:.6f}"
    ret2 = dataclasses.replace(retargeted, joint_q=joint_q, sample_rate=sample_rate, meta=meta)
    smpl_scale, z_offset, z_terrain = _resolve_export_scene_params(meta, source_motion)

    clip_dir = resolve_clip_export_dir(
        out_root, stem, source_path, has_scene=has_scene,
    )
    # Flat AMASS / LAFAN-style clips share one ``out_root`` (``clip_dir == out_root``).
    # Wiping ``clip_dir`` before each write would delete every prior CSV in a batch
    # export — only replace dedicated per-clip bundle directories.
    flat_shared_dir = (
        not has_scene
        and clip_dir.resolve() == out_root.resolve()
    )
    if flat_shared_dir:
        clip_dir.mkdir(parents=True, exist_ok=True)
    else:
        if clip_dir.exists() and clip_dir.is_dir():
            shutil.rmtree(clip_dir, ignore_errors=True)
        clip_dir.mkdir(parents=True, exist_ok=True)

    csv_meta = {"retarget_backend": backend}
    if "playback_mesh_z_lift" in meta:
        csv_meta["playback_mesh_z_lift"] = meta["playback_mesh_z_lift"]

    if fmt == "pkl":
        blob: dict[str, object] = {
            "hhtools_export": "retarget_v1",
            "format": "pkl",
            "retarget_backend": backend,
            "robot": _robot_pkl_blob(ret2, joint_q, sample_rate, meta),
            "objects": _object_track_blobs(ret2, source_motion, smpl_scale=smpl_scale, z_offset=z_offset),
        }
        terrain_robot = _scaled_terrain(source_motion, smpl_scale, z_terrain)
        if terrain_robot is not None:
            blob["terrain_data"] = terrain_robot.to_ms_terrain_data_dict()
        pkl_path = clip_dir / f"{stem}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(blob, f)
    else:
        from hhtools.io.robot_csv import save_robot_csv

        save_robot_csv(
            clip_dir / f"{stem}.csv",
            robot=model,
            joint_q=joint_q,
            sample_rate=sample_rate,
            meta=csv_meta,
            include_header=csv_header,
        )

    object_tracks: list[str] = []
    if getattr(source_motion, "objects", None):
        object_tracks = _write_object_tracks(
            clip_dir,
            ret2,
            source_motion,
            smpl_scale=smpl_scale,
            z_offset=z_offset,
            fmt=fmt,
            csv_header=csv_header,
        )

    mesh_names: list[str] = []
    if has_scene:
        mesh_names = _copy_scene_meshes(
            clip_dir, source_motion, stem, smpl_scale=smpl_scale, z_terrain=z_terrain,
        )

    if not has_scene and fmt == "csv":
        return clip_dir / f"{stem}.csv"

    if not has_scene:
        return clip_dir / (f"{stem}.pkl" if fmt == "pkl" else f"{stem}.csv")

    if not pack_scene:
        _log.info(
            "export folder %s (meshes=%s, object_tracks=%s)",
            clip_dir,
            mesh_names,
            object_tracks,
        )
        return clip_dir

    # ``make_archive(out_root/stem, root_dir=clip_dir)`` hangs when
    # ``clip_dir == out_root`` (batch upload layout: ``out/sub10/sub10/``):
    # the ``.zip`` is created *inside* the tree being walked and gets re-
    # included.  ``zip_directory`` always writes the archive beside ``clip_dir``.
    zip_path = zip_directory(clip_dir, stem)
    shutil.rmtree(clip_dir, ignore_errors=True)
    _log.info(
        "export bundle %s (meshes=%s, object_tracks=%s)",
        zip_path.name,
        mesh_names,
        object_tracks,
    )
    return zip_path


def zip_directory(
    src_dir: Path,
    zip_stem: str,
    *,
    compress: bool = False,
) -> Path:
    """Zip ``src_dir`` contents to ``zip_stem``.zip`` next to it.

    Batch exports default to ``compress=False`` (``ZIP_STORED``): CSV/PKL are
    already mostly unique floats; DEFLATE buys little and costs a lot on large
    trees (43×3000-frame clips).
    """
    import zipfile

    src_dir = Path(src_dir)
    archive_path = src_dir.parent / f"{zip_stem}.zip"
    if archive_path.exists():
        archive_path.unlink()
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    kwargs: dict = {"compression": compression}
    if compress:
        kwargs["compresslevel"] = 3
    with zipfile.ZipFile(archive_path, "w", **kwargs) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir).as_posix())
    return archive_path
