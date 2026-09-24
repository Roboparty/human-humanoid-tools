"""Export parsing and bundle-writing helpers for Web runtimes."""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path
from typing import Any


def _dataset_subdir(entry: dict[str, Any]) -> str:
    """Per-dataset export subfolder (e.g. ``AMASS``, ``PHUMA``).

    Prefers the dataset adapter name, falling back to the library folder label.
    """
    raw = entry.get("dataset") or entry.get("folder_label") or "misc"
    name = str(raw).strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "misc"
    return name.upper() if name.islower() and len(name) <= 12 else name


def _batch_export_subdir(entry: dict[str, Any]) -> str | None:
    """Preserve imported directory trees; group library clips by dataset."""
    if entry.get("origin") in {"upload", "local"}:
        sub = (entry.get("export_subdir") or "").strip().replace("\\", "/")
        return sub or None
    return _dataset_subdir(entry)


def _parse_csv_header(value: Any) -> bool:
    """Truthy unless the client explicitly disables comments + column headers."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s not in ("0", "false", "no", "off", "none", "raw", "numeric")


def _parse_optional_fps(value: Any) -> float | None:
    """Positive target fps from API/JSON, or ``None`` to keep the source rate."""
    if value is None or value == "":
        return None
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return None
    return fps if fps > 0 else None


def _parse_optional_time(value: Any, *, name: str = "t") -> float | None:
    """Non-negative seconds for export window bounds, or ``None`` if omitted."""
    if value is None or value == "":
        return None
    try:
        t = float(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{name} must be a number of seconds") from err
    if not math.isfinite(t) or t < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number of seconds")
    return t


def _resample_retargeted(retargeted: Any, fps: float | None):
    """Return a (joint_q, sample_rate) pair, optionally resampled to ``fps``."""
    import numpy as np

    from hhtools.io.scene_serialize import resample_joint_q

    src = float(getattr(retargeted, "sample_rate", 30.0))
    if fps is None or fps <= 0 or abs(fps - src) < 1e-6:
        return np.asarray(retargeted.joint_q, dtype=np.float32), src
    rc = int(getattr(retargeted, "root_coord_count", 7))
    jq = resample_joint_q(retargeted.joint_q, src, float(fps), root_coord_count=rc)
    return jq, float(fps)


def _write_r2r_export(
    retargeted: Any,
    target_model: Any,
    source_motion: Any,
    out_root: str | Path,
    *,
    source_model: Any,
    calibrated_joint_q: dict[str, float],
    entry: dict[str, Any],
    stem: str,
    fps: float | None,
    fmt: str,
    subdir: str | None = None,
    csv_header: bool = True,
    yellow_foot_z: float | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
):
    """R2R clip bundle: target robot traj + rescaled terrain/object sidecars."""
    from hhtools.io.export_bundle import resolve_clip_export_dir
    from hhtools.io.r2r_export_bundle import (
        clip_has_export_scene,
        write_r2r_export_bundle,
    )
    from hhtools.retarget.robot_to_robot import r2r_scene_scale_ratio

    out_dir = Path(out_root)
    if subdir:
        out_dir = out_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = write_r2r_export_bundle(
        retargeted,
        target_model,
        source_motion,
        scene_scale_ratio=r2r_scene_scale_ratio(
            source_model,
            target_model,
            source_motion,
            calibrated_joint_q,
        ),
        entry=entry,
        out_root=out_dir,
        stem=stem,
        fps=fps,
        fmt=fmt,
        resample_fn=_resample_retargeted,
        csv_header=csv_header,
        yellow_foot_z=yellow_foot_z,
        t_start=t_start,
        t_end=t_end,
    )
    if subdir is not None and path.suffix == ".zip":
        from hhtools.io.r2r_export_bundle import resolve_r2r_source_clip_dir

        source_clip_dir = resolve_r2r_source_clip_dir(entry)
        profile = str(entry.get("upload_profile") or "")
        has_scene = bool(entry.get("has_scene")) or (
            clip_has_export_scene(
                source_clip_dir,
                stem=stem,
                profile=profile,
            )
            if source_clip_dir is not None
            else False
        )
        clip_dir = resolve_clip_export_dir(
            out_dir,
            stem,
            entry.get("source_path"),
            has_scene=has_scene,
        )
        clip_dir.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(path), str(clip_dir))
        path.unlink(missing_ok=True)
        return clip_dir
    return path


def _write_export(
    retargeted: Any,
    model: Any,
    source_motion: Any,
    out_root: str | Path,
    *,
    stem: str,
    fps: float | None,
    fmt: str,
    backend: str,
    subdir: str | None = None,
    csv_header: bool = True,
    source_path: str | Path | None = None,
    yellow_foot_z: float | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
):
    """Write a browser-downloadable CSV/PKL bundle (zip when scene props exist)."""
    from hhtools.io.export_bundle import (
        motion_has_scene,
        resolve_clip_export_dir,
        write_retarget_export_bundle,
    )

    out_dir = Path(out_root)
    if subdir:
        out_dir = out_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = write_retarget_export_bundle(
        retargeted,
        model,
        source_motion,
        out_dir,
        stem=stem,
        fps=fps,
        fmt=fmt,
        backend=backend,
        resample_fn=_resample_retargeted,
        csv_header=csv_header,
        source_path=source_path,
        yellow_foot_z=yellow_foot_z,
        t_start=t_start,
        t_end=t_end,
    )
    # Batch jobs unpack per-clip zips into the job tree (final zip later).
    if subdir is not None and path.suffix == ".zip":
        clip_dir = resolve_clip_export_dir(
            out_dir,
            stem,
            source_path,
            has_scene=motion_has_scene(source_motion),
        )
        clip_dir.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(path), str(clip_dir))
        path.unlink(missing_ok=True)
        return clip_dir
    return path
