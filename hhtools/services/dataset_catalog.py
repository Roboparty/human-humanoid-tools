"""Transport-neutral discovery of human and robot clips for analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _clip_folder_label(root: Path, clip_path: Path) -> str:
    """Return the relative parent path used to group a clip in clients."""

    try:
        relative = clip_path.resolve().relative_to(root.resolve())
    except ValueError:
        return clip_path.parent.name or "uploads"
    if relative.parent == Path("."):
        return "uploads"
    return relative.parent.as_posix()


def _clip_stem(root: Path, clip_path: Path) -> str:
    """Return a stable display stem for standalone and clip-folder layouts."""

    if clip_path.stem.lower() == "motion_actor":
        return clip_path.parent.name or clip_path.stem
    if clip_path.parent.name == clip_path.stem:
        return clip_path.stem
    try:
        return clip_path.relative_to(root).stem
    except ValueError:
        return clip_path.stem


def build_analysis_entries(source_root: Path) -> list[dict[str, Any]]:
    """Merge all supported discovery strategies into analysis-ready entries."""

    from hhtools.io.robot_trajectory_detect import trajectory_robot_name
    from hhtools.services.motion_library import scan_library
    from hhtools.services.r2r_upload_resolve import enumerate_r2r_clips
    from hhtools.services.upload_resolve import enumerate_upload_clips

    root = source_root.resolve()
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    def append(
        *,
        source_path: Path,
        clip_id: str,
        dataset: str,
        folder_label: str,
        source_robot: str | None = None,
    ) -> None:
        resolved = str(source_path.resolve())
        if resolved in seen:
            return
        seen.add(resolved)
        entry = {
            "clip_id": clip_id,
            "source_path": resolved,
            "dataset": dataset,
            "folder_label": folder_label,
        }
        if source_robot:
            entry["source_robot"] = source_robot
        entries.append(entry)

    # A robot export can also look like a generic ``.pkl`` motion bundle to the
    # permissive upload scanner (for example, it used to be claimed as OMOMO).
    # Apply the stricter, structure-aware R2R detector first so the shared
    # ``seen`` boundary cannot relabel a valid robot trajectory as human motion.
    for reference in enumerate_r2r_clips(root, profile="auto"):
        path = reference.path.resolve()
        folder_label = _clip_folder_label(root, path) or "robot"
        stem = _clip_stem(root, path)
        append(
            source_path=path,
            clip_id=f"{folder_label}/{stem}" if folder_label else stem,
            dataset="robot",
            folder_label=folder_label,
            source_robot=trajectory_robot_name(path),
        )

    for reference in enumerate_upload_clips(root, profile="auto"):
        folder_label = _clip_folder_label(root, reference.path)
        stem = _clip_stem(root, reference.path)
        append(
            source_path=reference.path,
            clip_id=f"{folder_label}/{stem}" if folder_label else stem,
            dataset=str(reference.dataset or "unknown"),
            folder_label=folder_label or "uploads",
        )

    for entry in scan_library(root):
        append(
            source_path=entry.source_path,
            clip_id=f"{entry.folder_label}/{entry.stem}",
            dataset=entry.dataset,
            folder_label=entry.folder_label,
        )

    entries.sort(key=lambda item: (item["folder_label"].lower(), item["clip_id"].lower()))
    return entries


# Compatibility name used by the previous Web analysis module.
build_entries = build_analysis_entries


__all__ = ["build_analysis_entries"]
