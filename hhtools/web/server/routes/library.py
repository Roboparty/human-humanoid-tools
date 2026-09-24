"""Motion Library discovery and link-management routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from hhtools.services.motion_library_links import motions_library_root
from hhtools.web.server.library_runtime import (
    _enrich_basket_entry,
    _retained_robot_trajectory_entries,
    _source_robot_trajectory_entries,
)


def register_library_routes(
    app,
    *,
    state,
    motion_library_publish_lock,
) -> None:
    @app.get("/api/library")
    def library(source: str | None = None) -> dict:
        from hhtools.services.motion_library import scan_library
        from hhtools.services.motion_library_links import scan_motions_library

        root = Path(source) if source else state.source_root
        merged: list[dict] = []
        seen: set[str] = set()
        for e in scan_library(root):
            row = _enrich_basket_entry(
                {
                    "dataset": e.dataset,
                    "folder_label": e.folder_label,
                    "sequence_id": e.sequence_id,
                    "stem": e.stem,
                    "source_path": str(e.source_path),
                    "label": e.display_label,
                    "origin": "assets",
                }
            )
            seen.add(row["source_path"])
            merged.append(row)
        for raw in _source_robot_trajectory_entries(root):
            row = _enrich_basket_entry(raw)
            if row["source_path"] in seen:
                continue
            seen.add(row["source_path"])
            merged.append(row)
        # Avoid observing a half-copied same-process publish.  The filesystem
        # namespace is still only process-local; multi-worker deployments need
        # a cross-process file lock before they can offer this guarantee.
        with motion_library_publish_lock:
            lib_root = motions_library_root()
            motion_entries = scan_motions_library(lib_root)
        for raw in motion_entries:
            sp = str(raw.get("source_path") or "")
            if not sp or sp in seen:
                continue
            seen.add(sp)
            merged.append(_enrich_basket_entry(raw))
        for raw in _retained_robot_trajectory_entries(state.job_history):
            sp = str(raw.get("source_path") or "")
            if not sp or sp in seen:
                continue
            seen.add(sp)
            merged.append(_enrich_basket_entry(raw))
        merged.sort(
            key=lambda row: (
                str(row.get("folder_label") or "").lower(),
                str(row.get("stem") or "").lower(),
            ),
        )
        folders: list[str] = []
        for row in merged:
            label = str(row.get("folder_label") or "")
            if label and label not in folders:
                folders.append(label)
        return {
            "source_root": str(root),
            "motions_library_root": str(lib_root),
            "folders": folders,
            "entries": merged,
        }

    @app.post("/api/library/link")
    def library_link(body: dict) -> dict:
        from hhtools.services.motion_library_links import link_to_library, scan_motions_library

        path = str(body.get("path") or "").strip()
        folder_label = str(body.get("folder_label") or "").strip() or None
        if not path:
            raise HTTPException(status_code=400, detail="需要 path")
        with motion_library_publish_lock:
            lib_root = motions_library_root()
            dest = link_to_library(
                path,
                folder_label=folder_label,
                library_root=lib_root,
            )
            entries = [
                entry
                for entry in scan_motions_library(lib_root)
                if entry.get("folder_label") == dest.name
            ]
        return {
            "folder_label": dest.name,
            "kind": "directory",
            "clip_count": len(entries),
            "path": str(dest),
            "motions_library_root": str(lib_root),
        }

    @app.delete("/api/library/link/{folder_label}")
    def library_unlink(folder_label: str) -> dict:
        from hhtools.services.motion_library_links import remove_library_folder

        with motion_library_publish_lock:
            removed = remove_library_folder(folder_label)
        if not removed:
            raise HTTPException(status_code=404, detail="link not found")
        return {"removed": folder_label}
