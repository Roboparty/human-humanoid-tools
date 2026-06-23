# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Persist failed batch-retarget clips for easy re-import."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_CLIP_SIDECAR_SUFFIXES = (
    "_terrain.obj",
    "_cleaned_simplified.obj",
)


@dataclass
class BatchFailureLog:
    """On-disk failure bundle under ``<save_dir>/batch_failures/<name>``."""

    root: Path
    items: list[dict] = field(default_factory=list)

    def record(
        self,
        entry: dict,
        *,
        stage: str,
        reason: str,
        reference: str | None = None,
    ) -> dict:
        stem = entry.get("stem") or Path(entry.get("source_path", "?")).stem
        item: dict = {
            "stem": stem,
            "stage": stage,
            "reason": str(reason),
            "reference": reference,
            "dataset": entry.get("dataset"),
            "source_path": entry.get("source_path"),
            "log_rel": None,
        }
        try:
            log_rel = stash_failed_clip(entry, self.root)
            item["log_rel"] = log_rel.as_posix()
        except Exception as err:  # noqa: BLE001 — still surface the retarget error
            item["stash_error"] = str(err)
        self.items.append(item)
        return item

    def finalize(self, *, job_id: str, out_name: str) -> None:
        if not self.items:
            return
        payload = {
            "job_id": job_id,
            "export_name": out_name,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "count": len(self.items),
            "failures": self.items,
        }
        (self.root / "failures.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        lines = [
            "批量 Retarget 失败记录",
            f"时间 (UTC): {payload['failed_at']}",
            f"失败数量: {len(self.items)}",
            f"目录: {self.root}",
            "",
            "每条失败 clip 已按导入时的相对路径复制到本目录下。",
            "修复后可直接：",
            "  1) 把整个 batch_failures/... 文件夹（或其中子目录）拖入 Web「批量」篮子；",
            "  2) 或在本目录中找到对应 clip 文件/文件夹再拖入。",
            "",
            "明细：",
        ]
        for i, f in enumerate(self.items, 1):
            lines.append(
                f"{i}. {f['stem']}  [{f['stage']}]  {f['reason']}"
            )
            lines.append(f"   → {f['log_rel']}")
        (self.root / "失败说明.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def open_batch_failure_log(save_dir: Path, job_id: str, out_name: str) -> BatchFailureLog:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in out_name) or "batch"
    root = Path(save_dir) / "batch_failures" / f"{safe}_{job_id}"
    root.mkdir(parents=True, exist_ok=True)
    return BatchFailureLog(root=root)


def failure_rel_path(entry: dict) -> Path:
    """Relative path inside the failure log, mirroring import layout."""
    src = Path(entry.get("source_path") or "unknown")
    if entry.get("origin") == "upload":
        sub = (entry.get("export_subdir") or "").strip().replace("\\", "/")
        seq = entry.get("sequence_id") or src.name
        return Path(sub) / seq if sub else Path(seq)
    folder = entry.get("folder_label") or entry.get("dataset") or "misc"
    seq = entry.get("sequence_id") or src.name
    return Path(str(folder)) / seq


def _copy_sidecars(src_file: Path, dst_dir: Path) -> None:
    parent = src_file.parent
    stem = src_file.stem
    for suf in _CLIP_SIDECAR_SUFFIXES:
        side = parent / f"{stem}{suf}"
        if side.is_file():
            shutil.copy2(side, dst_dir / side.name)
    pkl = parent / f"{stem}.pkl"
    if pkl.is_file() and src_file.suffix.lower() in {".npz", ".npy"}:
        shutil.copy2(pkl, dst_dir / pkl.name)


def stash_failed_clip(entry: dict, log_root: Path) -> Path:
    """Copy a failed clip (and clip-folder sidecars) into ``log_root``."""
    from hhtools.web.motion_library_links import resolve_clip_on_disk

    src = resolve_clip_on_disk(
        entry["source_path"],
        extra_names=[entry.get("sequence_id") or ""],
        folder_label=entry.get("folder_label"),
        sequence_id=entry.get("sequence_id"),
        upload_drop=entry.get("upload_drop"),
    )
    rel = failure_rel_path(entry)
    if not src.is_file():
        raise FileNotFoundError(f"source clip missing: {src}")

    if src.parent.name == src.stem:
        dst_dir = log_root / rel.parent
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src.parent, dst_dir)
        return rel

    dst = log_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    _copy_sidecars(src, dst.parent)
    return dst.relative_to(log_root)
