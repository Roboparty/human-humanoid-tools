# SPDX-License-Identifier: Apache-2.0
"""Web glue for the "数据集可视化分析 / Dataset Visualization & Analysis" panel.

Bridges the library scan and the :mod:`hhtools.analysis` pipeline behind the
``/api/dataset/*`` endpoints.  Results are cached on disk so re-opening the panel
avoids recomputing features; the cache key binds inputs, sidecars, and configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from hhtools.services.dataset_catalog import build_analysis_entries as build_entries

ProgressCb = Callable[[float, str], None]

_SOURCE_HINT_FILE = ".hhtools_source.json"
# Bump when user-facing numeric metrics change so old manifests are not reused.
_METRIC_CACHE_SCHEMA = 3


def save_upload_source_hint(drop_dir: Path, user_source_root: str) -> None:
    root = str(user_source_root or "").strip()
    if not root:
        return
    hint = Path(drop_dir) / _SOURCE_HINT_FILE
    hint.write_text(
        json.dumps({"user_source_root": root}, ensure_ascii=False),
        encoding="utf-8",
    )


def read_upload_source_hint(source_root: str | Path | None) -> str | None:
    if not source_root:
        return None
    hint = Path(source_root) / _SOURCE_HINT_FILE
    if not hint.is_file():
        return None
    try:
        data = json.loads(hint.read_text(encoding="utf-8"))
        root = str(data.get("user_source_root") or "").strip()
        return root or None
    except Exception:
        return None


def _cache_dir(source_root: Path, fallback: Path) -> Path:
    """Prefer ``{source_root}/.hhtools_analysis`` if writable, else ``fallback``."""
    candidate = source_root / ".hhtools_analysis"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return candidate
    except Exception:
        d = fallback / ".hhtools_analysis"
        d.mkdir(parents=True, exist_ok=True)
        return d


def _manifest_path(source_root: Path, fallback: Path, embedding: str) -> Path:
    return _cache_dir(source_root, fallback) / f"manifest_{embedding}.json"


def _fingerprint(
    entries: list[dict[str, Any]],
    *,
    source_root: Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> str | None:
    """Bind paths, file contents, sidecars, config, and algorithm.

    Content hashing also detects edits on coarse-timestamp filesystems and
    replacements that preserve timestamps. Read in chunks; failed, missing,
    or concurrently changing inputs never form a reusable snapshot.
    """
    from hhtools.analysis.config import load_config

    try:
        sources = [Path(entry["source_path"]).absolute() for entry in entries]
        root = source_root.absolute() if source_root is not None else (
            Path(os.path.commonpath([str(path.parent) for path in sources])) if sources else None
        )
        paths = set(sources)
        if root is not None:
            def fail_walk(error: OSError) -> None:
                raise error

            for directory, children, files in os.walk(root, onerror=fail_walk):
                children[:] = [name for name in children if not name.startswith(".") and name != "__pycache__"]
                paths.update(Path(directory) / name for name in files if not name.startswith("."))
        identities = []
        for path in sorted(paths):
            info = path.stat()
            if not path.is_file():
                return None
            with path.open("rb") as stream:
                content_digest = hashlib.file_digest(stream, "sha256").hexdigest()
            after = path.stat()
            if (
                info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_dev, info.st_ino
            ) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_dev, after.st_ino
            ):
                return None
            identities.append([
                path.relative_to(root).as_posix() if root is not None else str(path),
                info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_dev, info.st_ino,
                content_digest,
            ])
        payload = {
            "source_root": str(root) if root is not None else None,
            "files": identities,
            "entries": sorted(entries, key=lambda entry: str(entry["source_path"])),
            "config": load_config() if cfg is None else cfg,
            "metric_schema": _METRIC_CACHE_SCHEMA,
        }
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
    except (OSError, KeyError, TypeError, ValueError):
        return None


def load_cached(
    source_root: Path, fallback: Path, embedding: str, entries: list[dict[str, Any]],
    *, cfg: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a cached result if present and still fresh, else ``None``."""
    path = _manifest_path(source_root, fallback, embedding)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("meta"), dict):
        return None
    fingerprint = _fingerprint(entries, source_root=source_root, cfg=cfg)
    if fingerprint is None or data["meta"].get("fingerprint") != fingerprint:
        return None
    if data.get("meta", {}).get("metric_schema") != _METRIC_CACHE_SCHEMA:
        return None
    return data


def run_analysis(
    source_root: Path,
    fallback: Path,
    *,
    embedding: str = "handcrafted",
    cfg_override: dict[str, Any] | None = None,
    force: bool = False,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Analyze the library under ``source_root`` and return ``{meta, clips, summary}``."""
    from hhtools.analysis.collection import analyze_entries, build_summary
    from hhtools.analysis.config import load_config

    entries = build_entries(source_root)
    cfg = load_config(cfg_override)
    if not force:
        cached = load_cached(source_root, fallback, embedding, entries, cfg=cfg)
        if cached is not None:
            if progress is not None:
                progress(1.0, "命中缓存")
            return cached

    fingerprint = _fingerprint(entries, source_root=source_root, cfg=cfg)
    clips = analyze_entries(
        entries, cfg=cfg, embedding_name=embedding, progress=progress
    )
    summary = build_summary(clips, cfg)
    payload = {
        "meta": {
            "source_root": str(source_root),
            "embedding": embedding,
            "fingerprint": fingerprint,
            "metric_schema": _METRIC_CACHE_SCHEMA,
            "generated_at": time.time(),
        },
        "clips": [c.to_dict() for c in clips],
        "summary": summary,
    }
    # Never publish a mixed snapshot under the identity of newer inputs.
    if fingerprint is None or fingerprint != _fingerprint(entries, source_root=source_root, cfg=cfg):
        payload["meta"]["fingerprint"] = None
        return payload
    path = _manifest_path(source_root, fallback, embedding)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(path)
    except Exception:
        pass
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def compute_subset(
    clips: list[dict[str, Any]], k: int, alpha: float = 0.99
) -> list[str]:
    """Run Global Weighted FPS over clip embeddings; return selected clip_ids."""
    import numpy as np

    from hhtools.analysis.subset import global_weighted_fps

    usable = [c for c in clips if c.get("embedding")]
    if not usable:
        return []
    Z = np.array([c["embedding"] for c in usable], dtype=np.float64)
    C = np.array([float(c.get("metrics", {}).get("complexity", 0.0)) for c in usable])
    sel = global_weighted_fps(Z, C, int(k), alpha=float(alpha))
    return [usable[i]["clip_id"] for i in sel]


def export_manifest(
    clips: list[dict[str, Any]],
    ids: list[str],
    *,
    analyze_source: str | None = None,
    user_source_root: str | None = None,
    archive_paths: dict[str, str] | None = None,
    path_basis: str | None = None,
) -> str:
    """Build a training-manifest JSON string for the selected clip ids."""
    rows = _manifest_rows(
        clips,
        ids,
        analyze_source=analyze_source,
        user_source_root=user_source_root,
        archive_paths=archive_paths,
    )
    meta: dict[str, Any] = {
        "path_basis": path_basis or ("user_local" if user_source_root else "analysis_relative")
    }
    if user_source_root:
        meta["user_source_root"] = user_source_root
    payload: dict[str, Any] = {"count": len(rows), "meta": meta, "clips": rows}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_manifest_csv(
    clips: list[dict[str, Any]],
    ids: list[str],
    *,
    analyze_source: str | None = None,
    user_source_root: str | None = None,
    archive_paths: dict[str, str] | None = None,
) -> str:
    """Compact CSV: path + tags + key metrics for downstream pipelines."""
    import csv
    import io

    rows = _manifest_rows(
        clips,
        ids,
        analyze_source=analyze_source,
        user_source_root=user_source_root,
        archive_paths=archive_paths,
    )
    buf = io.StringIO()
    fields = [
        "clip_id", "source_path", "dataset", "folder_label", "source_kind",
        "tags", "s_phy", "complexity", "duration_s",
    ]
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        m = r.get("metrics") or {}
        w.writerow({
            "clip_id": r["clip_id"],
            "source_path": r.get("source_path", ""),
            "dataset": r.get("dataset", ""),
            "folder_label": r.get("folder_label", ""),
            "source_kind": r.get("source_kind", ""),
            "tags": ";".join(r.get("tags") or []),
            "s_phy": m.get("s_phy", ""),
            "complexity": m.get("complexity", ""),
            "duration_s": m.get("duration_s", ""),
        })
    return buf.getvalue()


def _path_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_manifest_source_path(
    disk_path: str | Path,
    *,
    analyze_source: str | Path | None = None,
    user_source_root: str | Path | None = None,
    archive_relpath: str | None = None,
    folder_label: str | None = None,
    clip_id: str | None = None,
) -> str:
    """Map an on-disk clip path to a manifest ``source_path``.

    * **archive** export — path inside the ZIP (``mimic/foo.csv``, ``clip_export/…``)
    * **JSON** export — ``user_source_root`` + relative layout from upload / labels
    * otherwise — path relative to the analyzed directory
    """
    if archive_relpath:
        return archive_relpath.replace("\\", "/")

    disk_path = Path(disk_path).resolve()
    user_root = str(user_source_root or "").strip()
    if not user_root:
        analyze = str(analyze_source or "").strip()
        relative_path: str | None = None
        if analyze:
            try:
                relative_path = disk_path.relative_to(Path(analyze).resolve()).as_posix()
            except ValueError:
                pass
        label = str(folder_label or "").strip().replace("\\", "/")
        return relative_path or (f"{label}/{disk_path.name}" if label else disk_path.name)

    user_base = Path(user_root).resolve()
    analyze = str(analyze_source or "").strip()
    if analyze:
        try:
            rel = disk_path.relative_to(Path(analyze).resolve())
            parts = rel.parts
            if parts and parts[0] == user_base.name and len(parts) > 1:
                rel = Path(*parts[1:])
            return str(user_base / rel).replace("\\", "/")
        except ValueError:
            pass

    label = str(folder_label or "").strip().replace("\\", "/")
    fname = disk_path.name
    if label:
        leaf = label.split("/")[-1]
        if user_base.name == leaf:
            return str(user_base / fname).replace("\\", "/")
        return str(user_base / label / fname).replace("\\", "/")

    cid = str(clip_id or "").strip().replace("\\", "/")
    if cid and "/" in cid:
        folder = cid.rsplit("/", 1)[0]
        leaf = folder.split("/")[-1]
        if user_base.name == leaf:
            return str(user_base / fname).replace("\\", "/")
        return str(user_base / folder / fname).replace("\\", "/")

    return str(user_base / fname).replace("\\", "/")


def _clip_is_robot(clip: dict[str, Any]) -> bool:
    if clip.get("source_kind") == "robot":
        return True
    return str(clip.get("dataset") or "") == "robot"


def _resolve_allowed_roots(allowed_roots: list[Path] | None) -> list[Path]:
    roots: list[Path] = []
    for r in allowed_roots or []:
        p = Path(r).resolve()
        if p.is_dir() and p not in roots:
            roots.append(p)
    return roots


def _clip_export_dir(source_path: str | Path) -> Path:
    path = Path(source_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"trajectory not found: {path}")
    return path.parent


def _export_ignore(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n.startswith(".")}


def _clip_has_scene_folder(traj: Path) -> bool:
    """True when the trajectory lives in a meshmimic / intermimic clip folder."""
    traj = Path(traj).resolve()
    clip_dir = traj.parent
    stem = traj.stem
    if clip_dir.name.endswith("_export"):
        return True
    if (clip_dir / f"{stem}_terrain.obj").is_file():
        return True
    if any(clip_dir.glob("*_terrain.obj")):
        return True
    if (clip_dir / f"{stem}_cleaned_simplified.obj").is_file():
        return True
    if any(clip_dir.glob("*_cleaned_simplified.obj")):
        return True
    if any(clip_dir.glob("object_*.csv")):
        return True
    return False


def _robot_export_kind(traj: Path) -> str:
    """``folder`` = scene clip dir; ``mimic_file`` = standalone trajectory CSV."""
    return "folder" if _clip_has_scene_folder(traj) else "mimic_file"


def _unique_name(base: str, used: set[str], row: dict[str, Any]) -> str:
    if base not in used:
        used.add(base)
        return base
    label = str(row.get("folder_label") or row.get("clip_id") or "").replace("/", "__")
    name = f"{base}__{label}" if label and label != base else f"{base}_{len(used)}"
    used.add(name)
    return name


def _copy_scene_folder(src: Path, dest: Path) -> None:
    if dest.exists():
        return
    shutil.copytree(src, dest, ignore=_export_ignore)


def export_robot_clips_zip(
    clips: list[dict[str, Any]],
    ids: list[str],
    out_dir: str | Path,
    *,
    zip_stem: str = "robot_subset_export",
    allowed_roots: list[Path] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Pack selected robot clips into a ZIP.

    * **mimic** standalone trajectories (CSV in a batch root) → ``mimic/<file>``
    * **meshmimic / intermimic** clip folders → ``<folder>/`` as imported
    """
    roots = _resolve_allowed_roots(allowed_roots)
    id_set = set(ids)
    robot_clips = [
        c for c in clips
        if c.get("clip_id") in id_set and _clip_is_robot(c)
    ]
    if not robot_clips:
        raise ValueError("没有可打包的机器人 clip")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / f".staging_{int(time.time() * 1000)}"
    staging.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, str]] = []
    mimic_used: set[str] = set()
    folder_used: set[str] = set()
    folder_src: dict[str, Path] = {}
    archive_paths: dict[str, str] = {}
    seen_traj: set[str] = set()
    try:
        for clip in robot_clips:
            row = {
                "clip_id": clip["clip_id"],
                "source_path": clip.get("source_path", ""),
                "folder_label": clip.get("folder_label", ""),
            }
            traj = Path(str(row["source_path"])).resolve()
            traj_key = str(traj)
            if traj_key in seen_traj:
                continue
            seen_traj.add(traj_key)

            clip_dir = traj.parent
            if roots and not any(_path_under_root(clip_dir, root) for root in roots):
                raise PermissionError(f"clip path not under allowed roots: {clip_dir}")

            kind = _robot_export_kind(traj)
            if kind == "mimic_file":
                mimic_dir = staging / "mimic"
                mimic_dir.mkdir(parents=True, exist_ok=True)
                fname = _unique_name(traj.name, mimic_used, row)
                shutil.copy2(traj, mimic_dir / fname)
                archive_path = f"mimic/{fname}"
            else:
                folder_name = _unique_name(clip_dir.name, folder_used, row)
                dest = staging / folder_name
                prev = folder_src.get(folder_name)
                if prev is not None and prev == clip_dir:
                    archive_path = folder_name
                else:
                    folder_src[folder_name] = clip_dir
                    _copy_scene_folder(clip_dir, dest)
                    archive_path = folder_name
                archive_path = f"{archive_path}/{traj.name}"

            archive_paths[row["clip_id"]] = archive_path
            copied.append({
                "clip_id": row["clip_id"],
                "kind": kind,
                "folder": archive_path.rsplit("/", 1)[0] if "/" in archive_path else archive_path,
                "source_path": archive_path,
            })

        manifest_rows = _manifest_rows(
            clips,
            ids,
            archive_paths=archive_paths,
        )
        manifest = {
            "count": len(manifest_rows),
            "exported_at": time.time(),
            "meta": {"path_basis": "archive"},
            "clips": manifest_rows,
            "folders": copied,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        from hhtools.io.export_bundle import zip_directory

        zip_path = zip_directory(staging, zip_stem)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    stats = {
        "clip_count": len(copied),
        "folders": [c["folder"] for c in copied],
        "zip_name": zip_path.name,
    }
    return zip_path, stats


def _manifest_rows(
    clips: list[dict[str, Any]],
    ids: list[str],
    *,
    analyze_source: str | None = None,
    user_source_root: str | None = None,
    archive_paths: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    id_set = set(ids)
    rows: list[dict[str, Any]] = []
    for c in clips:
        if c["clip_id"] not in id_set:
            continue
        clip_id = c["clip_id"]
        disk_path = c.get("source_path", "")
        if archive_paths and clip_id in archive_paths:
            source_path = archive_paths[clip_id]
        else:
            source_path = resolve_manifest_source_path(
                disk_path,
                analyze_source=analyze_source,
                user_source_root=user_source_root,
                folder_label=c.get("folder_label"),
                clip_id=clip_id,
            )
        rows.append({
            "clip_id": clip_id,
            "source_path": source_path,
            "dataset": c.get("dataset", ""),
            "folder_label": c.get("folder_label", ""),
            "source_kind": c.get("source_kind", ""),
            "tags": c.get("tags", []),
            "metrics": c.get("metrics", {}),
        })
    return rows


def scan_upload_summary(source_root: Path) -> dict[str, Any]:
    """Return clip/folder counts after a folder upload lands on disk."""
    entries = build_entries(source_root)
    folders: dict[str, int] = {}
    kinds: dict[str, int] = {"human": 0, "robot": 0}
    for e in entries:
        folders[e["folder_label"]] = folders.get(e["folder_label"], 0) + 1
        if e.get("dataset") == "robot":
            kinds["robot"] += 1
        else:
            kinds["human"] += 1
    return {
        "source": str(source_root),
        "user_source_root": read_upload_source_hint(source_root),
        "clip_count": len(entries),
        "robot_count": kinds["robot"],
        "human_count": kinds["human"],
        "folders": folders,
        "clips": [
            {"clip_id": e["clip_id"], "folder_label": e["folder_label"]}
            for e in entries
        ],
        "entries_preview": entries[:5],
    }


def _prune_empty_dirs(root: Path) -> None:
    """Remove empty directories under ``root`` (bottom-up)."""
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        p = Path(dirpath)
        if p == root:
            continue
        if not any(p.iterdir()):
            p.rmdir()


def remove_upload_folder(source_root: Path, folder_label: str) -> dict[str, Any]:
    """Delete one folder group from an upload-analysis batch on disk."""

    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"上传批次不存在: {root}")

    label = str(folder_label or "").strip()
    if not label:
        raise ValueError("folder_label 不能为空")

    entries = build_entries(root)
    matching = [e for e in entries if e["folder_label"] == label]
    if not matching:
        raise FileNotFoundError(f"批次中无此目录: {label}")

    target = root.joinpath(*PurePosixPath(label).parts)
    if target.is_dir():
        shutil.rmtree(target)
    else:
        for entry in matching:
            Path(entry["source_path"]).unlink(missing_ok=True)

    _prune_empty_dirs(root)

    remaining = build_entries(root)
    if not remaining:
        shutil.rmtree(root, ignore_errors=True)
        return {
            "source": "",
            "user_source_root": read_upload_source_hint(root),
            "clip_count": 0,
            "robot_count": 0,
            "human_count": 0,
            "folders": {},
            "clips": [],
            "entries_preview": [],
            "removed_folder": label,
        }
    summary = scan_upload_summary(root)
    summary["removed_folder"] = label
    return summary


__all__ = [
    "build_entries",
    "compute_subset",
    "export_manifest",
    "export_manifest_csv",
    "export_robot_clips_zip",
    "resolve_manifest_source_path",
    "load_cached",
    "read_upload_source_hint",
    "run_analysis",
    "save_upload_source_hint",
    "remove_upload_folder",
    "scan_upload_summary",
]
