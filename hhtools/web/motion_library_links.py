"""User motion library under ``~/.config/hhtools/motions``.

Drag-and-drop from the browser cannot expose client absolute paths.  When the
same files already exist on the **server** (e.g. under ``~/下载``), we locate
them automatically and materialize a symlink directory here.  Otherwise the
upload bytes are copied into this tree.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

_DEFAULT_LOOSE_LABEL = "用户数据集"
_MOTIONS_DIRNAME = "motions"


def motions_library_root() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    user_cfg = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return user_cfg / "hhtools" / _MOTIONS_DIRNAME


def ensure_motions_library() -> Path:
    root = motions_library_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_folder_name(label: str) -> str:
    cleaned = re.sub(r"[^\w.\-+/]+", "_", str(label or "").strip())
    cleaned = cleaned.strip("._/") or _DEFAULT_LOOSE_LABEL
    return cleaned[:120]


def _normalize_relpaths(relative_paths: list[str]) -> list[str]:
    out: list[str] = []
    for raw in relative_paths:
        rel = str(raw or "").replace("\\", "/").lstrip("/")
        if rel:
            out.append(rel)
    return out


def _common_path_parts(rels: list[str]) -> tuple[str, ...]:
    posix_rels = [PurePosixPath(r) for r in rels]
    if not posix_rels:
        return ()
    common: tuple[str, ...] = posix_rels[0].parts
    for rel in posix_rels[1:]:
        common = tuple(
            a for a, b in zip(common, rel.parts, strict=False) if a == b
        )
        if not common:
            break
    return common


def _infer_folder_label(rels: list[str], hint: str | None = None) -> str:
    if hint:
        return _safe_folder_name(hint)
    if not rels:
        return _DEFAULT_LOOSE_LABEL
    if any("/" in r or "\\" in r for r in rels):
        return _safe_folder_name(PurePosixPath(rels[0]).parts[0])
    return _DEFAULT_LOOSE_LABEL


def candidate_search_roots() -> list[Path]:
    home = Path.home()
    raw: list[Path | str] = [
        motions_library_root(),
        home / "下载",
        home / "Downloads",
        home / "data",
        home / "datasets",
        home / "motions",
        Path("/home/motions"),
    ]
    extra = os.environ.get("HHTOOLS_MOTION_SEARCH_PATHS", "")
    if extra:
        raw.extend(p.strip() for p in extra.split(os.pathsep) if p.strip())
    seen: set[str] = set()
    out: list[Path] = []
    for item in raw:
        try:
            path = Path(item).expanduser().resolve()
        except OSError:
            continue
        key = str(path)
        if key in seen or not path.is_dir():
            continue
        seen.add(key)
        out.append(path)
    return out


def _discover_source_root(user_root: Path, rels: list[str]) -> Path | None:
    user_root = user_root.resolve()

    def _all_exist(base: Path) -> bool:
        return all((base / rel).is_file() for rel in rels)

    if _all_exist(user_root):
        return user_root
    if not user_root.is_dir():
        return None
    for child in sorted(user_root.iterdir()):
        if child.is_dir() and _all_exist(child):
            return child
        if not child.is_dir():
            continue
        for sub in sorted(child.iterdir()):
            if sub.is_dir() and _all_exist(sub):
                return sub
    return None


def _source_dir_from_resolved(
    root: Path,
    rels: list[str],
    resolved_files: list[Path],
) -> Path:
    common_parts = _common_path_parts(rels)
    parents = {p.parent.resolve() for p in resolved_files}
    if len(parents) == 1:
        link_dir = next(iter(parents))
        if common_parts:
            nested = root.joinpath(*common_parts).resolve()
            if nested.is_dir():
                link_dir = nested
        return link_dir
    if common_parts:
        candidate = root.joinpath(*common_parts).resolve()
        if candidate.is_dir():
            return candidate
    return resolved_files[0].parent.resolve()


def _resolve_source_files(relative_paths: list[str]) -> tuple[Path, list[Path]]:
    """Locate on-disk files for a browser drop; return ``(root, resolved)``."""

    rels = _normalize_relpaths(relative_paths)
    if not rels:
        raise ValueError("未收到任何相对路径")

    for search_root in candidate_search_roots():
        discovered = _discover_source_root(search_root, rels)
        if discovered is None:
            continue
        root = discovered
        resolved: list[Path] = []
        try:
            for rel in rels:
                candidate = (root / rel).resolve()
                candidate.relative_to(root)
                if not candidate.is_file():
                    raise FileNotFoundError(candidate)
                resolved.append(candidate)
        except (FileNotFoundError, ValueError):
            continue
        return root, resolved

    raise FileNotFoundError("在服务器常用目录中未找到与拖入文件匹配的数据集")


def auto_resolve_source_files(relative_paths: list[str]) -> list[Path]:
    """Resolve browser drop paths to on-disk files under a common root."""

    return _resolve_source_files(relative_paths)[1]


def auto_resolve_source_dir(relative_paths: list[str]) -> Path:
    """Find the on-disk directory for a browser folder drop (no user input)."""

    rels = _normalize_relpaths(relative_paths)
    root, resolved = _resolve_source_files(rels)
    return _source_dir_from_resolved(root, rels, resolved)


def _existing_library_link_for_dir(source_dir: Path) -> Path | None:
    """Return an existing ``motions/<label>`` symlink that already points at ``source_dir``."""

    source_dir = source_dir.resolve()
    root = motions_library_root()
    if not root.is_dir():
        return None
    for child in sorted(root.iterdir()):
        if not child.is_symlink():
            continue
        try:
            if child.resolve() == source_dir:
                return child
        except OSError:
            continue
    return None


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def materialize_symlink_dir(source_dir: Path, folder_label: str | None = None) -> Path:
    """Symlink ``source_dir`` into ``~/.config/hhtools/motions/<label>/``."""

    ensure_motions_library()
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"不是目录: {source_dir}")
    label = _safe_folder_name(folder_label or source_dir.name)
    dest = motions_library_root() / label
    if dest.exists() or dest.is_symlink():
        try:
            if dest.resolve() == source_dir:
                return dest
        except OSError:
            pass
        _remove_path(dest)
    dest.symlink_to(source_dir, target_is_directory=True)
    return dest


def materialize_upload_tree(drop_dir: Path, folder_label: str | None = None) -> Path:
    """Copy an upload drop into ``~/.config/hhtools/motions/<label>/``."""

    ensure_motions_library()
    drop_dir = drop_dir.resolve()
    label = _infer_folder_label([], folder_label)
    if not label or label == _DEFAULT_LOOSE_LABEL:
        children = [p for p in drop_dir.iterdir() if p.is_dir()]
        if len(children) == 1 and not any(drop_dir.glob("*.npz")):
            label = _safe_folder_name(children[0].name)
    dest = motions_library_root() / label
    if dest.exists() or dest.is_symlink():
        _remove_path(dest)
    shutil.copytree(drop_dir, dest)
    return dest


def materialize_drop(
    relative_paths: list[str],
    *,
    folder_label: str | None = None,
    upload_drop: Path | None = None,
) -> tuple[Path, str, str]:
    """Locate or copy data into the user motions library.

    Returns ``(library_dir, folder_label, mode)`` where ``mode`` is
    ``"symlink"`` or ``"copy"``.
    """

    label = _infer_folder_label(_normalize_relpaths(relative_paths), folder_label)
    rels = _normalize_relpaths(relative_paths)

    # Single loose file: reuse an existing folder symlink when present; otherwise
    # link only that clip (not the whole parent directory).
    if len(rels) == 1 and "/" not in rels[0]:
        try:
            source_file = auto_resolve_source_files(rels)[0]
            parent = source_file.parent.resolve()
            existing = _existing_library_link_for_dir(parent)
            if existing is not None:
                return existing, existing.name, "symlink"
            clip_label = _safe_folder_name(folder_label or parent.name)
            dest_root = link_to_library(source_file, folder_label=clip_label)
            return dest_root, dest_root.name, "symlink"
        except FileNotFoundError:
            pass

    try:
        source_dir = auto_resolve_source_dir(relative_paths)
        dest = materialize_symlink_dir(source_dir, label)
        return dest, dest.name, "symlink"
    except FileNotFoundError:
        if upload_drop is None:
            raise
        dest = materialize_upload_tree(upload_drop, label)
        return dest, dest.name, "copy"


def link_to_library(path: str | Path, *, folder_label: str | None = None) -> Path:
    target = Path(path).expanduser().resolve()
    if target.is_dir():
        return materialize_symlink_dir(target, folder_label or target.name)
    if target.is_file():
        label = _safe_folder_name(folder_label or _DEFAULT_LOOSE_LABEL)
        dest_root = ensure_motions_library() / label
        dest_root.mkdir(parents=True, exist_ok=True)
        dest = dest_root / target.name
        if dest.exists() or dest.is_symlink():
            dest.unlink(missing_ok=True)
        dest.symlink_to(target)
        return dest_root
    raise FileNotFoundError(f"路径不存在: {target}")


def remove_library_folder(folder_label: str) -> bool:
    label = _safe_folder_name(folder_label)
    dest = motions_library_root() / label
    if not (dest.exists() or dest.is_symlink()):
        return False
    _remove_path(dest)
    return True


def scan_motions_library() -> list[dict[str, Any]]:
    """Scan ``~/.config/hhtools/motions`` for library entries."""

    from hhtools.web.dataset_analysis import build_entries

    root = ensure_motions_library()
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not root.is_dir():
        return entries
    for child in sorted(root.iterdir()):
        if not child.exists():
            continue
        if not (child.is_dir() or child.is_symlink()):
            continue
        folder_label = child.name
        try:
            raw_entries = build_entries(child)
        except Exception:
            continue
        for raw in raw_entries:
            entry = _entry_with_link_label(raw, folder_label, child)
            sp = entry["source_path"]
            if sp in seen:
                continue
            seen.add(sp)
            entries.append(entry)
    entries.sort(
        key=lambda x: (
            str(x.get("folder_label") or "").lower(),
            str(x.get("stem") or x.get("clip_id") or "").lower(),
        ),
    )
    return entries


def _entry_with_link_label(
    raw: dict[str, Any],
    folder_label: str,
    link_root: Path,
) -> dict[str, Any]:
    source = Path(str(raw["source_path"])).resolve()
    try:
        rel = source.relative_to(link_root.resolve())
    except ValueError:
        rel = PurePosixPath(source.name)
    stem = rel.with_suffix("").as_posix() if rel.parts else source.stem
    return {
        "dataset": raw.get("dataset", "unknown"),
        "folder_label": folder_label,
        "sequence_id": rel.as_posix() if rel.parts else source.name,
        "source_path": str(source),
        "stem": stem,
        "label": f"{folder_label} · {stem}",
        "origin": "link",
    }


__all__ = [
    "auto_resolve_source_dir",
    "auto_resolve_source_files",
    "candidate_search_roots",
    "ensure_motions_library",
    "link_to_library",
    "materialize_drop",
    "materialize_symlink_dir",
    "materialize_upload_tree",
    "motions_library_root",
    "remove_library_folder",
    "scan_motions_library",
]
