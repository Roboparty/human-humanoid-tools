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
    # ``~/syj/motions``, ``~/projects/motions``, … without hard-coding usernames.
    try:
        for child in sorted(home.iterdir()):
            if child.is_dir():
                raw.append(child / "motions")
    except OSError:
        pass
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


def _clip_path_hint_tokens(folder_label: str, sequence_id: str) -> list[str]:
    """Extract path fragments used to disambiguate duplicate clip basenames."""
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in (folder_label, sequence_id):
        text = str(raw or "").strip()
        if not text:
            continue
        for part in re.split(r"[/\\·]+", text):
            part = part.strip().lower()
            if len(part) >= 5 and part not in seen:
                tokens.append(part)
                seen.add(part)
        for match in re.finditer(r"[A-Za-z]\d{4,}", text):
            token = match.group(0).lower()
            if token not in seen:
                tokens.append(token)
                seen.add(token)
    return tokens


def _score_clip_candidate(path: Path, hints: list[str], folder_label: str) -> int:
    resolved = str(path.resolve()).lower()
    score = 0
    for hint in hints:
        if hint in resolved:
            score += len(hint)
    label = str(folder_label or "").strip().lower()
    if label and label in resolved:
        score += 100
    return score


def _find_all_clips_named(
    name: str,
    search_roots: list[Path] | None = None,
) -> list[Path]:
    """Return every on-disk clip with ``name`` under known motion trees."""
    name = PurePosixPath(str(name or "").replace("\\", "/")).name
    if not name:
        return []
    roots = search_roots if search_roots is not None else candidate_search_roots()
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            matches = [p for p in root.rglob(name) if p.is_file()]
        except OSError:
            continue
        for path in matches:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(path.resolve())
    return out


def _pick_best_clip_candidate(
    candidates: list[Path],
    hints: list[str],
    folder_label: str,
) -> Path | None:
    if not candidates or not hints:
        return None
    scored = [
        (path, _score_clip_candidate(path, hints, folder_label))
        for path in candidates
    ]
    best_score = max(score for _, score in scored)
    if best_score <= 0:
        return None
    best = [path for path, score in scored if score == best_score]
    return best[0] if len(best) == 1 else None


def _recorded_path_matches_hints(
    path: Path,
    hints: list[str],
    folder_label: str,
) -> bool:
    if not hints:
        return True
    return _score_clip_candidate(path, hints, folder_label) > 0


def _resolve_single_clip_under_roots(
    rel: str,
    search_roots: list[Path] | None = None,
) -> Path | None:
    """Find one clip by relative path or basename under known motion trees."""
    rel = str(rel or "").replace("\\", "/").lstrip("/")
    if not rel:
        return None
    roots = search_roots if search_roots is not None else candidate_search_roots()
    name = PurePosixPath(rel).name
    for root in roots:
        direct = root / rel
        if direct.is_file():
            return direct.resolve()
        if rel != name:
            continue
        try:
            matches = [p for p in root.rglob(name) if p.is_file()]
        except OSError:
            continue
        if not matches:
            continue
        if len(matches) > 1:
            # Same basename in multiple capture folders (e.g. two ``Take_012_Skeleton0.bvh``
            # under ``20260429_mocap`` vs ``20260623_mocap``) — never pick arbitrarily.
            continue
        return matches[0].resolve()
    return None


def resolve_clip_on_disk(
    source_path: str | Path,
    *,
    extra_names: list[str] | None = None,
    folder_label: str | None = None,
    sequence_id: str | None = None,
    upload_drop: str | Path | None = None,
) -> Path:
    """Return an existing clip path, searching server motion trees when stale.

    Batch baskets often store ``~/.config/hhtools/motions/<label>/<clip>.bvh``
    even when only one clip was copied during a multi-file browser drop.  When
    the recorded path is missing, locate the same basename under
    :func:`candidate_search_roots` (``~/syj/motions``, ``HHTOOLS_MOTION_SEARCH_PATHS``, …).

    When the recorded path exists but points at the wrong capture folder (stale
    library symlink for a duplicate basename), ``folder_label`` / ``sequence_id``
    are used to pick the intended clip instead of trusting the symlink target.
    """
    recorded = Path(source_path).expanduser()
    folder_label = str(folder_label or "").strip()
    sequence_id = str(sequence_id or "").strip()

    names: list[str] = []
    if recorded.name:
        names.append(recorded.name)
    for raw in extra_names or ():
        n = PurePosixPath(str(raw).replace("\\", "/")).name
        if n and n not in names:
            names.append(n)
    if sequence_id:
        sid_name = PurePosixPath(sequence_id.replace("\\", "/")).name
        if sid_name and sid_name not in names:
            names.append(sid_name)

    if upload_drop is not None:
        drop = Path(upload_drop).expanduser()
        if sequence_id:
            rel_upload = drop / sequence_id.replace("\\", "/")
            if rel_upload.is_file():
                return rel_upload.resolve()
        for name in names:
            uploaded = _uploaded_path_for_rel(drop, name)
            if uploaded.is_file():
                return uploaded.resolve()

    hints = _clip_path_hint_tokens(folder_label, sequence_id)
    candidates: list[Path] = []
    seen_candidates: set[str] = set()
    for name in names:
        for path in _find_all_clips_named(name):
            key = str(path)
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            candidates.append(path)

    recorded_resolved: Path | None = None
    try:
        if recorded.is_file():
            recorded_resolved = recorded.resolve()
            key = str(recorded_resolved)
            if key not in seen_candidates:
                candidates.append(recorded_resolved)
                seen_candidates.add(key)
    except OSError:
        pass

    if recorded_resolved is not None and len(candidates) <= 1:
        return recorded_resolved

    if recorded_resolved is not None and _recorded_path_matches_hints(
        recorded_resolved, hints, folder_label,
    ):
        scores = {
            path: _score_clip_candidate(path, hints, folder_label)
            for path in candidates
        }
        best_score = max(scores.values(), default=0)
        if best_score > 0 and scores.get(recorded_resolved, 0) == best_score:
            return recorded_resolved

    picked = _pick_best_clip_candidate(candidates, hints, folder_label)
    if picked is not None:
        return picked

    if recorded_resolved is not None:
        return recorded_resolved

    for name in names:
        found = _resolve_single_clip_under_roots(name)
        if found is not None:
            return found
    raise FileNotFoundError(f"BVH sequence not found: {recorded}")


def library_entry_for_load(
    *,
    dataset: str,
    folder_label: str,
    sequence_id: str,
    source_path: str | Path,
    upload_drop: str | Path | None = None,
) -> "LibraryEntry":
    """Resolve a basket/library row to a load-safe :class:`LibraryEntry`."""
    from hhtools.viewer.library import LibraryEntry

    resolved = resolve_clip_on_disk(
        source_path,
        extra_names=[sequence_id or ""],
        folder_label=folder_label,
        sequence_id=sequence_id,
        upload_drop=upload_drop,
    )
    return LibraryEntry(
        dataset=dataset,
        folder_label=folder_label,
        sequence_id=sequence_id,
        source_path=resolved,
    )


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
                candidate = root / rel
                if not candidate.is_file():
                    raise FileNotFoundError(candidate)
                # Do not require ``resolve()`` to stay under ``root`` — library
                # folders are often symlinks to the user's motion tree.
                resolved.append(candidate.resolve())
        except (FileNotFoundError, ValueError):
            continue
        return root, resolved

    # Loose clips scattered in different subfolders (e.g. ``20260429_mocap/*/*.bvh``).
    per_file: list[Path] = []
    for rel in rels:
        found = _resolve_single_clip_under_roots(rel)
        if found is None:
            per_file = []
            break
        per_file.append(found)
    if per_file:
        return _source_dir_from_resolved(per_file[0].parent, rels, per_file), per_file

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


def _upload_tree_root(drop_dir: Path) -> Path:
    """Unwrap a single top-level wrapper folder from browser upload drops."""
    children = sorted(
        p for p in drop_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    files = [
        p for p in drop_dir.iterdir()
        if p.is_file() and not p.name.startswith(".")
    ]
    if len(children) == 1 and not files:
        return children[0]
    return drop_dir


def materialize_upload_tree(drop_dir: Path, folder_label: str | None = None) -> Path:
    """Copy an upload drop into ``~/.config/hhtools/motions/<label>/``."""

    ensure_motions_library()
    drop_dir = drop_dir.resolve()
    tree = _upload_tree_root(drop_dir)
    label = _infer_folder_label([], folder_label)
    if not label or label == _DEFAULT_LOOSE_LABEL:
        if tree == drop_dir:
            children = [p for p in drop_dir.iterdir() if p.is_dir()]
            if len(children) == 1 and not any(drop_dir.glob("*.npz")):
                label = _safe_folder_name(children[0].name)
        else:
            label = _safe_folder_name(tree.name)
    dest = motions_library_root() / label
    if dest.exists() or dest.is_symlink():
        _remove_path(dest)
    # Avoid ``motions/<label>/<label>/…`` when the drop wraps one folder.
    if _safe_folder_name(tree.name) == label:
        shutil.copytree(tree, dest)
    else:
        shutil.copytree(drop_dir, dest)
    return dest


def _uploaded_path_for_rel(upload_drop: Path, rel: str) -> Path:
    rel = str(rel or "").replace("\\", "/").lstrip("/")
    direct = upload_drop / rel
    if direct.is_file():
        return direct
    return upload_drop / PurePosixPath(rel).name


def _resolved_matches_upload(resolved: Path, upload_drop: Path, rel: str) -> bool:
    """True when an on-disk auto-resolve hit is the same bytes as the browser upload."""
    uploaded = _uploaded_path_for_rel(upload_drop, rel)
    if not uploaded.is_file() or not resolved.is_file():
        return False
    try:
        return uploaded.stat().st_size == resolved.stat().st_size
    except OSError:
        return False


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

    # Multiple loose files (often from different capture folders): link each clip
    # into the library instead of requiring a single shared parent directory.
    if rels and all("/" not in r and "\\" not in r for r in rels) and len(rels) > 1:
        try:
            resolved = auto_resolve_source_files(rels)
            dest_root = ensure_motions_library() / label
            dest_root.mkdir(parents=True, exist_ok=True)
            for rel, src in zip(rels, resolved, strict=True):
                dest = dest_root / PurePosixPath(rel).name
                try:
                    if dest.exists() and dest.resolve() == src.resolve():
                        continue
                except OSError:
                    pass
                if dest.exists() or dest.is_symlink():
                    dest.unlink(missing_ok=True)
                dest.symlink_to(src)
            return dest_root, dest_root.name, "symlink"
        except FileNotFoundError:
            pass

    # Single loose file: reuse an existing folder symlink when present; otherwise
    # link only that clip (not the whole parent directory).
    if len(rels) == 1 and "/" not in rels[0]:
        try:
            source_file = auto_resolve_source_files(rels)[0].resolve()
            if upload_drop is not None and not _resolved_matches_upload(
                source_file, upload_drop, rels[0],
            ):
                raise FileNotFoundError("auto-resolved file does not match upload")
            lib_root = motions_library_root().resolve()
            try:
                source_file.relative_to(lib_root)
                parent = source_file.parent
                return parent, parent.name, "symlink"
            except ValueError:
                pass
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
        if upload_drop is not None:
            resolved_files = auto_resolve_source_files(rels)
            for rel, src in zip(rels, resolved_files, strict=True):
                if not _resolved_matches_upload(src, upload_drop, rel):
                    raise FileNotFoundError("auto-resolved file does not match upload")
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
        # Already materialised (upload copy) — never replace with a self-symlink.
        try:
            if dest.exists() and dest.resolve() == target.resolve():
                return dest_root
        except OSError:
            pass
        try:
            target.relative_to(dest_root.resolve())
            return dest_root
        except ValueError:
            pass
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
    "library_entry_for_load",
    "link_to_library",
    "materialize_drop",
    "materialize_symlink_dir",
    "materialize_upload_tree",
    "motions_library_root",
    "remove_library_folder",
    "resolve_clip_on_disk",
    "scan_motions_library",
]
