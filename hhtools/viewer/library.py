"""Folder-indexed motion library with real-time fuzzy search.

The viewer scans raw dataset directories on disk and turns every recognised file
into a clickable :class:`LibraryEntry`. The scan is **recursive** so repos can
arrange datasets however they like — flat (``assets/motions/AMASS/*.npz``),
grouped by task (``assets/motions/mimic/AMASS/*``, ``assets/motions/intermimic/
OMOMO/*``, ``assets/motions/meshmimic/parkour_1/*``), or a mix of both. Only the
*innermost* directory name — the dataset folder — affects routing, so adding a
new grouping layer never requires a code change.

A :class:`LibraryEntry` describes a single clickable clip:

- ``dataset`` — registered adapter name (``"amass"``, ``"motion_x"``, ...). Maps to
  the converter that turns the raw file into a :class:`hhtools.core.motion.Motion`.
- ``folder_label`` — the dataset directory's name as shown in the UI
  (e.g. ``"Motion-X"``), preserving the original capitalisation so the tree
  matches what users see on disk. Any parent grouping dirs (``mimic``,
  ``intermimic``, ``meshmimic``, ...) are deliberately *not* part of this label.
- ``stem`` — filename without extension, used as the primary search text.
- ``source_path`` — absolute path to the raw file.

:func:`filter_entries` is the real-time search. Queries are tokenised on whitespace;
each token is an *unanchored, case-insensitive substring* that must appear somewhere
in the entry's search key (``"<folder> / <stem>"``). This gives sensible behaviour
like ``"kick kung"`` → ``Motion-X/Aerial_Kick_Kungfu_wushu_1_clip1`` even though the
tokens appear out of order.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# Directory-name → registered adapter name. Keys normalise via ``_normalise_dirname``.
# This is the single source of truth for the folder-name ↔ adapter mapping now that
# the old ``hhtools sync`` command has been retired in favour of the viewer's lazy
# ephemeral cache — add new datasets here and they light up everywhere.
_DIR_TO_ADAPTER: dict[str, str] = {
    "amass": "amass",
    "accad": "amass",
    "cmu": "amass",
    "motionx": "motion_x",
    "phuma": "phuma",
    "lafan": "lafan",
    "lafan1": "lafan",
    "soma": "soma",
    "xsens": "xsens_mocap",
    "xsensmocap": "xsens_mocap",
    "origin_data": "xsens_mocap",
    "gvhmr": "gvhmr",
    "kungfu": "kungfu_athlete",
    "kungfuathlete": "kungfu_athlete",
    "omomo": "omomo",
    # meshmimic sub-sources: every folder under `meshmimic/<source>/<clip>/<clip>.npy`
    # registers as its own dataset bucket so UI labels stay clean
    # ("holosoma · parkour_1" etc.) and different sources can coexist with
    # their own source.yaml manifests.
    "holosoma": "meshmimic_holosoma",
    # Authored-rig folders: each clip is a self-contained .glb/.gltf with its own
    # skeleton + animation + optional skinned mesh. These light up mesh rendering
    # automatically because the adapter loads with ``with_mesh=True`` by default.
    "glb": "glb",
    "gltf": "glb",
    "parc_ms": "parc_ms",
    "parcms": "parc_ms",
    "unified_npz": "unified_npz",
    "unifiednpz": "unified_npz",
    # meshmimic/20260429_mocap — same layout as parc_ms (<clip>/<clip>.npz + terrain)
    "20260429mocap": "unified_npz",
}

# File extensions the library will try to enumerate from each dataset directory. The
# actual parsing is delegated to the adapter's ``list_sequences`` / ``load_motion``.
_SUPPORTED_EXTS = {".npz", ".npy", ".pt", ".pkl", ".bvh", ".glb", ".gltf"}


def adapter_sequence_id(source_path: str | Path, sequence_id: str) -> str:
    """Return ``sequence_id`` relative to ``source_path.parent`` for dataset adapters.

    Library scans may keep a nested ``sequence_id`` (path under a linked batch
    folder) for UI labels.  Adapters resolve ``root / sequence_id`` with
    ``root = source_path.parent``, so a nested id duplicates the clip folder
    (``clip/Take_012.bvh`` looked up as ``clip/clip/Take_012.bvh``).
    """
    path = Path(source_path).expanduser()
    seq = PurePosixPath(str(sequence_id or "").replace("\\", "/"))
    if not seq.parts:
        return path.name
    try:
        if path.is_file() and (path.parent / seq).resolve() == path.resolve():
            return path.name
    except OSError:
        pass
    if seq.name == path.name and len(seq.parts) > 1:
        return path.name
    return seq.as_posix()


@dataclass(frozen=True)
class LibraryEntry:
    """A single clip discovered on disk."""

    dataset: str
    folder_label: str
    sequence_id: str  # filename, e.g. "kick.npy"
    source_path: Path

    @property
    def stem(self) -> str:
        return Path(self.sequence_id).stem

    @property
    def adapter_sequence_id(self) -> str:
        """Filename-style id passed to ``adapter.load_motion(root=source_path.parent)``."""
        return adapter_sequence_id(self.source_path, self.sequence_id)

    @property
    def cache_name(self) -> str:
        """Name the ephemeral cache / processed-npz file should have on disk."""
        return f"{self.dataset}__{self.stem}.npz"

    @property
    def display_label(self) -> str:
        """Tree-style label shown in the UI (folder · stem)."""
        return f"{self.folder_label} · {self.stem}"

    @property
    def search_key(self) -> str:
        """Lower-cased haystack used by :func:`filter_entries`."""
        return f"{self.folder_label} {self.stem}".lower()


def _normalise_dirname(name: str) -> str:
    """Strip non-alphanumeric characters and lowercase so ``"Motion-X"`` → ``"motionx"``."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def scan_library(source_root: str | Path) -> list[LibraryEntry]:
    """Recursively walk ``source_root`` yielding one :class:`LibraryEntry` per file.

    Two-pass approach:

    1. **Locate dataset directories.** We walk ``source_root`` and collect every
       directory whose name matches an adapter alias in :data:`_DIR_TO_ADAPTER`
       (case-insensitive, non-alnum-stripped). The tree is pruned below each
       match so nested dataset dirs under the same parent chain don't double-count.
    2. **Enumerate files beneath each dataset directory, at any depth.** This is
       what lets ``intermimic/OMOMO/`` hold a per-clip subfolder layout:

       ::

           OMOMO/
           ├── sub10_largebox_000/
           │   ├── sub10_largebox_000.pkl       ← discovered
           │   └── largebox_cleaned_simplified.obj
           └── sub12_woodchair_000/
               ├── sub12_woodchair_000.pkl     ← discovered
               └── woodchair_cleaned_simplified.obj

       The clip's own ``.pkl`` / ``.npz`` / ``.npy`` file becomes a library entry;
       sibling assets (meshes, terrain ``.obj``, etc.) are ignored here but
       picked up by the dataset adapter when it loads the clip.

    The directory's name is used as ``folder_label`` regardless of how deep it
    sits in the tree (``mimic/AMASS``, ``intermimic/OMOMO``, ``meshmimic/
    parkour_1`` all route to ``AMASS`` / ``OMOMO`` / ``parkour_1`` respectively).
    Intermediate grouping layers (``mimic``, ``intermimic``, ``meshmimic`` — or
    anything else the user invents) cost nothing.

    Hidden folders (``.git``, ``__pycache__`` etc.) are pruned for speed. Symlink
    loops are avoided by asking ``os.walk`` to not follow symlinks. Results are
    deduplicated by resolved source path and sorted by ``(folder_label, stem)``
    so the UI order is deterministic.
    """
    root = Path(source_root)
    if not root.exists():
        return []

    # Pass 1: find every dataset directory. Prune the tree under each match so
    # we never double-scan nested dataset roots (e.g. a misplaced OMOMO inside
    # an AMASS dir won't get enumerated twice).
    dataset_roots: list[tuple[str, Path, str]] = []  # (adapter_name, dir_path, dir_name)
    for dirpath, dirnames, _ in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__pycache__")
        dname = Path(dirpath).name
        adapter_name = _DIR_TO_ADAPTER.get(_normalise_dirname(dname))
        if adapter_name is None:
            continue
        dataset_roots.append((adapter_name, Path(dirpath), dname))
        dirnames.clear()  # don't descend past a dataset dir

    # Pass 2: enumerate supported files under each dataset root at any depth.
    #
    # Sidecar handling: meshmimic clips ship a PARC-format ``<stem>.pkl``
    # **next to** the canonical ``<stem>.npz`` / ``<stem>.npy`` (it carries
    # the rasterised terrain heightfield only, see
    # ``scripts/build_terrain_heightfield_sidecars.py``).  Surfacing both as
    # library entries would yield duplicate ``"folder · stem"`` labels and
    # crash the Mantine dropdown.  Rule: drop a ``.pkl`` if a non-``.pkl``
    # supported file with the same stem lives in the same directory — the
    # adapter resolves the sidecar via ``path.with_suffix(".pkl")`` anyway.
    entries: list[LibraryEntry] = []
    seen_sources: set[Path] = set()
    for adapter_name, ds_root, folder_label in dataset_roots:
        for dirpath, dirnames, filenames in os.walk(ds_root, followlinks=False):
            dirnames[:] = sorted(
                d for d in dirnames if not d.startswith(".") and d != "__pycache__"
            )
            primary_stems: set[str] = {
                Path(f).stem
                for f in filenames
                if Path(f).suffix.lower() in _SUPPORTED_EXTS
                and Path(f).suffix.lower() != ".pkl"
            }
            for fname in sorted(filenames):
                suffix = Path(fname).suffix.lower()
                if suffix not in _SUPPORTED_EXTS:
                    continue
                if suffix == ".pkl" and Path(fname).stem in primary_stems:
                    # sidecar — owned by the primary clip in this dir
                    continue
                sp = (Path(dirpath) / fname).resolve()
                if sp in seen_sources:
                    continue
                seen_sources.add(sp)
                entries.append(
                    LibraryEntry(
                        dataset=adapter_name,
                        folder_label=folder_label,
                        sequence_id=fname,
                        source_path=sp,
                    )
                )
    entries.sort(key=lambda e: (e.folder_label.lower(), e.stem.lower()))
    return entries


def group_by_folder(entries: list[LibraryEntry]) -> dict[str, list[LibraryEntry]]:
    """Group entries by their ``folder_label``, preserving sort order inside each bucket."""
    buckets: dict[str, list[LibraryEntry]] = {}
    for e in entries:
        buckets.setdefault(e.folder_label, []).append(e)
    return buckets


def filter_entries(
    entries: list[LibraryEntry],
    query: str,
    *,
    folder: str | None = None,
) -> list[LibraryEntry]:
    """Return entries matching ``query`` (and optionally a folder filter).

    Matching rules:

    - Case-insensitive.
    - ``query`` is split on whitespace. Every token must appear as a *substring*
      somewhere in ``search_key``, in any order. Empty / blank query → no restriction.
    - ``folder`` (if given and not ``"All"``) restricts results to entries whose
      ``folder_label`` equals it exactly. ``"All"`` or ``None`` disables the filter.

    The search is intentionally loose so users can type partial words ("kung kick")
    and still hit the file they meant.
    """
    tokens = [t for t in query.lower().split() if t]
    filtered: list[LibraryEntry] = []
    for e in entries:
        if folder is not None and folder != "All" and e.folder_label != folder:
            continue
        hay = e.search_key
        if tokens and not all(tok in hay for tok in tokens):
            continue
        filtered.append(e)
    return filtered


def list_folders(entries: list[LibraryEntry]) -> list[str]:
    """Return folder labels in the order they should appear in the UI selector."""
    seen: list[str] = []
    for e in entries:
        if e.folder_label not in seen:
            seen.append(e.folder_label)
    return seen


__all__ = [
    "LibraryEntry",
    "adapter_sequence_id",
    "filter_entries",
    "group_by_folder",
    "list_folders",
    "scan_library",
]
