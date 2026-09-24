"""Stable semantic categories for Motion Library entries.

The web UI exposes the same three concepts as the upload workflow, but library
rows can come from several places: bundled ``assets/motions`` data, linked user
directories, or a freshly uploaded basket.  Their display folder names are not
a reliable API contract, so category inference lives here on the server.

``"all"`` is intentionally not a :class:`MotionCategory`; it is a UI filter
sentinel rather than a property of an individual clip.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

type MotionCategory = Literal["motion", "object", "terrain"]

_CATEGORY_VALUES: dict[str, MotionCategory] = {
    "motion": "motion",
    "object": "object",
    "terrain": "terrain",
}

# Adapter names are more stable than user-facing dataset/folder labels.  Listing
# the known plain-motion adapters as well as scene adapters avoids probing the
# same large dataset directory once per clip.  Ambiguous/future adapters continue
# to the sidecar, profile, and path checks below.
_DATASET_CATEGORIES: dict[str, MotionCategory] = {
    "amass": "motion",
    "motionx": "motion",
    "phuma": "motion",
    "lafan": "motion",
    "lafanbvh": "motion",
    "mocap": "motion",
    "soma": "motion",
    "xsensmocap": "motion",
    "gvhmr": "motion",
    "kungfuathlete": "motion",
    "glb": "motion",
    "omomo": "object",
    "intermimic": "object",
    "parcms": "terrain",
    "meshmimic": "terrain",
    "meshmimicholosoma": "terrain",
    "holosoma": "terrain",
}

_PROFILE_CATEGORIES: dict[str, MotionCategory] = {
    "mimic": "motion",
    "intermimic": "object",
    "meshmimic": "terrain",
}

_PATH_CATEGORIES: dict[str, MotionCategory] = dict(_PROFILE_CATEGORIES)

# A library scan can contain hundreds of clips under one directory. Directory
# metadata changes when children are added, removed, or renamed, so the
# fingerprint invalidates stale results without rescanning once per clip. The
# LRU bound prevents long-lived servers from retaining every directory ever
# observed (including disconnected external libraries).
_SIDECAR_DIRECTORY_CACHE_MAXSIZE = 256


def _normalise(value: Any) -> str:
    """Normalise an adapter/profile/path token without trusting its type."""

    try:
        return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())
    except (TypeError, ValueError):
        return ""


def _entry_value(entry: object, key: str) -> Any:
    """Read ``key`` from API dictionaries and ``LibraryEntry``-like objects."""

    try:
        if isinstance(entry, Mapping):
            return entry.get(key)
        return getattr(entry, key, None)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


@lru_cache(maxsize=_SIDECAR_DIRECTORY_CACHE_MAXSIZE)
def _cached_sidecar_category(
    parent: Path,
    directory_fingerprint: tuple[int, int, int],
) -> MotionCategory | None:
    """Return the sidecar category for one version of ``parent``."""

    # The fingerprint is part of the cache key; enumeration itself only needs
    # the path. Keeping it explicit makes automatic invalidation auditable.
    del directory_fingerprint
    sibling_names = {child.name.casefold() for child in parent.iterdir()}

    # OMOMO's captured-object sidecar is more specific than a generic OBJ.
    if any(name.endswith("_cleaned_simplified.obj") for name in sibling_names):
        return "object"
    if "terrain.obj" in sibling_names or any(
        name.endswith("_terrain.obj") for name in sibling_names
    ):
        return "terrain"
    return None


def _category_from_sidecars(entry: object) -> MotionCategory | None:
    """Infer scene semantics from siblings of ``source_path``.

    Directory enumeration is deliberately best-effort.  Linked libraries may
    contain a broken symlink, live on a disconnected mount, or be unreadable to
    the current process; none of those conditions should break ``/api/library``.
    """

    raw_path = _entry_value(entry, "source_path")
    if raw_path in (None, ""):
        return None
    try:
        parent = Path(raw_path).expanduser().parent
        directory_stat = parent.stat()
        fingerprint = (
            directory_stat.st_mtime_ns,
            directory_stat.st_ctime_ns,
            directory_stat.st_size,
        )
        return _cached_sidecar_category(parent, fingerprint)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _category_from_path(entry: object) -> MotionCategory | None:
    """Use exact grouping-directory tokens as a final compatibility hint."""

    candidates = (
        _entry_value(entry, "source_path"),
        _entry_value(entry, "sequence_id"),
        _entry_value(entry, "folder_label"),
    )
    for candidate in candidates:
        try:
            # Split both separators so a persisted Windows path remains useful
            # when the same metadata is inspected on Linux, and vice versa.
            parts = re.split(r"[\\/]+", str(candidate or ""))
        except (TypeError, ValueError):
            continue
        for part in reversed(parts):
            category = _PATH_CATEGORIES.get(_normalise(part))
            if category is not None:
                return category
    return None


def infer_motion_category(entry: object) -> MotionCategory:
    """Return the semantic Motion Library category for ``entry``.

    Evidence is applied in this order:

    1. An already-normalised ``motion_category`` (idempotent API enrichment).
    2. Scene-specific dataset adapter names.
    3. Actual OMOMO/PARC/Holosoma sidecars beside the primary clip.
    4. The upload profile, when an entry still carries one.
    5. Exact ``mimic``/``intermimic``/``meshmimic`` path components.

    Unknown adapters and all failed filesystem probes safely fall back to plain
    motion.  This keeps new datasets visible while allowing their category to be
    added here once their scene semantics are known.
    """

    explicit = str(_entry_value(entry, "motion_category") or "").casefold()
    explicit_category = _CATEGORY_VALUES.get(explicit)
    if explicit_category is not None:
        return explicit_category

    dataset_category = _DATASET_CATEGORIES.get(_normalise(_entry_value(entry, "dataset")))
    if dataset_category is not None:
        return dataset_category

    sidecar_category = _category_from_sidecars(entry)
    if sidecar_category is not None:
        return sidecar_category

    for key in ("upload_profile", "profile"):
        profile_category = _PROFILE_CATEGORIES.get(_normalise(_entry_value(entry, key)))
        if profile_category is not None:
            return profile_category

    return _category_from_path(entry) or "motion"


__all__ = ["MotionCategory", "infer_motion_category"]
