"""Persistent server-side selection of the managed Motion Library root.

The path belongs to the Python service rather than to one browser profile.  A
renderer ``localStorage`` preference would point at the wrong filesystem for a
remote Web client and would let simultaneous clients disagree about where
uploads are published.

This module owns value validation, atomic persistence, and the canonical
ownership-marker contract.  The Web server remains responsible for its
local-admin authorization boundary, for serializing a live root switch with
library publication, and for deciding when a directory may be safely adopted.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hhtools.utils.paths import (
    HHTOOLS_MOTION_LIBRARY_ROOT_ENV,
    user_motion_library_root,
)

_log = logging.getLogger(__name__)

MOTION_LIBRARY_SETTINGS_SCHEMA_VERSION = 1
MOTION_LIBRARY_MARKER_FILENAME = ".hhtools-motion-library.json"
MOTION_LIBRARY_MARKER_SCHEMA_VERSION = 1
MOTION_LIBRARY_MARKER_MANAGED_BY = "hhtools"
_MAX_MOTION_LIBRARY_MARKER_BYTES = 4 * 1024


def _normalized_root(value: object) -> Path | None:
    """Validate one JSON-compatible root and return a stable absolute path.

    ``None`` means "follow the default root" and is intentionally persisted as
    JSON ``null``.  Empty strings, relative paths, and NUL-containing values are
    rejected so their meaning cannot depend on the server's launch directory.
    Filesystem existence/writability is checked by the server at apply time;
    keeping it out of deserialization allows an offline removable/network drive
    to remain configured and be reported accurately instead of silently reset.
    """

    if value is None:
        return None
    if isinstance(value, Path):
        raw = os.fspath(value)
    elif isinstance(value, str):
        raw = value
    else:
        raise ValueError("root must be an absolute path string or null")

    raw = raw.strip()
    if not raw:
        raise ValueError("root must not be empty")
    if "\x00" in raw:
        raise ValueError("root must not contain NUL bytes")
    expanded = Path(raw).expanduser()
    if not expanded.is_absolute():
        raise ValueError("root must be an absolute path")
    # ``strict=False`` normalizes ``..`` without requiring a currently mounted
    # network/removable directory to exist.
    return expanded.resolve(strict=False)


@dataclass(frozen=True)
class MotionLibrarySettings:
    """Persisted root selection; ``None`` follows the platform default."""

    root: Path | None = None

    def as_payload(self) -> dict[str, str | None]:
        return {"root": os.fspath(self.root) if self.root is not None else None}


def validate_motion_library_settings(root: object) -> MotionLibrarySettings:
    """Return a strict settings value or raise a user-facing ``ValueError``."""

    return MotionLibrarySettings(root=_normalized_root(root))


def updated_motion_library_settings(
    current: MotionLibrarySettings,
    patch: object,
) -> MotionLibrarySettings:
    """Apply a strict JSON PATCH-shaped mapping to ``current``."""

    if not isinstance(patch, Mapping):
        raise ValueError("motion library settings must be a JSON object")
    unknown = sorted(str(key) for key in patch if key != "root")
    if unknown:
        raise ValueError(f"unknown motion library setting: {', '.join(unknown)}")
    if "root" not in patch:
        raise ValueError("the root setting is required")
    return validate_motion_library_settings(patch["root"])


def effective_motion_library_root(settings: MotionLibrarySettings) -> Path:
    """Resolve environment, persistent, and platform defaults in precedence order."""

    override = os.environ.get(HHTOOLS_MOTION_LIBRARY_ROOT_ENV)
    if override:
        root = _normalized_root(override)
        assert root is not None
        return root
    if settings.root is not None:
        return settings.root
    return user_motion_library_root().expanduser().resolve(strict=False)


def motion_library_marker_path(root: str | Path) -> Path:
    """Return the reserved ownership-marker path for a managed library root.

    Merely computing this path performs no write.  Server integration should
    only create the marker after an explicit local-user selection and should
    never treat an unmarked, non-empty arbitrary dataset directory as safe for
    recursive replacement/removal.
    """

    normalized = _normalized_root(Path(root))
    if normalized is None:  # Defensive only; ``Path`` above cannot yield None.
        raise ValueError("root is required")
    return normalized / MOTION_LIBRARY_MARKER_FILENAME


def motion_library_marker_payload() -> dict[str, int | str]:
    """Return the canonical ownership marker written into managed roots."""

    return {
        "schema_version": MOTION_LIBRARY_MARKER_SCHEMA_VERSION,
        "managed_by": MOTION_LIBRARY_MARKER_MANAGED_BY,
    }


def validate_motion_library_marker(root: str | Path) -> bool:
    """Return whether ``root`` contains our exact, regular-file marker.

    A missing marker is not itself an error because an empty directory can be
    adopted safely.  A marker-shaped entry that is present but malformed is an
    error: silently accepting it would let an arbitrary non-empty directory opt
    into recursive library replacement merely by containing a same-named file.
    Symlinks are rejected even when their target contains valid JSON so the
    ownership claim remains physically inside the selected root.
    """

    marker = motion_library_marker_path(root)
    if not (marker.exists() or marker.is_symlink()):
        return False
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("invalid Motion Library ownership marker: expected a regular file")
    try:
        if marker.stat().st_size > _MAX_MOTION_LIBRARY_MARKER_BYTES:
            raise ValueError("invalid Motion Library ownership marker: file is too large")
        payload: Any = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as err:
        raise ValueError("invalid Motion Library ownership marker") from err
    expected = motion_library_marker_payload()
    if (
        not isinstance(payload, dict)
        or set(payload) != set(expected)
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != MOTION_LIBRARY_MARKER_SCHEMA_VERSION
        or not isinstance(payload.get("managed_by"), str)
        or payload.get("managed_by") != MOTION_LIBRARY_MARKER_MANAGED_BY
    ):
        raise ValueError("invalid Motion Library ownership marker contents")
    return True


class MotionLibrarySettingsStore:
    """Atomically persist one process-wide Motion Library root selection."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve(strict=False)
        self._lock = threading.RLock()

    def load(self) -> MotionLibrarySettings:
        default = MotionLibrarySettings()
        with self._lock:
            if not self.path.is_file():
                return default
            try:
                payload: Any = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("settings root must be an object")
                version = payload.get("schema_version")
                if version != MOTION_LIBRARY_SETTINGS_SCHEMA_VERSION:
                    raise ValueError(f"unsupported settings schema version: {version!r}")
                if set(payload) != {"schema_version", "root"}:
                    unknown = sorted(
                        str(key) for key in payload if key not in {"schema_version", "root"}
                    )
                    if unknown:
                        raise ValueError(f"unknown motion library setting: {', '.join(unknown)}")
                    raise ValueError("the root setting is required")
                return validate_motion_library_settings(payload["root"])
            except (OSError, ValueError, json.JSONDecodeError):
                _log.warning(
                    "ignoring invalid Motion Library settings at %s",
                    self.path,
                    exc_info=True,
                )
                return default

    def save(self, settings: MotionLibrarySettings) -> None:
        # Revalidate even though the dataclass is typed: Python callers can
        # still construct it with an invalid runtime value.
        validated = validate_motion_library_settings(settings.root)
        payload = {
            "schema_version": MOTION_LIBRARY_SETTINGS_SCHEMA_VERSION,
            **validated.as_payload(),
        }
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.{time.time_ns()}.tmp")
            try:
                temporary.write_text(encoded + "\n", encoding="utf-8")
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)


__all__ = [
    "MOTION_LIBRARY_MARKER_FILENAME",
    "MOTION_LIBRARY_MARKER_MANAGED_BY",
    "MOTION_LIBRARY_MARKER_SCHEMA_VERSION",
    "MOTION_LIBRARY_SETTINGS_SCHEMA_VERSION",
    "MotionLibrarySettings",
    "MotionLibrarySettingsStore",
    "effective_motion_library_root",
    "motion_library_marker_path",
    "motion_library_marker_payload",
    "updated_motion_library_settings",
    "validate_motion_library_marker",
    "validate_motion_library_settings",
]
