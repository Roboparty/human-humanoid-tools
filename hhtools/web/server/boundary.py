"""Request-boundary and upload-path safeguards for the Web server."""

from __future__ import annotations

import ipaddress
import logging
from pathlib import Path, PurePosixPath, PureWindowsPath

_log = logging.getLogger(__name__)

_UPLOAD_CHUNK_BYTES = 1024 * 1024

_UPLOAD_ENDPOINTS = frozenset(
    {
        "/api/dataset/upload",
        "/api/basket/upload",
        "/api/motion/upload",
        "/api/video-to-motion/upload",
        "/api/robot/upload",
        "/api/r2r/source/upload",
        "/api/r2r/basket/upload",
    }
)


def _safe_upload_relative_path(
    filename: str | None,
    *,
    default: str = "upload.bin",
) -> Path:
    """Return a normalized browser upload path that cannot escape its drop root."""

    raw = str(filename or default).strip()
    if not raw or "\x00" in raw:
        raise ValueError("invalid upload filename")

    normalized = raw.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(normalized)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError("upload filename must be relative")
    if not posix_path.parts or any(part == ".." for part in posix_path.parts):
        raise ValueError("upload filename contains a parent-directory segment")

    relative = Path(*posix_path.parts)
    if relative.name in ("", ".", ".."):
        raise ValueError("invalid upload filename")
    return relative


def _ensure_path_within(root: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and require it to remain below ``root``."""

    resolved_root = Path(root).resolve()
    resolved_candidate = Path(candidate).resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as err:
        raise ValueError("upload path escapes its destination") from err
    return resolved_candidate


def _safe_upload_destination(root: Path, relative: Path) -> Path:
    return _ensure_path_within(root, Path(root).resolve() / relative)


def _safe_upload_directory_name(name: str | None, *, default: str) -> str:
    relative = _safe_upload_relative_path(name, default=default)
    if len(relative.parts) != 1:
        raise ValueError("upload directory name must contain one path segment")
    return relative.name


def _is_loopback_address(value: str | None, *, allow_localhost: bool = False) -> bool:
    """Recognize loopback literals without trusting arbitrary DNS resolution."""

    if value is None:
        return False
    if allow_localhost and value.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped.is_loopback
    return address.is_loopback
