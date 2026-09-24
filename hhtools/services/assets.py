"""Persistent, content-addressed asset registration for Agent clients.

The registry is deliberately a control-plane service.  Public contracts only
contain a configured ``root_id`` and portable paths; absolute host paths are
resolved again at the service boundary and are never written to SQLite or
returned in an :class:`~hhtools.contracts.AssetBundle`.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from hhtools.contracts import (
    ApiError,
    AssetBundle,
    AssetCategory,
    AssetDetected,
    AssetFile,
    AssetFileRole,
    AssetKind,
    AssetRegistrationRequest,
    AssetSearchResponse,
    AssetSource,
    AssetSourceScheme,
    ErrorStage,
)

_HASH_CHUNK_SIZE = 1024 * 1024
_DEFAULT_SEARCH_LIMIT = 100
_MAX_SEARCH_LIMIT = 500

RootProvider = Path | Callable[[], Path]
AssetDiscoverer = Callable[[Path], "DiscoveredAsset"]


class AssetServiceError(RuntimeError):
    """Expected asset-service failure with a transport-neutral error body."""

    def __init__(self, error: ApiError) -> None:
        self.error = error
        super().__init__(f"{error.code}: {error.message}")

    @property
    def api_error(self) -> ApiError:
        """Alias used by protocol adapters that expose an API error payload."""

        return self.error

    @property
    def code(self) -> str:
        """Stable machine code without requiring callers to inspect text."""

        return self.error.code


@dataclass(frozen=True, slots=True)
class DiscoveredAssetFile:
    """One trusted inspector result that still needs root-bound validation."""

    path: Path
    role: AssetFileRole = AssetFileRole.OTHER
    required: bool = True
    media_type: str | None = None


@dataclass(frozen=True, slots=True)
class DiscoveredAsset:
    """Inspector output consumed by :meth:`AssetRegistry.register`.

    Paths are an internal hand-off only.  The registry resolves every path,
    proves it remains below the configured root, and converts it to a portable
    relative path before constructing or persisting a public contract.
    """

    primary_file: Path
    files: Sequence[DiscoveredAssetFile]
    kind: AssetKind = AssetKind.MOTION_BUNDLE
    category: AssetCategory = AssetCategory.PLAIN_MOTION
    display_name: str | None = None
    detected: AssetDetected | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _asset_error(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
) -> AssetServiceError:
    return AssetServiceError(
        ApiError(
            code=code,
            message=message,
            retryable=retryable,
            stage=ErrorStage.ASSET_REGISTRATION,
            details=dict(details or {}),
        )
    )


def _portable_relative_path(value: str) -> PurePosixPath:
    """Defensively validate a root-relative path, even for constructed models."""

    if not value or "\\" in value:
        raise _asset_error(
            "ASSET_OUTSIDE_ALLOWED_ROOT",
            "Asset paths must be normalized relative paths below an allowed root.",
        )
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise _asset_error(
            "ASSET_OUTSIDE_ALLOWED_ROOT",
            "Asset paths must be normalized relative paths below an allowed root.",
        )
    return posix


def _looks_like_absolute_path(value: str) -> bool:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return posix.is_absolute() or windows.is_absolute() or bool(windows.drive)


def _portable_json(value: Any, *, field_name: str) -> Any:
    """Copy JSON metadata while rejecting host paths and non-JSON objects."""

    def validate(item: Any) -> None:
        if item is None or isinstance(item, bool | int | float):
            return
        if isinstance(item, str):
            if _looks_like_absolute_path(item):
                raise _asset_error(
                    "ASSET_OUTSIDE_ALLOWED_ROOT",
                    f"{field_name} cannot contain an absolute host path.",
                )
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise _asset_error(
                        "INVALID_PARAMETER",
                        f"{field_name} object keys must be strings.",
                    )
                if _looks_like_absolute_path(key):
                    raise _asset_error(
                        "ASSET_OUTSIDE_ALLOWED_ROOT",
                        f"{field_name} cannot contain an absolute host path.",
                    )
                validate(child)
            return
        if isinstance(item, list | tuple):
            for child in item:
                validate(child)
            return
        raise _asset_error(
            "INVALID_PARAMETER",
            f"{field_name} must contain JSON-compatible values.",
        )

    validate(value)
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise _asset_error(
            "INVALID_PARAMETER",
            f"{field_name} must contain finite JSON-compatible values.",
        ) from exc
    return json.loads(encoded)


def _sha256_file(path: Path) -> tuple[str, int]:
    """Hash one stable file snapshot and reject concurrent modification."""

    try:
        before = path.stat()
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
                size += len(chunk)
        after = path.stat()
    except (OSError, PermissionError) as exc:
        raise _asset_error(
            "ASSET_NOT_FOUND",
            "An asset file is missing or unreadable.",
            retryable=True,
        ) from exc

    snapshot_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    snapshot_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if snapshot_before != snapshot_after or size != after.st_size:
        raise _asset_error(
            "ASSET_HASH_MISMATCH",
            "An asset file changed while it was being registered.",
            retryable=True,
        )
    return digest.hexdigest(), size


def _infer_role(path: Path) -> AssetFileRole:
    suffix = path.suffix.lower()
    if suffix in {".urdf", ".mjcf", ".xml"}:
        return AssetFileRole.ROBOT_DESCRIPTION
    if suffix in {".mp4", ".mov", ".avi", ".webm", ".mkv"}:
        return AssetFileRole.VIDEO
    if suffix in {
        ".bvh",
        ".csv",
        ".glb",
        ".gltf",
        ".npy",
        ".npz",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
    }:
        return AssetFileRole.MOTION
    return AssetFileRole.OTHER


def _raise_walk_error(error: OSError) -> None:
    raise error


def _default_discovery(candidate: Path, *, recursive: bool) -> DiscoveredAsset:
    if candidate.is_file():
        paths = [candidate]
    elif candidate.is_dir():
        try:
            if recursive:
                paths = []
                for current, directory_names, file_names in os.walk(
                    candidate,
                    topdown=True,
                    onerror=_raise_walk_error,
                    followlinks=False,
                ):
                    directory = Path(current)
                    paths.extend(
                        directory / name
                        for name in directory_names
                        if (directory / name).is_symlink()
                    )
                    paths.extend(directory / name for name in file_names)
            else:
                paths = list(candidate.iterdir())
            paths.sort(key=lambda item: item.as_posix().casefold())
        except (OSError, RuntimeError) as exc:
            raise _asset_error(
                "ASSET_NOT_FOUND",
                "The asset directory could not be read.",
                retryable=True,
            ) from exc
    else:
        raise _asset_error("ASSET_NOT_FOUND", "The requested asset does not exist.")

    # Directories and symlinks are retained temporarily so register() can
    # validate their resolved locations before filtering to regular files.
    discovered = [
        DiscoveredAssetFile(path=path, role=_infer_role(path))
        for path in paths
        if path.is_file() or path.is_symlink()
    ]
    if not discovered:
        raise _asset_error(
            "BUNDLE_INCOMPLETE",
            "The requested asset bundle contains no regular files.",
        )

    regular = [item for item in discovered if item.path.is_file()]
    primary = (
        next(
            (item for item in regular if item.role is AssetFileRole.MOTION),
            regular[0],
        )
        if regular
        else discovered[0]
    )
    role = primary.role
    kind = AssetKind.MOTION_BUNDLE
    category = AssetCategory.PLAIN_MOTION
    if role is AssetFileRole.ROBOT_DESCRIPTION:
        kind = AssetKind.ROBOT_BUNDLE
        category = AssetCategory.ROBOT_MODEL
    elif role is AssetFileRole.VIDEO:
        kind = AssetKind.VIDEO
    return DiscoveredAsset(
        primary_file=primary.path,
        files=discovered,
        kind=kind,
        category=category,
    )


def _identity_digest(
    *,
    kind: AssetKind,
    category: AssetCategory,
    primary_file: str,
    files: Sequence[AssetFile],
    detected: AssetDetected | None,
) -> str:
    """Hash immutable content and routing semantics; labels/source are excluded."""

    payload = {
        "schema_version": "1.0",
        "kind": kind.value,
        "category": category.value,
        "primary_file": primary_file,
        "detected": detected.model_dump(mode="json") if detected is not None else None,
        "files": [
            {
                "relative_path": item.relative_path,
                "role": item.role.value,
                "required": item.required,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in sorted(files, key=lambda value: (value.relative_path, value.role.value))
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class AssetRegistry:
    """SQLite-backed registry for immutable asset manifests.

    ``roots`` is deployment configuration, not request data.  A callable root
    provider is resolved on every operation so a WebUI setting can change
    without persisting an absolute machine path in the registry database.
    """

    def __init__(
        self,
        data_dir: Path,
        roots: Mapping[str, RootProvider],
        *,
        discoverer: AssetDiscoverer | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._database_path = self._data_dir / "assets.sqlite3"
        self._artifact_root = self._data_dir / "artifacts"
        self._roots = dict(roots)
        if any(
            not root_id
            or root_id in {".", ".."}
            or "/" in root_id
            or "\\" in root_id
            or _looks_like_absolute_path(root_id)
            for root_id in self._roots
        ):
            raise _asset_error(
                "INVALID_PARAMETER",
                "Asset root identifiers must be portable names, not paths.",
            )
        self._discoverer = discoverer
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @property
    def artifact_root(self) -> Path:
        """Internal local boundary reserved for the later ArtifactStore."""

        return self._artifact_root

    @property
    def allowed_root_ids(self) -> tuple[str, ...]:
        """Return safe configuration identifiers, never their host locations."""

        return tuple(sorted(self._roots))

    def registration_hint(
        self,
        trusted_path: Path,
        *,
        kind: AssetKind | None = None,
        category: AssetCategory | None = None,
        recursive: bool = True,
        preferred_root_id: str | None = None,
    ) -> AssetRegistrationRequest:
        """Convert one trusted local path to a portable registration request.

        This is an in-process service boundary, not a public path resolver.  It
        accepts a path already selected by trusted application code, proves the
        path is below a configured root, and returns only ``root_id`` plus a
        normalized relative path.  When roots overlap, the deepest usable root
        wins.  Equally specific root identifiers are rejected instead of being
        selected by an arbitrary ordering.
        """

        try:
            resolved = Path(trusted_path).resolve(strict=True)
            if not resolved.is_file() and not resolved.is_dir():
                raise OSError("trusted asset source is not a regular file or directory")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise _asset_error(
                "ASSET_NOT_FOUND",
                "The trusted asset source is unavailable.",
            ) from exc

        candidates: list[tuple[int, str, Path]] = []
        for root_id in sorted(self._roots):
            # Fail closed when any configured provider cannot be resolved.  A
            # silent fallback to a broader root could change the portable
            # identity selected for the same installed preset.
            root = self._root(root_id)
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            candidates.append((len(root.parts), root_id, relative))

        if not candidates:
            raise _asset_error(
                "ASSET_OUTSIDE_ALLOWED_ROOT",
                "The trusted asset source is not addressable below an allowed root.",
            )

        specificity = max(depth for depth, _root_id, _relative in candidates)
        selected = [candidate for candidate in candidates if candidate[0] == specificity]
        if len(selected) != 1 and preferred_root_id is not None:
            preferred = [item for item in selected if item[1] == preferred_root_id]
            if len(preferred) == 1:
                selected = preferred
        if len(selected) != 1:
            raise _asset_error(
                "ASSET_ROOT_AMBIGUOUS",
                "Multiple equally specific allowed roots identify the trusted asset source.",
                details={"root_ids": sorted(root_id for _depth, root_id, _relative in selected)},
            )

        _depth, root_id, relative = selected[0]
        if not relative.parts:
            raise _asset_error(
                "ASSET_ROOT_UNREPRESENTABLE",
                "The most specific allowed root cannot name itself as a portable relative path.",
                details={"root_id": root_id},
            )
        relative_path = relative.as_posix()
        try:
            return AssetRegistrationRequest(
                root_id=root_id,
                relative_path=relative_path,
                display_name=None,
                kind=kind,
                category=category,
                recursive=recursive,
            )
        except (TypeError, ValueError) as exc:
            raise _asset_error(
                "INVALID_PARAMETER",
                "The trusted asset source cannot be represented by the registration contract.",
            ) from exc

    def canonical_root_aliases(
        self,
        preferred_root_ids: tuple[str, ...] = (),
    ) -> dict[str, str]:
        """Collapse roots resolving to the same directory without exposing that directory."""

        priority = {root_id: index for index, root_id in enumerate(preferred_root_ids)}
        grouped: dict[Path, list[str]] = {}
        for root_id in sorted(self._roots):
            grouped.setdefault(self._root(root_id), []).append(root_id)
        aliases: dict[str, str] = {}
        for root_ids in grouped.values():
            canonical = min(
                root_ids,
                key=lambda root_id: (priority.get(root_id, len(priority)), root_id),
            )
            aliases.update({root_id: canonical for root_id in root_ids})
        return aliases

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS assets (
                        asset_id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        category TEXT NOT NULL,
                        display_name TEXT NOT NULL,
                        root_id TEXT NOT NULL,
                        logical_path TEXT NOT NULL,
                        dataset TEXT,
                        reference_model TEXT,
                        recommended_backend TEXT,
                        registered_at TEXT NOT NULL,
                        manifest_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute("CREATE INDEX IF NOT EXISTS assets_kind_idx ON assets(kind)")
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS assets_category_idx ON assets(category)"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS assets_dataset_idx ON assets(dataset)"
                )
        except sqlite3.Error as exc:
            raise _asset_error(
                "INTERNAL_ERROR",
                "The asset registry database could not be initialized.",
                retryable=True,
            ) from exc

    def _root(self, root_id: str) -> Path:
        provider = self._roots.get(root_id)
        if provider is None:
            details = (
                {"root_id": root_id}
                if root_id
                and root_id not in {".", ".."}
                and "/" not in root_id
                and "\\" not in root_id
                and not _looks_like_absolute_path(root_id)
                else {}
            )
            raise _asset_error(
                "ASSET_OUTSIDE_ALLOWED_ROOT",
                "The requested asset root is not allowed.",
                details=details,
            )
        try:
            configured = provider() if callable(provider) else provider
            root = Path(configured).resolve(strict=True)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise _asset_error(
                "ASSET_NOT_FOUND",
                "The configured asset root is unavailable.",
                retryable=True,
                details={"root_id": root_id},
            ) from exc
        if not root.is_dir():
            raise _asset_error(
                "ASSET_NOT_FOUND",
                "The configured asset root is not a directory.",
                details={"root_id": root_id},
            )
        return root

    @staticmethod
    def _contain(root: Path, path: Path, *, root_id: str) -> Path:
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise _asset_error(
                "ASSET_OUTSIDE_ALLOWED_ROOT",
                "An asset path resolves outside its allowed root.",
                details={"root_id": root_id},
            ) from exc
        return resolved

    def _registration_path(self, root: Path, request: AssetRegistrationRequest) -> Path:
        logical = _portable_relative_path(request.relative_path)
        candidate = root.joinpath(*logical.parts)
        try:
            return self._contain(root, candidate, root_id=request.root_id)
        except AssetServiceError as exc:
            if not candidate.exists() and not candidate.is_symlink():
                raise _asset_error(
                    "ASSET_NOT_FOUND",
                    "The requested asset does not exist below the configured root.",
                    details={
                        "root_id": request.root_id,
                        "logical_path": request.relative_path,
                    },
                ) from exc
            raise

    def resolve_registration_path(self, request: AssetRegistrationRequest) -> Path:
        """Resolve an input for a trusted inspector without exposing it publicly."""

        root = self._root(request.root_id)
        return self._registration_path(root, request)

    def _resolve_discovered_path(self, root: Path, root_id: str, value: Path) -> Path:
        candidate = value if value.is_absolute() else root / value
        return self._contain(root, candidate, root_id=root_id)

    @staticmethod
    def _relative_to_bundle(path: Path, bundle_base: Path) -> str:
        """Return a bundle-relative path after enforcing the bundle boundary."""

        try:
            return path.relative_to(bundle_base).as_posix()
        except ValueError as exc:
            raise _asset_error(
                "BUNDLE_INCOMPLETE",
                "A discovered file is outside the requested asset bundle.",
            ) from exc

    def register(
        self,
        request: AssetRegistrationRequest,
        *,
        discovery: DiscoveredAsset | None = None,
    ) -> AssetBundle:
        """Register a safe bundle and return its stable content identity."""

        root = self._root(request.root_id)
        candidate = self._registration_path(root, request)
        bundle_base = candidate if candidate.is_dir() else candidate.parent
        if discovery is None:
            if self._discoverer is not None:
                try:
                    discovery = self._discoverer(candidate)
                except AssetServiceError:
                    raise
                except Exception as exc:
                    raise _asset_error(
                        "BUNDLE_INCOMPLETE",
                        "Asset bundle discovery failed.",
                    ) from exc
            else:
                discovery = _default_discovery(candidate, recursive=request.recursive)

        metadata = _portable_json(dict(discovery.metadata), field_name="Asset metadata")
        detected = discovery.detected
        if detected is not None:
            detected = AssetDetected.model_validate(
                _portable_json(
                    detected.model_dump(mode="json"),
                    field_name="Detected asset metadata",
                )
            )
        display_name = request.display_name or discovery.display_name or candidate.stem
        if _looks_like_absolute_path(display_name):
            raise _asset_error(
                "ASSET_OUTSIDE_ALLOWED_ROOT",
                "Asset display names cannot contain an absolute host path.",
            )

        resolved_files: dict[str, tuple[Path, DiscoveredAssetFile]] = {}
        for item in discovery.files:
            resolved = self._resolve_discovered_path(root, request.root_id, Path(item.path))
            if not resolved.is_file():
                # A directory symlink is still containment-checked above, but
                # directories are not part of the immutable file manifest.
                continue
            relative = self._relative_to_bundle(resolved, bundle_base)
            if relative in resolved_files:
                previous = resolved_files[relative][1]
                if previous.role is not item.role or previous.required != item.required:
                    raise _asset_error(
                        "BUNDLE_INCOMPLETE",
                        "Asset discovery assigned conflicting roles to one file.",
                        details={"relative_path": relative},
                    )
                continue
            resolved_files[relative] = (resolved, item)

        if not resolved_files:
            raise _asset_error(
                "BUNDLE_INCOMPLETE",
                "The requested asset bundle contains no regular files.",
            )

        primary_path = self._resolve_discovered_path(
            root,
            request.root_id,
            Path(discovery.primary_file),
        )
        if not primary_path.is_file():
            raise _asset_error(
                "BUNDLE_INCOMPLETE",
                "The asset primary file is not a regular file.",
            )
        primary_relative = self._relative_to_bundle(primary_path, bundle_base)
        if primary_relative not in resolved_files:
            raise _asset_error(
                "BUNDLE_INCOMPLETE",
                "The asset primary file is missing from the discovered manifest.",
                details={"relative_path": primary_relative},
            )

        files: list[AssetFile] = []
        for relative, (path, item) in sorted(resolved_files.items()):
            digest, size = _sha256_file(path)
            media_type = item.media_type or mimetypes.guess_type(path.name)[0]
            files.append(
                AssetFile(
                    role=item.role,
                    relative_path=relative,
                    sha256=digest,
                    size_bytes=size,
                    media_type=media_type,
                    required=item.required,
                )
            )

        kind = request.kind or discovery.kind
        category = request.category or discovery.category
        digest = _identity_digest(
            kind=kind,
            category=category,
            primary_file=primary_relative,
            files=files,
            detected=detected,
        )
        registered_at = datetime.now(UTC)
        bundle = AssetBundle(
            asset_id=f"asset:sha256:{digest}",
            kind=kind,
            category=category,
            display_name=display_name,
            primary_file=primary_relative,
            files=files,
            source=AssetSource(
                scheme=AssetSourceScheme.MANAGED_FILE,
                root_id=request.root_id,
                registered_at=registered_at,
                logical_path=request.relative_path,
            ),
            detected=detected,
            metadata=metadata,
        )
        return self._persist(bundle)

    def _persist(self, bundle: AssetBundle) -> AssetBundle:
        source = bundle.source
        if source is None:
            raise _asset_error("INTERNAL_ERROR", "A registered asset must have a source.")
        detected = bundle.detected
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO assets (
                        asset_id, kind, category, display_name, root_id,
                        logical_path, dataset, reference_model,
                        recommended_backend, registered_at, manifest_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        bundle.asset_id,
                        bundle.kind.value,
                        bundle.category.value,
                        bundle.display_name,
                        source.root_id,
                        source.logical_path or bundle.primary_file,
                        detected.dataset if detected else None,
                        detected.reference if detected else None,
                        detected.recommended_backend if detected else None,
                        source.registered_at.isoformat(),
                        bundle.model_dump_json(),
                    ),
                )
                row = connection.execute(
                    "SELECT manifest_json FROM assets WHERE asset_id = ?",
                    (bundle.asset_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise _asset_error(
                "INTERNAL_ERROR",
                "The asset manifest could not be persisted.",
                retryable=True,
            ) from exc
        if row is None:
            raise _asset_error(
                "INTERNAL_ERROR",
                "The asset manifest was not available after registration.",
                retryable=True,
            )
        return self._decode_bundle(row["manifest_json"])

    @staticmethod
    def _decode_bundle(payload: str) -> AssetBundle:
        try:
            bundle = AssetBundle.model_validate_json(payload)
            if _looks_like_absolute_path(bundle.display_name):
                raise ValueError("absolute display name")
            _portable_json(bundle.metadata, field_name="Asset metadata")
            if bundle.detected is not None:
                _portable_json(
                    bundle.detected.model_dump(mode="json"),
                    field_name="Detected asset metadata",
                )
            source = bundle.source
            if source is not None and (
                not source.root_id
                or source.root_id in {".", ".."}
                or "/" in source.root_id
                or "\\" in source.root_id
                or _looks_like_absolute_path(source.root_id)
            ):
                raise ValueError("path-like root id")
            return bundle
        except (AssetServiceError, TypeError, ValueError) as exc:
            raise _asset_error(
                "INTERNAL_ERROR",
                "A persisted asset manifest is invalid.",
            ) from exc

    def get(self, asset_id: str) -> AssetBundle:
        """Return one portable manifest by stable id."""

        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT manifest_json FROM assets WHERE asset_id = ?",
                    (asset_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise _asset_error(
                "INTERNAL_ERROR",
                "The asset registry could not be read.",
                retryable=True,
            ) from exc
        if row is None:
            raise _asset_error(
                "ASSET_NOT_FOUND",
                "No registered asset has the requested id.",
                details={"asset_id": asset_id},
            )
        return self._decode_bundle(row["manifest_json"])

    def search(
        self,
        *,
        query: str | None = None,
        kind: AssetKind | str | None = None,
        category: AssetCategory | str | None = None,
        dataset: str | None = None,
        reference: str | None = None,
        limit: int = _DEFAULT_SEARCH_LIMIT,
        offset: int = 0,
    ) -> AssetSearchResponse:
        """Search compact manifest metadata with deterministic pagination."""

        if not 1 <= limit <= _MAX_SEARCH_LIMIT or offset < 0:
            raise _asset_error(
                "INVALID_PARAMETER",
                "Asset search requires limit 1..500 and a non-negative offset.",
            )
        try:
            kind_value = AssetKind(kind).value if kind is not None else None
            category_value = AssetCategory(category).value if category is not None else None
        except ValueError as exc:
            raise _asset_error(
                "INVALID_PARAMETER",
                "Asset search contains an unsupported kind or category.",
            ) from exc

        clauses: list[str] = []
        parameters: list[Any] = []
        if query:
            escaped = query.casefold().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append(
                "(LOWER(display_name) LIKE ? ESCAPE '\\' OR LOWER(logical_path) LIKE ? ESCAPE '\\')"
            )
            parameters.extend((f"%{escaped}%", f"%{escaped}%"))
        for column, value in (
            ("kind", kind_value),
            ("category", category_value),
            ("dataset", dataset),
            ("reference_model", reference),
        ):
            if value is not None:
                clauses.append(f"LOWER({column}) = ?")
                parameters.append(str(value).casefold())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        try:
            with self._connect() as connection:
                total_row = connection.execute(
                    f"SELECT COUNT(*) AS count FROM assets{where}",  # noqa: S608
                    parameters,
                ).fetchone()
                rows = connection.execute(
                    f"""
                    SELECT manifest_json FROM assets{where}
                    ORDER BY registered_at DESC, asset_id ASC
                    LIMIT ? OFFSET ?
                    """,  # noqa: S608
                    [*parameters, limit, offset],
                ).fetchall()
        except sqlite3.Error as exc:
            raise _asset_error(
                "INTERNAL_ERROR",
                "The asset registry search failed.",
                retryable=True,
            ) from exc
        total = int(total_row["count"]) if total_row is not None else 0
        return AssetSearchResponse(
            assets=[self._decode_bundle(row["manifest_json"]) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
        )

    def resolve_file(
        self,
        asset_id: str,
        relative_path: str | None = None,
        *,
        verify_hash: bool = True,
    ) -> Path:
        """Resolve a declared file for trusted services and optionally verify it.

        This method is not an Agent response.  It repeats root and symlink
        containment checks on every call, so a changed linked directory cannot
        silently expand the registry's filesystem authority.
        """

        bundle = self.get(asset_id)
        source = bundle.source
        if source is None:
            raise _asset_error("INTERNAL_ERROR", "The asset has no registered source.")
        selected = relative_path or bundle.primary_file
        declared = {item.relative_path: item for item in bundle.files}
        asset_file = declared.get(selected)
        if asset_file is None:
            raise _asset_error(
                "ASSET_NOT_FOUND",
                "The requested file is not declared by this asset bundle.",
                details={"asset_id": asset_id, "relative_path": selected},
            )
        logical = _portable_relative_path(selected)
        root = self._root(source.root_id)
        if source.logical_path is None:
            raise _asset_error("INTERNAL_ERROR", "The asset source has no logical path.")
        source_path = _portable_relative_path(source.logical_path)
        unresolved_candidate = root.joinpath(*source_path.parts)
        try:
            candidate = self._contain(
                root,
                unresolved_candidate,
                root_id=source.root_id,
            )
        except AssetServiceError as exc:
            if not unresolved_candidate.exists() and not unresolved_candidate.is_symlink():
                raise _asset_error(
                    "ASSET_NOT_FOUND",
                    "The registered asset source is no longer available.",
                    details={"asset_id": asset_id},
                ) from exc
            raise
        bundle_base = candidate if candidate.is_dir() else candidate.parent
        unresolved_file = bundle_base.joinpath(*logical.parts)
        try:
            path = self._contain(
                root,
                unresolved_file,
                root_id=source.root_id,
            )
        except AssetServiceError as exc:
            if not unresolved_file.exists() and not unresolved_file.is_symlink():
                raise _asset_error(
                    "ASSET_NOT_FOUND",
                    "A declared asset file is no longer available.",
                    details={"asset_id": asset_id, "relative_path": selected},
                ) from exc
            raise
        try:
            path.relative_to(bundle_base)
        except ValueError as exc:
            raise _asset_error(
                "ASSET_OUTSIDE_ALLOWED_ROOT",
                "A declared asset file resolves outside its bundle boundary.",
                details={"asset_id": asset_id, "relative_path": selected},
            ) from exc
        if not path.is_file():
            raise _asset_error(
                "ASSET_NOT_FOUND",
                "A declared asset file is no longer available.",
                details={"asset_id": asset_id, "relative_path": selected},
            )
        if verify_hash:
            digest, size = _sha256_file(path)
            if digest != asset_file.sha256 or size != asset_file.size_bytes:
                raise _asset_error(
                    "ASSET_HASH_MISMATCH",
                    "A registered asset file no longer matches its manifest.",
                    details={
                        "asset_id": asset_id,
                        "relative_path": selected,
                        "expected_sha256": asset_file.sha256,
                        "actual_sha256": digest,
                    },
                )
        return path


__all__ = [
    "AssetDiscoverer",
    "AssetRegistry",
    "AssetServiceError",
    "DiscoveredAsset",
    "DiscoveredAssetFile",
    "RootProvider",
]
