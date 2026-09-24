"""Read-only discovery of registerable assets below configured roots."""

from __future__ import annotations

import os
import re
import struct
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Protocol

from hhtools.contracts import (
    ApiError,
    AssetCategory,
    AssetKind,
    AssetRegistrationRequest,
    AvailableAssetCatalogEntry,
    AvailableAssetCatalogRequest,
    AvailableAssetCatalogResponse,
    ErrorStage,
)

from .assets import AssetServiceError


class _RegistrationHints(Protocol):
    @property
    def allowed_root_ids(self) -> tuple[str, ...]: ...

    def registration_hint(
        self,
        trusted_path: Path,
        *,
        kind: AssetKind | None = None,
        category: AssetCategory | None = None,
        recursive: bool = True,
        preferred_root_id: str | None = None,
    ) -> AssetRegistrationRequest: ...

    def canonical_root_aliases(
        self,
        preferred_root_ids: tuple[str, ...] = (),
    ) -> dict[str, str]: ...


@dataclass(frozen=True, slots=True)
class AvailableAssetCandidate:
    """Trusted in-process candidate; its path never crosses a transport boundary."""

    path: Path
    display_name: str
    kind: AssetKind
    category: AssetCategory
    recursive: bool = True
    dataset: str | None = None
    reference: str | None = None
    required_paths: tuple[Path, ...] = ()


AvailableAssetProvider = Callable[[], Iterable[AvailableAssetCandidate]]
MAX_AVAILABLE_ASSET_SCAN_ENTRIES = 20_000
MAX_AVAILABLE_ASSET_CANDIDATES = 10_000
MAX_CATALOG_CSV_PROBE_BYTES = 64 * 1024
MAX_CATALOG_NPZ_CENTRAL_DIRECTORY_BYTES = 2 * 1024 * 1024
MAX_CATALOG_NPZ_MEMBERS = 4_096
MAX_CATALOG_NPZ_NAME_BYTES = 256 * 1024
_ZIP_EOCD = struct.Struct("<4s4H2LH")
_ZIP_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_ZIP64_LOCATOR_BYTES = 20
_ZIP_EOCD_MAX_BYTES = _ZIP_EOCD.size + 65_535 + _ZIP64_LOCATOR_BYTES


class AvailableAssetCatalogLimitError(RuntimeError):
    """A trusted catalog provider exceeded its fixed discovery budget."""


def iter_bounded_catalog_files(
    root: Path,
    *,
    extensions: frozenset[str],
    max_entries: int = MAX_AVAILABLE_ASSET_SCAN_ENTRIES,
) -> Iterable[Path]:
    """Yield regular files from a bounded tree without following links or loading content."""

    visited = 0
    pending = [Path(root)]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as children:
            for child in children:
                visited += 1
                if visited > max_entries:
                    raise AvailableAssetCatalogLimitError
                if child.name.startswith(".") or child.name == "__pycache__":
                    continue
                if child.is_dir(follow_symlinks=False):
                    pending.append(Path(child.path))
                    continue
                if not child.is_file(follow_symlinks=False):
                    continue
                path = Path(child.path)
                if path.suffix.casefold() in extensions:
                    yield path


def is_catalog_motion_sidecar(path: Path) -> bool:
    """Return whether a supported-looking file is a known companion, not a primary."""

    name = path.name.casefold()
    suffix = path.suffix.casefold()
    if suffix == ".csv" and (name == "motion_actor.csv" or name.startswith(("object_", "prop_"))):
        return True
    if suffix not in {".pkl", ".pickle"}:
        return False
    if name == "terrain.pkl":
        return True
    return any(
        path.with_suffix(primary_suffix).is_file()
        for primary_suffix in (".npz", ".npy", ".bvh", ".glb", ".gltf")
    )


def _relative_robot_name_hint(relative_path: PurePosixPath) -> bool:
    normalized = relative_path.as_posix().casefold()
    tokens = {token for token in re.split(r"[^a-z0-9]+", normalized) if token}
    parts = tuple(part.casefold() for part in relative_path.parts)
    return bool(
        tokens.intersection({"robot", "r2r", "jointq", "qpos"})
        or any(part.endswith("_export") or part == "exports" for part in parts)
    )


def _pkl_has_human_bundle_marker(path: Path, relative_path: PurePosixPath) -> bool:
    normalized_parts = {
        re.sub(r"[^a-z0-9]", "", part.casefold()) for part in relative_path.parts[:-1]
    }
    if normalized_parts.intersection({"omomo", "parcms"}):
        return True
    return bool(
        (path.parent / f"{path.stem}_terrain.obj").is_file()
        or any(path.parent.glob("*_cleaned_simplified.obj"))
    )


def _bounded_npz_directory(path: Path) -> bool:
    """Validate the ZIP directory budget before :mod:`zipfile` allocates it."""

    try:
        size = path.stat().st_size
        if size < _ZIP_EOCD.size:
            return False
        tail_size = min(size, _ZIP_EOCD_MAX_BYTES)
        with path.open("rb") as stream:
            stream.seek(size - tail_size)
            tail = stream.read(tail_size)
        offset = tail.rfind(_ZIP_EOCD_SIGNATURE)
        if offset < 0 or len(tail) - offset < _ZIP_EOCD.size:
            return False
        (
            signature,
            disk_number,
            central_disk,
            disk_members,
            members,
            central_size,
            central_offset,
            comment_size,
        ) = _ZIP_EOCD.unpack_from(tail, offset)
    except (OSError, struct.error):
        return False
    eocd_position = size - tail_size + offset
    locator_offset = offset - _ZIP64_LOCATOR_BYTES
    has_zip64_locator = (
        locator_offset >= 0
        and tail[locator_offset : locator_offset + len(_ZIP64_LOCATOR_SIGNATURE)]
        == _ZIP64_LOCATOR_SIGNATURE
    )
    return bool(
        signature == _ZIP_EOCD_SIGNATURE
        and not has_zip64_locator
        and disk_number == central_disk == 0
        and disk_members == members
        and members < 0xFFFF
        and central_size < 0xFFFFFFFF
        and central_offset < 0xFFFFFFFF
        and members <= MAX_CATALOG_NPZ_MEMBERS
        and central_size <= MAX_CATALOG_NPZ_CENTRAL_DIRECTORY_BYTES
        and central_offset + central_size <= eocd_position
        and offset + _ZIP_EOCD.size + comment_size == len(tail)
    )


def _classify_npz_robot_trajectory(path: Path) -> bool | None:
    """Inspect bounded NPZ member names without loading or unpickling arrays."""

    if not _bounded_npz_directory(path):
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except (OSError, ValueError, zipfile.BadZipFile):
        return None
    if (
        len(names) > MAX_CATALOG_NPZ_MEMBERS
        or sum(len(name.encode("utf-8")) for name in names) > MAX_CATALOG_NPZ_NAME_BYTES
    ):
        return None
    keys = {
        PurePosixPath(name).name.removesuffix(".npy").casefold()
        for name in names
        if name.casefold().endswith(".npy")
    }
    return bool(keys.intersection({"joint_q", "qpos", "q"}))


def classify_catalog_robot_trajectory(
    path: Path,
    *,
    relative_path: PurePosixPath,
) -> bool | None:
    """Classify robot exports; ``None`` means the bounded probe was inconclusive."""

    suffix = path.suffix.casefold()
    if suffix not in {".csv", ".npz", ".pkl", ".pickle"}:
        return False
    if _relative_robot_name_hint(relative_path):
        return True
    if suffix == ".npz":
        return _classify_npz_robot_trajectory(path)
    if suffix in {".pkl", ".pickle"}:
        return not _pkl_has_human_bundle_marker(path, relative_path)
    if suffix != ".csv":
        return False
    robot_trajectory: bool | None = None
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_CATALOG_CSV_PROBE_BYTES + 1)
    except OSError:
        robot_trajectory = None
    else:
        payload = payload[:MAX_CATALOG_CSV_PROBE_BYTES]
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            robot_trajectory = None
        else:
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                cells = [cell.strip() for cell in stripped.split(",")]
                normalized = [cell.casefold() for cell in cells]
                if "root_x" in normalized or any(cell.startswith("dof_") for cell in normalized):
                    robot_trajectory = True
                    break
                try:
                    robot_trajectory = len(cells) >= 8 and all(
                        float(cell) == float(cell) for cell in cells
                    )
                except ValueError:
                    pass
                break
    return robot_trajectory


def require_bounded_catalog_root(
    root: Path,
    *,
    max_entries: int = MAX_AVAILABLE_ASSET_SCAN_ENTRIES,
) -> None:
    """Reject a tree before existing discovery helpers can scan it without bounds."""

    visited = 0
    pending = [(Path(root), True)]
    while pending:
        directory, is_root = pending.pop()
        try:
            with os.scandir(directory) as children:
                for child in children:
                    visited += 1
                    if visited > max_entries:
                        raise AvailableAssetCatalogLimitError
                    if is_root and child.name.startswith((".", "_")):
                        continue
                    if child.is_dir(follow_symlinks=False):
                        pending.append((Path(child.path), False))
        except AvailableAssetCatalogLimitError:
            raise
        except OSError:
            raise


def _catalog_error(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: dict[str, object] | None = None,
) -> AssetServiceError:
    return AssetServiceError(
        ApiError(
            code=code,
            message=message,
            retryable=retryable,
            stage=ErrorStage.ASSET_REGISTRATION,
            details=details or {},
        )
    )


class AvailableAssetCatalogService:
    """List trusted candidates only after proving a portable registration identity."""

    def __init__(
        self,
        registration_hints: _RegistrationHints,
        providers: Mapping[str, AvailableAssetProvider],
        *,
        preferred_root_ids: tuple[str, ...] = (),
    ) -> None:
        self._registration_hints = registration_hints
        self._providers = dict(providers)
        self._preferred_root_ids = preferred_root_ids

    def list_available(
        self,
        request: AvailableAssetCatalogRequest,
    ) -> AvailableAssetCatalogResponse:
        """Return a deterministic page without hashing, registration, or unsafe decoding."""

        allowed = set(self._registration_hints.allowed_root_ids)
        if request.root_id is not None and request.root_id not in allowed:
            raise _catalog_error(
                "ASSET_OUTSIDE_ALLOWED_ROOT",
                "The requested asset root is not allowed.",
                details={"root_id": request.root_id},
            )

        aliases = self._registration_hints.canonical_root_aliases(self._preferred_root_ids)
        canonical_roots = set(aliases.values())
        root_ids = (
            [aliases[request.root_id]] if request.root_id is not None else sorted(canonical_roots)
        )
        entries: dict[tuple[str, str, str], AvailableAssetCatalogEntry] = {}
        for root_id in root_ids:
            provider_root_ids = sorted(
                alias for alias, canonical in aliases.items() if canonical == root_id
            )
            missing_provider = next(
                (alias for alias in provider_root_ids if alias not in self._providers),
                None,
            )
            if missing_provider is not None:
                raise _catalog_error(
                    "AVAILABLE_ASSET_CATALOG_UNAVAILABLE",
                    "The allowed asset root has no catalog provider.",
                    retryable=True,
                    details={"root_id": missing_provider},
                )

            try:
                seen_candidates: set[tuple[str, str, AssetKind, AssetCategory]] = set()
                candidate_index = 0
                providers = tuple(self._providers[alias] for alias in provider_root_ids)
                for candidate in chain.from_iterable(provider() for provider in providers):
                    candidate_index += 1
                    if candidate_index > MAX_AVAILABLE_ASSET_CANDIDATES:
                        raise AvailableAssetCatalogLimitError
                    if request.kind is not None and candidate.kind != request.kind:
                        continue
                    try:
                        candidate_path = Path(candidate.path)
                        hint = self._registration_hints.registration_hint(
                            candidate_path,
                            kind=candidate.kind,
                            category=candidate.category,
                            recursive=candidate.recursive,
                            preferred_root_id=root_id,
                        )
                        bundle_base = (
                            candidate_path.resolve(strict=True)
                            if candidate_path.is_dir()
                            else candidate_path.resolve(strict=True).parent
                        )
                        for required_path in candidate.required_paths:
                            resolved_required = Path(required_path).resolve(strict=True)
                            resolved_required.relative_to(bundle_base)
                            required_hint = self._registration_hints.registration_hint(
                                resolved_required,
                                preferred_root_id=root_id,
                            )
                            if required_hint.root_id != root_id:
                                raise ValueError("required asset belongs to another root")
                    except AssetServiceError as error:
                        if error.api_error.code in {
                            "ASSET_NOT_FOUND",
                            "ASSET_OUTSIDE_ALLOWED_ROOT",
                            "ASSET_ROOT_UNREPRESENTABLE",
                            "INVALID_PARAMETER",
                        }:
                            continue
                        raise
                    except (OSError, RuntimeError, TypeError, ValueError):
                        continue
                    # Overlapping configured roots use the registry's most-specific
                    # identity. Do not advertise the same path under a broader root.
                    if hint.root_id != root_id:
                        continue
                    try:
                        entry = AvailableAssetCatalogEntry(
                            root_id=hint.root_id,
                            relative_path=hint.relative_path,
                            display_name=candidate.display_name,
                            kind=candidate.kind,
                            category=candidate.category,
                            recursive=candidate.recursive,
                            dataset=candidate.dataset,
                            reference=candidate.reference,
                        )
                    except (TypeError, ValueError):
                        continue
                    candidate_key = (
                        entry.root_id,
                        entry.relative_path,
                        entry.kind,
                        entry.category,
                    )
                    if candidate_key in seen_candidates:
                        continue
                    seen_candidates.add(candidate_key)
                    key = (entry.root_id, entry.relative_path, entry.kind.value)
                    entries[key] = entry
            except AssetServiceError:
                raise
            except AvailableAssetCatalogLimitError as error:
                raise _catalog_error(
                    "AVAILABLE_ASSET_CATALOG_LIMIT_EXCEEDED",
                    "The allowed asset root exceeds the catalog discovery limit.",
                    details={
                        "root_id": root_id,
                        "max_scan_entries": MAX_AVAILABLE_ASSET_SCAN_ENTRIES,
                        "max_candidates": MAX_AVAILABLE_ASSET_CANDIDATES,
                    },
                ) from error
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise _catalog_error(
                    "AVAILABLE_ASSET_CATALOG_UNAVAILABLE",
                    "The available asset catalog could not scan an allowed root.",
                    retryable=True,
                    details={"root_id": root_id},
                ) from error

        ordered = sorted(
            entries.values(),
            key=lambda item: (
                item.root_id.casefold(),
                item.display_name.casefold(),
                item.relative_path.casefold(),
                item.kind.value,
            ),
        )
        if request.query is not None:
            terms = request.query.casefold().split()
            ordered = [
                item
                for item in ordered
                if all(
                    term
                    in " ".join(
                        value
                        for value in (
                            item.display_name,
                            item.relative_path,
                            item.kind.value,
                            item.category.value,
                            item.dataset or "",
                            item.reference or "",
                        )
                    ).casefold()
                    for term in terms
                )
            ]
        total = len(ordered)
        return AvailableAssetCatalogResponse(
            assets=ordered[request.offset : request.offset + request.limit],
            total=total,
            limit=request.limit,
            offset=request.offset,
        )


__all__ = [
    "AvailableAssetCandidate",
    "AvailableAssetCatalogLimitError",
    "AvailableAssetCatalogService",
    "AvailableAssetProvider",
    "MAX_AVAILABLE_ASSET_CANDIDATES",
    "MAX_AVAILABLE_ASSET_SCAN_ENTRIES",
    "MAX_CATALOG_CSV_PROBE_BYTES",
    "MAX_CATALOG_NPZ_CENTRAL_DIRECTORY_BYTES",
    "MAX_CATALOG_NPZ_MEMBERS",
    "MAX_CATALOG_NPZ_NAME_BYTES",
    "is_catalog_motion_sidecar",
    "classify_catalog_robot_trajectory",
    "iter_bounded_catalog_files",
    "require_bounded_catalog_root",
]
