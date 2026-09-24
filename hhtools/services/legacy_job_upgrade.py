"""Safely upgrade one path-based H2R JobSpec v1 into JobSpec v2.

The v1 format predates content-addressed assets and immutable preflight plans.
This service therefore treats its absolute ``source_path`` as a one-time local
lookup hint only.  A trusted, deployment-owned :class:`DynamicRootLocator`
must map that hint and the selected robot preset back into purpose-specific
allowlisted roots.  The resulting portable paths are registered and inspected,
then normal preflight and :class:`~hhtools.services.retarget.RetargetService`
produce the plan and JobSpec v2.

No solver is imported, no scheduler slot is reserved, and no output path is
accepted from v1.  The migration receipt contains content identities and
digests only; host paths are never returned or persisted by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from hhtools.contracts import (
    ApiError,
    AssetBundle,
    AssetInspection,
    AssetInspectionRequest,
    AssetKind,
    AssetRegistrationRequest,
    ErrorStage,
    InspectionStatus,
    JobSpecV2,
    LegacyJobUpgradeResponse,
    LegacyMigrationReceipt,
    OutputPolicy,
    PreflightResponse,
    PreflightStatus,
    RetargetPreflightRequest,
)
from hhtools.contracts.legacy_jobs import JobSpecError, build_job_spec, normalize_job_spec
from hhtools.retarget.calibration.calibration import normalize_calibration_reference

from .assets import AssetServiceError
from .retarget import RetargetServiceError

RootUsage = Literal["motion", "robot"]
RootProvider = Path | Callable[[], Path]

_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_FIELD_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_RAW_SPEC_FIELDS = frozenset({"schema_version", "kind", "request"})
_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "job_id",
        "kind",
        "status",
        "created_at",
        "finished_at",
        "scope",
        "request",
        "cli",
        "spec",
        "replay",
        "parent_job_id",
    }
)
_REQUEST_FIELDS = frozenset(
    {
        "source_path",
        "source_entry",
        "robot",
        "reference",
        "backend",
        "ik_iterations",
        "human_height",
        "limit_frames",
        "retarget_fps",
        "foot_clamp_anti_penetration",
    }
)
_SOURCE_ENTRY_FIELDS = frozenset(
    {
        "dataset",
        "folder_label",
        "sequence_id",
        "source_path",
        "stem",
        "label",
        "name",
        "display_name",
        "origin",
        "reference",
        "upload_profile",
        "export_subdir",
        "suggested_backend",
        "motion_category",
        "asset_kind",
        "has_scene",
    }
)
_SOURCE_ENTRY_STRING_FIELDS = _SOURCE_ENTRY_FIELDS.difference({"has_scene"})
_BACKENDS = frozenset({"newton", "interaction_mesh"})
_REFERENCES = frozenset(
    {
        "smplx",
        "smpl",
        "gvhmr",
        "soma_bvh",
        "lafan_bvh",
        "mocap_bvh",
        "xsens_mocap",
        "glb",
    }
)
_MOTION_CATEGORY_CLAIMS = {
    "motion": "plain_motion",
    "object": "object_interaction",
    "terrain": "terrain_scene",
}
_UPLOAD_PROFILE_CLAIMS = {
    "mimic": "plain_motion",
    "intermimic": "object_interaction",
    "meshmimic": "terrain_scene",
}
_MAX_DOCUMENT_BYTES = 64 * 1024
_MAX_STRING_LENGTH = 16 * 1024
_MAX_DEPTH = 16
_MAX_NODES = 4_096
_HASH_CHUNK_SIZE = 1024 * 1024
_INTERACTION_IK_WARNING = "LEGACY_INTERACTION_IK_ITERATIONS_IGNORED"


class LegacyJobUpgradeError(RuntimeError):
    """Expected migration failure carrying the shared public error body."""

    def __init__(self, error: ApiError) -> None:
        self.error = error
        super().__init__(f"{error.code}: {error.message}")

    @property
    def api_error(self) -> ApiError:
        return self.error

    @property
    def code(self) -> str:
        return self.error.code


def _upgrade_error(
    code: str,
    message: str,
    *,
    stage: ErrorStage = ErrorStage.REQUEST,
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
) -> LegacyJobUpgradeError:
    return LegacyJobUpgradeError(
        ApiError(
            code=code,
            message=message,
            stage=stage,
            retryable=retryable,
            details=dict(details or {}),
        )
    )


def _invalid(
    message: str,
    *,
    field: str | None = None,
    fields: Iterable[str] | None = None,
) -> LegacyJobUpgradeError:
    details: dict[str, Any] = {}
    if field is not None:
        details["field"] = field
    if fields is not None:
        supplied = list(fields)
        safe_fields = sorted(
            value
            for value in supplied
            if isinstance(value, str) and _SAFE_FIELD_NAME.fullmatch(value) is not None
        )
        details["field_count"] = len(supplied)
        if safe_fields:
            details["fields"] = safe_fields
    return _upgrade_error("INVALID_JOB_SPEC", message, details=details)


@dataclass(frozen=True, slots=True)
class _PathSnapshot:
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int


class _TrackedRootProvider:
    """Record resolved-root generations across locator and registry calls."""

    def __init__(self, provider: RootProvider) -> None:
        self._provider = provider
        self._lock = Lock()
        self._last: Path | None = None
        self._generation = 0

    def resolve(self) -> tuple[Path, int]:
        with self._lock:
            supplied = self._provider() if callable(self._provider) else self._provider
            resolved = Path(supplied).resolve(strict=True)
            if self._last is not None and resolved != self._last:
                self._generation += 1
            self._last = resolved
            return resolved, self._generation

    def __call__(self) -> Path:
        """AssetRegistry-compatible provider that shares the generation log."""

        return self.resolve()[0]


def _snapshot(path: Path, *, root_id: str) -> _PathSnapshot:
    try:
        value = path.stat()
    except OSError as exc:
        raise _upgrade_error(
            "ASSET_NOT_FOUND",
            "An allowlisted asset changed or became unreadable during migration.",
            stage=ErrorStage.ASSET_REGISTRATION,
            retryable=True,
            details={"root_id": root_id},
        ) from exc
    return _PathSnapshot(
        device=value.st_dev,
        inode=value.st_ino,
        mode=value.st_mode,
        size=value.st_size,
        modified_ns=value.st_mtime_ns,
    )


@dataclass(frozen=True, slots=True)
class _RootMatch:
    """Internal root match; host paths must never cross a transport boundary."""

    usage: RootUsage
    root_id: str
    relative_path: str
    resolved_root: Path
    resolved_candidate: Path
    root_snapshot: _PathSnapshot
    root_generation: int
    candidate_snapshot: _PathSnapshot
    candidate_is_directory: bool


class DynamicRootLocator:
    """Reverse-map host paths into trusted, purpose-specific dynamic roots.

    Root mappings are supplied by the composition root, not by the legacy
    document.  Providers are called again before preflight so a settings or
    mount change cannot silently redirect an already-authorized migration.
    """

    def __init__(
        self,
        *,
        motion_roots: Mapping[str, RootProvider],
        robot_roots: Mapping[str, RootProvider],
    ) -> None:
        self._roots: dict[RootUsage, dict[str, _TrackedRootProvider]] = {
            "motion": self._validated_roots(motion_roots),
            "robot": self._validated_roots(robot_roots),
        }
        duplicated = set(self._roots["motion"]).intersection(self._roots["robot"])
        if duplicated:
            raise ValueError("motion and robot root ids must be purpose-specific")

    @staticmethod
    def _validated_roots(
        values: Mapping[str, RootProvider],
    ) -> dict[str, _TrackedRootProvider]:
        roots = dict(values)
        if not roots:
            raise ValueError("at least one allowlisted root is required per purpose")
        for root_id in roots:
            if len(root_id) > 128 or _PORTABLE_ID.fullmatch(root_id) is None:
                raise ValueError("root ids must be portable identifiers")
        return {root_id: _TrackedRootProvider(provider) for root_id, provider in roots.items()}

    def registry_root_providers(self) -> dict[str, Callable[[], Path]]:
        """Return the tracked providers for the in-process AssetRegistry.

        The composition root should configure ``AssetRegistry`` with this
        mapping.  Sharing these wrappers lets the locator detect an A→B→A
        root-settings race even when both trees contain identical bytes.
        """

        return {
            root_id: provider
            for providers in self._roots.values()
            for root_id, provider in providers.items()
        }

    def allowed_root_ids(self, usage: RootUsage) -> frozenset[str]:
        """Return safe identifiers for validating a deduplicated asset source."""

        return frozenset(self._roots[usage])

    def locate_motion_file(self, source_path: str) -> _RootMatch:
        return self._locate(Path(source_path), usage="motion", expect_directory=False)

    def locate_robot_directory(self, root_dir: Path) -> _RootMatch:
        return self._locate(Path(root_dir), usage="robot", expect_directory=True)

    def _resolve_root(self, usage: RootUsage, root_id: str) -> tuple[Path, int]:
        provider = self._roots[usage][root_id]
        try:
            root, generation = provider.resolve()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise _upgrade_error(
                "ALLOWED_ROOT_UNAVAILABLE",
                "A configured asset root is unavailable.",
                stage=ErrorStage.ASSET_REGISTRATION,
                retryable=True,
                details={"root_id": root_id, "usage": usage},
            ) from exc
        if not root.is_dir():
            raise _upgrade_error(
                "ALLOWED_ROOT_UNAVAILABLE",
                "A configured asset root is not a directory.",
                stage=ErrorStage.ASSET_REGISTRATION,
                details={"root_id": root_id, "usage": usage},
            )
        return root, generation

    def _locate(
        self,
        candidate: Path,
        *,
        usage: RootUsage,
        expect_directory: bool,
    ) -> _RootMatch:
        if not candidate.is_absolute():
            raise _upgrade_error(
                "ASSET_OUTSIDE_ALLOWED_ROOT",
                "Legacy asset paths must be absolute paths below an allowed root.",
                stage=ErrorStage.ASSET_REGISTRATION,
                details={"usage": usage},
            )
        try:
            resolved_candidate = candidate.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise _upgrade_error(
                "ASSET_NOT_FOUND",
                "The legacy asset path is missing or unreadable.",
                stage=ErrorStage.ASSET_REGISTRATION,
                retryable=True,
                details={"usage": usage},
            ) from exc
        if expect_directory != resolved_candidate.is_dir():
            expected = "directory" if expect_directory else "file"
            raise _upgrade_error(
                "ASSET_KIND_MISMATCH",
                f"The legacy {usage} asset must resolve to a regular {expected}.",
                stage=ErrorStage.ASSET_REGISTRATION,
                details={"usage": usage},
            )
        if not expect_directory and not resolved_candidate.is_file():
            raise _upgrade_error(
                "ASSET_KIND_MISMATCH",
                "The legacy motion asset must resolve to a regular file.",
                stage=ErrorStage.ASSET_REGISTRATION,
                details={"usage": usage},
            )

        matches: list[tuple[int, str, Path, Path, int]] = []
        for root_id in sorted(self._roots[usage]):
            root, generation = self._resolve_root(usage, root_id)
            try:
                relative = resolved_candidate.relative_to(root)
            except ValueError:
                continue
            if not relative.parts:
                continue
            matches.append((len(root.parts), root_id, root, relative, generation))
        if not matches:
            raise _upgrade_error(
                "ASSET_OUTSIDE_ALLOWED_ROOT",
                "The legacy asset is outside every allowed root for its purpose.",
                stage=ErrorStage.ASSET_REGISTRATION,
                details={"usage": usage},
            )

        matches.sort(key=lambda item: (-item[0], item[1]))
        specificity = matches[0][0]
        most_specific = [item for item in matches if item[0] == specificity]
        if len(most_specific) != 1:
            raise _upgrade_error(
                "AMBIGUOUS_ALLOWED_ROOT",
                "The legacy asset matches multiple equally specific allowed roots.",
                stage=ErrorStage.ASSET_REGISTRATION,
                details={
                    "usage": usage,
                    "root_ids": sorted(item[1] for item in most_specific),
                },
            )
        _, root_id, root, relative, generation = most_specific[0]
        relative_path = relative.as_posix()
        try:
            AssetRegistrationRequest(
                root_id=root_id,
                relative_path=relative_path,
                display_name=None,
            )
        except ValidationError as exc:
            raise _upgrade_error(
                "ASSET_OUTSIDE_ALLOWED_ROOT",
                "The allowlisted asset path is not portable.",
                stage=ErrorStage.ASSET_REGISTRATION,
                details={"root_id": root_id, "usage": usage},
            ) from exc
        return _RootMatch(
            usage=usage,
            root_id=root_id,
            relative_path=relative_path,
            resolved_root=root,
            resolved_candidate=resolved_candidate,
            root_snapshot=_snapshot(root, root_id=root_id),
            root_generation=generation,
            candidate_snapshot=_snapshot(resolved_candidate, root_id=root_id),
            candidate_is_directory=expect_directory,
        )

    def revalidate(self, match: _RootMatch) -> None:
        """Reject a root remap, path replacement, or candidate type change."""

        root, generation = self._resolve_root(match.usage, match.root_id)
        try:
            candidate = root.joinpath(*PurePosixPath(match.relative_path).parts).resolve(
                strict=True
            )
            candidate.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise _upgrade_error(
                "ASSET_CHANGED_DURING_UPGRADE",
                "An allowlisted asset changed while it was being migrated.",
                stage=ErrorStage.ASSET_REGISTRATION,
                retryable=True,
                details={"root_id": match.root_id, "usage": match.usage},
            ) from exc
        unchanged = (
            root == match.resolved_root
            and generation == match.root_generation
            and candidate == match.resolved_candidate
            and _snapshot(root, root_id=match.root_id) == match.root_snapshot
            and _snapshot(candidate, root_id=match.root_id) == match.candidate_snapshot
            and candidate.is_dir() == match.candidate_is_directory
            and (candidate.is_dir() or candidate.is_file())
        )
        if not unchanged:
            raise _upgrade_error(
                "ASSET_CHANGED_DURING_UPGRADE",
                "An allowlisted asset changed while it was being migrated.",
                stage=ErrorStage.ASSET_REGISTRATION,
                retryable=True,
                details={"root_id": match.root_id, "usage": match.usage},
            )


class _AssetService(Protocol):
    def register(self, request: AssetRegistrationRequest) -> AssetBundle: ...

    def inspect(self, request: AssetInspectionRequest) -> AssetInspection: ...


class _PreflightService(Protocol):
    def preflight_retarget(self, request: RetargetPreflightRequest) -> PreflightResponse: ...


class _RetargetService(Protocol):
    def get_job_spec(self, plan_id: str) -> JobSpecV2: ...


class _RobotPreset(Protocol):
    name: str
    root_dir: Path


LegacyJobUpgradeResult = LegacyJobUpgradeResponse


def _validate_json_shape(value: Any) -> bytes:
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_NODES:
            raise _invalid("The legacy JobSpec contains too many values.")
        if depth > _MAX_DEPTH:
            raise _invalid("The legacy JobSpec is nested too deeply.")
        if item is None or isinstance(item, bool | int):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise _invalid("The legacy JobSpec contains a non-finite number.")
            return
        if isinstance(item, str):
            if len(item) > _MAX_STRING_LENGTH:
                raise _invalid("The legacy JobSpec contains an oversized string.")
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise _invalid("The legacy JobSpec object keys must be strings.")
                if len(key) > 256:
                    raise _invalid("The legacy JobSpec contains an oversized field name.")
                visit(child, depth + 1)
            return
        if isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
            return
        raise _invalid("The legacy JobSpec must contain strict JSON values only.")

    visit(value, 0)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _invalid("The legacy JobSpec is not canonical JSON.") from exc
    if len(encoded) > _MAX_DOCUMENT_BYTES:
        raise _invalid("The legacy JobSpec exceeds the migration size limit.")
    return encoded


def _strict_canonical_v1(payload: Any) -> tuple[dict[str, Any], bytes]:
    _validate_json_shape(payload)
    if not isinstance(payload, dict):
        raise _invalid("The legacy JobSpec envelope must be an object.")

    if "spec" in payload:
        unknown_envelope = set(payload).difference(_ENVELOPE_FIELDS)
        if unknown_envelope:
            raise _invalid(
                "The downloaded JobSpec envelope contains unsupported fields.",
                fields=unknown_envelope,
            )
        job_id = payload.get("job_id")
        if job_id is not None and (not isinstance(job_id, str) or not job_id or len(job_id) > 256):
            raise _invalid("job_id must be a non-empty bounded string.", field="job_id")
        candidate = payload.get("spec")
    else:
        candidate = payload
    if not isinstance(candidate, dict):
        raise _invalid("spec must be a JSON object.", field="spec")

    unknown_spec = set(candidate).difference(_RAW_SPEC_FIELDS)
    if unknown_spec:
        raise _invalid("The v1 spec contains unsupported fields.", fields=unknown_spec)
    if set(candidate) != _RAW_SPEC_FIELDS:
        raise _invalid(
            "The v1 spec must contain schema_version, kind, and request.",
            fields=_RAW_SPEC_FIELDS.difference(candidate),
        )
    version = candidate.get("schema_version")
    if type(version) is not int or version != 1:
        raise _invalid("schema_version must be the integer 1.", field="schema_version")
    if candidate.get("kind") != "retarget":
        raise _invalid("Only single H2R retarget JobSpec v1 can be upgraded.", field="kind")
    raw_request = candidate.get("request")
    if not isinstance(raw_request, dict):
        raise _invalid("request must be a JSON object.", field="request")
    unknown_request = set(raw_request).difference(_REQUEST_FIELDS)
    if unknown_request:
        raise _invalid(
            "The retarget request contains unsupported fields.",
            fields=unknown_request,
        )

    try:
        normalized = normalize_job_spec(candidate)
    except JobSpecError as exc:
        raise _invalid("The legacy JobSpec is not a supported canonical v1 document.") from exc
    if "spec" in payload:
        outer_version = payload.get("schema_version")
        if outer_version is not None and (type(outer_version) is not int or outer_version != 1):
            raise _invalid(
                "The downloaded envelope schema_version must be the integer 1.",
                field="schema_version",
            )
        has_outer_kind = "kind" in payload
        has_outer_request = "request" in payload
        if has_outer_kind != has_outer_request:
            raise _invalid(
                "The downloaded envelope must provide kind and request together.",
            )
        if has_outer_kind:
            outer_kind = payload["kind"]
            outer_request = payload["request"]
            if outer_kind != "retarget" or not isinstance(outer_request, dict):
                raise _invalid(
                    "The downloaded envelope kind and request are invalid.",
                    field="request",
                )
            if build_job_spec(outer_kind, outer_request) != normalized:
                raise _invalid(
                    "The downloaded envelope conflicts with its nested JobSpec.",
                    field="spec",
                )
        for field in ("status", "scope"):
            value = payload.get(field)
            if value is not None and (not isinstance(value, str) or not value):
                raise _invalid(
                    f"The downloaded envelope {field} must be a non-empty string.",
                    field=field,
                )
        for field in ("created_at", "finished_at"):
            value = payload.get(field)
            if value is not None and not _is_finite_number(value):
                raise _invalid(
                    f"The downloaded envelope {field} must be a finite number or null.",
                    field=field,
                )
        for field in ("cli", "replay"):
            value = payload.get(field)
            if value is not None and not isinstance(value, dict):
                raise _invalid(
                    f"The downloaded envelope {field} must be an object.",
                    field=field,
                )
        parent_job_id = payload.get("parent_job_id")
        if parent_job_id is not None and (
            not isinstance(parent_job_id, str) or not parent_job_id or len(parent_job_id) > 256
        ):
            raise _invalid(
                "The downloaded envelope parent_job_id must be a bounded string or null.",
                field="parent_job_id",
            )
    canonical = _validate_json_shape(normalized)
    return normalized, canonical


def _nonempty_string(request: Mapping[str, Any], field: str) -> str:
    value = request.get(field)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise _invalid(f"{field} must be a non-empty canonical string.", field=field)
    return value


def _optional_string(request: Mapping[str, Any], field: str, default: str) -> str:
    if field not in request:
        return default
    return _nonempty_string(request, field)


def _strict_positive_number(value: Any, *, field: str, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _invalid(f"{field} must be a finite number.", field=field)
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise _invalid(f"{field} must be a finite number.", field=field) from exc
    if not math.isfinite(normalized) or normalized <= minimum:
        raise _invalid(f"{field} must be a finite positive number.", field=field)
    return normalized


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _strict_positive_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise _invalid(f"{field} must be a positive integer.", field=field)
    return value


def _parameters(request: Mapping[str, Any], backend: str) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    limit = request.get("limit_frames")
    if limit is not None and (type(limit) is not int or limit < 0):
        raise _invalid("limit_frames must be a non-negative integer or null.", field="limit_frames")
    if limit is None or limit == 0:
        parameters["run_mode"] = "full"
    else:
        parameters["run_mode"] = "smoke"
        parameters["limit_frames"] = _strict_positive_int(limit, field="limit_frames")

    ik_iterations = request.get("ik_iterations", 24)
    normalized_iterations = _strict_positive_int(ik_iterations, field="ik_iterations")
    # The legacy Web worker always captured an IK value, including for its
    # interaction path.  The current preflight contract correctly rejects this
    # Newton-only parameter, so validate but omit it for interaction jobs.
    if backend == "newton":
        parameters["ik_iterations"] = normalized_iterations

    human_height = request.get("human_height")
    if human_height is not None:
        parameters["human_height"] = _strict_positive_number(
            human_height,
            field="human_height",
            minimum=0.1,
        )
    retarget_fps = request.get("retarget_fps")
    if retarget_fps is not None:
        parameters["retarget_fps"] = _strict_positive_number(
            retarget_fps,
            field="retarget_fps",
            minimum=0.0,
        )
    foot_clamp = request.get("foot_clamp_anti_penetration", False)
    if not isinstance(foot_clamp, bool):
        raise _invalid(
            "foot_clamp_anti_penetration must be a boolean.",
            field="foot_clamp_anti_penetration",
        )
    parameters["foot_clamp_anti_penetration"] = foot_clamp
    return parameters


def _validate_source_entry(
    request: Mapping[str, Any],
    *,
    source_match: _RootMatch,
    reference: str,
) -> None:
    raw = request.get("source_entry")
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise _invalid("source_entry must be a JSON object.", field="source_entry")
    unknown = set(raw).difference(_SOURCE_ENTRY_FIELDS)
    if unknown:
        raise _invalid("source_entry contains unsupported fields.", fields=unknown)
    for field in sorted(_SOURCE_ENTRY_STRING_FIELDS.intersection(raw)):
        value = raw[field]
        if value is not None and not isinstance(value, str):
            raise _invalid(
                "source_entry text fields must be strings or null.",
                field=f"source_entry.{field}",
            )
    if "has_scene" in raw and not isinstance(raw["has_scene"], bool):
        raise _invalid("source_entry.has_scene must be a boolean.", field="source_entry.has_scene")
    if raw.get("asset_kind") not in {None, "human_motion"}:
        raise _invalid(
            "Robot trajectories cannot be upgraded as H2R motion inputs.",
            field="source_entry.asset_kind",
        )
    if raw.get("motion_category") not in {None, *_MOTION_CATEGORY_CLAIMS}:
        raise _invalid(
            "source_entry.motion_category is unsupported.",
            field="source_entry.motion_category",
        )
    if raw.get("suggested_backend") not in {None, *_BACKENDS}:
        raise _invalid(
            "source_entry.suggested_backend is unsupported.",
            field="source_entry.suggested_backend",
        )
    if raw.get("upload_profile") not in {None, "", "auto", *_UPLOAD_PROFILE_CLAIMS}:
        raise _invalid(
            "source_entry.upload_profile is unsupported.",
            field="source_entry.upload_profile",
        )

    nested_source = raw.get("source_path")
    if nested_source is not None:
        try:
            nested_path = Path(nested_source)
            if not nested_path.is_absolute():
                raise ValueError("nested source path is not absolute")
            nested_resolved = nested_path.resolve(strict=True)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise _invalid(
                "source_entry.source_path is not the selected source asset.",
                field="source_entry.source_path",
            ) from exc
        if nested_resolved != source_match.resolved_candidate:
            raise _invalid(
                "source_entry.source_path is not the selected source asset.",
                field="source_entry.source_path",
            )
    entry_reference = raw.get("reference")
    if (
        entry_reference is not None
        and normalize_calibration_reference(entry_reference) != reference
    ):
        raise _invalid(
            "source_entry.reference conflicts with the retarget request.",
            field="source_entry.reference",
        )


def _normalized_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def _validate_source_inspection_claims(
    request: Mapping[str, Any],
    *,
    inspection: AssetInspection,
    backend: str,
) -> None:
    raw = request.get("source_entry")
    if not isinstance(raw, dict):
        return
    # ``origin`` and ``export_subdir`` are historical display/storage labels,
    # not execution-routing claims. They remain in the canonical-v1 digest but
    # never influence registration, preflight, or the resulting JobSpec.

    def conflict(field: str) -> None:
        raise _upgrade_error(
            "LEGACY_METADATA_MISMATCH",
            "Legacy source metadata conflicts with safe asset inspection.",
            stage=ErrorStage.PREFLIGHT,
            details={"field": f"source_entry.{field}", "asset_id": inspection.asset_id},
        )

    dataset = raw.get("dataset")
    if isinstance(dataset, str) and dataset.strip() and dataset.casefold() != "unknown":
        inspected_dataset = inspection.dataset
        if not isinstance(inspected_dataset, str) or _normalized_label(
            dataset
        ) != _normalized_label(inspected_dataset):
            conflict("dataset")

    category = raw.get("motion_category")
    if isinstance(category, str):
        expected = _MOTION_CATEGORY_CLAIMS[category]
        if inspection.category.value != expected:
            conflict("motion_category")

    upload_profile = raw.get("upload_profile")
    if (
        isinstance(upload_profile, str)
        and upload_profile in _UPLOAD_PROFILE_CLAIMS
        and inspection.category.value != _UPLOAD_PROFILE_CLAIMS[upload_profile]
    ):
        conflict("upload_profile")

    has_scene = raw.get("has_scene")
    if isinstance(has_scene, bool):
        inspected_scene = bool(inspection.has_object or inspection.has_terrain)
        if has_scene != inspected_scene:
            conflict("has_scene")

    suggested_backend = raw.get("suggested_backend")
    if isinstance(suggested_backend, str) and suggested_backend != backend:
        conflict("suggested_backend")


def _verified_registration(
    asset_service: _AssetService,
    match: _RootMatch,
    *,
    kind: AssetKind,
    allowed_source_root_ids: frozenset[str],
) -> tuple[AssetBundle, AssetInspection]:
    try:
        bundle = asset_service.register(
            AssetRegistrationRequest(
                root_id=match.root_id,
                relative_path=match.relative_path,
                display_name=None,
                kind=kind,
                recursive=kind is AssetKind.ROBOT_BUNDLE,
            )
        )
        inspection = asset_service.inspect(
            AssetInspectionRequest(
                asset_id=bundle.asset_id,
                verify_hashes=True,
                parse_content=True,
            )
        )
    except AssetServiceError as exc:
        raise LegacyJobUpgradeError(exc.api_error) from exc

    source = bundle.source
    valid_source = source is not None and source.root_id in allowed_source_root_ids
    if (
        bundle.kind is not kind
        or inspection.asset_id != bundle.asset_id
        or inspection.kind is not kind
        or not valid_source
    ):
        raise _upgrade_error(
            "ASSET_REGISTRATION_MISMATCH",
            "The registered asset does not match the authorized root lookup.",
            stage=ErrorStage.ASSET_REGISTRATION,
            details={"root_id": match.root_id, "usage": match.usage},
        )
    if inspection.status is InspectionStatus.INVALID:
        if inspection.errors:
            raise LegacyJobUpgradeError(inspection.errors[0])
        raise _upgrade_error(
            "ASSET_INSPECTION_FAILED",
            "The registered asset did not pass safe content inspection.",
            stage=ErrorStage.ASSET_INSPECTION,
            details={"asset_id": bundle.asset_id},
        )
    return bundle, inspection


def _hash_stable_file(path: Path, *, root_id: str) -> tuple[str, int]:
    """Hash a located file without ever returning its host path."""

    before = _snapshot(path, root_id=root_id)
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise _upgrade_error(
            "ASSET_CHANGED_DURING_UPGRADE",
            "An allowlisted asset changed while it was being migrated.",
            stage=ErrorStage.ASSET_REGISTRATION,
            retryable=True,
            details={"root_id": root_id},
        ) from exc
    after = _snapshot(path, root_id=root_id)
    if before != after or size != after.size:
        raise _upgrade_error(
            "ASSET_CHANGED_DURING_UPGRADE",
            "An allowlisted asset changed while it was being migrated.",
            stage=ErrorStage.ASSET_REGISTRATION,
            retryable=True,
            details={"root_id": root_id},
        )
    return digest.hexdigest(), size


def _verify_registered_content(bundle: AssetBundle, match: _RootMatch) -> None:
    """Bind registration output back to the exact path authorized by lookup.

    ``AssetRegistry`` may itself use a dynamic root provider.  Comparing every
    registered manifest file with the originally located tree closes the race
    where that provider changes between reverse lookup and registration, even
    if it later changes back before :meth:`DynamicRootLocator.revalidate`.
    """

    base = (
        match.resolved_candidate
        if match.candidate_is_directory
        else match.resolved_candidate.parent
    )
    try:
        primary = base.joinpath(*PurePosixPath(bundle.primary_file).parts).resolve(strict=True)
        primary.relative_to(base)
    except (OSError, RuntimeError, ValueError) as exc:
        raise _upgrade_error(
            "ASSET_REGISTRATION_MISMATCH",
            "The registered asset does not match the authorized filesystem snapshot.",
            stage=ErrorStage.ASSET_REGISTRATION,
            details={"root_id": match.root_id, "usage": match.usage},
        ) from exc
    if not match.candidate_is_directory and primary != match.resolved_candidate:
        raise _upgrade_error(
            "ASSET_REGISTRATION_MISMATCH",
            "The registered asset does not match the authorized filesystem snapshot.",
            stage=ErrorStage.ASSET_REGISTRATION,
            details={"root_id": match.root_id, "usage": match.usage},
        )

    for item in bundle.files:
        try:
            candidate = base.joinpath(*PurePosixPath(item.relative_path).parts).resolve(strict=True)
            candidate.relative_to(base)
        except (OSError, RuntimeError, ValueError) as exc:
            raise _upgrade_error(
                "ASSET_REGISTRATION_MISMATCH",
                "The registered asset does not match the authorized filesystem snapshot.",
                stage=ErrorStage.ASSET_REGISTRATION,
                details={"root_id": match.root_id, "usage": match.usage},
            ) from exc
        digest, size = _hash_stable_file(candidate, root_id=match.root_id)
        if digest != item.sha256 or size != item.size_bytes:
            raise _upgrade_error(
                "ASSET_REGISTRATION_MISMATCH",
                "The registered asset does not match the authorized filesystem snapshot.",
                stage=ErrorStage.ASSET_REGISTRATION,
                details={"root_id": match.root_id, "usage": match.usage},
            )


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class LegacyJobUpgradeService:
    """Orchestrate a strict, read-only single-H2R v1 to v2 migration."""

    def __init__(
        self,
        asset_service: _AssetService,
        preflight_service: _PreflightService,
        retarget_service: _RetargetService,
        root_locator: DynamicRootLocator,
        *,
        robot_provider: Callable[[], Iterable[_RobotPreset]] | None = None,
    ) -> None:
        if robot_provider is None:
            from hhtools.robot.registry import list_presets_readonly

            robot_provider = list_presets_readonly
        self._asset_service = asset_service
        self._preflight_service = preflight_service
        self._retarget_service = retarget_service
        self._root_locator = root_locator
        self._robot_provider = robot_provider

    def _robot_preset(self, robot_id: str) -> _RobotPreset:
        try:
            matches = [preset for preset in self._robot_provider() if preset.name == robot_id]
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise _upgrade_error(
                "ROBOT_REGISTRY_UNAVAILABLE",
                "The trusted robot registry is unavailable.",
                stage=ErrorStage.ASSET_REGISTRATION,
                retryable=True,
            ) from exc
        if not matches:
            raise _upgrade_error(
                "ROBOT_NOT_FOUND",
                "The legacy target robot is not available in the trusted registry.",
                stage=ErrorStage.ASSET_REGISTRATION,
                details={"robot_id": robot_id},
            )
        if len(matches) != 1:
            raise _upgrade_error(
                "ROBOT_AMBIGUOUS",
                "The trusted robot registry contains duplicate robot identifiers.",
                stage=ErrorStage.ASSET_REGISTRATION,
                details={"robot_id": robot_id},
            )
        return matches[0]

    def upgrade(self, payload: Any) -> LegacyJobUpgradeResult:
        """Create a content-bound JobSpec v2 without starting any work."""

        spec, canonical_v1 = _strict_canonical_v1(payload)
        request = spec["request"]
        source_path = _nonempty_string(request, "source_path")
        robot_id = _nonempty_string(request, "robot")
        if len(robot_id) > 256 or _PORTABLE_ID.fullmatch(robot_id) is None:
            raise _invalid("robot must be a portable identifier.", field="robot")

        reference = normalize_calibration_reference(_optional_string(request, "reference", "smpl"))
        if reference not in _REFERENCES:
            raise _invalid(
                "reference is not a supported calibration reference.",
                field="reference",
            )
        backend = _optional_string(request, "backend", "newton")
        if backend not in _BACKENDS:
            raise _invalid("backend is not a supported H2R backend.", field="backend")
        parameters = _parameters(request, backend)

        motion_match = self._root_locator.locate_motion_file(source_path)
        _validate_source_entry(
            request,
            source_match=motion_match,
            reference=reference,
        )
        preset = self._robot_preset(robot_id)
        robot_match = self._root_locator.locate_robot_directory(preset.root_dir)

        motion_bundle, motion_inspection = _verified_registration(
            self._asset_service,
            motion_match,
            kind=AssetKind.MOTION_BUNDLE,
            allowed_source_root_ids=self._root_locator.allowed_root_ids("motion"),
        )
        robot_bundle, _robot_inspection = _verified_registration(
            self._asset_service,
            robot_match,
            kind=AssetKind.ROBOT_BUNDLE,
            allowed_source_root_ids=self._root_locator.allowed_root_ids("robot"),
        )
        # Re-read the tracked dynamic providers before trusting registration
        # output. Shared registry providers make even an A→B→A remap advance
        # the generation, while the snapshots catch in-place replacements.
        self._root_locator.revalidate(motion_match)
        self._root_locator.revalidate(robot_match)
        _verify_registered_content(motion_bundle, motion_match)
        _verify_registered_content(robot_bundle, robot_match)
        detected_reference = motion_inspection.reference_model
        if (
            not isinstance(detected_reference, str)
            or normalize_calibration_reference(detected_reference) != reference
        ):
            raise _upgrade_error(
                "REFERENCE_MISMATCH",
                "The legacy reference does not match the inspected motion asset.",
                stage=ErrorStage.PREFLIGHT,
                details={"asset_id": motion_bundle.asset_id},
            )
        _validate_source_inspection_claims(
            request,
            inspection=motion_inspection,
            backend=backend,
        )

        # Preflight and RetargetService perform their own content-hash
        # verification as separate, non-executing boundaries.
        try:
            preflight = self._preflight_service.preflight_retarget(
                RetargetPreflightRequest(
                    motion_asset_id=motion_bundle.asset_id,
                    robot_id=robot_id,
                    robot_asset_id=robot_bundle.asset_id,
                    backend=backend,
                    calibration_id=None,
                    output_format="csv",
                    output_policy=OutputPolicy.CREATE_NEW,
                    parameters=parameters,
                )
            )
        except ValidationError as exc:
            raise _invalid("The legacy parameters cannot form a safe preflight request.") from exc

        if preflight.status is not PreflightStatus.READY or preflight.plan is None:
            return LegacyJobUpgradeResult(
                preflight=preflight,
                job_spec=None,
                receipt=None,
            )

        self._root_locator.revalidate(motion_match)
        self._root_locator.revalidate(robot_match)
        try:
            job_spec = self._retarget_service.get_job_spec(preflight.plan.plan_id)
        except RetargetServiceError as exc:
            raise LegacyJobUpgradeError(exc.api_error) from exc
        if job_spec.plan_id != preflight.plan.plan_id:
            raise _upgrade_error(
                "JOB_SPEC_PLAN_MISMATCH",
                "RetargetService returned a JobSpec for a different immutable plan.",
                stage=ErrorStage.INTERNAL,
            )

        receipt = LegacyMigrationReceipt(
            canonical_v1_sha256=hashlib.sha256(canonical_v1).hexdigest(),
            motion_asset_id=motion_bundle.asset_id,
            robot_asset_id=robot_bundle.asset_id,
            plan_id=job_spec.plan_id,
            job_spec_sha256=_sha256_json(job_spec.model_dump(mode="json")),
            warnings=(
                (_INTERACTION_IK_WARNING,)
                if backend == "interaction_mesh" and "ik_iterations" in request
                else ()
            ),
        )
        return LegacyJobUpgradeResult(
            preflight=preflight,
            job_spec=job_spec,
            receipt=receipt,
        )


__all__ = [
    "DynamicRootLocator",
    "LegacyJobUpgradeError",
    "LegacyJobUpgradeResult",
    "LegacyJobUpgradeService",
    "LegacyMigrationReceipt",
    "RootProvider",
    "RootUsage",
]
