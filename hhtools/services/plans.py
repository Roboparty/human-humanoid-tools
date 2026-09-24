"""Immutable, content-addressed storage for resolved retarget plans.

``PlanStore`` is deliberately transport-neutral.  It persists only the public
``RetargetPlan`` document and the canonical JSON payload from which its
content id was derived.  In particular, local filesystem paths must never
cross this boundary into the database.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, NoReturn

from pydantic import ValidationError

from hhtools.contracts import ApiError, BatchPlan, ErrorStage, R2RPlan, RetargetPlan

_RETARGET_PLAN_SEMANTICS = "hhtools.retarget.plan.v1"
_R2R_PLAN_SEMANTICS = "hhtools.r2r.plan.v1"
_BATCH_PLAN_SEMANTICS = "hhtools.batch.plan.v1"
ExecutionPlan = RetargetPlan | R2RPlan | BatchPlan


class PlanStoreError(RuntimeError):
    """Expected plan-store failure with a transport-neutral error body."""

    def __init__(self, error: ApiError) -> None:
        self.error = error
        super().__init__(f"{error.code}: {error.message}")

    @property
    def api_error(self) -> ApiError:
        """Alias used by adapters that expose structured API errors."""

        return self.error

    @property
    def code(self) -> str:
        """Return the stable machine code without requiring message parsing."""

        return self.error.code


class _InvalidDocumentError(ValueError):
    """Private validation signal that never crosses the service boundary."""


def _raise_duplicate_key(key: str) -> NoReturn:
    raise _InvalidDocumentError(f"duplicate JSON object key: {key}")


def _object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _raise_duplicate_key(key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise _InvalidDocumentError(f"non-finite JSON number: {value}")


def _strict_json_loads(payload: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_object_from_pairs,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError) as exc:
        raise _InvalidDocumentError("invalid JSON document") from exc


def _looks_like_absolute_path(value: str) -> bool:
    """Recognize POSIX, drive-qualified, rooted, and UNC host paths."""

    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return posix.is_absolute() or windows.is_absolute() or bool(windows.drive) or bool(windows.root)


def _validate_portable_json(value: Any, *, location: str = "$") -> None:
    """Reject values that are not finite portable JSON.

    The location is intentionally used only for internal diagnostics.  Public
    errors never echo values or absolute paths supplied by a caller.
    """

    if value is None or isinstance(value, bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _InvalidDocumentError(f"non-finite number at {location}")
        return
    if isinstance(value, str):
        if _looks_like_absolute_path(value):
            raise _InvalidDocumentError(f"absolute host path at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_portable_json(item, location=f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _InvalidDocumentError(f"non-string object key at {location}")
            if _looks_like_absolute_path(key):
                raise _InvalidDocumentError(f"absolute host path key at {location}")
            _validate_portable_json(item, location=f"{location}.{key}")
        return
    raise _InvalidDocumentError(f"non-JSON value at {location}")


def _canonical_json(value: Any) -> str:
    _validate_portable_json(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise _InvalidDocumentError("document cannot be encoded as canonical JSON") from exc


def _error(
    code: str,
    message: str,
    *,
    stage: ErrorStage = ErrorStage.PREFLIGHT,
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
) -> PlanStoreError:
    return PlanStoreError(
        ApiError(
            code=code,
            message=message,
            retryable=retryable,
            stage=stage,
            details=dict(details or {}),
        )
    )


def _canonical_payload_document(canonical_payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if not isinstance(canonical_payload, dict):
        raise _error(
            "INVALID_PARAMETER",
            "A plan hash payload must be a JSON object with portable values.",
        )
    try:
        encoded = _canonical_json(canonical_payload)
        normalized = _strict_json_loads(encoded)
    except _InvalidDocumentError as exc:
        raise _error(
            "INVALID_PARAMETER",
            "A plan hash payload must be finite portable JSON without host paths.",
        ) from exc
    if not isinstance(normalized, dict):
        raise _error(
            "INVALID_PARAMETER",
            "A plan hash payload must be a JSON object with portable values.",
        )
    return encoded, normalized


def compute_plan_id(canonical_payload: Mapping[str, Any]) -> str:
    """Compute the deterministic ``plan:sha256`` id for a portable payload."""

    encoded, _ = _canonical_payload_document(canonical_payload)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"plan:sha256:{digest}"


def _encode_plan(plan: ExecutionPlan) -> str:
    """Validate and snapshot a plan without retaining nested caller objects."""

    try:
        # RetargetPlan is frozen, but ``parameters`` is intentionally an open
        # JSON object.  Validate it before Pydantic has an opportunity to turn
        # a Path or tuple into a superficially JSON-compatible representation.
        parameters = getattr(plan, "parameters", None)
        if parameters is not None:
            _validate_portable_json(parameters, location="$.parameters")
        document = plan.model_dump(mode="json")
        encoded = _canonical_json(document)
        restored = type(plan).model_validate_json(encoded)
    except (_InvalidDocumentError, TypeError, ValueError, ValidationError) as exc:
        raise _error(
            "INVALID_PARAMETER",
            "The retarget plan must be valid portable JSON without host paths.",
        ) from exc
    if restored != plan:
        raise _error(
            "INVALID_PARAMETER",
            "The retarget plan did not survive a lossless JSON round trip.",
        )
    return encoded


def _nested_object(document: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = document.get(field)
    if not isinstance(value, dict):
        raise _InvalidDocumentError(f"{field} must be a JSON object")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_portable_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return (
        not posix.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and all(part not in {"", ".", ".."} for part in posix.parts)
    )


def _validate_retarget_plan_projection(
    plan: RetargetPlan,
    canonical_payload: Mapping[str, Any],
) -> None:
    """Bind the public plan document to the canonical retarget semantics.

    The content id alone proves that ``canonical_payload`` has not changed; it
    does not prove that the separately persisted public ``RetargetPlan`` was
    projected from that payload.  Validate the fields exposed to callers so a
    cache hit cannot return a different robot, backend, calibration, or set of
    effective parameters under an otherwise valid payload hash.

    Unknown semantics remain opaque for backwards compatibility.  The v1
    retarget semantics are owned by HHTools, however, so malformed or divergent
    documents must never be accepted as immutable plans.
    """

    if canonical_payload.get("semantics") != _RETARGET_PLAN_SEMANTICS:
        return

    motion = _nested_object(canonical_payload, "motion")
    robot = _nested_object(canonical_payload, "robot")
    profile = _nested_object(canonical_payload, "retarget_profile")
    output = _nested_object(canonical_payload, "output")
    parameters = canonical_payload.get("parameters")
    if not isinstance(parameters, dict):
        raise _InvalidDocumentError("parameters must be a JSON object")

    profile_source = profile.get("source")
    # Older v1 plans predate writable per-user calibration overlays.  Treat a
    # missing discriminator as the original robot-bundle storage so persisted
    # plans retain their meaning after an upgrade.
    profile_storage = profile.get("storage", "robot_bundle")
    profile_digest = profile.get("digest")
    calibration_id = profile.get("calibration_id")
    profile_relative_path = profile.get("relative_path")
    if not _is_sha256(profile_digest):
        raise _InvalidDocumentError("retarget profile digest must be SHA-256")
    if not _is_portable_relative_path(profile_relative_path):
        raise _InvalidDocumentError("retarget profile path must be portable and relative")
    if profile_storage not in {"robot_bundle", "user_calibration"}:
        raise _InvalidDocumentError("unsupported retarget profile storage")
    if profile_storage == "user_calibration":
        robot_id = robot.get("robot_id")
        reference = motion.get("reference")
        if not isinstance(robot_id, str) or not isinstance(reference, str):
            raise _InvalidDocumentError("user calibration identity is incomplete")
        expected_paths = {
            f"{robot_id}/retarget_calibration_{reference}.yaml",
            f"{robot_id}/retarget_calibration.yaml",
            f".calibration-overlays/{robot_id}/retarget_calibration_{reference}.yaml",
            f".calibration-overlays/{robot_id}/retarget_calibration.yaml",
        }
        if profile_relative_path not in expected_paths:
            raise _InvalidDocumentError(
                "user calibration path must match the plan robot and reference"
            )
    if profile_source == "calibration":
        if not isinstance(calibration_id, str):
            raise _InvalidDocumentError("manual calibration must have an id")
        if calibration_id != f"cal:sha256:{profile_digest}":
            raise _InvalidDocumentError("manual calibration id must match the profile digest")
        projected_calibration_digest: Any = profile_digest
    elif profile_source == "bundled_scaler":
        if profile_storage != "robot_bundle":
            raise _InvalidDocumentError("bundled scaler must use robot-bundle storage")
        if calibration_id is not None:
            raise _InvalidDocumentError("bundled scaler cannot have a calibration id")
        projected_calibration_digest = None
    else:
        raise _InvalidDocumentError("unsupported retarget profile source")

    projected = {
        "motion_asset_id": motion.get("asset_id"),
        "robot_id": robot.get("robot_id"),
        "robot_asset_id": robot.get("asset_id"),
        "backend": canonical_payload.get("backend"),
        "calibration_id": calibration_id,
        "output_format": output.get("format"),
        "output_policy": output.get("policy"),
        "parameters": parameters,
        "input_digest": motion.get("digest"),
        "robot_digest": robot.get("digest"),
        "calibration_digest": projected_calibration_digest,
    }
    public_plan = plan.model_dump(mode="json")
    divergent = sorted(
        field for field, expected in projected.items() if public_plan.get(field) != expected
    )
    if divergent:
        raise _InvalidDocumentError(
            "retarget plan fields diverge from canonical payload: " + ", ".join(divergent)
        )


def _validate_r2r_plan_projection(
    plan: R2RPlan,
    canonical_payload: Mapping[str, Any],
) -> None:
    """Bind an R2R plan to its trajectory, robot pair, and calibration identity."""

    trajectory = _nested_object(canonical_payload, "trajectory")
    source_robot = _nested_object(canonical_payload, "source_robot")
    target_robot = _nested_object(canonical_payload, "target_robot")
    calibration = _nested_object(canonical_payload, "pair_calibration")
    output = _nested_object(canonical_payload, "output")
    parameters = canonical_payload.get("parameters")
    if not isinstance(parameters, dict):
        raise _InvalidDocumentError("parameters must be a JSON object")
    if trajectory.get("category") != "robot_trajectory":
        raise _InvalidDocumentError("R2R input category must be robot_trajectory")
    if trajectory.get("source_robot_id") != source_robot.get("robot_id"):
        raise _InvalidDocumentError("R2R trajectory source identity is inconsistent")
    if trajectory.get("profile") != "mimic":
        raise _InvalidDocumentError("R2R v1 supports only scene-free mimic trajectories")
    if canonical_payload.get("backend") != "newton":
        raise _InvalidDocumentError("R2R v1 supports only the Newton backend")

    digests = {
        "trajectory": trajectory.get("digest"),
        "source_robot": source_robot.get("digest"),
        "target_robot": target_robot.get("digest"),
        "pair_calibration": calibration.get("digest"),
    }
    if any(not _is_sha256(value) for value in digests.values()):
        raise _InvalidDocumentError("R2R identities must use SHA-256 digests")

    calibration_digest = calibration.get("digest")
    calibration_id = calibration.get("calibration_id")
    if calibration_id != f"cal:sha256:{calibration_digest}":
        raise _InvalidDocumentError("R2R calibration id must match its content digest")
    if calibration.get("source_robot_id") != source_robot.get("robot_id"):
        raise _InvalidDocumentError("R2R calibration source identity is inconsistent")
    if calibration.get("target_robot_id") != target_robot.get("robot_id"):
        raise _InvalidDocumentError("R2R calibration target identity is inconsistent")

    storage = calibration.get("storage")
    relative_path = calibration.get("relative_path")
    if storage not in {"robot_bundle", "user_calibration"}:
        raise _InvalidDocumentError("unsupported R2R calibration storage")
    if not _is_portable_relative_path(relative_path):
        raise _InvalidDocumentError("R2R calibration path must be portable and relative")
    if storage == "user_calibration":
        target_robot_id = target_robot.get("robot_id")
        path = PurePosixPath(str(relative_path))
        if (
            not isinstance(target_robot_id, str)
            or path.parent.as_posix()
            not in {target_robot_id, f".calibration-overlays/{target_robot_id}"}
            or not path.name.startswith("r2r_calibration_")
            or not path.name.endswith(".yaml")
        ):
            raise _InvalidDocumentError("user R2R calibration path must match the target robot")

    projected = {
        "trajectory_asset_id": trajectory.get("asset_id"),
        "source_robot_id": source_robot.get("robot_id"),
        "source_robot_asset_id": source_robot.get("asset_id"),
        "target_robot_id": target_robot.get("robot_id"),
        "target_robot_asset_id": target_robot.get("asset_id"),
        "backend": canonical_payload.get("backend"),
        "calibration_id": calibration_id,
        "output_format": output.get("format"),
        "output_policy": output.get("policy"),
        "parameters": parameters,
        "trajectory_digest": trajectory.get("digest"),
        "source_robot_digest": source_robot.get("digest"),
        "target_robot_digest": target_robot.get("digest"),
        "calibration_digest": calibration_digest,
    }
    public_plan = plan.model_dump(mode="json")
    divergent = sorted(
        field for field, expected in projected.items() if public_plan.get(field) != expected
    )
    if divergent:
        raise _InvalidDocumentError(
            "R2R plan fields diverge from canonical payload: " + ", ".join(divergent)
        )


def _validate_batch_plan_projection(
    plan: BatchPlan,
    canonical_payload: Mapping[str, Any],
) -> None:
    """Bind a batch plan to its ordered child projections and fixed ceilings."""

    target_robot = _nested_object(canonical_payload, "target_robot")
    output = _nested_object(canonical_payload, "output")
    resource_limits = _nested_object(canonical_payload, "resource_limits")
    source_robot = canonical_payload.get("source_robot")
    if source_robot is not None and not isinstance(source_robot, dict):
        raise _InvalidDocumentError("source_robot must be null or a JSON object")
    items = canonical_payload.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise _InvalidDocumentError("batch items must be an ordered object list")
    if items != [item.model_dump(mode="json") for item in plan.items]:
        raise _InvalidDocumentError("batch child plan projections are inconsistent")

    projected = {
        "workflow": canonical_payload.get("workflow"),
        "run_mode": canonical_payload.get("run_mode"),
        "target_robot_id": target_robot.get("robot_id"),
        "target_robot_asset_id": target_robot.get("asset_id"),
        "target_robot_digest": target_robot.get("digest"),
        "source_robot_id": source_robot.get("robot_id") if source_robot else None,
        "source_robot_asset_id": source_robot.get("asset_id") if source_robot else None,
        "source_robot_digest": source_robot.get("digest") if source_robot else None,
        "output_policy": output.get("policy"),
        "resource_limits": resource_limits,
    }
    public_plan = plan.model_dump(mode="json")
    divergent = sorted(
        field for field, expected in projected.items() if public_plan.get(field) != expected
    )
    if divergent:
        raise _InvalidDocumentError(
            "batch plan fields diverge from canonical payload: " + ", ".join(divergent)
        )


def _validate_plan_projection(
    plan: ExecutionPlan,
    canonical_payload: Mapping[str, Any],
) -> None:
    semantics = canonical_payload.get("semantics")
    if semantics == _BATCH_PLAN_SEMANTICS:
        if not isinstance(plan, BatchPlan):
            raise _InvalidDocumentError("batch semantics require a batch plan document")
        _validate_batch_plan_projection(plan, canonical_payload)
        return
    if semantics == _R2R_PLAN_SEMANTICS:
        if not isinstance(plan, R2RPlan):
            raise _InvalidDocumentError("R2R semantics require an R2R plan document")
        _validate_r2r_plan_projection(plan, canonical_payload)
        return
    if not isinstance(plan, RetargetPlan):
        raise _InvalidDocumentError("H2R semantics require a retarget plan document")
    _validate_retarget_plan_projection(plan, canonical_payload)


class PlanStore:
    """SQLite-backed immutable store for content-bound retarget plans."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)
        self._database_path = self._data_dir / "plans.sqlite3"
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The plan store directory could not be initialized.",
                stage=ErrorStage.INTERNAL,
                retryable=True,
            ) from exc
        self._initialize_database()

    @property
    def database_path(self) -> Path:
        """Return the internal database location for deployment diagnostics."""

        return self._database_path

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
                    CREATE TABLE IF NOT EXISTS plans (
                        plan_id TEXT PRIMARY KEY,
                        plan_json TEXT NOT NULL,
                        canonical_payload_json TEXT NOT NULL
                    )
                    """
                )
        except sqlite3.Error as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The plan store database could not be initialized.",
                stage=ErrorStage.INTERNAL,
                retryable=True,
            ) from exc

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> tuple[ExecutionPlan, dict[str, Any], str, str]:
        try:
            plan_id = row["plan_id"]
            plan_json = row["plan_json"]
            payload_json = row["canonical_payload_json"]
            if (
                not isinstance(plan_id, str)
                or not isinstance(plan_json, str)
                or not isinstance(payload_json, str)
            ):
                raise _InvalidDocumentError("persisted columns have invalid types")

            plan_document = _strict_json_loads(plan_json)
            payload_document = _strict_json_loads(payload_json)
            if not isinstance(plan_document, dict) or not isinstance(payload_document, dict):
                raise _InvalidDocumentError("persisted documents must be JSON objects")
            _validate_portable_json(plan_document)
            _validate_portable_json(payload_document)

            canonical_plan_json = _canonical_json(plan_document)
            canonical_payload_json = _canonical_json(payload_document)
            if canonical_plan_json != plan_json or canonical_payload_json != payload_json:
                raise _InvalidDocumentError("persisted documents are not canonical JSON")

            plan_model = (
                BatchPlan
                if payload_document.get("semantics") == _BATCH_PLAN_SEMANTICS
                else R2RPlan
                if payload_document.get("semantics") == _R2R_PLAN_SEMANTICS
                else RetargetPlan
            )
            plan = plan_model.model_validate(plan_document)
            expected_id = (
                f"plan:sha256:{hashlib.sha256(canonical_payload_json.encode('utf-8')).hexdigest()}"
            )
            if plan.plan_id != plan_id or plan_id != expected_id:
                raise _InvalidDocumentError("persisted plan identity is inconsistent")
            _validate_plan_projection(plan, payload_document)
        except (
            _InvalidDocumentError,
            KeyError,
            TypeError,
            ValueError,
            ValidationError,
        ) as exc:
            raise _error(
                "INTERNAL_ERROR",
                "A persisted retarget plan is invalid.",
                stage=ErrorStage.INTERNAL,
            ) from exc
        return plan, payload_document, canonical_plan_json, canonical_payload_json

    def put_if_absent(
        self,
        plan: ExecutionPlan,
        canonical_payload: Mapping[str, Any],
    ) -> ExecutionPlan:
        """Insert one plan exactly once, or return the identical stored plan.

        An existing id is never overwritten.  Reusing it with any different
        canonical payload or plan document is reported as ``PLAN_CONFLICT``.
        """

        payload_json, payload_document = _canonical_payload_document(canonical_payload)
        expected_id = f"plan:sha256:{hashlib.sha256(payload_json.encode('utf-8')).hexdigest()}"
        plan_json = _encode_plan(plan)
        if plan.plan_id != expected_id:
            raise _error(
                "PLAN_CONFLICT",
                "The plan id does not match its canonical hash payload.",
                details={"expected_plan_id": expected_id},
            )
        try:
            _validate_plan_projection(plan, payload_document)
        except _InvalidDocumentError as exc:
            raise _error(
                "PLAN_CONFLICT",
                "The retarget plan does not match its canonical hash payload.",
                details={"plan_id": plan.plan_id},
            ) from exc

        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT OR IGNORE INTO plans (
                        plan_id, plan_json, canonical_payload_json
                    ) VALUES (?, ?, ?)
                    """,
                    (plan.plan_id, plan_json, payload_json),
                )
                row = connection.execute(
                    """
                    SELECT plan_id, plan_json, canonical_payload_json
                    FROM plans
                    WHERE plan_id = ?
                    """,
                    (plan.plan_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The retarget plan could not be persisted.",
                stage=ErrorStage.INTERNAL,
                retryable=True,
            ) from exc
        if row is None:
            raise _error(
                "INTERNAL_ERROR",
                "The retarget plan was unavailable after persistence.",
                stage=ErrorStage.INTERNAL,
                retryable=True,
            )

        stored, _, stored_plan_json, stored_payload_json = self._decode_row(row)
        if stored_payload_json != payload_json or stored_plan_json != plan_json:
            raise _error(
                "PLAN_CONFLICT",
                "The plan id is already bound to a different immutable plan.",
                details={"plan_id": plan.plan_id},
            )
        return stored

    def get(self, plan_id: str) -> ExecutionPlan:
        """Load a fresh validated plan object by content id."""

        row = self._get_row(plan_id)
        plan, _, _, _ = self._decode_row(row)
        return plan

    def get_payload(self, plan_id: str) -> dict[str, Any]:
        """Load a fresh JSON copy of the canonical plan hash payload."""

        row = self._get_row(plan_id)
        _, payload, _, payload_json = self._decode_row(row)
        # Parsing again intentionally prevents a caller from retaining a
        # mutable object shared with another result in this operation.
        copied = _strict_json_loads(payload_json)
        if not isinstance(copied, dict):  # guarded by _decode_row
            raise _error(
                "INTERNAL_ERROR",
                "A persisted retarget plan payload is invalid.",
                stage=ErrorStage.INTERNAL,
            )
        return copied

    def _get_row(self, plan_id: str) -> sqlite3.Row:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT plan_id, plan_json, canonical_payload_json
                    FROM plans
                    WHERE plan_id = ?
                    """,
                    (plan_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The plan store could not be read.",
                stage=ErrorStage.INTERNAL,
                retryable=True,
            ) from exc
        if row is None:
            raise _error(
                "PLAN_NOT_FOUND",
                "No immutable retarget plan has the requested id.",
            )
        return row


__all__ = ["ExecutionPlan", "PlanStore", "PlanStoreError", "compute_plan_id"]
