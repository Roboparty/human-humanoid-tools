"""Read-only projection of immutable retarget plans into JobSpec v2.

This facade is the final non-executing boundary before a future JobManager.  It
does not import a solver, select or probe a device, reserve scheduler admission,
or create output files.  A specification is rebuilt from an immutable plan on
every call after re-verifying the registered asset bytes.

Runtime provenance is captured once when the service is constructed.  The
default provider reads the Git/build and installed-package identity only; it
never queries GPU memory, scheduler occupancy, or another transient device
state.  The returned spec uses the plan creation time, so the same plan and
service provenance produce the same JobSpec without a second persistence
layer.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from collections.abc import Callable, Mapping
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from hhtools._version import __version__
from hhtools.contracts import (
    ApiError,
    AssetBundle,
    AssetInspection,
    AssetInspectionRequest,
    AssetKind,
    ErrorStage,
    InspectionStatus,
    JobSpecCalibration,
    JobSpecInput,
    JobSpecKind,
    JobSpecProvenance,
    JobSpecRobot,
    JobSpecV2,
    NextAction,
    RetargetPlan,
)
from hhtools.utils.paths import user_robot_dir

from .assets import AssetServiceError
from .plans import PlanStore, PlanStoreError

_PLAN_SEMANTICS = "hhtools.retarget.plan.v1"
_ASSET_ID_PREFIX = "asset:sha256:"
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class RetargetServiceError(RuntimeError):
    """Expected facade failure with the shared transport-neutral error body."""

    def __init__(self, error: ApiError) -> None:
        self.error = error
        super().__init__(f"{error.code}: {error.message}")

    @property
    def api_error(self) -> ApiError:
        return self.error

    @property
    def code(self) -> str:
        return self.error.code


class _AssetProvider(Protocol):
    def get(self, asset_id: str) -> AssetBundle: ...

    def inspect(self, request: AssetInspectionRequest) -> AssetInspection: ...


ProvenanceProvider = Callable[[], JobSpecProvenance | Mapping[str, Any]]


def _service_error(
    code: str,
    message: str,
    *,
    stage: ErrorStage = ErrorStage.PREFLIGHT,
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
    next_action: NextAction | None = None,
) -> RetargetServiceError:
    return RetargetServiceError(
        ApiError(
            code=code,
            message=message,
            stage=stage,
            retryable=retryable,
            details=dict(details or {}),
            next_action=next_action,
        )
    )


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _git_output(*arguments: str) -> str | None:
    """Read one bounded Git fact without invoking a shell."""

    try:
        completed = subprocess.run(  # noqa: S603 - fixed executable and arguments
            ["git", "-C", str(_REPOSITORY_ROOT), *arguments],  # noqa: S607
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _default_provenance() -> JobSpecProvenance:
    """Capture static code/runtime identity without touching an accelerator."""

    commit = _git_output("rev-parse", "HEAD")
    if commit is None or _GIT_COMMIT.fullmatch(commit) is None:
        commit = "unknown"
        dirty = True
    else:
        status = _git_output("status", "--porcelain", "--untracked-files=normal")
        # Failure to prove a clean tree must never be reported as clean.
        dirty = status is None or bool(status)

    pytorch = _distribution_version("torch")
    newton = _distribution_version("newton", "newton-python")
    dependencies = {
        name: version
        for name, version in (
            ("hhtools", __version__),
            ("mujoco", _distribution_version("mujoco")),
            ("numpy", _distribution_version("numpy")),
            ("pydantic", _distribution_version("pydantic")),
            ("warp", _distribution_version("warp-lang", "warp")),
        )
        if version is not None
    }
    return JobSpecProvenance(
        hhtools_git_commit=commit,
        hhtools_dirty=dirty,
        python=platform.python_version(),
        pytorch=pytorch,
        # Importing Torch or a CUDA runtime merely to fill this field could
        # initialise device state.  The execution manifest records the actual
        # CUDA runtime and selected device later.
        cuda=None,
        newton=newton,
        device=None,
        platform=f"{platform.system()}-{platform.machine()}",
        dependencies=dict(sorted(dependencies.items())),
    )


def _snapshot_provenance(provider: ProvenanceProvider) -> str:
    """Validate, detach, and serialize one service-lifetime provenance fact."""

    try:
        supplied = provider()
        snapshot = JobSpecProvenance.model_validate(supplied)
        # This facade has not selected an execution device.  Never turn a
        # provider's current GPU choice into the immutable execution identity.
        snapshot = snapshot.model_copy(
            update={
                "device": None,
                "dependencies": dict(sorted(snapshot.dependencies.items())),
            }
        )
        encoded = snapshot.model_dump_json()
        JobSpecProvenance.model_validate_json(encoded)
    except (TypeError, ValueError, ValidationError) as exc:
        raise _service_error(
            "INTERNAL_ERROR",
            "The JobSpec provenance provider returned an invalid snapshot.",
            stage=ErrorStage.INTERNAL,
        ) from exc
    return encoded


def _payload_object(payload: Mapping[str, Any], field: str, plan_id: str) -> Mapping[str, Any]:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise _service_error(
            "PLAN_STALE",
            "The immutable plan payload is incomplete.",
            details={"plan_id": plan_id, "field": field},
            next_action=_preflight_action(plan_id),
        )
    return value


def _preflight_action(plan_id: str) -> NextAction:
    return NextAction(
        actor="agent",
        action="run_preflight",
        message="Run retarget preflight again to create a current immutable plan.",
        parameters={"plan_id": plan_id},
    )


def _asset_digest(asset_id: str) -> str:
    if not asset_id.startswith(_ASSET_ID_PREFIX):
        return ""
    return asset_id.removeprefix(_ASSET_ID_PREFIX)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_user_calibration_profile(
    *,
    plan_id: str,
    relative_path: Any,
    expected_digest: Any,
) -> None:
    """Verify one managed user calibration without trusting the robot manifest.

    User calibration overlays intentionally live outside immutable packaged
    robot bundles.  The canonical plan therefore binds their portable path and
    exact content digest directly.  Resolve the path under the current user's
    managed robot root and re-hash it immediately before materializing a
    JobSpec, so switching users, changing ``HHTOOLS_ROBOT_DIR``, path escape,
    deletion, and post-preflight edits all make the plan stale.
    """

    if not isinstance(relative_path, str) or not isinstance(expected_digest, str):
        raise _service_error(
            "PLAN_STALE",
            "The managed calibration identity recorded by the plan is incomplete.",
            details={"plan_id": plan_id},
            next_action=_preflight_action(plan_id),
        )
    try:
        root = user_robot_dir().resolve(strict=True)
        candidate = (root / Path(relative_path)).resolve(strict=True)
        candidate.relative_to(root)
        if not candidate.is_file():
            raise ValueError("managed calibration is not a regular file")
        before = candidate.stat()
        actual_digest = _sha256_file(candidate)
        after = candidate.stat()
        stable_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
    except (OSError, RuntimeError, ValueError):
        actual_digest = None
        stable_identity = False
    if not stable_identity or actual_digest != expected_digest:
        raise _service_error(
            "PLAN_STALE",
            "The managed user calibration no longer matches the immutable plan.",
            details={"plan_id": plan_id},
            next_action=_preflight_action(plan_id),
        )


class RetargetService:
    """Resolve a verified immutable plan into a stable, non-executing JobSpec."""

    def __init__(
        self,
        plan_store: PlanStore,
        asset_service: _AssetProvider,
        *,
        provenance_provider: ProvenanceProvider = _default_provenance,
    ) -> None:
        self._plan_store = plan_store
        self._asset_service = asset_service
        # JSON storage prevents callers from mutating the nested dependency map
        # retained by this otherwise frozen Pydantic model.
        self._provenance_json = _snapshot_provenance(provenance_provider)

    def _plan_record(self, plan_id: str) -> tuple[RetargetPlan, dict[str, Any]]:
        try:
            plan = self._plan_store.get(plan_id)
            payload = self._plan_store.get_payload(plan_id)
        except PlanStoreError as exc:
            raise RetargetServiceError(exc.api_error) from exc
        if payload.get("semantics") != _PLAN_SEMANTICS:
            raise _service_error(
                "UNSUPPORTED_PLAN_SEMANTICS",
                "The requested plan is not a supported retarget plan.",
                details={"plan_id": plan_id},
            )
        return plan, payload

    def _verified_asset(
        self,
        *,
        plan_id: str,
        asset_id: str,
        expected_digest: Any,
        expected_kind: AssetKind,
    ) -> tuple[AssetBundle, AssetInspection]:
        reason: str | None = None
        try:
            bundle = self._asset_service.get(asset_id)
            inspection = self._asset_service.inspect(
                AssetInspectionRequest(
                    asset_id=asset_id,
                    verify_hashes=True,
                    parse_content=False,
                )
            )
        except AssetServiceError as exc:
            reason = exc.code
        else:
            if (
                bundle.asset_id != asset_id
                or inspection.asset_id != asset_id
                or expected_digest != _asset_digest(asset_id)
                or bundle.kind is not expected_kind
                or inspection.kind is not expected_kind
            ):
                reason = "BUNDLE_METADATA_MISMATCH"
            elif inspection.status is InspectionStatus.INVALID:
                reason = ",".join(sorted({error.code for error in inspection.errors}))

        if reason is not None:
            raise _service_error(
                "PLAN_STALE",
                "A content-bound asset no longer matches the immutable plan.",
                details={
                    "plan_id": plan_id,
                    "asset_id": asset_id,
                    "reason_code": reason,
                },
                next_action=_preflight_action(plan_id),
            )
        return bundle, inspection

    def get_job_spec(self, plan_id: str) -> JobSpecV2:
        """Return a stable JobSpec v2 after read-only plan and hash checks."""

        plan, payload = self._plan_record(plan_id)
        motion_payload = _payload_object(payload, "motion", plan_id)
        robot_payload = _payload_object(payload, "robot", plan_id)
        profile_payload = _payload_object(payload, "retarget_profile", plan_id)

        _motion_bundle, motion_inspection = self._verified_asset(
            plan_id=plan_id,
            asset_id=plan.motion_asset_id,
            expected_digest=motion_payload.get("digest"),
            expected_kind=AssetKind.MOTION_BUNDLE,
        )
        robot_bundle, robot_inspection = self._verified_asset(
            plan_id=plan_id,
            asset_id=plan.robot_asset_id,
            expected_digest=robot_payload.get("digest"),
            expected_kind=AssetKind.ROBOT_BUNDLE,
        )

        routing = {
            "category": motion_inspection.category.value,
            "dataset": motion_inspection.dataset,
            "reference": motion_inspection.reference_model,
        }
        expected_routing = {
            "category": motion_payload.get("category"),
            "dataset": motion_payload.get("dataset"),
            "reference": motion_payload.get("reference"),
        }
        recommended_backend = motion_inspection.metadata.get("recommended_backend")
        if (
            routing != expected_routing
            or robot_inspection.category.value != "robot_model"
            or (isinstance(recommended_backend, str) and recommended_backend != plan.backend)
        ):
            raise _service_error(
                "PLAN_STALE",
                "Asset routing metadata no longer matches the immutable plan.",
                details={"plan_id": plan_id},
                next_action=_preflight_action(plan_id),
            )

        profile_source = profile_payload.get("source")
        profile_storage = profile_payload.get("storage", "robot_bundle")
        profile_digest = profile_payload.get("digest")
        profile_relative_path = profile_payload.get("relative_path")
        if profile_storage == "robot_bundle":
            profile_file = next(
                (
                    item
                    for item in robot_bundle.files
                    if item.relative_path == profile_relative_path
                ),
                None,
            )
            if (
                profile_file is None
                or profile_file.role.value != "metadata"
                or profile_file.sha256 != profile_digest
            ):
                raise _service_error(
                    "PLAN_STALE",
                    "The retarget profile is not bound into the current robot bundle.",
                    details={"plan_id": plan_id},
                    next_action=_preflight_action(plan_id),
                )
        elif profile_storage == "user_calibration" and profile_source == "calibration":
            _verify_user_calibration_profile(
                plan_id=plan_id,
                relative_path=profile_relative_path,
                expected_digest=profile_digest,
            )
        else:
            raise _service_error(
                "PLAN_STALE",
                "The retarget profile storage recorded by the plan is unsupported.",
                details={"plan_id": plan_id},
                next_action=_preflight_action(plan_id),
            )
        calibration = None
        if profile_source == "calibration":
            assert plan.calibration_id is not None
            assert plan.calibration_digest is not None
            calibration = JobSpecCalibration(
                calibration_id=plan.calibration_id,
                sha256=plan.calibration_digest,
            )
        elif profile_source != "bundled_scaler":
            raise _service_error(
                "PLAN_STALE",
                "The retarget profile recorded by the plan is unsupported.",
                details={"plan_id": plan_id},
                next_action=_preflight_action(plan_id),
            )

        # JSON round-trip gives each caller an independent nested parameter map.
        effective_parameters = json.loads(
            json.dumps(
                plan.parameters,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        effective_parameters["output_format"] = plan.output_format
        provenance = JobSpecProvenance.model_validate_json(self._provenance_json)
        spec = JobSpecV2(
            kind=JobSpecKind.RETARGET,
            plan_id=plan.plan_id,
            inputs=[
                JobSpecInput(
                    asset_id=plan.motion_asset_id,
                    sha256=plan.input_digest,
                )
            ],
            robot=JobSpecRobot(
                robot_id=plan.robot_id,
                asset_id=plan.robot_asset_id,
                config_sha256=plan.robot_digest,
            ),
            calibration=calibration,
            backend=plan.backend,
            effective_parameters=effective_parameters,
            output_policy=plan.output_policy,
            provenance=provenance,
            created_at=plan.created_at,
        )
        # Return a fully detached model because JobSpec's open JSON maps are
        # intentionally mutable even though top-level assignment is frozen.
        return JobSpecV2.model_validate_json(spec.model_dump_json())


__all__ = [
    "ProvenanceProvider",
    "RetargetService",
    "RetargetServiceError",
]
