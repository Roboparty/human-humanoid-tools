"""Project immutable R2R plans into executable JobSpec v2 documents."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from hhtools.contracts import (
    AssetInspectionRequest,
    AssetKind,
    InspectionStatus,
    JobSpecCalibration,
    JobSpecInput,
    JobSpecKind,
    JobSpecProvenance,
    JobSpecRobot,
    JobSpecV2,
    NextAction,
    R2RPlan,
)
from hhtools.utils.paths import user_robot_dir

from .assets import AssetServiceError
from .plans import PlanStore, PlanStoreError
from .r2r_preflight import R2R_PLAN_SEMANTICS
from .retarget import (
    ProvenanceProvider,
    RetargetService,
    RetargetServiceError,
    _default_provenance,
    _service_error,
    _snapshot_provenance,
)


def _preflight_action(plan_id: str) -> NextAction:
    return NextAction(
        actor="agent",
        action="preflight_r2r",
        message="Run R2R preflight again before starting or retrying this plan.",
        parameters={"plan_id": plan_id},
    )


def _asset_digest(asset_id: str) -> str:
    return asset_id.removeprefix("asset:sha256:")


def _payload_object(payload: Mapping[str, Any], field: str, plan_id: str) -> Mapping[str, Any]:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise _service_error(
            "PLAN_STALE",
            "The immutable R2R plan payload is incomplete.",
            details={"plan_id": plan_id, "field": field},
            next_action=_preflight_action(plan_id),
        )
    return value


def _sha256_file(path: Path) -> tuple[str, tuple[int, int, int, int]]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise OSError("managed file changed while hashing")
    return digest.hexdigest(), after_identity


class R2RRetargetService:
    """Re-verify all four R2R content identities before materializing a JobSpec."""

    def __init__(
        self,
        plan_store: PlanStore,
        asset_service: Any,
        *,
        provenance_provider: ProvenanceProvider = _default_provenance,
    ) -> None:
        self._plan_store = plan_store
        self._asset_service = asset_service
        self._provenance_json = _snapshot_provenance(provenance_provider)

    def _plan_record(self, plan_id: str) -> tuple[R2RPlan, dict[str, Any]]:
        try:
            plan = self._plan_store.get(plan_id)
            payload = self._plan_store.get_payload(plan_id)
        except PlanStoreError as error:
            raise RetargetServiceError(error.api_error) from error
        if not isinstance(plan, R2RPlan) or payload.get("semantics") != R2R_PLAN_SEMANTICS:
            raise _service_error(
                "UNSUPPORTED_PLAN_SEMANTICS",
                "The requested plan is not an R2R plan.",
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
    ):
        reason: str | None = None
        try:
            bundle = self._asset_service.get(asset_id)
            inspection = self._asset_service.inspect(
                AssetInspectionRequest(
                    asset_id=asset_id,
                    verify_hashes=True,
                    parse_content=True,
                )
            )
        except AssetServiceError as error:
            reason = error.code
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
                "An R2R asset no longer matches the immutable plan.",
                details={
                    "plan_id": plan_id,
                    "asset_id": asset_id,
                    "reason_code": reason,
                },
                next_action=_preflight_action(plan_id),
            )
        return bundle, inspection

    def _verify_pair_calibration(
        self,
        *,
        plan: R2RPlan,
        payload: Mapping[str, Any],
        target_bundle: Any,
    ) -> None:
        relative_path = payload.get("relative_path")
        expected_digest = payload.get("digest")
        storage = payload.get("storage")
        if not isinstance(relative_path, str) or not isinstance(expected_digest, str):
            raise _service_error(
                "PLAN_STALE",
                "The pair calibration identity recorded by the plan is incomplete.",
                details={"plan_id": plan.plan_id},
                next_action=_preflight_action(plan.plan_id),
            )
        if storage == "robot_bundle":
            profile = next(
                (item for item in target_bundle.files if item.relative_path == relative_path),
                None,
            )
            valid = bool(
                profile is not None
                and profile.role.value == "metadata"
                and profile.sha256 == expected_digest
            )
        elif storage == "user_calibration":
            try:
                root = user_robot_dir().resolve(strict=True)
                relative = PurePosixPath(relative_path)
                candidate = root.joinpath(*relative.parts).resolve(strict=True)
                candidate.relative_to(root)
                valid = candidate.is_file() and _sha256_file(candidate)[0] == expected_digest
            except (OSError, RuntimeError, ValueError):
                valid = False
        else:
            valid = False
        if not valid:
            raise _service_error(
                "PLAN_STALE",
                "The pair calibration no longer matches the immutable R2R plan.",
                details={"plan_id": plan.plan_id},
                next_action=_preflight_action(plan.plan_id),
            )

    def get_job_spec(self, plan_id: str) -> JobSpecV2:
        plan, payload = self._plan_record(plan_id)
        trajectory_payload = _payload_object(payload, "trajectory", plan_id)
        source_payload = _payload_object(payload, "source_robot", plan_id)
        target_payload = _payload_object(payload, "target_robot", plan_id)
        calibration_payload = _payload_object(payload, "pair_calibration", plan_id)

        _trajectory_bundle, trajectory_inspection = self._verified_asset(
            plan_id=plan_id,
            asset_id=plan.trajectory_asset_id,
            expected_digest=trajectory_payload.get("digest"),
            expected_kind=AssetKind.ROBOT_TRAJECTORY_BUNDLE,
        )
        source_bundle, source_inspection = self._verified_asset(
            plan_id=plan_id,
            asset_id=plan.source_robot_asset_id,
            expected_digest=source_payload.get("digest"),
            expected_kind=AssetKind.ROBOT_BUNDLE,
        )
        target_bundle, target_inspection = self._verified_asset(
            plan_id=plan_id,
            asset_id=plan.target_robot_asset_id,
            expected_digest=target_payload.get("digest"),
            expected_kind=AssetKind.ROBOT_BUNDLE,
        )
        routing_matches = bool(
            trajectory_inspection.category.value == trajectory_payload.get("category")
            and trajectory_inspection.source_robot_id
            == trajectory_payload.get("declared_source_robot_id")
            and trajectory_inspection.metadata.get("trajectory_profile")
            == trajectory_payload.get("profile")
            and not trajectory_inspection.has_object
            and not trajectory_inspection.has_terrain
            and source_inspection.category.value == "robot_model"
            and target_inspection.category.value == "robot_model"
            and source_payload.get("robot_id") == plan.source_robot_id
            and target_payload.get("robot_id") == plan.target_robot_id
        )
        if not routing_matches:
            raise _service_error(
                "PLAN_STALE",
                "R2R routing metadata no longer matches the immutable plan.",
                details={"plan_id": plan_id},
                next_action=_preflight_action(plan_id),
            )
        self._verify_pair_calibration(
            plan=plan,
            payload=calibration_payload,
            target_bundle=target_bundle,
        )

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
            kind=JobSpecKind.R2R_RETARGET,
            plan_id=plan.plan_id,
            inputs=[
                JobSpecInput(
                    asset_id=plan.trajectory_asset_id,
                    sha256=plan.trajectory_digest,
                )
            ],
            source_robot=JobSpecRobot(
                robot_id=plan.source_robot_id,
                asset_id=source_bundle.asset_id,
                config_sha256=plan.source_robot_digest,
            ),
            robot=JobSpecRobot(
                robot_id=plan.target_robot_id,
                asset_id=target_bundle.asset_id,
                config_sha256=plan.target_robot_digest,
            ),
            calibration=JobSpecCalibration(
                calibration_id=plan.calibration_id,
                sha256=plan.calibration_digest,
            ),
            backend=plan.backend,
            effective_parameters=effective_parameters,
            output_policy=plan.output_policy,
            provenance=provenance,
            created_at=plan.created_at,
        )
        return JobSpecV2.model_validate_json(spec.model_dump_json())


class WorkflowRetargetService:
    """Dispatch one plan id to its workflow-specific immutable-spec projector."""

    def __init__(
        self,
        plan_store: PlanStore,
        h2r: RetargetService,
        r2r: R2RRetargetService,
    ) -> None:
        self._plan_store = plan_store
        self._h2r = h2r
        self._r2r = r2r

    def get_job_spec(self, plan_id: str) -> JobSpecV2:
        try:
            semantics = self._plan_store.get_payload(plan_id).get("semantics")
        except PlanStoreError as error:
            raise RetargetServiceError(error.api_error) from error
        if semantics == R2R_PLAN_SEMANTICS:
            return self._r2r.get_job_spec(plan_id)
        if semantics == "hhtools.retarget.plan.v1":
            return self._h2r.get_job_spec(plan_id)
        raise _service_error(
            "UNSUPPORTED_PLAN_SEMANTICS",
            "The requested plan does not belong to a supported workflow.",
            details={"plan_id": plan_id},
        )


__all__ = ["R2RRetargetService", "WorkflowRetargetService"]
