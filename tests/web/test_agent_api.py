from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hhtools.agent.api import _error_response, _looks_like_host_path, router
from hhtools.contracts import (
    ApiError,
    ArtifactDescriptor,
    AssetBundle,
    AssetCategory,
    AssetDetected,
    AssetFile,
    AssetFileRole,
    AssetInspection,
    AssetRegistrationRequest,
    AssetSearchResponse,
    AvailableAssetCatalogEntry,
    AvailableAssetCatalogRequest,
    AvailableAssetCatalogResponse,
    BatchPreflightRequest,
    BatchPreflightResponse,
    CalibrationCandidate,
    CalibrationPreview,
    CalibrationProposalResponse,
    CalibrationSaveReceipt,
    CalibrationStatusResponse,
    CalibrationValidationReport,
    CapabilityResponse,
    ErrorStage,
    InspectionStatus,
    JobOutcome,
    JobSpecInput,
    JobSpecKind,
    JobSpecProvenance,
    JobSpecRobot,
    JobSpecV2,
    LegacyJobUpgradeResponse,
    LegacyMigrationReceipt,
    NextAction,
    OutputPolicy,
    PreflightCheck,
    PreflightResponse,
    R2RCalibrationCandidate,
    R2RCalibrationPreview,
    R2RCalibrationProposalResponse,
    R2RCalibrationSaveReceipt,
    R2RCalibrationStatusResponse,
    R2RPreflightRequest,
    R2RPreflightResponse,
    RetargetPlan,
    RetargetPreflightRequest,
    SchedulerCapability,
)
from hhtools.robot.registry import clear_cache
from hhtools.services.artifacts import ArtifactStore, StoredArtifact
from hhtools.services.assets import AssetServiceError
from hhtools.services.job_store import JobStore
from hhtools.services.jobs import (
    JobExecutionContext,
    JobExecutionResult,
    JobManager,
)
from hhtools.services.legacy_job_upgrade import LegacyJobUpgradeError
from hhtools.web import server
from hhtools.web.jobs.job_scheduler import JobScheduler
from hhtools.web.server import state as server_state

_DIGEST = "a" * 64
_ASSET_ID = f"asset:sha256:{_DIGEST}"
_R2R_TARGET_ASSET_ID = f"asset:sha256:{'b' * 64}"
_CALIBRATION_CANDIDATE_ID = f"cal-candidate:sha256:{'c' * 64}"


def _asset_bundle() -> AssetBundle:
    return AssetBundle(
        asset_id=_ASSET_ID,
        kind="motion_bundle",
        category=AssetCategory.PLAIN_MOTION,
        display_name="Walk",
        primary_file="walk.npz",
        files=[
            AssetFile(
                role=AssetFileRole.MOTION,
                relative_path="walk.npz",
                sha256=_DIGEST,
                size_bytes=128,
            )
        ],
        detected=AssetDetected(
            dataset="unified_npz",
            reference="smpl",
            recommended_backend="newton",
        ),
    )


class _FakeCapabilities:
    def get_capabilities(self) -> CapabilityResponse:
        return CapabilityResponse(
            service_version="test",
            scheduler=SchedulerCapability(
                max_running_jobs=0,
                max_queued_jobs=0,
                mode="unlimited",
            ),
            supported_input_formats=["bvh"],
            supported_output_formats=["csv", "pkl"],
            features={"agent_rest": True},
        )


class _FakeAssets:
    allowed_root_ids = ("motion-library",)

    def register(self, request: AssetRegistrationRequest) -> AssetBundle:
        assert request.root_id == "motion-library"
        return _asset_bundle()

    def get(self, asset_id: str) -> AssetBundle:
        if asset_id != _ASSET_ID:
            raise AssetServiceError(
                ApiError(
                    code="ASSET_NOT_FOUND",
                    message="No registered asset has the requested id.",
                    stage=ErrorStage.ASSET_REGISTRATION,
                    details={"asset_id": asset_id},
                )
            )
        return _asset_bundle()

    def search(self, **filters) -> AssetSearchResponse:
        return AssetSearchResponse(
            assets=[_asset_bundle()],
            total=1,
            limit=int(filters["limit"]),
            offset=int(filters["offset"]),
        )

    def inspect(self, request) -> AssetInspection:
        assert request.asset_id == _ASSET_ID
        return AssetInspection(
            asset_id=_ASSET_ID,
            status=InspectionStatus.VALID,
            kind="motion_bundle",
            category=AssetCategory.PLAIN_MOTION,
            source_format="npz",
            dataset="unified_npz",
            reference_model="smpl",
            frame_count=3,
            frame_rate_hz=30.0,
        )


class _FakeAvailableAssets:
    def list_available(
        self,
        request: AvailableAssetCatalogRequest,
    ) -> AvailableAssetCatalogResponse:
        if request.root_id == "unknown":
            raise AssetServiceError(
                ApiError(
                    code="ASSET_OUTSIDE_ALLOWED_ROOT",
                    message="The requested asset root is not allowed.",
                    stage=ErrorStage.ASSET_REGISTRATION,
                    details={"root_id": "unknown"},
                )
            )
        entry = AvailableAssetCatalogEntry(
            root_id="motion-library",
            relative_path="AMASS/walk.npz",
            display_name="Walk",
            kind="motion_bundle",
            category="plain_motion",
            dataset="amass",
            reference="smpl",
        )
        matches = request.query is None or request.query.casefold() in "walk amass"
        assets = [entry] if matches and request.offset == 0 else []
        return AvailableAssetCatalogResponse(
            assets=assets[: request.limit],
            total=1 if matches else 0,
            limit=request.limit,
            offset=request.offset,
        )


class _FakePreflight:
    def preflight_retarget(
        self,
        request: RetargetPreflightRequest,
    ) -> PreflightResponse:
        assert request.motion_asset_id == _ASSET_ID
        error = ApiError(
            code="ROBOT_ASSET_REQUIRED",
            message="A robot bundle is required.",
            stage=ErrorStage.PREFLIGHT,
        )
        return PreflightResponse(
            request_id="req_rest_test",
            status="rejected",
            error=error,
        )


class _FakeR2RPreflight:
    def preflight_r2r(
        self,
        request: R2RPreflightRequest,
    ) -> R2RPreflightResponse:
        assert request.trajectory_asset_id == _ASSET_ID
        return R2RPreflightResponse(
            request_id="req_r2r_rest_test",
            status="rejected",
            recommended_backend="newton",
            error=ApiError(
                code="R2R_CALIBRATION_REQUIRED",
                message="Pair calibration is required.",
                stage=ErrorStage.PREFLIGHT,
            ),
        )


class _FakeBatchPreflight:
    def preflight_batch(
        self,
        request: BatchPreflightRequest,
    ) -> BatchPreflightResponse:
        assert request.workflow.value == "h2r"
        return BatchPreflightResponse(
            request_id="req_batch_rest_test",
            status="rejected",
            error=ApiError(
                code="PLAN_NOT_FOUND",
                message="A child plan is unavailable.",
                stage=ErrorStage.PREFLIGHT,
            ),
        )


def _calibration_validation() -> CalibrationValidationReport:
    return CalibrationValidationReport(
        candidate_id=_CALIBRATION_CANDIDATE_ID,
        valid=True,
        score=0.95,
        changed_joint_count=2,
        mapped_slots=16,
        edge_errors_deg={"left_upper_arm": 4.0},
        checks=[
            PreflightCheck(
                code="CALIBRATION_POSE_ALIGNED",
                level="pass",
                message="Calibration pose is aligned.",
            )
        ],
    )


class _FakeCalibration:
    def status(self, request) -> CalibrationStatusResponse:
        return CalibrationStatusResponse(
            request_id="req_calibration_rest",
            state="missing",
            robot_id=request.robot_id,
            robot_asset_id=request.robot_asset_id,
            robot_digest=request.robot_asset_id.rsplit(":", 1)[-1],
            reference=request.reference,
            source="none",
            joint_count=0,
            mapped_slots=16,
            can_propose=True,
            can_silent_save=True,
        )

    def propose(self, request) -> CalibrationProposalResponse:
        return CalibrationProposalResponse(
            candidate=CalibrationCandidate(
                candidate_id=_CALIBRATION_CANDIDATE_ID,
                robot_id=request.robot_id,
                robot_asset_id=request.robot_asset_id,
                robot_digest=request.robot_asset_id.rsplit(":", 1)[-1],
                reference=request.reference,
                baseline="urdf_zero",
                joint_q={"left_shoulder_roll_joint": 1.2},
            ),
            validation=_calibration_validation(),
        )

    def validate(self, _request) -> CalibrationValidationReport:
        return _calibration_validation()

    def preview(self, _request) -> tuple[CalibrationPreview, bytes]:
        payload = b"\x89PNG\r\n\x1a\nrest-test"
        return (
            CalibrationPreview(
                candidate_id=_CALIBRATION_CANDIDATE_ID,
                sha256=hashlib.sha256(payload).hexdigest(),
                width=1200,
                height=700,
                validation=_calibration_validation(),
            ),
            payload,
        )

    def save(self, request) -> CalibrationSaveReceipt:
        digest = "d" * 64
        return CalibrationSaveReceipt(
            candidate_id=request.candidate_id,
            calibration_id=f"cal:sha256:{digest}",
            calibration_digest=digest,
            robot_id="g1_29dof",
            reference="smplx",
            save_mode=request.save_mode,
            validation=_calibration_validation(),
            visual_review=request.visual_review,
        )


class _FakeR2RCalibration:
    def status(self, request) -> R2RCalibrationStatusResponse:
        return R2RCalibrationStatusResponse(
            request_id="req_r2r_calibration_rest",
            state="missing",
            source_robot_id=request.source_robot_id,
            source_robot_asset_id=request.source_robot_asset_id,
            source_robot_digest=request.source_robot_asset_id.rsplit(":", 1)[-1],
            target_robot_id=request.target_robot_id,
            target_robot_asset_id=request.target_robot_asset_id,
            target_robot_digest=request.target_robot_asset_id.rsplit(":", 1)[-1],
            storage="none",
            joint_count=0,
            source_mapped_slots=16,
            target_mapped_slots=16,
            can_propose=True,
            can_silent_save=True,
        )

    def propose(self, request) -> R2RCalibrationProposalResponse:
        return R2RCalibrationProposalResponse(
            candidate=R2RCalibrationCandidate(
                candidate_id=_CALIBRATION_CANDIDATE_ID,
                source_robot_id=request.source_robot_id,
                source_robot_asset_id=request.source_robot_asset_id,
                source_robot_digest=request.source_robot_asset_id.rsplit(":", 1)[-1],
                target_robot_id=request.target_robot_id,
                target_robot_asset_id=request.target_robot_asset_id,
                target_robot_digest=request.target_robot_asset_id.rsplit(":", 1)[-1],
                baseline="urdf_zero",
                joint_q={"left_shoulder_roll_joint": 1.2},
            ),
            validation=_calibration_validation(),
        )

    def validate(self, _request) -> CalibrationValidationReport:
        return _calibration_validation()

    def preview(self, _request) -> tuple[R2RCalibrationPreview, bytes]:
        payload = b"\x89PNG\r\n\x1a\nr2r-rest-test"
        return (
            R2RCalibrationPreview(
                candidate_id=_CALIBRATION_CANDIDATE_ID,
                sha256=hashlib.sha256(payload).hexdigest(),
                width=1200,
                height=700,
                validation=_calibration_validation(),
            ),
            payload,
        )

    def save(self, request) -> R2RCalibrationSaveReceipt:
        digest = "e" * 64
        return R2RCalibrationSaveReceipt(
            candidate_id=request.candidate_id,
            calibration_id=f"cal:sha256:{digest}",
            calibration_digest=digest,
            source_robot_id="source_bot",
            target_robot_id="target_bot",
            save_mode=request.save_mode,
            validation=_calibration_validation(),
            visual_review=request.visual_review,
        )


def _agent_app() -> FastAPI:
    app = FastAPI()
    app.state.agent_capabilities_service = _FakeCapabilities()
    app.state.agent_asset_service = _FakeAssets()
    app.state.agent_available_asset_catalog_service = _FakeAvailableAssets()
    app.state.agent_preflight_service = _FakePreflight()
    app.state.agent_r2r_preflight_service = _FakeR2RPreflight()
    app.state.agent_batch_preflight_service = _FakeBatchPreflight()
    app.state.agent_calibration_service = _FakeCalibration()
    app.state.agent_r2r_calibration_service = _FakeR2RCalibration()
    app.include_router(router)
    return app


def _job_spec(marker: str = "1") -> JobSpecV2:
    return JobSpecV2(
        kind=JobSpecKind.RETARGET,
        plan_id=f"plan:sha256:{marker * 64}",
        inputs=[
            JobSpecInput(
                asset_id=f"asset:sha256:{'a' * 64}",
                sha256="a" * 64,
            )
        ],
        robot=JobSpecRobot(
            robot_id="g1_29dof",
            asset_id=f"asset:sha256:{'b' * 64}",
            config_sha256="b" * 64,
        ),
        calibration=None,
        backend="newton",
        effective_parameters={
            "run_mode": "smoke",
            "limit_frames": 30,
            "output_format": "csv",
        },
        output_policy=OutputPolicy.CREATE_NEW,
        provenance=JobSpecProvenance(
            hhtools_git_commit="f" * 40,
            hhtools_dirty=False,
            python="3.12",
        ),
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )


class _JobSpecs:
    def __init__(self, *specs: JobSpecV2) -> None:
        self.specs = {spec.plan_id: spec for spec in specs}

    def get_job_spec(self, plan_id: str) -> JobSpecV2:
        return JobSpecV2.model_validate_json(self.specs[plan_id].model_dump_json())


def _job_app(
    tmp_path: Path,
    *specs: JobSpecV2,
    executor: Any,
    scheduler: JobScheduler | None = None,
) -> tuple[FastAPI, JobManager, ArtifactStore, JobScheduler]:
    active_scheduler = scheduler or JobScheduler()
    job_store = JobStore(tmp_path / "agent-state")
    artifact_store = ArtifactStore(tmp_path / "agent-state")
    manager = JobManager(
        job_store,
        artifact_store,
        _JobSpecs(*specs),  # type: ignore[arg-type]
        active_scheduler,
        executor=executor,
        recover_interrupted=False,
    )
    app = FastAPI()
    app.state.agent_job_manager = manager
    app.include_router(router)
    return app, manager, artifact_store, active_scheduler


def _wait_for_http_terminal(
    client: TestClient,
    job_id: str,
    *,
    timeout: float = 3.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/agent/v1/jobs/{job_id}")
        assert response.status_code == 200
        latest = response.json()
        if latest["state"] in {"completed", "failed", "cancelled"}:
            return latest
        time.sleep(0.005)
    raise AssertionError(f"job did not become terminal: {latest}")


def test_agent_router_serializes_the_service_contract() -> None:
    response = TestClient(_agent_app()).get("/api/agent/v1/capabilities")

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": "1.0",
        "service_name": "hhtools",
        "service_version": "test",
        "agent_api_version": "v1",
        "backends": [],
        "devices": [],
        "robots": [],
        "scheduler": {
            "max_running_jobs": 0,
            "max_queued_jobs": 0,
            "running": 0,
            "queued": 0,
            "reserved": 0,
            "mode": "unlimited",
            "closed": False,
        },
        "asset_root_ids": [],
        "supported_input_formats": ["bvh"],
        "supported_output_formats": ["csv", "pkl"],
        "features": {"agent_rest": True},
    }


def test_agent_router_exposes_calibration_assistance_and_png_preview() -> None:
    client = TestClient(_agent_app())
    identity = {
        "schema_version": "1.0",
        "robot_id": "g1_29dof",
        "robot_asset_id": _ASSET_ID,
        "reference": "smplx",
    }

    status = client.post("/api/agent/v1/calibrations/status", json=identity)
    proposal = client.post("/api/agent/v1/calibrations/proposals", json=identity)
    candidate_id = proposal.json()["candidate"]["candidate_id"]
    candidate = {"schema_version": "1.0", "candidate_id": candidate_id}
    validation = client.post("/api/agent/v1/calibrations/validate", json=candidate)
    preview = client.post("/api/agent/v1/calibrations/preview", json=candidate)
    saved = client.post(
        "/api/agent/v1/calibrations/save",
        json={
            **candidate,
            "save_mode": "gpt_vision_silent",
            "visual_review": {
                "reviewer": "gpt_vision",
                "verdict": "pass",
                "model_hint": "gpt-test",
                "summary": "The front and side overlays are aligned.",
            },
        },
    )

    assert status.status_code == 200
    assert status.json()["state"] == "missing"
    assert proposal.status_code == 200
    assert validation.json()["valid"] is True
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"
    assert preview.content.startswith(b"\x89PNG")
    assert preview.headers["x-hhtools-calibration-candidate"] == candidate_id
    assert saved.status_code == 200
    assert saved.json()["saved"] is True


def test_agent_router_exposes_r2r_calibration_assistance_and_png_preview() -> None:
    client = TestClient(_agent_app())
    identity = {
        "schema_version": "1.0",
        "source_robot_id": "source_bot",
        "source_robot_asset_id": _ASSET_ID,
        "target_robot_id": "target_bot",
        "target_robot_asset_id": _R2R_TARGET_ASSET_ID,
    }

    status = client.post("/api/agent/v1/calibrations/r2r/status", json=identity)
    proposal = client.post("/api/agent/v1/calibrations/r2r/proposals", json=identity)
    candidate_id = proposal.json()["candidate"]["candidate_id"]
    candidate = {"schema_version": "1.0", "candidate_id": candidate_id}
    validation = client.post("/api/agent/v1/calibrations/r2r/validate", json=candidate)
    preview = client.post("/api/agent/v1/calibrations/r2r/preview", json=candidate)
    saved = client.post(
        "/api/agent/v1/calibrations/r2r/save",
        json={**candidate, "save_mode": "validated_silent"},
    )

    assert status.status_code == 200
    assert status.json()["state"] == "missing"
    assert proposal.status_code == 200
    assert validation.json()["valid"] is True
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"
    assert preview.content.startswith(b"\x89PNG")
    assert saved.status_code == 200
    assert saved.json()["saved"] is True
    assert preview.headers["x-hhtools-calibration-candidate"] == candidate_id


def test_agent_router_lists_available_assets_without_route_shadowing() -> None:
    client = TestClient(_agent_app())

    response = client.get(
        "/api/agent/v1/assets/available",
        params={
            "root_id": "motion-library",
            "query": "walk",
            "kind": "motion_bundle",
            "limit": 5,
            "offset": 0,
        },
    )
    unknown = client.get(
        "/api/agent/v1/assets/available",
        params={"root_id": "unknown"},
    )
    invalid = client.get(
        "/api/agent/v1/assets/available",
        params={"limit": 501},
    )

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": "1.0",
        "assets": [
            {
                "root_id": "motion-library",
                "relative_path": "AMASS/walk.npz",
                "display_name": "Walk",
                "kind": "motion_bundle",
                "category": "plain_motion",
                "recursive": True,
                "dataset": "amass",
                "reference": "smpl",
            }
        ],
        "total": 1,
        "limit": 5,
        "offset": 0,
    }
    assert unknown.status_code == 403
    assert unknown.json()["code"] == "ASSET_OUTSIDE_ALLOWED_ROOT"
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "INVALID_PARAMETER"


def test_agent_router_rejects_duplicate_keys_and_nonfinite_json() -> None:
    client = TestClient(_agent_app())

    duplicate = client.post(
        "/api/agent/v1/assets",
        content=(
            b'{"root_id":"motion-library","root_id":"motion-library","relative_path":"walk.npz"}'
        ),
        headers={"Content-Type": "application/json"},
    )
    nonfinite = client.post(
        "/api/agent/v1/assets",
        content=(b'{"root_id":"motion-library","relative_path":"walk.npz","recursive":NaN}'),
        headers={"Content-Type": "application/json"},
    )
    too_deep = client.post(
        "/api/agent/v1/assets",
        content=(b"[" * 5_000) + b"0" + (b"]" * 5_000),
        headers={"Content-Type": "application/json"},
    )
    overflowed_float = client.post(
        "/api/agent/v1/assets",
        content=(b'{"root_id":"motion-library","relative_path":"walk.npz","recursive":1e9999}'),
        headers={"Content-Type": "application/json"},
    )

    assert duplicate.status_code == 400
    assert duplicate.json()["code"] == "INVALID_JSON"
    assert nonfinite.status_code == 400
    assert nonfinite.json()["code"] == "INVALID_JSON"
    assert too_deep.status_code == 400
    assert too_deep.json()["code"] == "INVALID_JSON"
    assert overflowed_float.status_code == 400
    assert overflowed_float.json()["code"] == "INVALID_JSON"
    assert "detail" not in duplicate.json()
    assert duplicate.headers["cache-control"] == "no-store"


def test_agent_router_blocks_host_paths_in_success_and_error_json() -> None:
    class UnsafeSuccessAssets(_FakeAssets):
        def get(self, asset_id: str) -> AssetBundle:
            bundle = super().get(asset_id)
            return bundle.model_copy(
                update={"metadata": {"debug_path": "debug:C:\\Users\\Nora\\secret"}}
            )

    class UnsafeErrorAssets(_FakeAssets):
        def get(self, asset_id: str) -> AssetBundle:
            raise AssetServiceError(
                ApiError(
                    code="ASSET_NOT_FOUND",
                    message="debug:/srv/secret",
                    stage=ErrorStage.ASSET_REGISTRATION,
                    details={
                        "debug_path": "C:\\private\\asset.bin",
                        "url": "/srv/secret",
                    },
                )
            )

    class UnsafeFileUriAssets(_FakeAssets):
        def get(self, asset_id: str) -> AssetBundle:
            bundle = super().get(asset_id)
            return bundle.model_copy(update={"metadata": {"debug_uri": "file:///etc/passwd"}})

    class UnsafeEmbeddedFileUriAssets(_FakeAssets):
        def get(self, asset_id: str) -> AssetBundle:
            raise AssetServiceError(
                ApiError(
                    code="ASSET_NOT_FOUND",
                    message="debug file:///etc/passwd",
                    stage=ErrorStage.ASSET_REGISTRATION,
                )
            )

    success_app = _agent_app()
    success_app.state.agent_asset_service = UnsafeSuccessAssets()
    error_app = _agent_app()
    error_app.state.agent_asset_service = UnsafeErrorAssets()
    file_uri_app = _agent_app()
    file_uri_app.state.agent_asset_service = UnsafeFileUriAssets()
    embedded_file_uri_app = _agent_app()
    embedded_file_uri_app.state.agent_asset_service = UnsafeEmbeddedFileUriAssets()

    blocked_success = TestClient(success_app).get(f"/api/agent/v1/assets/{_ASSET_ID}")
    blocked_error = TestClient(error_app).get(f"/api/agent/v1/assets/{_ASSET_ID}")
    blocked_file_uri = TestClient(file_uri_app).get(f"/api/agent/v1/assets/{_ASSET_ID}")
    blocked_embedded_file_uri = TestClient(embedded_file_uri_app).get(
        f"/api/agent/v1/assets/{_ASSET_ID}"
    )

    assert blocked_success.status_code == 500
    assert blocked_success.json()["code"] == "INTERNAL_ERROR"
    assert blocked_error.status_code == 500
    assert blocked_error.json()["code"] == "INTERNAL_ERROR"
    assert blocked_file_uri.status_code == 500
    assert blocked_file_uri.json()["code"] == "INTERNAL_ERROR"
    assert blocked_embedded_file_uri.status_code == 500
    assert blocked_embedded_file_uri.json()["code"] == "INTERNAL_ERROR"
    assert "C:\\Users\\Nora\\secret" not in blocked_success.text
    assert "C:\\private" not in blocked_error.text
    assert "/srv/secret" not in blocked_error.text
    assert "file:///etc/passwd" not in blocked_file_uri.text
    assert "file:///etc/passwd" not in blocked_embedded_file_uri.text
    assert _looks_like_host_path("debug:C:\\Users\\Nora\\secret") is True
    assert _looks_like_host_path("debug:/srv/secret") is True
    assert _looks_like_host_path("https://example.com/path") is False
    assert _looks_like_host_path("hhtools://jobs/job:1/artifacts/report") is False


@pytest.mark.parametrize(
    "value",
    [
        "debug:C:\\Users\\Nora\\secret",
        "path[C:\\Users\\Nora\\secret]",
        "path{C:\\Users\\Nora\\secret}",
        "path|C:\\Users\\Nora\\secret",
        "path[\\\\server\\share\\secret]",
        "debug:/srv/secret",
        "path[/srv/secret]",
        "path{/srv/secret}",
        "path|/srv/secret",
        "path[//remote/share]",
        "prefix%2Fetc%2Fpasswd",
        "https://example.com/report?path=%252Fetc%252Fpasswd",
        "https://prefixC:%5CUsers%5CNora%5Csecret@example.com",
        "https://example.com/path]C:\\Users\\Nora\\secret",
        "hhtools://jobs/job:1/artifacts/report|/srv/secret",
    ],
)
def test_agent_portable_guard_detects_paths_after_punctuation(value: str) -> None:
    assert _looks_like_host_path(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "https://example.com/path",
        "hhtools://jobs/job:1/artifacts/report",
        "debug:https://example.com/path",
        "path[https://example.com/path]",
        "path{hhtools://jobs/job:1/artifacts/report}",
        "https://[2001:db8::1]/docs",
        "https://example.com/callback?next=/agent/v1",
        "https://example.com/docs?next=https%3A%2F%2Fdocs.example%2Fguide",
    ],
)
def test_agent_portable_guard_preserves_allowlisted_uri_fragments(value: str) -> None:
    assert _looks_like_host_path(value) is False


def test_agent_queue_full_error_emits_retry_after_header() -> None:
    error = ApiError(
        code="QUEUE_FULL",
        message="The Agent queue is full.",
        retryable=True,
        stage=ErrorStage.ADMISSION,
        next_action=NextAction(
            actor="agent",
            action="poll_admission",
            parameters={"poll_after_ms": 1_501},
        ),
    )

    response = _error_response(error)
    assert error.next_action is not None
    oversized = _error_response(
        error.model_copy(
            update={
                "next_action": error.next_action.model_copy(
                    update={"parameters": {"poll_after_ms": 1e300}}
                )
            }
        )
    )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "2"
    assert response.headers["cache-control"] == "no-store"
    assert "retry-after" not in oversized.headers


class _SwitchAfterFirstOpenPath:
    """Serve the replacement if production ever reopens after verification."""

    def __init__(self, original: Path, replacement: Path) -> None:
        self.current = original
        self.replacement = replacement
        self.open_calls = 0

    def open(self, mode: str):
        self.open_calls += 1
        handle = self.current.open(mode)
        self.current = self.replacement
        return handle


class _SingleArtifactManager:
    def __init__(self, stored: StoredArtifact) -> None:
        self.stored = stored
        self.lookups = 0

    def get_artifact(
        self,
        job_id: str,
        artifact_id: str,
        *,
        verify: bool = False,
    ) -> StoredArtifact:
        self.lookups += 1
        assert job_id == self.stored.descriptor.job_id
        assert artifact_id == self.stored.descriptor.artifact_id
        assert verify is False
        return self.stored

    def start_retarget(self, *args, **kwargs):
        raise AssertionError("not called")

    def get_job(self, *args, **kwargs):
        raise AssertionError("not called")

    def wait_job(self, *args, **kwargs):
        raise AssertionError("not called")

    def lookup_job(self, *args, **kwargs):
        raise AssertionError("not called")

    def cancel_job(self, *args, **kwargs):
        raise AssertionError("not called")

    def retry_job(self, *args, **kwargs):
        raise AssertionError("not called")

    def list_artifacts(self, *args, **kwargs):
        raise AssertionError("not called")


def test_agent_artifact_streams_the_verified_open_handle_with_strong_headers(
    tmp_path: Path,
) -> None:
    original_bytes = b"canonical artifact bytes\n"
    original = tmp_path / "canonical.csv"
    replacement = tmp_path / "private.csv"
    original.write_bytes(original_bytes)
    replacement.write_bytes(b"must never be served")
    switch_path = _SwitchAfterFirstOpenPath(original, replacement)
    descriptor = ArtifactDescriptor(
        artifact_id="artifact:retargeted_motion:one-open",
        job_id="job:one-open",
        kind="retargeted_motion",
        format="csv",
        resource_uri="hhtools://jobs/job:one-open/artifacts/one-open",
        media_type="text/csv",
        size_bytes=len(original_bytes),
        sha256=hashlib.sha256(original_bytes).hexdigest(),
    )
    manager = _SingleArtifactManager(
        StoredArtifact(descriptor=descriptor, path=switch_path)  # type: ignore[arg-type]
    )
    app = FastAPI()
    app.state.agent_job_manager = manager
    app.include_router(router)

    with TestClient(app) as client:
        response = client.get(
            f"/api/agent/v1/jobs/{descriptor.job_id}/artifacts/{descriptor.artifact_id}/content"
        )
        ranged = client.get(
            f"/api/agent/v1/jobs/{descriptor.job_id}/artifacts/{descriptor.artifact_id}/content",
            headers={"Range": "bytes=0-3"},
        )

    assert response.status_code == 200
    assert response.content == original_bytes
    assert switch_path.open_calls == 1
    assert manager.lookups == 2  # canonical authorization precedes Range rejection
    assert response.headers["content-length"] == str(len(original_bytes))
    assert response.headers["x-content-sha256"] == descriptor.sha256
    assert response.headers["content-digest"].startswith("sha-256=:")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-disposition"] == (
        'attachment; filename="retargeted_motion.csv"'
    )
    assert "accept-ranges" not in response.headers
    assert str(tmp_path) not in str(response.headers)
    assert ranged.status_code == 416
    assert ranged.json()["code"] == "RANGE_NOT_SUPPORTED"


def test_agent_artifact_rejects_unsafe_descriptor_headers(tmp_path: Path) -> None:
    payload = b"safe bytes"
    path = tmp_path / "artifact.bin"
    path.write_bytes(payload)
    descriptor = ArtifactDescriptor(
        artifact_id="artifact:report:unsafe-mime",
        job_id="job:unsafe-mime",
        kind="report",
        format="bin",
        resource_uri="hhtools://jobs/job:unsafe-mime/artifacts/unsafe-mime",
        media_type="application/octet-stream",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    ).model_copy(update={"media_type": "text/plain\r\nX-Leak: yes"})
    manager = _SingleArtifactManager(StoredArtifact(descriptor=descriptor, path=path))
    app = FastAPI()
    app.state.agent_job_manager = manager
    app.include_router(router)

    response = TestClient(app).get(
        f"/api/agent/v1/jobs/{descriptor.job_id}/artifacts/{descriptor.artifact_id}/content"
    )

    assert response.status_code == 500
    assert response.json()["code"] == "INTERNAL_ERROR"
    assert "x-leak" not in response.headers


@pytest.mark.parametrize("kind", ["job_spec", "preview", "evaluation_report", "manifest"])
@pytest.mark.parametrize("media_type", ["application/json", "Application/Problem+JSON"])
@pytest.mark.parametrize("tampered", [False, True])
def test_agent_json_artifact_downloads_exact_verified_bytes(
    tmp_path: Path,
    kind: str,
    media_type: str,
    tampered: bool,
) -> None:
    payload = b'{ "schema_version": "1.0", "status": "ok" }\n'
    original = tmp_path / "original.json"
    original.write_bytes(payload)
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b'{"private":"must never be served"}')
    switch_path = _SwitchAfterFirstOpenPath(original, replacement)
    descriptor = ArtifactDescriptor(
        artifact_id=f"artifact:{kind}:json-download",
        job_id="job:json-download",
        kind=kind,
        format="json",
        media_type=media_type,
        resource_uri=f"hhtools://jobs/job:json-download/artifacts/artifact:{kind}:json-download",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    app = FastAPI()
    app.state.agent_job_manager = _SingleArtifactManager(
        StoredArtifact(descriptor=descriptor, path=switch_path)  # type: ignore[arg-type]
    )
    app.include_router(router)
    if tampered:
        original.write_bytes(payload + b" ")
    with TestClient(app) as client:
        response = client.get(
            f"/api/agent/v1/jobs/{descriptor.job_id}/artifacts/{descriptor.artifact_id}/content"
        )
    if tampered:
        assert response.status_code == 409
        assert response.json()["code"] == "ARTIFACT_HASH_MISMATCH"
        return
    assert response.status_code == 200, response.text
    assert response.content == payload
    assert switch_path.open_calls == 1
    assert response.headers["x-content-sha256"] == hashlib.sha256(response.content).hexdigest()
    assert response.headers["content-length"] == str(len(payload))
    assert response.headers["content-disposition"] == f'attachment; filename="{kind}.json"'
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize(
    "payload",
    [
        b'{"value":NaN}',
        b'{"value":1,"value":2}',
        b'{"value":"/home/private/file"}',
        b'{"secret":"private"}',
        b"not json",
    ],
)
def test_agent_json_artifacts_still_enforce_content_validation(
    tmp_path: Path,
    payload: bytes,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(payload)
    descriptor = ArtifactDescriptor(
        artifact_id="artifact:manifest:invalid-json",
        job_id="job:invalid-json",
        kind="manifest",
        format="json",
        media_type="application/json",
        resource_uri="hhtools://jobs/job:invalid-json/artifacts/artifact:manifest:invalid-json",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    app = FastAPI()
    app.state.agent_job_manager = _SingleArtifactManager(
        StoredArtifact(descriptor=descriptor, path=path)
    )
    app.include_router(router)
    with TestClient(app) as client:
        response = client.get(
            f"/api/agent/v1/jobs/{descriptor.job_id}/artifacts/{descriptor.artifact_id}/content"
        )
    assert response.status_code == 500
    assert response.json()["code"] == "INTERNAL_ERROR"
    assert "private" not in response.text


def test_agent_asset_routes_serialize_versioned_contracts() -> None:
    client = TestClient(_agent_app())

    registered = client.post(
        "/api/agent/v1/assets",
        json={
            "schema_version": "1.0",
            "root_id": "motion-library",
            "relative_path": "walk.npz",
        },
    )
    searched = client.get("/api/agent/v1/assets?category=plain_motion&limit=20")
    fetched = client.get(f"/api/agent/v1/assets/{_ASSET_ID}")
    inspected = client.get(f"/api/agent/v1/assets/{_ASSET_ID}/inspect")

    assert registered.status_code == 201
    assert registered.json()["asset_id"] == _ASSET_ID
    assert searched.status_code == 200
    assert searched.json()["assets"][0]["schema_version"] == "1.0"
    assert searched.json()["limit"] == 20
    assert fetched.json()["detected"]["recommended_backend"] == "newton"
    assert inspected.status_code == 200
    assert inspected.json()["status"] == "valid"
    assert inspected.json()["frame_count"] == 3


def test_agent_asset_routes_use_structured_errors_for_input_and_service_failures() -> None:
    client = TestClient(_agent_app())

    invalid = client.post(
        "/api/agent/v1/assets",
        json={"root_id": "motion-library", "relative_path": "../secret.npz"},
    )
    missing = client.get(f"/api/agent/v1/assets/asset:sha256:{'0' * 64}")

    assert invalid.status_code == 422
    assert invalid.json()["schema_version"] == "1.0"
    assert invalid.json()["code"] == "INVALID_PARAMETER"
    assert invalid.json()["stage"] == "request"
    assert missing.status_code == 404
    assert missing.json()["code"] == "ASSET_NOT_FOUND"
    assert "detail" not in missing.json()


def test_agent_preflight_route_returns_a_business_evaluation_contract() -> None:
    client = TestClient(_agent_app())

    response = client.post(
        "/api/agent/v1/preflight/retarget",
        json={
            "motion_asset_id": _ASSET_ID,
            "robot_id": "test_robot",
        },
    )

    assert response.status_code == 200
    assert response.json()["schema_version"] == "1.0"
    assert response.json()["status"] == "rejected"
    assert response.json()["error"]["code"] == "ROBOT_ASSET_REQUIRED"
    assert "detail" not in response.json()


def test_agent_r2r_preflight_route_uses_the_versioned_pair_contract() -> None:
    response = TestClient(_agent_app()).post(
        "/api/agent/v1/preflight/r2r",
        json={
            "trajectory_asset_id": _ASSET_ID,
            "source_robot_id": "source_bot",
            "source_robot_asset_id": f"asset:sha256:{'b' * 64}",
            "target_robot_id": "target_bot",
            "target_robot_asset_id": f"asset:sha256:{'c' * 64}",
        },
    )

    assert response.status_code == 200
    assert response.json()["schema_version"] == "1.0"
    assert response.json()["status"] == "rejected"
    assert response.json()["recommended_backend"] == "newton"
    assert response.json()["error"]["code"] == "R2R_CALIBRATION_REQUIRED"


def test_agent_batch_preflight_route_uses_ordered_child_plan_contract() -> None:
    response = TestClient(_agent_app()).post(
        "/api/agent/v1/preflight/batch",
        json={
            "workflow": "h2r",
            "item_plan_ids": [f"plan:sha256:{'a' * 64}"],
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert response.json()["error"]["code"] == "PLAN_NOT_FOUND"


def test_agent_job_rest_lifecycle_idempotency_retry_and_canonical_artifacts(
    tmp_path: Path,
) -> None:
    first = _job_spec("1")
    second = _job_spec("2")
    execution_calls: list[str] = []

    def execute(spec: JobSpecV2, context: JobExecutionContext) -> JobExecutionResult:
        execution_calls.append(spec.plan_id)
        context.publish_bytes(
            kind="retargeted_motion",
            payload=b"frame,joint\n0,0.0\n",
            format="csv",
            media_type="text/csv",
        )
        return JobExecutionResult(
            outcome=JobOutcome.SUCCESS,
            summary={"num_frames": 1},
        )

    app, _manager, artifact_store, scheduler = _job_app(
        tmp_path,
        first,
        second,
        executor=execute,
    )
    try:
        with TestClient(app) as client:
            submitted = client.post(
                "/api/agent/v1/jobs",
                json={
                    "schema_version": "1.0",
                    "plan_id": first.plan_id,
                    "idempotency_key": "rest-root-1",
                },
            )
            repeated = client.post(
                "/api/agent/v1/jobs",
                json={
                    "schema_version": "1.0",
                    "plan_id": first.plan_id,
                    "idempotency_key": "rest-root-1",
                },
            )

            assert submitted.status_code == 202
            assert repeated.status_code == 202
            root_job_id = submitted.json()["job_id"]
            assert repeated.json()["job_id"] == root_job_id
            root = _wait_for_http_terminal(client, root_job_id)
            assert root["state"] == "completed"
            assert root["outcome"] == "success"
            assert execution_calls == [first.plan_id]

            unchanged = client.get(
                f"/api/agent/v1/jobs/{root_job_id}",
                params={"after_revision": root["progress"]["revision"]},
            )
            assert unchanged.status_code == 200
            assert unchanged.json()["progress"]["revision"] == root["progress"]["revision"]

            recovered = client.post(
                "/api/agent/v1/jobs/lookup",
                json={
                    "schema_version": "1.0",
                    "plan_id": first.plan_id,
                    "idempotency_key": "rest-root-1",
                    "after_revision": root["progress"]["revision"],
                },
            )
            assert recovered.status_code == 200
            assert recovered.json()["job_id"] == root_job_id

            lookup_conflict = client.post(
                "/api/agent/v1/jobs/lookup",
                json={
                    "plan_id": second.plan_id,
                    "idempotency_key": "rest-root-1",
                },
            )
            assert lookup_conflict.status_code == 409
            assert lookup_conflict.json()["code"] == "JOB_CONFLICT"

            conflict = client.post(
                "/api/agent/v1/jobs",
                json={
                    "plan_id": second.plan_id,
                    "idempotency_key": "rest-root-1",
                },
            )
            assert conflict.status_code == 409
            assert conflict.json()["code"] == "JOB_CONFLICT"

            artifact_page = client.get(
                f"/api/agent/v1/jobs/{root_job_id}/artifacts",
                params={"limit": 2},
            )
            assert artifact_page.status_code == 200
            assert artifact_page.json()["job_id"] == root_job_id
            assert artifact_page.json()["total"] == 4
            assert len(artifact_page.json()["artifacts"]) == 2

            motion = next(item for item in root["artifacts"] if item["kind"] == "retargeted_motion")
            descriptor = client.get(
                f"/api/agent/v1/jobs/{root_job_id}/artifacts/{motion['artifact_id']}"
            )
            content = client.get(
                f"/api/agent/v1/jobs/{root_job_id}/artifacts/{motion['artifact_id']}/content",
                params={"verify": "true"},
            )
            assert descriptor.status_code == 200
            assert descriptor.json() == motion
            assert content.status_code == 200
            assert content.content == b"frame,joint\n0,0.0\n"
            assert content.headers["x-content-sha256"] == motion["sha256"]
            assert "retargeted_motion.csv" in content.headers["content-disposition"]
            assert str(tmp_path) not in descriptor.text
            assert str(tmp_path) not in str(content.headers)

            # Raw ArtifactStore candidates are not canonical job membership and
            # must remain indistinguishable from a nonexistent artifact.
            orphan = artifact_store.put_bytes(
                job_id=root_job_id,
                kind="log_tail",
                payload=b"orphan candidate",
                format="txt",
                media_type="text/plain",
            )
            denied_orphan = client.get(
                f"/api/agent/v1/jobs/{root_job_id}/artifacts/{orphan.artifact_id}"
            )
            assert denied_orphan.status_code == 404
            assert denied_orphan.json()["code"] == "ARTIFACT_NOT_FOUND"

            retried = client.post(
                f"/api/agent/v1/jobs/{root_job_id}/retry",
                json={"idempotency_key": "rest-child-1"},
            )
            assert retried.status_code == 202
            child_job_id = retried.json()["job_id"]
            child = _wait_for_http_terminal(client, child_job_id)
            assert child["parent_job_id"] == root_job_id
            assert child["root_job_id"] == root_job_id
            assert child["attempt"] == 2
            repeated_retry = client.post(
                f"/api/agent/v1/jobs/{root_job_id}/retry",
                json={"idempotency_key": "rest-child-1"},
            )
            assert repeated_retry.status_code == 202
            assert repeated_retry.json()["job_id"] == child_job_id
            assert execution_calls == [first.plan_id, first.plan_id]

            child_motion = next(
                item for item in child["artifacts"] if item["kind"] == "retargeted_motion"
            )
            denied_cross_job = client.get(
                f"/api/agent/v1/jobs/{root_job_id}/artifacts/{child_motion['artifact_id']}"
            )
            assert denied_cross_job.status_code == 404
            assert denied_cross_job.json()["code"] == "ARTIFACT_NOT_FOUND"

            terminal_cancel = client.post(f"/api/agent/v1/jobs/{root_job_id}/cancel")
            assert terminal_cancel.status_code == 409
            assert terminal_cancel.json()["code"] == "JOB_CANCEL_UNSUPPORTED"

            managed_motion = artifact_store.get(motion["artifact_id"])
            managed_motion.path.write_bytes(b"corrupted")
            verified_descriptor = client.get(
                f"/api/agent/v1/jobs/{root_job_id}/artifacts/{motion['artifact_id']}",
                params={"verify": "true"},
            )
            assert verified_descriptor.status_code == 409
            assert verified_descriptor.json()["code"] == "ARTIFACT_HASH_MISMATCH"
    finally:
        assert scheduler.shutdown(wait=True, timeout=3.0)


def test_agent_job_wait_blocks_for_revision_and_validates_timeout(
    tmp_path: Path,
) -> None:
    spec = _job_spec("4")
    started = threading.Event()
    publish_progress = threading.Event()
    release = threading.Event()

    def execute(_spec: JobSpecV2, context: JobExecutionContext) -> JobExecutionResult:
        started.set()
        assert publish_progress.wait(timeout=3.0)
        context.report_progress(
            phase="ik_solve",
            fraction=0.5,
            message="Halfway",
        )
        assert release.wait(timeout=3.0)
        return JobExecutionResult(outcome=JobOutcome.SUCCESS)

    app, manager, _artifact_store, scheduler = _job_app(
        tmp_path,
        spec,
        executor=execute,
    )
    wait_entered = threading.Event()
    original_wait = manager.wait_job

    def observed_wait(job_id: str, *, after_revision: int, timeout: float = 30.0):
        wait_entered.set()
        return original_wait(
            job_id,
            after_revision=after_revision,
            timeout=timeout,
        )

    manager.wait_job = observed_wait  # type: ignore[method-assign]
    try:
        with TestClient(app) as client:
            submitted = client.post(
                "/api/agent/v1/jobs",
                json={"plan_id": spec.plan_id, "idempotency_key": "rest-wait-1"},
            )
            job_id = submitted.json()["job_id"]
            assert started.wait(timeout=2.0)
            current = client.get(f"/api/agent/v1/jobs/{job_id}").json()

            with ThreadPoolExecutor(max_workers=1) as executor:
                waiting = executor.submit(
                    client.get,
                    f"/api/agent/v1/jobs/{job_id}/wait",
                    params={
                        "after_revision": current["progress"]["revision"],
                        "timeout": 1.0,
                    },
                )
                assert wait_entered.wait(timeout=1.0)
                publish_progress.set()
                changed = waiting.result(timeout=2.0)

            assert changed.status_code == 200
            changed_document = changed.json()
            assert changed_document["progress"]["revision"] > current["progress"]["revision"]
            assert changed_document["progress"]["message"] == "Halfway"

            unchanged = client.get(
                f"/api/agent/v1/jobs/{job_id}/wait",
                params={
                    "after_revision": changed_document["progress"]["revision"],
                    "timeout": 0,
                },
            )
            assert unchanged.status_code == 200
            assert (
                unchanged.json()["progress"]["revision"] == changed_document["progress"]["revision"]
            )

            invalid_timeout = client.get(
                f"/api/agent/v1/jobs/{job_id}/wait",
                params={"after_revision": 0, "timeout": 60.1},
            )
            assert invalid_timeout.status_code == 422
            assert invalid_timeout.json()["code"] == "INVALID_PARAMETER"

            release.set()
            terminal = client.get(
                f"/api/agent/v1/jobs/{job_id}/wait",
                params={
                    "after_revision": changed_document["progress"]["revision"],
                    "timeout": 2.0,
                },
            )
            assert terminal.status_code == 200
            assert terminal.json()["state"] == "completed"
    finally:
        release.set()
        publish_progress.set()
        assert scheduler.shutdown(wait=True, timeout=3.0)


def test_agent_running_cancel_is_cooperative_and_missing_jobs_are_structured(
    tmp_path: Path,
) -> None:
    spec = _job_spec("3")
    started = threading.Event()

    def execute(_spec: JobSpecV2, context: JobExecutionContext) -> JobExecutionResult:
        started.set()
        while not context.cancellation_requested:
            time.sleep(0.005)
        context.raise_if_cancelled()
        raise AssertionError("cancel acknowledgement must stop execution")

    app, _manager, _artifact_store, scheduler = _job_app(
        tmp_path,
        spec,
        executor=execute,
    )
    try:
        with TestClient(app) as client:
            submitted = client.post(
                "/api/agent/v1/jobs",
                json={
                    "plan_id": spec.plan_id,
                    "idempotency_key": "rest-cancel-1",
                },
            )
            job_id = submitted.json()["job_id"]
            assert started.wait(timeout=2.0)

            active_retry = client.post(
                f"/api/agent/v1/jobs/{job_id}/retry",
                json={"idempotency_key": "too-early"},
            )
            assert active_retry.status_code == 409
            assert active_retry.json()["code"] == "INVALID_JOB_TRANSITION"

            cancellation = client.post(f"/api/agent/v1/jobs/{job_id}/cancel")
            assert cancellation.status_code == 200
            assert cancellation.json()["cancellation_requested"] is True
            cancelled = _wait_for_http_terminal(client, job_id)
            assert cancelled["state"] == "cancelled"
            assert cancelled["cancellation_requested"] is True

            repeated = client.post(f"/api/agent/v1/jobs/{job_id}/cancel")
            assert repeated.status_code == 200
            assert repeated.json()["state"] == "cancelled"

            missing = client.get("/api/agent/v1/jobs/job:does-not-exist")
            assert missing.status_code == 404
            assert missing.json()["code"] == "JOB_NOT_FOUND"
            assert "detail" not in missing.json()
    finally:
        assert scheduler.shutdown(wait=True, timeout=3.0)


class _FakeLegacyUpgrade:
    def __init__(self, response: LegacyJobUpgradeResponse) -> None:
        self.response = response
        self.payloads: list[dict[str, Any]] = []

    def upgrade(self, payload: Any) -> LegacyJobUpgradeResponse:
        assert isinstance(payload, dict)
        self.payloads.append(payload)
        if payload.get("kind") == "unsupported":
            raise LegacyJobUpgradeError(
                ApiError(
                    code="INVALID_JOB_SPEC",
                    message="Only single H2R JobSpec v1 can be upgraded.",
                    stage=ErrorStage.REQUEST,
                )
            )
        return self.response


def test_agent_legacy_upgrade_is_a_thin_versioned_adapter() -> None:
    spec = _job_spec("4")
    plan = RetargetPlan(
        plan_id=spec.plan_id,
        created_at=spec.created_at,
        motion_asset_id=spec.inputs[0].asset_id,
        robot_id=spec.robot.robot_id,
        robot_asset_id=spec.robot.asset_id,
        backend=spec.backend,
        output_format="csv",
        output_policy=OutputPolicy.CREATE_NEW,
        parameters=dict(spec.effective_parameters),
        input_digest=spec.inputs[0].sha256,
        robot_digest=spec.robot.config_sha256,
    )
    upgrade_response = LegacyJobUpgradeResponse(
        preflight=PreflightResponse(
            request_id="req_upgrade_rest",
            status="ready",
            plan=plan,
            recommended_backend="newton",
        ),
        job_spec=spec,
        receipt=LegacyMigrationReceipt(
            canonical_v1_sha256="c" * 64,
            motion_asset_id=spec.inputs[0].asset_id,
            robot_asset_id=spec.robot.asset_id,
            plan_id=spec.plan_id,
            job_spec_sha256="d" * 64,
        ),
    )
    service = _FakeLegacyUpgrade(upgrade_response)
    app = FastAPI()
    app.state.agent_legacy_job_upgrade_service = service
    app.include_router(router)
    client = TestClient(app)
    legacy_payload = {
        "schema_version": 1,
        "kind": "retarget",
        "request": {
            "source_path": "/srv/hhtools/motions/walk.bvh",
            "robot": "g1_29dof",
        },
    }
    unsupported_payload = {
        "schema_version": 1,
        "kind": "unsupported",
        "request": {},
    }

    upgraded = client.post(
        "/api/agent/v1/legacy/jobspec-v1/upgrade",
        json={"schema_version": "1.0", "payload": legacy_payload},
    )
    rejected = client.post(
        "/api/agent/v1/legacy/jobspec-v1/upgrade",
        json={"payload": unsupported_payload},
    )

    assert upgraded.status_code == 200
    assert upgraded.json() == upgrade_response.model_dump(mode="json", exclude_none=True)
    assert service.payloads == [legacy_payload, unsupported_payload]
    assert rejected.status_code == 400
    assert rejected.json()["code"] == "INVALID_JOB_SPEC"
    assert "detail" not in rejected.json()
    assert "/srv/hhtools" not in upgraded.text


def test_agent_phase5_routes_and_examples_are_visible_in_openapi() -> None:
    schema = TestClient(_agent_app()).get("/openapi.json").json()
    expected_paths = {
        "/api/agent/v1/preflight/r2r",
        "/api/agent/v1/preflight/batch",
        "/api/agent/v1/calibrations/status",
        "/api/agent/v1/calibrations/proposals",
        "/api/agent/v1/calibrations/validate",
        "/api/agent/v1/calibrations/preview",
        "/api/agent/v1/calibrations/save",
        "/api/agent/v1/calibrations/r2r/status",
        "/api/agent/v1/calibrations/r2r/proposals",
        "/api/agent/v1/calibrations/r2r/validate",
        "/api/agent/v1/calibrations/r2r/preview",
        "/api/agent/v1/calibrations/r2r/save",
        "/api/agent/v1/jobs",
        "/api/agent/v1/jobs/lookup",
        "/api/agent/v1/jobs/{job_id}",
        "/api/agent/v1/jobs/{job_id}/cancel",
        "/api/agent/v1/jobs/{job_id}/retry",
        "/api/agent/v1/jobs/{job_id}/artifacts",
        "/api/agent/v1/jobs/{job_id}/artifacts/{artifact_id}",
        "/api/agent/v1/jobs/{job_id}/artifacts/{artifact_id}/content",
        "/api/agent/v1/legacy/jobspec-v1/upgrade",
    }
    assert expected_paths <= set(schema["paths"])

    for path, method in (
        ("/api/agent/v1/jobs", "post"),
        ("/api/agent/v1/jobs/lookup", "post"),
        ("/api/agent/v1/jobs/{job_id}/retry", "post"),
        ("/api/agent/v1/legacy/jobspec-v1/upgrade", "post"),
    ):
        media = schema["paths"][path][method]["requestBody"]["content"]["application/json"]
        assert media["examples"]


def test_full_web_app_registers_agent_api_before_the_static_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def local_tmpdir(tag: str) -> Path:
        path = tmp_path / f"runtime-{tag}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(server_state, "_tmpdir", local_tmpdir)
    monkeypatch.setattr(server_state, "_robot_library_root", lambda: tmp_path / "robots")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv(
        "HHTOOLS_MOTION_LIBRARY_SETTINGS_PATH",
        str(tmp_path / "motion-library-settings.json"),
    )
    clear_cache()
    source_root = tmp_path / "motions"
    source_root.mkdir()
    robot_root = tmp_path / "robots" / "catalog_bot"
    robot_root.mkdir(parents=True)
    (robot_root / "robot.yaml").write_text(
        "name: catalog_bot\n"
        "display_name: Catalog Bot\n"
        "urdf: robot.urdf\n"
        "dof_order: [hip]\n"
        "ik_map:\n"
        "  hips: base\n",
        encoding="utf-8",
    )
    (robot_root / "robot.urdf").write_text(
        """<?xml version="1.0"?>
<robot name="catalog_bot">
  <link name="base"/>
  <link name="torso"/>
  <joint name="hip" type="revolute">
    <parent link="base"/>
    <child link="torso"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="2"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    positions = np.zeros((2, 1, 3), dtype=np.float32)
    quaternions = np.zeros((2, 1, 4), dtype=np.float32)
    quaternions[..., 3] = 1.0
    np.savez(
        source_root / "walk.npz",
        schema_version=np.array("1"),
        name=np.array("walk"),
        framerate=np.array(30.0),
        up_axis=np.array("Z"),
        bone_names=np.array(["root"]),
        parent_indices=np.array([-1], dtype=np.int32),
        positions=positions,
        quaternions=quaternions,
    )
    (source_root / "robot.csv").write_text(
        "root_x,root_y,root_z,root_qx,root_qy,root_qz,root_qw,dof_hip\n0,0,0,0,0,0,1,0\n",
        encoding="utf-8",
    )
    np.savez(source_root / "robot-trajectory.npz", joint_q=np.zeros((2, 8)))
    escaped_clip = source_root / "parc_ms" / "escaped" / "escaped.npz"
    escaped_clip.parent.mkdir(parents=True)
    np.savez(
        escaped_clip,
        schema_version=np.array("1"),
        name=np.array("escaped"),
        framerate=np.array(30.0),
        up_axis=np.array("Z"),
        bone_names=np.array(["root"]),
        parent_indices=np.array([-1], dtype=np.int32),
        positions=positions,
        quaternions=quaternions,
    )
    outside_terrain = tmp_path / "private-terrain.obj"
    outside_terrain.write_text("o private\n", encoding="utf-8")
    (escaped_clip.parent / "escaped_terrain.obj").symlink_to(outside_terrain)
    (source_root / "malformed.npz").write_bytes(b"not a zip archive")
    app = server.create_app(
        source_root=source_root,
        save_dir=tmp_path / "save",
        cache_dir=tmp_path / "cache",
        job_history_dir=tmp_path / "history",
        job_settings_path=tmp_path / "job-settings.json",
    )
    boundary_payload = b"boundary streaming stays live\n"
    boundary_artifact_path = tmp_path / "boundary-stream.bin"
    boundary_artifact_path.write_bytes(boundary_payload)
    boundary_descriptor = ArtifactDescriptor(
        artifact_id="artifact:report:boundary-stream",
        job_id="job:boundary-stream",
        kind="report",
        format="bin",
        resource_uri=("hhtools://jobs/job:boundary-stream/artifacts/boundary-stream"),
        media_type="application/octet-stream",
        size_bytes=len(boundary_payload),
        sha256=hashlib.sha256(boundary_payload).hexdigest(),
    )
    boundary_manager = _SingleArtifactManager(
        StoredArtifact(
            descriptor=boundary_descriptor,
            path=boundary_artifact_path,
        )
    )
    production_job_manager = app.state.agent_job_manager

    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50_000),
    ) as client:
        response = client.get("/api/agent/v1/capabilities")
        agent_missing_route = client.get("/api/agent/v1/does-not-exist")
        agent_wrong_method = client.delete("/api/agent/v1/capabilities")
        service_missing = client.get(f"/api/agent/v1/assets/asset:sha256:{'0' * 64}")
        non_agent_missing = client.get("/api/does-not-exist")
        motion_catalog = client.get(
            "/api/agent/v1/assets/available",
            params={"root_id": "source", "kind": "motion_bundle"},
        )
        robot_catalog = client.get(
            "/api/agent/v1/assets/available",
            params={"root_id": "robot-library", "kind": "robot_bundle"},
        )
        motion_entry = motion_catalog.json()["assets"][0]
        registered = client.post(
            "/api/agent/v1/assets",
            json={
                field: motion_entry[field]
                for field in (
                    "root_id",
                    "relative_path",
                    "display_name",
                    "kind",
                    "category",
                    "recursive",
                )
            },
        )
        inspected = client.get(f"/api/agent/v1/assets/{registered.json()['asset_id']}/inspect")
        robot_entry = robot_catalog.json()["assets"][0]
        registered_robot = client.post(
            "/api/agent/v1/assets",
            json={
                field: robot_entry[field]
                for field in (
                    "root_id",
                    "relative_path",
                    "display_name",
                    "kind",
                    "category",
                    "recursive",
                )
            },
        )
        inspected_robot = client.get(
            f"/api/agent/v1/assets/{registered_robot.json()['asset_id']}/inspect"
        )
        try:
            app.state.agent_job_manager = boundary_manager
            streamed_artifact = client.get(
                f"/api/agent/v1/jobs/{boundary_descriptor.job_id}/artifacts/"
                f"{boundary_descriptor.artifact_id}/content"
            )
        finally:
            app.state.agent_job_manager = production_job_manager

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "1.0"
    assert payload["scheduler"]["mode"] == "unlimited"
    assert payload["scheduler"]["max_running_jobs"] == 0
    assert payload["features"]["agent_rest"] is True
    assert payload["features"]["asset_registry"] is True
    assert payload["features"]["available_asset_catalog"] is True
    assert payload["features"]["asset_inspection"] is True
    assert payload["features"]["preflight"] is True
    assert payload["features"]["artifact_store"] is True
    assert payload["features"]["persistent_jobs"] is True
    assert payload["features"]["idempotent_jobs"] is True
    assert payload["features"]["revision_polling"] is True
    assert payload["features"]["revision_waiting"] is True
    assert payload["features"]["job_execution"] is True
    assert payload["features"]["job_cancellation"] is True
    assert payload["features"]["job_retry"] is True
    assert payload["features"]["calibration_status"] is True
    assert payload["features"]["calibration_proposals"] is True
    assert payload["features"]["calibration_validation"] is True
    assert payload["features"]["calibration_silent_save"] is True
    assert payload["features"]["calibration_visual_preview"] is True
    assert agent_missing_route.status_code == 404
    assert agent_missing_route.json()["code"] == "ENDPOINT_NOT_FOUND"
    assert "detail" not in agent_missing_route.json()
    assert agent_wrong_method.status_code == 405
    assert agent_wrong_method.json()["code"] == "METHOD_NOT_ALLOWED"
    assert "GET" in agent_wrong_method.headers["allow"]
    assert service_missing.status_code == 404
    assert service_missing.json()["code"] == "ASSET_NOT_FOUND"
    assert non_agent_missing.status_code == 404
    assert "detail" in non_agent_missing.json()
    assert streamed_artifact.status_code == 200
    assert streamed_artifact.content == boundary_payload
    assert payload["asset_root_ids"] == ["motion-library", "robot-library", "source"]
    assert app.state.agent_artifact_store.database_path.parent == (
        tmp_path / "save" / ".hhtools-agent"
    )
    assert app.state.agent_job_store.database_path.parent == (tmp_path / "save" / ".hhtools-agent")
    assert app.state.agent_job_manager.execution_available is True
    assert app.state.agent_job_manager._scheduler is app.state.job_scheduler  # noqa: SLF001
    assert (
        app.state.agent_legacy_job_upgrade_service._asset_service  # noqa: SLF001
        is app.state.agent_asset_service
    )
    assert (
        app.state.agent_legacy_job_upgrade_service._root_locator  # noqa: SLF001
        is app.state.agent_legacy_root_locator
    )
    assert "source_root" not in payload
    assert "save_dir" not in payload
    assert registered.status_code == 201
    assert inspected.status_code == 200
    assert inspected.json()["status"] == "valid"
    assert motion_catalog.status_code == 200
    assert motion_catalog.json()["total"] == 1
    assert motion_entry["relative_path"] == "walk.npz"
    assert robot_catalog.status_code == 200
    assert robot_catalog.json()["total"] == 1
    assert robot_entry["relative_path"] == "catalog_bot"
    assert registered_robot.status_code == 201
    assert inspected_robot.status_code == 200
    assert inspected_robot.json()["status"] == "valid"
    assert str(tmp_path) not in motion_catalog.text
    assert str(tmp_path) not in robot_catalog.text
    assert str(tmp_path) not in registered.text
    assert str(tmp_path) not in inspected.text
    clear_cache()


def test_agent_robot_loader_uses_and_releases_an_isolated_manifest_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def local_tmpdir(tag: str) -> Path:
        path = tmp_path / f"runtime-{tag}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    robot_library = tmp_path / "robots"
    robot_root = robot_library / "snapshot_bot"
    robot_root.mkdir(parents=True)
    (robot_root / "robot.yaml").write_text(
        "name: snapshot_bot\n"
        "display_name: Snapshot Bot\n"
        "urdf: robot.urdf\n"
        "dof_order: [hip]\n"
        "ik_map:\n"
        "  hips: base\n",
        encoding="utf-8",
    )
    original_urdf = """<?xml version="1.0"?>
<robot name="snapshot_bot">
  <link name="world"/>
  <joint name="floating_base_joint" type="floating">
    <parent link="world"/>
    <child link="base"/>
  </joint>
  <link name="base">
    <inertial>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
  <joint name="hip" type="revolute">
    <parent link="base"/>
    <child link="leg"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="2"/>
  </joint>
  <link name="leg">
    <inertial>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
</robot>
"""
    source_urdf = robot_root / "robot.urdf"
    source_urdf.write_text(original_urdf, encoding="utf-8")

    from hhtools.robot import urdf_normalize

    original_convert = urdf_normalize.convert_unsupported_meshes_for_mujoco

    def record_conversion_workspace(urdf_path: Path, **kwargs):
        (Path(urdf_path).parent / "mesh-conversion.marker").write_text(
            "workspace-only",
            encoding="utf-8",
        )
        return original_convert(urdf_path, **kwargs)

    monkeypatch.setattr(
        urdf_normalize,
        "convert_unsupported_meshes_for_mujoco",
        record_conversion_workspace,
    )

    monkeypatch.setattr(server_state, "_tmpdir", local_tmpdir)
    monkeypatch.setattr(server_state, "_robot_library_root", lambda: robot_library)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv(
        "HHTOOLS_MOTION_LIBRARY_SETTINGS_PATH",
        str(tmp_path / "motion-library-settings.json"),
    )
    clear_cache()
    source_root = tmp_path / "motions"
    source_root.mkdir()
    app = server.create_app(
        source_root=source_root,
        save_dir=tmp_path / "save",
        cache_dir=tmp_path / "cache",
        job_history_dir=tmp_path / "history",
        job_settings_path=tmp_path / "job-settings.json",
    )
    bundle = app.state.agent_asset_service.register(
        AssetRegistrationRequest(
            root_id="robot-library",
            relative_path="snapshot_bot",
            kind="robot_bundle",
            recursive=True,
        )
    )
    robot_digest = bundle.asset_id.removeprefix("asset:sha256:")
    motion_digest = "a" * 64
    spec = JobSpecV2(
        kind=JobSpecKind.RETARGET,
        plan_id=f"plan:sha256:{'b' * 64}",
        inputs=[
            JobSpecInput(
                asset_id=f"asset:sha256:{motion_digest}",
                sha256=motion_digest,
            )
        ],
        robot=JobSpecRobot(
            robot_id="snapshot_bot",
            asset_id=bundle.asset_id,
            config_sha256=robot_digest,
        ),
        calibration=None,
        backend="newton",
        effective_parameters={},
        output_policy=OutputPolicy.CREATE_NEW,
        provenance=JobSpecProvenance(
            hhtools_git_commit="test",
            hhtools_dirty=False,
            python="3.12",
        ),
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )
    executor = app.state.agent_job_manager._executor  # noqa: SLF001
    bindings = executor.h2r._bindings  # noqa: SLF001

    model = bindings.get_robot_model(spec)
    snapshot_root = model.preset.root_dir
    try:
        assert snapshot_root != robot_root
        assert snapshot_root.parent.parent == (
            tmp_path / "save" / ".hhtools-agent" / "temporary" / "robots"
        )
        assert source_urdf.read_text(encoding="utf-8") == original_urdf
        assert "floating_base_joint" not in model.preset.urdf_path.read_text(encoding="utf-8")
        assert (snapshot_root / "mesh-conversion.marker").is_file()
        assert not (robot_root / "mesh-conversion.marker").exists()

        # Interaction-Mesh retries compilation when the first cached MuJoCo
        # model is unavailable.  Its preset path must remain inside the same
        # disposable snapshot, never point back at the registered bundle.
        from hhtools.retarget.interaction_mesh.mujoco_scene import (
            require_mujoco_model,
        )

        model.mujoco_model = None
        model.mjcf_xml = ""
        assert require_mujoco_model(model) is not None
        assert source_urdf.read_text(encoding="utf-8") == original_urdf
    finally:
        bindings.release_robot_model(model)

    assert not snapshot_root.parent.exists()
    assert source_urdf.read_text(encoding="utf-8") == original_urdf
    clear_cache()
