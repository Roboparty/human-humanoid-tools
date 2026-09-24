"""Contract tests shared by future REST, JSON CLI, and MCP adapters."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from hhtools.contracts import (
    AgentCliArgumentDiagnostic,
    AgentCliHelp,
    AgentCliHelpArgument,
    AgentCliHelpSubcommand,
    AgentH2RExecutionParameters,
    AgentJobView,
    ApiError,
    ArtifactDescriptor,
    ArtifactListResponse,
    AssetBundle,
    AssetCategory,
    AssetFile,
    AssetInspection,
    AssetInspectionRequest,
    AssetKind,
    AssetRegistrationRequest,
    AssetSearchResponse,
    AssetSource,
    AvailableAssetCatalogEntry,
    AvailableAssetCatalogRequest,
    AvailableAssetCatalogResponse,
    BackendCapability,
    CalibrationCandidate,
    CalibrationJointLimit,
    CalibrationPreview,
    CalibrationPreviewRequest,
    CalibrationProposalRequest,
    CalibrationProposalResponse,
    CalibrationSaveReceipt,
    CalibrationSaveRequest,
    CalibrationStatusRequest,
    CalibrationStatusResponse,
    CalibrationValidationReport,
    CalibrationValidationRequest,
    CalibrationVisualReview,
    CapabilityResponse,
    DeviceCapability,
    EvaluationReport,
    ExecutionProvenance,
    FailureItem,
    FailureReport,
    HumanRetargetExecutionOptions,
    InspectionStatus,
    JobManifest,
    JobOutcome,
    JobProgress,
    JobQueueView,
    JobRetryRequest,
    JobSpecCalibration,
    JobSpecInput,
    JobSpecProvenance,
    JobSpecRobot,
    JobSpecV2,
    JobStartRequest,
    JobState,
    LegacyJobUpgradeRequest,
    LegacyJobUpgradeResponse,
    LegacyMigrationReceipt,
    NextAction,
    OutputPolicy,
    PreflightCheck,
    PreflightCheckLevel,
    PreflightResponse,
    PreflightStatus,
    R2RCalibrationStatusRequest,
    RetargetPlan,
    RetargetPreflightRequest,
    RobotCapability,
    RobotListResponse,
    SchedulerCapability,
    SchedulerMode,
    SchemaVersion,
)

NOW = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
ASSET_MOTION = f"asset:sha256:{SHA_A}"
ASSET_ROBOT = f"asset:sha256:{SHA_B}"
PLAN_ID = f"plan:sha256:{SHA_A}"
CALIBRATION_ID = f"cal:sha256:{SHA_C}"
ARTIFACT_ID = f"artifact:retargeted_motion:{SHA_A}"


def motion_file(path: str = "motions/walk.npz") -> AssetFile:
    return AssetFile(
        role="motion",
        relative_path=path,
        sha256=SHA_A,
        size_bytes=128,
        media_type="application/octet-stream",
    )


def ready_plan() -> RetargetPlan:
    return RetargetPlan(
        plan_id=PLAN_ID,
        created_at=NOW,
        expires_at=NOW + timedelta(hours=1),
        motion_asset_id=ASSET_MOTION,
        robot_id="g1_29dof",
        robot_asset_id=ASSET_ROBOT,
        backend="newton",
        calibration_id=CALIBRATION_ID,
        output_format="csv",
        output_policy="create_new",
        parameters={"solver": {"max_iterations": 20}},
        input_digest=SHA_A,
        robot_digest=SHA_B,
        calibration_digest=SHA_C,
    )


def test_common_contracts_use_stable_machine_values_and_forbid_extra_fields() -> None:
    error = ApiError(
        code="CALIBRATION_MISSING",
        message="The robot needs calibration.",
        retryable=False,
        stage="preflight",
        details={"robot_id": "g1_29dof"},
    )
    action = NextAction(
        actor="human",
        action="open_calibration_ui",
        message="Calibrate the selected robot.",
        url="http://127.0.0.1:8009/?view=calibration",
    )

    assert error.model_dump(mode="json")["code"] == "CALIBRATION_MISSING"
    assert action.model_dump(mode="json")["action"] == "open_calibration_ui"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ApiError(code="BAD_INPUT", message="Bad input", stage="request", typo=True)
    with pytest.raises(ValidationError):
        ApiError(code="calibration_missing", message="Bad machine code", stage="request")


def test_cli_metadata_contracts_are_versioned_and_reject_unlisted_diagnostics() -> None:
    diagnostic = AgentCliArgumentDiagnostic(
        reason_code="MISSING_ARGUMENT",
        command="hhtools agent job start",
        argument="--plan",
        expected="A preflight plan id.",
        usage="hhtools agent job start --plan PLAN_ID --idempotency-key KEY",
    )
    help_document = AgentCliHelp(
        command="hhtools agent job",
        summary="Manage jobs.",
        usage="hhtools agent job COMMAND --help",
        options=[AgentCliHelpArgument(name="--help", description="Return JSON help.")],
        subcommands=[AgentCliHelpSubcommand(name="start", summary="Start a job.")],
    )

    assert diagnostic.model_dump(mode="json")["reason_code"] == "MISSING_ARGUMENT"
    assert help_document.model_dump(mode="json")["schema_version"] == "1.0"
    with pytest.raises(ValidationError):
        AgentCliArgumentDiagnostic(
            reason_code="MISSING_ARGUMENT",
            command="hhtools agent job start",
            argument=r"C:\Users\Nora\secret.txt",
            usage="hhtools agent job start",
        )
    with pytest.raises(ValidationError, match="Extra inputs"):
        AgentCliArgumentDiagnostic(
            reason_code="MISSING_ARGUMENT",
            command="hhtools agent job start",
            argument="--plan",
            usage="hhtools agent job start",
            raw_argv=["secret"],
        )


def test_public_url_contracts_match_the_portable_transport_boundary() -> None:
    documentation = NextAction(
        actor="human",
        action="read_documentation",
        url="https://docs.example.invalid/guide?next=/agent/v1",
    )
    assert documentation.url is not None

    with pytest.raises(ValidationError):
        NextAction(
            actor="human",
            action="open_calibration_ui",
            url="http://127.0.0.1:8009/private/session",
        )

    with pytest.raises(ValidationError):
        ArtifactDescriptor(
            artifact_id=ARTIFACT_ID,
            job_id="job_test",
            kind="retargeted_motion",
            resource_uri="hhtools://custom/private",
            sha256=SHA_A,
        )


@pytest.mark.parametrize(
    "bad_path",
    ["../secret.npz", "/srv/motion.npz", r"C:\motions\walk.npz", r"motions\walk.npz"],
)
def test_asset_file_rejects_non_portable_or_escaping_paths(bad_path: str) -> None:
    with pytest.raises(ValidationError):
        motion_file(bad_path)


def test_asset_bundle_is_content_addressed_and_primary_file_is_declared() -> None:
    bundle = AssetBundle(
        asset_id=ASSET_MOTION,
        kind="motion_bundle",
        category="plain_motion",
        display_name="Walk",
        primary_file="motions/walk.npz",
        files=[motion_file()],
        source=AssetSource(
            scheme="managed_file",
            root_id="motion-library",
            registered_at=NOW,
        ),
        metadata={"dataset": "AMASS"},
    )

    payload = bundle.model_dump(mode="json")
    assert payload["schema_version"] == "1.0"
    assert payload["files"][0]["role"] == "motion"
    assert "content" not in payload["files"][0]

    with pytest.raises(ValidationError, match="primary_file"):
        AssetBundle(
            asset_id=ASSET_MOTION,
            kind="motion_bundle",
            category="plain_motion",
            display_name="Walk",
            primary_file="motions/missing.npz",
            files=[motion_file()],
        )
    with pytest.raises(ValidationError, match="duplicate relative paths"):
        AssetBundle(
            asset_id=ASSET_MOTION,
            kind="motion_bundle",
            category="plain_motion",
            display_name="Walk",
            primary_file="motions/walk.npz",
            files=[motion_file(), motion_file()],
        )


def test_asset_requests_only_accept_paths_below_a_configured_root() -> None:
    request = AssetRegistrationRequest(
        root_id="motion-library",
        relative_path="omomo/sub10",
        category="object_interaction",
    )
    inspection = AssetInspectionRequest(asset_id=ASSET_MOTION)

    assert request.relative_path == "omomo/sub10"
    assert inspection.verify_hashes is True
    with pytest.raises(ValidationError):
        AssetRegistrationRequest(root_id="motion-library", relative_path="C:/motions/walk.npz")
    with pytest.raises(ValidationError):
        AssetSource(
            scheme="managed_file",
            root_id="motion-library",
            registered_at=NOW,
            logical_path="../outside",
        )


def test_asset_search_response_is_versioned_and_bounded() -> None:
    response = AssetSearchResponse(assets=[], total=0, limit=50, offset=0)

    assert response.model_dump(mode="json")["schema_version"] == "1.0"
    with pytest.raises(ValidationError):
        AssetSearchResponse(assets=[], total=0, limit=501, offset=0)


def test_asset_inspection_status_matches_issues() -> None:
    inspection = AssetInspection(
        asset_id=ASSET_MOTION,
        status="valid_with_warnings",
        kind="motion_bundle",
        category="plain_motion",
        source_format="npz",
        reference_model="smpl",
        frame_count=120,
        frame_rate_hz=30.0,
        duration_seconds=4.0,
        joint_count=24,
        warnings=["Root height contains a large discontinuity."],
    )
    assert inspection.status is InspectionStatus.VALID_WITH_WARNINGS

    with pytest.raises(ValidationError, match="at least one error"):
        AssetInspection(
            asset_id=ASSET_MOTION,
            status="invalid",
            kind="motion_bundle",
            category="plain_motion",
        )

    with pytest.raises(ValidationError, match="errors must use invalid status"):
        AssetInspection(
            asset_id=ASSET_MOTION,
            status="valid_with_warnings",
            kind="motion_bundle",
            category="plain_motion",
            warnings=["Partial metadata."],
            errors=[
                ApiError(
                    code="BUNDLE_INCOMPLETE",
                    message="A sidecar is missing.",
                    stage="asset_inspection",
                )
            ],
        )


def test_capabilities_are_compact_and_json_serializable() -> None:
    response = CapabilityResponse(
        service_version="0.2.0",
        backends=[
            BackendCapability(
                backend_id="newton",
                display_name="Newton",
                available=True,
                supported_categories=["plain_motion"],
                output_formats=["csv", "pkl"],
                features={"gpu": True},
            )
        ],
        devices=[
            DeviceCapability(
                device_id="cuda:0",
                kind="cuda",
                display_name="NVIDIA RTX 4090",
                available=True,
                total_memory_bytes=24 * 1024**3,
            )
        ],
        robots=[
            RobotCapability(
                robot_id="g1_29dof",
                display_name="Unitree G1",
                available=True,
                has_urdf=True,
                has_ik_mapping=True,
                dof_count=29,
                supported_references=["smpl", "smplx"],
                calibrated_references=["smpl"],
                scaler_references=["smplx"],
            )
        ],
        scheduler=SchedulerCapability(
            max_running_jobs=0,
            max_queued_jobs=0,
            running=1,
            queued=2,
            reserved=0,
            mode="unlimited",
        ),
        supported_input_formats=["npz", "bvh", "glb"],
        supported_output_formats=["csv", "pkl"],
        features={"preflight": True},
    )

    payload = response.model_dump(mode="json")
    assert payload["backends"][0]["supported_categories"] == ["plain_motion"]
    assert payload["devices"][0]["kind"] == "cuda"
    assert payload["scheduler"]["max_running_jobs"] == 0
    assert payload["scheduler"]["mode"] == "unlimited"
    assert payload["supported_output_formats"] == ["csv", "pkl"]

    configured_but_ignored_queue_limit = SchedulerCapability(
        max_running_jobs=0,
        max_queued_jobs=8,
        mode="unlimited",
    )
    assert configured_but_ignored_queue_limit.mode is SchedulerMode.UNLIMITED

    with pytest.raises(ValidationError, match="mode must be mixed"):
        SchedulerCapability(max_running_jobs=8, max_queued_jobs=0, mode="limited")


def test_preflight_request_and_ready_response_round_trip_through_json() -> None:
    request = RetargetPreflightRequest(
        motion_asset_id=ASSET_MOTION,
        robot_id="g1_29dof",
        backend=None,
    )
    check = PreflightCheck(
        code="CALIBRATION_PRESENT",
        level="pass",
        message="A compatible calibration is available.",
    )
    response = PreflightResponse(
        request_id="req_01",
        status="ready",
        plan=ready_plan(),
        checks=[check],
        recommended_backend="newton",
    )

    assert request.output_policy is OutputPolicy.CREATE_NEW
    assert request.output_format == "csv"
    encoded = response.model_dump_json()
    decoded = PreflightResponse.model_validate_json(encoded)
    assert decoded.status is PreflightStatus.READY
    assert decoded.plan is not None
    assert decoded.plan.backend == "newton"


def test_preflight_state_requires_plan_action_or_error_as_appropriate() -> None:
    with pytest.raises(ValidationError, match="must include a plan"):
        PreflightResponse(request_id="req_ready", status="ready")

    action = NextAction(actor="human", action="open_calibration_ui")
    response = PreflightResponse(
        request_id="req_action",
        checks=[
            PreflightCheck(
                code="CALIBRATION_MISSING",
                level="error",
                message="Calibration is required.",
                next_action=action,
            )
        ],
        status="human_action_required",
        required_actions=[action],
    )
    assert response.required_actions[0].actor == "human"

    with pytest.raises(ValidationError, match="must include an error"):
        PreflightResponse(request_id="req_rejected", status="rejected")
    assert response.status is PreflightStatus.HUMAN_ACTION_REQUIRED


def test_job_spec_v2_round_trips_complete_execution_identity() -> None:
    spec = JobSpecV2(
        kind="retarget",
        plan_id=PLAN_ID,
        inputs=[JobSpecInput(asset_id=ASSET_MOTION, sha256=SHA_A)],
        robot=JobSpecRobot(
            robot_id="g1_29dof",
            asset_id=ASSET_ROBOT,
            config_sha256=SHA_B,
        ),
        calibration=JobSpecCalibration(
            calibration_id=CALIBRATION_ID,
            sha256=SHA_C,
        ),
        backend="newton",
        effective_parameters={
            "reference": "smpl",
            "output_format": "csv",
            "start_time": 0.0,
            "end_time": None,
        },
        output_policy="create_new",
        provenance=JobSpecProvenance(
            hhtools_git_commit="0123456789abcdef",
            hhtools_dirty=False,
            python="3.12.11",
            pytorch="2.8.0",
            cuda="12.8",
            newton="1.0.0",
            device="NVIDIA RTX 4090",
        ),
        created_at=NOW,
    )

    decoded = JobSpecV2.model_validate_json(spec.model_dump_json())
    assert decoded.schema_version == 2
    assert decoded.inputs[0].asset_id == ASSET_MOTION
    assert decoded.effective_parameters["output_format"] == "csv"
    with pytest.raises(ValidationError, match="frozen"):
        spec.backend = "interaction_mesh"
    with pytest.raises(ValidationError, match="duplicate asset ids"):
        JobSpecV2(**{**spec.model_dump(), "inputs": [spec.inputs[0], spec.inputs[0]]})


def test_retarget_plan_is_frozen_and_content_bound() -> None:
    plan = ready_plan()
    with pytest.raises(ValidationError, match="frozen"):
        plan.backend = "interaction_mesh"
    with pytest.raises(ValidationError, match="expires_at"):
        RetargetPlan(
            **{
                **plan.model_dump(),
                "expires_at": NOW - timedelta(seconds=1),
            }
        )


def test_job_progress_validates_counts_and_fraction() -> None:
    progress = JobProgress(
        phase="ik_solve",
        fraction=0.625,
        revision=17,
        completed_items=5,
        total_items=8,
        updated_at=NOW,
    )
    assert progress.fraction == 0.625
    with pytest.raises(ValidationError, match="cannot exceed"):
        JobProgress(completed_items=9, total_items=8)
    with pytest.raises(ValidationError):
        JobProgress(fraction=1.01)


def test_agent_job_view_separates_lifecycle_from_outcome() -> None:
    running = AgentJobView(
        job_id="job_01",
        state="running",
        progress=JobProgress(phase="ik_solve", fraction=0.62, revision=17),
        summary={"robot_id": "g1_29dof", "backend": "newton", "input_count": 1},
        submitted_at=NOW,
        started_at=NOW + timedelta(seconds=1),
        poll_after_ms=1500,
    )
    assert running.outcome is None
    assert running.cancellation_requested is False
    assert running.attempt == 1
    assert running.model_dump(mode="json")["state"] == "running"

    completed = AgentJobView(
        job_id="job_01",
        state="completed",
        outcome="success",
        progress=JobProgress(phase="done", fraction=1.0, revision=22),
        artifacts=[
            ArtifactDescriptor(
                artifact_id=ARTIFACT_ID,
                job_id="job_01",
                kind="retargeted_motion",
                format="csv",
                resource_uri=f"hhtools://jobs/job_01/artifacts/{ARTIFACT_ID}",
                sha256=SHA_A,
            )
        ],
        artifact_count=1,
        submitted_at=NOW,
        started_at=NOW + timedelta(seconds=1),
        completed_at=NOW + timedelta(seconds=20),
    )
    assert completed.state is JobState.COMPLETED
    assert completed.outcome is JobOutcome.SUCCESS
    assert completed.artifacts[0].uri.startswith("hhtools://")
    artifact_payload = completed.artifacts[0].model_dump(mode="json")
    assert "resource_uri" in artifact_payload
    assert "uri" not in artifact_payload
    with pytest.raises(ValidationError, match="safe MIME type"):
        ArtifactDescriptor(
            artifact_id=ARTIFACT_ID,
            job_id="job_01",
            kind="preview",
            resource_uri=f"hhtools://jobs/job_01/artifacts/{ARTIFACT_ID}",
            media_type="text/plain\r\nX-Injected: true",
        )
    with pytest.raises(ValidationError):
        ArtifactDescriptor(
            artifact_id=ARTIFACT_ID,
            job_id="job_01",
            kind="../preview",
            format="csv\r\nX-Injected",
            resource_uri=f"hhtools://jobs/job_01/artifacts/{ARTIFACT_ID}",
        )


def test_agent_job_view_rejects_inconsistent_terminal_states() -> None:
    with pytest.raises(ValidationError, match="outcome is only valid"):
        AgentJobView(
            job_id="job_running",
            state="running",
            outcome="success",
            progress=JobProgress(),
            submitted_at=NOW,
        )
    with pytest.raises(ValidationError, match="must include an outcome"):
        AgentJobView(
            job_id="job_done",
            state="completed",
            progress=JobProgress(fraction=1.0),
            submitted_at=NOW,
        )
    with pytest.raises(ValidationError, match="must include an error"):
        AgentJobView(
            job_id="job_failed",
            state="failed",
            progress=JobProgress(),
            submitted_at=NOW,
        )


def test_agent_job_view_enforces_timestamps_lineage_and_compact_artifact_count() -> None:
    with pytest.raises(ValidationError, match="queued jobs cannot"):
        AgentJobView(
            job_id="job_queued",
            state="queued",
            progress=JobProgress(),
            submitted_at=NOW,
            started_at=NOW,
        )
    with pytest.raises(ValidationError, match="running jobs must"):
        AgentJobView(
            job_id="job_running",
            state="running",
            progress=JobProgress(),
            submitted_at=NOW,
        )
    with pytest.raises(ValidationError, match="terminal jobs must"):
        AgentJobView(
            job_id="job_cancelled",
            state="cancelled",
            progress=JobProgress(),
            submitted_at=NOW,
        )
    with pytest.raises(ValidationError, match="artifact_count"):
        AgentJobView(
            job_id="job_done",
            state="completed",
            outcome="success",
            progress=JobProgress(fraction=1.0),
            artifacts=[
                ArtifactDescriptor(
                    artifact_id=ARTIFACT_ID,
                    job_id="job_done",
                    kind="manifest",
                    resource_uri=f"hhtools://jobs/job_done/artifacts/{ARTIFACT_ID}",
                )
            ],
            artifact_count=0,
            submitted_at=NOW,
            completed_at=NOW + timedelta(seconds=1),
        )
    with pytest.raises(ValidationError, match="retry jobs require"):
        AgentJobView(
            job_id="job_child",
            parent_job_id="job_parent",
            state="queued",
            progress=JobProgress(),
            submitted_at=NOW,
        )

    child = AgentJobView(
        job_id="job_child",
        parent_job_id="job_parent",
        root_job_id="job_root",
        attempt=3,
        state="queued",
        progress=JobProgress(),
        submitted_at=NOW,
        cancellation_requested=True,
        cancellable=True,
    )
    assert child.parent_job_id == "job_parent"
    assert child.root_job_id == "job_root"
    assert child.attempt == 3


def test_job_operation_requests_and_artifact_pages_are_strict_and_versioned() -> None:
    start = JobStartRequest(plan_id=PLAN_ID, idempotency_key="agent:run-001")
    retry = JobRetryRequest(idempotency_key="agent:retry-001")
    descriptor = ArtifactDescriptor(
        artifact_id=ARTIFACT_ID,
        job_id="job_01",
        kind="retargeted_motion",
        format="csv",
        resource_uri=f"hhtools://jobs/job_01/artifacts/{ARTIFACT_ID}",
    )
    page = ArtifactListResponse(
        job_id="job_01",
        artifacts=[descriptor],
        total=1,
    )

    assert start.model_dump(mode="json")["schema_version"] == "1.0"
    assert retry.idempotency_key == "agent:retry-001"
    assert page.artifacts == [descriptor]
    with pytest.raises(ValidationError):
        JobStartRequest(plan_id=PLAN_ID, idempotency_key="contains whitespace")
    with pytest.raises(ValidationError, match="requested job"):
        ArtifactListResponse(job_id="job_other", artifacts=[descriptor], total=1)
    with pytest.raises(ValidationError, match="exceed limit"):
        ArtifactListResponse(
            job_id="job_01",
            artifacts=[descriptor, descriptor],
            total=2,
            limit=1,
        )
    with pytest.raises(ValidationError, match="beyond total"):
        ArtifactListResponse(
            job_id="job_01",
            artifacts=[descriptor],
            total=1,
            offset=1,
        )


def test_legacy_upgrade_transport_request_is_wrapped_and_strict() -> None:
    request = LegacyJobUpgradeRequest(payload={"schema_version": 1, "kind": "h2r", "request": {}})

    assert request.schema_version is SchemaVersion.V1
    with pytest.raises(ValidationError, match="Extra inputs"):
        LegacyJobUpgradeRequest(payload={}, typo=True)


@pytest.mark.parametrize(
    "model",
    [
        AgentCliArgumentDiagnostic,
        AgentCliHelpArgument,
        AgentCliHelpSubcommand,
        AgentCliHelp,
        ApiError,
        NextAction,
        AssetFile,
        AssetBundle,
        AssetInspection,
        AssetRegistrationRequest,
        AssetInspectionRequest,
        AssetSearchResponse,
        AvailableAssetCatalogEntry,
        AvailableAssetCatalogRequest,
        AvailableAssetCatalogResponse,
        BackendCapability,
        DeviceCapability,
        EvaluationReport,
        FailureItem,
        FailureReport,
        RobotCapability,
        RobotListResponse,
        SchedulerCapability,
        CapabilityResponse,
        CalibrationCandidate,
        CalibrationJointLimit,
        CalibrationPreview,
        CalibrationPreviewRequest,
        CalibrationProposalRequest,
        CalibrationProposalResponse,
        CalibrationSaveReceipt,
        CalibrationSaveRequest,
        CalibrationStatusRequest,
        CalibrationStatusResponse,
        CalibrationValidationReport,
        CalibrationValidationRequest,
        CalibrationVisualReview,
        RetargetPreflightRequest,
        PreflightCheck,
        RetargetPlan,
        PreflightResponse,
        JobProgress,
        JobManifest,
        ArtifactDescriptor,
        ArtifactListResponse,
        JobQueueView,
        JobRetryRequest,
        JobStartRequest,
        AgentJobView,
        LegacyJobUpgradeRequest,
        LegacyJobUpgradeResponse,
        LegacyMigrationReceipt,
        JobSpecInput,
        JobSpecRobot,
        JobSpecCalibration,
        JobSpecProvenance,
        JobSpecV2,
        HumanRetargetExecutionOptions,
        AgentH2RExecutionParameters,
        ExecutionProvenance,
    ],
)
def test_every_public_object_schema_forbids_unknown_fields(model: type) -> None:
    schema = model.model_json_schema()
    assert schema["additionalProperties"] is False


def test_public_enum_values_are_stable_english_tokens() -> None:
    assert SchemaVersion.V1.value == "1.0"
    assert AssetKind.MOTION_BUNDLE.value == "motion_bundle"
    assert AssetCategory.PLAIN_MOTION.value == "plain_motion"
    assert AssetCategory.OBJECT_INTERACTION.value == "object_interaction"
    assert PreflightCheckLevel.PASS.value == "pass"
    assert PreflightStatus.HUMAN_ACTION_REQUIRED.value == "human_action_required"
    assert JobState.CANCELLED.value == "cancelled"
    assert JobOutcome.PARTIAL.value == "partial"
    assert JobOutcome.REVIEW_REQUIRED.value == "review_required"


def test_web_and_agent_h2r_execution_options_share_strict_solver_fields() -> None:
    web = HumanRetargetExecutionOptions(
        backend="newton",
        ik_iterations=24,
        retarget_fps=30.0,
        reference="smpl",
        foot_clamp_anti_penetration=True,
    )
    agent = AgentH2RExecutionParameters(
        **web.model_dump(exclude_none=True),
        run_mode="smoke",
        human_height=1.72,
        limit_frames=30,
        retarget_profile="bundled_scaler",
        output_format="csv",
    )
    assert agent.ik_iterations == web.ik_iterations
    assert agent.retarget_fps == web.retarget_fps
    with pytest.raises(ValidationError):
        HumanRetargetExecutionOptions(foot_clamp_anti_penetration="yes")


def test_execution_provenance_requires_truthful_fallback_and_cuda_graph_flags() -> None:
    observed = ExecutionProvenance(
        executor="web_h2r_v1",
        backend="newton",
        device="cuda:0",
        device_kind="cuda",
        precision="float32",
        cuda_graph_requested=True,
        cuda_graph_used=True,
    )
    assert observed.fallback_used is False
    with pytest.raises(ValidationError):
        ExecutionProvenance(cuda_graph_requested=False, cuda_graph_used=True)
    with pytest.raises(ValidationError):
        ExecutionProvenance(fallback_reason="silent fallback")


def test_gpt_silent_calibration_requires_a_portable_passing_visual_review() -> None:
    candidate_id = f"cal-candidate:sha256:{SHA_C}"

    with pytest.raises(ValidationError):
        CalibrationSaveRequest(
            candidate_id=candidate_id,
            save_mode="gpt_vision_silent",
        )
    with pytest.raises(ValidationError):
        CalibrationSaveRequest(
            candidate_id=candidate_id,
            save_mode="gpt_vision_silent",
            visual_review=CalibrationVisualReview(
                reviewer="gpt_vision",
                verdict="pass",
                summary="Loaded /home/private/calibration.png",
            ),
        )

    request = CalibrationSaveRequest(
        candidate_id=candidate_id,
        save_mode="gpt_vision_silent",
        visual_review=CalibrationVisualReview(
            reviewer="gpt_vision",
            verdict="pass",
            model_hint="gpt-test",
            summary="Front and side landmark directions are aligned.",
        ),
    )
    assert request.visual_review is not None
    assert request.visual_review.verdict.value == "pass"


def test_r2r_calibration_identity_requires_a_distinct_robot_pair() -> None:
    valid = R2RCalibrationStatusRequest(
        source_robot_id="source_bot",
        source_robot_asset_id=ASSET_MOTION,
        target_robot_id="target_bot",
        target_robot_asset_id=ASSET_ROBOT,
    )
    assert valid.source_robot_id != valid.target_robot_id

    with pytest.raises(ValidationError):
        R2RCalibrationStatusRequest(
            source_robot_id="same_bot",
            source_robot_asset_id=ASSET_MOTION,
            target_robot_id="same_bot",
            target_robot_asset_id=ASSET_ROBOT,
        )
    with pytest.raises(ValidationError):
        R2RCalibrationStatusRequest(
            source_robot_id="source_bot",
            source_robot_asset_id=ASSET_ROBOT,
            target_robot_id="target_bot",
            target_robot_asset_id=ASSET_ROBOT,
        )
