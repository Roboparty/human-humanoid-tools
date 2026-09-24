"""MCP v2 contract, service-parity, and real stdio integration tests."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import anyio
import numpy as np
import pytest
from mcp import Client, MCPError, StdioServerParameters

from hhtools.contracts import (
    AgentJobView,
    ApiError,
    ArtifactDescriptor,
    ArtifactExportReceipt,
    AvailableAssetCatalogEntry,
    AvailableAssetCatalogResponse,
    CalibrationCandidate,
    CalibrationPreview,
    CalibrationProposalResponse,
    CalibrationSaveReceipt,
    CalibrationStatusResponse,
    CalibrationValidationReport,
    CapabilityResponse,
    ErrorStage,
    JobProgress,
    NextAction,
    PreflightCheck,
    PreflightResponse,
    R2RPreflightResponse,
    SchedulerCapability,
)
from hhtools.mcp.runtime import AgentRuntime
from hhtools.mcp.server import create_mcp_server
from hhtools.robot.registry import clear_cache
from hhtools.services.jobs import JobManagerError
from hhtools.web import server as web_server
from hhtools.web.server import state as server_state

_DIGEST = "a" * 64
_ASSET_ID = f"asset:sha256:{_DIGEST}"
_PLAN_ID = f"plan:sha256:{_DIGEST}"
_JOB_ID = "job:mcp-test"
_ARTIFACT_ID = "artifact:retargeted_motion:mcp-test"
_NOW = datetime(2026, 8, 31, tzinfo=UTC)
_CALIBRATION_CANDIDATE_ID = f"cal-candidate:sha256:{'c' * 64}"

_EXPECTED_TOOLS = {
    "get_capabilities",
    "get_calibration_status",
    "get_r2r_calibration_status",
    "register_asset_bundle",
    "search_assets",
    "list_available_assets",
    "inspect_asset_bundle",
    "list_robots",
    "preview_calibration",
    "preview_r2r_calibration",
    "preflight_retarget",
    "preflight_r2r",
    "propose_calibration",
    "propose_r2r_calibration",
    "save_calibration",
    "save_r2r_calibration",
    "preflight_batch",
    "start_job",
    "start_retarget",
    "validate_calibration",
    "validate_r2r_calibration",
    "get_job",
    "wait_job",
    "lookup_job",
    "cancel_job",
    "retry_job",
    "list_job_artifacts",
    "export_artifact",
}
_EXPECTED_RESOURCES = {"hhtools://capabilities"}
_EXPECTED_RESOURCE_TEMPLATES = {
    "hhtools://schemas/agent/v1/{schema_name}",
    "hhtools://robots/{robot_id}",
    "hhtools://assets/{asset_id}/manifest",
    "hhtools://plans/{plan_id}",
    "hhtools://jobs/{job_id}/status",
    "hhtools://jobs/{job_id}/manifest",
    "hhtools://jobs/{job_id}/evaluation",
    "hhtools://jobs/{job_id}/batch",
    "hhtools://jobs/{job_id}/failures",
    "hhtools://jobs/{job_id}/artifacts/{artifact_id}",
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _capabilities() -> CapabilityResponse:
    return CapabilityResponse(
        service_version="mcp-test",
        scheduler=SchedulerCapability(
            max_running_jobs=0,
            max_queued_jobs=0,
            mode="unlimited",
        ),
        asset_root_ids=["source", "motion-library", "robot-library"],
        supported_input_formats=["bvh", "npz"],
        supported_output_formats=["csv"],
        features={"agent_rest": False, "json_cli": False, "mcp": True},
    )


def _job() -> AgentJobView:
    return AgentJobView(
        job_id=_JOB_ID,
        state="queued",
        progress=JobProgress(
            phase="queued",
            fraction=0.0,
            revision=7,
            message="No change since revision 7.",
        ),
        artifact_count=1,
        cancellable=True,
        submitted_at=_NOW,
        poll_after_ms=750,
    )


def _artifact() -> ArtifactDescriptor:
    return ArtifactDescriptor(
        artifact_id=_ARTIFACT_ID,
        job_id=_JOB_ID,
        kind="retargeted_motion",
        format="csv",
        resource_uri=f"hhtools://jobs/{_JOB_ID}/artifacts/{_ARTIFACT_ID}",
        media_type="text/csv",
        size_bytes=3,
        sha256=_DIGEST,
    )


class _MutatingReportPath:
    """Emulate a same-size report rewrite when a reader seeks for a second pass."""

    def __init__(self, path: Path, replacement: bytes) -> None:
        self.path = path
        self.replacement = replacement
        self.seek_calls = 0

    def open(self, _mode: str):
        owner = self
        stream = self.path.open("r+b")

        class _Stream:
            def __enter__(self):
                return self

            def __exit__(self, *args: Any) -> None:
                stream.close()

            def read(self, size: int = -1) -> bytes:
                return stream.read(size)

            def fileno(self) -> int:
                return stream.fileno()

            def seek(self, offset: int, whence: int = 0) -> int:
                owner.seek_calls += 1
                stream.seek(0)
                stream.write(owner.replacement)
                stream.flush()
                return stream.seek(offset, whence)

        return _Stream()


class _CapabilitiesService:
    def __init__(self) -> None:
        self.calls = 0

    def get_capabilities(self) -> CapabilityResponse:
        self.calls += 1
        return _capabilities()


class _AssetsService:
    def register(self, _request: Any) -> Any:
        raise AssertionError("asset mutation is outside this focused fixture")

    def search(self, **_kwargs: Any) -> Any:
        raise AssertionError("asset search is outside this focused fixture")

    def inspect(self, _request: Any) -> Any:
        raise AssertionError("asset inspection is outside this focused fixture")

    def get(self, _asset_id: str) -> Any:
        raise AssertionError("asset resource is outside this focused fixture")


class _AvailableAssetsService:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def list_available(self, request: Any) -> AvailableAssetCatalogResponse:
        self.calls.append(request)
        entry = AvailableAssetCatalogEntry(
            root_id="source",
            relative_path="AMASS/walk.npz",
            display_name="Walk",
            kind="motion_bundle",
            category="plain_motion",
            dataset="amass",
            reference="smpl",
        )
        return AvailableAssetCatalogResponse(
            assets=[entry],
            total=1,
            limit=request.limit,
            offset=request.offset,
        )


class _PreflightService:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def preflight_retarget(self, request: Any) -> PreflightResponse:
        self.calls.append(request)
        return PreflightResponse(
            request_id="request-mcp-human-action",
            status="human_action_required",
            required_actions=[
                NextAction(
                    actor="human",
                    action="open_calibration_ui",
                    message="Create and review the robot calibration before retrying.",
                    url="http://127.0.0.1:8009/?view=calibration",
                    parameters={"robot_id": request.robot_id},
                )
            ],
        )


class _R2RPreflightService:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def preflight_r2r(self, request: Any) -> R2RPreflightResponse:
        self.calls.append(request)
        return R2RPreflightResponse(
            request_id="request-mcp-r2r",
            status="human_action_required",
            recommended_backend="newton",
            required_actions=[
                NextAction(
                    actor="human",
                    action="open_calibration_ui",
                    message="Calibrate this exact robot pair before retrying.",
                    url="http://127.0.0.1:8009/?panel=r2r&calibrate=1",
                    parameters={
                        "source_robot_id": request.source_robot_id,
                        "target_robot_id": request.target_robot_id,
                    },
                )
            ],
        )


class _Plans:
    def get(self, _plan_id: str) -> Any:
        raise AssertionError("plan resource is outside this focused fixture")


def _calibration_validation() -> CalibrationValidationReport:
    return CalibrationValidationReport(
        candidate_id=_CALIBRATION_CANDIDATE_ID,
        valid=True,
        score=0.95,
        changed_joint_count=2,
        mapped_slots=16,
        edge_errors_deg={"left_upper_arm": 4.0, "right_upper_arm": 4.0},
        checks=[
            PreflightCheck(
                code="CALIBRATION_POSE_ALIGNED",
                level="pass",
                message="Calibration pose is aligned.",
            )
        ],
    )


class _CalibrationService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def status(self, request: Any) -> CalibrationStatusResponse:
        self.calls.append("status")
        return CalibrationStatusResponse(
            request_id="req_calibration_mcp",
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

    def propose(self, request: Any) -> CalibrationProposalResponse:
        self.calls.append("propose")
        candidate = CalibrationCandidate(
            candidate_id=_CALIBRATION_CANDIDATE_ID,
            robot_id=request.robot_id,
            robot_asset_id=request.robot_asset_id,
            robot_digest=request.robot_asset_id.rsplit(":", 1)[-1],
            reference=request.reference,
            baseline="urdf_zero",
            joint_q={"left_shoulder_roll_joint": 1.2},
        )
        return CalibrationProposalResponse(
            candidate=candidate,
            validation=_calibration_validation(),
        )

    def validate(self, _request: Any) -> CalibrationValidationReport:
        self.calls.append("validate")
        return _calibration_validation()

    def preview(self, _request: Any) -> tuple[CalibrationPreview, bytes]:
        self.calls.append("preview")
        payload = b"\x89PNG\r\n\x1a\nvision-test"
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

    def save(self, request: Any) -> CalibrationSaveReceipt:
        self.calls.append("save")
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


class _Exports:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def export(self, job_id: str, artifact_id: str) -> ArtifactExportReceipt:
        self.calls.append((job_id, artifact_id))
        job_token = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
        artifact_token = hashlib.sha256(artifact_id.encode("utf-8")).hexdigest()
        return ArtifactExportReceipt(
            relative_path=f"jobs/{job_token}/{artifact_token}.csv",
            job_id=job_id,
            artifact_id=artifact_id,
            kind="retargeted_motion",
            format="csv",
            media_type="text/csv",
            size_bytes=3,
            sha256=_DIGEST,
        )


class _Jobs:
    def __init__(self) -> None:
        self.start_calls: list[tuple[str, str]] = []
        self.get_calls: list[tuple[str, int | None]] = []
        self.wait_calls: list[tuple[str, int, float]] = []
        self.lookup_calls: list[tuple[str, str, int | None]] = []
        self.list_calls: list[tuple[str, int, int]] = []
        self.artifact_calls: list[tuple[str, str, bool]] = []

    def start_retarget(self, plan_id: str, *, idempotency_key: str) -> AgentJobView:
        self.start_calls.append((plan_id, idempotency_key))
        return _job()

    def lookup_job(
        self,
        plan_id: str,
        *,
        idempotency_key: str,
        after_revision: int | None = None,
    ) -> AgentJobView:
        self.lookup_calls.append((plan_id, idempotency_key, after_revision))
        return _job()

    def get_job(
        self,
        job_id: str,
        *,
        after_revision: int | None = None,
    ) -> AgentJobView:
        self.get_calls.append((job_id, after_revision))
        if job_id != _JOB_ID:
            raise JobManagerError(
                ApiError(
                    code="JOB_NOT_FOUND",
                    message="No canonical job has the requested id.",
                    stage=ErrorStage.ARTIFACT,
                )
            )
        return _job()

    def wait_job(
        self,
        job_id: str,
        *,
        after_revision: int,
        timeout: float = 30.0,
    ) -> AgentJobView:
        self.wait_calls.append((job_id, after_revision, timeout))
        return self.get_job(job_id, after_revision=after_revision)

    def cancel_job(self, job_id: str) -> AgentJobView:
        return self.get_job(job_id)

    def retry_job(self, job_id: str, *, idempotency_key: str) -> AgentJobView:
        del idempotency_key
        return self.get_job(job_id)

    def list_artifacts(
        self,
        job_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> list[ArtifactDescriptor]:
        self.list_calls.append((job_id, offset, limit))
        if job_id != _JOB_ID:
            raise JobManagerError(
                ApiError(
                    code="JOB_NOT_FOUND",
                    message="No canonical job has the requested id.",
                    stage=ErrorStage.ARTIFACT,
                )
            )
        return [_artifact()][offset : offset + limit]

    def get_artifact(
        self,
        job_id: str,
        artifact_id: str,
        *,
        verify: bool = False,
    ) -> Any:
        self.artifact_calls.append((job_id, artifact_id, verify))
        if job_id != _JOB_ID or artifact_id != _ARTIFACT_ID:
            raise JobManagerError(
                ApiError(
                    code="ARTIFACT_NOT_FOUND",
                    message="The artifact is not canonically attached to this job.",
                    stage=ErrorStage.ARTIFACT,
                    details={"job_id": job_id},
                )
            )
        return SimpleNamespace(descriptor=_artifact())


class _Fixture:
    def __init__(self) -> None:
        self.capabilities = _CapabilitiesService()
        self.assets = _AssetsService()
        self.available_assets = _AvailableAssetsService()
        self.preflight = _PreflightService()
        self.r2r_preflight = _R2RPreflightService()
        self.batch_preflight = cast(Any, object())
        self.calibration = _CalibrationService()
        self.plans = _Plans()
        self.jobs = _Jobs()
        self.exports = _Exports()
        self.runtime = AgentRuntime(
            capabilities=cast(Any, self.capabilities),
            assets=cast(Any, self.assets),
            available_assets=cast(Any, self.available_assets),
            preflight=cast(Any, self.preflight),
            r2r_preflight=self.r2r_preflight,
            batch_preflight=self.batch_preflight,
            calibration=cast(Any, self.calibration),
            plans=cast(Any, self.plans),
            jobs=cast(Any, self.jobs),
            exports=cast(Any, self.exports),
        )

    def server(self):
        @asynccontextmanager
        async def runtime_factory() -> AsyncIterator[AgentRuntime]:
            yield self.runtime

        return create_mcp_server(runtime_factory=runtime_factory)

    def replace_jobs(self, jobs: Any) -> None:
        self.jobs = jobs
        self.runtime = AgentRuntime(
            capabilities=cast(Any, self.capabilities),
            assets=cast(Any, self.assets),
            available_assets=cast(Any, self.available_assets),
            preflight=cast(Any, self.preflight),
            r2r_preflight=self.r2r_preflight,
            batch_preflight=self.batch_preflight,
            calibration=cast(Any, self.calibration),
            plans=cast(Any, self.plans),
            jobs=cast(Any, jobs),
            exports=cast(Any, self.exports),
        )


def _tool_by_name(tools: list[Any], name: str) -> Any:
    return next(tool for tool in tools if tool.name == name)


def _assert_invalid_parameter_result(result: Any, *forbidden_values: str) -> None:
    expected = ApiError(
        code="INVALID_PARAMETER",
        message="The MCP tool arguments do not match the advertised schema.",
        retryable=False,
        stage=ErrorStage.REQUEST,
    ).model_dump(mode="json", exclude_none=True)
    assert result.is_error is True
    assert result.structured_content == expected
    assert json.loads(result.content[0].text) == expected
    serialized = json.dumps(result.model_dump(mode="json"))
    assert all(value not in serialized for value in forbidden_values)
    assert "pydantic" not in serialized.casefold()
    assert "errors.pydantic.dev" not in serialized.casefold()


@pytest.mark.anyio
async def test_mcp_enumerates_only_bounded_tools_and_resources() -> None:
    fixture = _Fixture()

    async with Client(fixture.server(), raise_exceptions=True) as client:
        tools = (await client.list_tools()).tools
        resources = await client.list_resources()
        templates = await client.list_resource_templates()

    assert {tool.name for tool in tools} == _EXPECTED_TOOLS
    assert {str(resource.uri) for resource in resources.resources} == _EXPECTED_RESOURCES
    assert {
        str(template.uri_template) for template in templates.resource_templates
    } == _EXPECTED_RESOURCE_TEMPLATES

    serialized = json.dumps(
        [tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in tools]
    ).casefold()
    for forbidden in (
        "base64",
        "shell",
        "run_command",
        "source_path",
        "output_path",
        "absolute_path",
        '"argv"',
        '"command"',
    ):
        assert forbidden not in serialized


@pytest.mark.anyio
async def test_mcp_lists_available_assets_with_closed_bounded_schema() -> None:
    fixture = _Fixture()

    async with Client(fixture.server(), raise_exceptions=True) as client:
        tools = (await client.list_tools()).tools
        result = await client.call_tool(
            "list_available_assets",
            {
                "root_id": "source",
                "query": "walk",
                "kind": "motion_bundle",
                "limit": 5,
                "offset": 0,
            },
        )

    tool = _tool_by_name(tools, "list_available_assets")
    assert tool.input_schema["additionalProperties"] is False
    assert tool.input_schema["properties"]["limit"]["maximum"] == 500
    assert tool.output_schema["title"] == "AvailableAssetCatalogResponse"
    assert result.is_error is False
    assert result.structured_content["assets"][0]["relative_path"] == "AMASS/walk.npz"
    assert fixture.available_assets.calls[0].root_id == "source"
    assert fixture.available_assets.calls[0].limit == 5


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool_name", "arguments", "rejected_value"),
    [
        ("list_available_assets", {"limit": 501}, "501"),
        ("list_available_assets", {"root_id": "../private"}, "../private"),
        (
            "get_capabilities",
            {"unexpected": "https://private.invalid/token?secret=mcp-validation"},
            "https://private.invalid/token?secret=mcp-validation",
        ),
    ],
)
async def test_mcp_returns_versioned_errors_for_invalid_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    rejected_value: str,
) -> None:
    fixture = _Fixture()

    async with Client(fixture.server(), raise_exceptions=True) as client:
        result = await client.call_tool(tool_name, arguments)

    _assert_invalid_parameter_result(result, rejected_value)
    assert fixture.available_assets.calls == []
    assert fixture.capabilities.calls == 0


@pytest.mark.anyio
async def test_real_mcp_catalog_entries_register_and_inspect_without_host_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def local_tmpdir(tag: str) -> Path:
        path = tmp_path / f"runtime-{tag}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    source_root = tmp_path / "motions"
    source_root.mkdir()
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
    monkeypatch.setattr(server_state, "_tmpdir", local_tmpdir)
    monkeypatch.setattr(server_state, "_robot_library_root", lambda: tmp_path / "robots")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv(
        "HHTOOLS_MOTION_LIBRARY_SETTINGS_PATH",
        str(tmp_path / "motion-library-settings.json"),
    )
    clear_cache()
    app = web_server.create_app(
        source_root=source_root,
        save_dir=tmp_path / "save",
        cache_dir=tmp_path / "cache",
        job_history_dir=tmp_path / "history",
        job_settings_path=tmp_path / "job-settings.json",
        agent_mcp_available=True,
        agent_rest_available=False,
        agent_json_cli_available=False,
    )

    @asynccontextmanager
    async def runtime_factory() -> AsyncIterator[AgentRuntime]:
        async with app.router.lifespan_context(app):
            yield AgentRuntime.from_application(app)

    try:
        async with Client(
            create_mcp_server(runtime_factory=runtime_factory),
            raise_exceptions=True,
        ) as client:
            motion_catalog = await client.call_tool(
                "list_available_assets",
                {"root_id": "source", "kind": "motion_bundle"},
            )
            robot_catalog = await client.call_tool(
                "list_available_assets",
                {"root_id": "robot-library", "kind": "robot_bundle"},
            )
            motion_entry = motion_catalog.structured_content["assets"][0]
            robot_entry = robot_catalog.structured_content["assets"][0]

            async def register_and_inspect(entry: dict[str, Any]) -> Any:
                request = {
                    field: entry[field]
                    for field in (
                        "root_id",
                        "relative_path",
                        "display_name",
                        "kind",
                        "category",
                        "recursive",
                    )
                }
                registered = await client.call_tool(
                    "register_asset_bundle",
                    {"request": request},
                )
                return await client.call_tool(
                    "inspect_asset_bundle",
                    {
                        "request": {
                            "asset_id": registered.structured_content["asset_id"],
                            "verify_hashes": True,
                            "parse_content": False,
                        }
                    },
                )

            motion_inspection = await register_and_inspect(motion_entry)
            robot_inspection = await register_and_inspect(robot_entry)
    finally:
        clear_cache()

    assert motion_entry["relative_path"] == "walk.npz"
    assert robot_entry["relative_path"] == "catalog_bot"
    assert motion_inspection.structured_content["status"] == "valid"
    assert robot_inspection.structured_content["status"] == "valid"
    serialized = json.dumps([motion_entry, robot_entry])
    assert str(tmp_path) not in serialized


@pytest.mark.anyio
async def test_mcp_tool_schemas_are_generated_from_public_pydantic_contracts() -> None:
    fixture = _Fixture()

    async with Client(fixture.server(), raise_exceptions=True) as client:
        tools = (await client.list_tools()).tools

    assert all(tool.input_schema["additionalProperties"] is False for tool in tools)

    register = _tool_by_name(tools, "register_asset_bundle")
    registration = register.input_schema["$defs"]["AssetRegistrationRequest"]
    assert register.input_schema["required"] == ["request"]
    assert set(registration["properties"]) == {
        "schema_version",
        "root_id",
        "relative_path",
        "display_name",
        "kind",
        "category",
        "recursive",
    }
    assert registration["additionalProperties"] is False

    start = _tool_by_name(tools, "start_retarget")
    start_request = start.input_schema["$defs"]["JobStartRequest"]
    assert set(start_request["properties"]) == {
        "schema_version",
        "plan_id",
        "idempotency_key",
    }
    assert "run_mode" not in json.dumps(start.input_schema)

    wait = _tool_by_name(tools, "wait_job")
    assert set(wait.input_schema["required"]) == {"job_id", "after_revision"}
    assert wait.input_schema["properties"]["after_revision"]["minimum"] == 0
    assert wait.input_schema["properties"]["timeout"]["minimum"] == 0.0
    assert wait.input_schema["properties"]["timeout"]["maximum"] == 60.0

    batch = _tool_by_name(tools, "preflight_batch")
    batch_request = batch.input_schema["$defs"]["BatchPreflightRequest"]
    assert "maxItems" not in batch_request["properties"]["item_plan_ids"]
    assert batch_request["additionalProperties"] is False

    calibration = _tool_by_name(tools, "propose_calibration")
    calibration_request = calibration.input_schema["$defs"]["CalibrationProposalRequest"]
    assert calibration_request["additionalProperties"] is False
    assert "robot_asset_id" in calibration_request["required"]
    preview = _tool_by_name(tools, "preview_calibration")
    assert preview.output_schema["title"] == "CalibrationPreview"
    save_calibration = _tool_by_name(tools, "save_calibration")
    assert save_calibration.annotations.destructive_hint is False
    r2r_calibration = _tool_by_name(tools, "propose_r2r_calibration")
    r2r_request = r2r_calibration.input_schema["$defs"]["R2RCalibrationProposalRequest"]
    assert r2r_request["additionalProperties"] is False
    assert {
        "source_robot_asset_id",
        "target_robot_asset_id",
    } <= set(r2r_request["required"])
    r2r_preview = _tool_by_name(tools, "preview_r2r_calibration")
    assert r2r_preview.output_schema["title"] == "R2RCalibrationPreview"
    r2r_save = _tool_by_name(tools, "save_r2r_calibration")
    assert r2r_save.annotations.destructive_hint is False

    capabilities = _tool_by_name(tools, "get_capabilities")
    assert capabilities.output_schema["title"] == "CapabilityResponse"
    assert "features" in capabilities.output_schema["properties"]


@pytest.mark.anyio
async def test_mcp_rejects_unknown_arguments_without_echoing_their_values() -> None:
    fixture = _Fixture()
    sensitive_value = r"C:\Users\Nora\secret.txt"

    async with Client(fixture.server(), raise_exceptions=True) as client:
        result = await client.call_tool(
            "search_assets",
            {"request": {"query": sensitive_value, "limit": 5}},
        )

    _assert_invalid_parameter_result(result, sensitive_value)


@pytest.mark.anyio
async def test_capabilities_report_mcp_true_for_tool_and_resource() -> None:
    fixture = _Fixture()

    async with Client(fixture.server(), raise_exceptions=True) as client:
        tool_result = await client.call_tool("get_capabilities", {})
        resource_result = await client.read_resource("hhtools://capabilities")

    assert tool_result.is_error is False
    assert tool_result.structured_content["features"] == {
        "agent_rest": False,
        "json_cli": False,
        "mcp": True,
    }
    resource_document = json.loads(resource_result.contents[0].text)
    assert resource_document["features"] == {
        "agent_rest": False,
        "json_cli": False,
        "mcp": True,
    }
    assert fixture.capabilities.calls == 2


@pytest.mark.anyio
async def test_calibration_tools_return_a_vision_preview_and_silent_save_receipt() -> None:
    fixture = _Fixture()
    identity = {
        "schema_version": "1.0",
        "robot_id": "g1_29dof",
        "robot_asset_id": _ASSET_ID,
        "reference": "smplx",
    }

    async with Client(fixture.server(), raise_exceptions=True) as client:
        status = await client.call_tool(
            "get_calibration_status",
            {"request": identity},
        )
        proposal = await client.call_tool(
            "propose_calibration",
            {"request": identity},
        )
        candidate_id = proposal.structured_content["candidate"]["candidate_id"]
        validation = await client.call_tool(
            "validate_calibration",
            {"request": {"schema_version": "1.0", "candidate_id": candidate_id}},
        )
        preview = await client.call_tool(
            "preview_calibration",
            {"request": {"schema_version": "1.0", "candidate_id": candidate_id}},
        )
        saved = await client.call_tool(
            "save_calibration",
            {
                "request": {
                    "schema_version": "1.0",
                    "candidate_id": candidate_id,
                    "save_mode": "gpt_vision_silent",
                    "visual_review": {
                        "reviewer": "gpt_vision",
                        "verdict": "pass",
                        "model_hint": "gpt-test",
                        "summary": "The front and side overlays are aligned.",
                    },
                }
            },
        )

    assert status.structured_content["state"] == "missing"
    assert validation.structured_content["valid"] is True
    assert preview.structured_content["media_type"] == "image/png"
    assert [content.type for content in preview.content] == ["text", "image"]
    assert preview.content[1].mime_type == "image/png"
    assert saved.structured_content["saved"] is True
    assert fixture.calibration.calls == ["status", "propose", "validate", "preview", "save"]


@pytest.mark.anyio
async def test_human_action_preflight_never_starts_a_job() -> None:
    fixture = _Fixture()
    request = {
        "schema_version": "1.0",
        "motion_asset_id": _ASSET_ID,
        "robot_id": "g1_29dof",
        "robot_asset_id": _ASSET_ID,
        "parameters": {"run_mode": "smoke"},
    }

    async with Client(fixture.server(), raise_exceptions=True) as client:
        result = await client.call_tool("preflight_retarget", {"request": request})

    assert result.is_error is False
    assert result.structured_content["status"] == "human_action_required"
    action = result.structured_content["required_actions"][0]
    assert action["actor"] == "human"
    assert action["action"] == "open_calibration_ui"
    assert action["url"].startswith("http://127.0.0.1:8009/")
    assert len(fixture.preflight.calls) == 1
    assert fixture.jobs.start_calls == []


@pytest.mark.anyio
async def test_r2r_preflight_and_generic_start_use_the_shared_job_lifecycle() -> None:
    fixture = _Fixture()
    request = {
        "schema_version": "1.0",
        "trajectory_asset_id": _ASSET_ID,
        "source_robot_id": "source_bot",
        "source_robot_asset_id": f"asset:sha256:{'b' * 64}",
        "target_robot_id": "target_bot",
        "target_robot_asset_id": f"asset:sha256:{'c' * 64}",
    }

    async with Client(fixture.server(), raise_exceptions=True) as client:
        preflight = await client.call_tool("preflight_r2r", {"request": request})
        started = await client.call_tool(
            "start_job",
            {
                "request": {
                    "schema_version": "1.0",
                    "plan_id": _PLAN_ID,
                    "idempotency_key": "r2r-mcp-test",
                }
            },
        )

    assert preflight.is_error is False
    assert preflight.structured_content["status"] == "human_action_required"
    assert preflight.structured_content["recommended_backend"] == "newton"
    assert fixture.r2r_preflight.calls[0].source_robot_id == "source_bot"
    assert started.is_error is False
    assert fixture.jobs.start_calls == [(_PLAN_ID, "r2r-mcp-test")]


@pytest.mark.anyio
async def test_revision_polling_forwards_after_revision_and_stays_compact() -> None:
    fixture = _Fixture()

    async with Client(fixture.server(), raise_exceptions=True) as client:
        result = await client.call_tool(
            "get_job",
            {"job_id": _JOB_ID, "after_revision": 7},
        )

    assert result.is_error is False
    assert result.structured_content["progress"]["revision"] == 7
    assert result.structured_content["poll_after_ms"] == 750
    assert fixture.jobs.get_calls == [(_JOB_ID, 7)]
    serialized = json.dumps(result.structured_content).casefold()
    assert "trajectory" not in serialized
    assert "base64" not in serialized


@pytest.mark.anyio
async def test_revision_wait_forwards_revision_and_bounded_timeout() -> None:
    fixture = _Fixture()

    async with Client(fixture.server(), raise_exceptions=True) as client:
        result = await client.call_tool(
            "wait_job",
            {"job_id": _JOB_ID, "after_revision": 7, "timeout": 12.5},
        )

    assert result.is_error is False
    assert result.structured_content["progress"]["revision"] == 7
    assert fixture.jobs.wait_calls == [(_JOB_ID, 7, 12.5)]


@pytest.mark.anyio
async def test_lookup_job_recovers_only_the_caller_owned_submission() -> None:
    fixture = _Fixture()

    async with Client(fixture.server(), raise_exceptions=True) as client:
        result = await client.call_tool(
            "lookup_job",
            {
                "request": {
                    "schema_version": "1.0",
                    "plan_id": _PLAN_ID,
                    "idempotency_key": "caller-owned-key",
                    "after_revision": 7,
                }
            },
        )

    assert result.is_error is False
    assert result.structured_content["job_id"] == _JOB_ID
    assert fixture.jobs.lookup_calls == [(_PLAN_ID, "caller-owned-key", 7)]


@pytest.mark.anyio
async def test_artifacts_remain_job_scoped_and_errors_are_structured() -> None:
    fixture = _Fixture()

    async with Client(fixture.server(), raise_exceptions=True) as client:
        page = await client.call_tool(
            "list_job_artifacts",
            {"job_id": _JOB_ID, "limit": 25, "offset": 0},
        )
        resource = await client.read_resource(f"hhtools://jobs/{_JOB_ID}/artifacts/{_ARTIFACT_ID}")
        denied = await client.call_tool(
            "list_job_artifacts",
            {"job_id": "job-other", "limit": 25, "offset": 0},
        )

    assert page.is_error is False
    assert page.structured_content["job_id"] == _JOB_ID
    assert page.structured_content["artifacts"][0]["artifact_id"] == _ARTIFACT_ID
    assert fixture.jobs.list_calls == [
        (_JOB_ID, 0, 25),
        ("job-other", 0, 25),
    ]

    resource_document = json.loads(resource.contents[0].text)
    assert resource_document["job_id"] == _JOB_ID
    assert resource_document["artifact_id"] == _ARTIFACT_ID
    assert fixture.jobs.artifact_calls == [(_JOB_ID, _ARTIFACT_ID, True)]
    assert "base64" not in json.dumps(resource_document).casefold()
    assert "path" not in resource_document

    assert denied.is_error is True
    assert denied.structured_content["schema_version"] == "1.0"
    assert denied.structured_content["code"] == "JOB_NOT_FOUND"
    assert denied.structured_content["stage"] == "artifact"
    assert json.loads(denied.content[0].text) == denied.structured_content


@pytest.mark.anyio
async def test_export_artifact_returns_only_a_portable_delivery_receipt() -> None:
    fixture = _Fixture()

    async with Client(fixture.server(), raise_exceptions=True) as client:
        result = await client.call_tool(
            "export_artifact",
            {"job_id": _JOB_ID, "artifact_id": _ARTIFACT_ID},
        )

    assert result.is_error is False
    receipt = result.structured_content
    assert receipt["root_id"] == "agent-exports"
    assert receipt["relative_path"].startswith("jobs/")
    assert receipt["sha256"] == _DIGEST
    assert fixture.exports.calls == [(_JOB_ID, _ARTIFACT_ID)]
    serialized = json.dumps(receipt).casefold()
    assert "base64" not in serialized
    assert "artifact-objects" not in serialized
    assert "c:\\" not in serialized


@pytest.mark.anyio
async def test_report_hash_covers_the_exact_payload_returned(tmp_path: Path) -> None:
    fixture = _Fixture()
    original = json.dumps(
        {
            "schema_version": "1.0",
            "job_id": _JOB_ID,
            "outcome": "success",
            "summary": "A",
            "created_at": _NOW.isoformat(),
        },
        separators=(",", ":"),
    ).encode()
    replacement = original.replace(b'"summary":"A"', b'"summary":"B"')
    assert len(original) == len(replacement)
    report_file = tmp_path / "evaluation.json"
    report_file.write_bytes(original)
    mutating_path = _MutatingReportPath(report_file, replacement)
    descriptor = ArtifactDescriptor(
        artifact_id="artifact:evaluation_report:mcp-test",
        job_id=_JOB_ID,
        kind="evaluation_report",
        format="json",
        resource_uri=f"hhtools://jobs/{_JOB_ID}/artifacts/evaluation-report",
        media_type="application/json",
        size_bytes=len(replacement),
        sha256=hashlib.sha256(replacement).hexdigest(),
    )

    class _ReportJobs:
        def get_job(self, _job_id: str) -> Any:
            return SimpleNamespace(artifact_count=1)

        def list_artifacts(self, _job_id: str, *, offset: int, limit: int) -> list[Any]:
            return [descriptor][offset : offset + limit]

        def get_artifact(
            self,
            _job_id: str,
            _artifact_id: str,
            *,
            verify: bool,
        ) -> Any:
            assert verify is False
            return SimpleNamespace(descriptor=descriptor, path=mutating_path)

    fixture.replace_jobs(_ReportJobs())
    async with Client(fixture.server(), raise_exceptions=True) as client:
        with pytest.raises(MCPError) as raised:
            await client.read_resource(f"hhtools://jobs/{_JOB_ID}/evaluation")

    error = json.loads(raised.value.message)
    assert error["code"] == "ARTIFACT_HASH_MISMATCH"
    # The fixed reader hashes the payload it already read and never starts a
    # second pass that a concurrent writer could swap underneath it.
    assert mutating_path.seek_calls == 0
    assert report_file.read_bytes() == original


@pytest.mark.anyio
async def test_oversized_batch_report_remains_exportable_instead_of_looking_corrupt() -> None:
    fixture = _Fixture()
    descriptor = ArtifactDescriptor(
        artifact_id="artifact:batch_report:mcp-test",
        job_id=_JOB_ID,
        kind="batch_report",
        format="json",
        resource_uri=f"hhtools://jobs/{_JOB_ID}/artifacts/batch-report",
        media_type="application/json",
        size_bytes=2 * 1024 * 1024 + 1,
        sha256=_DIGEST,
    )

    class _OversizedReportJobs:
        def get_job(self, _job_id: str) -> Any:
            return SimpleNamespace(artifact_count=1)

        def list_artifacts(self, _job_id: str, *, offset: int, limit: int) -> list[Any]:
            return [descriptor][offset : offset + limit]

        def get_artifact(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("oversized reports must not be loaded into model context")

    fixture.replace_jobs(_OversizedReportJobs())
    async with Client(fixture.server(), raise_exceptions=True) as client:
        with pytest.raises(MCPError) as raised:
            await client.read_resource(f"hhtools://jobs/{_JOB_ID}/batch")

    error = json.loads(raised.value.message)
    assert error["code"] == "REPORT_TOO_LARGE"
    assert error["stage"] == "artifact"


@pytest.mark.anyio
async def test_non_portable_service_errors_are_sanitized_for_tools_and_resources() -> None:
    fixture = _Fixture()
    secret_path = r"C:\Users\Nora\private\robot.npz"

    class _UnsafeJobs(_Jobs):
        def get_job(
            self,
            _job_id: str,
            *,
            after_revision: int | None = None,
        ) -> AgentJobView:
            del after_revision
            raise JobManagerError(
                ApiError(
                    code="SERVICE_FAILURE",
                    message=f"Could not read {secret_path}",
                    stage=ErrorStage.INTERNAL,
                    details={"source_path": secret_path},
                )
            )

    fixture.replace_jobs(_UnsafeJobs())
    async with Client(fixture.server(), raise_exceptions=True) as client:
        tool_result = await client.call_tool("get_job", {"job_id": _JOB_ID})
        with pytest.raises(MCPError) as raised:
            await client.read_resource(f"hhtools://jobs/{_JOB_ID}/status")

    expected = {
        "schema_version": "1.0",
        "code": "INTERNAL_ERROR",
        "message": "The HHTools MCP service could not complete the request.",
        "retryable": True,
        "stage": "internal",
        "details": {},
    }
    assert tool_result.is_error is True
    assert tool_result.structured_content == expected
    assert json.loads(tool_result.content[0].text) == expected
    resource_error = json.loads(raised.value.message)
    assert resource_error == expected
    assert secret_path not in json.dumps(tool_result.model_dump(mode="json"))
    assert secret_path not in raised.value.message


@pytest.mark.anyio
async def test_non_portable_success_payloads_are_sanitized_for_tools_and_resources() -> None:
    fixture = _Fixture()
    secret_path = r"C:\Users\Nora\private\robot.npz"

    class _UnsafeSuccessJobs(_Jobs):
        def get_job(
            self,
            _job_id: str,
            *,
            after_revision: int | None = None,
        ) -> AgentJobView:
            del after_revision
            return _job().model_copy(update={"summary": {"source_path": secret_path}})

    fixture.replace_jobs(_UnsafeSuccessJobs())
    async with Client(fixture.server(), raise_exceptions=True) as client:
        tool_result = await client.call_tool("get_job", {"job_id": _JOB_ID})
        with pytest.raises(MCPError) as raised:
            await client.read_resource(f"hhtools://jobs/{_JOB_ID}/status")

    expected_code = "INTERNAL_ERROR"
    assert tool_result.is_error is True
    assert tool_result.structured_content["code"] == expected_code
    assert json.loads(raised.value.message)["code"] == expected_code
    assert secret_path not in json.dumps(tool_result.model_dump(mode="json"))
    assert secret_path not in raised.value.message


@pytest.mark.anyio
async def test_real_stdio_subprocess_preserves_framing() -> None:
    fixture_server = Path(__file__).with_name("stdio_fixture_server.py")
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(fixture_server)],
        cwd=Path(__file__).parents[2],
        encoding="utf-8",
        encoding_error_handler="strict",
    )

    with anyio.fail_after(20):
        async with Client(parameters, read_timeout_seconds=10) as client:
            tools = await client.list_tools()
            result = await client.call_tool("get_capabilities", {})
            protocol_version = client.protocol_version

    # A successful discovery and tool round-trip over the child process proves
    # stdout contained only valid MCP frames; any prose or traceback corrupts
    # negotiation before these assertions are reachable.
    assert protocol_version == "2026-07-28"
    assert {tool.name for tool in tools.tools} == _EXPECTED_TOOLS
    assert result.is_error is False
    assert result.structured_content["service_version"] == "stdio-test"
    assert result.structured_content["features"]["mcp"] is True
