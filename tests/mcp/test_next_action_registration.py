"""Live MCP replay coverage for executable preflight asset actions."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from mcp import Client

from hhtools.contracts import (
    AssetRegistrationRequest,
    BackendCapability,
    CapabilityResponse,
    DeviceCapability,
    RetargetPreflightRequest,
    SchedulerCapability,
)
from hhtools.mcp.runtime import AgentRuntime
from hhtools.mcp.server import create_mcp_server
from hhtools.robot.registry import preset_from_dir
from hhtools.services.asset_service import AgentAssetService
from hhtools.services.assets import AssetRegistry
from hhtools.services.plans import PlanStore
from hhtools.services.preflight import PreflightService

_NOW = datetime(2026, 8, 31, 2, 0, tzinfo=UTC)


def _write_motion(path: Path) -> None:
    path.parent.mkdir(parents=True)
    positions = np.zeros((12, 2, 3), dtype=np.float32)
    quaternions = np.zeros((12, 2, 4), dtype=np.float32)
    quaternions[..., 3] = 1.0
    np.savez(
        path,
        schema_version=np.array("1"),
        name=np.array("walk"),
        framerate=np.array(30.0),
        up_axis=np.array("Z"),
        source_format=np.array("npz"),
        bone_names=np.array(["root", "joint"]),
        parent_indices=np.array([-1, 0], dtype=np.int32),
        positions=positions,
        quaternions=quaternions,
    )


def _write_robot(root: Path) -> None:
    urdf_dir = root / "urdf"
    urdf_dir.mkdir(parents=True)
    (root / "robot.yaml").write_text(
        "name: test_robot\n"
        "display_name: Test Robot\n"
        "urdf: urdf/robot.urdf\n"
        "dof_order: [hip]\n"
        "ik_map:\n"
        "  hips: base\n",
        encoding="utf-8",
    )
    (urdf_dir / "robot.urdf").write_text(
        """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base"/>
  <link name="torso"/>
  <joint name="hip" type="revolute">
    <parent link="base"/>
    <child link="torso"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="10" velocity="2"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    (urdf_dir / "retarget_calibration_smpl.yaml").write_text(
        "robot: test_robot\n"
        "reference: smpl\n"
        "calibrated_joint_q:\n"
        "  hip: 0.0\n"
        "notes: live MCP fixture\n",
        encoding="utf-8",
    )


def _capabilities() -> CapabilityResponse:
    return CapabilityResponse(
        service_version="next-action-test",
        backends=[
            BackendCapability(
                backend_id="newton",
                display_name="Newton IK",
                available=True,
                supported_categories=["plain_motion"],
                output_formats=["csv", "pkl"],
            )
        ],
        devices=[
            DeviceCapability(
                device_id="cpu",
                kind="cpu",
                display_name="Test CPU",
                available=True,
            )
        ],
        scheduler=SchedulerCapability(
            max_running_jobs=0,
            max_queued_jobs=0,
            mode="unlimited",
        ),
        supported_output_formats=["csv", "pkl"],
        features={"preflight": True, "mcp": True},
    )


def _fixture(
    tmp_path: Path,
) -> tuple[PreflightService, AgentAssetService, PlanStore, RetargetPreflightRequest]:
    motion_root = tmp_path / "motions"
    robot_root = tmp_path / "robots"
    robot = robot_root / "test_robot"
    _write_motion(motion_root / "walk.npz")
    _write_robot(robot)
    assets = AgentAssetService(
        AssetRegistry(
            tmp_path / "agent-state",
            {"motions": motion_root, "robots": robot_root},
        )
    )
    motion = assets.register(
        AssetRegistrationRequest(
            root_id="motions",
            relative_path="walk.npz",
            display_name=None,
        )
    )
    plans = PlanStore(tmp_path / "plans")
    preflight = PreflightService(
        assets,
        plans,
        capabilities_provider=_capabilities,
        robot_provider=lambda: [preset_from_dir(robot)],
        clock=lambda: _NOW,
        request_id_provider=lambda: "request-next-action-mcp",
    )
    request = RetargetPreflightRequest(
        motion_asset_id=motion.asset_id,
        robot_id="test_robot",
        robot_asset_id=None,
        parameters={"run_mode": "smoke"},
    )
    return preflight, assets, plans, request


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_robot_registration_next_action_replays_unchanged_through_live_mcp(
    tmp_path: Path,
) -> None:
    preflight, assets, plans, initial_request = _fixture(tmp_path)
    runtime = AgentRuntime(
        capabilities=cast(Any, object()),
        assets=assets,
        available_assets=cast(Any, object()),
        preflight=preflight,
        r2r_preflight=cast(Any, object()),
        batch_preflight=cast(Any, object()),
        plans=plans,
        jobs=cast(Any, object()),
        exports=cast(Any, object()),
    )

    @asynccontextmanager
    async def runtime_factory() -> AsyncIterator[AgentRuntime]:
        yield runtime

    server = create_mcp_server(runtime_factory=runtime_factory)

    async with Client(server, raise_exceptions=True) as client:
        tools = await client.list_tools()
        rejected = await client.call_tool(
            "preflight_retarget",
            {"request": initial_request.model_dump(mode="json")},
        )
        action = rejected.structured_content["error"]["next_action"]
        registration = AssetRegistrationRequest.model_validate(action["parameters"]["request"])
        registered = await client.call_tool(action["action"], action["parameters"])
        converged_request = initial_request.model_copy(
            update={"robot_asset_id": registered.structured_content["asset_id"]}
        )
        ready = await client.call_tool(
            "preflight_retarget",
            {"request": converged_request.model_dump(mode="json")},
        )

    assert rejected.is_error is False
    assert rejected.structured_content["status"] == "rejected"
    assert action["actor"] == "agent"
    assert action["action"] == "register_asset_bundle"
    assert set(action["parameters"]) == {"request"}
    assert registration.root_id == "robots"
    assert registration.relative_path == "test_robot"
    assert registered.is_error is False
    assert registered.structured_content["kind"] == "robot_bundle"
    assert ready.is_error is False
    assert ready.structured_content["status"] == "human_action_required"
    assert ready.structured_content["required_actions"][0]["action"] == "get_calibration_status"
    assert (
        ready.structured_content["required_actions"][0]["parameters"]["request"]["robot_asset_id"]
        == registered.structured_content["asset_id"]
    )
    tool_schemas = json.dumps(
        [{"name": tool.name, "input_schema": tool.input_schema} for tool in tools.tools]
    )
    assert "trusted_path" not in tool_schemas
    assert "registration_hint" not in tool_schemas
    serialized = json.dumps(
        [
            rejected.structured_content,
            registered.structured_content,
            ready.structured_content,
        ]
    )
    assert str(tmp_path) not in serialized
    assert rejected.structured_content["checks"][-1]["next_action"] == action
