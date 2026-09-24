from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from mcp import Client

from hhtools.application.runtime import ApplicationPaths, build_application_runtime
from hhtools.io.npz import save_npz
from hhtools.mcp.runtime import AgentRuntime
from hhtools.mcp.server import create_mcp_server
from hhtools.services.retarget import RetargetServiceError


def _write_robot_bundle(model, root: Path) -> Path:
    target = root / model.preset.name
    target.mkdir(parents=True)
    shutil.copy2(model.preset.urdf_path, target / "robot.urdf")
    (target / "robot.yaml").write_text(
        yaml.safe_dump(
            {
                "name": model.preset.name,
                "display_name": model.preset.display_name,
                "urdf": "robot.urdf",
                "dof_order": list(model.dof_names()),
                "ik_map": dict(model.preset.ik_map),
                "feet": dict(model.preset.feet),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return target


@pytest.mark.parametrize("shared_robot_root", [False, True])
def test_vision_agent_can_silently_calibrate_then_preflight(
    tmp_path: Path,
    monkeypatch,
    grounding_robot_pair,
    synthetic_terrain_motion,
    shared_robot_root: bool,
) -> None:
    _source_model, target_model = grounding_robot_pair
    source_root = tmp_path / "source"
    robot_root = tmp_path / "robots"
    user_robot_root = robot_root if shared_robot_root else tmp_path / "user-robots"
    source_root.mkdir()
    user_robot_root.mkdir()
    monkeypatch.setenv("HHTOOLS_ROBOT_DIR", str(user_robot_root))
    robot_path = _write_robot_bundle(target_model, robot_root)
    motion = synthetic_terrain_motion
    motion.terrain = None
    motion.meta = {"dataset": "unified_npz"}
    motion_path = source_root / "walk.npz"
    save_npz(motion, motion_path)

    paths = replace(
        ApplicationPaths.isolated(tmp_path / "runtime", source_root=source_root),
        robot_root=robot_root,
        workspace_robot_root=tmp_path / "workspace-robots",
    )
    runtime = build_application_runtime(
        paths,
        max_running_jobs=1,
        max_queued_jobs=1,
        agent_mcp_available=True,
        agent_rest_available=False,
        agent_json_cli_available=False,
    )

    async def exercise() -> None:
        async with runtime.lifespan():

            @asynccontextmanager
            async def runtime_factory() -> AsyncIterator[AgentRuntime]:
                yield AgentRuntime.from_services(runtime.services)

            async with Client(
                create_mcp_server(runtime_factory=runtime_factory),
                raise_exceptions=True,
            ) as client:
                robot = await client.call_tool(
                    "register_asset_bundle",
                    {
                        "request": {
                            "schema_version": "1.0",
                            "root_id": "robot-library",
                            "relative_path": robot_path.name,
                            "kind": "robot_bundle",
                        }
                    },
                )
                motion_asset = await client.call_tool(
                    "register_asset_bundle",
                    {
                        "request": {
                            "schema_version": "1.0",
                            "root_id": "source",
                            "relative_path": motion_path.name,
                        }
                    },
                )
                identity = {
                    "schema_version": "1.0",
                    "robot_id": target_model.preset.name,
                    "robot_asset_id": robot.structured_content["asset_id"],
                    "reference": "smpl",
                }
                preflight_request = {
                    "schema_version": "1.0",
                    "motion_asset_id": motion_asset.structured_content["asset_id"],
                    "robot_id": target_model.preset.name,
                    "robot_asset_id": robot.structured_content["asset_id"],
                    "parameters": {"run_mode": "smoke", "limit_frames": 1},
                }

                blocked = await client.call_tool(
                    "preflight_retarget",
                    {"request": preflight_request},
                )
                status = await client.call_tool(
                    "get_calibration_status",
                    {"request": identity},
                )
                proposal = await client.call_tool(
                    "propose_calibration",
                    {"request": identity},
                )
                candidate_id = proposal.structured_content["candidate"]["candidate_id"]
                candidate_request = {
                    "schema_version": "1.0",
                    "candidate_id": candidate_id,
                }
                preview = await client.call_tool(
                    "preview_calibration",
                    {"request": candidate_request},
                )
                saved = await client.call_tool(
                    "save_calibration",
                    {
                        "request": {
                            **candidate_request,
                            "save_mode": "gpt_vision_silent",
                            "visual_review": {
                                "reviewer": "gpt_vision",
                                "verdict": "pass",
                                "model_hint": "gpt-e2e",
                                "summary": "Front and side landmark overlays are coherent.",
                            },
                        }
                    },
                )
                current = await client.call_tool(
                    "get_calibration_status",
                    {"request": identity},
                )
                ready = await client.call_tool(
                    "preflight_retarget",
                    {"request": preflight_request},
                )
                assert (
                    ready.structured_content["plan"]["calibration_digest"]
                    == (saved.structured_content["calibration_digest"])
                )
                plan_id = ready.structured_content["plan"]["plan_id"]
                projector = runtime.services.agent_h2r_retarget_service
                spec = projector.get_job_spec(plan_id)
                assert spec.calibration.sha256 == saved.structured_content["calibration_digest"]
                directory = (
                    user_robot_root
                    / (".calibration-overlays" if shared_robot_root else ".")
                    / target_model.preset.name
                )
                calibration = directory / "retarget_calibration_smpl.yaml"
                calibration.write_bytes(calibration.read_bytes() + b"\n# changed after preflight\n")
                with pytest.raises(RetargetServiceError) as stale:
                    projector.get_job_spec(plan_id)
                assert stale.value.code == "PLAN_STALE"

        assert blocked.structured_content["status"] == "human_action_required"
        action = blocked.structured_content["required_actions"][0]
        assert action["actor"] == "agent"
        assert action["action"] == "get_calibration_status"
        assert status.structured_content["state"] == "missing"
        assert proposal.structured_content["validation"]["valid"] is True
        assert [content.type for content in preview.content] == ["text", "image"]
        assert saved.structured_content["saved"] is True
        assert current.structured_content["state"] == "valid"
        assert ready.structured_content["status"] == "ready"

    try:
        asyncio.run(exercise())
    finally:
        runtime.scheduler.shutdown(wait=True)

    assert (
        user_robot_root
        / (".calibration-overlays" if shared_robot_root else ".")
        / target_model.preset.name
        / "retarget_calibration_smpl.yaml"
    ).is_file()
