from __future__ import annotations

import asyncio
import json
import shutil
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from mcp import Client

from hhtools.application.runtime import ApplicationPaths, build_application_runtime
from hhtools.contracts import JobManifest
from hhtools.io.robot_csv import save_robot_csv
from hhtools.mcp.runtime import AgentRuntime
from hhtools.mcp.server import create_mcp_server
from hhtools.retarget.robot_to_robot import save_r2r_calibration


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


def test_agent_runs_self_contained_r2r_plan_to_verified_artifacts(
    tmp_path: Path,
    monkeypatch,
    grounding_robot_pair,
) -> None:
    source_model, target_model = grounding_robot_pair
    source_root = tmp_path / "source"
    robot_root = tmp_path / "robots"
    user_robot_root = tmp_path / "user-robots"
    source_root.mkdir()
    user_robot_root.mkdir()
    monkeypatch.setenv("HHTOOLS_ROBOT_DIR", str(user_robot_root))
    source_robot_path = _write_robot_bundle(source_model, robot_root)
    target_robot_path = _write_robot_bundle(target_model, robot_root)
    save_r2r_calibration(
        target_robot_path,
        target_robot=target_model.preset.name,
        source_robot=source_model.preset.name,
        calibrated_joint_q={name: 0.0 for name in target_model.dof_names()},
        user_root=user_robot_root,
    )
    trajectory_path = source_root / "walk.csv"
    joint_q = np.zeros((1, 7 + len(source_model.dof_names())), dtype=np.float32)
    joint_q[:, 2] = 1.0
    joint_q[:, 6] = 1.0
    save_robot_csv(
        trajectory_path,
        robot=source_model,
        joint_q=joint_q,
        sample_rate=50.0,
    )

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

            server = create_mcp_server(runtime_factory=runtime_factory)
            async with Client(server, raise_exceptions=True) as client:
                catalog = await client.call_tool(
                    "list_available_assets",
                    {"root_id": "source", "kind": "robot_trajectory_bundle"},
                )
                assert catalog.structured_content["total"] == 1
                assert catalog.structured_content["assets"][0]["relative_path"] == "walk.csv"
                assert (
                    catalog.structured_content["assets"][0]["category"]
                    == "robot_trajectory"
                )
                trajectory = await client.call_tool(
                    "register_asset_bundle",
                    {
                        "request": {
                            "schema_version": "1.0",
                            "root_id": "source",
                            "relative_path": trajectory_path.name,
                        }
                    },
                )
                source_robot = await client.call_tool(
                    "register_asset_bundle",
                    {
                        "request": {
                            "schema_version": "1.0",
                            "root_id": "robot-library",
                            "relative_path": source_robot_path.name,
                            "kind": "robot_bundle",
                        }
                    },
                )
                target_robot = await client.call_tool(
                    "register_asset_bundle",
                    {
                        "request": {
                            "schema_version": "1.0",
                            "root_id": "robot-library",
                            "relative_path": target_robot_path.name,
                            "kind": "robot_bundle",
                        }
                    },
                )
                preflight_result = await client.call_tool(
                    "preflight_r2r",
                    {
                        "request": {
                            "schema_version": "1.0",
                            "trajectory_asset_id": trajectory.structured_content["asset_id"],
                            "source_robot_id": source_model.preset.name,
                            "source_robot_asset_id": source_robot.structured_content["asset_id"],
                            "target_robot_id": target_model.preset.name,
                            "target_robot_asset_id": target_robot.structured_content["asset_id"],
                            "output_format": "csv",
                            "parameters": {
                                "run_mode": "smoke",
                                "limit_frames": 1,
                            },
                        }
                    },
                )
                preflight = preflight_result.structured_content
                assert preflight["status"] == "ready", preflight
                assert preflight["plan"]["backend"] == "newton"
                assert preflight["plan"]["source_robot_id"] == source_model.preset.name
                assert preflight["plan"]["target_robot_id"] == target_model.preset.name

                started = await client.call_tool(
                    "start_job",
                    {
                        "request": {
                            "schema_version": "1.0",
                            "plan_id": preflight["plan"]["plan_id"],
                            "idempotency_key": "r2r-e2e-smoke",
                        }
                    },
                )
                view = started.structured_content
                deadline = time.monotonic() + 30.0
                while view["state"] in {"queued", "running"}:
                    remaining = deadline - time.monotonic()
                    assert remaining > 0, view
                    waited = await client.call_tool(
                        "wait_job",
                        {
                            "job_id": view["job_id"],
                            "after_revision": view["progress"]["revision"],
                            "timeout": min(2.0, remaining),
                        },
                    )
                    view = waited.structured_content

                assert view["state"] == "completed", view
                assert view["outcome"] == "review_required"
                artifacts_result = await client.call_tool(
                    "list_job_artifacts",
                    {"job_id": view["job_id"], "limit": 100},
                )
                artifacts = artifacts_result.structured_content["artifacts"]
                assert {
                    "job_spec",
                    "preview",
                    "retargeted_motion",
                    "evaluation_report",
                    "manifest",
                }.issubset({artifact["kind"] for artifact in artifacts})
                manifest_resource = await client.read_resource(
                    f"hhtools://jobs/{view['job_id']}/manifest"
                )
                manifest = JobManifest.model_validate(
                    json.loads(manifest_resource.contents[0].text)
                )
                assert manifest.job_spec.kind.value == "r2r_retarget"
                assert manifest.job_spec.source_robot is not None
                assert manifest.job_spec.source_robot.robot_id == source_model.preset.name
                assert manifest.job_spec.robot.robot_id == target_model.preset.name
                assert manifest.execution_provenance.backend == "newton"
                assert (
                    manifest.execution_provenance.source_robot_asset_id
                    == source_robot.structured_content["asset_id"]
                )
                assert (
                    manifest.execution_provenance.robot_asset_id
                    == target_robot.structured_content["asset_id"]
                )

                motion_artifact = next(
                    artifact for artifact in artifacts if artifact["kind"] == "retargeted_motion"
                )
                assert motion_artifact["format"] == "csv"
                receipt = await client.call_tool(
                    "export_artifact",
                    {
                        "job_id": view["job_id"],
                        "artifact_id": motion_artifact["artifact_id"],
                    },
                )
                assert receipt.is_error is False
                assert receipt.structured_content["root_id"] == "agent-exports"
                assert receipt.structured_content["format"] == "csv"
                assert str(tmp_path) not in json.dumps(receipt.structured_content)

    asyncio.run(exercise())
