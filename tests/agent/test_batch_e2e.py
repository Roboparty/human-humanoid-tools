from __future__ import annotations

import asyncio
import json
import shutil
import time
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from mcp import Client

from hhtools.application.runtime import ApplicationPaths, build_application_runtime
from hhtools.contracts import BatchReport, JobManifest
from hhtools.io.npz import save_npz
from hhtools.io.robot_csv import save_robot_csv
from hhtools.mcp.runtime import AgentRuntime
from hhtools.mcp.server import create_mcp_server
from hhtools.retarget.calibration import RobotRetargetCalibration, save_calibration
from hhtools.retarget.robot_to_robot import save_r2r_calibration


def _write_robot_bundle(
    model,
    root: Path,
    *,
    h2r_calibration: bool,
) -> Path:
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
    if h2r_calibration:
        save_calibration(
            RobotRetargetCalibration(
                robot=model.preset.name,
                reference="smpl",
                calibrated_joint_q={name: 0.0 for name in model.dof_names()},
                notes="self-contained batch fixture",
            ),
            target / "retarget_calibration_smpl.yaml",
        )
    return target


async def _register(client: Client, root_id: str, relative_path: str, *, kind=None):
    request = {
        "schema_version": "1.0",
        "root_id": root_id,
        "relative_path": relative_path,
    }
    if kind is not None:
        request["kind"] = kind
    result = await client.call_tool("register_asset_bundle", {"request": request})
    return result.structured_content["asset_id"]


@pytest.mark.parametrize("workflow", ["h2r", "r2r"])
def test_agent_executes_bounded_two_item_batch_with_portable_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grounding_robot_pair,
    synthetic_terrain_motion,
    workflow: str,
) -> None:
    source_model, target_model = grounding_robot_pair
    source_root = tmp_path / "source"
    robot_root = tmp_path / "robots"
    user_robot_root = tmp_path / "user-robots"
    source_root.mkdir()
    user_robot_root.mkdir()
    monkeypatch.setenv("HHTOOLS_ROBOT_DIR", str(user_robot_root))
    target_path = _write_robot_bundle(
        target_model,
        robot_root,
        h2r_calibration=workflow == "h2r",
    )
    source_robot_path = None
    if workflow == "r2r":
        source_robot_path = _write_robot_bundle(
            source_model,
            robot_root,
            h2r_calibration=False,
        )
        save_r2r_calibration(
            target_path,
            target_robot=target_model.preset.name,
            source_robot=source_model.preset.name,
            calibrated_joint_q={name: 0.0 for name in target_model.dof_names()},
            user_root=user_robot_root,
        )

    input_paths: list[Path] = []
    for index in range(2):
        if workflow == "h2r":
            motion = deepcopy(synthetic_terrain_motion)
            motion.name = f"batch_motion_{index}"
            motion.terrain = None
            motion.meta = {"dataset": "unified_npz"}
            path = source_root / f"batch_motion_{index}.npz"
            save_npz(motion, path)
        else:
            path = source_root / f"batch_trajectory_{index}.csv"
            joint_q = np.zeros(
                (1, 7 + len(source_model.dof_names())),
                dtype=np.float32,
            )
            joint_q[:, 0] = index * 0.05
            joint_q[:, 2] = 1.0
            joint_q[:, 6] = 1.0
            save_robot_csv(
                path,
                robot=source_model,
                joint_q=joint_q,
                sample_rate=50.0,
            )
        input_paths.append(path)

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
                target_asset_id = await _register(
                    client,
                    "robot-library",
                    target_path.name,
                    kind="robot_bundle",
                )
                source_robot_asset_id = (
                    await _register(
                        client,
                        "robot-library",
                        source_robot_path.name,
                        kind="robot_bundle",
                    )
                    if source_robot_path is not None
                    else None
                )
                child_plan_ids: list[str] = []
                for path in input_paths:
                    input_asset_id = await _register(client, "source", path.name)
                    if workflow == "h2r":
                        calibration = await client.call_tool(
                            "get_calibration_status",
                            {
                                "request": {
                                    "robot_id": target_model.preset.name,
                                    "robot_asset_id": target_asset_id,
                                    "reference": "smpl",
                                }
                            },
                        )
                        assert calibration.structured_content["state"] == "valid"
                        result = await client.call_tool(
                            "preflight_retarget",
                            {
                                "request": {
                                    "schema_version": "1.0",
                                    "motion_asset_id": input_asset_id,
                                    "robot_id": target_model.preset.name,
                                    "robot_asset_id": target_asset_id,
                                    "parameters": {
                                        "run_mode": "smoke",
                                        "limit_frames": 1,
                                        "foot_clamp_anti_penetration": False,
                                    },
                                }
                            },
                        )
                    else:
                        result = await client.call_tool(
                            "preflight_r2r",
                            {
                                "request": {
                                    "schema_version": "1.0",
                                    "trajectory_asset_id": input_asset_id,
                                    "source_robot_id": source_model.preset.name,
                                    "source_robot_asset_id": source_robot_asset_id,
                                    "target_robot_id": target_model.preset.name,
                                    "target_robot_asset_id": target_asset_id,
                                    "parameters": {
                                        "run_mode": "smoke",
                                        "limit_frames": 1,
                                    },
                                }
                            },
                        )
                    assert result.structured_content["status"] == "ready"
                    child_plan_ids.append(result.structured_content["plan"]["plan_id"])

                batch_preflight = await client.call_tool(
                    "preflight_batch",
                    {
                        "request": {
                            "schema_version": "1.0",
                            "workflow": workflow,
                            "item_plan_ids": child_plan_ids,
                        }
                    },
                )
                batch = batch_preflight.structured_content
                assert batch["status"] == "ready", batch
                assert batch["plan"]["resource_limits"]["item_count"] == 2
                assert batch["plan"]["resource_limits"]["estimated_total_frames"] == 2

                started = await client.call_tool(
                    "start_job",
                    {
                        "request": {
                            "schema_version": "1.0",
                            "plan_id": batch["plan"]["plan_id"],
                            "idempotency_key": f"batch-{workflow}-e2e",
                        }
                    },
                )
                view = started.structured_content
                deadline = time.monotonic() + 45.0
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
                page = await client.call_tool(
                    "list_job_artifacts",
                    {"job_id": view["job_id"], "limit": 100},
                )
                artifacts = page.structured_content["artifacts"]
                kinds = [artifact["kind"] for artifact in artifacts]
                assert kinds.count("preview") == 2
                assert kinds.count("retargeted_motion") == 2
                assert "batch_report" in kinds
                assert "batch_archive" in kinds

                batch_resource = await client.read_resource(
                    f"hhtools://jobs/{view['job_id']}/batch"
                )
                report = BatchReport.model_validate(json.loads(batch_resource.contents[0].text))
                assert report.workflow.value == workflow
                assert report.total_items == report.succeeded_items == 2
                assert report.failed_items == 0
                assert [item.plan_id for item in report.items] == child_plan_ids

                manifest_resource = await client.read_resource(
                    f"hhtools://jobs/{view['job_id']}/manifest"
                )
                manifest = JobManifest.model_validate(
                    json.loads(manifest_resource.contents[0].text)
                )
                assert manifest.job_spec.kind.value == "batch_retarget"
                assert manifest.summary["completed_items"] == 2

                archive_artifact = next(
                    artifact for artifact in artifacts if artifact["kind"] == "batch_archive"
                )
                receipt = await client.call_tool(
                    "export_artifact",
                    {
                        "job_id": view["job_id"],
                        "artifact_id": archive_artifact["artifact_id"],
                    },
                )
                assert receipt.structured_content["format"] == "zip"
                relative = Path(receipt.structured_content["relative_path"])
                exported = paths.save_dir / "agent-exports" / relative
                with zipfile.ZipFile(exported) as archive:
                    members = archive.namelist()
                    assert "batch-report.json" in members
                    assert len([name for name in members if name.endswith(".csv")]) == 2
                assert str(tmp_path) not in json.dumps(receipt.structured_content)

    asyncio.run(exercise())
