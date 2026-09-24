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
from hhtools.contracts import JobManifest
from hhtools.core.scene import SceneObject
from hhtools.io.npz import save_npz
from hhtools.mcp.runtime import AgentRuntime
from hhtools.mcp.server import create_mcp_server
from hhtools.retarget.calibration import RobotRetargetCalibration, save_calibration


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
    for reference in ("smpl", "smplx"):
        save_calibration(
            RobotRetargetCalibration(
                robot=model.preset.name,
                reference=reference,
                calibrated_joint_q={name: 0.0 for name in model.dof_names()},
                notes="self-contained Agent Interaction-Mesh fixture",
            ),
            target / f"retarget_calibration_{reference}.yaml",
        )
    return target


def _object_motion(terrain_motion):
    motion = deepcopy(terrain_motion)
    motion.name = "synthetic_object_smoke"
    motion.terrain = None
    motion.meta = {"dataset": "omomo"}
    object_quaternions = np.zeros((motion.num_frames, 4), dtype=np.float32)
    object_quaternions[:, 3] = 1.0
    motion.objects = [
        SceneObject(
            name="box",
            positions=np.full((motion.num_frames, 3), (0.4, 0.0, 0.8), dtype=np.float32),
            quaternions=object_quaternions,
            extents=np.array([0.2, 0.2, 0.2], dtype=np.float32),
            mesh_path="box_cleaned_simplified.obj",
        )
    ]
    return motion


def _write_motion_bundle(motion, source_root: Path, *, dataset: str) -> Path:
    target = source_root / dataset / motion.name
    target.mkdir(parents=True)
    motion_path = target / f"{motion.name}.npz"
    save_npz(motion, motion_path)
    if motion.terrain is not None:
        (target / f"{motion.name}_terrain.obj").write_text(
            "v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nf 1 2 3\nf 1 3 4\n",
            encoding="utf-8",
        )
    for scene_object in motion.objects:
        if scene_object.mesh_path:
            (target / Path(scene_object.mesh_path).name).write_text(
                "v -.1 -.1 -.1\nv .1 -.1 -.1\nv 0 .1 -.1\nv 0 0 .1\n"
                "f 1 2 3\nf 1 2 4\nf 2 3 4\nf 3 1 4\n",
                encoding="utf-8",
            )
    return motion_path


@pytest.mark.parametrize("scene_kind", ["terrain", "object"])
def test_agent_runs_self_contained_interaction_mesh_plan_to_verified_artifacts(
    tmp_path: Path,
    grounding_robot_pair,
    synthetic_terrain_motion,
    scene_kind: str,
) -> None:
    model = grounding_robot_pair[1]
    source_root = tmp_path / "source"
    robot_root = tmp_path / "robots"
    if scene_kind == "terrain":
        source_motion = synthetic_terrain_motion
        dataset = "parc_ms"
    else:
        source_motion = _object_motion(synthetic_terrain_motion)
        dataset = "OMOMO"
    motion_path = _write_motion_bundle(source_motion, source_root, dataset=dataset)
    robot_path = _write_robot_bundle(model, robot_root)
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
                motion_result = await client.call_tool(
                    "register_asset_bundle",
                    {
                        "request": {
                            "schema_version": "1.0",
                            "root_id": "source",
                            "relative_path": motion_path.relative_to(source_root).as_posix(),
                        }
                    },
                )
                robot_result = await client.call_tool(
                    "register_asset_bundle",
                    {
                        "request": {
                            "schema_version": "1.0",
                            "root_id": "robot-library",
                            "relative_path": robot_path.relative_to(robot_root).as_posix(),
                            "kind": "robot_bundle",
                        }
                    },
                )
                calibration = await client.call_tool(
                    "get_calibration_status",
                    {
                        "request": {
                            "robot_id": model.preset.name,
                            "robot_asset_id": robot_result.structured_content["asset_id"],
                            "reference": "smpl" if scene_kind == "terrain" else "smplx",
                        }
                    },
                )
                assert calibration.structured_content["state"] == "valid"
                preflight_result = await client.call_tool(
                    "preflight_retarget",
                    {
                        "request": {
                            "schema_version": "1.0",
                            "motion_asset_id": motion_result.structured_content["asset_id"],
                            "robot_id": model.preset.name,
                            "robot_asset_id": robot_result.structured_content["asset_id"],
                            "output_format": "csv",
                            "parameters": {
                                "run_mode": "smoke",
                                "limit_frames": 1,
                                "foot_clamp_anti_penetration": False,
                            },
                        }
                    },
                )
                preflight = preflight_result.structured_content
                assert preflight["status"] == "ready", preflight
                assert preflight["plan"]["backend"] == "interaction_mesh"
                assert preflight["plan"]["output_format"] == "csv"
                assert "ik_iterations" not in preflight["plan"]["parameters"]

                started = await client.call_tool(
                    "start_retarget",
                    {
                        "request": {
                            "schema_version": "1.0",
                            "plan_id": preflight["plan"]["plan_id"],
                            "idempotency_key": f"interaction-mesh-e2e-{scene_kind}",
                        }
                    },
                )
                view = started.structured_content
                deadline = time.monotonic() + 15.0
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

                assert view["state"] == "completed"
                assert view["outcome"] == "review_required"
                page = await client.call_tool(
                    "list_job_artifacts",
                    {"job_id": view["job_id"], "limit": 100},
                )
                artifacts = page.structured_content["artifacts"]
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
                assert manifest.execution_provenance.backend == "interaction_mesh"
                assert manifest.execution_provenance.runtime == "mujoco"
                assert manifest.execution_provenance.device_kind == "cpu"
                assert manifest.summary["has_scene"] is True

                motion_artifact = next(
                    artifact for artifact in artifacts if artifact["kind"] == "retargeted_motion"
                )
                assert motion_artifact["format"] == "zip"
                assert motion_artifact["media_type"] == "application/zip"
                assert motion_artifact["metadata"]["requested_format"] == "csv"
                assert motion_artifact["metadata"]["content_format"] == "zip"
                receipt = await client.call_tool(
                    "export_artifact",
                    {
                        "job_id": view["job_id"],
                        "artifact_id": motion_artifact["artifact_id"],
                    },
                )
                assert receipt.is_error is False
                receipt_document = receipt.structured_content
                assert receipt_document["job_id"] == view["job_id"]
                assert receipt_document["root_id"] == "agent-exports"
                assert receipt_document["format"] == "zip"
                assert receipt_document["sha256"] == motion_artifact["sha256"]
                assert receipt_document["size_bytes"] == motion_artifact["size_bytes"]
                assert str(tmp_path) not in json.dumps(receipt_document)

                relative_export = Path(receipt_document["relative_path"])
                assert not relative_export.is_absolute()
                assert ".." not in relative_export.parts
                exported_path = paths.save_dir / "agent-exports" / relative_export
                with zipfile.ZipFile(exported_path) as archive:
                    members = set(archive.namelist())
                    if scene_kind == "terrain":
                        assert members == {
                            "synthetic_terrain_smoke.csv",
                            "synthetic_terrain_smoke_terrain.obj",
                        }
                    else:
                        assert members == {
                            "box_cleaned_simplified.obj",
                            "object_0_box.csv",
                            "synthetic_object_smoke.csv",
                        }
                    assert str(tmp_path).encode() not in b"".join(
                        archive.read(member) for member in sorted(members)
                    )

    asyncio.run(exercise())
