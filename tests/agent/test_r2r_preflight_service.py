from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from hhtools.contracts import (
    ApiError,
    AssetCategory,
    AssetRegistrationRequest,
    BackendCapability,
    CapabilityResponse,
    DeviceCapability,
    ErrorStage,
    JobSpecKind,
    PreflightStatus,
    R2RPreflightRequest,
    SchedulerCapability,
)
from hhtools.io.robot_csv import save_robot_csv
from hhtools.retarget.robot_to_robot import save_r2r_calibration
from hhtools.robot.registry import list_presets_in_root_readonly
from hhtools.services.asset_service import AgentAssetService
from hhtools.services.assets import AssetRegistry
from hhtools.services.plans import PlanStore, PlanStoreError
from hhtools.services.r2r_preflight import R2RPreflightService
from hhtools.services.r2r_retarget import R2RRetargetService
from hhtools.services.retarget import RetargetServiceError


def _robot_bundle(model, root: Path) -> Path:
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


def _capabilities() -> CapabilityResponse:
    return CapabilityResponse(
        service_version="test",
        backends=[
            BackendCapability(
                backend_id="newton",
                display_name="Newton IK",
                available=True,
                supported_categories=[AssetCategory.ROBOT_TRAJECTORY],
                output_formats=["csv", "pkl"],
                limits={
                    "max_ik_iterations": 200,
                    "max_retarget_fps": 1_000.0,
                    "max_retarget_frames": 100_000,
                },
            )
        ],
        devices=[
            DeviceCapability(
                device_id="cpu",
                kind="cpu",
                display_name="CPU",
                available=True,
            )
        ],
        scheduler=SchedulerCapability(
            max_running_jobs=1,
            max_queued_jobs=1,
            mode="limited",
        ),
        supported_output_formats=["csv", "pkl"],
    )


def _setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grounding_robot_pair,
    *,
    calibration: bool = True,
    declared_source: str | None = None,
    scene: bool = False,
):
    source_model, target_model = grounding_robot_pair
    robot_root = tmp_path / "robots"
    motion_root = tmp_path / "trajectories"
    user_root = tmp_path / "user-robots"
    motion_root.mkdir()
    user_root.mkdir()
    monkeypatch.setenv("HHTOOLS_ROBOT_DIR", str(user_root))
    source_dir = _robot_bundle(source_model, robot_root)
    target_dir = _robot_bundle(target_model, robot_root)
    if calibration:
        save_r2r_calibration(
            target_dir,
            target_robot=target_model.preset.name,
            source_robot=source_model.preset.name,
            calibrated_joint_q={name: 0.0 for name in target_model.dof_names()},
            user_root=user_root,
        )

    clip_dir = motion_root / "walk" if scene else motion_root
    clip_dir.mkdir(exist_ok=True)
    trajectory_path = clip_dir / "walk.csv"
    joint_q = np.zeros((2, 7 + len(source_model.dof_names())), dtype=np.float32)
    joint_q[:, 2] = 1.0
    joint_q[:, 6] = 1.0
    save_robot_csv(
        trajectory_path,
        robot=source_model,
        joint_q=joint_q,
        sample_rate=50.0,
        meta=(
            {"robot": declared_source}
            if declared_source is not None
            else {"robot": source_model.preset.name}
        ),
    )
    if scene:
        (clip_dir / "walk_terrain.obj").write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
            encoding="utf-8",
        )

    assets = AgentAssetService(
        AssetRegistry(
            tmp_path / "state",
            {"source": motion_root, "robots": robot_root},
        )
    )
    trajectory = assets.register(
        AssetRegistrationRequest(
            root_id="source",
            relative_path="walk" if scene else "walk.csv",
        )
    )
    source_bundle = assets.register(
        AssetRegistrationRequest(
            root_id="robots",
            relative_path=source_dir.name,
            kind="robot_bundle",
        )
    )
    target_bundle = assets.register(
        AssetRegistrationRequest(
            root_id="robots",
            relative_path=target_dir.name,
            kind="robot_bundle",
        )
    )
    plans = PlanStore(tmp_path / "state")
    service = R2RPreflightService(
        assets,
        plans,
        capabilities_provider=_capabilities,
        robot_provider=lambda: list_presets_in_root_readonly(robot_root),
        request_id_provider=lambda: "req_r2r_test",
    )
    request = R2RPreflightRequest(
        trajectory_asset_id=trajectory.asset_id,
        source_robot_id=source_model.preset.name,
        source_robot_asset_id=source_bundle.asset_id,
        target_robot_id=target_model.preset.name,
        target_robot_asset_id=target_bundle.asset_id,
        parameters={"run_mode": "smoke", "limit_frames": 1},
    )
    return service, plans, request, assets


def test_r2r_preflight_freezes_trajectory_robot_pair_and_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grounding_robot_pair,
) -> None:
    service, plans, request, _assets = _setup(
        tmp_path,
        monkeypatch,
        grounding_robot_pair,
    )

    response = service.preflight_r2r(request)

    assert response.status is PreflightStatus.READY
    assert response.plan is not None
    assert response.plan.trajectory_asset_id == request.trajectory_asset_id
    assert response.plan.source_robot_id == request.source_robot_id
    assert response.plan.source_robot_asset_id == request.source_robot_asset_id
    assert response.plan.target_robot_id == request.target_robot_id
    assert response.plan.target_robot_asset_id == request.target_robot_asset_id
    assert response.plan.backend == "newton"
    assert response.plan.calibration_id.startswith("cal:sha256:")
    assert response.plan.parameters == {
        "run_mode": "smoke",
        "limit_frames": 1,
        "ik_iterations": 24,
        "source_fps": None,
        "retarget_fps": None,
        "trajectory_profile": "mimic",
    }
    payload = plans.get_payload(response.plan.plan_id)
    assert payload["semantics"] == "hhtools.r2r.plan.v1"
    assert payload["source_robot"]["asset_id"] == request.source_robot_asset_id
    assert payload["target_robot"]["asset_id"] == request.target_robot_asset_id
    assert str(tmp_path) not in response.model_dump_json()


def test_missing_pair_calibration_returns_exact_agent_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grounding_robot_pair,
) -> None:
    service, _plans, request, _assets = _setup(
        tmp_path,
        monkeypatch,
        grounding_robot_pair,
        calibration=False,
    )

    response = service.preflight_r2r(request)

    assert response.status is PreflightStatus.HUMAN_ACTION_REQUIRED
    assert response.plan is None
    action = response.required_actions[0]
    assert action.actor == "agent"
    assert action.action == "get_r2r_calibration_status"
    assert action.parameters == {
        "request": {
            "schema_version": "1.0",
            "source_robot_id": request.source_robot_id,
            "source_robot_asset_id": request.source_robot_asset_id,
            "target_robot_id": request.target_robot_id,
            "target_robot_asset_id": request.target_robot_asset_id,
        }
    }


def test_declared_source_robot_mismatch_is_rejected_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grounding_robot_pair,
) -> None:
    service, _plans, request, _assets = _setup(
        tmp_path,
        monkeypatch,
        grounding_robot_pair,
        declared_source="another_robot",
    )

    response = service.preflight_r2r(request)

    assert response.status is PreflightStatus.REJECTED
    assert response.error is not None
    assert response.error.code == "SOURCE_ROBOT_MISMATCH"


def test_scene_trajectory_is_rejected_by_initial_r2r_agent_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grounding_robot_pair,
) -> None:
    service, _plans, request, _assets = _setup(
        tmp_path,
        monkeypatch,
        grounding_robot_pair,
        scene=True,
    )

    response = service.preflight_r2r(request)

    assert response.status is PreflightStatus.REJECTED
    assert response.error is not None
    assert response.error.code == "R2R_SCENE_UNSUPPORTED"


def test_ready_r2r_plan_projects_to_a_two_robot_jobspec_and_detects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grounding_robot_pair,
) -> None:
    service, plans, request, assets = _setup(
        tmp_path,
        monkeypatch,
        grounding_robot_pair,
    )
    response = service.preflight_r2r(request)
    assert response.plan is not None
    projector = R2RRetargetService(
        plans,
        assets,
        provenance_provider=lambda: {
            "hhtools_git_commit": "test",
            "hhtools_dirty": False,
            "python": "3.12",
        },
    )

    spec = projector.get_job_spec(response.plan.plan_id)

    assert spec.kind is JobSpecKind.R2R_RETARGET
    assert spec.inputs[0].asset_id == request.trajectory_asset_id
    assert spec.source_robot is not None
    assert spec.source_robot.robot_id == request.source_robot_id
    assert spec.source_robot.asset_id == request.source_robot_asset_id
    assert spec.robot.robot_id == request.target_robot_id
    assert spec.robot.asset_id == request.target_robot_asset_id
    assert spec.calibration is not None
    assert spec.calibration.calibration_id == response.plan.calibration_id
    assert spec.effective_parameters["output_format"] == "csv"

    assets.resolve_primary(request.trajectory_asset_id).write_text(
        "changed after preflight\n",
        encoding="utf-8",
    )
    with pytest.raises(RetargetServiceError) as captured:
        projector.get_job_spec(response.plan.plan_id)
    assert captured.value.code == "PLAN_STALE"


def test_identical_concurrent_preflight_recovers_the_winning_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grounding_robot_pair,
) -> None:
    service, plans, request, _assets = _setup(
        tmp_path,
        monkeypatch,
        grounding_robot_pair,
    )
    first = service.preflight_r2r(request)
    assert first.plan is not None
    original_get = plans.get
    miss_once = True

    def stale_read(plan_id: str):
        nonlocal miss_once
        if miss_once:
            miss_once = False
            raise PlanStoreError(
                ApiError(
                    code="PLAN_NOT_FOUND",
                    message="Simulated concurrent cache miss.",
                    stage=ErrorStage.PREFLIGHT,
                )
            )
        return original_get(plan_id)

    monkeypatch.setattr(plans, "get", stale_read)

    repeated = service.preflight_r2r(request)

    assert repeated.status is PreflightStatus.READY
    assert repeated.plan == first.plan
