from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from hhtools.contracts import (
    AssetInspectionRequest,
    AssetRegistrationRequest,
    JobSpecProvenance,
    OutputPolicy,
    RetargetPlan,
)
from hhtools.services.asset_service import AgentAssetService
from hhtools.services.assets import AssetRegistry
from hhtools.services.plans import PlanStore, compute_plan_id
from hhtools.services.retarget import RetargetService, RetargetServiceError

NOW = datetime(2026, 8, 31, 4, 0, tzinfo=UTC)


@dataclass(frozen=True)
class _Fixture:
    assets: AgentAssetService
    store: PlanStore
    plan: RetargetPlan
    payload: dict[str, object]
    motion_path: Path
    robot_urdf: Path
    calibration_path: Path | None


class _RecordingAssets:
    def __init__(self, delegate: AgentAssetService) -> None:
        self._delegate = delegate
        self.inspections: list[AssetInspectionRequest] = []

    def get(self, asset_id: str):
        return self._delegate.get(asset_id)

    def inspect(self, request: AssetInspectionRequest):
        self.inspections.append(request)
        return self._delegate.inspect(request)


def _write_motion(path: Path) -> None:
    path.parent.mkdir(parents=True)
    positions = np.zeros((48, 2, 3), dtype=np.float32)
    quaternions = np.zeros((48, 2, 4), dtype=np.float32)
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


def _write_robot(root: Path) -> tuple[Path, Path]:
    urdf = root / "urdf" / "robot.urdf"
    scaler = root / "config" / "smpl_scaler.yaml"
    urdf.parent.mkdir(parents=True)
    scaler.parent.mkdir(parents=True)
    scaler.write_text(
        "human_height_assumption: 1.7\n"
        "model_height: 1.0\n"
        "joint_scales:\n"
        "  hips: 1.0\n"
        "joint_offsets: {}\n",
        encoding="utf-8",
    )
    (root / "robot.yaml").write_text(
        "name: test_robot\n"
        "display_name: Test Robot\n"
        "urdf: urdf/robot.urdf\n"
        "dof_order: [hip]\n"
        "ik_map:\n"
        "  hips: base\n"
        "retarget:\n"
        "  references:\n"
        "    smpl:\n"
        "      scaler_config: config/smpl_scaler.yaml\n",
        encoding="utf-8",
    )
    urdf.write_text(
        """<?xml version="1.0"?>
<robot name="test_robot">
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
    return urdf, scaler


def _write_calibration(root: Path) -> Path:
    path = root / "urdf" / "retarget_calibration_smpl.yaml"
    path.write_text(
        "robot: test_robot\nreference: smpl\ncalibrated_joint_q:\n  hip: 0.0\n",
        encoding="utf-8",
    )
    return path


def _write_user_calibration(root: Path) -> Path:
    path = root / "test_robot" / "retarget_calibration_smpl.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "robot: test_robot\nreference: smpl\ncalibrated_joint_q:\n  hip: 0.0\n",
        encoding="utf-8",
    )
    return path


def _fixed_provenance() -> JobSpecProvenance:
    return JobSpecProvenance(
        hhtools_git_commit="f" * 40,
        hhtools_dirty=False,
        python="3.13.7",
        pytorch="2.8.0",
        cuda="12.8",
        newton="1.0.0",
        # RetargetService must discard this current device selection because
        # it has not yet admitted or started an execution.
        device="cuda:7",
        platform="Linux-x86_64",
        dependencies={"z-package": "2", "a-package": "1"},
    )


def _fixture(
    tmp_path: Path,
    *,
    profile_source: str = "bundled_scaler",
    profile_storage: str = "robot_bundle",
    semantics: str = "hhtools.retarget.plan.v1",
    category: str | None = None,
) -> _Fixture:
    motion_root = tmp_path / "motions"
    robot_root = tmp_path / "robots"
    motion_path = motion_root / "walk.npz"
    robot_dir = robot_root / "test_robot"
    _write_motion(motion_path)
    robot_urdf, scaler = _write_robot(robot_dir)
    if profile_source == "calibration" and profile_storage == "user_calibration":
        calibration_path = _write_user_calibration(tmp_path / "user-robots")
    elif profile_source == "calibration":
        calibration_path = _write_calibration(robot_dir)
    else:
        calibration_path = None

    assets = AgentAssetService(
        AssetRegistry(
            tmp_path / "asset-state",
            {"motions": motion_root, "robots": robot_root},
        )
    )
    motion = assets.register(AssetRegistrationRequest(root_id="motions", relative_path="walk.npz"))
    robot = assets.register(
        AssetRegistrationRequest(
            root_id="robots",
            relative_path="test_robot",
            kind="robot_bundle",
        )
    )
    motion_inspection = assets.inspect(AssetInspectionRequest(asset_id=motion.asset_id))
    assert motion_inspection.dataset is not None
    assert motion_inspection.reference_model is not None

    profile_digest = (
        hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        if calibration_path is not None
        else hashlib.sha256(scaler.read_bytes()).hexdigest()
    )
    calibration_id = f"cal:sha256:{profile_digest}" if profile_source == "calibration" else None
    parameters = {
        "run_mode": "smoke",
        "limit_frames": 30,
        "human_height": 1.7,
        "retarget_fps": 30.0,
        "foot_clamp_anti_penetration": False,
        "reference": motion_inspection.reference_model,
        "retarget_profile": profile_source,
        "ik_iterations": 24,
    }
    profile_payload: dict[str, object] = {
        "source": profile_source,
        "calibration_id": calibration_id,
        "digest": profile_digest,
        "relative_path": (
            "test_robot/retarget_calibration_smpl.yaml"
            if profile_storage == "user_calibration"
            else (
                "urdf/retarget_calibration_smpl.yaml"
                if profile_source == "calibration"
                else "config/smpl_scaler.yaml"
            )
        ),
    }
    if profile_storage != "robot_bundle":
        profile_payload["storage"] = profile_storage
    payload: dict[str, object] = {
        "semantics": semantics,
        "motion": {
            "asset_id": motion.asset_id,
            "digest": motion.asset_id.removeprefix("asset:sha256:"),
            "category": category or motion_inspection.category.value,
            "dataset": motion_inspection.dataset,
            "reference": motion_inspection.reference_model,
        },
        "robot": {
            "asset_id": robot.asset_id,
            "digest": robot.asset_id.removeprefix("asset:sha256:"),
            "robot_id": "test_robot",
        },
        "backend": "newton",
        "retarget_profile": profile_payload,
        "output": {"format": "csv", "policy": "create_new"},
        "parameters": parameters,
    }
    plan = RetargetPlan(
        plan_id=compute_plan_id(payload),
        created_at=NOW,
        motion_asset_id=motion.asset_id,
        robot_id="test_robot",
        robot_asset_id=robot.asset_id,
        backend="newton",
        calibration_id=calibration_id,
        output_format="csv",
        output_policy=OutputPolicy.CREATE_NEW,
        parameters=parameters,
        input_digest=motion.asset_id.removeprefix("asset:sha256:"),
        robot_digest=robot.asset_id.removeprefix("asset:sha256:"),
        calibration_digest=profile_digest if profile_source == "calibration" else None,
    )
    store = PlanStore(tmp_path / "plan-state")
    store.put_if_absent(plan, payload)
    return _Fixture(
        assets=assets,
        store=store,
        plan=plan,
        payload=payload,
        motion_path=motion_path,
        robot_urdf=robot_urdf,
        calibration_path=calibration_path,
    )


def _assert_error(captured: pytest.ExceptionInfo[RetargetServiceError], code: str) -> None:
    assert captured.value.code == code
    assert captured.value.api_error.code == code


def test_get_job_spec_is_stable_detached_and_rechecks_both_asset_hashes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    recording_assets = _RecordingAssets(fixture.assets)
    calls = 0

    def provenance() -> JobSpecProvenance:
        nonlocal calls
        calls += 1
        return _fixed_provenance()

    service = RetargetService(
        fixture.store,
        recording_assets,
        provenance_provider=provenance,
    )
    first = service.get_job_spec(fixture.plan.plan_id)
    same = service.get_job_spec(fixture.plan.plan_id)

    assert first == same
    assert calls == 1
    assert first.schema_version == 2
    assert first.kind.value == "retarget"
    assert first.created_at == fixture.plan.created_at
    assert first.inputs[0].asset_id == fixture.plan.motion_asset_id
    assert first.inputs[0].sha256 == fixture.plan.input_digest
    assert first.robot.asset_id == fixture.plan.robot_asset_id
    assert first.robot.config_sha256 == fixture.plan.robot_digest
    assert first.calibration is None
    assert first.effective_parameters["output_format"] == "csv"
    assert first.provenance.device is None
    assert first.provenance.cuda == "12.8"
    assert list(first.provenance.dependencies) == ["a-package", "z-package"]
    assert str(tmp_path) not in first.model_dump_json()
    assert len(recording_assets.inspections) == 4
    assert {request.asset_id for request in recording_assets.inspections} == {
        fixture.plan.motion_asset_id,
        fixture.plan.robot_asset_id,
    }
    assert all(request.verify_hashes for request in recording_assets.inspections)
    assert all(not request.parse_content for request in recording_assets.inspections)

    first.effective_parameters["run_mode"] = "full"
    first.provenance.dependencies["a-package"] = "changed"
    restored = service.get_job_spec(fixture.plan.plan_id)
    assert restored.effective_parameters["run_mode"] == "smoke"
    assert restored.provenance.dependencies["a-package"] == "1"


def test_manual_calibration_is_materialized_in_job_spec(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, profile_source="calibration")
    service = RetargetService(
        fixture.store,
        fixture.assets,
        provenance_provider=_fixed_provenance,
    )

    spec = service.get_job_spec(fixture.plan.plan_id)

    assert spec.calibration is not None
    assert spec.calibration.calibration_id == fixture.plan.calibration_id
    assert spec.calibration.sha256 == fixture.plan.calibration_digest


def test_user_calibration_overlay_is_verified_outside_the_robot_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_root = tmp_path / "user-robots"
    monkeypatch.setenv("HHTOOLS_ROBOT_DIR", str(user_root))
    fixture = _fixture(
        tmp_path,
        profile_source="calibration",
        profile_storage="user_calibration",
    )
    service = RetargetService(
        fixture.store,
        fixture.assets,
        provenance_provider=_fixed_provenance,
    )

    spec = service.get_job_spec(fixture.plan.plan_id)

    assert spec.calibration is not None
    assert spec.calibration.calibration_id == fixture.plan.calibration_id
    assert spec.calibration.sha256 == fixture.plan.calibration_digest


def test_changed_user_calibration_overlay_makes_the_plan_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_root = tmp_path / "user-robots"
    monkeypatch.setenv("HHTOOLS_ROBOT_DIR", str(user_root))
    fixture = _fixture(
        tmp_path,
        profile_source="calibration",
        profile_storage="user_calibration",
    )
    assert fixture.calibration_path is not None
    service = RetargetService(
        fixture.store,
        fixture.assets,
        provenance_provider=_fixed_provenance,
    )
    fixture.calibration_path.write_text(
        fixture.calibration_path.read_text(encoding="utf-8").replace(
            "hip: 0.0",
            "hip: 0.5",
        ),
        encoding="utf-8",
    )

    with pytest.raises(RetargetServiceError) as captured:
        service.get_job_spec(fixture.plan.plan_id)

    _assert_error(captured, "PLAN_STALE")


def test_changed_manual_calibration_makes_the_plan_stale(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, profile_source="calibration")
    assert fixture.calibration_path is not None
    service = RetargetService(
        fixture.store,
        fixture.assets,
        provenance_provider=_fixed_provenance,
    )
    fixture.calibration_path.write_text(
        fixture.calibration_path.read_text(encoding="utf-8").replace(
            "hip: 0.0",
            "hip: 0.5",
        ),
        encoding="utf-8",
    )

    with pytest.raises(RetargetServiceError) as captured:
        service.get_job_spec(fixture.plan.plan_id)

    _assert_error(captured, "PLAN_STALE")


@pytest.mark.parametrize("changed_asset", ["motion", "robot"])
def test_changed_asset_bytes_make_the_plan_stale(
    tmp_path: Path,
    changed_asset: str,
) -> None:
    fixture = _fixture(tmp_path)
    service = RetargetService(
        fixture.store,
        fixture.assets,
        provenance_provider=_fixed_provenance,
    )
    target = fixture.motion_path if changed_asset == "motion" else fixture.robot_urdf
    target.write_bytes(target.read_bytes() + b"\nchanged")

    with pytest.raises(RetargetServiceError) as captured:
        service.get_job_spec(fixture.plan.plan_id)

    _assert_error(captured, "PLAN_STALE")
    assert "ASSET_HASH_MISMATCH" in captured.value.error.details["reason_code"]
    assert captured.value.error.next_action is not None
    assert captured.value.error.next_action.action == "run_preflight"


def test_changed_routing_projection_makes_the_plan_stale(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, category="terrain_scene")
    service = RetargetService(
        fixture.store,
        fixture.assets,
        provenance_provider=_fixed_provenance,
    )

    with pytest.raises(RetargetServiceError) as captured:
        service.get_job_spec(fixture.plan.plan_id)

    _assert_error(captured, "PLAN_STALE")


def test_unknown_plan_semantics_are_not_reinterpreted_as_job_spec_v2(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, semantics="future.retarget.plan.v9")
    service = RetargetService(
        fixture.store,
        fixture.assets,
        provenance_provider=_fixed_provenance,
    )

    with pytest.raises(RetargetServiceError) as captured:
        service.get_job_spec(fixture.plan.plan_id)

    _assert_error(captured, "UNSUPPORTED_PLAN_SEMANTICS")


def test_missing_plan_preserves_the_structured_plan_store_error(tmp_path: Path) -> None:
    motion_root = tmp_path / "motions"
    motion_root.mkdir()
    assets = AgentAssetService(AssetRegistry(tmp_path / "asset-state", {"motions": motion_root}))
    service = RetargetService(
        PlanStore(tmp_path / "plan-state"),
        assets,
        provenance_provider=_fixed_provenance,
    )

    with pytest.raises(RetargetServiceError) as captured:
        service.get_job_spec(f"plan:sha256:{'0' * 64}")

    _assert_error(captured, "PLAN_NOT_FOUND")
