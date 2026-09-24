from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from hhtools.contracts import (
    ApiError,
    AssetKind,
    AssetRegistrationRequest,
    ErrorStage,
    JobSpecInput,
    JobSpecKind,
    JobSpecProvenance,
    JobSpecRobot,
    JobSpecV2,
    NextAction,
    PreflightResponse,
    PreflightStatus,
    RetargetPlan,
    RetargetPreflightRequest,
)
from hhtools.robot.base import RobotPreset
from hhtools.robot.registry import preset_from_dir
from hhtools.services.asset_service import AgentAssetService
from hhtools.services.assets import AssetRegistry
from hhtools.services.legacy_job_upgrade import (
    DynamicRootLocator,
    LegacyJobUpgradeError,
    LegacyJobUpgradeService,
)

NOW = datetime(2026, 8, 31, 4, 0, tzinfo=UTC)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_motion(path: Path, *, frame_count: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    positions: NDArray[np.float32] = np.zeros((frame_count, 2, 3), dtype=np.float32)
    quaternions: NDArray[np.float32] = np.zeros((frame_count, 2, 4), dtype=np.float32)
    quaternions[..., 3] = 1.0
    np.savez(
        path,
        schema_version=np.array("1"),
        name=np.array(path.stem),
        framerate=np.array(30.0),
        up_axis=np.array("Z"),
        source_format=np.array("npz"),
        bone_names=np.array(["root", "joint"]),
        parent_indices=np.array([-1, 0], dtype=np.int32),
        positions=positions,
        quaternions=quaternions,
    )


def _write_robot(root: Path) -> RobotPreset:
    urdf = root / "urdf" / "robot.urdf"
    urdf.parent.mkdir(parents=True)
    (root / "robot.yaml").write_text(
        "name: test_robot\n"
        "display_name: Test Robot\n"
        "urdf: urdf/robot.urdf\n"
        "dof_order: [hip]\n"
        "ik_map:\n"
        "  hips: base\n",
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
    <limit lower="-1.0" upper="1.0" effort="10" velocity="2"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return preset_from_dir(root)


class _FakeBoundaries:
    def __init__(self) -> None:
        self.preflight_calls: list[RetargetPreflightRequest] = []
        self.retarget_calls: list[str] = []
        self.plans: dict[str, RetargetPlan] = {}
        self.forced_preflight: PreflightResponse | None = None

    def preflight_retarget(self, request: RetargetPreflightRequest) -> PreflightResponse:
        self.preflight_calls.append(request)
        if self.forced_preflight is not None:
            return self.forced_preflight
        plan_id = f"plan:sha256:{_digest(request.model_dump(mode='json'))}"
        assert request.robot_asset_id is not None
        plan = RetargetPlan(
            plan_id=plan_id,
            created_at=NOW,
            motion_asset_id=request.motion_asset_id,
            robot_id=request.robot_id,
            robot_asset_id=request.robot_asset_id,
            backend=request.backend or "newton",
            output_format=request.output_format,
            output_policy=request.output_policy,
            parameters=json.loads(json.dumps(request.parameters)),
            input_digest=request.motion_asset_id.removeprefix("asset:sha256:"),
            robot_digest=request.robot_asset_id.removeprefix("asset:sha256:"),
        )
        self.plans[plan_id] = plan
        return PreflightResponse(
            request_id="req_legacy_upgrade",
            status=PreflightStatus.READY,
            plan=plan,
            recommended_backend=plan.backend,
        )

    def get_job_spec(self, plan_id: str) -> JobSpecV2:
        self.retarget_calls.append(plan_id)
        plan = self.plans[plan_id]
        parameters = json.loads(json.dumps(plan.parameters))
        parameters["output_format"] = plan.output_format
        return JobSpecV2(
            kind=JobSpecKind.RETARGET,
            plan_id=plan.plan_id,
            inputs=[
                JobSpecInput(
                    asset_id=plan.motion_asset_id,
                    sha256=plan.input_digest,
                )
            ],
            robot=JobSpecRobot(
                robot_id=plan.robot_id,
                asset_id=plan.robot_asset_id,
                config_sha256=plan.robot_digest,
            ),
            calibration=None,
            backend=plan.backend,
            effective_parameters=parameters,
            output_policy=plan.output_policy,
            provenance=JobSpecProvenance(
                hhtools_git_commit="a" * 40,
                hhtools_dirty=False,
                python="3.12.5",
                device=None,
            ),
            created_at=plan.created_at,
        )


@dataclass(slots=True)
class _Harness:
    service: LegacyJobUpgradeService
    asset_service: AgentAssetService
    boundaries: _FakeBoundaries
    motion_root: Path
    motion_path: Path
    robot_root: Path
    robot: RobotPreset


def _harness(
    tmp_path: Path,
    *,
    motion_roots: dict[str, Path] | None = None,
    robot_roots: dict[str, Path] | None = None,
    locator: DynamicRootLocator | None = None,
) -> _Harness:
    motion_root = tmp_path / "motion-library"
    motion_path = motion_root / "nested" / "walk.npz"
    _write_motion(motion_path)
    robot_root = tmp_path / "robot-library"
    robot = _write_robot(robot_root / "test_robot")
    registered_motion_roots = motion_roots or {"motion-main": motion_root}
    registered_robot_roots = robot_roots or {"robot-main": robot_root}
    root_locator = locator or DynamicRootLocator(
        motion_roots=registered_motion_roots,
        robot_roots=registered_robot_roots,
    )
    asset_service = AgentAssetService(
        AssetRegistry(tmp_path / "state", root_locator.registry_root_providers())
    )
    boundaries = _FakeBoundaries()
    service = LegacyJobUpgradeService(
        asset_service,
        boundaries,
        boundaries,
        root_locator,
        robot_provider=lambda: [robot],
    )
    return _Harness(
        service=service,
        asset_service=asset_service,
        boundaries=boundaries,
        motion_root=motion_root,
        motion_path=motion_path,
        robot_root=robot_root,
        robot=robot,
    )


def _payload(harness: _Harness) -> dict[str, Any]:
    source = str(harness.motion_path)
    return {
        "schema_version": 1,
        "kind": "retarget",
        "request": {
            "source_path": source,
            "source_entry": {
                "dataset": "unified_npz",
                "folder_label": "nested",
                "sequence_id": "walk.npz",
                "source_path": source,
                "stem": "walk",
                "reference": "smpl",
                "motion_category": "motion",
                "asset_kind": "human_motion",
            },
            "robot": "test_robot",
            "reference": "smpl",
            "backend": "newton",
            "ik_iterations": 31,
            "human_height": 1.72,
            "limit_frames": 12,
            "retarget_fps": 60.0,
            "foot_clamp_anti_penetration": True,
        },
    }


def test_upgrade_registers_inspects_preflights_and_maps_parameters(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    payload = _payload(harness)
    before = json.loads(json.dumps(payload))

    result = harness.service.upgrade(payload)

    assert payload == before
    assert len(harness.boundaries.preflight_calls) == 1
    request = harness.boundaries.preflight_calls[0]
    assert request.robot_id == "test_robot"
    assert request.backend == "newton"
    assert request.output_format == "csv"
    assert request.output_policy.value == "create_new"
    assert request.calibration_id is None
    assert request.parameters == {
        "run_mode": "smoke",
        "limit_frames": 12,
        "ik_iterations": 31,
        "human_height": 1.72,
        "retarget_fps": 60.0,
        "foot_clamp_anti_penetration": True,
    }
    assert result.preflight.plan is not None
    assert result.job_spec is not None
    assert result.receipt is not None
    assert harness.boundaries.retarget_calls == [result.preflight.plan.plan_id]
    assert result.job_spec.backend == "newton"
    assert result.receipt.motion_asset_id == request.motion_asset_id
    assert result.receipt.robot_asset_id == request.robot_asset_id
    assert result.receipt.output_format == "csv"
    assert result.receipt.output_policy == "create_new"
    assert result.receipt.canonical_v1_sha256 == _digest(payload)

    portable = json.dumps(
        {
            "preflight": result.preflight.model_dump(mode="json"),
            "job_spec": result.job_spec.model_dump(mode="json"),
            "receipt": result.receipt.model_dump(mode="json"),
        },
        ensure_ascii=False,
    )
    assert str(tmp_path) not in portable
    assert "source_path" not in portable


def test_upgrade_is_idempotent_and_wrapper_job_id_is_not_execution_identity(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    payload = _payload(harness)

    first = harness.service.upgrade(payload)
    second = harness.service.upgrade({"job_id": "legacy_a", "spec": payload})
    outer_request = json.loads(json.dumps(payload["request"]))
    outer_request["motion_token"] = "expired-session-token"
    third = harness.service.upgrade(
        {
            "schema_version": 1,
            "job_id": "legacy_download",
            "kind": "retarget",
            "status": "done",
            "created_at": 1_775_000_000.0,
            "finished_at": 1_775_000_001.0,
            "scope": "persistent",
            "request": outer_request,
            "cli": {"available": True, "command": f"legacy {harness.motion_path}"},
            "spec": payload,
            "replay": {"available": True, "reason": None, "source_count": 1},
            "parent_job_id": None,
        }
    )

    assert first.job_spec == second.job_spec
    assert first.receipt == second.receipt
    assert first.job_spec == third.job_spec
    assert first.receipt == third.receipt
    assert first.receipt is not None
    assert first.receipt.canonical_v1_sha256 == _digest(payload)
    assert len(harness.boundaries.preflight_calls) == 3
    assert len(harness.boundaries.retarget_calls) == 3


def test_downloaded_wrapper_conflict_is_rejected(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    payload = _payload(harness)
    outer_request = json.loads(json.dumps(payload["request"]))
    outer_request["robot"] = "another_robot"

    with pytest.raises(LegacyJobUpgradeError) as captured:
        harness.service.upgrade(
            {
                "job_id": "legacy_download",
                "kind": "retarget",
                "request": outer_request,
                "spec": payload,
            }
        )

    assert captured.value.code == "INVALID_JOB_SPEC"
    assert captured.value.api_error.details == {"field": "spec"}
    assert harness.boundaries.preflight_calls == []


def test_absent_or_zero_frame_limit_maps_to_full_execution(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    payload = _payload(harness)
    request = payload["request"]
    for field in (
        "source_entry",
        "reference",
        "backend",
        "ik_iterations",
        "human_height",
        "limit_frames",
        "retarget_fps",
        "foot_clamp_anti_penetration",
    ):
        request.pop(field)

    harness.service.upgrade(payload)
    first_parameters = harness.boundaries.preflight_calls[-1].parameters
    request["limit_frames"] = 0
    harness.service.upgrade(payload)
    second_parameters = harness.boundaries.preflight_calls[-1].parameters

    assert first_parameters == {
        "run_mode": "full",
        "ik_iterations": 24,
        "foot_clamp_anti_penetration": False,
    }
    assert second_parameters == first_parameters


def test_interaction_backend_validates_but_omits_newton_only_iterations(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    payload = _payload(harness)
    payload["request"]["backend"] = "interaction_mesh"

    result = harness.service.upgrade(payload)

    assert harness.boundaries.preflight_calls[-1].parameters == {
        "run_mode": "smoke",
        "limit_frames": 12,
        "human_height": 1.72,
        "retarget_fps": 60.0,
        "foot_clamp_anti_penetration": True,
    }
    assert result.receipt is not None
    assert result.receipt.warnings == ("LEGACY_INTERACTION_IK_ITERATIONS_IGNORED",)


@pytest.mark.parametrize(
    ("mutate", "expected_field"),
    [
        (lambda body: body.update({"unexpected": True}), None),
        (lambda body: body.update({"schema_version": "1"}), "schema_version"),
        (lambda body: body.update({"kind": "batch"}), "kind"),
        (lambda body: body["request"].update({"out_dir": "result"}), None),
        (lambda body: body["request"].update({"robot": "../test_robot"}), "robot"),
        (lambda body: body["request"].update({"limit_frames": True}), "limit_frames"),
        (lambda body: body["request"].update({"ik_iterations": "24"}), "ik_iterations"),
    ],
)
def test_strict_v1_contract_rejects_unknowns_and_coercion(
    tmp_path: Path,
    mutate: Any,
    expected_field: str | None,
) -> None:
    harness = _harness(tmp_path)
    payload = _payload(harness)
    mutate(payload)

    with pytest.raises(LegacyJobUpgradeError) as captured:
        harness.service.upgrade(payload)

    assert captured.value.code == "INVALID_JOB_SPEC"
    if expected_field is not None:
        assert captured.value.api_error.details.get("field") == expected_field
    assert harness.boundaries.preflight_calls == []
    assert str(tmp_path) not in captured.value.api_error.model_dump_json()


def test_document_depth_and_size_limits_run_before_filesystem_access(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    deep = _payload(harness)
    nested: list[Any] = []
    cursor = nested
    for _ in range(20):
        child: list[Any] = []
        cursor.append(child)
        cursor = child
    deep["request"]["source_entry"]["label"] = nested

    with pytest.raises(LegacyJobUpgradeError, match="nested too deeply"):
        harness.service.upgrade(deep)

    oversized = _payload(harness)
    oversized["request"]["source_path"] = "x" * (64 * 1024)
    with pytest.raises(LegacyJobUpgradeError, match="oversized string"):
        harness.service.upgrade(oversized)
    assert harness.boundaries.preflight_calls == []


def test_large_numbers_are_rejected_as_structured_errors(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    payload = _payload(harness)
    payload["request"]["human_height"] = 10**400

    with pytest.raises(LegacyJobUpgradeError) as captured:
        harness.service.upgrade(payload)
    assert captured.value.code == "INVALID_JOB_SPEC"
    assert captured.value.api_error.details == {"field": "human_height"}

    wrapper = {"created_at": 10**400, "spec": _payload(harness)}
    with pytest.raises(LegacyJobUpgradeError) as captured:
        harness.service.upgrade(wrapper)
    assert captured.value.code == "INVALID_JOB_SPEC"
    assert captured.value.api_error.details == {"field": "created_at"}
    assert harness.boundaries.preflight_calls == []


def test_unknown_path_shaped_field_name_is_not_reflected(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    payload = _payload(harness)
    payload["request"][str(tmp_path / "secret-field")] = True

    with pytest.raises(LegacyJobUpgradeError) as captured:
        harness.service.upgrade(payload)

    serialized = captured.value.api_error.model_dump_json()
    assert captured.value.code == "INVALID_JOB_SPEC"
    assert captured.value.api_error.details == {"field_count": 1}
    assert str(tmp_path) not in serialized


def test_outside_path_and_nested_second_path_are_rejected_without_leaking_paths(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    outside = tmp_path / "private" / "outside.npz"
    _write_motion(outside)
    payload = _payload(harness)
    payload["request"]["source_path"] = str(outside)
    payload["request"]["source_entry"]["source_path"] = str(outside)

    with pytest.raises(LegacyJobUpgradeError) as captured:
        harness.service.upgrade(payload)
    assert captured.value.code == "ASSET_OUTSIDE_ALLOWED_ROOT"
    assert str(tmp_path) not in captured.value.api_error.model_dump_json()

    payload = _payload(harness)
    payload["request"]["source_entry"]["source_path"] = str(outside)
    with pytest.raises(LegacyJobUpgradeError) as captured:
        harness.service.upgrade(payload)
    assert captured.value.code == "INVALID_JOB_SPEC"
    assert captured.value.api_error.details == {"field": "source_entry.source_path"}
    assert harness.boundaries.preflight_calls == []


def test_robot_trajectory_cannot_cross_the_h2r_boundary(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    payload = _payload(harness)
    payload["request"]["source_entry"]["asset_kind"] = "robot_trajectory"

    with pytest.raises(LegacyJobUpgradeError) as captured:
        harness.service.upgrade(payload)

    assert captured.value.code == "INVALID_JOB_SPEC"
    assert captured.value.api_error.details == {"field": "source_entry.asset_kind"}
    assert harness.boundaries.preflight_calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset", "different_dataset"),
        ("motion_category", "object"),
        ("has_scene", True),
        ("suggested_backend", "interaction_mesh"),
        ("upload_profile", "intermimic"),
    ],
)
def test_source_entry_execution_claims_must_match_safe_inspection(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    harness = _harness(tmp_path)
    payload = _payload(harness)
    payload["request"]["source_entry"][field] = value

    with pytest.raises(LegacyJobUpgradeError) as captured:
        harness.service.upgrade(payload)

    assert captured.value.code == "LEGACY_METADATA_MISMATCH"
    assert captured.value.api_error.details["field"] == f"source_entry.{field}"
    assert harness.boundaries.preflight_calls == []


@pytest.mark.parametrize("status", ["rejected", "human_action_required"])
def test_non_ready_preflight_is_preserved_without_creating_v2(
    tmp_path: Path,
    status: str,
) -> None:
    harness = _harness(tmp_path)
    if status == "rejected":
        response = PreflightResponse(
            request_id="req_rejected",
            status=PreflightStatus.REJECTED,
            error=ApiError(
                code="BACKEND_UNAVAILABLE",
                message="Backend unavailable.",
                stage=ErrorStage.PREFLIGHT,
            ),
        )
    else:
        response = PreflightResponse(
            request_id="req_human",
            status=PreflightStatus.HUMAN_ACTION_REQUIRED,
            required_actions=[
                NextAction(
                    actor="human",
                    action="open_calibration",
                    message="Calibrate this robot.",
                )
            ],
        )
    harness.boundaries.forced_preflight = response

    result = harness.service.upgrade(_payload(harness))

    assert result.preflight == response
    assert result.job_spec is None
    assert result.receipt is None
    assert harness.boundaries.retarget_calls == []


def test_locator_uses_the_most_specific_allowlisted_root(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    specific = harness.motion_root / "nested"
    locator = DynamicRootLocator(
        motion_roots={
            "motion-general": harness.motion_root,
            "motion-specific": specific,
        },
        robot_roots={"robot-main": harness.robot_root},
    )
    asset_service = AgentAssetService(
        AssetRegistry(tmp_path / "specific-state", locator.registry_root_providers())
    )
    boundaries = _FakeBoundaries()
    service = LegacyJobUpgradeService(
        asset_service,
        boundaries,
        boundaries,
        locator,
        robot_provider=lambda: [harness.robot],
    )

    result = service.upgrade(_payload(harness))

    assert result.receipt is not None
    bundle = asset_service.get(result.receipt.motion_asset_id)
    assert bundle.source is not None
    assert bundle.source.root_id == "motion-specific"
    assert bundle.source.logical_path == "walk.npz"


def test_dynamic_root_change_is_detected_before_preflight(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    replacement = tmp_path / "replacement-motion-library"
    replacement.mkdir()
    calls = 0

    def dynamic_motion_root() -> Path:
        nonlocal calls
        calls += 1
        return harness.motion_root if calls == 1 else replacement

    boundaries = _FakeBoundaries()
    locator = DynamicRootLocator(
        motion_roots={"motion-main": dynamic_motion_root},
        robot_roots={"robot-main": harness.robot_root},
    )
    service = LegacyJobUpgradeService(
        harness.asset_service,
        boundaries,
        boundaries,
        locator,
        robot_provider=lambda: [harness.robot],
    )

    with pytest.raises(LegacyJobUpgradeError) as captured:
        service.upgrade(_payload(harness))

    assert captured.value.code == "ASSET_CHANGED_DURING_UPGRADE"
    assert boundaries.preflight_calls == []
    assert str(tmp_path) not in captured.value.api_error.model_dump_json()


@pytest.mark.parametrize("same_bytes", [False, True])
def test_a_to_b_to_a_root_race_is_detected_even_for_equivalent_bytes(
    tmp_path: Path,
    same_bytes: bool,
) -> None:
    harness = _harness(tmp_path)
    replacement = tmp_path / "replacement-motion-library"
    replacement_path = replacement / "nested" / "walk.npz"
    if same_bytes:
        replacement_path.parent.mkdir(parents=True)
        replacement_path.write_bytes(harness.motion_path.read_bytes())
    else:
        _write_motion(replacement_path, frame_count=17)
    calls = 0

    def switching_motion_root() -> Path:
        nonlocal calls
        calls += 1
        # Lookup authorizes A. Asset registration and inspection see B. The
        # provider then switches back to A before explicit root revalidation.
        if calls == 1 or calls >= 5:
            return harness.motion_root
        return replacement

    locator = DynamicRootLocator(
        motion_roots={"motion-main": switching_motion_root},
        robot_roots={"robot-main": harness.robot_root},
    )
    asset_service = AgentAssetService(
        AssetRegistry(tmp_path / "race-state", locator.registry_root_providers())
    )
    boundaries = _FakeBoundaries()
    service = LegacyJobUpgradeService(
        asset_service,
        boundaries,
        boundaries,
        locator,
        robot_provider=lambda: [harness.robot],
    )

    with pytest.raises(LegacyJobUpgradeError) as captured:
        service.upgrade(_payload(harness))

    assert captured.value.code == "ASSET_CHANGED_DURING_UPGRADE"
    assert boundaries.preflight_calls == []
    assert str(tmp_path) not in captured.value.api_error.model_dump_json()


def test_content_dedup_from_another_allowed_source_remains_upgradeable(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    second_root = tmp_path / "second-motion-library"
    second_path = second_root / "different-folder" / "walk.npz"
    second_path.parent.mkdir(parents=True)
    second_path.write_bytes(harness.motion_path.read_bytes())
    locator = DynamicRootLocator(
        motion_roots={
            "motion-first": harness.motion_root,
            "motion-second": second_root,
        },
        robot_roots={"robot-main": harness.robot_root},
    )
    asset_service = AgentAssetService(
        AssetRegistry(tmp_path / "dedup-state", locator.registry_root_providers())
    )
    first = asset_service.register(
        AssetRegistrationRequest(
            root_id="motion-first",
            relative_path="nested/walk.npz",
            display_name=None,
            kind=AssetKind.MOTION_BUNDLE,
        )
    )
    boundaries = _FakeBoundaries()
    service = LegacyJobUpgradeService(
        asset_service,
        boundaries,
        boundaries,
        locator,
        robot_provider=lambda: [harness.robot],
    )
    payload = _payload(harness)
    payload["request"]["source_path"] = str(second_path)
    payload["request"]["source_entry"]["source_path"] = str(second_path)

    result = service.upgrade(payload)

    assert result.receipt is not None
    assert result.receipt.motion_asset_id == first.asset_id
    assert boundaries.retarget_calls == [result.receipt.plan_id]


def test_equally_specific_root_aliases_are_rejected(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    locator = DynamicRootLocator(
        motion_roots={
            "motion-a": harness.motion_root,
            "motion-b": harness.motion_root,
        },
        robot_roots={"robot-main": harness.robot_root},
    )
    boundaries = _FakeBoundaries()
    service = LegacyJobUpgradeService(
        harness.asset_service,
        boundaries,
        boundaries,
        locator,
        robot_provider=lambda: [harness.robot],
    )

    with pytest.raises(LegacyJobUpgradeError) as captured:
        service.upgrade(_payload(harness))

    assert captured.value.code == "AMBIGUOUS_ALLOWED_ROOT"
    assert captured.value.api_error.details == {
        "usage": "motion",
        "root_ids": ["motion-a", "motion-b"],
    }
    assert boundaries.preflight_calls == []
