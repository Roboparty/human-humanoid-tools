from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from hhtools.contracts import (
    AssetCategory,
    AssetFileRole,
    AssetInspectionRequest,
    AssetKind,
    AssetRegistrationRequest,
    InspectionStatus,
)
from hhtools.services.asset_service import AgentAssetService
from hhtools.services.assets import AssetRegistry, AssetServiceError


def _robot_csv(path: Path, *, robot: str = "source_bot", frames: int = 2) -> Path:
    rows = [
        f"# robot: {robot}",
        "# sample_rate: 50",
        ("time,root_x,root_y,root_z,root_qx,root_qy,root_qz,root_qw,dof_left_knee,dof_right_knee"),
    ]
    rows.extend(f"{frame / 50:.2f},0,0,1,0,0,0,1,0.1,-0.1" for frame in range(frames))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _service(tmp_path: Path) -> tuple[AgentAssetService, Path]:
    root = tmp_path / "assets"
    root.mkdir()
    return AgentAssetService(AssetRegistry(tmp_path / "state", {"source": root})), root


def test_robot_csv_auto_registers_as_a_typed_r2r_bundle(tmp_path: Path) -> None:
    service, root = _service(tmp_path)
    trajectory = _robot_csv(root / "walk.csv")

    bundle = service.register(
        AssetRegistrationRequest(root_id="source", relative_path=trajectory.name)
    )
    inspection = service.inspect(AssetInspectionRequest(asset_id=bundle.asset_id))

    assert bundle.kind is AssetKind.ROBOT_TRAJECTORY_BUNDLE
    assert bundle.category is AssetCategory.ROBOT_TRAJECTORY
    assert bundle.detected is not None
    assert bundle.detected.dataset == "robot_trajectory"
    assert bundle.detected.source_robot_id == "source_bot"
    assert bundle.detected.trajectory_profile == "mimic"
    assert bundle.files[0].role is AssetFileRole.ROBOT_TRAJECTORY
    assert inspection.status is InspectionStatus.VALID
    assert inspection.source_robot_id == "source_bot"
    assert inspection.frame_count == 2
    assert inspection.frame_rate_hz == 50.0
    assert inspection.joint_count == 2
    assert inspection.metadata["dof_names"] == ["left_knee", "right_knee"]
    assert inspection.metadata["content_parsed"] is True


def test_scene_sidecars_are_content_bound_and_reported_for_preflight_rejection(
    tmp_path: Path,
) -> None:
    service, root = _service(tmp_path)
    clip = root / "carry"
    _robot_csv(clip / "carry.csv")
    (clip / "object_0_box.csv").write_text("time,pos_x\n0,0\n", encoding="utf-8")
    (clip / "box.obj").write_text("v 0 0 0\n", encoding="utf-8")

    bundle = service.register(AssetRegistrationRequest(root_id="source", relative_path="carry"))
    inspection = service.inspect(AssetInspectionRequest(asset_id=bundle.asset_id))

    assert bundle.detected is not None
    assert bundle.detected.trajectory_profile == "intermimic"
    assert {item.role for item in bundle.files} == {
        AssetFileRole.ROBOT_TRAJECTORY,
        AssetFileRole.OBJECT_TRAJECTORY,
        AssetFileRole.OBJECT_MESH,
    }
    assert inspection.has_object is True
    assert inspection.has_terrain is False
    assert inspection.metadata["trajectory_profile"] == "intermimic"


def test_npz_trajectory_is_inspected_without_pickle_loading(tmp_path: Path) -> None:
    service, root = _service(tmp_path)
    joint_q = np.zeros((3, 9), dtype=np.float32)
    joint_q[:, 6] = 1.0
    np.savez(
        root / "walk.npz",
        joint_q=joint_q,
        dof_names=np.array(["left_knee", "right_knee"]),
        sample_rate=np.array(60.0),
        robot=np.array("source_bot"),
    )

    bundle = service.register(AssetRegistrationRequest(root_id="source", relative_path="walk.npz"))
    inspection = service.inspect(AssetInspectionRequest(asset_id=bundle.asset_id))

    assert bundle.kind is AssetKind.ROBOT_TRAJECTORY_BUNDLE
    assert inspection.status is InspectionStatus.VALID
    assert inspection.frame_count == 3
    assert inspection.frame_rate_hz == 60.0
    assert inspection.source_robot_id == "source_bot"


def test_pickle_can_be_registered_explicitly_but_cannot_pass_semantic_inspection(
    tmp_path: Path,
) -> None:
    service, root = _service(tmp_path)
    trajectory = root / "walk.pkl"
    trajectory.write_bytes(
        pickle.dumps(
            {
                "joint_q": [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.1]],
                "dof_names": ["left_knee"],
            }
        )
    )

    bundle = service.register(
        AssetRegistrationRequest(
            root_id="source",
            relative_path=trajectory.name,
            kind="robot_trajectory_bundle",
            category="robot_trajectory",
        )
    )
    inspection = service.inspect(AssetInspectionRequest(asset_id=bundle.asset_id))

    assert inspection.status is InspectionStatus.VALID_WITH_WARNINGS
    assert inspection.metadata["content_parsed"] is False
    assert inspection.metadata["content_validation_code"] == "CONTENT_REQUIRES_ISOLATED_VALIDATION"


def test_ambiguous_r2r_directory_requires_an_exact_clip(tmp_path: Path) -> None:
    service, root = _service(tmp_path)
    _robot_csv(root / "clips" / "first.csv")
    _robot_csv(root / "clips" / "second.csv")

    with pytest.raises(AssetServiceError) as captured:
        service.register(
            AssetRegistrationRequest(
                root_id="source",
                relative_path="clips",
                kind="robot_trajectory_bundle",
                category="robot_trajectory",
            )
        )

    assert captured.value.code == "BUNDLE_AMBIGUOUS"
    assert captured.value.api_error.details == {"candidates": ["first.csv", "second.csv"]}
