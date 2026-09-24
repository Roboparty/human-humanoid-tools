from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hhtools.contracts import (
    AssetCategory,
    AssetFileRole,
    AssetInspectionRequest,
    AssetKind,
    AssetRegistrationRequest,
    ErrorStage,
    InspectionStatus,
)
from hhtools.services.asset_service import AgentAssetService
from hhtools.services.assets import AssetRegistry, AssetServiceError


def _request(relative_path: str) -> AssetRegistrationRequest:
    return AssetRegistrationRequest(
        root_id="motion-library",
        relative_path=relative_path,
    )


def _write_unified_npz(path: Path, *, frame_count: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.zeros((frame_count, 2, 3), dtype=np.float32)
    quaternions = np.zeros((frame_count, 2, 4), dtype=np.float32)
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


def test_plain_npz_register_inspect_and_persist_without_host_paths(
    tmp_path: Path,
) -> None:
    library = tmp_path / "private-motion-library"
    primary = library / "bundle" / "nested" / "walk.npz"
    _write_unified_npz(primary)
    data_dir = tmp_path / "agent-state"
    service = AgentAssetService(AssetRegistry(data_dir, {"motion-library": library}))

    bundle = service.register(_request("bundle"))
    inspection = service.inspect(AssetInspectionRequest(asset_id=bundle.asset_id))

    assert bundle.primary_file == "nested/walk.npz"
    assert bundle.detected is not None
    assert bundle.detected.dataset == "unified_npz"
    assert bundle.detected.reference == "smpl"
    assert bundle.detected.recommended_backend == "newton"
    assert inspection.status is InspectionStatus.VALID
    assert inspection.frame_count == 4
    assert inspection.joint_count == 2

    reopened = AgentAssetService(AssetRegistry(data_dir, {"motion-library": lambda: library}))
    assert reopened.get(bundle.asset_id) == bundle
    assert reopened.resolve_primary(bundle.asset_id) == primary.resolve()
    assert reopened.resolve_file(bundle.asset_id, bundle.primary_file) == primary.resolve()
    assert reopened.search(dataset="unified_npz").assets == [bundle]
    assert (
        reopened.inspect(AssetInspectionRequest(asset_id=bundle.asset_id)).asset_id
        == bundle.asset_id
    )
    serialized = f"{bundle.model_dump_json()} {inspection.model_dump_json()}"
    assert str(tmp_path) not in serialized
    assert service.allowed_root_ids == ("motion-library",)


def test_trusted_registration_hint_prefers_the_unique_most_specific_root(
    tmp_path: Path,
) -> None:
    assets_root = tmp_path / "private-assets"
    robot_root = assets_root / "robots"
    preset_root = robot_root / "test_robot"
    preset_root.mkdir(parents=True)
    service = AgentAssetService(
        AssetRegistry(
            tmp_path / "data",
            {
                "all-assets": assets_root,
                "robot-library": robot_root,
            },
        )
    )

    hint = service.registration_hint(
        preset_root,
        kind=AssetKind.ROBOT_BUNDLE,
        category=AssetCategory.ROBOT_MODEL,
    )

    assert hint == AssetRegistrationRequest(
        root_id="robot-library",
        relative_path="test_robot",
        kind=AssetKind.ROBOT_BUNDLE,
        category=AssetCategory.ROBOT_MODEL,
    )
    assert str(tmp_path) not in hint.model_dump_json()


def test_trusted_registration_hint_rejects_ambiguous_and_outside_paths_safely(
    tmp_path: Path,
) -> None:
    robot_root = tmp_path / "private-robots"
    preset_root = robot_root / "test_robot"
    preset_root.mkdir(parents=True)
    ambiguous = AgentAssetService(
        AssetRegistry(
            tmp_path / "ambiguous-data",
            {
                "robot-library-a": robot_root,
                "robot-library-b": lambda: robot_root,
            },
        )
    )

    with pytest.raises(AssetServiceError) as ambiguous_error:
        ambiguous.registration_hint(preset_root, kind=AssetKind.ROBOT_BUNDLE)

    assert ambiguous_error.value.code == "ASSET_ROOT_AMBIGUOUS"
    assert ambiguous_error.value.api_error.details == {
        "root_ids": ["robot-library-a", "robot-library-b"]
    }
    assert str(tmp_path) not in ambiguous_error.value.api_error.model_dump_json()

    allowed = tmp_path / "other-allowed-root"
    allowed.mkdir()
    outside = AgentAssetService(
        AssetRegistry(tmp_path / "outside-data", {"robot-library": allowed})
    )
    with pytest.raises(AssetServiceError) as outside_error:
        outside.registration_hint(preset_root, kind=AssetKind.ROBOT_BUNDLE)

    assert outside_error.value.code == "ASSET_OUTSIDE_ALLOWED_ROOT"
    assert str(tmp_path) not in outside_error.value.api_error.model_dump_json()


def test_trusted_registration_hint_does_not_fall_back_when_a_root_is_unavailable(
    tmp_path: Path,
) -> None:
    broad_root = tmp_path / "private-assets"
    preset_root = broad_root / "robots" / "test_robot"
    preset_root.mkdir(parents=True)

    def unavailable_root() -> Path:
        raise RuntimeError("private provider failure")

    service = AgentAssetService(
        AssetRegistry(
            tmp_path / "data",
            {
                "all-assets": broad_root,
                "robot-library": unavailable_root,
            },
        )
    )

    with pytest.raises(AssetServiceError) as captured:
        service.registration_hint(preset_root, kind=AssetKind.ROBOT_BUNDLE)

    assert captured.value.code == "ASSET_NOT_FOUND"
    assert captured.value.api_error.details == {"root_id": "robot-library"}
    assert str(tmp_path) not in captured.value.api_error.model_dump_json()


def test_trusted_registration_hint_checks_exact_root_ambiguity_before_contract_shape(
    tmp_path: Path,
) -> None:
    broad_root = tmp_path / "private-assets"
    preset_root = broad_root / "robots" / "test_robot"
    preset_root.mkdir(parents=True)
    ambiguous = AgentAssetService(
        AssetRegistry(
            tmp_path / "ambiguous-data",
            {
                "all-assets": broad_root,
                "preset-a": preset_root,
                "preset-b": lambda: preset_root,
            },
        )
    )

    with pytest.raises(AssetServiceError) as ambiguous_error:
        ambiguous.registration_hint(preset_root, kind=AssetKind.ROBOT_BUNDLE)

    assert ambiguous_error.value.code == "ASSET_ROOT_AMBIGUOUS"
    assert ambiguous_error.value.api_error.details == {"root_ids": ["preset-a", "preset-b"]}
    assert str(tmp_path) not in ambiguous_error.value.api_error.model_dump_json()

    exact = AgentAssetService(
        AssetRegistry(
            tmp_path / "exact-data",
            {
                "all-assets": broad_root,
                "preset": preset_root,
            },
        )
    )
    with pytest.raises(AssetServiceError) as exact_error:
        exact.registration_hint(preset_root, kind=AssetKind.ROBOT_BUNDLE)

    assert exact_error.value.code == "ASSET_ROOT_UNREPRESENTABLE"
    assert exact_error.value.api_error.details == {"root_id": "preset"}
    assert str(tmp_path) not in exact_error.value.api_error.model_dump_json()


def test_registration_translates_every_discovered_sidecar(tmp_path: Path) -> None:
    library = tmp_path / "library"
    bundle_dir = library / "OMOMO" / "sub10_largebox_000"
    bundle_dir.mkdir(parents=True)
    primary = bundle_dir / "sub10_largebox_000.pkl"
    primary.write_bytes(b"opaque pickle bytes are not executed during discovery")
    mesh = bundle_dir / "largebox_cleaned_simplified.obj"
    mesh.write_text("v 0 0 0\n", encoding="utf-8")
    service = AgentAssetService(AssetRegistry(tmp_path / "data", {"motion-library": library}))

    bundle = service.register(_request("OMOMO/sub10_largebox_000"))

    assert bundle.detected is not None
    assert bundle.detected.dataset == "omomo"
    assert {item.role for item in bundle.files} == {
        AssetFileRole.MOTION,
        AssetFileRole.OBJECT_MESH,
    }
    assert {item.relative_path for item in bundle.files} == {
        primary.name,
        mesh.name,
    }


def test_ambiguous_directory_has_only_relative_candidate_details(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    _write_unified_npz(library / "ambiguous" / "one.npz")
    _write_unified_npz(library / "ambiguous" / "nested" / "two.npz")
    service = AgentAssetService(AssetRegistry(tmp_path / "data", {"motion-library": library}))

    with pytest.raises(AssetServiceError) as captured:
        service.register(_request("ambiguous"))

    error = captured.value.api_error
    assert error.code == "BUNDLE_AMBIGUOUS"
    assert error.stage is ErrorStage.ASSET_REGISTRATION
    assert error.details == {"candidates": ["nested/two.npz", "one.npz"]}
    assert str(tmp_path) not in str(error.model_dump(mode="json"))


def test_registration_path_errors_are_preserved_without_host_paths(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    library.mkdir()
    outside = tmp_path / "outside.npz"
    _write_unified_npz(outside)
    service = AgentAssetService(AssetRegistry(tmp_path / "data", {"motion-library": library}))
    request = AssetRegistrationRequest.model_construct(
        root_id="motion-library",
        relative_path="../outside.npz",
        display_name=None,
        kind=None,
        category=None,
        recursive=True,
    )

    with pytest.raises(AssetServiceError) as captured:
        service.register(request)

    assert captured.value.code == "ASSET_OUTSIDE_ALLOWED_ROOT"
    assert str(tmp_path) not in str(captured.value.api_error.model_dump(mode="json"))


def test_inspection_reports_changed_content_as_a_hash_error(tmp_path: Path) -> None:
    library = tmp_path / "library"
    primary = library / "walk.npz"
    _write_unified_npz(primary)
    service = AgentAssetService(AssetRegistry(tmp_path / "data", {"motion-library": library}))
    bundle = service.register(_request("walk.npz"))
    primary.write_bytes(b"changed after registration")

    inspection = service.inspect(AssetInspectionRequest(asset_id=bundle.asset_id))

    assert inspection.status is InspectionStatus.INVALID
    assert "ASSET_HASH_MISMATCH" in {error.code for error in inspection.errors}
    assert str(tmp_path) not in str(inspection.model_dump(mode="json"))

    with pytest.raises(AssetServiceError) as captured:
        service.resolve_primary(bundle.asset_id)
    assert captured.value.code == "ASSET_HASH_MISMATCH"
    assert str(tmp_path) not in captured.value.api_error.model_dump_json()


def test_robot_bundle_registers_urdf_metadata_meshes_and_inspects(tmp_path: Path) -> None:
    library = tmp_path / "robots"
    robot = library / "test_robot"
    (robot / "urdf").mkdir(parents=True)
    (robot / "meshes").mkdir()
    (robot / "robot.yaml").write_text(
        "name: test_robot\nurdf: urdf/robot.urdf\ndof_order: [hip]\n",
        encoding="utf-8",
    )
    (robot / "meshes" / "body.stl").write_text(
        "solid body\nendsolid body\n",
        encoding="utf-8",
    )
    (robot / "urdf" / "robot.urdf").write_text(
        """<robot name="test_robot">
  <link name="base">
    <visual><geometry><mesh filename="../meshes/body.stl"/></geometry></visual>
  </link>
  <link name="hip_link"/>
  <joint name="hip" type="revolute">
    <parent link="base"/><child link="hip_link"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    service = AgentAssetService(AssetRegistry(tmp_path / "data", {"robots": library}))

    bundle = service.register(
        AssetRegistrationRequest(root_id="robots", relative_path="test_robot")
    )
    inspection = service.inspect(AssetInspectionRequest(asset_id=bundle.asset_id))

    assert bundle.kind.value == "robot_bundle"
    assert bundle.category.value == "robot_model"
    assert {item.role.value for item in bundle.files} == {
        "metadata",
        "robot_description",
        "visual_mesh",
    }
    assert inspection.status is InspectionStatus.VALID
    assert inspection.joint_count == 1
