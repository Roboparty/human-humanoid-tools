from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from hhtools.contracts import (
    AssetBundle,
    AssetCategory,
    AssetFile,
    AssetFileRole,
    AssetKind,
    ErrorStage,
    InspectionStatus,
)
from hhtools.services.robot_asset_inspection import (
    RobotAssetDiscovery,
    RobotAssetDiscoveryError,
    RobotAssetInspector,
    discover_robot_bundle,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset_id(seed: str) -> str:
    return f"asset:sha256:{hashlib.sha256(seed.encode()).hexdigest()}"


def _bundle(root: Path, discovery: RobotAssetDiscovery) -> AssetBundle:
    files = []
    for item in discovery.files:
        relative = item.path.relative_to(root.resolve()).as_posix()
        files.append(
            AssetFile(
                role=item.role,
                relative_path=relative,
                sha256=_digest(item.path),
                size_bytes=item.path.stat().st_size,
                required=item.required,
            )
        )
    return AssetBundle(
        asset_id=_asset_id(discovery.primary_urdf.name),
        kind=AssetKind.ROBOT_BUNDLE,
        category=AssetCategory.ROBOT_MODEL,
        display_name="Test Robot",
        primary_file=discovery.primary_urdf.relative_to(root.resolve()).as_posix(),
        files=files,
        metadata=dict(discovery.metadata),
    )


def _write_valid_robot(root: Path, *, mesh_reference: str = "../meshes/body.stl") -> Path:
    urdf = root / "urdf" / "robot.urdf"
    mesh = root / "meshes" / "body.stl"
    urdf.parent.mkdir(parents=True)
    mesh.parent.mkdir(parents=True)
    mesh.write_text("solid body\nendsolid body\n", encoding="utf-8")
    (root / "robot.yaml").write_text("name: test_robot\n", encoding="utf-8")
    urdf.write_text(
        f"""<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base">
    <visual><geometry><mesh filename="{mesh_reference}"/></geometry></visual>
    <collision><geometry><mesh filename="{mesh_reference}"/></geometry></collision>
  </link>
  <link name="arm"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <limit lower="-1.0" upper="1.0" effort="10" velocity="2"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return urdf


def _assert_no_host_path(value: object, root: Path) -> None:
    serialized = str(value)
    assert str(root) not in serialized
    assert root.as_posix() not in serialized


def test_valid_urdf_bundle_discovers_meshes_metadata_and_joint_counts(tmp_path: Path) -> None:
    bundle_root = tmp_path / "robot"
    urdf = _write_valid_robot(bundle_root)

    discovery = discover_robot_bundle(bundle_root)

    assert discovery.primary_urdf == urdf.resolve()
    roles = {item.path.name: item.role for item in discovery.files}
    assert roles["robot.urdf"] is AssetFileRole.ROBOT_DESCRIPTION
    assert roles["robot.yaml"] is AssetFileRole.METADATA
    # A mesh shared by visual and collision is represented once using the
    # stricter collision role; the shared-use count remains explicit metadata.
    assert roles["body.stl"] is AssetFileRole.COLLISION_MESH
    assert len([item for item in discovery.files if item.path.name == "body.stl"]) == 1
    assert discovery.metadata["link_count"] == 2
    assert discovery.metadata["urdf_joint_count"] == 1
    assert discovery.metadata["joint_count"] == 1
    assert discovery.metadata["shared_visual_collision_mesh_count"] == 1

    manifest = _bundle(bundle_root, discovery)
    inspection = RobotAssetInspector().inspect(manifest, bundle_root)

    assert inspection.status is InspectionStatus.VALID
    assert inspection.kind is AssetKind.ROBOT_BUNDLE
    assert inspection.category is AssetCategory.ROBOT_MODEL
    assert inspection.source_format == "urdf"
    assert inspection.joint_count == 1
    assert inspection.metadata["robot_name"] == "test_robot"
    assert inspection.metadata["visual_mesh_declaration_count"] == 1
    assert inspection.metadata["collision_mesh_declaration_count"] == 1
    assert inspection.metadata["unique_mesh_count"] == 1
    _assert_no_host_path(inspection.model_dump(mode="json"), tmp_path)


def test_package_uri_resolves_within_bundle(tmp_path: Path) -> None:
    bundle_root = tmp_path / "robot"
    _write_valid_robot(bundle_root, mesh_reference="package://robot_description/meshes/body.stl")

    discovery = discover_robot_bundle(bundle_root)

    assert {item.path.name for item in discovery.files} == {
        "body.stl",
        "robot.urdf",
        "robot.yaml",
    }


def test_declared_scaler_config_is_bound_into_robot_manifest(tmp_path: Path) -> None:
    bundle_root = tmp_path / "robot"
    _write_valid_robot(bundle_root)
    scaler = bundle_root / "config" / "smpl_scaler.yaml"
    scaler.parent.mkdir()
    scaler.write_text("joint_scales: {}\n", encoding="utf-8")
    (bundle_root / "robot.yaml").write_text(
        "name: test_robot\n"
        "retarget:\n"
        "  references:\n"
        "    smpl:\n"
        "      scaler_config: config/smpl_scaler.yaml\n",
        encoding="utf-8",
    )

    discovery = discover_robot_bundle(bundle_root)

    assert any(item.path == scaler.resolve() for item in discovery.files)
    assert all(
        item.role is AssetFileRole.METADATA
        for item in discovery.files
        if item.path.name in {"robot.yaml", "smpl_scaler.yaml"}
    )
    assert discovery.metadata["metadata_file_count"] == 2


def test_declared_scaler_config_cannot_escape_robot_bundle(tmp_path: Path) -> None:
    bundle_root = tmp_path / "robot"
    _write_valid_robot(bundle_root)
    outside = tmp_path / "outside_scaler.yaml"
    outside.write_text("joint_scales: {}\n", encoding="utf-8")
    (bundle_root / "robot.yaml").write_text(
        "name: test_robot\n"
        "retarget:\n"
        "  references:\n"
        "    smpl:\n"
        "      scaler_config: ../outside_scaler.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(RobotAssetDiscoveryError) as caught:
        discover_robot_bundle(bundle_root)

    assert caught.value.code == "ASSET_OUTSIDE_ALLOWED_ROOT"
    _assert_no_host_path(caught.value.api_error.model_dump(mode="json"), tmp_path)


@pytest.mark.parametrize("path_field", ["urdf", "mesh_search_paths", "scaler_config"])
def test_robot_metadata_execution_paths_must_be_bundle_relative(
    tmp_path: Path,
    path_field: str,
) -> None:
    bundle_root = tmp_path / "robot"
    urdf = _write_valid_robot(bundle_root)
    mesh_directory = bundle_root / "meshes"
    scaler = bundle_root / "config" / "smpl_scaler.yaml"
    scaler.parent.mkdir()
    scaler.write_text("joint_scales: {}\n", encoding="utf-8")

    if path_field == "urdf":
        metadata = f"name: test_robot\nurdf: {urdf.as_posix()}\n"
    elif path_field == "mesh_search_paths":
        metadata = f"name: test_robot\nmesh_search_paths:\n  - {mesh_directory.as_posix()}\n"
    else:
        metadata = (
            "name: test_robot\n"
            "retarget:\n"
            "  references:\n"
            "    smpl:\n"
            f"      scaler_config: {scaler.as_posix()}\n"
        )
    (bundle_root / "robot.yaml").write_text(metadata, encoding="utf-8")

    with pytest.raises(RobotAssetDiscoveryError) as caught:
        discover_robot_bundle(bundle_root)

    assert caught.value.code == "ROBOT_METADATA_INVALID"
    _assert_no_host_path(caught.value.api_error.model_dump(mode="json"), tmp_path)


def test_missing_referenced_mesh_is_a_stable_bundle_error(tmp_path: Path) -> None:
    bundle_root = tmp_path / "robot"
    urdf = _write_valid_robot(bundle_root)
    (bundle_root / "meshes" / "body.stl").unlink()

    with pytest.raises(RobotAssetDiscoveryError) as caught:
        discover_robot_bundle(bundle_root)

    assert caught.value.code == "BUNDLE_INCOMPLETE"
    assert caught.value.api_error.stage is ErrorStage.ASSET_INSPECTION
    assert caught.value.api_error.details == {
        "relative_path": "meshes/body.stl",
        "role": "visual_mesh",
    }
    _assert_no_host_path(caught.value.api_error.model_dump(mode="json"), tmp_path)

    # A previously registered URDF-only manifest detects the same missing
    # dependency through its content inspection, without needing discovery.
    manifest = AssetBundle(
        asset_id=_asset_id("missing-mesh"),
        kind=AssetKind.ROBOT_BUNDLE,
        category=AssetCategory.ROBOT_MODEL,
        display_name="Missing Mesh",
        primary_file="urdf/robot.urdf",
        files=[
            AssetFile(
                role=AssetFileRole.ROBOT_DESCRIPTION,
                relative_path="urdf/robot.urdf",
                sha256=_digest(urdf),
                size_bytes=urdf.stat().st_size,
            )
        ],
    )
    inspection = RobotAssetInspector().inspect(manifest, bundle_root)
    assert inspection.status is InspectionStatus.INVALID
    assert "BUNDLE_INCOMPLETE" in {error.code for error in inspection.errors}
    _assert_no_host_path(inspection.model_dump(mode="json"), tmp_path)


@pytest.mark.parametrize("reference_kind", ["parent", "absolute", "file_uri"])
def test_mesh_references_cannot_escape_candidate_bundle(
    tmp_path: Path,
    reference_kind: str,
) -> None:
    outside = tmp_path / "outside.stl"
    outside.write_text("solid outside\nendsolid outside\n", encoding="utf-8")
    bundle_root = tmp_path / "robot"
    if reference_kind == "parent":
        reference = "../../outside.stl"
    elif reference_kind == "file_uri":
        reference = outside.as_uri()
    else:
        reference = str(outside.resolve())
    _write_valid_robot(bundle_root, mesh_reference=reference)

    with pytest.raises(RobotAssetDiscoveryError) as caught:
        discover_robot_bundle(bundle_root)

    assert caught.value.code == "ASSET_OUTSIDE_ALLOWED_ROOT"
    assert caught.value.api_error.stage is ErrorStage.ASSET_INSPECTION
    _assert_no_host_path(caught.value.api_error.model_dump(mode="json"), tmp_path)


def test_absolute_mesh_inside_bundle_is_rejected_for_new_and_persisted_assets(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "robot"
    urdf = _write_valid_robot(bundle_root)
    metadata = bundle_root / "robot.yaml"
    dae = bundle_root / "meshes" / "source_body.dae"
    dae.write_text("<COLLADA/>", encoding="utf-8")
    urdf.write_text(
        urdf.read_text(encoding="utf-8").replace(
            "../meshes/body.stl",
            dae.resolve().as_posix(),
        ),
        encoding="utf-8",
    )
    generated_stl = dae.with_suffix(".stl")

    with pytest.raises(RobotAssetDiscoveryError) as caught:
        discover_robot_bundle(bundle_root)

    assert caught.value.code == "ASSET_OUTSIDE_ALLOWED_ROOT"
    assert not generated_stl.exists()
    _assert_no_host_path(caught.value.api_error.model_dump(mode="json"), tmp_path)

    # Bundles persisted by an older release receive the same fail-closed check
    # during the full inspection that precedes Agent snapshot materialization.
    manifest = AssetBundle(
        asset_id=_asset_id("legacy-absolute-mesh"),
        kind=AssetKind.ROBOT_BUNDLE,
        category=AssetCategory.ROBOT_MODEL,
        display_name="Legacy absolute mesh",
        primary_file="urdf/robot.urdf",
        files=[
            AssetFile(
                role=AssetFileRole.ROBOT_DESCRIPTION,
                relative_path="urdf/robot.urdf",
                sha256=_digest(urdf),
                size_bytes=urdf.stat().st_size,
            ),
            AssetFile(
                role=AssetFileRole.METADATA,
                relative_path="robot.yaml",
                sha256=_digest(metadata),
                size_bytes=metadata.stat().st_size,
            ),
            AssetFile(
                role=AssetFileRole.COLLISION_MESH,
                relative_path="meshes/source_body.dae",
                sha256=_digest(dae),
                size_bytes=dae.stat().st_size,
            ),
        ],
    )

    inspection = RobotAssetInspector().inspect(manifest, bundle_root)

    assert inspection.status is InspectionStatus.INVALID
    assert "ASSET_OUTSIDE_ALLOWED_ROOT" in {error.code for error in inspection.errors}
    assert not generated_stl.exists()
    _assert_no_host_path(inspection.model_dump(mode="json"), tmp_path)


@pytest.mark.parametrize("meshdir_kind", ["absolute_inside", "relative_escape"])
def test_mujoco_compiler_directories_must_remain_inside_snapshot(
    tmp_path: Path,
    meshdir_kind: str,
) -> None:
    bundle_root = tmp_path / "robot"
    urdf = _write_valid_robot(bundle_root)
    original_discovery = discover_robot_bundle(bundle_root)
    if meshdir_kind == "absolute_inside":
        meshdir = (bundle_root / "meshes").resolve().as_posix()
    else:
        meshdir = "../../outside-meshes"
    urdf.write_text(
        urdf.read_text(encoding="utf-8").replace(
            "</robot>",
            f'  <mujoco><compiler meshdir="{meshdir}"/></mujoco>\n</robot>',
        ),
        encoding="utf-8",
    )

    with pytest.raises(RobotAssetDiscoveryError) as caught:
        discover_robot_bundle(bundle_root)

    assert caught.value.code == "ASSET_OUTSIDE_ALLOWED_ROOT"
    assert caught.value.api_error.details == {"attribute": "meshdir"}
    _assert_no_host_path(caught.value.api_error.model_dump(mode="json"), tmp_path)

    # Recompute hashes from the modified files while retaining the old
    # manifest shape to model a bundle registered by a previous release.
    persisted_manifest = _bundle(bundle_root, original_discovery)
    inspection = RobotAssetInspector().inspect(persisted_manifest, bundle_root)
    assert inspection.status is InspectionStatus.INVALID
    assert "ASSET_OUTSIDE_ALLOWED_ROOT" in {error.code for error in inspection.errors}
    _assert_no_host_path(inspection.model_dump(mode="json"), tmp_path)


def test_directory_discovery_ignores_urdfs_in_hidden_subtrees(tmp_path: Path) -> None:
    bundle_root = tmp_path / "robot"
    primary = _write_valid_robot(bundle_root)
    hidden = bundle_root / ".private"
    hidden.mkdir()
    (hidden / "other.urdf").write_text(primary.read_text(encoding="utf-8"), encoding="utf-8")

    discovery = discover_robot_bundle(bundle_root)

    assert discovery.primary_urdf == primary.resolve()
    assert all(".private" not in item.path.parts for item in discovery.files)


def test_mesh_symlink_cannot_escape_candidate_bundle(tmp_path: Path) -> None:
    outside = tmp_path / "outside.stl"
    outside.write_text("solid outside\nendsolid outside\n", encoding="utf-8")
    bundle_root = tmp_path / "robot"
    _write_valid_robot(bundle_root)
    mesh = bundle_root / "meshes" / "body.stl"
    mesh.unlink()
    try:
        os.symlink(outside, mesh)
    except OSError:
        pytest.skip("The current Windows account cannot create symlinks.")

    with pytest.raises(RobotAssetDiscoveryError) as caught:
        discover_robot_bundle(bundle_root)

    assert caught.value.code == "ASSET_OUTSIDE_ALLOWED_ROOT"
    _assert_no_host_path(caught.value.api_error.model_dump(mode="json"), tmp_path)


def test_directory_discovery_rejects_ambiguous_urdfs_with_relative_candidates(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "robot"
    first = _write_valid_robot(bundle_root)
    second = bundle_root / "variants" / "other.urdf"
    second.parent.mkdir(parents=True)
    second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(RobotAssetDiscoveryError) as caught:
        discover_robot_bundle(bundle_root)

    assert caught.value.code == "BUNDLE_AMBIGUOUS"
    assert caught.value.candidates == ("urdf/robot.urdf", "variants/other.urdf")
    assert caught.value.api_error.details["candidates"] == list(caught.value.candidates)
    _assert_no_host_path(caught.value.api_error.model_dump(mode="json"), tmp_path)


def test_corrupt_xml_returns_parse_error_without_parser_or_host_details(tmp_path: Path) -> None:
    bundle_root = tmp_path / "robot"
    urdf = bundle_root / "robot.urdf"
    bundle_root.mkdir()
    urdf.write_text("<robot name='broken'><link></robot>", encoding="utf-8")

    with pytest.raises(RobotAssetDiscoveryError) as caught:
        discover_robot_bundle(bundle_root)

    assert caught.value.code == "ROBOT_URDF_PARSE_FAILED"
    assert caught.value.api_error.details == {"exception_type": "ParseError"}
    _assert_no_host_path(caught.value.api_error.model_dump(mode="json"), tmp_path)

    manifest = AssetBundle(
        asset_id=_asset_id("corrupt"),
        kind=AssetKind.ROBOT_BUNDLE,
        category=AssetCategory.ROBOT_MODEL,
        display_name="Corrupt",
        primary_file="robot.urdf",
        files=[
            AssetFile(
                role=AssetFileRole.ROBOT_DESCRIPTION,
                relative_path="robot.urdf",
                sha256=_digest(urdf),
                size_bytes=urdf.stat().st_size,
            )
        ],
    )
    inspection = RobotAssetInspector().inspect(manifest, bundle_root)
    assert inspection.status is InspectionStatus.INVALID
    assert "ROBOT_URDF_PARSE_FAILED" in {error.code for error in inspection.errors}
    _assert_no_host_path(inspection.model_dump(mode="json"), tmp_path)


def test_hash_mutation_blocks_primary_content_parsing(tmp_path: Path) -> None:
    bundle_root = tmp_path / "robot"
    urdf = _write_valid_robot(bundle_root)
    discovery = discover_robot_bundle(bundle_root)
    manifest = _bundle(bundle_root, discovery)
    original_size = urdf.stat().st_size
    payload = urdf.read_text(encoding="utf-8")
    urdf.write_text(payload.replace("test_robot", "best_robot"), encoding="utf-8")
    assert urdf.stat().st_size == original_size

    inspection = RobotAssetInspector().inspect(manifest, bundle_root)

    assert inspection.status is InspectionStatus.INVALID
    assert "ASSET_HASH_MISMATCH" in {error.code for error in inspection.errors}
    assert inspection.metadata["content_parsed"] is False
    _assert_no_host_path(inspection.model_dump(mode="json"), tmp_path)


def test_duplicate_names_bad_references_and_limits_are_structured_errors(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "robot"
    urdf = bundle_root / "robot.urdf"
    bundle_root.mkdir()
    urdf.write_text(
        """<robot name="invalid">
  <link name="base"/><link name="base"/>
  <joint name="bad" type="revolute">
    <parent link="missing"/><child link="base"/>
    <limit lower="2" upper="1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )

    with pytest.raises(RobotAssetDiscoveryError) as caught:
        discover_robot_bundle(bundle_root)

    assert caught.value.code == "ROBOT_URDF_INVALID"
    _assert_no_host_path(caught.value.api_error.model_dump(mode="json"), tmp_path)
