from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from hhtools.contracts import (
    AssetBundle,
    AssetCategory,
    AssetDetected,
    AssetFile,
    AssetFileRole,
    AssetKind,
    InspectionStatus,
)
from hhtools.services import asset_inspection as asset_inspection_module
from hhtools.services.asset_inspection import (
    MotionAssetDiscoveryError,
    MotionAssetInspector,
    discover_primary,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset_id(seed: str) -> str:
    return f"asset:sha256:{hashlib.sha256(seed.encode()).hexdigest()}"


def _manifest_file(
    root: Path,
    relative_path: str,
    role: AssetFileRole,
    *,
    required: bool = True,
    missing_hash: str | None = None,
) -> AssetFile:
    path = root.joinpath(*relative_path.split("/"))
    return AssetFile(
        role=role,
        relative_path=relative_path,
        sha256=_digest(path) if path.is_file() else (missing_hash or "0" * 64),
        size_bytes=path.stat().st_size if path.is_file() else 0,
        required=required,
    )


def _bundle(
    root: Path,
    *,
    primary: str,
    category: AssetCategory,
    dataset: str,
    files: list[tuple[str, AssetFileRole]],
) -> AssetBundle:
    return AssetBundle(
        asset_id=_asset_id(primary),
        kind=AssetKind.MOTION_BUNDLE,
        category=category,
        display_name=Path(primary).stem,
        primary_file=primary,
        files=[_manifest_file(root, path, role) for path, role in files],
        detected=AssetDetected(dataset=dataset),
    )


def _write_unified_npz(
    path: Path,
    *,
    positions: np.ndarray | None = None,
    framerate: float = 30.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pos = (
        np.asarray(positions, dtype=np.float32)
        if positions is not None
        else np.zeros((3, 2, 3), dtype=np.float32)
    )
    quat = np.zeros((pos.shape[0], pos.shape[1], 4), dtype=np.float32)
    quat[..., 3] = 1.0
    np.savez(
        path,
        schema_version=np.array("1"),
        name=np.array(path.stem),
        framerate=np.array(framerate),
        up_axis=np.array("Z"),
        source_format=np.array("npz"),
        bone_names=np.array([f"joint_{index}" for index in range(pos.shape[1])]),
        parent_indices=np.array([-1, *range(max(0, pos.shape[1] - 1))], dtype=np.int32),
        positions=pos,
        quaternions=quat,
    )


def _write_holosoma_manifest(path: Path, joint_count: int = 53) -> None:
    path.write_text(
        json.dumps(
            {
                "framerate": {"raw_hz": 120.0, "recommended_downsample": 4},
                "skeleton": {
                    "joint_names": [f"joint_{index}" for index in range(joint_count)],
                    "parent_indices": [-1, *range(joint_count - 1)],
                },
            }
        ),
        encoding="utf-8",
    )


def test_plain_motion_inspection_reads_lightweight_npz_metadata(tmp_path: Path) -> None:
    primary = "unified_npz/walk.npz"
    _write_unified_npz(tmp_path / primary)
    bundle = _bundle(
        tmp_path,
        primary=primary,
        category=AssetCategory.PLAIN_MOTION,
        dataset="unified_npz",
        files=[(primary, AssetFileRole.MOTION)],
    )

    inspection = MotionAssetInspector().inspect(bundle, tmp_path)

    assert inspection.status is InspectionStatus.VALID
    assert inspection.dataset == "unified_npz"
    assert inspection.reference_model == "smpl"
    assert inspection.frame_count == 3
    assert inspection.frame_rate_hz == 30.0
    assert inspection.duration_seconds == pytest.approx(2 / 30)
    assert inspection.joint_count == 2
    assert inspection.has_object is False
    assert inspection.has_terrain is False
    assert inspection.metadata["recommended_backend"] == "newton"
    assert str(tmp_path) not in str(inspection.model_dump(mode="json"))


def test_npz_metadata_is_bounded_before_decompression(tmp_path: Path) -> None:
    primary = tmp_path / "unified_npz" / "oversized-meta.npz"
    _write_unified_npz(primary)
    with np.load(primary, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["meta_json"] = np.array("x" * (64 * 1024 + 1))
    np.savez_compressed(primary, **values)

    discovery = discover_primary(primary)
    bundle = _bundle(
        tmp_path,
        primary=primary.relative_to(tmp_path).as_posix(),
        category=AssetCategory.PLAIN_MOTION,
        dataset=discovery.dataset,
        files=[(primary.relative_to(tmp_path).as_posix(), AssetFileRole.MOTION)],
    )
    inspection = MotionAssetInspector().inspect(bundle, tmp_path)

    assert discovery.dataset == "unified_npz"
    assert inspection.status is InspectionStatus.INVALID
    assert any(
        "meta_json exceeds" in str(error.details.get("reason", ""))
        for error in inspection.errors
    )


def test_unified_npz_requires_the_scalar_fields_consumed_by_the_real_loader(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "unified_npz" / "missing_loader_fields.npz"
    primary.parent.mkdir(parents=True)
    positions = np.zeros((2, 2, 3), dtype=np.float32)
    quaternions = np.zeros((2, 2, 4), dtype=np.float32)
    quaternions[..., 3] = 1.0
    # These four arrays used to be enough for inspection, but load_npz also
    # unconditionally consumes schema_version, name, framerate, and up_axis.
    np.savez(
        primary,
        bone_names=np.array(["root", "child"]),
        parent_indices=np.array([-1, 0], dtype=np.int32),
        positions=positions,
        quaternions=quaternions,
    )
    relative = primary.relative_to(tmp_path).as_posix()
    bundle = _bundle(
        tmp_path,
        primary=relative,
        category=AssetCategory.PLAIN_MOTION,
        dataset="unified_npz",
        files=[(relative, AssetFileRole.MOTION)],
    )

    inspection = MotionAssetInspector().inspect(bundle, tmp_path)

    assert inspection.status is InspectionStatus.INVALID
    error = next(error for error in inspection.errors if error.code == "MOTION_PARSE_FAILED")
    assert "required fields" in error.details["reason"]


def test_raw_amass_requires_a_complete_loader_parameter_schema(tmp_path: Path) -> None:
    primary = tmp_path / "AMASS" / "subject" / "incomplete_stageii.npz"
    primary.parent.mkdir(parents=True)
    # trans + pose_body previously produced a frame count and a false success,
    # although AmassAdapter then fails because betas and root_orient are required.
    np.savez(
        primary,
        trans=np.zeros((3, 3), dtype=np.float32),
        pose_body=np.zeros((3, 63), dtype=np.float32),
        mocap_frame_rate=np.array(30.0),
    )
    relative = primary.relative_to(tmp_path).as_posix()
    bundle = _bundle(
        tmp_path,
        primary=relative,
        category=AssetCategory.PLAIN_MOTION,
        dataset="amass",
        files=[(relative, AssetFileRole.MOTION)],
    )

    inspection = MotionAssetInspector().inspect(bundle, tmp_path)

    assert inspection.status is InspectionStatus.INVALID
    error = next(error for error in inspection.errors if error.code == "MOTION_PARSE_FAILED")
    assert "betas" in error.details["reason"]


def test_holosoma_npy_must_match_the_manifest_joint_position_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip_dir = tmp_path / "meshmimic" / "holosoma" / "parkour_1"
    clip_dir.mkdir(parents=True)
    primary = clip_dir / "parkour_1.npy"
    # A (T, J) matrix previously passed because the inspector only required
    # two axes; MeshmimicHolosomaAdapter requires exactly (T, J, 3).
    np.save(primary, np.zeros((4, 53), dtype=np.float32))
    manifest = clip_dir / "source.yaml"
    _write_holosoma_manifest(manifest)
    terrain = clip_dir / "terrain.obj"
    terrain.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    primary_relative = primary.relative_to(tmp_path).as_posix()
    manifest_relative = manifest.relative_to(tmp_path).as_posix()
    terrain_relative = terrain.relative_to(tmp_path).as_posix()
    bundle = _bundle(
        tmp_path,
        primary=primary_relative,
        category=AssetCategory.TERRAIN_SCENE,
        dataset="meshmimic_holosoma",
        files=[
            (primary_relative, AssetFileRole.MOTION),
            (manifest_relative, AssetFileRole.METADATA),
            (terrain_relative, AssetFileRole.TERRAIN_MESH),
        ],
    )
    numpy_load = np.load

    def safe_numpy_load(*args, **kwargs):
        assert kwargs.get("allow_pickle") is False
        return numpy_load(*args, **kwargs)

    monkeypatch.setattr(asset_inspection_module.np, "load", safe_numpy_load)

    inspection = MotionAssetInspector().inspect(bundle, tmp_path)

    assert inspection.status is InspectionStatus.INVALID
    error = next(error for error in inspection.errors if error.code == "MOTION_PARSE_FAILED")
    assert "(frames, joints, 3)" in error.details["reason"]


def test_complete_raw_amass_stageii_structure_remains_accepted(tmp_path: Path) -> None:
    primary = tmp_path / "AMASS" / "subject" / "valid_stageii.npz"
    primary.parent.mkdir(parents=True)
    np.savez(
        primary,
        trans=np.zeros((3, 3), dtype=np.float32),
        betas=np.zeros(16, dtype=np.float32),
        root_orient=np.zeros((3, 3), dtype=np.float32),
        pose_body=np.zeros((3, 63), dtype=np.float32),
        mocap_frame_rate=np.array(60.0),
    )
    relative = primary.relative_to(tmp_path).as_posix()
    bundle = _bundle(
        tmp_path,
        primary=relative,
        category=AssetCategory.PLAIN_MOTION,
        dataset="amass",
        files=[(relative, AssetFileRole.MOTION)],
    )

    inspection = MotionAssetInspector().inspect(bundle, tmp_path)

    assert inspection.status is InspectionStatus.VALID
    assert inspection.frame_count == 3
    assert inspection.frame_rate_hz == 60.0
    assert inspection.metadata["parameter_schema"] == "stageii"


def test_valid_holosoma_positions_and_manifest_remain_accepted(tmp_path: Path) -> None:
    clip_dir = tmp_path / "meshmimic" / "holosoma" / "parkour_2"
    clip_dir.mkdir(parents=True)
    primary = clip_dir / "parkour_2.npy"
    np.save(primary, np.zeros((8, 53, 3), dtype=np.float32))
    manifest = clip_dir / "source.yaml"
    _write_holosoma_manifest(manifest)
    terrain = clip_dir / "terrain.obj"
    terrain.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    primary_relative = primary.relative_to(tmp_path).as_posix()
    bundle = _bundle(
        tmp_path,
        primary=primary_relative,
        category=AssetCategory.TERRAIN_SCENE,
        dataset="meshmimic_holosoma",
        files=[
            (primary_relative, AssetFileRole.MOTION),
            (manifest.relative_to(tmp_path).as_posix(), AssetFileRole.METADATA),
            (terrain.relative_to(tmp_path).as_posix(), AssetFileRole.TERRAIN_MESH),
        ],
    )

    inspection = MotionAssetInspector().inspect(bundle, tmp_path)

    assert inspection.status is InspectionStatus.VALID
    assert inspection.frame_count == 2
    assert inspection.frame_rate_hz == 30.0
    assert inspection.joint_count == 53


def test_object_interaction_requires_and_discovers_object_mesh(tmp_path: Path) -> None:
    primary = tmp_path / "OMOMO" / "sub10_largebox_000" / "sub10_largebox_000.pkl"
    primary.parent.mkdir(parents=True)
    with primary.open("wb") as handle:
        pickle.dump({"motion": [[0.0]], "seq_name": primary.stem}, handle)
    mesh = primary.parent / "largebox_cleaned_simplified.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    primary_relative = primary.relative_to(tmp_path).as_posix()
    mesh_relative = mesh.relative_to(tmp_path).as_posix()
    bundle = _bundle(
        tmp_path,
        primary=primary_relative,
        category=AssetCategory.OBJECT_INTERACTION,
        dataset="omomo",
        files=[
            (primary_relative, AssetFileRole.MOTION),
            (mesh_relative, AssetFileRole.OBJECT_MESH),
        ],
    )

    discovery = discover_primary(primary.parent)
    inspection = MotionAssetInspector().inspect(bundle, tmp_path)

    assert discovery.primary_path == primary.resolve()
    assert discovery.category is AssetCategory.OBJECT_INTERACTION
    assert discovery.sidecars[AssetFileRole.OBJECT_MESH] == (mesh.resolve(),)
    assert inspection.status is InspectionStatus.VALID_WITH_WARNINGS
    assert inspection.dataset == "omomo"
    assert inspection.reference_model == "smplx"
    assert inspection.has_object is True
    assert inspection.metadata["recommended_backend"] == "interaction_mesh"
    assert inspection.metadata["pickle_executed"] is False
    assert inspection.metadata["content_parsed"] is False
    assert inspection.metadata["content_validation_code"] == "CONTENT_REQUIRES_ISOLATED_VALIDATION"


def test_terrain_scene_inspection_keeps_terrain_sidecar_bound_to_bundle(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "parc_ms" / "parkour_1" / "parkour_1.pkl"
    primary.parent.mkdir(parents=True)
    with primary.open("wb") as handle:
        pickle.dump({"motion_data": {}, "terrain_data": {}}, handle)
    terrain = primary.parent / "parkour_1_terrain.obj"
    terrain.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    primary_relative = primary.relative_to(tmp_path).as_posix()
    terrain_relative = terrain.relative_to(tmp_path).as_posix()
    bundle = _bundle(
        tmp_path,
        primary=primary_relative,
        category=AssetCategory.TERRAIN_SCENE,
        dataset="parc_ms",
        files=[
            (primary_relative, AssetFileRole.MOTION),
            (terrain_relative, AssetFileRole.TERRAIN_MESH),
        ],
    )

    discovery = discover_primary(primary.parent)
    inspection = MotionAssetInspector().inspect(bundle, tmp_path)

    assert discovery.dataset == "parc_ms"
    assert discovery.reference == "smpl"
    assert discovery.sidecars[AssetFileRole.TERRAIN_MESH] == (terrain.resolve(),)
    assert inspection.status is InspectionStatus.VALID_WITH_WARNINGS
    assert inspection.category is AssetCategory.TERRAIN_SCENE
    assert inspection.has_terrain is True
    assert inspection.metadata["recommended_backend"] == "interaction_mesh"


def test_missing_required_object_sidecar_returns_stable_bundle_error(tmp_path: Path) -> None:
    primary = tmp_path / "OMOMO" / "sub10_largebox_000" / "sub10_largebox_000.pkl"
    primary.parent.mkdir(parents=True)
    with primary.open("wb") as handle:
        pickle.dump({"motion": [[0.0]]}, handle)
    primary_relative = primary.relative_to(tmp_path).as_posix()
    mesh_relative = f"{primary.parent.relative_to(tmp_path).as_posix()}/largebox.obj"
    bundle = _bundle(
        tmp_path,
        primary=primary_relative,
        category=AssetCategory.OBJECT_INTERACTION,
        dataset="omomo",
        files=[
            (primary_relative, AssetFileRole.MOTION),
            (mesh_relative, AssetFileRole.OBJECT_MESH),
        ],
    )

    inspection = MotionAssetInspector().inspect(bundle, tmp_path)

    assert inspection.status is InspectionStatus.INVALID
    assert "BUNDLE_INCOMPLETE" in {error.code for error in inspection.errors}
    assert all(
        str(tmp_path) not in str(error.model_dump(mode="json")) for error in inspection.errors
    )


def test_parc_requires_an_actual_terrain_mesh_not_an_other_sidecar(tmp_path: Path) -> None:
    primary = tmp_path / "parc_ms" / "parkour_1" / "parkour_1.pkl"
    primary.parent.mkdir(parents=True)
    with primary.open("wb") as handle:
        pickle.dump({"motion_data": {}, "terrain_data": {}}, handle)
    other = primary.with_suffix(".npz")
    _write_unified_npz(other)
    primary_relative = primary.relative_to(tmp_path).as_posix()
    other_relative = other.relative_to(tmp_path).as_posix()
    bundle = _bundle(
        tmp_path,
        primary=primary_relative,
        category=AssetCategory.TERRAIN_SCENE,
        dataset="parc_ms",
        files=[
            (primary_relative, AssetFileRole.MOTION),
            (other_relative, AssetFileRole.OTHER),
        ],
    )

    inspection = MotionAssetInspector().inspect(bundle, tmp_path)

    assert inspection.status is InspectionStatus.INVALID
    assert inspection.has_terrain is False
    assert any(
        error.code == "BUNDLE_INCOMPLETE"
        and error.details.get("missing_roles") == [AssetFileRole.TERRAIN_MESH.value]
        for error in inspection.errors
    )


def test_corrupt_motion_content_returns_parse_error_without_exception_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = tmp_path / "unified_npz" / "broken.npz"
    primary.parent.mkdir(parents=True)
    primary.write_bytes(b"not a zip archive")
    relative = primary.relative_to(tmp_path).as_posix()
    bundle = _bundle(
        tmp_path,
        primary=relative,
        category=AssetCategory.PLAIN_MOTION,
        dataset="unified_npz",
        files=[(relative, AssetFileRole.MOTION)],
    )
    numpy_load = np.load

    def safe_numpy_load(*args, **kwargs):
        assert kwargs.get("allow_pickle", False) is not True
        return numpy_load(*args, **kwargs)

    monkeypatch.setattr(asset_inspection_module.np, "load", safe_numpy_load)

    inspection = MotionAssetInspector().inspect(bundle, tmp_path)

    assert inspection.status is InspectionStatus.INVALID
    error = next(error for error in inspection.errors if error.code == "MOTION_PARSE_FAILED")
    assert error.details["exception_type"] == "ValueError"
    assert str(tmp_path) not in str(error.model_dump(mode="json"))


def test_nonfinite_motion_values_have_a_distinct_machine_code(tmp_path: Path) -> None:
    primary = tmp_path / "unified_npz" / "nan.npz"
    positions = np.zeros((2, 2, 3), dtype=np.float32)
    positions[1, 0, 2] = np.nan
    _write_unified_npz(primary, positions=positions)
    relative = primary.relative_to(tmp_path).as_posix()
    bundle = _bundle(
        tmp_path,
        primary=relative,
        category=AssetCategory.PLAIN_MOTION,
        dataset="unified_npz",
        files=[(relative, AssetFileRole.MOTION)],
    )

    inspection = MotionAssetInspector().inspect(bundle, tmp_path)

    assert inspection.status is InspectionStatus.INVALID
    assert "MOTION_NONFINITE_VALUES" in {error.code for error in inspection.errors}


def test_content_routing_truth_rejects_declared_dataset_reference_mismatch(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "unified_npz" / "walk.npz"
    _write_unified_npz(primary)
    relative = primary.relative_to(tmp_path).as_posix()
    bundle = _bundle(
        tmp_path,
        primary=relative,
        category=AssetCategory.PLAIN_MOTION,
        dataset="motion_x",
        files=[(relative, AssetFileRole.MOTION)],
    ).model_copy(
        update={
            "detected": AssetDetected(
                dataset="motion_x",
                reference="smplx",
                recommended_backend="interaction_mesh",
            )
        }
    )

    inspection = MotionAssetInspector().inspect(bundle, tmp_path)

    assert inspection.status is InspectionStatus.INVALID
    assert inspection.dataset == "unified_npz"
    assert inspection.reference_model == "smpl"
    mismatch = next(
        error for error in inspection.errors if error.code == "BUNDLE_METADATA_MISMATCH"
    )
    assert set(mismatch.details["mismatches"]) == {
        "dataset",
        "reference",
        "recommended_backend",
    }


def test_directory_discovery_rejects_multiple_logical_clips(tmp_path: Path) -> None:
    _write_unified_npz(tmp_path / "unified_npz" / "one.npz")
    _write_unified_npz(tmp_path / "unified_npz" / "two.npz")

    with pytest.raises(MotionAssetDiscoveryError) as caught:
        discover_primary(tmp_path / "unified_npz")

    assert caught.value.code == "BUNDLE_AMBIGUOUS"
    assert caught.value.candidates == ("one.npz", "two.npz")
