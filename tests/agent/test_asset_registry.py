from __future__ import annotations

import os
from pathlib import Path

import pytest

from hhtools.contracts import (
    AssetCategory,
    AssetDetected,
    AssetFileRole,
    AssetRegistrationRequest,
)
from hhtools.services.assets import (
    AssetRegistry,
    AssetServiceError,
    DiscoveredAsset,
    DiscoveredAssetFile,
)


def _request(relative_path: str, **overrides) -> AssetRegistrationRequest:
    return AssetRegistrationRequest(
        root_id="motion-library",
        relative_path=relative_path,
        **overrides,
    )


def _assert_code(captured: pytest.ExceptionInfo[AssetServiceError], code: str) -> None:
    assert captured.value.code == code
    assert captured.value.api_error.code == code


def test_registration_is_content_addressed_and_never_exposes_host_paths(
    tmp_path: Path,
) -> None:
    library = tmp_path / "private-host-library"
    library.mkdir()
    motion = library / "walk.npz"
    motion.write_bytes(b"stable motion bytes")
    data_dir = tmp_path / "registry-data"
    registry = AssetRegistry(data_dir, {"motion-library": library})

    first = registry.register(_request("walk.npz", display_name="Walk"))
    second = registry.register(_request("walk.npz", display_name="Renamed duplicate"))

    assert first.asset_id == second.asset_id
    assert first.asset_id.startswith("asset:sha256:")
    assert first == second
    assert first.primary_file == "walk.npz"
    assert first.files[0].sha256 != "0" * 64
    assert first.source is not None
    assert first.source.root_id == "motion-library"
    assert first.source.logical_path == "walk.npz"
    assert registry.artifact_root == data_dir / "artifacts"
    assert registry.artifact_root.is_dir()

    payload = first.model_dump_json()
    assert str(library) not in payload
    assert str(tmp_path) not in payload
    for database_file in data_dir.glob("assets.sqlite3*"):
        assert str(library).encode() not in database_file.read_bytes()


def test_explicit_bundle_persists_and_searches_across_registry_instances(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    bundle_dir = library / "omomo" / "sub10"
    bundle_dir.mkdir(parents=True)
    motion = bundle_dir / "motion.pkl"
    mesh = bundle_dir / "largebox.obj"
    motion.write_bytes(b"motion")
    mesh.write_bytes(b"mesh")
    discovery = DiscoveredAsset(
        primary_file=motion,
        files=(
            DiscoveredAssetFile(motion, role=AssetFileRole.MOTION),
            DiscoveredAssetFile(mesh, role=AssetFileRole.OBJECT_MESH),
        ),
        category=AssetCategory.OBJECT_INTERACTION,
        display_name="OMOMO Large Box",
        detected=AssetDetected(
            dataset="omomo",
            reference="smplx",
            recommended_backend="interaction_mesh",
        ),
        metadata={"subject": "sub10", "tags": ["box", "interaction"]},
    )
    data_dir = tmp_path / "data"
    first_registry = AssetRegistry(data_dir, {"motion-library": library})

    registered = first_registry.register(_request("omomo/sub10"), discovery=discovery)

    second_registry = AssetRegistry(data_dir, {"motion-library": lambda: library})
    restored = second_registry.get(registered.asset_id)
    result = second_registry.search(
        query="large box",
        category="object_interaction",
        dataset="OMOMO",
        reference="SMPLX",
    )

    assert restored == registered
    assert result.total == 1
    assert result.limit == 100
    assert result.offset == 0
    assert result.assets == [registered]
    assert {item.role.value for item in restored.files} == {"motion", "object_mesh"}
    assert restored.primary_file == "motion.pkl"
    assert {item.relative_path for item in restored.files} == {"motion.pkl", "largebox.obj"}


def test_bundle_identity_is_stable_when_the_same_tree_moves_between_roots(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first-root"
    second_root = tmp_path / "second-root"
    first_bundle = first_root / "dataset-a" / "clip"
    second_bundle = second_root / "different-parent" / "clip"
    first_bundle.mkdir(parents=True)
    second_bundle.mkdir(parents=True)
    for bundle in (first_bundle, second_bundle):
        (bundle / "motion.npz").write_bytes(b"same motion")
        (bundle / "metadata.json").write_bytes(b'{"same": true}')

    first_registry = AssetRegistry(tmp_path / "data-a", {"motion-library": first_root})
    second_registry = AssetRegistry(tmp_path / "data-b", {"motion-library": second_root})

    first = first_registry.register(_request("dataset-a/clip"))
    second = second_registry.register(_request("different-parent/clip"))

    assert first.asset_id == second.asset_id
    assert first.primary_file == second.primary_file == "motion.npz"
    assert [item.relative_path for item in first.files] == [
        item.relative_path for item in second.files
    ]


def test_bundle_identity_binds_routing_semantics(tmp_path: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    motion = library / "motion.npz"
    motion.write_bytes(b"same bytes")
    registry = AssetRegistry(tmp_path / "data", {"motion-library": library})

    amass = registry.register(
        _request("motion.npz"),
        discovery=DiscoveredAsset(
            primary_file=motion,
            files=(DiscoveredAssetFile(motion, role=AssetFileRole.MOTION),),
            detected=AssetDetected(
                dataset="amass",
                reference="smpl",
                recommended_backend="newton",
            ),
        ),
    )
    motion_x = registry.register(
        _request("motion.npz"),
        discovery=DiscoveredAsset(
            primary_file=motion,
            files=(DiscoveredAssetFile(motion, role=AssetFileRole.MOTION),),
            detected=AssetDetected(
                dataset="motion_x",
                reference="smplx",
                recommended_backend="newton",
            ),
        ),
    )

    assert amass.asset_id != motion_x.asset_id


@pytest.mark.parametrize(
    "unsafe_path",
    ["../secret.npz", "/private/secret.npz", "C:/private/secret.npz", r"..\secret.npz"],
)
def test_registration_defensively_rejects_non_relative_paths(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    library = tmp_path / "library"
    library.mkdir()
    registry = AssetRegistry(tmp_path / "data", {"motion-library": library})
    # model_construct simulates a future adapter bug or direct service caller
    # bypassing normal Pydantic validation.
    request = AssetRegistrationRequest.model_construct(
        root_id="motion-library",
        relative_path=unsafe_path,
        display_name=None,
        kind=None,
        category=None,
        recursive=True,
    )

    with pytest.raises(AssetServiceError) as captured:
        registry.register(request)

    _assert_code(captured, "ASSET_OUTSIDE_ALLOWED_ROOT")
    assert str(tmp_path) not in captured.value.error.message
    assert str(tmp_path) not in str(captured.value.error.details)


def test_registration_rejects_unknown_roots_with_a_structured_error(tmp_path: Path) -> None:
    registry = AssetRegistry(tmp_path / "data", {})
    request = AssetRegistrationRequest(root_id="not-configured", relative_path="motion.npz")

    with pytest.raises(AssetServiceError) as captured:
        registry.register(request)

    _assert_code(captured, "ASSET_OUTSIDE_ALLOWED_ROOT")
    assert captured.value.error.stage.value == "asset_registration"
    assert captured.value.error.details == {"root_id": "not-configured"}


def test_registration_rejects_a_symlink_that_escapes_the_allowed_root(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    library.mkdir()
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"not allowed")
    linked = library / "escape.npz"
    try:
        os.symlink(outside, linked)
    except OSError as exc:
        pytest.skip(f"File symlinks are unavailable in this environment: {exc}")
    registry = AssetRegistry(tmp_path / "data", {"motion-library": library})

    with pytest.raises(AssetServiceError) as captured:
        registry.register(_request("escape.npz"))

    _assert_code(captured, "ASSET_OUTSIDE_ALLOWED_ROOT")


def test_registration_revalidates_every_file_reported_by_an_inspector(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    library.mkdir()
    motion = library / "motion.npz"
    motion.write_bytes(b"motion")
    outside_mesh = tmp_path / "secret.obj"
    outside_mesh.write_bytes(b"mesh")
    discovery = DiscoveredAsset(
        primary_file=motion,
        files=(
            DiscoveredAssetFile(motion, role=AssetFileRole.MOTION),
            DiscoveredAssetFile(outside_mesh, role=AssetFileRole.OBJECT_MESH),
        ),
        category=AssetCategory.OBJECT_INTERACTION,
    )
    registry = AssetRegistry(tmp_path / "data", {"motion-library": library})

    with pytest.raises(AssetServiceError) as captured:
        registry.register(_request("motion.npz"), discovery=discovery)

    _assert_code(captured, "ASSET_OUTSIDE_ALLOWED_ROOT")


def test_resolve_file_rechecks_hash_and_can_return_a_trusted_internal_path(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    library.mkdir()
    motion = library / "motion.npz"
    motion.write_bytes(b"version one")
    registry = AssetRegistry(tmp_path / "data", {"motion-library": library})
    bundle = registry.register(_request("motion.npz"))

    assert registry.resolve_file(bundle.asset_id) == motion.resolve()
    motion.write_bytes(b"version two")

    with pytest.raises(AssetServiceError) as captured:
        registry.resolve_file(bundle.asset_id)

    _assert_code(captured, "ASSET_HASH_MISMATCH")
    assert registry.resolve_file(bundle.asset_id, verify_hash=False) == motion.resolve()
    assert str(library) not in str(captured.value.error.model_dump(mode="json"))


def test_missing_assets_and_invalid_search_parameters_have_stable_codes(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    library.mkdir()
    registry = AssetRegistry(tmp_path / "data", {"motion-library": library})

    with pytest.raises(AssetServiceError) as missing:
        registry.get(f"asset:sha256:{'0' * 64}")
    with pytest.raises(AssetServiceError) as invalid_limit:
        registry.search(limit=0)
    with pytest.raises(AssetServiceError) as invalid_category:
        registry.search(category="anything")

    _assert_code(missing, "ASSET_NOT_FOUND")
    _assert_code(invalid_limit, "INVALID_PARAMETER")
    _assert_code(invalid_category, "INVALID_PARAMETER")
