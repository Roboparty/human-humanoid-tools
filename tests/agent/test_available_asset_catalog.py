from __future__ import annotations

import json
import struct
from pathlib import Path, PurePosixPath

import numpy as np
import pytest
from pydantic import ValidationError

import hhtools.services.available_assets as available_assets_module
from hhtools.contracts import (
    AssetInspectionRequest,
    AssetRegistrationRequest,
    AvailableAssetCatalogRequest,
)
from hhtools.contracts.portability import validate_portable_json
from hhtools.services import (
    AgentAssetService,
    AssetRegistry,
    AssetServiceError,
    AvailableAssetCandidate,
    AvailableAssetCatalogLimitError,
    AvailableAssetCatalogService,
    classify_catalog_robot_trajectory,
    iter_bounded_catalog_files,
    require_bounded_catalog_root,
)


def _catalog(
    tmp_path: Path,
    candidates: list[AvailableAssetCandidate],
) -> tuple[AgentAssetService, AvailableAssetCatalogService]:
    root = tmp_path / "motions"
    root.mkdir(exist_ok=True)
    registry = AssetRegistry(tmp_path / "agent-state", {"source": root})
    assets = AgentAssetService(registry)
    return assets, AvailableAssetCatalogService(
        registry,
        {"source": lambda: list(candidates)},
    )


def test_catalog_returns_portable_registration_metadata_and_exact_page(tmp_path: Path) -> None:
    root = tmp_path / "motions"
    root.mkdir()
    walk = root / "walk.bvh"
    run = root / "run.bvh"
    walk.write_text("HIERARCHY\n", encoding="utf-8")
    run.write_text("HIERARCHY\n", encoding="utf-8")
    candidates = [
        AvailableAssetCandidate(
            path=walk,
            display_name="Walk",
            kind="motion_bundle",
            category="plain_motion",
            dataset="lafan",
            reference="lafan_bvh",
        ),
        AvailableAssetCandidate(
            path=run,
            display_name="Run",
            kind="motion_bundle",
            category="plain_motion",
            dataset="lafan",
            reference="lafan_bvh",
        ),
    ]
    assets, catalog = _catalog(tmp_path, candidates)

    page = catalog.list_available(
        AvailableAssetCatalogRequest(query="lafan", limit=1, offset=1)
    )

    assert page.total == 2
    assert page.limit == 1
    assert page.offset == 1
    assert len(page.assets) == 1
    entry = page.assets[0]
    assert entry.display_name == "Walk"
    assert entry.root_id == "source"
    assert entry.relative_path == "walk.bvh"
    assert entry.kind.value == "motion_bundle"
    assert entry.category.value == "plain_motion"
    document = page.model_dump(mode="json", exclude_none=True)
    validate_portable_json(document)
    assert str(tmp_path) not in json.dumps(document)

    bundle = assets.register(
        AssetRegistrationRequest(
            root_id=entry.root_id,
            relative_path=entry.relative_path,
            display_name=entry.display_name,
            kind=entry.kind,
            category=entry.category,
            recursive=entry.recursive,
        )
    )
    inspection = assets.inspect(
        AssetInspectionRequest(
            asset_id=bundle.asset_id,
            verify_hashes=True,
            parse_content=False,
        )
    )
    assert inspection.asset_id == bundle.asset_id
    assert inspection.kind.value == "motion_bundle"


def test_catalog_rejects_unknown_root_and_contract_bounds(tmp_path: Path) -> None:
    _assets, catalog = _catalog(tmp_path, [])

    with pytest.raises(AssetServiceError) as raised:
        catalog.list_available(AvailableAssetCatalogRequest(root_id="unknown"))

    assert raised.value.api_error.code == "ASSET_OUTSIDE_ALLOWED_ROOT"
    assert raised.value.api_error.details == {"root_id": "unknown"}
    with pytest.raises(ValidationError):
        AvailableAssetCatalogRequest(limit=501)
    with pytest.raises(ValidationError):
        AvailableAssetCatalogRequest(offset=-1)
    with pytest.raises(ValidationError):
        AvailableAssetCatalogRequest(typo=True)


def test_catalog_skips_symlink_escape_without_hiding_valid_sibling(tmp_path: Path) -> None:
    root = tmp_path / "motions"
    outside = tmp_path / "private" / "secret.bvh"
    root.mkdir()
    outside.parent.mkdir()
    outside.write_text("private", encoding="utf-8")
    escaped = root / "escape.bvh"
    escaped.symlink_to(outside)
    valid = root / "valid.bvh"
    valid.write_text("HIERARCHY\n", encoding="utf-8")
    _assets, catalog = _catalog(
        tmp_path,
        [
            AvailableAssetCandidate(
                path=valid,
                display_name="Valid",
                kind="motion_bundle",
                category="plain_motion",
            ),
            AvailableAssetCandidate(
                path=escaped,
                display_name="Escaped",
                kind="motion_bundle",
                category="plain_motion",
            )
        ],
    )

    page = catalog.list_available(AvailableAssetCatalogRequest(root_id="source"))

    document = page.model_dump(mode="json")
    assert [(item.display_name, item.relative_path) for item in page.assets] == [
        ("Valid", "valid.bvh")
    ]
    assert str(tmp_path) not in json.dumps(document)


def test_catalog_skips_candidate_with_sidecar_outside_bundle(tmp_path: Path) -> None:
    root = tmp_path / "motions"
    root.mkdir()
    valid = root / "valid.bvh"
    invalid = root / "invalid.npz"
    outside = tmp_path / "private.obj"
    valid.write_text("HIERARCHY\n", encoding="utf-8")
    invalid.write_bytes(b"catalog does not parse candidate bytes")
    outside.write_text("o private\n", encoding="utf-8")
    _assets, catalog = _catalog(
        tmp_path,
        [
            AvailableAssetCandidate(
                path=valid,
                display_name="Valid",
                kind="motion_bundle",
                category="plain_motion",
            ),
            AvailableAssetCandidate(
                path=invalid,
                display_name="Invalid",
                kind="motion_bundle",
                category="terrain_scene",
                required_paths=(outside,),
            ),
        ],
    )

    page = catalog.list_available(AvailableAssetCatalogRequest(root_id="source"))

    assert [item.display_name for item in page.assets] == ["Valid"]


def test_catalog_skips_invalid_candidate_contract_without_hiding_valid_sibling(
    tmp_path: Path,
) -> None:
    root = tmp_path / "motions"
    root.mkdir()
    valid = root / "valid.bvh"
    invalid = root / "invalid.bvh"
    valid.write_text("HIERARCHY\n", encoding="utf-8")
    invalid.write_text("HIERARCHY\n", encoding="utf-8")
    _assets, catalog = _catalog(
        tmp_path,
        [
            AvailableAssetCandidate(
                path=invalid,
                display_name="Invalid",
                kind="motion_bundle",
                category="not-a-category",  # type: ignore[arg-type]
            ),
            AvailableAssetCandidate(
                path=valid,
                display_name="Valid",
                kind="motion_bundle",
                category="plain_motion",
            ),
        ],
    )

    page = catalog.list_available(AvailableAssetCatalogRequest(root_id="source"))

    assert [(item.display_name, item.relative_path) for item in page.assets] == [
        ("Valid", "valid.bvh")
    ]


def test_duplicate_roots_merge_provider_semantics_under_canonical_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    clip = root / "walk.bvh"
    clip.write_text("HIERARCHY\n", encoding="utf-8")
    outside = tmp_path / "outside.obj"
    outside.write_text("o outside\n", encoding="utf-8")
    robot = root / "catalog_bot"
    robot.mkdir()
    (robot / "robot.yaml").write_text(
        "name: catalog_bot\n"
        "display_name: Catalog Bot\n"
        "urdf: robot.urdf\n"
        "dof_order: [hip]\n"
        "ik_map:\n"
        "  hips: base\n",
        encoding="utf-8",
    )
    (robot / "robot.urdf").write_text(
        """<?xml version="1.0"?>
<robot name="catalog_bot">
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
    registry = AssetRegistry(
        tmp_path / "agent-state",
        {"source": root, "motion-library": root, "robot-library": root},
    )
    motion_candidate = AvailableAssetCandidate(
        path=clip,
        display_name="Walk",
        kind="motion_bundle",
        category="plain_motion",
    )
    invalid_motion_candidate = AvailableAssetCandidate(
        path=clip,
        display_name="Invalid walk",
        kind="motion_bundle",
        category="plain_motion",
        required_paths=(outside,),
    )
    robot_candidate = AvailableAssetCandidate(
        path=robot,
        display_name="Catalog Bot",
        kind="robot_bundle",
        category="robot_model",
    )
    catalog = AvailableAssetCatalogService(
        registry,
        {
            "source": lambda: [motion_candidate],
            "motion-library": lambda: [invalid_motion_candidate],
            "robot-library": lambda: [robot_candidate],
        },
        preferred_root_ids=("motion-library", "robot-library", "source"),
    )

    canonical = catalog.list_available(AvailableAssetCatalogRequest())
    via_alias = catalog.list_available(AvailableAssetCatalogRequest(root_id="robot-library"))

    assert canonical.total == 2
    assert {item.kind.value for item in canonical.assets} == {
        "motion_bundle",
        "robot_bundle",
    }
    assert {item.root_id for item in canonical.assets} == {"motion-library"}
    assert via_alias == canonical
    assets = AgentAssetService(registry)
    registered = {
        assets.register(
            AssetRegistrationRequest(
                root_id=item.root_id,
                relative_path=item.relative_path,
                display_name=item.display_name,
                kind=item.kind,
                category=item.category,
                recursive=item.recursive,
            )
        ).kind.value
        for item in canonical.assets
    }
    assert registered == {"motion_bundle", "robot_bundle"}


def test_catalog_has_hard_tree_and_candidate_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "motions"
    root.mkdir()
    for name in ("one", "two", "three"):
        (root / name).write_text(name, encoding="utf-8")

    with pytest.raises(AvailableAssetCatalogLimitError):
        require_bounded_catalog_root(root, max_entries=2)

    candidates = [
        AvailableAssetCandidate(
            path=root / name,
            display_name=name,
            kind="motion_bundle",
            category="plain_motion",
        )
        for name in ("one", "two", "three")
    ]
    _assets, catalog = _catalog(tmp_path, candidates)
    monkeypatch.setattr("hhtools.services.available_assets.MAX_AVAILABLE_ASSET_CANDIDATES", 2)
    with pytest.raises(AssetServiceError) as raised:
        catalog.list_available(AvailableAssetCatalogRequest())

    assert raised.value.api_error.code == "AVAILABLE_ASSET_CATALOG_LIMIT_EXCEEDED"
    assert raised.value.api_error.details["max_candidates"] == 2

    duplicate_catalog = AvailableAssetCatalogService(
        catalog._registration_hints,  # noqa: SLF001 - exact work-budget regression
        {"source": lambda: [candidates[0]] * 3},
    )
    with pytest.raises(AssetServiceError) as duplicate_raised:
        duplicate_catalog.list_available(AvailableAssetCatalogRequest())
    assert duplicate_raised.value.api_error.code == "AVAILABLE_ASSET_CATALOG_LIMIT_EXCEEDED"


def test_bounded_iterator_counts_but_never_descends_into_hidden_trees(tmp_path: Path) -> None:
    root = tmp_path / "motions"
    hidden = root / ".private" / "AMASS"
    hidden.mkdir(parents=True)
    (hidden / "secret.bvh").write_text("HIERARCHY\n", encoding="utf-8")

    with pytest.raises(AvailableAssetCatalogLimitError):
        list(iter_bounded_catalog_files(root, extensions=frozenset({".bvh"}), max_entries=0))
    visible = list(
        iter_bounded_catalog_files(root, extensions=frozenset({".bvh"}), max_entries=1)
    )

    assert visible == []


def test_robot_catalog_scanner_ignores_hidden_presets_and_symlinks(
    tmp_path: Path,
) -> None:
    from hhtools.robot.registry import list_presets_in_root_readonly

    root = tmp_path / "robots"
    hidden = root / ".private"
    hidden.mkdir(parents=True)
    (hidden / "robot.yaml").write_text(
        "name: private\nurdf: robot.urdf\n",
        encoding="utf-8",
    )
    (hidden / "robot.urdf").write_text(
        '<robot name="private"><link name="base"/></robot>\n',
        encoding="utf-8",
    )
    template = root / "_template"
    template.mkdir()
    (template / "robot.yaml").write_text(
        "name: template\nurdf: robot.urdf\n",
        encoding="utf-8",
    )
    (template / "robot.urdf").write_text(
        '<robot name="template"><link name="base"/></robot>\n',
        encoding="utf-8",
    )
    for index in range(10):
        (hidden / f"ignored-{index}.mesh").write_text("mesh", encoding="utf-8")
        (template / f"ignored-{index}.mesh").write_text("mesh", encoding="utf-8")
    (root / "visible-link").symlink_to(hidden, target_is_directory=True)

    with pytest.raises(AvailableAssetCatalogLimitError):
        require_bounded_catalog_root(root, max_entries=2)
    require_bounded_catalog_root(root, max_entries=3)

    assert list_presets_in_root_readonly(root) == []


def test_robot_trajectory_detection_uses_relative_names_and_safe_npz_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "robot" / "human-motions"
    human = root / "AMASS" / "walk.npz"
    robot = root / "exports" / "trajectory.npz"
    keyed_robot = root / "misc" / "unknown.npz"
    opaque_pickle = root / "OMOMO" / "clip" / "clip.pkl"
    ambiguous_pickle = root / "misc" / "trajectory.pkl"
    human.parent.mkdir(parents=True)
    robot.parent.mkdir(parents=True)
    keyed_robot.parent.mkdir(parents=True)
    opaque_pickle.parent.mkdir(parents=True)
    ambiguous_pickle.parent.mkdir(parents=True, exist_ok=True)
    np.savez(human, poses=np.zeros((2, 72)), trans=np.zeros((2, 3)))
    np.savez(robot, joint_q=np.zeros((2, 8)))
    np.savez(keyed_robot, qpos=np.zeros((400_000,), dtype=np.float64))
    opaque_pickle.write_bytes(b"not a pickle and must never be opened")
    (opaque_pickle.parent / "object_cleaned_simplified.obj").write_text(
        "o object\n",
        encoding="utf-8",
    )
    ambiguous_pickle.write_bytes(b"also never opened")
    original_open = Path.open

    def reject_pickle_open(path: Path, *args: object, **kwargs: object):
        if path.suffix.casefold() in {".pkl", ".pickle"}:
            raise AssertionError("catalog classification must not open pickle content")
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", reject_pickle_open)

    assert classify_catalog_robot_trajectory(
        human,
        relative_path=PurePosixPath("AMASS/walk.npz"),
    ) is False
    assert classify_catalog_robot_trajectory(
        robot,
        relative_path=PurePosixPath("exports/trajectory.npz"),
    ) is True
    assert classify_catalog_robot_trajectory(
        keyed_robot,
        relative_path=PurePosixPath("misc/unknown.npz"),
    ) is True
    assert classify_catalog_robot_trajectory(
        opaque_pickle,
        relative_path=PurePosixPath("OMOMO/clip/clip.pkl"),
    ) is False
    assert classify_catalog_robot_trajectory(
        ambiguous_pickle,
        relative_path=PurePosixPath("misc/trajectory.pkl"),
    ) is True
    assert classify_catalog_robot_trajectory(
        opaque_pickle,
        relative_path=PurePosixPath("r2r/human.pkl"),
    ) is True


def test_npz_trajectory_detection_is_inconclusive_for_malformed_or_over_budget_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = tmp_path / "malformed.npz"
    over_budget = tmp_path / "over-budget.npz"
    malformed.write_bytes(b"not a zip archive")
    np.savez(over_budget, poses=np.zeros((2, 72)), trans=np.zeros((2, 3)))

    assert (
        classify_catalog_robot_trajectory(
            malformed,
            relative_path=PurePosixPath("misc/malformed.npz"),
        )
        is None
    )
    monkeypatch.setattr(
        "hhtools.services.available_assets.MAX_CATALOG_NPZ_CENTRAL_DIRECTORY_BYTES",
        1,
    )
    assert (
        classify_catalog_robot_trajectory(
            over_budget,
            relative_path=PurePosixPath("misc/over-budget.npz"),
        )
        is None
    )


def test_npz_trajectory_detection_rejects_zip64_metadata(tmp_path: Path) -> None:
    path = tmp_path / "zip64.npz"
    np.savez(path, poses=np.zeros((2, 72)), trans=np.zeros((2, 3)))
    payload = path.read_bytes()
    eocd = payload.rfind(b"PK\x05\x06")
    assert eocd >= 0
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, 0, 1)
    path.write_bytes(payload[:eocd] + locator + payload[eocd:])

    assert (
        classify_catalog_robot_trajectory(
            path,
            relative_path=PurePosixPath("misc/zip64.npz"),
        )
        is None
    )
    assert not available_assets_module._bounded_npz_directory(path)  # noqa: SLF001
