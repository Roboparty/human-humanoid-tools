from __future__ import annotations

import pytest

from hhtools.contracts import AssetCategory
from hhtools.services.routing import (
    backend_for_category,
    category_for_dataset,
    reference_for_dataset,
)


@pytest.mark.parametrize(
    ("dataset", "reference"),
    [
        ("amass", "smpl"),
        ("phuma", "smpl"),
        ("unified_npz", "smpl"),
        ("parc_ms", "smpl"),
        ("motion_x", "smplx"),
        ("omomo", "smplx"),
        ("meshmimic_holosoma", "smplx"),
        ("lafan", "lafan_bvh"),
        ("omnicontact", "lafan_bvh"),
        ("mocap", "mocap_bvh"),
        ("soma", "soma_bvh"),
        ("xsens_mocap", "xsens_mocap"),
        ("gvhmr", "gvhmr"),
        ("kungfu_athlete", "gvhmr"),
        ("glb", "glb"),
    ],
)
def test_dataset_reference_routing_is_centralized(dataset: str, reference: str) -> None:
    assert reference_for_dataset(dataset, ".npz") == reference


@pytest.mark.parametrize(
    ("dataset", "category", "backend"),
    [
        ("amass", AssetCategory.PLAIN_MOTION, "newton"),
        ("omomo", AssetCategory.OBJECT_INTERACTION, "interaction_mesh"),
        ("parc_ms", AssetCategory.TERRAIN_SCENE, "interaction_mesh"),
    ],
)
def test_dataset_category_selects_only_declared_compatible_backend(
    dataset: str,
    category: AssetCategory,
    backend: str,
) -> None:
    detected = category_for_dataset(dataset)
    assert detected is category
    assert backend_for_category(detected) == backend


def test_non_motion_category_has_no_silent_backend_fallback() -> None:
    with pytest.raises(ValueError, match="not retargetable motion"):
        backend_for_category(AssetCategory.ROBOT_MODEL)
