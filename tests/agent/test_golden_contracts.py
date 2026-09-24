from __future__ import annotations

import json
from pathlib import Path

import pytest

from hhtools.contracts import (
    AssetBundle,
    PreflightResponse,
    RetargetPreflightRequest,
)
from hhtools.contracts.schema_registry import PUBLIC_AGENT_SCHEMAS

FIXTURES = Path(__file__).parent / "fixtures"
SCHEMAS = Path(__file__).parents[2] / "docs" / "schemas" / "agent" / "v1"
PUBLIC_SCHEMAS = PUBLIC_AGENT_SCHEMAS


def test_public_schema_directory_contains_exactly_the_exported_contracts() -> None:
    snapshots = {path.name for path in SCHEMAS.glob("*.schema.json")}
    expected = {f"{name}.schema.json" for name in PUBLIC_SCHEMAS}

    assert snapshots == expected
    assert len(snapshots) == 53


@pytest.mark.parametrize(
    ("filename", "category", "sidecar_role"),
    [
        ("plain_motion.json", "plain_motion", None),
        ("object_interaction.json", "object_interaction", "object_mesh"),
        ("terrain_scene.json", "terrain_scene", "terrain_mesh"),
    ],
)
def test_asset_golden_fixtures_are_portable_and_typed(
    filename: str,
    category: str,
    sidecar_role: str | None,
) -> None:
    payload = json.loads((FIXTURES / "assets" / filename).read_text(encoding="utf-8"))

    bundle = AssetBundle.model_validate(payload)

    assert bundle.category.value == category
    assert bundle.source is not None
    assert bundle.source.root_id == "motion-library"
    assert not bundle.source.logical_path.startswith(("/", "C:/"))
    if sidecar_role is not None:
        assert sidecar_role in {item.role.value for item in bundle.files}


def test_calibrated_robot_preflight_golden_request_and_response_round_trip() -> None:
    payload = json.loads(
        (FIXTURES / "preflight" / "g1_smpl_ready.json").read_text(encoding="utf-8")
    )

    request = RetargetPreflightRequest.model_validate(payload["request"])
    response = PreflightResponse.model_validate(payload["response"])

    assert request.parameters["limit_frames"] == 30
    assert response.plan is not None
    assert response.plan.robot_id == "g1_29dof"
    assert response.plan.calibration_id == request.calibration_id


@pytest.mark.parametrize(("name", "model"), sorted(PUBLIC_SCHEMAS.items()))
def test_public_json_schemas_match_reviewed_snapshots(name: str, model: type) -> None:
    snapshot = json.loads((SCHEMAS / f"{name}.schema.json").read_text(encoding="utf-8"))

    assert snapshot == model.model_json_schema()
