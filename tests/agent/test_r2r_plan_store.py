from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from hhtools.contracts import R2RPlan
from hhtools.services.plans import PlanStore, PlanStoreError, compute_plan_id
from hhtools.services.r2r_preflight import R2R_PLAN_SEMANTICS


def _payload() -> dict[str, object]:
    return {
        "semantics": R2R_PLAN_SEMANTICS,
        "trajectory": {
            "asset_id": f"asset:sha256:{'1' * 64}",
            "digest": "1" * 64,
            "category": "robot_trajectory",
            "source_robot_id": "source_bot",
            "declared_source_robot_id": "source_bot",
            "profile": "mimic",
        },
        "source_robot": {
            "asset_id": f"asset:sha256:{'2' * 64}",
            "digest": "2" * 64,
            "robot_id": "source_bot",
        },
        "target_robot": {
            "asset_id": f"asset:sha256:{'3' * 64}",
            "digest": "3" * 64,
            "robot_id": "target_bot",
        },
        "backend": "newton",
        "pair_calibration": {
            "source_robot_id": "source_bot",
            "target_robot_id": "target_bot",
            "calibration_id": f"cal:sha256:{'4' * 64}",
            "digest": "4" * 64,
            "storage": "robot_bundle",
            "relative_path": "r2r_calibration_source_bot.yaml",
        },
        "output": {"format": "csv", "policy": "create_new"},
        "parameters": {
            "run_mode": "smoke",
            "limit_frames": 1,
            "ik_iterations": 24,
            "source_fps": None,
            "retarget_fps": None,
            "trajectory_profile": "mimic",
        },
    }


def _plan(payload: dict[str, object]) -> R2RPlan:
    return R2RPlan(
        plan_id=compute_plan_id(payload),
        created_at=datetime(2026, 9, 8, tzinfo=UTC),
        trajectory_asset_id=f"asset:sha256:{'1' * 64}",
        source_robot_id="source_bot",
        source_robot_asset_id=f"asset:sha256:{'2' * 64}",
        target_robot_id="target_bot",
        target_robot_asset_id=f"asset:sha256:{'3' * 64}",
        backend="newton",
        calibration_id=f"cal:sha256:{'4' * 64}",
        output_format="csv",
        output_policy="create_new",
        parameters=payload["parameters"],  # type: ignore[arg-type]
        trajectory_digest="1" * 64,
        source_robot_digest="2" * 64,
        target_robot_digest="3" * 64,
        calibration_digest="4" * 64,
    )


def test_r2r_plan_round_trip_binds_all_four_content_identities(tmp_path: Path) -> None:
    payload = _payload()
    plan = _plan(payload)
    store = PlanStore(tmp_path / "state")

    inserted = store.put_if_absent(plan, payload)

    assert inserted == plan
    assert store.get(plan.plan_id) == plan
    assert store.get_payload(plan.plan_id) == payload


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trajectory_asset_id", f"asset:sha256:{'a' * 64}"),
        ("source_robot_id", "other_source"),
        ("source_robot_asset_id", f"asset:sha256:{'b' * 64}"),
        ("target_robot_id", "other_target"),
        ("target_robot_asset_id", f"asset:sha256:{'c' * 64}"),
        ("backend", "interaction_mesh"),
        ("calibration_id", f"cal:sha256:{'d' * 64}"),
        ("output_format", "pkl"),
        ("trajectory_digest", "a" * 64),
        ("source_robot_digest", "b" * 64),
        ("target_robot_digest", "c" * 64),
        ("calibration_digest", "d" * 64),
    ],
)
def test_r2r_semantics_reject_divergent_public_projection(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = _payload()
    plan = _plan(payload).model_copy(update={field: value})

    with pytest.raises(PlanStoreError) as captured:
        PlanStore(tmp_path / "state").put_if_absent(plan, payload)

    assert captured.value.code == "PLAN_CONFLICT"


def test_r2r_pair_calibration_id_must_match_digest(tmp_path: Path) -> None:
    payload = _payload()
    calibration = payload["pair_calibration"]
    assert isinstance(calibration, dict)
    calibration["calibration_id"] = f"cal:sha256:{'5' * 64}"

    with pytest.raises(PlanStoreError) as captured:
        PlanStore(tmp_path / "state").put_if_absent(_plan(payload), payload)

    assert captured.value.code == "PLAN_CONFLICT"


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("trajectory", "source_robot_id", "another_source"),
        ("trajectory", "profile", "intermimic"),
        (None, "backend", "interaction_mesh"),
    ],
)
def test_r2r_v1_semantics_reject_unsupported_or_inconsistent_routing(
    tmp_path: Path,
    section: str | None,
    field: str,
    value: str,
) -> None:
    payload = _payload()
    target = payload if section is None else payload[section]
    assert isinstance(target, dict)
    target[field] = value

    with pytest.raises(PlanStoreError) as captured:
        PlanStore(tmp_path / "state").put_if_absent(_plan(payload), payload)

    assert captured.value.code == "PLAN_CONFLICT"
