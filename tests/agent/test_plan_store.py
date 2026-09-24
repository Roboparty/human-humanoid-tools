from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hhtools.contracts import OutputPolicy, RetargetPlan
from hhtools.services.plans import PlanStore, PlanStoreError, compute_plan_id


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "motion_asset_id": f"asset:sha256:{'1' * 64}",
        "robot_asset_id": f"asset:sha256:{'2' * 64}",
        "robot_id": "g1",
        "backend": "newton",
        "output_format": "csv",
        "parameters": {"solver": {"iterations": 24, "gain": 0.5}},
    }
    payload.update(overrides)
    return payload


def _retarget_payload(*, profile_source: str = "calibration") -> dict[str, object]:
    calibration_id = f"cal:sha256:{'5' * 64}" if profile_source == "calibration" else None
    return {
        "semantics": "hhtools.retarget.plan.v1",
        "motion": {
            "asset_id": f"asset:sha256:{'1' * 64}",
            "digest": "3" * 64,
            "category": "plain_motion",
            "dataset": "amass",
            "reference": "smpl",
        },
        "robot": {
            "asset_id": f"asset:sha256:{'2' * 64}",
            "digest": "4" * 64,
            "robot_id": "g1",
        },
        "backend": "newton",
        "retarget_profile": {
            "source": profile_source,
            "calibration_id": calibration_id,
            "digest": "5" * 64,
            "relative_path": (
                "urdf/retarget_calibration_smpl.yaml"
                if profile_source == "calibration"
                else "config/smpl_scaler.yaml"
            ),
        },
        "output": {"format": "csv", "policy": "create_new"},
        "parameters": {
            "run_mode": "smoke",
            "limit_frames": 30,
            "reference": "smpl",
            "retarget_profile": profile_source,
        },
    }


def _plan(
    canonical_payload: dict[str, object],
    *,
    created_at: datetime | None = None,
    parameters: dict[str, object] | None = None,
) -> RetargetPlan:
    return RetargetPlan(
        plan_id=compute_plan_id(canonical_payload),
        created_at=created_at or datetime(2026, 8, 31, 12, 30, tzinfo=UTC),
        expires_at=(created_at or datetime(2026, 8, 31, 12, 30, tzinfo=UTC)) + timedelta(hours=1),
        motion_asset_id=f"asset:sha256:{'1' * 64}",
        robot_id="g1",
        robot_asset_id=f"asset:sha256:{'2' * 64}",
        backend="newton",
        output_format="csv",
        output_policy=OutputPolicy.CREATE_NEW,
        parameters=parameters or {"solver": {"iterations": 24, "gain": 0.5}},
        input_digest="3" * 64,
        robot_digest="4" * 64,
    )


def _retarget_plan(
    canonical_payload: dict[str, object],
    *,
    profile_source: str = "calibration",
) -> RetargetPlan:
    plan = _plan(
        canonical_payload,
        parameters={
            "run_mode": "smoke",
            "limit_frames": 30,
            "reference": "smpl",
            "retarget_profile": profile_source,
        },
    )
    return plan.model_copy(
        update={
            "calibration_id": (
                f"cal:sha256:{'5' * 64}" if profile_source == "calibration" else None
            ),
            "calibration_digest": "5" * 64 if profile_source == "calibration" else None,
        }
    )


def _assert_code(captured: pytest.ExceptionInfo[PlanStoreError], code: str) -> None:
    assert captured.value.code == code
    assert captured.value.api_error.code == code


def test_compute_plan_id_uses_exact_canonical_json_and_is_deterministic() -> None:
    first = {"z": [3, {"b": 2, "a": "雪"}], "a": True}
    reordered = {"a": True, "z": [3, {"a": "雪", "b": 2}]}
    canonical = json.dumps(
        first,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected = f"plan:sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"

    assert compute_plan_id(first) == expected
    assert compute_plan_id(reordered) == expected


@pytest.mark.parametrize("profile_source", ["calibration", "bundled_scaler"])
def test_retarget_semantics_bind_the_public_plan_projection(
    tmp_path: Path,
    profile_source: str,
) -> None:
    payload = _retarget_payload(profile_source=profile_source)
    plan = _retarget_plan(payload, profile_source=profile_source)
    store = PlanStore(tmp_path / "state")

    inserted = store.put_if_absent(plan, payload)

    assert inserted == plan
    assert store.get(plan.plan_id) == plan
    assert store.get_payload(plan.plan_id) == payload


def test_manual_calibration_id_must_match_its_content_digest(tmp_path: Path) -> None:
    payload = _retarget_payload()
    profile = payload["retarget_profile"]
    assert isinstance(profile, dict)
    profile["calibration_id"] = f"cal:sha256:{'6' * 64}"
    plan = _retarget_plan(payload).model_copy(update={"calibration_id": f"cal:sha256:{'6' * 64}"})

    with pytest.raises(PlanStoreError) as captured:
        PlanStore(tmp_path / "state").put_if_absent(plan, payload)

    _assert_code(captured, "PLAN_CONFLICT")


def test_user_calibration_storage_is_a_valid_content_bound_profile(
    tmp_path: Path,
) -> None:
    payload = _retarget_payload()
    profile = payload["retarget_profile"]
    assert isinstance(profile, dict)
    profile["storage"] = "user_calibration"
    profile["relative_path"] = "g1/retarget_calibration_smpl.yaml"
    plan = _retarget_plan(payload)

    inserted = PlanStore(tmp_path / "state").put_if_absent(plan, payload)

    assert inserted == plan


@pytest.mark.parametrize(
    ("profile_source", "storage"),
    [
        ("calibration", "remote_calibration"),
        ("bundled_scaler", "user_calibration"),
    ],
)
def test_retarget_profile_storage_must_match_the_profile_source(
    tmp_path: Path,
    profile_source: str,
    storage: str,
) -> None:
    payload = _retarget_payload(profile_source=profile_source)
    profile = payload["retarget_profile"]
    assert isinstance(profile, dict)
    profile["storage"] = storage
    plan = _retarget_plan(payload, profile_source=profile_source)

    with pytest.raises(PlanStoreError) as captured:
        PlanStore(tmp_path / "state").put_if_absent(plan, payload)

    _assert_code(captured, "PLAN_CONFLICT")


@pytest.mark.parametrize(
    "relative_path",
    [
        "other_robot/retarget_calibration_smpl.yaml",
        ".calibration-overlays/other_robot/retarget_calibration_smpl.yaml",
        ".calibration-overlays/test_robot/retarget_calibration_smplx.yaml",
        ".calibration-overlays/nested/test_robot/retarget_calibration_smpl.yaml",
    ],
)
def test_user_calibration_path_must_match_plan_robot_and_reference(
    tmp_path: Path,
    relative_path: str,
) -> None:
    payload = _retarget_payload()
    profile = payload["retarget_profile"]
    assert isinstance(profile, dict)
    profile["storage"] = "user_calibration"
    profile["relative_path"] = relative_path
    plan = _retarget_plan(payload)

    with pytest.raises(PlanStoreError) as captured:
        PlanStore(tmp_path / "state").put_if_absent(plan, payload)

    _assert_code(captured, "PLAN_CONFLICT")


@pytest.mark.parametrize(
    "relative_path",
    ["../retarget_calibration.yaml", "C:/robot/scaler.yaml", "config\\scaler.yaml"],
)
def test_retarget_profile_path_must_be_portable_and_bundle_relative(
    tmp_path: Path,
    relative_path: str,
) -> None:
    payload = _retarget_payload()
    profile = payload["retarget_profile"]
    assert isinstance(profile, dict)
    profile["relative_path"] = relative_path

    with pytest.raises(PlanStoreError) as captured:
        plan = _retarget_plan(payload)
        PlanStore(tmp_path / "state").put_if_absent(plan, payload)

    assert captured.value.code in {"INVALID_PARAMETER", "PLAN_CONFLICT"}


@pytest.mark.parametrize(
    ("field", "different_value"),
    [
        ("motion_asset_id", f"asset:sha256:{'a' * 64}"),
        ("robot_id", "different_robot"),
        ("robot_asset_id", f"asset:sha256:{'b' * 64}"),
        ("backend", "interaction_mesh"),
        ("calibration_id", f"cal:sha256:{'c' * 64}"),
        ("output_format", "pkl"),
        ("output_policy", OutputPolicy.FAIL_IF_EXISTS),
        ("parameters", {"run_mode": "full"}),
        ("input_digest", "d" * 64),
        ("robot_digest", "e" * 64),
        ("calibration_digest", "f" * 64),
    ],
)
def test_retarget_semantics_reject_a_divergent_public_plan_before_insert(
    tmp_path: Path,
    field: str,
    different_value: object,
) -> None:
    payload = _retarget_payload()
    plan = _retarget_plan(payload).model_copy(update={field: different_value})
    store = PlanStore(tmp_path / "state")

    with pytest.raises(PlanStoreError) as captured:
        store.put_if_absent(plan, payload)

    _assert_code(captured, "PLAN_CONFLICT")
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM plans").fetchone()[0] == 0


def test_put_and_get_round_trip_isolates_all_nested_mutation(tmp_path: Path) -> None:
    payload = _payload()
    plan = _plan(payload)
    store = PlanStore(tmp_path / "state")

    inserted = store.put_if_absent(plan, payload)
    plan.parameters["solver"]["gain"] = 99  # type: ignore[index]
    payload["parameters"]["solver"]["gain"] = 88  # type: ignore[index]
    inserted.parameters["solver"]["gain"] = 77  # type: ignore[index]

    restored = store.get(inserted.plan_id)
    restored_payload = store.get_payload(inserted.plan_id)
    assert restored.parameters == {"solver": {"iterations": 24, "gain": 0.5}}
    assert restored_payload["parameters"] == {"solver": {"iterations": 24, "gain": 0.5}}

    restored.parameters["solver"]["gain"] = 66  # type: ignore[index]
    restored_payload["parameters"]["solver"]["gain"] = 55  # type: ignore[index]
    assert store.get(inserted.plan_id).parameters["solver"]["gain"] == 0.5  # type: ignore[index]
    assert store.get_payload(inserted.plan_id)["parameters"] == {
        "solver": {"iterations": 24, "gain": 0.5}
    }


def test_store_persists_across_instances_and_enables_wal(tmp_path: Path) -> None:
    data_dir = tmp_path / "agent-state"
    payload = _payload()
    plan = _plan(payload)

    first = PlanStore(data_dir)
    inserted = first.put_if_absent(plan, payload)
    second = PlanStore(data_dir)

    assert first.database_path == data_dir / "plans.sqlite3"
    assert second.get(inserted.plan_id) == inserted
    assert second.get_payload(inserted.plan_id) == payload
    with sqlite3.connect(first.database_path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_concurrent_identical_puts_preserve_the_first_plan(tmp_path: Path) -> None:
    store = PlanStore(tmp_path / "state")
    payload = _payload()
    plan = _plan(payload)

    def insert(_: int) -> RetargetPlan:
        return store.put_if_absent(plan, payload)

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(insert, range(48)))

    assert all(result == plan for result in results)
    assert {result.created_at for result in results} == {plan.created_at}
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM plans").fetchone()[0] == 1


def test_same_plan_id_cannot_be_reused_for_a_different_plan(tmp_path: Path) -> None:
    store = PlanStore(tmp_path / "state")
    payload = _payload()
    first = _plan(payload)
    store.put_if_absent(first, payload)
    different = _plan(payload, created_at=first.created_at + timedelta(minutes=1))

    with pytest.raises(PlanStoreError) as captured:
        store.put_if_absent(different, payload)

    _assert_code(captured, "PLAN_CONFLICT")
    assert store.get(first.plan_id) == first


def test_same_plan_id_cannot_be_reused_for_a_different_payload(tmp_path: Path) -> None:
    store = PlanStore(tmp_path / "state")
    payload = _payload()
    plan = _plan(payload)
    store.put_if_absent(plan, payload)
    changed_payload = _payload(backend="interaction_mesh")

    with pytest.raises(PlanStoreError) as captured:
        store.put_if_absent(plan, changed_payload)

    _assert_code(captured, "PLAN_CONFLICT")
    assert store.get_payload(plan.plan_id) == payload


def test_missing_and_corrupt_rows_have_stable_structured_errors(tmp_path: Path) -> None:
    store = PlanStore(tmp_path / "state")
    payload = _payload()
    plan = _plan(payload)

    with pytest.raises(PlanStoreError) as missing:
        store.get(f"plan:sha256:{'0' * 64}")
    _assert_code(missing, "PLAN_NOT_FOUND")

    store.put_if_absent(plan, payload)
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE plans SET plan_json = ? WHERE plan_id = ?",
            ("{not-json", plan.plan_id),
        )

    with pytest.raises(PlanStoreError) as corrupt:
        store.get(plan.plan_id)
    _assert_code(corrupt, "INTERNAL_ERROR")
    assert corrupt.value.error.stage.value == "internal"


def test_existing_retarget_row_with_divergent_public_projection_is_corrupt(
    tmp_path: Path,
) -> None:
    store = PlanStore(tmp_path / "state")
    payload = _retarget_payload()
    plan = _retarget_plan(payload)
    store.put_if_absent(plan, payload)

    # Keep both JSON documents canonical and keep the payload hash/row id
    # intact.  Only the separately persisted public projection is changed;
    # this is the corruption that a preflight cache hit must not return.
    plan_document = plan.model_dump(mode="json")
    plan_document["backend"] = "interaction_mesh"
    tampered_plan_json = json.dumps(
        plan_document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE plans SET plan_json = ? WHERE plan_id = ?",
            (tampered_plan_json, plan.plan_id),
        )

    with pytest.raises(PlanStoreError) as cache_hit:
        store.get(plan.plan_id)
    _assert_code(cache_hit, "INTERNAL_ERROR")
    assert cache_hit.value.error.stage.value == "internal"

    with pytest.raises(PlanStoreError) as repeated_preflight:
        store.put_if_absent(plan, payload)
    _assert_code(repeated_preflight, "INTERNAL_ERROR")


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "/srv/private/motion.npz",
        r"C:\private\motion.npz",
        r"\\server\share\motion.npz",
        float("nan"),
        float("inf"),
        ("tuple", "is-not-json"),
        Path("relative-but-not-json"),
    ],
)
def test_rejects_host_paths_nonfinite_numbers_and_non_json_payloads(
    tmp_path: Path,
    unsafe_value: object,
) -> None:
    store = PlanStore(tmp_path / "state")
    payload = _payload(unsafe=unsafe_value)
    safe_plan = _plan(_payload())

    with pytest.raises(PlanStoreError) as captured:
        store.put_if_absent(safe_plan, payload)

    _assert_code(captured, "INVALID_PARAMETER")
    for database_file in (tmp_path / "state").glob("plans.sqlite3*"):
        database_bytes = database_file.read_bytes()
        assert b"/srv/private/motion.npz" not in database_bytes
        assert b"C:\\private\\motion.npz" not in database_bytes


def test_rejects_absolute_paths_nested_in_the_plan_document(tmp_path: Path) -> None:
    store = PlanStore(tmp_path / "state")
    payload = _payload()
    plan = _plan(payload, parameters={"solver": {"cache": "/private/cache"}})

    with pytest.raises(PlanStoreError) as captured:
        store.put_if_absent(plan, payload)

    _assert_code(captured, "INVALID_PARAMETER")
    for database_file in (tmp_path / "state").glob("plans.sqlite3*"):
        assert b"/private/cache" not in database_file.read_bytes()
