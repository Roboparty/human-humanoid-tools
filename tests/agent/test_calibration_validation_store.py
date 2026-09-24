from pathlib import Path

import pytest

from hhtools.contracts import CalibrationValidationReport
from hhtools.services.calibration_validation import (
    CalibrationValidationStore,
    calibration_validation_identity,
)


def _identity(**changes):
    values = dict(
        robot_id="test_robot",
        robot_asset_id=f"asset:sha256:{'a' * 64}",
        reference="glb",
        calibration_digest="b" * 64,
        motion_asset_id=f"asset:sha256:{'c' * 64}",
    )
    values.update(changes)
    return calibration_validation_identity(**values)


@pytest.mark.parametrize("valid", [False, True])
def test_validation_survives_restart_without_changing_its_verdict(tmp_path: Path, valid: bool):
    report = CalibrationValidationReport(
        valid=valid,
        score=1 if valid else 0,
        changed_joint_count=0,
        mapped_slots=16,
        checks=[
            dict(
                code="CALIBRATION_POSE_ALIGNED",
                level="pass" if valid else "error",
                message="Measured limb directions.",
            )
        ],
    )
    store = CalibrationValidationStore(tmp_path)
    store.put(_identity(), report)
    assert CalibrationValidationStore(tmp_path).get(_identity()) == report


@pytest.mark.parametrize(
    "changes",
    [
        {"robot_id": "different_robot"},
        {"robot_asset_id": f"asset:sha256:{'d' * 64}"},
        {"reference": "smpl"},
        {"calibration_digest": "d" * 64},
        {"motion_asset_id": f"asset:sha256:{'d' * 64}"},
    ],
)
def test_validation_cannot_be_reused_for_another_input(tmp_path: Path, changes):
    store = CalibrationValidationStore(tmp_path)
    report = CalibrationValidationReport(
        valid=True,
        score=1,
        changed_joint_count=0,
        mapped_slots=16,
        checks=[dict(code="CALIBRATION_POSE_ALIGNED", level="pass", message="Aligned.")],
    )
    store.put(_identity(), report)
    assert store.get(_identity(**changes)) is None
    assert store.get({**_identity(), "version": "future-validator"}) is None
    path = store._path(_identity())
    path.write_text(path.read_text().replace('"valid":true', '"valid":false'))
    assert store.get(_identity()) is None
