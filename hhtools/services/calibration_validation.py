"""Persistent, content-bound calibration evidence for lightweight preflight."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from hhtools.contracts import CalibrationValidationReport

CALIBRATION_VALIDATION_VERSION = "hhtools.calibration.validation.v1"


def calibration_validation_identity(
    *,
    robot_id: str,
    robot_asset_id: str,
    reference: str,
    calibration_digest: str,
    motion_asset_id: str | None = None,
) -> dict[str, str | None]:
    """Bind manual calibration evidence to the robot, reference, and GLB clip."""

    return {
        "version": CALIBRATION_VALIDATION_VERSION,
        "robot_id": robot_id,
        "robot_asset_id": robot_asset_id,
        "reference": reference,
        "calibration_digest": calibration_digest,
        "motion_asset_id": motion_asset_id if reference == "glb" else None,
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class CalibrationValidationStore:
    """Keep assessments out of robot bundles; missing evidence is never valid."""

    def __init__(self, data_dir: Path) -> None:
        self._root = Path(data_dir) / "calibration-validations"

    def _path(self, identity: dict[str, str | None]) -> Path:
        digest = hashlib.sha256(_canonical(identity).encode()).hexdigest()
        return self._root / f"{digest}.json"

    def put(
        self,
        identity: dict[str, str | None],
        report: CalibrationValidationReport,
    ) -> None:
        payload = {
            "identity": identity,
            "validation": report.model_dump(mode="json", exclude={"candidate_id"}),
        }
        encoded = _canonical(payload)
        document = _canonical(
            {
                "payload": payload,
                "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            }
        )
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._path(identity)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(document, encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def get(self, identity: dict[str, str | None]) -> CalibrationValidationReport | None:
        path = self._path(identity)
        try:
            with path.open("r", encoding="utf-8") as stream:
                encoded = stream.read(256 * 1024 + 1)
            if len(encoded) > 256 * 1024:
                return None
            document = json.loads(encoded)
            payload = document["payload"]
            if (
                _canonical(document) != encoded
                or payload["identity"] != identity
                or document["sha256"] != hashlib.sha256(_canonical(payload).encode()).hexdigest()
            ):
                return None
            return CalibrationValidationReport.model_validate(payload["validation"])
        except (OSError, KeyError, TypeError, ValueError, RecursionError):
            return None


__all__ = [
    "CALIBRATION_VALIDATION_VERSION",
    "CalibrationValidationStore",
    "calibration_validation_identity",
]
