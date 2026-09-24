"""Immutable local storage for content-addressed calibration candidates."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from hhtools.contracts import CalibrationCandidate, R2RCalibrationCandidate
from hhtools.services.calibration_validation import CalibrationValidationStore


class CalibrationCandidateStoreError(RuntimeError):
    """A candidate is missing, conflicting, or corrupt."""


def _canonical_payload(payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    try:
        document = json.loads(
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        encoded = json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (OverflowError, TypeError, ValueError) as error:
        raise CalibrationCandidateStoreError("candidate payload is not canonical JSON") from error
    return encoded, document


def compute_calibration_candidate_id(payload: Mapping[str, Any]) -> str:
    encoded, _document = _canonical_payload(payload)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"cal-candidate:sha256:{digest}"


def calibration_candidate_payload(
    candidate: CalibrationCandidate | R2RCalibrationCandidate,
) -> dict[str, Any]:
    return candidate.model_dump(mode="json", exclude={"candidate_id"})


class CalibrationCandidateStore:
    """Persist candidates across MCP reconnects without exposing host paths."""

    def __init__(self, data_dir: Path) -> None:
        self.validation_store = CalibrationValidationStore(data_dir)
        self._root = Path(data_dir) / "calibration-candidates"
        self._history_root = Path(data_dir) / "calibration-history"
        self._lock = threading.RLock()
        self._root.mkdir(parents=True, exist_ok=True)
        self._history_root.mkdir(parents=True, exist_ok=True)

    def _path(self, candidate_id: str) -> Path:
        prefix = "cal-candidate:sha256:"
        if not candidate_id.startswith(prefix):
            raise CalibrationCandidateStoreError("candidate id is invalid")
        digest = candidate_id.removeprefix(prefix)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise CalibrationCandidateStoreError("candidate id is invalid")
        return self._root / f"{digest}.json"

    def _put(
        self,
        candidate: CalibrationCandidate | R2RCalibrationCandidate,
    ) -> CalibrationCandidate | R2RCalibrationCandidate:
        payload = calibration_candidate_payload(candidate)
        expected = compute_calibration_candidate_id(payload)
        if candidate.candidate_id != expected:
            raise CalibrationCandidateStoreError("candidate content identity is invalid")
        encoded, _document = _canonical_payload(candidate.model_dump(mode="json"))
        path = self._path(candidate.candidate_id)
        with self._lock:
            if path.exists():
                try:
                    existing = path.read_text(encoding="utf-8")
                except OSError as error:
                    raise CalibrationCandidateStoreError("candidate could not be read") from error
                if existing != encoded:
                    raise CalibrationCandidateStoreError(
                        "candidate identity conflicts with storage"
                    )
                if isinstance(candidate, R2RCalibrationCandidate):
                    return self.get_r2r(candidate.candidate_id)
                return self.get(candidate.candidate_id)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_text(encoded, encoding="utf-8")
                temporary.replace(path)
            except OSError as error:
                raise CalibrationCandidateStoreError("candidate could not be persisted") from error
            finally:
                temporary.unlink(missing_ok=True)
        return candidate

    def put(self, candidate: CalibrationCandidate) -> CalibrationCandidate:
        stored = self._put(candidate)
        if not isinstance(stored, CalibrationCandidate):  # pragma: no cover - typing guard
            raise CalibrationCandidateStoreError("stored candidate has the wrong workflow")
        return stored

    def put_r2r(self, candidate: R2RCalibrationCandidate) -> R2RCalibrationCandidate:
        stored = self._put(candidate)
        if not isinstance(stored, R2RCalibrationCandidate):  # pragma: no cover - typing guard
            raise CalibrationCandidateStoreError("stored candidate has the wrong workflow")
        return stored

    def _get(
        self,
        candidate_id: str,
        model: type[CalibrationCandidate] | type[R2RCalibrationCandidate],
    ) -> CalibrationCandidate | R2RCalibrationCandidate:
        path = self._path(candidate_id)
        with self._lock:
            try:
                encoded = path.read_text(encoding="utf-8")
            except FileNotFoundError as error:
                raise CalibrationCandidateStoreError("candidate was not found") from error
            except OSError as error:
                raise CalibrationCandidateStoreError("candidate could not be read") from error
        try:
            document = json.loads(encoded)
            candidate = model.model_validate(document)
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as error:
            raise CalibrationCandidateStoreError("stored candidate is invalid") from error
        canonical, _document = _canonical_payload(candidate.model_dump(mode="json"))
        expected = compute_calibration_candidate_id(calibration_candidate_payload(candidate))
        if (
            encoded != canonical
            or candidate.candidate_id != candidate_id
            or expected != candidate_id
        ):
            raise CalibrationCandidateStoreError("stored candidate identity is invalid")
        return candidate

    def get(self, candidate_id: str) -> CalibrationCandidate:
        candidate = self._get(candidate_id, CalibrationCandidate)
        if not isinstance(candidate, CalibrationCandidate):  # pragma: no cover - typing guard
            raise CalibrationCandidateStoreError("stored candidate has the wrong workflow")
        return candidate

    def get_r2r(self, candidate_id: str) -> R2RCalibrationCandidate:
        candidate = self._get(candidate_id, R2RCalibrationCandidate)
        if not isinstance(candidate, R2RCalibrationCandidate):  # pragma: no cover - typing guard
            raise CalibrationCandidateStoreError("stored candidate has the wrong workflow")
        return candidate

    def archive_calibration(self, calibration_id: str, payload: bytes) -> None:
        """Retain exact previous bytes before a validated silent replacement."""

        prefix = "cal:sha256:"
        digest = calibration_id.removeprefix(prefix)
        if (
            not calibration_id.startswith(prefix)
            or len(digest) != 64
            or hashlib.sha256(payload).hexdigest() != digest
        ):
            raise CalibrationCandidateStoreError("calibration archive identity is invalid")
        path = self._history_root / f"{digest}.yaml"
        with self._lock:
            if path.exists():
                try:
                    existing = path.read_bytes()
                except OSError as error:
                    raise CalibrationCandidateStoreError(
                        "calibration archive could not be read"
                    ) from error
                if existing != payload:
                    raise CalibrationCandidateStoreError(
                        "calibration archive identity conflicts with storage"
                    )
                return
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_bytes(payload)
                temporary.replace(path)
            except OSError as error:
                raise CalibrationCandidateStoreError(
                    "calibration archive could not be persisted"
                ) from error
            finally:
                temporary.unlink(missing_ok=True)

    def has_calibration_archive(self, calibration_id: str) -> bool:
        prefix = "cal:sha256:"
        digest = calibration_id.removeprefix(prefix)
        if not calibration_id.startswith(prefix) or len(digest) != 64:
            return False
        path = self._history_root / f"{digest}.yaml"
        try:
            return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
        except OSError:
            return False


__all__ = [
    "CalibrationCandidateStore",
    "CalibrationCandidateStoreError",
    "calibration_candidate_payload",
    "compute_calibration_candidate_id",
]
