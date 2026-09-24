"""Persistent, validated settings for scheduler and Agent-batch admission.

The renderer is deliberately not the source of truth for these values.  Keeping
them beside the Python service means every browser connected to one backend sees
the same limits.  Saved values survive Web and Electron sidecar restarts unless
an explicit CLI or environment setting overrides them at the next launch.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

JOB_ADMISSION_SETTINGS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class JobAdmissionSettings:
    """Optional scheduler and batch caps; zero means unlimited."""

    max_running_jobs: int = 0
    max_queued_jobs: int = 0
    max_batch_items: int = 0
    max_batch_total_frames: int = 0

    def as_payload(self) -> dict[str, int]:
        return asdict(self)


def _non_negative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def validate_job_admission_settings(
    max_running_jobs: object,
    max_queued_jobs: object,
    max_batch_items: object = 0,
    max_batch_total_frames: object = 0,
) -> JobAdmissionSettings:
    """Return a strict settings value or raise a user-facing ``ValueError``."""

    return JobAdmissionSettings(
        max_running_jobs=_non_negative_integer(
            max_running_jobs,
            name="max_running_jobs",
        ),
        max_queued_jobs=_non_negative_integer(
            max_queued_jobs,
            name="max_queued_jobs",
        ),
        max_batch_items=_non_negative_integer(
            max_batch_items,
            name="max_batch_items",
        ),
        max_batch_total_frames=_non_negative_integer(
            max_batch_total_frames,
            name="max_batch_total_frames",
        ),
    )


def updated_job_admission_settings(
    current: JobAdmissionSettings,
    patch: object,
) -> JobAdmissionSettings:
    """Apply a strict JSON PATCH-shaped mapping to ``current``."""

    if not isinstance(patch, dict):
        raise ValueError("job admission settings must be a JSON object")
    allowed = {
        "max_running_jobs",
        "max_queued_jobs",
        "max_batch_items",
        "max_batch_total_frames",
    }
    unknown = sorted(str(key) for key in patch if key not in allowed)
    if unknown:
        raise ValueError(f"unknown job admission setting: {', '.join(unknown)}")
    if not patch:
        raise ValueError("at least one job admission setting is required")
    return validate_job_admission_settings(
        patch.get("max_running_jobs", current.max_running_jobs),
        patch.get("max_queued_jobs", current.max_queued_jobs),
        patch.get("max_batch_items", current.max_batch_items),
        patch.get("max_batch_total_frames", current.max_batch_total_frames),
    )


class JobAdmissionSettingsStore:
    """Atomically persist one process-wide job-admission configuration."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._lock = threading.RLock()

    def load(self) -> JobAdmissionSettings:
        default = JobAdmissionSettings()
        with self._lock:
            if not self.path.is_file():
                return default
            try:
                payload: Any = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("settings root must be an object")
                version = payload.get("schema_version")
                if version != JOB_ADMISSION_SETTINGS_SCHEMA_VERSION:
                    raise ValueError(f"unsupported settings schema version: {version!r}")
                return validate_job_admission_settings(
                    payload.get("max_running_jobs"),
                    payload.get("max_queued_jobs"),
                    payload.get("max_batch_items", 0),
                    payload.get("max_batch_total_frames", 0),
                )
            except (OSError, ValueError, json.JSONDecodeError):
                _log.warning(
                    "ignoring invalid job admission settings at %s",
                    self.path,
                    exc_info=True,
                )
                return default

    def save(self, settings: JobAdmissionSettings) -> None:
        payload = {
            "schema_version": JOB_ADMISSION_SETTINGS_SCHEMA_VERSION,
            **settings.as_payload(),
        }
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(
                f".{self.path.name}.{time.time_ns()}.tmp",
            )
            try:
                temporary.write_text(encoded + "\n", encoding="utf-8")
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)


__all__ = [
    "JOB_ADMISSION_SETTINGS_SCHEMA_VERSION",
    "JobAdmissionSettings",
    "JobAdmissionSettingsStore",
    "updated_job_admission_settings",
    "validate_job_admission_settings",
]
