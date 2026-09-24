"""Resource-limit defaults and persisted job/batch admission settings."""

from __future__ import annotations

import logging
from pathlib import Path

from hhtools.services.job_settings import (
    JobAdmissionSettings,
    JobAdmissionSettingsStore,
    validate_job_admission_settings,
)

_log = logging.getLogger(__name__)

# Upload/retention bounds are active by default; job admission is deliberately
# unlimited unless an expert deployment opts into a concurrency cap and queue.
_DEFAULT_MAX_UPLOAD_FILES = 4096
_DEFAULT_MAX_UPLOAD_FILE_BYTES = 2 * 1024**3
_DEFAULT_MAX_UPLOAD_REQUEST_BYTES = 8 * 1024**3
_DEFAULT_MAX_RUNNING_JOBS = 0
_DEFAULT_MAX_QUEUED_JOBS = 0
_DEFAULT_MAX_RETAINED_JOBS = 64
_DEFAULT_JOB_TTL_SECONDS = 60 * 60.0

# Injected into the served shell so browser assets and backend stay in sync.
UI_BUILD_ID = "20260902-react-workbench"


def effective_job_admission_settings(
    *,
    max_running_jobs: int | None,
    max_queued_jobs: int | None,
    job_settings_path: Path | None,
    max_batch_items: int | None = None,
    max_batch_total_frames: int | None = None,
) -> tuple[JobAdmissionSettings, Path]:
    """Merge persistent settings with explicit CLI/environment overrides."""

    from hhtools.utils.paths import user_web_settings_path

    path = Path(job_settings_path or user_web_settings_path())
    persisted = JobAdmissionSettingsStore(path).load()
    settings = validate_job_admission_settings(
        persisted.max_running_jobs if max_running_jobs is None else max_running_jobs,
        persisted.max_queued_jobs if max_queued_jobs is None else max_queued_jobs,
        persisted.max_batch_items if max_batch_items is None else max_batch_items,
        (
            persisted.max_batch_total_frames
            if max_batch_total_frames is None
            else max_batch_total_frames
        ),
    )
    return settings, path
