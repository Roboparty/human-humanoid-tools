from __future__ import annotations

import json
from pathlib import Path

from hhtools.web.jobs.job_settings import (
    JobAdmissionSettings,
    JobAdmissionSettingsStore,
    updated_job_admission_settings,
)
from hhtools.web.server import settings as server_settings


def test_settings_store_round_trips_atomically(tmp_path: Path) -> None:
    path = tmp_path / "config" / "web-settings.json"
    store = JobAdmissionSettingsStore(path)

    store.save(JobAdmissionSettings(max_running_jobs=2, max_queued_jobs=32))

    assert store.load() == JobAdmissionSettings(max_running_jobs=2, max_queued_jobs=32)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "max_batch_items": 0,
        "max_batch_total_frames": 0,
        "max_running_jobs": 2,
        "max_queued_jobs": 32,
    }
    assert not list(path.parent.glob("*.tmp"))


def test_settings_store_uses_defaults_for_invalid_content(tmp_path: Path) -> None:
    path = tmp_path / "web-settings.json"
    path.write_text('{"schema_version": 1, "max_running_jobs": -1}', encoding="utf-8")

    assert JobAdmissionSettingsStore(path).load() == JobAdmissionSettings()


def test_settings_store_migrates_existing_v1_scheduler_settings(tmp_path: Path) -> None:
    path = tmp_path / "web-settings.json"
    path.write_text(
        '{"schema_version": 1, "max_running_jobs": 2, "max_queued_jobs": 32}',
        encoding="utf-8",
    )

    assert JobAdmissionSettingsStore(path).load() == JobAdmissionSettings(
        max_running_jobs=2,
        max_queued_jobs=32,
        max_batch_items=0,
        max_batch_total_frames=0,
    )


def test_settings_patch_is_partial_but_strict() -> None:
    current = JobAdmissionSettings(max_running_jobs=1, max_queued_jobs=16)

    assert updated_job_admission_settings(
        current,
        {"max_running_jobs": 2},
    ) == JobAdmissionSettings(max_running_jobs=2, max_queued_jobs=16)

    assert updated_job_admission_settings(
        current,
        {"max_batch_items": 500, "max_batch_total_frames": 0},
    ) == JobAdmissionSettings(
        max_running_jobs=1,
        max_queued_jobs=16,
        max_batch_items=500,
        max_batch_total_frames=0,
    )


def test_effective_settings_restore_saved_values_and_keep_explicit_overrides(
    tmp_path: Path,
) -> None:
    path = tmp_path / "web-settings.json"
    JobAdmissionSettingsStore(path).save(
        JobAdmissionSettings(
            max_running_jobs=2,
            max_queued_jobs=32,
            max_batch_items=500,
            max_batch_total_frames=2_000_000,
        ),
    )

    restored, restored_path = server_settings.effective_job_admission_settings(
        max_running_jobs=None,
        max_queued_jobs=None,
        job_settings_path=path,
    )
    overridden, _ = server_settings.effective_job_admission_settings(
        max_running_jobs=8,
        max_queued_jobs=None,
        max_batch_items=0,
        max_batch_total_frames=None,
        job_settings_path=path,
    )

    assert restored == JobAdmissionSettings(
        max_running_jobs=2,
        max_queued_jobs=32,
        max_batch_items=500,
        max_batch_total_frames=2_000_000,
    )
    assert restored_path == path
    assert overridden == JobAdmissionSettings(
        max_running_jobs=8,
        max_queued_jobs=32,
        max_batch_items=0,
        max_batch_total_frames=2_000_000,
    )
