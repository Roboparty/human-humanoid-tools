from __future__ import annotations

from pathlib import Path

import pytest

from hhtools.web.jobs.job_history import JobHistoryStore


def test_store_persists_records_and_adopts_artifacts(tmp_path: Path) -> None:
    store = JobHistoryStore(tmp_path / "history", max_records=4)
    generated = tmp_path / "runtime" / "result.zip"
    generated.parent.mkdir()
    generated.write_bytes(b"artifact")

    adopted = store.adopt_artifact("job-one", generated, download_name="motion.zip")
    store.put(
        {
            "id": "job-one",
            "kind": "batch",
            "status": "done",
            "created_at": 10.0,
            "finished_at": 12.0,
            "artifact_path": str(adopted),
        }
    )

    reopened = JobHistoryStore(tmp_path / "history", max_records=4)
    record = reopened.get("job-one")

    assert generated.exists() is False
    assert record is not None
    assert reopened.artifact_path(record) == adopted
    assert adopted.read_bytes() == b"artifact"


@pytest.mark.parametrize("active_status", ["pending", "running"])
def test_store_marks_interrupted_jobs_as_errors(
    tmp_path: Path,
    active_status: str,
) -> None:
    root = tmp_path / "history"
    store = JobHistoryStore(root, max_records=4)
    store.put(
        {
            "id": "interrupted",
            "kind": "retarget",
            "status": active_status,
            "created_at": 10.0,
            "finished_at": None,
        }
    )

    recovered = JobHistoryStore(root, max_records=4).get("interrupted")

    assert recovered is not None
    assert recovered["status"] == "error"
    assert "重启" in recovered["error"]


def test_store_retention_never_prunes_accepted_active_jobs(tmp_path: Path) -> None:
    store = JobHistoryStore(tmp_path / "history", max_records=1)
    for job_id, status, created_at in (
        ("pending-job", "pending", 1.0),
        ("running-job", "running", 2.0),
        ("latest-terminal", "done", 4.0),
        ("old-terminal", "error", 3.0),
    ):
        store.put(
            {
                "id": job_id,
                "kind": "test",
                "status": status,
                "created_at": created_at,
            }
        )

    records = {record["id"]: record for record in store.list_records()}

    assert set(records) == {"pending-job", "running-job", "latest-terminal"}


def test_store_retention_uses_completion_time_for_long_running_job(tmp_path: Path) -> None:
    store = JobHistoryStore(tmp_path / "history", max_records=1)
    store.put(
        {
            "id": "long-running",
            "kind": "batch",
            "status": "running",
            "created_at": 1.0,
        }
    )
    store.put(
        {
            "id": "quick-job",
            "kind": "batch",
            "status": "done",
            "created_at": 10.0,
            "finished_at": 20.0,
        }
    )
    generated = tmp_path / "runtime" / "long-running.zip"
    generated.parent.mkdir()
    generated.write_bytes(b"artifact")
    adopted = store.adopt_artifact("long-running", generated)

    store.put(
        {
            "id": "long-running",
            "kind": "batch",
            "status": "done",
            "created_at": 1.0,
            "finished_at": 30.0,
            "artifact_path": str(adopted),
        }
    )

    assert store.get("quick-job") is None
    assert store.get("long-running") is not None
    assert adopted.read_bytes() == b"artifact"
