"""Small disk-backed store for reproducible Web job records.

The live job owns large preview payloads in memory. This store only retains compact JSON
metadata, the effective request, and browser-downloadable artifacts. Keeping that boundary
explicit prevents a long Motion or robot trajectory from being serialized into history.
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

JOB_HISTORY_SCHEMA_VERSION = 1
_ACTIVE_STATUSES = frozenset({"pending", "running"})


class JobHistoryStore:
    """Persist compact job records with atomic, one-file-per-job updates."""

    def __init__(self, root: Path, *, max_records: int) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self.root = Path(root).expanduser().resolve()
        self.records_dir = self.root / "records"
        self.artifacts_dir = self.root / "artifacts"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.max_records = int(max_records)
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._load_records()
        self._recover_interrupted_records()
        with self._lock:
            self._prune_locked()

    def list_records(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return newest-first detached records safe for API serialization."""
        with self._lock:
            records = sorted(
                self._records.values(),
                key=lambda record: float(record.get("created_at") or 0.0),
                reverse=True,
            )
            if limit is not None:
                records = records[: max(0, int(limit))]
            return copy.deepcopy(records)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(job_id)
            return copy.deepcopy(record) if record is not None else None

    def put(self, record: dict[str, Any]) -> None:
        """Atomically insert or replace one JSON-safe record."""
        job_id = _validated_job_id(record.get("id"))
        payload = copy.deepcopy(record)
        payload["id"] = job_id
        payload["schema_version"] = JOB_HISTORY_SCHEMA_VERSION
        # Serializing before opening the temp file prevents partial writes when a
        # future caller accidentally supplies a non-JSON value.
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        with self._lock:
            destination = self.records_dir / f"{job_id}.json"
            temporary = self.records_dir / f".{job_id}.{time.time_ns()}.tmp"
            try:
                temporary.write_text(encoded + "\n", encoding="utf-8")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            self._records[job_id] = payload
            self._prune_locked()

    def adopt_artifact(
        self,
        job_id: str,
        source: Path,
        *,
        download_name: str | None = None,
    ) -> Path:
        """Move a generated artifact into the persistent history directory."""
        safe_job_id = _validated_job_id(job_id)
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        filename = Path(download_name or source_path.name).name
        if filename in {"", ".", ".."}:
            filename = source_path.name
        destination_dir = (self.artifacts_dir / safe_job_id).resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = (destination_dir / filename).resolve()
        destination.relative_to(destination_dir)
        if source_path == destination:
            return destination
        destination.unlink(missing_ok=True)
        shutil.move(str(source_path), str(destination))
        return destination

    def artifact_path(self, record: dict[str, Any]) -> Path | None:
        """Return a managed artifact only when it still exists below this store."""
        raw = record.get("artifact_path")
        if not isinstance(raw, str) or not raw:
            return None
        try:
            path = Path(raw).resolve()
            path.relative_to(self.artifacts_dir)
        except (OSError, ValueError):
            return None
        return path if path.is_file() else None

    def _load_records(self) -> None:
        with self._lock:
            for path in self.records_dir.glob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    job_id = _validated_job_id(payload.get("id"))
                    if int(payload.get("schema_version") or 0) != JOB_HISTORY_SCHEMA_VERSION:
                        continue
                    self._records[job_id] = payload
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    _log.warning("ignoring malformed job-history record %s", path)

    def _recover_interrupted_records(self) -> None:
        """Convert jobs left active by a prior process into actionable failures."""
        now = time.time()
        with self._lock:
            interrupted = [
                record
                for record in self._records.values()
                if record.get("status") in _ACTIVE_STATUSES
            ]
        for record in interrupted:
            recovered = copy.deepcopy(record)
            recovered["status"] = "error"
            recovered["message"] = "服务在任务完成前关闭"
            recovered["error"] = "任务因 Web 服务重启或意外退出而中断，请重新运行。"
            recovered["finished_at"] = now
            recovered["duration_seconds"] = max(
                0.0, now - float(recovered.get("created_at") or now)
            )
            recovered["can_download"] = False
            self.put(recovered)

    def _prune_locked(self) -> None:
        # Active jobs are accepted work, not retained history.  Keeping every
        # pending/running record lets a restart recover it as interrupted even
        # when an unlimited queue temporarily exceeds ``max_records``.
        terminal_records = sorted(
            (
                record
                for record in self._records.values()
                if record.get("status") not in _ACTIVE_STATUSES
            ),
            key=lambda record: float(
                record.get("finished_at")
                if record.get("finished_at") is not None
                else record.get("created_at") or 0.0
            ),
            reverse=True,
        )
        for record in terminal_records[self.max_records :]:
            job_id = str(record.get("id") or "")
            self._records.pop(job_id, None)
            (self.records_dir / f"{job_id}.json").unlink(missing_ok=True)
            shutil.rmtree(self.artifacts_dir / job_id, ignore_errors=True)


def _validated_job_id(raw: Any) -> str:
    value = str(raw or "")
    if not value or len(value) > 64 or not all(ch.isalnum() or ch in "-_" for ch in value):
        raise ValueError("invalid job id")
    return value


__all__ = ["JOB_HISTORY_SCHEMA_VERSION", "JobHistoryStore"]
