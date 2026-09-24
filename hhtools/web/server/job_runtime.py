"""Legacy Web job scheduling, retention, persistence, and presentation."""

from __future__ import annotations

import logging
import shlex
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hhtools.contracts.legacy_jobs import build_job_spec, replay_capability
from hhtools.services.job_scheduler import (
    JobQueueFullError,
    JobReservation,
    JobScheduler,
    JobSchedulerClosedError,
)

from .boundary import _is_loopback_address
from .state import Job, SessionState, _snapshot_job_request

if TYPE_CHECKING:
    from fastapi import Request


_log = logging.getLogger(__name__)

_ACTIVE_JOB_STATUSES = frozenset({"pending", "running"})


def _has_retryable_batch_entries(kind: object, request: object) -> bool:
    """Whether failed clips can be reconstructed without rescanning a directory."""
    if kind != "batch" or not isinstance(request, dict):
        return False
    source = request.get("source")
    if isinstance(source, str) and source.strip():
        return False
    entries = request.get("entries")
    return isinstance(entries, list) and bool(entries)


class WebJobRuntime:
    """Own the mutable legacy Web job lifecycle for one application instance."""

    def __init__(
        self,
        state: SessionState,
        scheduler: JobScheduler,
        max_retained_jobs: int,
        job_ttl_seconds: float,
    ) -> None:
        self.state = state
        self.scheduler = scheduler
        self.max_retained_jobs = max_retained_jobs
        self.job_ttl_seconds = job_ttl_seconds

    def _remove_artifact(self, job: Job) -> None:
        """Delete only generated artifacts that belong to this server instance."""
        artifact = (job.result or {}).get("artifact_path")
        if not artifact:
            return
        try:
            path = Path(artifact).resolve()
            export_root = self.state.export_root.resolve()
            path.relative_to(export_root)
        except (OSError, ValueError, TypeError):
            # A job result may point at user-owned data. Never remove a path
            # unless it is provably contained by the ephemeral export root.
            return
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except OSError:
            _log.warning("failed to remove expired job artifact %s", path, exc_info=True)

    def prune_locked(self, now: float) -> None:
        terminal: list[Job] = []
        for job in self.state.jobs.values():
            if job.status in _ACTIVE_JOB_STATUSES:
                continue
            if job.terminal_since is None:
                job.terminal_since = now
            terminal.append(job)

        expired = {
            job.id
            for job in terminal
            if now - max(job.terminal_since or now, job.last_accessed_at) >= self.job_ttl_seconds
        }
        for job_id in expired:
            removed = self.state.jobs.pop(job_id, None)
            if removed is not None:
                self._remove_artifact(removed)

        retained = sorted(
            (job for job in terminal if job.id not in expired),
            key=lambda job: job.terminal_since or job.created_at,
        )
        overflow = max(0, len(retained) - self.max_retained_jobs)
        for job in retained[:overflow]:
            removed = self.state.jobs.pop(job.id, None)
            if removed is not None:
                self._remove_artifact(removed)

    def cli_reproduction(  # noqa: PLR0911 - each unsupported case needs its reason
        self,
        kind: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Build an exact CLI equivalent when the public CLI covers this Web job."""
        if kind not in {"retarget", "batch"}:
            return {
                "available": False,
                "command": None,
                "reason": "该任务类型暂时没有等价的 hhtools CLI 命令。",
            }

        if request.get("retarget_fps") is not None:
            return {
                "available": False,
                "command": None,
                "reason": "当前 CLI 尚未提供 Retarget FPS 重采样参数。",
            }
        if request.get("export_fps") is not None or request.get("fps") is not None:
            return {
                "available": False,
                "command": None,
                "reason": "当前 CLI 尚未提供 Web 导出 FPS 参数。",
            }
        if request.get("t_start") is not None or request.get("t_end") is not None:
            return {
                "available": False,
                "command": None,
                "reason": "当前 CLI 尚未提供 Web 时间区间导出参数。",
            }
        if bool(request.get("foot_clamp_anti_penetration")):
            return {
                "available": False,
                "command": None,
                "reason": "当前 CLI 尚未提供同等的脚部防穿透参数。",
            }
        if str(request.get("format") or "csv").lower() != "csv":
            return {
                "available": False,
                "command": None,
                "reason": "当前 hhtools retarget CLI 仅能复现 CSV 导出。",
            }

        entries = request.get("entries")
        if kind == "batch" and isinstance(entries, list):
            source_paths = [
                str(entry.get("source_path"))
                for entry in entries
                if isinstance(entry, dict) and entry.get("source_path")
            ]
        else:
            source = request.get("source_path")
            source_paths = [str(source)] if source else []
        if not source_paths:
            return {
                "available": False,
                "command": None,
                "reason": "任务只保留了会话 token，没有可供 CLI 重开的源文件路径。",
            }
        if any(not Path(path).is_file() for path in source_paths):
            return {
                "available": False,
                "command": None,
                "reason": "一个或多个源文件已不存在，无法生成可执行的 CLI 命令。",
            }
        try:
            ephemeral_root = self.state.upload_root.resolve()
            if any(Path(path).resolve().is_relative_to(ephemeral_root) for path in source_paths):
                return {
                    "available": False,
                    "command": None,
                    "reason": "源文件来自临时上传目录，请先保存到 Motion Library。",
                }
        except OSError:
            return {
                "available": False,
                "command": None,
                "reason": "无法确认源文件是否可供 CLI 访问。",
            }

        robot = str(request.get("robot") or "").strip()
        if not robot:
            return {
                "available": False,
                "command": None,
                "reason": "任务记录缺少目标机器人名称。",
            }

        backend = str(request.get("backend") or "newton").strip().lower()
        if backend == "newton" and any(
            Path(path).suffix.lower() != ".npz" for path in source_paths
        ):
            return {
                "available": False,
                "command": None,
                "reason": "Newton CLI 当前只直接接受 NPZ 输入。",
            }
        if kind == "batch" and isinstance(entries, list):
            references = {
                str(entry.get("reference") or request.get("reference") or "smpl")
                for entry in entries
                if isinstance(entry, dict)
            }
            if len(references) > 1:
                return {
                    "available": False,
                    "command": None,
                    "reason": "该批次包含多个校准参考，当前 CLI 需要按参考分别运行。",
                }
        if request.get("csv_header") is False:
            return {
                "available": False,
                "command": None,
                "reason": "当前 CLI 尚未提供无表头 CSV 导出参数。",
            }
        output = str(
            request.get("out_dir") or ("batch_export" if kind == "batch" else "retarget_output.csv")
        )
        reference = str(request.get("reference") or "smpl")
        human_height = float(request.get("human_height") or 1.7)
        ik_iterations = int(request.get("ik_iterations") or 24)

        if backend == "interaction_mesh":
            args = ["hhtools", "retarget", "interaction-mesh", "run", *source_paths]
            args.extend(
                [
                    "--robot",
                    robot,
                    "--output",
                    output,
                    "--human-height",
                    f"{human_height:g}",
                    "--calibration-reference",
                    reference,
                ]
            )
            if request.get("limit_frames") is not None:
                args.extend(["--limit-frames", str(int(request["limit_frames"]))])
        elif backend == "newton":
            args = ["hhtools", "retarget", "run", *source_paths]
            args.extend(
                [
                    "--robot",
                    robot,
                    "--output",
                    output,
                    "--ik-iterations",
                    str(ik_iterations),
                    "--human-height",
                    f"{human_height:g}",
                    "--calibration-reference",
                    reference,
                ]
            )
            if request.get("limit_frames") is not None:
                args.extend(["--limit-frames", str(int(request["limit_frames"]))])
        else:
            return {
                "available": False,
                "command": None,
                "reason": f"未知求解器 {backend!r} 无法映射到 CLI。",
            }
        return {"available": True, "command": shlex.join(args), "reason": None}

    def reserve_slot(self) -> JobReservation:
        """Reserve scheduler admission before a route performs durable writes."""
        from fastapi import HTTPException

        try:
            return self.scheduler.reserve()
        except JobQueueFullError as err:
            raise HTTPException(status_code=429, detail=str(err)) from err
        except JobSchedulerClosedError as err:
            raise HTTPException(status_code=503, detail=str(err)) from err

    def scheduler_payload(
        self,
        *,
        editable: bool | None = None,
    ) -> dict[str, int | bool | str]:
        snapshot = self.scheduler.snapshot()
        payload: dict[str, int | bool | str] = {
            "mode": "unlimited" if snapshot.max_running_jobs == 0 else "queued",
            "max_running_jobs": snapshot.max_running_jobs,
            "max_queued_jobs": snapshot.max_queued_jobs,
            "running_jobs": snapshot.running_jobs,
            "queued_jobs": snapshot.queued_jobs,
            "reserved_jobs": snapshot.reserved_jobs,
            "cancelling_jobs": snapshot.cancelling_jobs,
            "closed": snapshot.closed,
        }
        if editable is not None:
            payload["editable"] = editable
        return payload

    def settings_editable(self, request: Request) -> bool:
        """Expose the same local-admin boundary enforced by the PATCH route."""
        client_host = request.client.host if request.client is not None else None
        return _is_loopback_address(client_host) and _is_loopback_address(
            request.url.hostname,
            allow_localhost=True,
        )

    def schedule(
        self,
        kind: str,
        request: dict[str, Any] | None = None,
        target: Callable[..., None] | None = None,
        *,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        parent_job_id: str | None = None,
        reservation: JobReservation | None = None,
    ) -> Job:
        """Create, persist, and submit one background Job through the scheduler."""
        from fastapi import HTTPException

        if target is None:
            raise TypeError("scheduled job target is required")
        admission = reservation or self.reserve_slot()
        submitted = False
        try:
            now = time.monotonic()
            job = Job(
                id=uuid.uuid4().hex[:12],
                kind=kind,
                request=_snapshot_job_request(request or {}),
                status="pending",
                message="等待可用的任务执行槽位…",
                parent_job_id=parent_job_id,
                on_terminal=self.persist_terminal,
            )

            def run() -> None:
                try:
                    job.mark_running()
                    if job.message == "等待可用的任务执行槽位…":
                        job.message = "任务已开始…"
                    self.persist(job)
                    target(job, *args, **(kwargs or {}))
                except Exception as err:  # noqa: BLE001 - expose worker failure
                    _log.exception("unhandled %s job failure", kind)
                    if job.status in _ACTIVE_JOB_STATUSES:
                        job.error = str(err)
                        job.mark_terminal("error")
                finally:
                    if job.status in _ACTIVE_JOB_STATUSES:
                        job.error = "后台任务提前结束，未发布完成状态。"
                        job.mark_terminal("error")

            def cancel(reason: str) -> None:
                if job.status not in _ACTIVE_JOB_STATUSES:
                    return
                job.message = "任务未开始"
                job.error = reason
                job.mark_terminal("error")

            try:
                with self.state.job_lock:
                    self.prune_locked(now)
                    self.state.jobs[job.id] = job
                    self.persist(job)
            except Exception:
                with self.state.job_lock:
                    self.state.jobs.pop(job.id, None)
                raise
            try:
                admission.submit(run, on_cancel=cancel)
            except JobSchedulerClosedError as err:
                cancel(str(err))
                raise HTTPException(status_code=503, detail=str(err)) from err
            except Exception:
                cancel("无法启动后台任务。")
                raise
            submitted = True
            return job
        finally:
            if not submitted:
                admission.cancel()

    def get(self, job_id: str) -> Job | None:
        now = time.monotonic()
        with self.state.job_lock:
            self.prune_locked(now)
            job = self.state.jobs.get(job_id)
            if job is not None:
                job.last_accessed_at = now
            return job

    def parameter_summary(self, job: Job) -> dict[str, str | int | float | bool]:
        """Return compact, stable parameters suitable for the always-on job drawer."""
        request = job.request
        summary: dict[str, str | int | float | bool] = {}
        keys = (
            "robot",
            "target",
            "target_robot",
            "source_robot",
            "source",
            "profile",
            "reference",
            "backend",
            "embedding",
            "format",
            "retarget_fps",
            "export_fps",
            "source_fps",
            "batch_size",
            "out_dir",
            "folder_label",
            "library_folder_label",
        )
        for key in keys:
            value = request.get(key)
            if isinstance(value, (str, int, float, bool)) and value != "":
                summary[key] = value

        entries = request.get("entries")
        files = request.get("files")
        if isinstance(entries, list):
            summary["entry_count"] = len(entries)
        elif isinstance(request.get("entry_count"), int):
            summary["entry_count"] = request["entry_count"]
        if isinstance(files, list):
            summary["file_count"] = len(files)
        elif isinstance(request.get("file_count"), int):
            summary["file_count"] = request["file_count"]
        return summary

    def result_summary(self, job: Job) -> dict[str, str | int | float | bool]:
        result = job.result or {}
        summary: dict[str, str | int | float | bool] = {}
        for key in ("download_name", "num_frames", "clip_count", "solver_mode", "format"):
            value = result.get(key)
            if isinstance(value, (str, int, float, bool)) and value != "":
                summary[key] = value
        written = result.get("written")
        failures = result.get("failures")
        errors = result.get("errors")
        if isinstance(written, list):
            summary["success_count"] = len(written)
        if isinstance(failures, list):
            summary["failure_count"] = len(failures)
        elif isinstance(errors, list):
            summary["failure_count"] = len(errors)
        return summary

    def can_download(self, job: Job) -> bool:
        artifact = (job.result or {}).get("artifact_path")
        if job.status != "done" or not isinstance(artifact, str):
            return False
        try:
            return Path(artifact).is_file()
        except OSError:
            return False

    def record(self, job: Job) -> dict[str, Any]:
        finished = job.finished_wall_time
        duration_end = finished if finished is not None else time.time()
        cli = self.cli_reproduction(job.kind, job.request)
        spec = build_job_spec(job.kind, job.request)
        replay = replay_capability(spec, ephemeral_root=self.state.upload_root)
        failures = (job.result or {}).get("failures")
        failed_item_count = len(failures) if isinstance(failures, list) else 0
        return {
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "progress": job.progress,
            "clip_progress": job.clip_progress,
            "message": job.message,
            "error": job.error,
            "created_at": job.created_wall_time,
            "finished_at": finished,
            "duration_seconds": max(0.0, duration_end - job.created_wall_time),
            "parameters": self.parameter_summary(job),
            "result_summary": self.result_summary(job),
            "can_download": self.can_download(job),
            "can_copy_cli": bool(cli["available"]),
            "can_retry": (bool(replay["available"]) and job.status not in _ACTIVE_JOB_STATUSES),
            "retry_reason": (
                "任务仍在排队或运行中。" if job.status in _ACTIVE_JOB_STATUSES else replay["reason"]
            ),
            "can_retry_failed": (
                _has_retryable_batch_entries(job.kind, job.request)
                and job.status not in _ACTIVE_JOB_STATUSES
                and failed_item_count > 0
                and bool(replay["available"])
            ),
            "failed_item_count": failed_item_count,
            "parent_job_id": job.parent_job_id,
            "scope": "current_session",
        }

    def persistent_record(self, job: Job) -> dict[str, Any]:
        record = {
            **self.record(job),
            "schema_version": 1,
            "scope": "persistent",
            "request": _snapshot_job_request(job.request),
            "cli": self.cli_reproduction(job.kind, job.request),
            "parent_job_id": job.parent_job_id,
        }
        failures = (job.result or {}).get("failures")
        if isinstance(failures, list):
            record["failures"] = _snapshot_job_request(failures)
        artifact = (job.result or {}).get("artifact_path")
        if isinstance(artifact, str):
            record["artifact_path"] = artifact
        download_name = (job.result or {}).get("download_name")
        if isinstance(download_name, str):
            record["download_name"] = download_name
        return record

    def persist(self, job: Job) -> None:
        self.state.job_history.put(self.persistent_record(job))

    def persist_terminal(self, job: Job, status: str) -> None:
        """Retain generated output before publishing and persisting terminal status."""
        artifact = (job.result or {}).get("artifact_path")
        if isinstance(artifact, str):
            try:
                artifact_path = Path(artifact).resolve()
                artifact_path.relative_to(self.state.export_root.resolve())
                adopted = self.state.job_history.adopt_artifact(
                    job.id,
                    artifact_path,
                    download_name=(job.result or {}).get("download_name"),
                )
                if job.result is not None:
                    job.result["artifact_path"] = str(adopted)
                    job.result["download_name"] = adopted.name
            except (OSError, ValueError):
                _log.warning(
                    "could not retain generated artifact for job %s",
                    job.id,
                    exc_info=True,
                )
        job._publish_terminal(status)
        self.persist(job)

    def stored_record(self, record: dict[str, Any]) -> dict[str, Any]:
        artifact = self.state.job_history.artifact_path(record)
        cli = self.cli_reproduction(
            str(record.get("kind") or ""),
            record.get("request") or {},
        )
        spec = build_job_spec(
            str(record.get("kind") or ""),
            record.get("request") or {},
        )
        replay = replay_capability(spec, ephemeral_root=self.state.upload_root)
        failures = record.get("failures")
        failed_item_count = len(failures) if isinstance(failures, list) else 0
        return {
            key: value
            for key, value in {
                **record,
                "can_download": artifact is not None,
                "can_copy_cli": bool(cli["available"]),
                "can_retry": bool(replay["available"]),
                "retry_reason": replay["reason"],
                "can_retry_failed": (
                    _has_retryable_batch_entries(
                        record.get("kind"),
                        record.get("request"),
                    )
                    and failed_item_count > 0
                    and bool(replay["available"])
                ),
                "failed_item_count": failed_item_count,
                "scope": "persistent",
            }.items()
            if key
            not in {
                "request",
                "cli",
                "artifact_path",
                "download_name",
                "schema_version",
                "failures",
            }
        }

    def config_payload(
        self,
        job: Job | None,
        stored: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if job is not None:
            spec = build_job_spec(job.kind, job.request)
            return {
                "schema_version": 1,
                "job_id": job.id,
                "kind": job.kind,
                "status": job.status,
                "created_at": job.created_wall_time,
                "finished_at": job.finished_wall_time,
                "scope": "current_session",
                "request": job.request,
                "cli": self.cli_reproduction(job.kind, job.request),
                "spec": spec,
                "replay": replay_capability(
                    spec,
                    ephemeral_root=self.state.upload_root,
                ),
                "parent_job_id": job.parent_job_id,
            }
        if stored is None:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="unknown job")
        spec = build_job_spec(
            str(stored.get("kind") or ""),
            stored.get("request") or {},
        )
        return {
            "schema_version": int(stored.get("schema_version") or 1),
            "job_id": stored["id"],
            "kind": stored["kind"],
            "status": stored["status"],
            "created_at": stored["created_at"],
            "finished_at": stored.get("finished_at"),
            "scope": "persistent",
            "request": stored.get("request") or {},
            "cli": self.cli_reproduction(
                str(stored.get("kind") or ""),
                stored.get("request") or {},
            ),
            "spec": spec,
            "replay": replay_capability(
                spec,
                ephemeral_root=self.state.upload_root,
            ),
            "parent_job_id": stored.get("parent_job_id"),
        }
