"""Legacy Web job replay, history, status, and download routes."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response

from hhtools.application.motions import (
    _load_motion_file,
    _load_motion_for_web,
    _load_via_adapter,
)
from hhtools.application.previews import _load_robot_export_for_web
from hhtools.application.state import Job, _snapshot_job_request
from hhtools.contracts.legacy_jobs import (
    JobSpecError,
    build_job_spec,
    normalize_job_spec,
    replay_capability,
)
from hhtools.web.server.job_runtime import _ACTIVE_JOB_STATUSES

_log = logging.getLogger(__name__)


def register_job_routes(
    app,
    *,
    state,
    jobs,
    motion_operations,
    robot_operations,
    h2r_operations,
    batch_operations,
) -> None:
    _get_job = jobs.get
    _job_cli_reproduction = jobs.cli_reproduction
    _job_config_payload = jobs.config_payload
    _job_record = jobs.record
    _prune_jobs_locked = jobs.prune_locked
    _schedule_job = jobs.schedule
    _scheduler_payload = jobs.scheduler_payload
    _stored_job_record = jobs.stored_record
    _register_motion = motion_operations.register_motion
    _serialize_and_store_robot = robot_operations.serialize_and_store_robot
    _run_retarget_job = h2r_operations.run_retarget_job
    _run_batch_job = batch_operations.run_batch_job

    def _job_source_for_replay(job_id: str) -> tuple[dict[str, Any], str, list[dict]]:
        """Return ``(spec, status, failures)`` without exposing stored internals."""
        job = _get_job(job_id)
        if job is not None:
            failures = (job.result or {}).get("failures")
            return (
                build_job_spec(job.kind, job.request),
                job.status,
                failures if isinstance(failures, list) else [],
            )
        stored = state.job_history.get(job_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="unknown job")
        failures = stored.get("failures")
        return (
            build_job_spec(
                str(stored.get("kind") or ""),
                stored.get("request") or {},
            ),
            str(stored.get("status") or ""),
            failures if isinstance(failures, list) else [],
        )

    def _ensure_replay_robot(job: Job, robot: str) -> None:
        if robot in state.robots:
            return
        job.progress = max(job.progress, 0.01)
        job.message = f"正在重新加载机器人 {robot}…"
        _serialize_and_store_robot(robot)

    def _load_replay_motion(job: Job, request: dict[str, Any]) -> str:
        """Rebuild a motion token from the source path captured by JobSpec."""
        from hhtools.services.r2r_upload_resolve import _is_robot_export_trajectory

        source_path = Path(str(request["source_path"])).expanduser().resolve()
        job.progress = max(job.progress, 0.02)
        job.message = f"正在重新加载 {source_path.name}…"
        source_entry = request.get("source_entry")
        motion = None
        dataset: str | None = None
        library_entry: dict[str, Any] | None = None

        if isinstance(source_entry, dict):
            candidate = dict(source_entry)
            candidate["source_path"] = str(source_path)
            try:
                from hhtools.services.motion_library_links import library_entry_for_load

                entry = library_entry_for_load(
                    dataset=str(candidate.get("dataset") or "unknown"),
                    folder_label=str(candidate.get("folder_label") or source_path.parent.name),
                    sequence_id=str(candidate.get("sequence_id") or source_path.name),
                    source_path=source_path,
                    upload_drop=candidate.get("upload_drop"),
                )
                if candidate.get("dataset") == "robot" or _is_robot_export_trajectory(source_path):
                    motion = _load_robot_export_for_web(source_path, state)
                    dataset = "robot"
                else:
                    motion = _load_motion_for_web(entry, state.cache)
                    dataset = str(candidate.get("dataset") or "unknown")
                library_entry = candidate
            except Exception as err:  # noqa: BLE001 - direct file load is the fallback
                _log.info("library replay load fell back to direct IO: %s", err)

        if motion is None:
            if _is_robot_export_trajectory(source_path):
                motion = _load_robot_export_for_web(source_path, state)
                dataset = "robot"
            else:
                try:
                    motion = _load_motion_file(source_path)
                except Exception as direct_error:  # noqa: BLE001 - adapter fallback
                    motion, dataset = _load_via_adapter(source_path)
                    if motion is None:
                        raise ValueError(
                            f"无法重新加载源动作 {source_path}: {direct_error}"
                        ) from direct_error
            library_entry = {
                "dataset": dataset or "unknown",
                "folder_label": source_path.parent.name or "replay",
                "sequence_id": source_path.name,
                "source_path": str(source_path),
                "stem": source_path.stem,
                "origin": "replay",
            }

        payload = _register_motion(
            motion,
            dataset,
            "replay",
            library_entry=library_entry,
        )
        return str(payload["token"])

    def _normalise_replay_batch_entries(request: dict[str, Any]) -> dict[str, Any]:
        """Fill the library-shaped fields required by the existing batch worker."""
        replay_request = dict(request)
        normalized: list[dict[str, Any]] = []
        for raw_entry in replay_request.get("entries") or []:
            entry = dict(raw_entry)
            source = Path(str(entry["source_path"])).expanduser().resolve()
            entry["source_path"] = str(source)
            entry.setdefault("dataset", "unknown")
            entry.setdefault("folder_label", source.parent.name or "replay")
            entry.setdefault("sequence_id", source.name)
            entry.setdefault("stem", source.stem)
            if entry.get("dataset") == "unknown" and not entry.get("origin"):
                # The upload resolver is also the generic single-file loader. It
                # avoids requiring a dataset adapter for an imported NPZ/BVH/GLB.
                entry["origin"] = "upload"
                entry["upload_profile"] = "auto"
            normalized.append(entry)
        replay_request["entries"] = normalized
        return replay_request

    def _run_replayed_retarget_job(job: Job, request: dict[str, Any]) -> None:
        try:
            robot = str(request["robot"])
            _ensure_replay_robot(job, robot)
            motion_token = _load_replay_motion(job, request)
            _run_retarget_job(job, {**request, "motion_token": motion_token})
        except Exception as err:  # noqa: BLE001 - worker errors belong on the job
            _log.exception("replayed retarget job failed")
            job.error = str(err)
            job.mark_terminal("error")

    def _run_replayed_batch_job(job: Job, request: dict[str, Any]) -> None:
        try:
            _ensure_replay_robot(job, str(request["robot"]))
            _run_batch_job(job, _normalise_replay_batch_entries(request))
        except Exception as err:  # noqa: BLE001 - worker errors belong on the job
            _log.exception("replayed batch job failed")
            job.error = str(err)
            job.mark_terminal("error")

    def _failed_only_spec(
        spec: dict[str, Any],
        failures: list[dict],
    ) -> dict[str, Any]:
        if spec["kind"] != "batch":
            raise HTTPException(
                status_code=400,
                detail={"code": "not_batch", "msg": "只有批处理任务支持仅重试失败项。"},
            )
        request = dict(spec["request"])
        if isinstance(request.get("source"), str) and request["source"].strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "failed_subset_unavailable",
                    "msg": "目录批处理不保存文件清单；请重新运行整个目录。",
                },
            )
        failed_paths = {
            str(Path(str(item["source_path"])).expanduser().resolve())
            for item in failures
            if isinstance(item, dict) and item.get("source_path")
        }
        entries = [
            entry
            for entry in request.get("entries") or []
            if isinstance(entry, dict)
            and entry.get("source_path")
            and str(Path(str(entry["source_path"])).expanduser().resolve()) in failed_paths
        ]
        if not entries:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "no_failed_entries",
                    "msg": "该记录没有可定位到源文件的失败条目。",
                },
            )
        request["entries"] = entries
        out_name = str(request.get("out_dir") or "batch_export")
        request["out_dir"] = f"{out_name}_failed_retry"
        return build_job_spec("batch", request)

    def _start_replayed_job(
        spec: dict[str, Any],
        *,
        parent_job_id: str | None,
    ) -> Job:
        capability = replay_capability(spec, ephemeral_root=state.upload_root)
        if not capability["available"]:
            raise HTTPException(
                status_code=400,
                detail={"code": "job_not_replayable", "msg": capability["reason"]},
            )
        request = _snapshot_job_request(spec["request"])
        target = (
            _run_replayed_retarget_job if spec["kind"] == "retarget" else _run_replayed_batch_job
        )
        return _schedule_job(
            str(spec["kind"]),
            request,
            target,
            args=(request,),
            parent_job_id=parent_job_id,
        )

    @app.post("/api/jobs/spec/validate")
    async def validate_job_spec(body: dict) -> dict:
        """Normalize imported JSON and report whether it can be run locally."""
        try:
            spec = normalize_job_spec(body)
        except JobSpecError as err:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_job_spec", "msg": str(err)},
            ) from err
        return {
            "spec": spec,
            "replay": replay_capability(spec, ephemeral_root=state.upload_root),
        }

    @app.post("/api/jobs/replay")
    async def replay_job(body: dict) -> dict:
        """Start a new job from history or from an edited/imported JobSpec."""
        source_job_id = body.get("job_id")
        failed_only = bool(body.get("failed_only", False))
        failures: list[dict] = []
        if isinstance(source_job_id, str) and source_job_id:
            spec, source_status, failures = _job_source_for_replay(source_job_id)
            if source_status in _ACTIVE_JOB_STATUSES:
                raise HTTPException(
                    status_code=409,
                    detail={
                        # Keep the established machine-readable code for API
                        # compatibility; its message now also covers pending.
                        "code": "job_running",
                        "msg": "原任务仍在排队或运行中。",
                    },
                )
        else:
            try:
                spec = normalize_job_spec(body)
            except JobSpecError as err:
                raise HTTPException(
                    status_code=400,
                    detail={"code": "invalid_job_spec", "msg": str(err)},
                ) from err
            source_job_id = None
        if failed_only:
            spec = _failed_only_spec(spec, failures)
        job = _start_replayed_job(spec, parent_job_id=source_job_id)
        return {
            "job_id": job.id,
            "parent_job_id": source_job_id,
            "spec": build_job_spec(job.kind, job.request),
        }

    @app.get("/api/jobs")
    def job_list(limit: int = 50) -> dict:
        """List compact live and disk-backed records, newest first."""
        bounded_limit = max(1, min(100, limit))
        now = time.monotonic()
        persisted = {
            record["id"]: _stored_job_record(record)
            for record in state.job_history.list_records(limit=100)
        }
        with state.job_lock:
            _prune_jobs_locked(now)
            # Listing must not refresh last_accessed_at: otherwise the drawer's
            # periodic polling would prevent terminal jobs from expiring.
            live = {job.id: _job_record(job) for job in state.jobs.values()}
        records = sorted(
            {**persisted, **live}.values(),
            key=lambda record: float(record.get("created_at") or 0.0),
            reverse=True,
        )[:bounded_limit]
        return {
            "jobs": records,
            "session_only": False,
            "persistence": "disk",
            "scheduler": _scheduler_payload(),
        }

    @app.get("/api/job/{job_id}/config")
    def job_config(job_id: str) -> dict:
        """Return the exact effective request captured when this job was started."""
        job = _get_job(job_id)
        stored = None if job is not None else state.job_history.get(job_id)
        return _job_config_payload(job, stored)

    @app.get("/api/job/{job_id}/config/download")
    def job_config_download(job_id: str) -> Response:
        """Download the effective request as a UTF-8 JSON reproduction record."""
        job = _get_job(job_id)
        stored = None if job is not None else state.job_history.get(job_id)
        payload = _job_config_payload(job, stored)
        return Response(
            content=json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="hhtools-job-{job_id}.json"'},
        )

    @app.get("/api/job/{job_id}/cli")
    def job_cli(job_id: str) -> dict:
        """Return the exact public CLI equivalent, or why none exists yet."""
        job = _get_job(job_id)
        if job is not None:
            return _job_cli_reproduction(job.kind, job.request)
        stored = state.job_history.get(job_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return _job_cli_reproduction(
            str(stored.get("kind") or ""),
            stored.get("request") or {},
        )

    @app.get("/api/job/{job_id}")
    def job_status(job_id: str) -> dict:
        job = _get_job(job_id)
        if job is not None:
            return {**_job_record(job), "result": job.result}
        stored = state.job_history.get(job_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="unknown job")
        result = {
            "artifact_path": stored.get("artifact_path"),
            "download_name": stored.get("download_name"),
        }
        return {**_stored_job_record(stored), "result": result}

    @app.get("/api/job/{job_id}/download")
    def job_download(job_id: str):
        job = _get_job(job_id)
        if job is not None:
            if job.status != "done":
                raise HTTPException(status_code=404, detail="job not ready")
            artifact = (job.result or {}).get("artifact_path")
            path = Path(artifact) if artifact else None
            name = (job.result or {}).get("download_name") or (path.name if path else None)
        else:
            stored = state.job_history.get(job_id)
            if stored is None or stored.get("status") != "done":
                raise HTTPException(status_code=404, detail="job not ready")
            path = state.job_history.artifact_path(stored)
            name = stored.get("download_name") or (path.name if path else None)
        if path is None:
            raise HTTPException(status_code=404, detail="no download artifact")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="artifact missing")
        media_type = {
            ".csv": "text/csv",
            ".zip": "application/zip",
        }.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(
            path,
            filename=name or path.name,
            media_type=media_type,
        )
