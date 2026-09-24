"""Basket, H2R batch retarget, and artifact export routes."""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse

from hhtools.application.export import (
    _parse_csv_header,
    _parse_optional_fps,
    _parse_optional_time,
    _write_export,
    _write_r2r_export,
)
from hhtools.application.motions import _load_batch_motion
from hhtools.application.progress import _BATCH_ZIP_PROGRESS, _set_batch_job_progress
from hhtools.application.retarget import _request_human_height
from hhtools.application.state import Job, _snapshot_job_request
from hhtools.web.server.batch_runtime import (
    _batch_export_retargeted_chunk,
    _record_batch_failure,
    _retarget_newton_batch_chunk,
    _run_batch_entries_sequential,
)
from hhtools.web.server.library_runtime import (
    _enrich_basket_entry,
    _entries_for_batch_source,
    _entry_reference,
    _normalise_batch_profile,
    _resolve_batch_source,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BatchRouteOperations:
    run_batch_job: Callable


def register_batch_routes(app, *, state, jobs) -> BatchRouteOperations:
    _schedule_job = jobs.schedule

    @app.get("/api/basket")
    def basket_get() -> dict:
        return {"basket": state.basket}

    @app.post("/api/basket/add")
    async def basket_add(body: dict) -> dict:
        fallback = (body.get("reference") or "smpl").strip()
        for e in body.get("entries", []):
            enriched = _enrich_basket_entry(e, fallback)
            if not any(x.get("source_path") == enriched.get("source_path") for x in state.basket):
                state.basket.append(enriched)
        return {"basket": state.basket}

    @app.post("/api/basket/clear")
    async def basket_clear() -> dict:
        state.basket.clear()
        return {"basket": state.basket}

    def _run_batch_job(job: Job, body: dict) -> None:
        try:
            robot = body["robot"]
            default_reference = body.get("reference", "smpl")
            backend = body.get("backend", "newton")
            ik_iters = int(body.get("ik_iterations", 24))
            from hhtools.robot.registry import get as _get_preset

            human_height = _request_human_height(body, _get_preset(robot), default_reference)
            out_name = body.get("out_dir") or "batch_export"
            fmt = (body.get("format") or "csv").lower()
            csv_header = _parse_csv_header(body.get("csv_header", True))
            export_fps = _parse_optional_fps(body.get("export_fps", body.get("fps")))
            retarget_fps = _parse_optional_fps(body.get("retarget_fps"))
            export_t_start = _parse_optional_time(body.get("t_start"), name="t_start")
            export_t_end = _parse_optional_time(body.get("t_end"), name="t_end")
            limit_frames = body.get("limit_frames")
            foot_clamp_anti_penetration = bool(body.get("foot_clamp_anti_penetration", False))
            requested_batch = max(1, min(256, int(body.get("batch_size", 16))))
            batch_size = requested_batch
            source_root: Path | None = None
            source_profile: str | None = None
            if "source" in body:
                source_root, source_profile, entries = _entries_for_batch_source(
                    body.get("source"), body.get("profile")
                )
            else:
                entries = [
                    _enrich_basket_entry(e, default_reference)
                    for e in (body.get("entries") or state.basket)
                ]
            model = state.robots[robot]
            if backend != "interaction_mesh":
                from hhtools.retarget.newton_basic.batch_limits import clamp_gpu_batch_size

                batch_size = clamp_gpu_batch_size(model, requested_batch)
                if batch_size < requested_batch:
                    _log.info(
                        "GPU batch_size clamped %d → %d for robot %r",
                        requested_batch,
                        batch_size,
                        robot,
                    )

            effective_request = {
                **job.request,
                "robot": robot,
                "reference": default_reference,
                "backend": backend,
                "ik_iterations": ik_iters,
                "human_height": human_height,
                "limit_frames": limit_frames,
                "batch_size": batch_size,
                "retarget_fps": retarget_fps,
                "export_fps": export_fps,
                "format": fmt,
                "csv_header": csv_header,
                "out_dir": out_name,
            }
            if source_root is None:
                effective_request["entries"] = entries
            else:
                # Keep large directory batches out of the persisted job JSON.
                effective_request.pop("entries", None)
                effective_request.update(
                    source=str(source_root),
                    profile=source_profile,
                    entry_count=len(entries),
                )
            job.request = _snapshot_job_request(effective_request)
            out_dir = state.export_root / job.id
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            total = len(entries)
            written: list[str] = []
            errors: list[str] = []
            failures: list[dict] = []
            failure_log = None
            done_clips = 0
            batch_t0 = time.monotonic()
            clamp_note = ""
            if backend != "interaction_mesh" and batch_size < requested_batch:
                clamp_note = f"（GPU 上限，批量 {requested_batch}→{batch_size}）"
            _set_batch_job_progress(
                job,
                f"批量开始 · 0/{total}{clamp_note}",
                0.0,
                batch_t0,
                clip_progress=0.0,
            )

            if backend == "interaction_mesh":
                failure_log = _run_batch_entries_sequential(
                    entries,
                    model,
                    robot,
                    default_reference,
                    backend,
                    ik_iters,
                    human_height,
                    limit_frames,
                    retarget_fps,
                    export_fps,
                    fmt,
                    csv_header,
                    out_dir,
                    state,
                    job=job,
                    job_id=job.id,
                    out_name=out_name,
                    written=written,
                    errors=errors,
                    failures=failures,
                    failure_log=failure_log,
                    batch_t0=batch_t0,
                    foot_clamp_anti_penetration=foot_clamp_anti_penetration,
                    t_start=export_t_start,
                    t_end=export_t_end,
                )
            else:
                from collections import defaultdict

                by_ref: dict[str, list[dict]] = defaultdict(list)
                for e in entries:
                    by_ref[_entry_reference(e, default_reference)].append(e)

                ref_groups = list(by_ref.items())
                for reference, ref_entries in ref_groups:
                    for chunk_start in range(0, len(ref_entries), batch_size):
                        chunk = ref_entries[chunk_start : chunk_start + batch_size]
                        loaded_chunk: list[tuple[dict, object, object]] = []
                        for e in chunk:
                            # ``done_clips`` only advances on export/failure, so
                            # add the count already loaded in this chunk
                            # (``len(loaded_chunk)``) to keep the counter moving
                            # while a large chunk loads clip-by-clip.
                            loading_pos = done_clips + len(loaded_chunk) + 1
                            _set_batch_job_progress(
                                job,
                                f"加载 {e.get('stem', '?')} · {loading_pos}/{total}",
                                (done_clips + len(loaded_chunk)) / max(1, total),
                                batch_t0,
                                clip_progress=0.0,
                            )
                            try:
                                from hhtools.services.motion_library_links import (
                                    library_entry_for_load,
                                )

                                entry = library_entry_for_load(
                                    dataset=e["dataset"],
                                    folder_label=e["folder_label"],
                                    sequence_id=e["sequence_id"],
                                    source_path=e["source_path"],
                                    upload_drop=e.get("upload_drop"),
                                )
                                motion = _load_batch_motion(
                                    e,
                                    entry,
                                    state.cache,
                                    retarget_fps=retarget_fps,
                                    limit_frames=limit_frames,
                                )
                                loaded_chunk.append((e, motion, entry))
                            except Exception as err:  # noqa: BLE001
                                failure_log = _record_batch_failure(
                                    failure_log,
                                    state,
                                    job.id,
                                    out_name,
                                    e,
                                    stage="load",
                                    reason=str(err),
                                    reference=reference,
                                    errors=errors,
                                    failures=failures,
                                )
                                done_clips += 1
                                _set_batch_job_progress(
                                    job,
                                    f"加载失败 {e.get('stem', '?')} · {done_clips}/{total}",
                                    done_clips / max(1, total),
                                    batch_t0,
                                    clip_progress=1.0,
                                )
                        if not loaded_chunk:
                            continue

                        chunk_label = (
                            f"GPU×{len(loaded_chunk)}" if len(loaded_chunk) > 1 else "逐条"
                        )
                        _set_batch_job_progress(
                            job,
                            (
                                f"参考 {reference} · {chunk_label} · "
                                f"clip {done_clips + 1}–"
                                f"{min(done_clips + len(loaded_chunk), total)}/{total}"
                            ),
                            done_clips / max(1, total),
                            batch_t0,
                            clip_progress=0.0,
                        )
                        base_prog = done_clips / max(1, total)
                        span_prog = len(loaded_chunk) / max(1, total)
                        try:
                            exports, failure_log = _retarget_newton_batch_chunk(
                                loaded_chunk,
                                model=model,
                                robot_name=robot,
                                reference=reference,
                                ik_iters=ik_iters,
                                human_height=human_height,
                                state=state,
                                job=job,
                                job_id=job.id,
                                out_name=out_name,
                                failure_log=failure_log,
                                failures=failures,
                                errors=errors,
                                progress_base=base_prog,
                                progress_span=span_prog,
                                batch_t0=batch_t0,
                                chunk_label=chunk_label,
                                foot_clamp_anti_penetration=(foot_clamp_anti_penetration),
                            )
                            done_clips, failure_log = _batch_export_retargeted_chunk(
                                exports,
                                model=model,
                                motion_out_dir=out_dir,
                                export_fps=export_fps,
                                fmt=fmt,
                                backend=backend,
                                csv_header=csv_header,
                                base_prog=base_prog,
                                span_prog=span_prog,
                                job=job,
                                batch_t0=batch_t0,
                                done_clips=done_clips,
                                total=total,
                                written=written,
                                failure_log=failure_log,
                                state=state,
                                job_id=job.id,
                                out_name=out_name,
                                reference=reference,
                                errors=errors,
                                failures=failures,
                                t_start=export_t_start,
                                t_end=export_t_end,
                            )
                        except Exception as err:  # noqa: BLE001
                            for e, _, _ in loaded_chunk:
                                failure_log = _record_batch_failure(
                                    failure_log,
                                    state,
                                    job.id,
                                    out_name,
                                    e,
                                    stage="retarget",
                                    reason=str(err),
                                    reference=reference,
                                    errors=errors,
                                    failures=failures,
                                )
                                done_clips += 1
                            _set_batch_job_progress(
                                job,
                                f"批量失败 · {done_clips}/{total}",
                                done_clips / max(1, total),
                                batch_t0,
                                clip_progress=1.0,
                            )

            if failure_log is not None:
                failure_log.finalize(job_id=job.id, out_name=out_name)

            _set_batch_job_progress(
                job,
                "正在打包 ZIP…",
                _BATCH_ZIP_PROGRESS,
                batch_t0,
                clip_progress=1.0,
            )
            from hhtools.io.export_bundle import zip_directory

            zip_path = zip_directory(out_dir, out_name, compress=False)
            gpu_note = (
                "GPU-parallel Newton"
                if backend != "interaction_mesh" and batch_size > 1
                else "per-clip"
            )
            job.result = {
                "written": written,
                "errors": errors,
                "failures": failures,
                "failure_log": str(failure_log.root) if failure_log else None,
                "format": fmt,
                "download_name": f"{out_name}.zip",
                "artifact_path": str(zip_path),
                "clip_count": len(written),
                "batch_size": batch_size,
                "requested_batch_size": requested_batch,
                "solver_mode": gpu_note,
            }
            job.progress = 1.0
            job.clip_progress = 1.0
            fail_note = f"，{len(failures)} 失败" if failures else ""
            job.message = f"{len(written)} 成功{fail_note}" + (
                f" · {gpu_note}" if backend != "interaction_mesh" else ""
            )
            job.mark_terminal("done")
        except Exception as err:  # noqa: BLE001
            _log.exception("batch job failed")
            job.error = str(err)
            job.mark_terminal("error")

    @app.post("/api/batch/retarget")
    async def batch_retarget(body: dict) -> dict:
        request = dict(body)
        if "source" in request:
            try:
                request["source"] = str(_resolve_batch_source(request.get("source")))
            except ValueError as err:
                raise HTTPException(status_code=400, detail=str(err)) from err
            request["profile"] = _normalise_batch_profile(request.get("profile"))
            request.pop("entries", None)
        job = _schedule_job("batch", request, _run_batch_job, args=(request,))
        return {"job_id": job.id}

    # ----------------------------------------------------------------- export

    @app.get("/api/export/{export_token}")
    def export(
        export_token: str,
        fps: float | None = None,
        fmt: str = "csv",
        csv_header: bool = True,
        t_start: float | None = None,
        t_end: float | None = None,
    ):
        rec = state.motions.get(f"export::{export_token}")
        if rec is None:
            raise HTTPException(status_code=404, detail="unknown export token")
        if "path" in rec:
            path = Path(rec["path"])
            media = "application/zip" if path.suffix == ".zip" else "text/csv"
            return FileResponse(path, filename=path.name, media_type=media)

        ret = rec["retargeted"]
        stem = rec["stem"]
        fmt = (fmt or "csv").lower()
        try:
            t0 = _parse_optional_time(t_start, name="t_start")
            t1 = _parse_optional_time(t_end, name="t_end")
            # The robot may have been unloaded/swapped since this clip was
            # retargeted (``/api/robot`` unload pops ``state.robots``).  A bare
            # ``state.robots[name]`` here used to raise KeyError *outside* this
            # try block → unhandled 500.  Reload the preset on demand; pkl
            # export does not need the model at all, so tolerate its absence.
            model = state.robots.get(rec["robot"])
            if model is None:
                try:
                    from hhtools.robot.loader import load_robot
                    from hhtools.robot.registry import get as _get_preset

                    model = load_robot(_get_preset(rec["robot"]), compile_mjcf=False)
                    state.robots[rec["robot"]] = model
                except Exception as load_err:  # noqa: BLE001
                    if fmt != "pkl":
                        raise RuntimeError(
                            f"robot '{rec['robot']}' is no longer loaded and "
                            f"could not be reloaded for CSV export: {load_err}"
                        ) from load_err
                    model = None  # pkl branch never dereferences the model
            if rec.get("r2r"):
                src_name = rec.get("source_robot")
                src_model = state.robots.get(src_name) if src_name else None
                if src_model is None and src_name:
                    from hhtools.robot.loader import load_robot
                    from hhtools.robot.registry import get as _get_preset

                    src_model = load_robot(_get_preset(src_name), compile_mjcf=False)
                    state.robots[src_name] = src_model
                calib = None
                if src_name and model is not None:
                    from hhtools.retarget import robot_to_robot as r2r

                    calib = r2r.load_r2r_calibration(
                        model.preset.urdf_path.parent,
                        src_name,
                        target_robot=model.preset.name,
                    )
                if src_model is None or not calib:
                    raise RuntimeError("R2R export needs source robot loaded and calibration saved")
                path = _write_r2r_export(
                    ret,
                    model,
                    rec["source_motion"],
                    state.export_root,
                    source_model=src_model,
                    calibrated_joint_q=calib,
                    entry=rec.get("r2r_entry")
                    or {
                        "source_path": rec.get("source_path"),
                        "stem": stem,
                        "has_scene": rec.get("has_scene"),
                    },
                    stem=stem,
                    fps=fps,
                    fmt=fmt,
                    csv_header=_parse_csv_header(csv_header),
                    yellow_foot_z=rec.get("yellow_foot_z"),
                    t_start=t0,
                    t_end=t1,
                )
            else:
                path = _write_export(
                    ret,
                    model,
                    rec["source_motion"],
                    state.export_root,
                    stem=stem,
                    fps=fps,
                    fmt=fmt,
                    backend=rec["backend"],
                    csv_header=_parse_csv_header(csv_header),
                    source_path=rec.get("source_path"),
                    yellow_foot_z=rec.get("yellow_foot_z"),
                    t_start=t0,
                    t_end=t1,
                )
        except Exception as err:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"export failed: {err}") from err
        if path.suffix == ".zip":
            return FileResponse(
                path,
                filename=f"{stem}_export.zip",
                media_type="application/zip",
            )
        return FileResponse(path, filename=path.name, media_type="text/csv")

    return BatchRouteOperations(run_batch_job=_run_batch_job)
