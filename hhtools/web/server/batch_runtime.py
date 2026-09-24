"""Human-to-robot batch execution helpers."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from hhtools.application.export import _batch_export_subdir, _write_export
from hhtools.application.motions import _load_batch_motion
from hhtools.application.progress import (
    _BATCH_CHUNK_EXPORT_FRAC,
    _BATCH_CHUNK_IK_FRAC,
    _BATCH_EXPORT_WORKERS,
    _batch_chunk_export_progress,
    _batch_chunk_ik_progress,
    _set_batch_job_progress,
)
from hhtools.application.retarget import _retarget_single
from hhtools.application.robots import _join_robot_prewarm
from hhtools.web.server.library_runtime import _entry_reference

if TYPE_CHECKING:
    from hhtools.web.jobs.batch_failure_log import BatchFailureLog

_log = logging.getLogger(__name__)


def _batch_export_retargeted_chunk(
    exports: list[tuple[dict, object, object, object]],
    *,
    model,
    motion_out_dir,
    export_fps,
    fmt: str,
    backend: str,
    csv_header: bool,
    base_prog: float,
    span_prog: float,
    job,
    batch_t0: float,
    done_clips: int,
    total: int,
    written: list[str],
    failure_log,
    state,
    job_id: str,
    out_name: str,
    reference: str,
    errors: list[str],
    failures: list[dict],
    t_start: float | None = None,
    t_end: float | None = None,
):
    """Write retarget results for one GPU chunk."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_export = len(exports)
    if n_export == 0:
        return done_clips, failure_log

    workers = 1 if n_export <= 1 else min(_BATCH_EXPORT_WORKERS, n_export)
    prog_lock = threading.Lock()
    export_done = 0

    def _write_one(
        export_i: int,
        entry_dict: dict,
        motion: object,
        entry: object,
        ret: object,
    ) -> tuple[int, dict, str | None, str | None]:
        try:
            subdir = _batch_export_subdir(entry_dict)
            out_path = _write_export(
                ret,
                model,
                motion,
                motion_out_dir,
                stem=(motion.name or entry.stem),
                fps=export_fps,
                fmt=fmt,
                backend=backend,
                subdir=subdir,
                csv_header=csv_header,
                source_path=entry_dict.get("source_path"),
                t_start=t_start,
                t_end=t_end,
            )
            return (
                export_i,
                entry_dict,
                str(out_path.relative_to(motion_out_dir)),
                None,
            )
        except Exception as err:  # noqa: BLE001
            return export_i, entry_dict, None, str(err)

    def _record_success(rel_path: str) -> None:
        nonlocal export_done, done_clips
        with prog_lock:
            written.append(rel_path)
            export_done += 1
            done_clips += 1
            export_frac = export_done / n_export
            prog, clip_p = _batch_chunk_export_progress(
                base_prog,
                span_prog,
                export_frac,
            )
            _set_batch_job_progress(
                job,
                f"导出 · {done_clips}/{total}",
                prog,
                batch_t0,
                clip_progress=clip_p,
            )

    def _record_failure(entry_dict: dict, reason: str) -> None:
        nonlocal export_done, done_clips, failure_log
        with prog_lock:
            failure_log = _record_batch_failure(
                failure_log,
                state,
                job_id,
                out_name,
                entry_dict,
                stage="export",
                reason=reason,
                reference=reference,
                errors=errors,
                failures=failures,
            )
            export_done += 1
            done_clips += 1
            export_frac = export_done / n_export
            prog, clip_p = _batch_chunk_export_progress(
                base_prog,
                span_prog,
                export_frac,
            )
            _set_batch_job_progress(
                job,
                f"导出失败 {entry_dict.get('stem', '?')} · {done_clips}/{total}",
                prog,
                batch_t0,
                clip_progress=clip_p,
            )

    if workers == 1:
        for export_i, (entry_dict, motion, entry, ret) in enumerate(exports):
            _, _, rel, err = _write_one(export_i, entry_dict, motion, entry, ret)
            if err is not None:
                _record_failure(entry_dict, err)
            else:
                _record_success(rel)
        return done_clips, failure_log

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [
            pool.submit(_write_one, i, entry_dict, motion, entry, ret)
            for i, (entry_dict, motion, entry, ret) in enumerate(exports)
        ]
        for fut in as_completed(futs):
            _, entry_dict, rel, err = fut.result()
            if err is not None:
                _record_failure(entry_dict, err)
            else:
                _record_success(rel)
    return done_clips, failure_log


def _record_batch_failure(
    failure_log,
    state,
    job_id: str,
    out_name: str,
    entry: dict,
    *,
    stage: str,
    reason: str,
    reference: str | None,
    errors: list[str],
    failures: list[dict],
):
    from hhtools.web.jobs.batch_failure_log import open_batch_failure_log

    if failure_log is None:
        failure_log = open_batch_failure_log(state.save_dir, job_id, out_name)
    item = failure_log.record(
        entry,
        stage=stage,
        reason=reason,
        reference=reference,
    )
    failures.append(item)
    errors.append(f"{item['stem']} [{stage}]: {reason}")
    return failure_log


def _run_batch_entries_sequential(
    entries,
    model,
    robot_name,
    reference,
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
    *,
    job,
    job_id,
    out_name,
    written,
    errors,
    failures,
    failure_log,
    batch_t0: float,
    foot_clamp_anti_penetration: bool = False,
    t_start: float | None = None,
    t_end: float | None = None,
) -> BatchFailureLog | None:
    from hhtools.services.motion_library_links import library_entry_for_load

    total = len(entries)
    for i, entry_dict in enumerate(entries):
        _set_batch_job_progress(
            job,
            f"{i + 1}/{total}: {entry_dict.get('stem', '?')}",
            i / max(1, total),
            batch_t0,
            clip_progress=0.0,
        )
        ref = _entry_reference(entry_dict, reference)
        entry = library_entry_for_load(
            dataset=entry_dict["dataset"],
            folder_label=entry_dict["folder_label"],
            sequence_id=entry_dict["sequence_id"],
            source_path=entry_dict["source_path"],
            upload_drop=entry_dict.get("upload_drop"),
        )
        try:
            motion = _load_batch_motion(
                entry_dict,
                entry,
                state.cache,
                retarget_fps=retarget_fps,
                limit_frames=limit_frames,
            )
        except Exception as err:  # noqa: BLE001
            failure_log = _record_batch_failure(
                failure_log,
                state,
                job_id,
                out_name,
                entry_dict,
                stage="load",
                reason=str(err),
                reference=ref,
                errors=errors,
                failures=failures,
            )
            _set_batch_job_progress(
                job,
                f"加载失败 {entry_dict.get('stem', '?')} · {i + 1}/{total}",
                (i + 1) / max(1, total),
                batch_t0,
                clip_progress=1.0,
            )
            continue
        try:
            ret = _retarget_single(
                model,
                robot_name,
                motion,
                ref,
                backend,
                ik_iters,
                human_height,
                limit_frames,
                job,
                state=state,
                foot_clamp_anti_penetration=foot_clamp_anti_penetration,
            )
        except Exception as err:  # noqa: BLE001
            failure_log = _record_batch_failure(
                failure_log,
                state,
                job_id,
                out_name,
                entry_dict,
                stage="retarget",
                reason=str(err),
                reference=ref,
                errors=errors,
                failures=failures,
            )
            _set_batch_job_progress(
                job,
                f"重定向失败 {entry_dict.get('stem', '?')} · {i + 1}/{total}",
                (i + 1) / max(1, total),
                batch_t0,
                clip_progress=1.0,
            )
            continue
        try:
            subdir = _batch_export_subdir(entry_dict)
            out_path = _write_export(
                ret,
                model,
                motion,
                out_dir,
                stem=(motion.name or entry.stem),
                fps=export_fps,
                fmt=fmt,
                backend=backend,
                subdir=subdir,
                csv_header=csv_header,
                source_path=entry_dict.get("source_path"),
                t_start=t_start,
                t_end=t_end,
            )
            written.append(str(out_path.relative_to(out_dir)))
            _set_batch_job_progress(
                job,
                f"完成 {entry_dict.get('stem', '?')} · {i + 1}/{total}",
                (i + 1) / max(1, total),
                batch_t0,
                clip_progress=1.0,
            )
        except Exception as err:  # noqa: BLE001
            failure_log = _record_batch_failure(
                failure_log,
                state,
                job_id,
                out_name,
                entry_dict,
                stage="export",
                reason=str(err),
                reference=ref,
                errors=errors,
                failures=failures,
            )
            _set_batch_job_progress(
                job,
                f"失败 {entry_dict.get('stem', '?')} · {i + 1}/{total}",
                (i + 1) / max(1, total),
                batch_t0,
                clip_progress=1.0,
            )
    _set_batch_job_progress(
        job,
        f"批量完成 · {total}/{total}",
        1.0,
        batch_t0,
        clip_progress=1.0,
    )
    return failure_log


def _retarget_newton_batch_chunk(
    loaded: list[tuple[dict, object, object]],
    *,
    model,
    robot_name: str,
    reference: str,
    ik_iters: int,
    human_height: float,
    state,
    job,
    job_id: str,
    out_name: str,
    failure_log,
    failures: list[dict],
    errors: list[str],
    progress_base: float,
    progress_span: float,
    batch_t0: float,
    chunk_label: str,
    foot_clamp_anti_penetration: bool = False,
) -> tuple[list[tuple[dict, object, object, object]], object]:
    """Retarget pre-loaded clips; multi-env GPU when more than one is loaded."""
    from hhtools.retarget.calibration import (
        load_calibration,
        resolve_preset_calibration_file,
    )
    from hhtools.retarget.newton_basic import NewtonBasicPipeline
    from hhtools.retarget.newton_basic._warp_config import configure as configure_warp_cache
    from hhtools.robot.registry import get as get_preset

    if not loaded:
        return [], failure_log

    if len(loaded) == 1:
        entry_dict, motion, entry = loaded[0]
        ret = _retarget_single(
            model,
            robot_name,
            motion,
            reference,
            "newton",
            ik_iters,
            human_height,
            None,
            job,
            state=state,
            foot_clamp_anti_penetration=foot_clamp_anti_penetration,
        )
        return [(entry_dict, motion, entry, ret)], failure_log

    from collections import defaultdict as _defaultdict

    by_skeleton: dict[tuple, list] = _defaultdict(list)
    for item in loaded:
        sig = tuple(item[1].hierarchy.bone_names)
        by_skeleton[sig].append(item)
    if len(by_skeleton) > 1:
        merged: list[tuple[dict, object, object, object]] = []
        for group in by_skeleton.values():
            sub, failure_log = _retarget_newton_batch_chunk(
                group,
                model=model,
                robot_name=robot_name,
                reference=reference,
                ik_iters=ik_iters,
                human_height=human_height,
                state=state,
                job=job,
                job_id=job_id,
                out_name=out_name,
                failure_log=failure_log,
                failures=failures,
                errors=errors,
                progress_base=progress_base,
                progress_span=progress_span,
                batch_t0=batch_t0,
                chunk_label=chunk_label,
                foot_clamp_anti_penetration=foot_clamp_anti_penetration,
            )
            merged.extend(sub)
        return merged, failure_log

    from hhtools.robot.retarget_profile import (
        build_feet_stabilizer_config,
        build_pipeline_config_for_preset,
        bundled_scaler_path,
        resolve_retarget_scaler_config,
    )

    preset = get_preset(robot_name)
    cal_path = resolve_preset_calibration_file(preset, reference)
    if cal_path is None and bundled_scaler_path(preset, reference) is None:
        raise ValueError(
            f"robot {robot_name!r} not calibrated for reference {reference!r}; calibrate first"
        )

    _join_robot_prewarm(state, robot_name, job)
    configure_warp_cache()
    calibration = load_calibration(cal_path) if cal_path is not None else None
    scaler_cfg = resolve_retarget_scaler_config(
        preset,
        reference,
        calibration=calibration,
        model=model,
        motion=loaded[0][1],
        human_height=human_height,
    )
    feet_cfg = build_feet_stabilizer_config(preset, reference, model=model)
    _set_batch_job_progress(
        job,
        f"并行 IK {chunk_label} · 参考 {reference} · 编译内核…",
        progress_base + 0.02 * progress_span,
        batch_t0,
        clip_progress=0.02,
    )

    pipeline = NewtonBasicPipeline(
        model,
        scaler_config=scaler_cfg,
        pipeline_config=build_pipeline_config_for_preset(
            preset,
            reference,
            ik_iterations=ik_iters,
            foot_clamp_anti_penetration=foot_clamp_anti_penetration,
        ),
        feet_stabilizer_config=feet_cfg,
        human_height=human_height,
        configure_warp=False,
    )

    motions = [motion for _, motion, _ in loaded]
    ik_total = {"n": 0}

    def _frame_cb(done: int, total: int) -> None:
        if job is None:
            return
        if ik_total["n"] == 0:
            ik_total["n"] = max(1, int(total))
        if int(total) > ik_total["n"]:
            post_done = max(0, int(done) - ik_total["n"])
            post_tot = max(1, int(total) - ik_total["n"])
            total_p, clip_p = _batch_chunk_ik_progress(
                progress_base,
                progress_span,
                1.0,
            )
            post_frac = post_done / post_tot
            total_p = progress_base + progress_span * (
                _BATCH_CHUNK_IK_FRAC + 0.5 * _BATCH_CHUNK_EXPORT_FRAC * post_frac
            )
            _set_batch_job_progress(
                job,
                (f"并行 IK {chunk_label} · 参考 {reference} · 后处理 {post_done}/{post_tot}"),
                total_p,
                batch_t0,
                clip_progress=clip_p,
            )
            return
        ik_total["n"] = max(ik_total["n"], int(total))
        ik_count = ik_total["n"]
        frac = min(1.0, float(done) / max(1, ik_count))
        total_p, clip_p = _batch_chunk_ik_progress(
            progress_base,
            progress_span,
            frac,
        )
        _set_batch_job_progress(
            job,
            (
                f"并行 IK {chunk_label} · 参考 {reference} · "
                f"帧 {min(int(done), ik_count)}/{ik_count}（本批最长 clip）"
            ),
            total_p,
            batch_t0,
            clip_progress=clip_p,
        )

    try:
        results = pipeline.run_batch(motions, progress_callback=_frame_cb)
    except Exception as err:
        from hhtools.retarget.newton_basic.batch_limits import (
            is_ik_shared_memory_error,
            shared_memory_error_hint,
        )

        if not is_ik_shared_memory_error(err):
            raise
        _log.warning(
            "GPU batch IK failed (shared memory), falling back to sequential: %s",
            err,
        )
        hint = shared_memory_error_hint(getattr(pipeline.ctx, "joint_dof_count", None))
        if job is not None:
            _set_batch_job_progress(
                job,
                f"内核共享内存不足，改逐条 IK ×{len(loaded)}（参考 {reference}）…",
                progress_base + 0.05 * progress_span,
                batch_t0,
                clip_progress=0.0,
            )
        out: list[tuple[dict, object, object, object]] = []
        for i, (entry_dict, motion, entry) in enumerate(loaded):
            if job is not None:
                _set_batch_job_progress(
                    job,
                    (f"逐条 IK {i + 1}/{len(loaded)} · {entry_dict.get('stem', '?')}（{hint}）"),
                    progress_base + progress_span * (i / max(1, len(loaded))),
                    batch_t0,
                    clip_progress=0.0,
                )
            try:
                ret = _retarget_single(
                    model,
                    robot_name,
                    motion,
                    reference,
                    "newton",
                    ik_iters,
                    human_height,
                    None,
                    job,
                    state=state,
                    foot_clamp_anti_penetration=foot_clamp_anti_penetration,
                )
                out.append((entry_dict, motion, entry, ret))
            except Exception as single_err:  # noqa: BLE001
                failure_log = _record_batch_failure(
                    failure_log,
                    state,
                    job_id,
                    out_name,
                    entry_dict,
                    stage="retarget",
                    reason=str(single_err),
                    reference=reference,
                    errors=errors,
                    failures=failures,
                )
        if not out:
            raise RuntimeError(
                f"GPU batch IK failed and all {len(loaded)} sequential retries failed "
                f"(first error: {failures[-1]['reason'] if failures else err})"
            ) from err
        return out, failure_log

    if len(results) != len(loaded):
        raise RuntimeError(f"run_batch returned {len(results)} results for {len(loaded)} motions")
    return [
        (loaded[i][0], loaded[i][1], loaded[i][2], results[i]) for i in range(len(loaded))
    ], failure_log


__all__ = [
    "_batch_export_retargeted_chunk",
    "_record_batch_failure",
    "_retarget_newton_batch_chunk",
    "_run_batch_entries_sequential",
]
