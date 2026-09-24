"""Progress accounting helpers shared by Web batch and retarget runtimes."""

from __future__ import annotations

import math
import time
from typing import Any


def _format_duration(seconds: float) -> str:
    """Human-readable duration for batch ETA."""
    if not math.isfinite(seconds) or seconds < 0:
        return "估算中…"
    sec = int(seconds + 0.5)
    if sec < 60:
        return f"{sec} 秒"
    minutes, sec = divmod(sec, 60)
    if minutes < 60:
        return f"{minutes} 分 {sec} 秒"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} 时 {minutes} 分"


# GPU batch: IK frame progress uses only part of each chunk's budget; export + zip
# follow. Previously IK reached 100% of the chunk span before CSV/ZIP I/O, so
# ETA showed "1 s left" while dozens of large exports still ran.
_BATCH_CHUNK_IK_FRAC = 0.82
_BATCH_CHUNK_EXPORT_FRAC = 0.18
_BATCH_ZIP_PROGRESS = 0.985
_BATCH_EXPORT_WORKERS = 8


def _batch_chunk_ik_progress(
    progress_base: float,
    progress_span: float,
    frame_frac: float,
) -> tuple[float, float]:
    ik_clip = 0.05 + 0.95 * min(1.0, max(0.0, frame_frac))
    total = progress_base + progress_span * ik_clip * _BATCH_CHUNK_IK_FRAC
    return total, ik_clip * _BATCH_CHUNK_IK_FRAC


def _batch_chunk_export_progress(
    progress_base: float,
    progress_span: float,
    export_frac: float,
) -> tuple[float, float]:
    export_frac = min(1.0, max(0.0, export_frac))
    total = progress_base + progress_span * (
        _BATCH_CHUNK_IK_FRAC + _BATCH_CHUNK_EXPORT_FRAC * export_frac
    )
    clip_p = _BATCH_CHUNK_IK_FRAC + _BATCH_CHUNK_EXPORT_FRAC * export_frac
    return total, clip_p


def _batch_eta_suffix(progress: float, t0: float) -> str:
    """Linear ETA from elapsed time and fractional progress."""
    if progress <= 0.02 or progress >= 0.88:
        return ""
    elapsed = time.monotonic() - t0
    if elapsed <= 0:
        return ""
    remaining = elapsed * (1.0 - progress) / progress
    return f" · 预计剩余 {_format_duration(remaining)}"


def _set_batch_job_progress(
    job: Any | None,
    message: str,
    progress: float,
    t0: float,
    *,
    clip_progress: float | None = None,
) -> None:
    if job is None:
        return
    job.progress = min(0.99, max(0.0, float(progress)))
    if clip_progress is not None:
        job.clip_progress = min(1.0, max(0.0, float(clip_progress)))
    job.message = message + _batch_eta_suffix(job.progress, t0)


def _job_is_batch(job: Any | None) -> bool:
    return job is not None and job.kind in ("batch", "r2r_batch")


def _set_retarget_job_clip_progress(
    job: Any | None,
    value: float,
    message: str,
) -> None:
    """Update per-clip progress during batch retarget; otherwise ``job.progress``."""
    if job is None:
        return
    v = min(0.99, max(0.0, float(value)))
    if _job_is_batch(job):
        job.clip_progress = v
    else:
        job.progress = v
    job.message = message
