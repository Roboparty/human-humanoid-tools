# SPDX-License-Identifier: Apache-2.0
"""Collection-level orchestration: analyze many clips, embed, cluster, summarise.

This is the entry point the web layer calls.  It runs the per-clip pipeline
(:func:`hhtools.analysis.clip.analyze_clip`), then the collection-level steps that
need the whole distribution: embedding fit, 2-D scatter, clustering,
distribution-relative tags, and a histogram / tag-count summary for the UI.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np

from hhtools.analysis import cluster as _cluster
from hhtools.analysis import tags as _tags
from hhtools.analysis.clip import AnalyzableClip, analyze_clip
from hhtools.analysis.config import load_config
from hhtools.analysis.embedding import make_embedding

ProgressCb = Callable[[float, str], None]

# Metric keys surfaced as histograms / scatter axes in the UI.
_NUMERIC_METRIC_KEYS: tuple[str, ...] = (
    "duration_s",
    "complexity",
    "joint_kinetic_energy",
    "joint_accel_energy",
    "root_speed_xy",
    "root_speed_z",
    "root_turn_rate",
    "com_height_range",
    "airborne_ratio",
    "path_efficiency",
    "step_freq",
    "leg_energy",
    "arm_energy",
    "inverted_ratio",
    "max_torso_tilt_deg",
    "s_phy",
)

_AUTO = frozenset({None, 0, "0", "auto"})
_BLAS_THREAD_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _resolve_workers(
    workers: int | None, n_entries: int, cfg: dict[str, Any]
) -> int:
    if n_entries <= 1:
        return 1
    parallel = cfg.get("parallel") or {}
    cap_raw = parallel.get("max_workers")
    cap = (
        min(8, os.cpu_count() or 4)
        if cap_raw in _AUTO
        else max(1, int(cap_raw))
    )
    if workers is None:
        raw = parallel.get("workers")
        workers = cap if raw in _AUTO else int(raw)
    workers = min(int(workers), cap)
    return max(1, min(workers, n_entries))


def _pin_parallel_thread_env() -> None:
    """Set before spawning workers so numpy/OpenBLAS import with one thread."""
    for key in _BLAS_THREAD_VARS:
        os.environ[key] = "1"


def _parallel_worker_init() -> None:
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass


def _analyze_one(index: int, entry: dict[str, Any], cfg: dict[str, Any]) -> AnalyzableClip:
    return analyze_clip(
        entry["source_path"],
        clip_id=entry.get("clip_id") or str(index),
        source_path=entry["source_path"],
        dataset=entry.get("dataset", ""),
        folder_label=entry.get("folder_label", ""),
        cfg=cfg,
    )


def _analyze_entry_task(
    args: tuple[int, dict[str, Any], dict[str, Any]],
) -> tuple[int, AnalyzableClip]:
    index, entry, cfg = args
    return index, _analyze_one(index, entry, cfg)


def _analyze_entries(
    entries: list[dict[str, Any]],
    cfg: dict[str, Any],
    workers: int,
    progress: ProgressCb | None,
) -> list[AnalyzableClip]:
    total = len(entries)
    if workers <= 1:
        clips: list[AnalyzableClip] = []
        for i, e in enumerate(entries):
            if progress is not None:
                progress(0.05 + 0.7 * (i / max(total, 1)), f"分析 {e.get('clip_id', '')}")
            clips.append(_analyze_one(i, e, cfg))
        return clips

    _pin_parallel_thread_env()
    slots: list[AnalyzableClip | None] = [None] * total
    done = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_parallel_worker_init,
    ) as pool:
        futures = [
            pool.submit(_analyze_entry_task, (i, e, cfg))
            for i, e in enumerate(entries)
        ]
        for fut in as_completed(futures):
            index, clip = fut.result()
            slots[index] = clip
            done += 1
            if progress is not None:
                progress(0.05 + 0.7 * (done / total), f"分析 {done}/{total}")
    assert all(c is not None for c in slots)
    return slots  # type: ignore[return-value]


def analyze_entries(
    entries: list[dict[str, Any]],
    *,
    cfg: dict[str, Any] | None = None,
    embedding_name: str | None = None,
    workers: int | None = None,
    progress: ProgressCb | None = None,
) -> list[AnalyzableClip]:
    """Analyze a list of ``{clip_id, source_path, dataset, folder_label}`` dicts.

    Per-clip loading runs in parallel when ``workers > 1`` (default: auto, capped
    by ``parallel.max_workers`` in the analysis YAML).  ``workers=1`` forces
    sequential execution.
    """
    cfg = cfg or load_config()
    embedding_name = embedding_name or cfg.get("embedding", {}).get("backend", "handcrafted")
    n_workers = _resolve_workers(workers, len(entries), cfg)
    clips = _analyze_entries(entries, cfg, n_workers, progress)

    ok = [c for c in clips if c.error is None and c.metrics]
    if progress is not None:
        progress(0.8, "计算 embedding 与聚类…")

    if ok:
        backend = make_embedding(embedding_name, cfg)
        try:
            vecs = backend.fit_encode(ok)
            emb = np.stack(vecs, axis=0)
            scatter = _cluster.project_2d(emb)
            labels = _cluster.cluster(emb)
            for c, v, xy, lab in zip(ok, vecs, scatter, labels):
                c.embedding = [round(float(x), 5) for x in v.tolist()]
                c.scatter = (round(float(xy[0]), 5), round(float(xy[1]), 5))
                c.cluster_id = int(lab)
        except Exception:  # noqa: BLE001 - embedding optional; metrics still usable
            pass

        _tags.assign_dataset_tags(ok, cfg)

    if progress is not None:
        progress(0.95, "汇总分布…")
    return clips


def build_summary(clips: list[AnalyzableClip], cfg: dict[str, Any]) -> dict[str, Any]:
    """Histograms per numeric metric + tag counts + cluster counts."""
    ok = [c for c in clips if c.error is None and c.metrics]
    n = len(ok)

    histograms: dict[str, Any] = {}
    for key in _NUMERIC_METRIC_KEYS:
        vals = np.array(
            [float(c.metrics.get(key)) for c in ok if c.metrics.get(key) is not None],
            dtype=np.float64,
        )
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        nbins = int(min(30, max(5, round(np.sqrt(vals.size)))))
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-9:
            hi = lo + 1.0
        counts, edges = np.histogram(vals, bins=nbins, range=(lo, hi))
        histograms[key] = {
            "counts": counts.astype(int).tolist(),
            "edges": [round(float(e), 5) for e in edges.tolist()],
            "min": round(lo, 5),
            "max": round(hi, 5),
            "mean": round(float(vals.mean()), 5),
            "median": round(float(np.median(vals)), 5),
        }

    tag_counts: dict[str, int] = {}
    for c in ok:
        for t in c.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    cluster_counts: dict[str, int] = {}
    for c in ok:
        if c.cluster_id is not None:
            cluster_counts[str(c.cluster_id)] = cluster_counts.get(str(c.cluster_id), 0) + 1

    folder_counts: dict[str, int] = {}
    for c in ok:
        folder_counts[c.folder_label] = folder_counts.get(c.folder_label, 0) + 1

    return {
        "num_clips": len(clips),
        "num_ok": n,
        "num_error": len(clips) - n,
        "numeric_keys": list(histograms.keys()),
        "histograms": histograms,
        "tag_counts": tag_counts,
        "tag_order": _tags.all_known_tags(),
        "cluster_counts": cluster_counts,
        "folder_counts": folder_counts,
    }


__all__ = ["analyze_entries", "build_summary"]
