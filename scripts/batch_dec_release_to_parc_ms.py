#!/usr/bin/env python3
"""Batch-convert PARC ``dec_release`` MS pickles into ``parc_ms`` clip folders.

Each output clip mirrors the hand-authored demos under
``assets/motions/meshmimic/parc_ms``::

    <out>/<clip_name>/<clip_name>.pkl
    <out>/<clip_name>/<clip_name>_terrain.obj

Example (full dec_release → local parkour library)::

    python scripts/batch_dec_release_to_parc_ms.py \\
        --src ~/motions/dec_release \\
        --out ~/motions/parc_ms \\
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hhtools.io.parc_import import (  # noqa: E402
    export_ms_pkl_to_parc_ms_clip_dir,
    load_ms_pickle_container,
)


@dataclass
class ClipJob:
    src: str
    clip_name: str


@dataclass
class ClipResult:
    src: str
    clip_name: str
    out_dir: str | None = None
    ok: bool = False
    error: str | None = None


def _safe_clip_name(rel_parent: Path, stem: str) -> str:
    prefix = "_".join(rel_parent.parts)
    raw = f"{prefix}_{stem}" if prefix else stem
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in raw)


def _plan_jobs(src_root: Path) -> tuple[list[ClipJob], list[str]]:
    pkls = sorted(src_root.rglob("*.pkl"))
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for p in pkls:
        by_stem[p.stem].append(p)

    dup_stems = {stem for stem, paths in by_stem.items() if len(paths) > 1}
    jobs: list[ClipJob] = []
    skipped: list[str] = []
    used_names: set[str] = set()

    for p in pkls:
        motion_ms, _, _ = load_ms_pickle_container(p)
        if motion_ms is None:
            skipped.append(str(p.resolve()))
            continue
        stem = p.stem
        if stem in dup_stems:
            rel_parent = p.parent.relative_to(src_root)
            clip_name = _safe_clip_name(rel_parent, stem)
        else:
            clip_name = stem

        if clip_name in used_names:
            raise RuntimeError(f"clip name collision after disambiguation: {clip_name}")
        used_names.add(clip_name)
        jobs.append(ClipJob(src=str(p.resolve()), clip_name=clip_name))

    return jobs, skipped


def _run_one(job: ClipJob, out_root: str, overwrite: bool) -> ClipResult:
    try:
        out_dir = export_ms_pkl_to_parc_ms_clip_dir(
            job.src,
            out_root,
            clip_name=job.clip_name,
            overwrite=overwrite,
        )
        return ClipResult(
            src=job.src,
            clip_name=job.clip_name,
            out_dir=str(out_dir),
            ok=True,
        )
    except Exception as exc:
        return ClipResult(
            src=job.src,
            clip_name=job.clip_name,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="dec_release root (recursive *.pkl scan).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output parc_ms library root.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel worker processes (1 = serial).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing clip folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the conversion plan.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest path (default: <out>/manifest.json).",
    )
    args = parser.parse_args()

    src_root = args.src.resolve()
    out_root = args.out.resolve()
    if not src_root.is_dir():
        print(f"[error] source not found: {src_root}", file=sys.stderr)
        return 2

    jobs, skipped = _plan_jobs(src_root)
    print(
        f"planned {len(jobs)} clip(s) from {src_root} -> {out_root}"
        + (f" (skipped {len(skipped)} without motion_data)" if skipped else "")
    )

    if args.dry_run:
        for job in jobs[:10]:
            print(f"  {job.clip_name} <= {job.src}")
        if len(jobs) > 10:
            print(f"  ... and {len(jobs) - 10} more")
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    results: list[ClipResult] = []
    workers = max(1, int(args.workers))
    out_str = str(out_root)

    if workers == 1:
        for i, job in enumerate(jobs, 1):
            res = _run_one(job, out_str, args.overwrite)
            results.append(res)
            if i % 200 == 0 or i == len(jobs):
                ok = sum(1 for r in results if r.ok)
                print(f"progress {i}/{len(jobs)} ok={ok} fail={len(results) - ok}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_one, job, out_str, args.overwrite): job
                for job in jobs
            }
            for i, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                results.append(res)
                if not res.ok:
                    print(f"[fail] {res.clip_name}: {res.error}", file=sys.stderr)
                if i % 200 == 0 or i == len(jobs):
                    ok = sum(1 for r in results if r.ok)
                    print(f"progress {i}/{len(jobs)} ok={ok} fail={len(results) - ok}")

    ok_n = sum(1 for r in results if r.ok)
    fail_n = len(results) - ok_n
    print(f"done: ok={ok_n} fail={fail_n} -> {out_root}")

    manifest_path = args.manifest or (out_root / "manifest.json")
    payload = {
        "src_root": str(src_root),
        "out_root": str(out_root),
        "total": len(results),
        "ok": ok_n,
        "fail": fail_n,
        "skipped_no_motion": skipped,
        "clips": [asdict(r) for r in sorted(results, key=lambda r: r.clip_name)],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")

    if fail_n:
        print("failures:", file=sys.stderr)
        for r in sorted(results, key=lambda x: x.clip_name):
            if not r.ok:
                print(f"  {r.clip_name}: {r.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        raise SystemExit(130)
