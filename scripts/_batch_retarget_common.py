"""Shared helpers for offline batch retarget scripts.

Mirrors Web export contents via :func:`hhtools.web.export_bundle.write_retarget_export_bundle`
(with ``pack_scene=False`` so scene clips stay as folders, not zips).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

# Dataset adapter name → default calibration reference (same map as the Web UI).
DATASET_TO_REFERENCE: dict[str, str] = {
    "amass": "smpl",
    "motion_x": "smplx",
    "phuma": "smpl",
    "lafan": "lafan_bvh",
    "mocap": "mocap_bvh",
    "soma": "soma_bvh",
    "xsens_mocap": "xsens_mocap",
    "gvhmr": "gvhmr",
    "omomo": "smplx",
    "meshmimic_holosoma": "smplx",
    "glb": "glb",
    "unified_npz": "smpl",
    "parc_ms": "smpl",
}

_log = logging.getLogger("batch_retarget")


@dataclass(frozen=True)
class BatchClipConfig:
    robot: str
    in_root: Path
    out_root: Path
    dataset: str
    reference: str
    human_height: float
    backend: str
    limit_frames: int | None
    fmt: str = "csv"
    csv_header: bool = True
    fps: float | None = None


def exit_reason(returncode: int) -> str:
    if returncode < 0:
        return f"killed by signal {-returncode}"
    if returncode > 128:
        return f"killed by signal {returncode - 128}"
    return f"exit code {returncode}"


def append_failure_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def list_sequences(dataset: str, in_root: Path) -> list[str]:
    """List relative sequence ids via the registered dataset adapter."""
    import hhtools.io.datasets  # noqa: F401 — register adapters
    from hhtools.io.datasets import get_dataset

    adapter_cls = get_dataset(dataset)
    adapter = adapter_cls(root=in_root)
    return list(adapter.list_sequences())


def export_layout(seq_id: str, out_root: Path, *, has_scene: bool) -> tuple[Path, Path, str]:
    """Return ``(export_root, expected_csv, stem)`` for skip-existing checks.

    Folder-style clips (``<clip>/<clip>.ext``) land in ``out_root/<clip>/``.
    Flat / nested files mirror the relative parent under ``out_root``.
    """
    rel = Path(seq_id)
    stem = rel.stem
    if rel.parent.name == stem:
        export_root = out_root
        csv_path = out_root / stem / f"{stem}.csv"
    elif has_scene:
        export_root = out_root
        csv_path = out_root / stem / f"{stem}.csv"
    else:
        export_root = out_root / rel.parent if str(rel.parent) != "." else out_root
        csv_path = export_root / f"{stem}.csv"
    return export_root, csv_path, stem


def process_sequence(seq_id: str, cfg: BatchClipConfig) -> Path:
    """Retarget one sequence and write a Web-compatible export bundle."""
    import hhtools.io.datasets  # noqa: F401
    from hhtools.io.datasets import get_dataset
    from hhtools.retarget.calibration import resolve_calibration_file
    from hhtools.robot.loader import load_robot
    from hhtools.robot.registry import get as get_preset
    from hhtools.robot.registry import refresh
    from hhtools.web.export_bundle import (
        identity_resample,
        motion_has_scene,
        write_retarget_export_bundle,
    )

    refresh()
    preset = get_preset(cfg.robot)
    robot_model = load_robot(preset)
    if preset.urdf_path is None:
        raise RuntimeError(f"robot {cfg.robot!r} has no URDF on disk")
    cal_path = resolve_calibration_file(preset.urdf_path.parent, cfg.reference)
    if cal_path is None:
        raise RuntimeError(
            f"no calibration for robot {cfg.robot!r} reference {cfg.reference!r}",
        )

    adapter = get_dataset(cfg.dataset)(root=cfg.in_root)
    motion = adapter.load_motion(seq_id)
    if cfg.limit_frames is not None and motion.num_frames > cfg.limit_frames:
        n = int(cfg.limit_frames)
        motion.positions = motion.positions[:n]
        motion.quaternions = motion.quaternions[:n]
        for ob in getattr(motion, "objects", None) or []:
            if getattr(ob, "positions", None) is not None:
                ob.positions = ob.positions[:n]
            if getattr(ob, "quaternions", None) is not None:
                ob.quaternions = ob.quaternions[:n]

    backend = cfg.backend.strip().lower()
    if backend == "interaction_mesh":
        from hhtools.retarget.interaction_mesh.pipeline import InteractionMeshPipeline

        pipe = InteractionMeshPipeline.from_calibration(
            robot_model, motion, str(cal_path), human_height=cfg.human_height,
        )
        retargeted = pipe.run(motion)
    elif backend == "newton":
        from hhtools.retarget.calibration import load_calibration
        from hhtools.retarget.newton_basic import NewtonBasicPipeline
        from hhtools.robot.retarget_profile import (
            build_feet_stabilizer_config,
            build_pipeline_config_for_preset,
            build_scaler_config_for_robot,
        )

        calibration = load_calibration(cal_path)
        scaler_cfg = build_scaler_config_for_robot(
            calibration, robot_model, motion, human_height=cfg.human_height,
        )
        pipeline_cfg = build_pipeline_config_for_preset(
            preset, cfg.reference, ik_iterations=50,
        )
        feet_cfg = None
        if pipeline_cfg.apply_feet_stabilizer:
            feet_cfg = build_feet_stabilizer_config(
                preset, cfg.reference, model=robot_model,
            )
        pipe = NewtonBasicPipeline(
            robot_model,
            scaler_config=scaler_cfg,
            feet_stabilizer_config=feet_cfg,
            pipeline_config=pipeline_cfg,
            human_height=cfg.human_height,
            configure_warp=False,
        )
        retargeted = pipe.run(motion)
    else:
        raise ValueError(f"unknown backend {cfg.backend!r}")

    rel = Path(seq_id)
    stem = rel.stem
    source_path = (cfg.in_root / seq_id).resolve()
    has_scene = motion_has_scene(motion)
    if rel.parent.name == stem or has_scene:
        export_root = cfg.out_root
    else:
        export_root = cfg.out_root / rel.parent if str(rel.parent) != "." else cfg.out_root

    return write_retarget_export_bundle(
        retargeted,
        robot_model,
        motion,
        export_root,
        stem=stem,
        fps=cfg.fps,
        fmt=cfg.fmt,
        backend=backend,
        resample_fn=identity_resample,
        csv_header=cfg.csv_header,
        source_path=source_path,
        pack_scene=False,
    )


def add_common_args(
    p: argparse.ArgumentParser,
    *,
    default_reference: str | None = None,
    default_backend: str = "interaction_mesh",
    datasets: Sequence[str] | None = None,
) -> None:
    p.add_argument("--robot", required=True, help="Registered robot preset name (e.g. rp1).")
    p.add_argument("--in", dest="in_root", type=Path, required=True, help="Dataset root.")
    p.add_argument("--out", dest="out_root", type=Path, required=True, help="Output root.")
    if datasets:
        p.add_argument(
            "--dataset",
            choices=list(datasets),
            default=datasets[0],
            help=f"Dataset adapter (default: {datasets[0]}).",
        )
    else:
        p.add_argument("--dataset", required=True, help="Registered dataset adapter name.")
    p.add_argument(
        "--reference",
        default=default_reference,
        help="Calibration reference (default: dataset-specific).",
    )
    p.add_argument(
        "--backend",
        choices=("newton", "interaction_mesh"),
        default=default_backend,
        help=f"Retarget backend (default: {default_backend}).",
    )
    p.add_argument("--human-height", type=float, default=1.7, help="Subject height in metres.")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N clips.")
    p.add_argument("--clip", action="append", default=None, help="Process only this stem (repeatable).")
    p.add_argument(
        "--limit-frames", type=int, default=None,
        help="Cap frames per clip (smoke test).",
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip clips whose CSV already exists.")
    p.add_argument(
        "--fmt", choices=("csv", "pkl"), default="csv",
        help="Robot trajectory format (default: csv; matches Web).",
    )
    p.add_argument(
        "--no-csv-header",
        action="store_true",
        help="Write headerless robot CSV (legacy training layouts).",
    )
    p.add_argument("--fps", type=float, default=None, help="Optional resample FPS for export.")
    p.add_argument(
        "--in-process",
        action="store_true",
        help="Run clips in this process (a native crash aborts the whole batch).",
    )
    p.add_argument(
        "--failure-log",
        type=Path,
        default=None,
        help="Append per-clip failures as JSON lines.",
    )
    p.add_argument(
        "--_worker-seq",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--verbose", "-v", action="store_true")


def worker_command(
    script: Path,
    cfg: BatchClipConfig,
    seq_id: str,
    *,
    verbose: bool,
    extra: Sequence[str] | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script),
        "--robot", cfg.robot,
        "--in", str(cfg.in_root),
        "--out", str(cfg.out_root),
        "--dataset", cfg.dataset,
        "--reference", cfg.reference,
        "--backend", cfg.backend,
        "--human-height", str(cfg.human_height),
        "--fmt", cfg.fmt,
        "--_worker-seq", seq_id,
        "--in-process",
    ]
    if not cfg.csv_header:
        cmd.append("--no-csv-header")
    if cfg.limit_frames is not None:
        cmd.extend(["--limit-frames", str(cfg.limit_frames)])
    if cfg.fps is not None:
        cmd.extend(["--fps", str(cfg.fps)])
    if verbose:
        cmd.append("--verbose")
    if extra:
        cmd.extend(extra)
    return cmd


def filter_sequences(
    seqs: list[str],
    *,
    clips: list[str] | None,
    limit: int | None,
) -> list[str]:
    if clips is not None:
        want = set(clips)
        seqs = [s for s in seqs if Path(s).stem in want or s in want]
    if limit is not None:
        seqs = seqs[:limit]
    return seqs


def run_batch_main(
    args: argparse.Namespace,
    *,
    script_path: Path,
    scene_default: bool,
    resolve_reference: Callable[[argparse.Namespace], str] | None = None,
    extra_worker_args: Sequence[str] | None = None,
) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    reference = (
        resolve_reference(args)
        if resolve_reference is not None
        else (args.reference or DATASET_TO_REFERENCE.get(args.dataset, "smpl"))
    )

    cfg = BatchClipConfig(
        robot=args.robot,
        in_root=args.in_root.resolve(),
        out_root=args.out_root.resolve(),
        dataset=args.dataset,
        reference=reference,
        human_height=args.human_height,
        backend=args.backend,
        limit_frames=args.limit_frames,
        fmt=args.fmt,
        csv_header=not args.no_csv_header,
        fps=args.fps,
    )

    if args._worker_seq is not None:
        from hhtools.retarget.newton_basic._warp_config import configure as configure_warp_cache

        configure_warp_cache()
        try:
            process_sequence(args._worker_seq, cfg)
        except Exception as err:  # noqa: BLE001
            _log.exception("FAILED %s: %s", args._worker_seq, err)
            return 1
        return 0

    from hhtools.retarget.newton_basic._warp_config import configure as configure_warp_cache

    configure_warp_cache()

    if not cfg.in_root.is_dir():
        _log.error("input root not found: %s", cfg.in_root)
        return 2

    try:
        from hhtools.robot.registry import get as get_preset
        from hhtools.robot.registry import refresh

        refresh()
        get_preset(cfg.robot)
    except KeyError as err:
        _log.error("robot %r not registered: %s", cfg.robot, err)
        return 2

    seqs = filter_sequences(
        list_sequences(cfg.dataset, cfg.in_root),
        clips=args.clip,
        limit=args.limit,
    )
    if not seqs:
        _log.error("no %s sequences found under %s", cfg.dataset, cfg.in_root)
        return 1

    cfg.out_root.mkdir(parents=True, exist_ok=True)
    isolate = not args.in_process
    mode = "subprocess" if isolate else "in-process"
    _log.info(
        "retargeting %d clip(s) → %s (robot=%s, dataset=%s, backend=%s, ref=%s, mode=%s)",
        len(seqs), cfg.out_root, cfg.robot, cfg.dataset, cfg.backend, cfg.reference, mode,
    )

    written: list[str] = []
    failed: list[tuple[str, str]] = []
    t_start = time.time()
    repo_root = script_path.resolve().parents[1]

    for i, seq_id in enumerate(seqs, start=1):
        stem = Path(seq_id).stem
        _, csv_path, _ = export_layout(seq_id, cfg.out_root, has_scene=scene_default)
        if args.skip_existing and csv_path.is_file():
            _log.info("[%d/%d] skip existing %s", i, len(seqs), stem)
            written.append(stem)
            continue

        _log.info("[%d/%d] %s", i, len(seqs), seq_id)
        t0 = time.time()

        if isolate:
            proc = subprocess.run(
                worker_command(
                    script_path, cfg, seq_id,
                    verbose=args.verbose,
                    extra=extra_worker_args,
                ),
                cwd=str(repo_root),
            )
            if proc.returncode != 0:
                reason = exit_reason(proc.returncode)
                _log.error("  FAILED %s: %s", stem, reason)
                failed.append((stem, reason))
                if args.failure_log is not None:
                    append_failure_log(
                        args.failure_log,
                        {
                            "stem": stem,
                            "sequence_id": seq_id,
                            "reason": reason,
                            "returncode": proc.returncode,
                            "ts": time.time(),
                        },
                    )
                continue
        else:
            try:
                out_path = process_sequence(seq_id, cfg)
            except Exception as err:  # noqa: BLE001
                reason = str(err)
                _log.exception("  FAILED %s: %s", stem, err)
                failed.append((stem, reason))
                if args.failure_log is not None:
                    append_failure_log(
                        args.failure_log,
                        {
                            "stem": stem,
                            "sequence_id": seq_id,
                            "reason": reason,
                            "returncode": 1,
                            "ts": time.time(),
                        },
                    )
                continue
            csv_path = out_path if out_path.suffix else out_path / f"{stem}.csv"

        dt = time.time() - t0
        _log.info("  → %s (%.1fs)", csv_path, dt)
        written.append(stem)

    elapsed = time.time() - t_start
    _log.info("done: %d ok, %d failed in %.1fs", len(written), len(failed), elapsed)
    if failed:
        _log.warning("failed clips:")
        for stem, reason in failed:
            _log.warning("  %s: %s", stem, reason)
    return 0 if written else 1
