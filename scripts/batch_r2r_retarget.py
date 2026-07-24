"""Batch robot→robot retarget (offline; matches Web R2R export).

Input is a tree of **already-exported source-robot** trajectories (CSV/PKL/NPZ),
optionally with meshmimic terrain / intermimic object sidecars — the same layout
Web accepts via R2R upload.

Requires a saved R2R calibration on the **target** robot::

    <target_urdf_dir>/r2r_calibration_<source_robot>.yaml

(Calibrate once in the Web UI, or pass ``--calibration`` / ``--init-zero-calibration``.)

Output matches Web contents; scene clips stay as folders (not zip)::

    # meshmimic / intermimic
    <out>/<clip>/<clip>.csv + terrain/object sidecars

    # mimic (flat)
    <out>/<optional/rel>/<stem>.csv

Usage::

    python scripts/batch_r2r_retarget.py \\
        --source-robot rp1 \\
        --target-robot unitree_g1__g1_29dof \\
        --in ~/motions/rp1_exports \\
        --out ~/motions/g1_from_rp1 \\
        --profile auto \\
        --limit 5

    # smoke with a temporary zero-pose calibration:
    python scripts/batch_r2r_retarget.py \\
        --source-robot rp1 --target-robot unitree_g1__g1_29dof \\
        --in /tmp/meshmimic_batch_smoke --out /tmp/r2r_smoke \\
        --init-zero-calibration --limit 1 --limit-frames 8
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _batch_retarget_common import (  # noqa: E402
    append_failure_log,
    exit_reason,
)

_log = logging.getLogger("batch_r2r_retarget")


@dataclass(frozen=True)
class _R2rConfig:
    source_robot: str
    target_robot: str
    in_root: Path
    out_root: Path
    profile: str
    backend: str
    calibration: Path | None
    ik_iterations: int
    limit_frames: int | None
    fmt: str
    csv_header: bool
    fps: float | None
    t_start: float | None = None
    t_end: float | None = None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source-robot", required=True, help="Source robot preset name.")
    p.add_argument("--target-robot", required=True, help="Target robot preset name.")
    p.add_argument("--in", dest="in_root", type=Path, required=True, help="Source export root.")
    p.add_argument("--out", dest="out_root", type=Path, required=True, help="Output root.")
    p.add_argument(
        "--profile",
        choices=("auto", "mimic", "meshmimic", "intermimic"),
        default="auto",
        help="Input layout profile (default: auto).",
    )
    p.add_argument(
        "--backend",
        choices=("auto", "newton", "interaction_mesh"),
        default="auto",
        help="Retarget backend (default: auto from profile/scene).",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Override R2R calibration YAML (default: target_dir/r2r_calibration_<source>.yaml).",
    )
    p.add_argument(
        "--init-zero-calibration",
        action="store_true",
        help="If no calibration is found, write a zero-pose calibration beside the target URDF.",
    )
    p.add_argument("--ik-iterations", type=int, default=24)
    p.add_argument("--human-height", type=float, default=1.7, help=argparse.SUPPRESS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--clip", action="append", default=None, help="Process only this stem (repeatable).")
    p.add_argument("--limit-frames", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--fmt", choices=("csv", "pkl"), default="csv")
    p.add_argument("--no-csv-header", action="store_true")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument(
        "--t-start",
        type=float,
        default=None,
        help="Export window start (seconds on retargeted timeline).",
    )
    p.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="Export window end (seconds, exclusive). Exported time restarts at 0.",
    )
    p.add_argument("--in-process", action="store_true")
    p.add_argument("--failure-log", type=Path, default=None)
    p.add_argument("--_worker-seq", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def _expected_csv(out_root: Path, in_root: Path, traj_path: Path, *, has_scene: bool) -> Path:
    from hhtools.web.r2r_upload_resolve import export_subdir_for_r2r_clip

    stem = traj_path.stem
    sub = export_subdir_for_r2r_clip(in_root, traj_path)
    export_root = out_root / sub if sub else out_root
    if has_scene:
        return export_root / stem / f"{stem}.csv"
    return export_root / f"{stem}.csv"


def _load_or_init_calibration(
    target_model,
    source_name: str,
    *,
    calibration_path: Path | None,
    init_zero: bool,
) -> dict[str, float]:
    from hhtools.retarget import robot_to_robot as r2r

    if calibration_path is not None:
        path = calibration_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"calibration not found: {path}")
        import yaml

        with path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        jq = data.get("calibrated_joint_q") or {}
        if not isinstance(jq, dict) or not jq:
            raise ValueError(f"invalid R2R calibration (no calibrated_joint_q): {path}")
        return {str(k): float(v) for k, v in jq.items()}

    if target_model.preset.urdf_path is None:
        raise RuntimeError(f"target robot {target_model.preset.name!r} has no URDF on disk")
    target_dir = target_model.preset.urdf_path.parent
    calib = r2r.load_r2r_calibration(target_dir, source_name)
    if calib:
        return calib
    if not init_zero:
        raise RuntimeError(
            f"no R2R calibration for target={target_model.preset.name!r} "
            f"source={source_name!r} under {target_dir}. "
            "Calibrate in the Web UI, pass --calibration, or use --init-zero-calibration."
        )
    joint_order = [
        j.name for j in target_model.actuated_joints if j.joint_type != "fixed"
    ]
    zero = {n: 0.0 for n in joint_order}
    path = r2r.save_r2r_calibration(
        target_dir,
        target_robot=target_model.preset.name,
        source_robot=source_name,
        calibrated_joint_q=zero,
    )
    _log.warning("wrote zero-pose R2R calibration → %s", path)
    return zero


def process_r2r_clip(seq_key: str, cfg: _R2rConfig) -> Path:
    """Retarget one source trajectory path (absolute or relative to in_root)."""
    from hhtools.retarget import robot_to_robot as r2r
    from hhtools.retarget.newton_basic._warp_config import configure as configure_warp_cache
    from hhtools.robot.loader import load_robot
    from hhtools.robot.registry import get as get_preset
    from hhtools.robot.registry import refresh
    from hhtools.web.export_bundle import identity_resample
    from hhtools.web.r2r_export_bundle import write_r2r_export_bundle
    from hhtools.web.r2r_upload_resolve import export_subdir_for_r2r_clip

    configure_warp_cache()
    refresh()
    src_preset = get_preset(cfg.source_robot)
    tgt_preset = get_preset(cfg.target_robot)
    source_model = load_robot(src_preset, compile_mjcf=False)
    target_model = load_robot(tgt_preset, compile_mjcf=True)

    traj_path = Path(seq_key)
    if not traj_path.is_file():
        traj_path = (cfg.in_root / seq_key).resolve()
    if not traj_path.is_file():
        raise FileNotFoundError(f"source trajectory not found: {seq_key}")

    # Re-detect profile/scene for this clip (worker may not carry R2rClipRef).
    from hhtools.web.r2r_upload_resolve import enumerate_r2r_clips

    refs = {
        str(r.path.resolve()): r
        for r in enumerate_r2r_clips(cfg.in_root, cfg.profile)
    }
    ref = refs.get(str(traj_path.resolve()))
    profile = ref.profile if ref is not None else cfg.profile
    has_scene = bool(ref.has_scene) if ref is not None else False
    if cfg.backend == "auto":
        backend = r2r.suggested_r2r_backend(profile, has_scene=has_scene)
    else:
        backend = cfg.backend

    calib = _load_or_init_calibration(
        target_model,
        cfg.source_robot,
        calibration_path=cfg.calibration,
        init_zero=False,  # init only in parent process
    )

    traj = r2r.load_source_trajectory(traj_path, source_model=source_model)
    joint_q = traj.joint_q
    if cfg.limit_frames is not None and joint_q.shape[0] > cfg.limit_frames:
        joint_q = joint_q[: cfg.limit_frames]

    motion = r2r.source_trajectory_to_motion(
        source_model,
        joint_q,
        traj.dof_names,
        framerate=traj.framerate,
        name=traj_path.stem,
    )
    if backend == "interaction_mesh" and has_scene:
        from hhtools.web.r2r_scene import attach_r2r_clip_scene_to_motion

        motion = attach_r2r_clip_scene_to_motion(
            motion,
            traj_path.parent,
            profile=profile,
            robot_path=traj_path,
        )

    retargeted = r2r.retarget_robot_to_robot(
        source_model,
        target_model,
        calibrated_joint_q=calib,
        source_motion=motion,
        backend=backend,
        ik_iterations=cfg.ik_iterations,
    )

    stem = traj_path.stem
    sub = export_subdir_for_r2r_clip(cfg.in_root, traj_path)
    export_root = cfg.out_root / sub if sub else cfg.out_root
    entry = {
        "source_path": str(traj_path),
        "clip_dir": str(traj_path.parent),
        "stem": stem,
        "has_scene": has_scene,
        "upload_profile": profile,
    }
    return write_r2r_export_bundle(
        retargeted,
        target_model,
        motion,
        source_model=source_model,
        calibrated_joint_q=calib,
        entry=entry,
        out_root=export_root,
        stem=stem,
        fps=cfg.fps,
        fmt=cfg.fmt,
        resample_fn=identity_resample,
        csv_header=cfg.csv_header,
        pack_scene=False,
        t_start=cfg.t_start,
        t_end=cfg.t_end,
    )


def _worker_command(cfg: _R2rConfig, seq_key: str, *, verbose: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--source-robot", cfg.source_robot,
        "--target-robot", cfg.target_robot,
        "--in", str(cfg.in_root),
        "--out", str(cfg.out_root),
        "--profile", cfg.profile,
        "--backend", cfg.backend,
        "--ik-iterations", str(cfg.ik_iterations),
        "--fmt", cfg.fmt,
        "--_worker-seq", seq_key,
        "--in-process",
    ]
    if cfg.calibration is not None:
        cmd.extend(["--calibration", str(cfg.calibration)])
    if not cfg.csv_header:
        cmd.append("--no-csv-header")
    if cfg.limit_frames is not None:
        cmd.extend(["--limit-frames", str(cfg.limit_frames)])
    if cfg.fps is not None:
        cmd.extend(["--fps", str(cfg.fps)])
    if cfg.t_start is not None:
        cmd.extend(["--t-start", str(cfg.t_start)])
    if cfg.t_end is not None:
        cmd.extend(["--t-end", str(cfg.t_end)])
    if verbose:
        cmd.append("--verbose")
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = _R2rConfig(
        source_robot=args.source_robot,
        target_robot=args.target_robot,
        in_root=args.in_root.resolve(),
        out_root=args.out_root.resolve(),
        profile=args.profile,
        backend=args.backend,
        calibration=args.calibration.resolve() if args.calibration else None,
        ik_iterations=args.ik_iterations,
        limit_frames=args.limit_frames,
        fmt=args.fmt,
        csv_header=not args.no_csv_header,
        fps=args.fps,
        t_start=args.t_start,
        t_end=args.t_end,
    )

    if args._worker_seq is not None:
        try:
            process_r2r_clip(args._worker_seq, cfg)
        except Exception as err:  # noqa: BLE001
            _log.exception("FAILED %s: %s", args._worker_seq, err)
            return 1
        return 0

    from hhtools.retarget.newton_basic._warp_config import configure as configure_warp_cache
    from hhtools.robot.loader import load_robot
    from hhtools.robot.registry import get as get_preset
    from hhtools.robot.registry import refresh
    from hhtools.web.r2r_upload_resolve import enumerate_r2r_clips

    configure_warp_cache()

    if not cfg.in_root.is_dir():
        _log.error("input root not found: %s", cfg.in_root)
        return 2

    try:
        refresh()
        get_preset(cfg.source_robot)
        tgt_preset = get_preset(cfg.target_robot)
    except KeyError as err:
        _log.error("robot not registered: %s", err)
        return 2

    # Ensure calibration exists before spawning workers.
    target_model = load_robot(tgt_preset, compile_mjcf=False)
    try:
        _load_or_init_calibration(
            target_model,
            cfg.source_robot,
            calibration_path=cfg.calibration,
            init_zero=args.init_zero_calibration,
        )
    except Exception as err:  # noqa: BLE001
        _log.error("%s", err)
        return 2

    refs = enumerate_r2r_clips(cfg.in_root, cfg.profile)
    if args.clip:
        want = set(args.clip)
        refs = [r for r in refs if r.path.stem in want or r.path.name in want]
    if args.limit is not None:
        refs = refs[: args.limit]
    if not refs:
        _log.error("no R2R clips found under %s (profile=%s)", cfg.in_root, cfg.profile)
        return 1

    cfg.out_root.mkdir(parents=True, exist_ok=True)
    isolate = not args.in_process
    mode = "subprocess" if isolate else "in-process"
    _log.info(
        "R2R %d clip(s) %s → %s → %s (profile=%s, backend=%s, mode=%s)",
        len(refs), cfg.source_robot, cfg.target_robot, cfg.out_root,
        cfg.profile, cfg.backend, mode,
    )

    written: list[str] = []
    failed: list[tuple[str, str]] = []
    t_start = time.time()
    repo_root = Path(__file__).resolve().parents[1]

    for i, ref in enumerate(refs, start=1):
        stem = ref.path.stem
        try:
            seq_key = str(ref.path.resolve().relative_to(cfg.in_root))
        except ValueError:
            seq_key = str(ref.path.resolve())
        csv_path = _expected_csv(cfg.out_root, cfg.in_root, ref.path, has_scene=ref.has_scene)
        if args.skip_existing and csv_path.is_file():
            _log.info("[%d/%d] skip existing %s", i, len(refs), stem)
            written.append(stem)
            continue

        _log.info("[%d/%d] %s (%s)", i, len(refs), seq_key, ref.profile)
        t0 = time.time()

        if isolate:
            proc = subprocess.run(
                _worker_command(cfg, seq_key, verbose=args.verbose),
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
                            "sequence_id": seq_key,
                            "reason": reason,
                            "returncode": proc.returncode,
                            "ts": time.time(),
                        },
                    )
                continue
        else:
            try:
                out_path = process_r2r_clip(seq_key, cfg)
            except Exception as err:  # noqa: BLE001
                reason = str(err)
                _log.exception("  FAILED %s: %s", stem, err)
                failed.append((stem, reason))
                if args.failure_log is not None:
                    append_failure_log(
                        args.failure_log,
                        {
                            "stem": stem,
                            "sequence_id": seq_key,
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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
