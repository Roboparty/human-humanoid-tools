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
import select
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
    source_fps: float | None = None
    t_start: float | None = None
    t_end: float | None = None
    gpu_batch_size: int = 1
    clip_floor_snap: bool = True


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
    p.add_argument("--fps", type=float, default=None, help="Export resample FPS.")
    p.add_argument(
        "--source-fps",
        type=float,
        default=None,
        help=(
            "Source trajectory FPS when the file omits time/sample_rate "
            "(e.g. MotionDecode CSV). Default: 50."
        ),
    )
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
    p.add_argument(
        "--gpu-batch-size",
        type=int,
        default=1,
        help=(
            "Parallel Newton IK envs per GPU step (in-process newton only). "
            "Try 8–16 on a free 4090. Default 1 (sequential)."
        ),
    )
    p.add_argument(
        "--no-clip-floor-snap",
        action="store_true",
        help=(
            "Disable post-IK clip-wide sole→z=0 snap (slow CPU mesh FK). "
            "Prefer a separate post-pass for large MotionDecode batches."
        ),
    )
    p.add_argument("--failure-log", type=Path, default=None)
    p.add_argument("--_worker-seq", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--_worker-serve",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def _expected_csv(out_root: Path, in_root: Path, traj_path: Path, *, has_scene: bool) -> Path:
    from hhtools.web.library.r2r_upload_resolve import export_subdir_for_r2r_clip

    stem = traj_path.stem
    sub = export_subdir_for_r2r_clip(in_root, traj_path)
    export_root = out_root / sub if sub else out_root
    if has_scene:
        return export_root / stem / f"{stem}.csv"
    return export_root / f"{stem}.csv"


def _skip_marker_path(csv_path: Path) -> Path:
    """Sidecar written for permanently-skipped clips (empty / poison sources)."""
    return csv_path.with_suffix(csv_path.suffix + ".skip")


def _write_skip_marker(csv_path: Path, reason: str) -> Path:
    marker = _skip_marker_path(csv_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{reason.strip()}\n", encoding="utf-8")
    return marker


def _is_permanent_skip_reason(reason: str) -> bool:
    low = (reason or "").lower()
    return (
        "0 frames" in low
        or "empty source" in low
        or "axis 0 with size 0" in low
        or "zero-size array" in low
        or "segfault" in low
        or "sigsegv" in low
        or "killed by signal 11" in low
        or "signal 11" in low
        or "worker hung/timeout" in low
        or "soft failure after retry" in low
        or "bad argument to internal function" in low
        or "error while parsing function" in low
    )


def _is_crash_returncode(returncode: int | None) -> bool:
    """True for SIGSEGV / SIGABRT-style worker deaths."""
    if returncode is None:
        return False
    if returncode < 0:
        return -returncode in (6, 7, 11)  # ABRT, BUS, SEGV
    if returncode > 128:
        return (returncode - 128) in (6, 7, 11)
    return returncode in (134, 135, 139)


def _reset_newton_pipeline(runtime: _R2rRuntime) -> None:
    """Drop a poisoned Newton/Warp pipeline so the next call rebuilds it."""
    runtime.pipeline = None


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


@dataclass
class _R2rRuntime:
    """Cached robots/calibration/pipeline for fast in-process batching."""

    source_model: object
    target_model: object
    calib: dict[str, float]
    pipeline: object | None = None


def _resolve_traj_path(seq_key: str, in_root: Path) -> Path:
    traj_path = Path(seq_key)
    if not traj_path.is_file():
        traj_path = (in_root / seq_key).resolve()
    if not traj_path.is_file():
        raise FileNotFoundError(f"source trajectory not found: {seq_key}")
    return traj_path


def _build_runtime(cfg: _R2rConfig, *, init_zero: bool = False) -> _R2rRuntime:
    from hhtools.retarget.newton_basic._warp_config import configure as configure_warp_cache
    from hhtools.robot.loader import load_robot
    from hhtools.robot.registry import get as get_preset
    from hhtools.robot.registry import refresh

    configure_warp_cache()
    refresh()
    source_model = load_robot(get_preset(cfg.source_robot), compile_mjcf=False)
    target_model = load_robot(get_preset(cfg.target_robot), compile_mjcf=True)
    calib = _load_or_init_calibration(
        target_model,
        cfg.source_robot,
        calibration_path=cfg.calibration,
        init_zero=init_zero,
    )
    return _R2rRuntime(
        source_model=source_model,
        target_model=target_model,
        calib=calib,
    )


def _ensure_newton_pipeline(cfg: _R2rConfig, runtime: _R2rRuntime) -> object:
    if runtime.pipeline is not None:
        return runtime.pipeline
    from dataclasses import replace

    from hhtools.retarget import robot_to_robot as r2r
    from hhtools.retarget.newton_basic import NewtonBasicPipeline
    from hhtools.robot.retarget_profile import (
        build_feet_stabilizer_config,
        build_pipeline_config_for_preset,
    )

    scaler_cfg, ref = r2r._build_scaler_config(  # noqa: SLF001
        runtime.source_model, runtime.target_model, runtime.calib,
    )
    reference_key = f"robot_{runtime.source_model.preset.name}"
    identity_map = {n: n for n in ref.joint_names}
    feet_cfg = build_feet_stabilizer_config(
        runtime.target_model.preset, reference_key, model=runtime.target_model,
    )
    pipe_cfg = build_pipeline_config_for_preset(
        runtime.target_model.preset, reference_key, ik_iterations=cfg.ik_iterations,
    )
    if not cfg.clip_floor_snap:
        pipe_cfg = replace(pipe_cfg, clip_floor_snap=False)
        _log.info("clip_floor_snap disabled for this batch")
    _log.info("building Newton pipeline (ik_iterations=%d)", cfg.ik_iterations)
    runtime.pipeline = NewtonBasicPipeline(
        runtime.target_model,
        scaler_config=scaler_cfg,
        pipeline_config=pipe_cfg,
        feet_stabilizer_config=feet_cfg,
        human_height=float(ref.height_m),
        source_to_canonical=identity_map,
        configure_warp=False,
    )
    return runtime.pipeline


def _resolve_profile_scene(
    cfg: _R2rConfig,
    traj_path: Path,
    *,
    profile: str | None,
    has_scene: bool | None,
) -> tuple[str, bool]:
    if profile is not None and has_scene is not None:
        return profile, bool(has_scene)
    prof = profile if profile is not None else cfg.profile
    scene = bool(has_scene) if has_scene is not None else False
    if has_scene is None and cfg.profile in ("auto", "meshmimic", "intermimic"):
        from hhtools.web.library.r2r_upload_resolve import enumerate_r2r_clips

        refs = {
            str(r.path.resolve()): r
            for r in enumerate_r2r_clips(cfg.in_root, cfg.profile)
        }
        ref = refs.get(str(traj_path.resolve()))
        if ref is not None:
            return ref.profile, bool(ref.has_scene)
    if profile is None and cfg.profile == "mimic":
        return "mimic", False
    return prof, scene


def _prepare_source_motion(
    traj_path: Path,
    cfg: _R2rConfig,
    runtime: _R2rRuntime,
    *,
    profile: str,
    has_scene: bool,
    backend: str,
):
    """Load source traj, optionally downsample to ``cfg.fps`` before FK."""
    import numpy as np
    from hhtools.core.resample import resample_time_series
    from hhtools.retarget import robot_to_robot as r2r

    traj = r2r.load_source_trajectory(
        traj_path, source_model=runtime.source_model, source_fps=cfg.source_fps,
    )
    joint_q = np.asarray(traj.joint_q, dtype=np.float32)
    src_fps = float(traj.framerate)
    if joint_q.ndim != 2 or joint_q.shape[0] == 0:
        raise ValueError("empty source trajectory (0 frames)")
    if cfg.limit_frames is not None and joint_q.shape[0] > cfg.limit_frames:
        joint_q = joint_q[: cfg.limit_frames]

    # Retarget at export FPS when possible — avoids IK on 120 Hz then throwaway.
    dst_fps = float(cfg.fps) if cfg.fps is not None and cfg.fps > 0 else src_fps
    if dst_fps > 0 and abs(dst_fps - src_fps) > 1e-6 and joint_q.shape[0] > 1:
        joint_q = resample_time_series(joint_q, src_fps, dst_fps).astype(np.float32)
        src_fps = dst_fps
    if joint_q.shape[0] == 0:
        raise ValueError("empty source trajectory (0 frames after resample)")

    motion = r2r.source_trajectory_to_motion(
        runtime.source_model,
        joint_q,
        traj.dof_names,
        framerate=src_fps,
        name=traj_path.stem,
    )
    if backend == "interaction_mesh" and has_scene:
        from hhtools.web.output.r2r_scene import attach_r2r_clip_scene_to_motion

        motion = attach_r2r_clip_scene_to_motion(
            motion,
            traj_path.parent,
            profile=profile,
            robot_path=traj_path,
        )
    return motion


def _export_retargeted(
    *,
    retargeted,
    motion,
    traj_path: Path,
    cfg: _R2rConfig,
    runtime: _R2rRuntime,
    profile: str,
    has_scene: bool,
) -> Path:
    from hhtools.web.output.export_bundle import identity_resample
    from hhtools.web.output.r2r_export_bundle import write_r2r_export_bundle
    from hhtools.web.library.r2r_upload_resolve import export_subdir_for_r2r_clip

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
    # Already retargeted at cfg.fps when set — identity export.
    return write_r2r_export_bundle(
        retargeted,
        runtime.target_model,
        motion,
        source_model=runtime.source_model,
        calibrated_joint_q=runtime.calib,
        entry=entry,
        out_root=export_root,
        stem=stem,
        fps=None if cfg.fps is not None else cfg.fps,
        fmt=cfg.fmt,
        resample_fn=identity_resample,
        csv_header=cfg.csv_header,
        pack_scene=False,
        t_start=cfg.t_start,
        t_end=cfg.t_end,
    )


def process_r2r_clip(
    seq_key: str,
    cfg: _R2rConfig,
    *,
    runtime: _R2rRuntime | None = None,
    profile: str | None = None,
    has_scene: bool | None = None,
) -> Path:
    """Retarget one source trajectory path (absolute or relative to in_root)."""
    from hhtools.retarget import robot_to_robot as r2r

    if runtime is None:
        runtime = _build_runtime(cfg, init_zero=False)

    traj_path = _resolve_traj_path(seq_key, cfg.in_root)
    prof, scene = _resolve_profile_scene(
        cfg, traj_path, profile=profile, has_scene=has_scene,
    )
    if cfg.backend == "auto":
        backend = r2r.suggested_r2r_backend(prof, has_scene=scene)
    else:
        backend = cfg.backend

    motion = _prepare_source_motion(
        traj_path, cfg, runtime, profile=prof, has_scene=scene, backend=backend,
    )
    if backend == "newton":
        pipeline = _ensure_newton_pipeline(cfg, runtime)
        try:
            retargeted = pipeline.run(motion)
        except TypeError:
            retargeted = pipeline.run(motion)
    else:
        retargeted = r2r.retarget_robot_to_robot(
            runtime.source_model,
            runtime.target_model,
            calibrated_joint_q=runtime.calib,
            source_motion=motion,
            backend=backend,
            ik_iterations=cfg.ik_iterations,
        )
    return _export_retargeted(
        retargeted=retargeted,
        motion=motion,
        traj_path=traj_path,
        cfg=cfg,
        runtime=runtime,
        profile=prof,
        has_scene=scene,
    )


def _process_newton_batch(
    refs: list,
    cfg: _R2rConfig,
    runtime: _R2rRuntime,
) -> list[tuple[object, Path | None, str | None]]:
    """GPU-parallel retarget a chunk of mimic clips. Returns (ref, out|None, err|None)."""
    from hhtools.retarget import robot_to_robot as r2r

    pipeline = _ensure_newton_pipeline(cfg, runtime)
    prepared: list[tuple[object, Path, object, str, bool]] = []
    results: list[tuple[object, Path | None, str | None]] = []

    for ref in refs:
        traj_path = Path(ref.path).resolve()
        csv_path = _expected_csv(
            cfg.out_root, cfg.in_root, traj_path, has_scene=bool(ref.has_scene),
        )
        try:
            seq_key = str(traj_path.relative_to(cfg.in_root))
        except ValueError:
            seq_key = str(traj_path)
        try:
            prof, scene = ref.profile, bool(ref.has_scene)
            if cfg.backend == "auto":
                backend = r2r.suggested_r2r_backend(prof, has_scene=scene)
            else:
                backend = cfg.backend
            if backend != "newton":
                out = process_r2r_clip(
                    seq_key, cfg, runtime=runtime, profile=prof, has_scene=scene,
                )
                results.append((ref, out, None))
                continue
            motion = _prepare_source_motion(
                traj_path, cfg, runtime, profile=prof, has_scene=scene, backend=backend,
            )
            if int(getattr(motion, "num_frames", 0) or 0) <= 0:
                reason = "empty source trajectory (0 frames)"
                _write_skip_marker(csv_path, reason)
                results.append((ref, None, reason))
                continue
            prepared.append((ref, traj_path, motion, prof, scene))
        except Exception as err:  # noqa: BLE001
            reason = str(err)
            if _is_permanent_skip_reason(reason):
                _write_skip_marker(csv_path, reason)
            results.append((ref, None, reason))

    if not prepared:
        return results

    # Sequential run+export (not run_batch): Warp/Newton occasionally SIGSEGV
    # mid-batch; exporting each clip immediately preserves progress across
    # watchdog restarts. Soft failures rebuild the pipeline and continue.
    for ref, traj_path, motion, prof, scene in prepared:
        csv_path = _expected_csv(
            cfg.out_root, cfg.in_root, traj_path, has_scene=scene,
        )
        try:
            if runtime.pipeline is None:
                pipeline = _ensure_newton_pipeline(cfg, runtime)
            ret = pipeline.run(motion)
            out = _export_retargeted(
                retargeted=ret, motion=motion, traj_path=traj_path,
                cfg=cfg, runtime=runtime, profile=prof, has_scene=scene,
            )
            results.append((ref, out, None))
        except Exception as err:  # noqa: BLE001
            reason = str(err)
            if _is_permanent_skip_reason(reason):
                _write_skip_marker(csv_path, reason)
            _reset_newton_pipeline(runtime)
            pipeline = _ensure_newton_pipeline(cfg, runtime)
            results.append((ref, None, reason))
    return results


def _base_worker_command(cfg: _R2rConfig, *, verbose: bool) -> list[str]:
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
    if cfg.source_fps is not None:
        cmd.extend(["--source-fps", str(cfg.source_fps)])
    if cfg.t_start is not None:
        cmd.extend(["--t-start", str(cfg.t_start)])
    if cfg.t_end is not None:
        cmd.extend(["--t-end", str(cfg.t_end)])
    if not cfg.clip_floor_snap:
        cmd.append("--no-clip-floor-snap")
    if verbose:
        cmd.append("--verbose")
    return cmd


def _worker_command(cfg: _R2rConfig, seq_key: str, *, verbose: bool) -> list[str]:
    return _base_worker_command(cfg, verbose=verbose) + ["--_worker-seq", seq_key]


def _worker_serve_command(cfg: _R2rConfig, *, verbose: bool) -> list[str]:
    return _base_worker_command(cfg, verbose=verbose) + ["--_worker-serve"]


def _run_worker_serve(cfg: _R2rConfig) -> int:
    """Long-lived worker: read seq keys from stdin, emit OK/ERR on stdout."""
    # Keep protocol lines on stdout; route logs to stderr for the parent log.
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        runtime = _build_runtime(cfg, init_zero=False)
    except Exception as err:  # noqa: BLE001
        print(f"ERR\tbootstrap: {err}", flush=True)
        return 2
    print("READY", flush=True)
    for raw in sys.stdin:
        seq_key = raw.strip()
        if not seq_key:
            continue
        if seq_key in {".quit", "QUIT"}:
            break
        try:
            out = process_r2r_clip(seq_key, cfg, runtime=runtime)
            print(f"OK\t{out}", flush=True)
        except Exception as err:  # noqa: BLE001
            _log.exception("FAILED %s: %s", seq_key, err)
            _reset_newton_pipeline(runtime)
            print(f"ERR\t{err}", flush=True)
    return 0


def _read_protocol_line(
    fp,
    timeout_s: float,
    state: dict,
) -> str | None:
    """Read until a READY/OK/ERR protocol line; ignore Warp banners on stdout.

    Uses ``fileno()`` + ``os.read`` because ``select`` on text-mode PIPE
    wrappers is unreliable and was causing false TIMEOUT kills.
    """
    import os

    deadline = time.monotonic() + timeout_s
    buf = str(state.get("buf", ""))
    fd = fp.fileno()
    while True:
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            state["buf"] = buf
            s = line.strip()
            if not s:
                continue
            if s == "READY" or s.startswith("OK\t") or s.startswith("ERR\t"):
                return s
            # Non-protocol noise (Warp banners, etc.)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            state["buf"] = buf
            return None
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            state["buf"] = buf
            return None
        try:
            data = os.read(fd, 4096)
        except OSError:
            state["buf"] = buf
            return None
        if not data:
            state["buf"] = buf
            return None
        buf += data.decode("utf-8", errors="replace")
        state["buf"] = buf


def _process_pending_warm_worker(
    pending: list,
    cfg: _R2rConfig,
    *,
    verbose: bool,
    failure_log: Path | None,
    repo_root: Path,
    clip_timeout_s: float = 600.0,
) -> tuple[list[str], list[tuple[str, str]], int, int]:
    """Drive a warm subprocess worker; skip clips that SIGSEGV the worker.

    Returns ``(written, failed, skipped, remaining_unprocessed)``.
    """
    written: list[str] = []
    failed: list[tuple[str, str]] = []
    skipped = 0
    worker: subprocess.Popen[bytes] | None = None
    io_state: dict = {"buf": ""}
    total = len(pending)
    processed = 0

    def _record(stem: str, seq_key: str, reason: str, returncode: int = 1) -> None:
        failed.append((stem, reason))
        if failure_log is not None:
            append_failure_log(
                failure_log,
                {
                    "stem": stem,
                    "sequence_id": seq_key,
                    "reason": reason,
                    "returncode": returncode,
                    "ts": time.time(),
                },
            )

    def _stop_worker() -> int | None:
        nonlocal worker
        io_state["buf"] = ""
        if worker is None:
            return None
        rc = worker.poll()
        if rc is None:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
            rc = worker.poll()
        old = worker
        worker = None
        return old.returncode if old is not None else rc

    def _start_worker() -> subprocess.Popen[bytes]:
        nonlocal worker
        _stop_worker()
        proc = subprocess.Popen(
            _worker_serve_command(cfg, verbose=verbose),
            cwd=str(repo_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit → batch log
            bufsize=0,
        )
        assert proc.stdout is not None
        line = _read_protocol_line(proc.stdout, 180.0, io_state)
        if line is None or line != "READY":
            rc = proc.poll()
            _stop_worker()
            raise RuntimeError(
                f"warm worker failed to start (rc={rc}, line={line!r})"
            )
        worker = proc
        return proc

    try:
        _start_worker()
    except Exception as err:  # noqa: BLE001
        _log.error("warm worker bootstrap failed: %s", err)
        return written, failed, skipped, total

    for i, ref in enumerate(pending, start=1):
        stem = ref.path.stem
        try:
            seq_key = str(ref.path.resolve().relative_to(cfg.in_root))
        except ValueError:
            seq_key = str(ref.path.resolve())
        csv_path = _expected_csv(
            cfg.out_root, cfg.in_root, ref.path, has_scene=bool(ref.has_scene),
        )
        _log.info("[%d/%d] %s (%s)", i, total, seq_key, ref.profile)
        t0 = time.time()
        assert worker is not None and worker.stdin is not None and worker.stdout is not None
        try:
            worker.stdin.write((seq_key + "\n").encode("utf-8"))
            worker.stdin.flush()
        except BrokenPipeError:
            reason = "segfault during retarget/export (worker pipe broken)"
            _log.error("  FAILED %s: %s", stem, reason)
            _write_skip_marker(csv_path, reason)
            _record(stem, seq_key, reason, 139)
            skipped += 1
            written.append(stem)
            processed += 1
            try:
                _start_worker()
            except Exception as err:  # noqa: BLE001
                _log.error("worker restart failed: %s", err)
                break
            continue

        line = _read_protocol_line(worker.stdout, clip_timeout_s, io_state)
        rc = worker.poll()
        # EOF often races ahead of waitpid after SIGSEGV — poll briefly.
        if line is None and rc is None:
            for _ in range(20):
                time.sleep(0.05)
                rc = worker.poll()
                if rc is not None:
                    break
        if line is None:
            elapsed = time.time() - t0
            if rc is None:
                _log.error(
                    "  TIMEOUT %s after %.1fs — killing worker",
                    stem, elapsed,
                )
                reason = "worker hung/timeout during retarget/export"
                _stop_worker()
            elif _is_crash_returncode(rc):
                reason = "segfault during retarget/export"
                _log.error("  FAILED %s: %s (rc=%s, %.1fs)", stem, reason, rc, elapsed)
                _stop_worker()
            else:
                reason = exit_reason(rc if rc is not None else 1)
                _log.error("  FAILED %s: %s (%.1fs)", stem, reason, elapsed)
                _stop_worker()
            # Permanent-skip poison clips so resume can advance.
            _write_skip_marker(csv_path, reason)
            skipped += 1
            written.append(stem)
            _record(stem, seq_key, reason, rc or 139)
            processed += 1
            time.sleep(1.0)
            try:
                _start_worker()
            except Exception as err:  # noqa: BLE001
                _log.error("worker restart failed: %s", err)
                break
            continue

        status, _, payload = line.partition("\t")
        if status == "OK":
            processed += 1
            dt = time.time() - t0
            _log.info("  → %s (%.1fs)", payload or csv_path, dt)
            written.append(stem)
            continue

        reason = payload or line or "unknown"
        _log.error("  FAILED %s: %s — restarting worker and retrying once", stem, reason)
        _record(stem, seq_key, reason)
        # Soft ERR often means the warm worker is poisoned (Warp/Python
        # corruption). Restart and retry once; then permanent-skip.
        time.sleep(1.0)
        try:
            _start_worker()
        except Exception as err:  # noqa: BLE001
            _log.error("worker restart failed: %s", err)
            break
        assert worker is not None and worker.stdin is not None and worker.stdout is not None
        try:
            worker.stdin.write((seq_key + "\n").encode("utf-8"))
            worker.stdin.flush()
            line2 = _read_protocol_line(worker.stdout, clip_timeout_s, io_state)
        except BrokenPipeError:
            line2 = None
        if line2 is not None and line2.startswith("OK\t"):
            processed += 1
            dt = time.time() - t0
            _log.info("  → %s (%.1fs, retry)", line2.partition("\t")[2] or csv_path, dt)
            written.append(stem)
            continue
        skip_reason = reason if _is_permanent_skip_reason(reason) else (
            f"soft failure after retry: {reason}"
        )
        _write_skip_marker(csv_path, skip_reason)
        skipped += 1
        written.append(stem)
        processed += 1
        _log.error("  skip marked %s: %s", stem, skip_reason)

    _stop_worker()
    remaining = max(0, total - processed)
    return written, failed, skipped, remaining


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
        source_fps=args.source_fps,
        t_start=args.t_start,
        t_end=args.t_end,
        gpu_batch_size=max(1, int(args.gpu_batch_size)),
        clip_floor_snap=not bool(args.no_clip_floor_snap),
    )

    if getattr(args, "_worker_serve", False):
        return _run_worker_serve(cfg)

    if args._worker_seq is not None:
        try:
            process_r2r_clip(args._worker_seq, cfg)
        except Exception as err:  # noqa: BLE001
            _log.exception("FAILED %s: %s", args._worker_seq, err)
            return 1
        return 0

    from hhtools.robot.registry import get as get_preset
    from hhtools.robot.registry import refresh
    from hhtools.web.library.r2r_upload_resolve import enumerate_r2r_clips

    if not cfg.in_root.is_dir():
        _log.error("input root not found: %s", cfg.in_root)
        return 2

    try:
        refresh()
        get_preset(cfg.source_robot)
        get_preset(cfg.target_robot)
    except KeyError as err:
        _log.error("robot not registered: %s", err)
        return 2

    # Warm-worker parent stays CUDA-free; only build runtime here when needed
    # (calib init, or legacy non-isolated single-process path).
    use_warm_worker = bool(args.in_process)
    runtime = None
    if args.init_zero_calibration or not use_warm_worker:
        try:
            runtime = _build_runtime(cfg, init_zero=args.init_zero_calibration)
        except Exception as err:  # noqa: BLE001
            _log.error("%s", err)
            return 2
        if use_warm_worker:
            # Calib ensured; child worker will load models itself.
            runtime = None

    _log.info("enumerating clips under %s (profile=%s) ...", cfg.in_root, cfg.profile)
    t_enum = time.time()
    refs = enumerate_r2r_clips(cfg.in_root, cfg.profile)
    _log.info("found %d clip(s) in %.1fs", len(refs), time.time() - t_enum)
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
    mode = "subprocess" if isolate else "warm-worker"
    use_gpu_batch = False
    batch_n = 1

    _log.info(
        "R2R %d clip(s) %s → %s → %s (profile=%s, backend=%s, mode=%s, "
        "source_fps=%s, fps=%s, gpu_batch=%d, clip_floor_snap=%s)",
        len(refs), cfg.source_robot, cfg.target_robot, cfg.out_root,
        cfg.profile, cfg.backend, mode, cfg.source_fps, cfg.fps, batch_n,
        cfg.clip_floor_snap,
    )

    written: list[str] = []
    failed: list[tuple[str, str]] = []
    skipped = 0
    t_start = time.time()
    repo_root = Path(__file__).resolve().parents[1]

    pending: list = []
    for i, ref in enumerate(refs, start=1):
        stem = ref.path.stem
        csv_path = _expected_csv(cfg.out_root, cfg.in_root, ref.path, has_scene=ref.has_scene)
        skip_path = _skip_marker_path(csv_path)
        if args.skip_existing and skip_path.is_file():
            skipped += 1
            if i == 1 or i % 500 == 0 or i == len(refs):
                _log.info("[%d/%d] skip marked (%d skipped so far)", i, len(refs), skipped)
            written.append(stem)
            continue
        if args.skip_existing and csv_path.is_file() and csv_path.stat().st_size > 0:
            skipped += 1
            if i == 1 or i % 500 == 0 or i == len(refs):
                _log.info("[%d/%d] skip existing (%d skipped so far)", i, len(refs), skipped)
            written.append(stem)
            continue
        # Cheap pre-filter: header-only MotionDecode CSVs are ~1KB / 0 frames.
        try:
            if ref.path.is_file() and ref.path.stat().st_size < 2048:
                with ref.path.open("r", encoding="utf-8", errors="ignore") as fp:
                    data_lines = 0
                    for line in fp:
                        s = line.strip()
                        if not s or s.startswith("#"):
                            continue
                        # header row has non-numeric tokens
                        first = s.split(",", 1)[0]
                        try:
                            float(first)
                            data_lines += 1
                            break
                        except ValueError:
                            continue
                if data_lines == 0:
                    reason = "empty source trajectory (0 frames)"
                    _write_skip_marker(csv_path, reason)
                    skipped += 1
                    _log.warning("skip empty source %s", stem)
                    written.append(stem)
                    continue
        except OSError:
            pass
        pending.append(ref)

    # Pack similar-length clips together (stable order across resumes).
    pending.sort(key=lambda r: r.path.stat().st_size if r.path.is_file() else 0)

    def _record_failure(stem: str, seq_key: str, reason: str, returncode: int = 1) -> None:
        failed.append((stem, reason))
        if args.failure_log is not None:
            append_failure_log(
                args.failure_log,
                {
                    "stem": stem,
                    "sequence_id": seq_key,
                    "reason": reason,
                    "returncode": returncode,
                    "ts": time.time(),
                },
            )

    remaining_pending = 0
    if use_warm_worker:
        w_ok, w_fail, w_skip, remaining_pending = _process_pending_warm_worker(
            pending,
            cfg,
            verbose=bool(args.verbose),
            failure_log=args.failure_log,
            repo_root=repo_root,
        )
        written.extend(w_ok)
        failed.extend(w_fail)
        skipped += w_skip
    else:
        total_pending = len(pending)
        for done_pending, ref in enumerate(pending, start=1):
            stem = ref.path.stem
            try:
                seq_key = str(ref.path.resolve().relative_to(cfg.in_root))
            except ValueError:
                seq_key = str(ref.path.resolve())
            _log.info(
                "[%d/%d] %s (%s)",
                done_pending, total_pending, seq_key, ref.profile,
            )
            t0 = time.time()
            csv_path = _expected_csv(
                cfg.out_root, cfg.in_root, ref.path, has_scene=ref.has_scene,
            )
            proc = subprocess.run(
                _worker_command(cfg, seq_key, verbose=args.verbose),
                cwd=str(repo_root),
            )
            if proc.returncode != 0:
                reason = exit_reason(proc.returncode)
                if _is_crash_returncode(proc.returncode):
                    reason = "segfault during retarget/export"
                _log.error("  FAILED %s: %s", stem, reason)
                if _is_permanent_skip_reason(reason) or _is_crash_returncode(proc.returncode):
                    _write_skip_marker(csv_path, reason)
                    skipped += 1
                    written.append(stem)
                _record_failure(stem, seq_key, reason, proc.returncode)
                continue
            dt = time.time() - t0
            _log.info("  → %s (%.1fs)", csv_path, dt)
            written.append(stem)

    elapsed = time.time() - t_start
    _log.info(
        "done: %d ok (%d skipped), %d failed, %d remaining in %.1fs",
        len(written), skipped, len(failed), remaining_pending, elapsed,
    )
    if failed:
        _log.warning("failed clips:")
        for stem, reason in failed[:50]:
            _log.warning("  %s: %s", stem, reason)
        if len(failed) > 50:
            _log.warning("  ... and %d more", len(failed) - 50)
    # Incomplete run (worker died / restart failed mid-queue) → non-zero so
    # the watchdog resumes.
    if remaining_pending > 0:
        _log.error(
            "incomplete: %d pending clip(s) not processed; exit 3 for resume",
            remaining_pending,
        )
        return 3
    # Belt-and-suspenders: recount unfinished outputs (csv or .skip).
    unfinished = 0
    for ref in refs:
        out_csv = _expected_csv(
            cfg.out_root, cfg.in_root, ref.path, has_scene=bool(ref.has_scene),
        )
        if out_csv.is_file() and out_csv.stat().st_size > 0:
            continue
        if _skip_marker_path(out_csv).is_file():
            continue
        unfinished += 1
        if unfinished >= 5:
            break
    if unfinished > 0:
        # Full recount for the log message.
        unfinished = 0
        for ref in refs:
            out_csv = _expected_csv(
                cfg.out_root, cfg.in_root, ref.path, has_scene=bool(ref.has_scene),
            )
            if out_csv.is_file() and out_csv.stat().st_size > 0:
                continue
            if _skip_marker_path(out_csv).is_file():
                continue
            unfinished += 1
        _log.error(
            "incomplete: %d clip(s) still lack csv/.skip; exit 3 for resume",
            unfinished,
        )
        return 3
    if not written and not skipped:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
