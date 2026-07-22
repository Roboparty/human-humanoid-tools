"""Batch-retarget ``parc_ms`` clips onto a humanoid robot (interaction-mesh).

Each ``parc_ms`` clip is ``<root>/<clip>/<clip>.pkl`` (+ ``<clip>_terrain.obj``).
The interaction-mesh backend is used because the clip carries a terrain
heightfield (foot ↔ ground non-penetration + global Z-snap).

Output mirrors the ``assets/motions/meshmimic/parc_ms`` layout — one folder per
clip, **uncompressed**::

    <out_root>/<clip>/
        <clip>.csv           # headerless robot trajectory (time + root7 + dofs)
        <clip>_terrain.obj   # terrain scaled into the robot frame (smpl_scale)

Usage (smoke test on the first 5 clips)::

    python scripts/batch_parc_ms_retarget.py \\
        --robot rp1 \\
        --in ~/motions/parc_ms \\
        --out ~/motions/parc_ms_rp1 \\
        --limit 5

Run the full dataset by dropping ``--limit`` (very slow: interaction-mesh SQP
solves every frame in MuJoCo).  ``--skip-existing`` makes the job resumable.

Each clip runs in a **subprocess** by default so native crashes (e.g. SIGFPE from
MuJoCo/OSQP) skip that clip and the batch continues.  Use ``--in-process`` only
when debugging a single clip.  Optional ``--failure-log failures.jsonl`` records
failed clip names for later retry.

Requires ``mujoco`` and ``osqp`` (``pip install osqp``).
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

_log = logging.getLogger("batch_parc_ms_retarget")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", required=True, help="Registered robot preset name (e.g. rp1).")
    p.add_argument(
        "--in", dest="in_root", type=Path, required=True,
        help="parc_ms dataset root (folder of <clip>/<clip>.pkl).",
    )
    p.add_argument(
        "--out", dest="out_root", type=Path, required=True,
        help="Output root; one <clip>/ folder is written per clip.",
    )
    p.add_argument("--reference", default="smpl", help="Calibration reference (default: smpl).")
    p.add_argument("--human-height", type=float, default=1.7, help="Subject height in metres.")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N clips (smoke test).")
    p.add_argument("--clip", action="append", default=None, help="Process only this clip name (repeatable).")
    p.add_argument(
        "--limit-frames", type=int, default=None,
        help="Cap frames per clip (smoke test; reduces solve time).",
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip clips whose CSV already exists.")
    p.add_argument(
        "--in-process",
        action="store_true",
        help="Run clips in this process (faster; a native crash aborts the whole batch).",
    )
    p.add_argument(
        "--failure-log",
        type=Path,
        default=None,
        help="Append per-clip failures as JSON lines (stem, reason, returncode).",
    )
    p.add_argument(
        "--_worker-pkl",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


@dataclass(frozen=True)
class _ClipConfig:
    robot: str
    in_root: Path
    out_root: Path
    reference: str
    human_height: float
    limit_frames: int | None


def _exit_reason(returncode: int) -> str:
    if returncode < 0:
        return f"killed by signal {-returncode}"
    if returncode > 128:
        return f"killed by signal {returncode - 128}"
    return f"exit code {returncode}"


def _append_failure_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _worker_command(cfg: _ClipConfig, pkl: Path, *, verbose: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--robot",
        cfg.robot,
        "--in",
        str(cfg.in_root),
        "--out",
        str(cfg.out_root),
        "--reference",
        cfg.reference,
        "--human-height",
        str(cfg.human_height),
        "--_worker-pkl",
        str(pkl),
        "--in-process",
    ]
    if cfg.limit_frames is not None:
        cmd.extend(["--limit-frames", str(cfg.limit_frames)])
    if verbose:
        cmd.append("--verbose")
    return cmd


def _iter_clip_pkls(in_root: Path, *, clips: list[str] | None) -> list[Path]:
    """Return clip pkl paths, skipping terrain-only sidecars / legacy npz dupes."""
    pkls: list[Path] = []
    for pkl in sorted(in_root.rglob("*.pkl")):
        if not pkl.is_file():
            continue
        stem = pkl.stem
        if stem.endswith("_terrain"):
            continue
        # Legacy layout: a primary npz/npy/bvh means the pkl is a terrain sidecar.
        if any((pkl.parent / f"{stem}{ext}").is_file() for ext in (".npz", ".npy", ".bvh", ".glb", ".gltf")):
            continue
        if clips is not None and stem not in clips:
            continue
        pkls.append(pkl)
    return pkls


def _process_clip(pkl: Path, cfg: _ClipConfig) -> None:
    """Retarget one clip and write CSV (+ scaled terrain OBJ). Raises on failure."""
    from hhtools.io.datasets.parc_ms import ParcMsAdapter
    from hhtools.io.parc_import import heightfield_to_wavefront_obj
    from hhtools.io.robot_csv import save_robot_csv
    from hhtools.retarget.calibration import resolve_calibration_file
    from hhtools.retarget.interaction_mesh.pipeline import InteractionMeshPipeline
    from hhtools.robot.loader import load_robot
    from hhtools.robot.registry import get as get_preset
    from hhtools.robot.registry import refresh
    from hhtools.web.export_bundle import _resolve_export_scene_params, _scaled_terrain

    stem = pkl.stem
    clip_dir = cfg.out_root / stem
    csv_path = clip_dir / f"{stem}.csv"

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

    adapter = ParcMsAdapter(root=cfg.in_root)
    seq = str(pkl.relative_to(cfg.in_root))
    motion = adapter.load_motion(seq)
    if cfg.limit_frames is not None and motion.num_frames > cfg.limit_frames:
        motion.positions = motion.positions[: cfg.limit_frames]
        motion.quaternions = motion.quaternions[: cfg.limit_frames]

    pipe = InteractionMeshPipeline.from_calibration(
        robot_model, motion, str(cal_path), human_height=cfg.human_height,
    )
    ret = pipe.run(motion)

    clip_dir.mkdir(parents=True, exist_ok=True)
    from hhtools.web.export_bundle import bake_export_root_z

    joint_q, _lift = bake_export_root_z(robot_model, ret, source_motion=motion)
    save_robot_csv(
        csv_path,
        robot=robot_model,
        joint_q=joint_q,
        sample_rate=ret.sample_rate,
        include_header=False,
    )

    smpl_scale, _z_off, z_terrain = _resolve_export_scene_params(ret.meta, motion)
    terrain_robot = _scaled_terrain(motion, smpl_scale, z_terrain)
    if terrain_robot is not None:
        heightfield_to_wavefront_obj(terrain_robot, clip_dir / f"{stem}_terrain.obj")
    else:
        _log.warning("  no terrain for %s (csv written without obj)", stem)


def _run_worker(args: argparse.Namespace) -> int:
    from hhtools.retarget.newton_basic._warp_config import configure as configure_warp_cache

    configure_warp_cache()
    pkl = Path(args._worker_pkl).resolve()
    cfg = _ClipConfig(
        robot=args.robot,
        in_root=args.in_root.resolve(),
        out_root=args.out_root.resolve(),
        reference=args.reference,
        human_height=args.human_height,
        limit_frames=args.limit_frames,
    )
    try:
        _process_clip(pkl, cfg)
    except Exception as err:  # noqa: BLE001
        _log.exception("FAILED %s: %s", pkl.stem, err)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args._worker_pkl is not None:
        return _run_worker(args)

    from hhtools.retarget.newton_basic._warp_config import configure as configure_warp_cache

    configure_warp_cache()

    in_root = args.in_root.resolve()
    out_root = args.out_root.resolve()
    if not in_root.is_dir():
        _log.error("input root not found: %s", in_root)
        return 2

    try:
        from hhtools.robot.registry import get as get_preset
        from hhtools.robot.registry import refresh

        refresh()
        get_preset(args.robot)
    except KeyError as err:
        _log.error("robot %r not registered: %s", args.robot, err)
        return 2

    pkls = _iter_clip_pkls(in_root, clips=args.clip)
    if args.limit is not None:
        pkls = pkls[: args.limit]
    if not pkls:
        _log.error("no parc_ms clips found under %s", in_root)
        return 1

    cfg = _ClipConfig(
        robot=args.robot,
        in_root=in_root,
        out_root=out_root,
        reference=args.reference,
        human_height=args.human_height,
        limit_frames=args.limit_frames,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    isolate = not args.in_process
    mode = "subprocess" if isolate else "in-process"
    _log.info(
        "retargeting %d clip(s) → %s (robot=%s, mode=%s)",
        len(pkls), out_root, args.robot, mode,
    )

    written: list[str] = []
    failed: list[tuple[str, str]] = []
    t_start = time.time()

    for i, pkl in enumerate(pkls, start=1):
        stem = pkl.stem
        clip_dir = out_root / stem
        csv_path = clip_dir / f"{stem}.csv"
        if args.skip_existing and csv_path.is_file():
            _log.info("[%d/%d] skip existing %s", i, len(pkls), stem)
            written.append(stem)
            continue

        _log.info("[%d/%d] %s", i, len(pkls), stem)
        t0 = time.time()

        if isolate:
            proc = subprocess.run(
                _worker_command(cfg, pkl, verbose=args.verbose),
                cwd=str(Path(__file__).resolve().parents[1]),
            )
            if proc.returncode != 0:
                reason = _exit_reason(proc.returncode)
                _log.error("  FAILED %s: %s", stem, reason)
                failed.append((stem, reason))
                if args.failure_log is not None:
                    _append_failure_log(
                        args.failure_log,
                        {
                            "stem": stem,
                            "source_pkl": str(pkl),
                            "reason": reason,
                            "returncode": proc.returncode,
                            "ts": time.time(),
                        },
                    )
                continue
        else:
            try:
                _process_clip(pkl, cfg)
            except Exception as err:  # noqa: BLE001
                reason = str(err)
                _log.exception("  FAILED %s: %s", stem, err)
                failed.append((stem, reason))
                if args.failure_log is not None:
                    _append_failure_log(
                        args.failure_log,
                        {
                            "stem": stem,
                            "source_pkl": str(pkl),
                            "reason": reason,
                            "returncode": 1,
                            "ts": time.time(),
                        },
                    )
                continue

        dt = time.time() - t0
        _log.info("  → %s (%s, %.1fs)", csv_path, stem, dt)
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
