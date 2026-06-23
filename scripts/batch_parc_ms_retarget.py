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
Requires ``mujoco`` and ``osqp`` (``pip install osqp``).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
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
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from hhtools.retarget.newton_basic._warp_config import configure as configure_warp_cache

    configure_warp_cache()

    from hhtools.io.datasets.parc_ms import ParcMsAdapter
    from hhtools.io.parc_import import heightfield_to_wavefront_obj
    from hhtools.io.robot_csv import save_robot_csv
    from hhtools.retarget.calibration import resolve_calibration_file
    from hhtools.retarget.interaction_mesh.pipeline import InteractionMeshPipeline
    from hhtools.robot.loader import load_robot
    from hhtools.robot.registry import get as get_preset
    from hhtools.robot.registry import refresh
    from hhtools.web.export_bundle import _resolve_export_scene_params, _scaled_terrain

    in_root = args.in_root.resolve()
    out_root = args.out_root.resolve()
    if not in_root.is_dir():
        _log.error("input root not found: %s", in_root)
        return 2

    refresh()
    try:
        preset = get_preset(args.robot)
    except KeyError as err:
        _log.error("robot %r not registered: %s", args.robot, err)
        return 2
    robot_model = load_robot(preset)
    if preset.urdf_path is None:
        _log.error("robot %r has no URDF on disk", args.robot)
        return 2
    cal_path = resolve_calibration_file(preset.urdf_path.parent, args.reference)
    if cal_path is None:
        _log.error(
            "no calibration for robot %r reference %r under %s",
            args.robot, args.reference, preset.urdf_path.parent,
        )
        return 2

    adapter = ParcMsAdapter(root=in_root)
    pkls = _iter_clip_pkls(in_root, clips=args.clip)
    if args.limit is not None:
        pkls = pkls[: args.limit]
    if not pkls:
        _log.error("no parc_ms clips found under %s", in_root)
        return 1

    out_root.mkdir(parents=True, exist_ok=True)
    _log.info("retargeting %d clip(s) → %s (robot=%s)", len(pkls), out_root, args.robot)

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
        try:
            seq = str(pkl.relative_to(in_root))
            motion = adapter.load_motion(seq)
            if args.limit_frames is not None and motion.num_frames > args.limit_frames:
                motion.positions = motion.positions[: args.limit_frames]
                motion.quaternions = motion.quaternions[: args.limit_frames]

            pipe = InteractionMeshPipeline.from_calibration(
                robot_model, motion, str(cal_path), human_height=args.human_height,
            )
            ret = pipe.run(motion)

            clip_dir.mkdir(parents=True, exist_ok=True)
            save_robot_csv(
                csv_path,
                robot=robot_model,
                joint_q=ret.joint_q,
                sample_rate=ret.sample_rate,
                include_header=False,
            )

            # Terrain scaled into the robot frame (same chain as the web export
            # bundle): smpl_scale + foot-floor / terrain z_offset from ret.meta.
            smpl_scale, _z_off, z_terrain = _resolve_export_scene_params(ret.meta, motion)
            terrain_robot = _scaled_terrain(motion, smpl_scale, z_terrain)
            if terrain_robot is not None:
                heightfield_to_wavefront_obj(terrain_robot, clip_dir / f"{stem}_terrain.obj")
            else:
                _log.warning("  no terrain for %s (csv written without obj)", stem)

            dt = time.time() - t0
            _log.info("  → %s (%d frames, %.1fs)", csv_path, ret.num_frames, dt)
            written.append(stem)
        except Exception as err:  # noqa: BLE001 — keep batch going
            _log.exception("  FAILED %s: %s", stem, err)
            failed.append((stem, str(err)))

    elapsed = time.time() - t_start
    _log.info("done: %d ok, %d failed in %.1fs", len(written), len(failed), elapsed)
    if failed:
        _log.warning("failed clips:")
        for stem, reason in failed:
            _log.warning("  %s: %s", stem, reason)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
