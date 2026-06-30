#!/usr/bin/env python3
"""Batch audit: scaffold, load, validate ik_map, calibrate, retarget smoke test.

Usage (from repo root, hhtools venv active):
  python scripts/batch_robot_audit.py [--import] [--retarget] [--json out.json]

``--import`` copies/scaffolds into configs/robots/<name>/ (skip if exists).
``--retarget`` runs Newton retarget on a short SOMA BVH clip after zero-pose calib.

Asset roots (required — no bundled defaults; set ``HHTOOLS_ROBOT_AUDIT_ROOTS`` or ``--roots``)::

  export HHTOOLS_ROBOT_AUDIT_ROOTS=$HOME/GMR/assets:$HOME/Downloads/X2_URDF-v1.3.0
  python scripts/batch_robot_audit.py --roots "$HHTOOLS_ROBOT_AUDIT_ROOTS"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Partial / lab assemblies — not full humanoids for retarget.
_SKIP_PATH_RE = re.compile(
    r"""
    /assemblies/(?:sysID_|left_|right_) |
    /assemblies/(?:left|right)_(?:arm|leg)_ |
    toddlerbot_(?:active|arms|legs)\.urdf$
    """,
    re.VERBOSE | re.IGNORECASE,
)

REFERENCES = ("soma_bvh", "lafan_bvh", "smpl", "smplx", "gvhmr")
RETARGET_REFS = ("soma_bvh", "lafan_bvh", "smplx")
TEST_BVH = REPO / "assets/motions/mimic/SOMA/big_light_one_hand_pick_up_front_low_R_005__A508.bvh"
CONFIGS = REPO / "configs/robots"


@dataclass
class RobotReport:
    name: str
    urdf: str
    exists: bool = False
    imported: bool = False
    load_ok: bool = False
    mjcf_ok: bool = False
    num_dof: int = 0
    ik_slots: int = 0
    ik_critical: list[str] = field(default_factory=list)
    mesh_issues: list[str] = field(default_factory=list)
    calib: dict[str, str] = field(default_factory=dict)
    retarget: dict[str, str] = field(default_factory=dict)
    error: str | None = None


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "robot"


def _preset_name_for(urdf: Path, root: Path) -> str:
    rel = urdf.relative_to(root)
    parts = [p for p in rel.parts if p not in {"urdf", "basic_urdf", "models", "robots"}]
    stem = urdf.stem
    if len(parts) >= 2:
        prefix = _slugify(parts[-2])
        if prefix != _slugify(stem):
            return _slugify(f"{prefix}_{stem}")
    return _slugify(stem)


# Vendor URDFs with unrecoverable XML (manual fix required).
_BROKEN_URDF_STEMS = frozenset({"GR1T2_inspire_hand"})

def _should_skip_urdf(urdf: Path) -> bool:
    path = str(urdf.as_posix())
    if urdf.stem in _BROKEN_URDF_STEMS:
        return True
    if _SKIP_PATH_RE.search(path):
        return True
    return False


def discover_urdf_candidates(
    roots: tuple[Path, ...] | None = None,
) -> list[tuple[str, Path]]:
    """Return ``(preset_name, urdf_path)`` for importable full-body URDFs."""
    roots = roots or ()
    out: list[tuple[str, Path]] = []
    seen_names: dict[str, Path] = {}

    for root in roots:
        if not root.is_dir():
            continue
        for urdf in sorted(root.rglob("*.urdf")):
            if _should_skip_urdf(urdf):
                continue
            name = _preset_name_for(urdf, root)
            if name in seen_names and seen_names[name] != urdf:
                # Disambiguate duplicate stems from different vendors.
                vendor = _slugify(root.name)
                name = f"{vendor}_{name}"
            seen_names[name] = urdf
            out.append((name, urdf))

    return sorted(out, key=lambda item: item[0])


def _import_robot(name: str, urdf: Path, *, link_meshes: bool = False) -> Path:
    import shutil

    from hhtools.robot.scaffold import scaffold_yaml_file
    from hhtools.robot.urdf_normalize import ensure_urdf_meshes_resolvable

    dest = CONFIGS / name
    dest.mkdir(parents=True, exist_ok=True)
    target_urdf = dest / urdf.name
    if not target_urdf.exists():
        shutil.copy2(urdf, target_urdf)

    src_root = urdf.parent
    mesh_candidates = [
        src_root / "meshes",
        src_root.parent / "meshes",
        urdf.parent / "meshes",
        urdf.parent.parent / "meshes",
    ]
    dest_mesh = dest / "meshes"
    for mesh_dir in mesh_candidates:
        if not mesh_dir.is_dir():
            continue
        if not dest_mesh.exists():
            if link_meshes:
                dest_mesh.symlink_to(mesh_dir.resolve(), target_is_directory=True)
            else:
                shutil.copytree(mesh_dir, dest_mesh)
        elif not link_meshes:
            for f in mesh_dir.iterdir():
                if not f.is_file():
                    continue
                out = dest_mesh / f.name
                if not out.exists():
                    shutil.copy2(f, out)
        break

    ensure_urdf_meshes_resolvable(
        target_urdf,
        search_dirs=[dest_mesh, dest, urdf.parent, urdf.parent / "meshes"],
        output_path=target_urdf,
    )

    yaml_path = dest / "robot.yaml"
    if not yaml_path.exists():
        scaffold_yaml_file(target_urdf, root_dir=dest, overwrite=True)
    return dest


def _truncate_motion(motion, *, max_frames: int = 30):
    n = min(max_frames, int(motion.positions.shape[0]))
    return replace(
        motion,
        positions=motion.positions[:n],
        quaternions=motion.quaternions[:n],
    )


def _audit_one(name: str, urdf: Path, *, do_import: bool, do_retarget: bool) -> RobotReport:
    rep = RobotReport(name=name, urdf=str(urdf), exists=urdf.is_file())
    if not rep.exists:
        rep.error = "URDF missing"
        return rep

    saved_calibrations: dict[str, object] = {}

    try:
        if do_import:
            _import_robot(name, urdf)
            rep.imported = True

        from hhtools.robot.registry import get, refresh
        from hhtools.robot.loader import load_robot
        from hhtools.robot.kinematics import CRITICAL_IK_SLOTS, prepare_ik_map, validate_ik_map
        from hhtools.robot.urdf_normalize import detect_mesh_path_issues

        refresh()
        preset = get(name)
        mesh_issues = detect_mesh_path_issues(preset.urdf_path)
        rep.mesh_issues = [str(m) for m in mesh_issues[:5]]

        try:
            model = load_robot(preset, compile_mjcf=True)
            rep.load_ok = True
            rep.mjcf_ok = model.mujoco_model is not None
            rep.num_dof = len(model.actuated_joints)
            rep.ik_slots = len(preset.ik_map)
        except Exception as err:
            model = load_robot(preset, compile_mjcf=False)
            rep.load_ok = True
            rep.mjcf_ok = False
            rep.num_dof = len(model.actuated_joints)
            rep.ik_slots = len(preset.ik_map)
            rep.error = f"MJCF: {err}"

        issues = validate_ik_map(preset.urdf_path, preset.ik_map)
        rep.ik_critical = [
            i.format() for i in issues
            if i.slot in CRITICAL_IK_SLOTS
            or i.slot.endswith("_knee")
            or "is shared with" in i.message
        ]
        if rep.ik_critical:
            repaired, _ = prepare_ik_map(preset.urdf_path, dict(preset.ik_map))
            from hhtools.robot.kinematics import infer_smooth_joint_filter_masks
            from hhtools.robot.yaml_io import (
                update_robot_yaml_ik_map,
                update_robot_yaml_smooth_joint_filter_masks,
            )

            yaml_file = Path(preset.meta.get("yaml_path", preset.urdf_path.parent / "robot.yaml"))
            update_robot_yaml_ik_map(yaml_file, repaired)
            smooth_masks = infer_smooth_joint_filter_masks(preset.urdf_path, repaired)
            if smooth_masks:
                update_robot_yaml_smooth_joint_filter_masks(yaml_file, smooth_masks)
            refresh()
            preset = get(name)
            model = load_robot(preset, compile_mjcf=False)
            issues2 = validate_ik_map(preset.urdf_path, preset.ik_map)
            rep.ik_critical = [
                i.format() for i in issues2
                if i.slot in CRITICAL_IK_SLOTS
                or i.slot.endswith("_knee")
                or "is shared with" in i.message
            ]

        from hhtools.retarget.calibration import (
            RobotRetargetCalibration,
            calibration_path_for,
            derive_calibration_params,
            save_calibration,
        )

        joint_q = model.zero_configuration()
        for ref in REFERENCES:
            try:
                cal = RobotRetargetCalibration(
                    robot=name, reference=ref, calibrated_joint_q=joint_q,
                    notes="batch audit zero-pose",
                )
                derived = derive_calibration_params(cal, model)
                path = calibration_path_for(preset.urdf_path.parent, reference=ref)
                save_calibration(cal, path, derived=derived)
                saved_calibrations[ref] = cal
                rep.calib[ref] = "ok"
            except Exception as err:
                rep.calib[ref] = str(err)[:120]

        if do_retarget and TEST_BVH.is_file() and rep.mjcf_ok:
            from hhtools.io import load_motion
            from hhtools.retarget.newton_basic import NewtonBasicPipeline, PipelineConfig
            from hhtools.robot.retarget_profile import resolve_retarget_scaler_config

            motion = _truncate_motion(load_motion(TEST_BVH))
            refresh()
            preset = get(name)
            model = load_robot(preset, compile_mjcf=False)
            for ref in RETARGET_REFS:
                try:
                    cal = saved_calibrations.get(ref)
                    if cal is None:
                        rep.retarget[ref] = "no calibration"
                        continue
                    scaler_cfg = resolve_retarget_scaler_config(
                        preset, ref, calibration=cal, model=model, motion=motion, human_height=1.65,
                    )
                    pipe = NewtonBasicPipeline(
                        model,
                        scaler_config=scaler_cfg,
                        pipeline_config=PipelineConfig(ik_iterations=8),
                        human_height=1.65,
                        configure_warp=False,
                    )
                    out = pipe.run(motion)
                    rep.retarget[ref] = f"ok F={out.num_frames}"
                except Exception as err:
                    rep.retarget[ref] = str(err)[:120]
    except Exception as err:
        rep.error = traceback.format_exc()[-400:]
    return rep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--import", dest="do_import", action="store_true")
    parser.add_argument("--retarget", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--roots",
        type=str,
        default=os.environ.get("HHTOOLS_ROBOT_AUDIT_ROOTS", ""),
        help="Colon-separated asset roots (or set HHTOOLS_ROBOT_AUDIT_ROOTS)",
    )
    args = parser.parse_args()

    raw_roots = args.roots.strip() or os.environ.get("HHTOOLS_ROBOT_AUDIT_ROOTS", "").strip()
    if not raw_roots:
        print(
            "No asset roots configured. Set HHTOOLS_ROBOT_AUDIT_ROOTS "
            "or pass --roots path1:path2",
            file=sys.stderr,
        )
        return 2
    roots = tuple(Path(p).expanduser() for p in raw_roots.split(":") if p.strip())

    candidates = discover_urdf_candidates(roots)
    if not candidates:
        print("No URDF candidates found under:", ", ".join(str(r) for r in roots))
        return 1

    reports: list[RobotReport] = []
    for name, urdf in candidates:
        print(f"=== {name} ===", flush=True)
        rep = _audit_one(name, urdf, do_import=args.do_import, do_retarget=args.retarget)
        reports.append(rep)
        ik_ok = not rep.ik_critical
        cal_ok = all(v == "ok" for v in rep.calib.values())
        rt_ok = all(v.startswith("ok") for v in rep.retarget.values()) if rep.retarget else True
        status = "OK" if rep.load_ok and ik_ok and cal_ok and rt_ok else "FAIL"
        print(
            f"  {status} load={rep.load_ok} mjcf={rep.mjcf_ok} dof={rep.num_dof} "
            f"ik_crit={len(rep.ik_critical)} calib={sum(v=='ok' for v in rep.calib.values())}/"
            f"{len(rep.calib)} retarget={sum(v.startswith('ok') for v in rep.retarget.values())}/"
            f"{len(rep.retarget)}",
            flush=True,
        )
        if rep.error:
            print(f"  err: {rep.error[:200]}")
        if rep.ik_critical:
            for line in rep.ik_critical[:3]:
                print(f"  ik: {line}")
        if rep.retarget:
            bad_rt = {k: v for k, v in rep.retarget.items() if not v.startswith("ok")}
            for ref, msg in list(bad_rt.items())[:2]:
                print(f"  retarget[{ref}]: {msg[:100]}")

    out = [asdict(r) for r in reports]
    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    fails = sum(
        1 for r in reports
        if not r.load_ok
        or r.ik_critical
        or any(v != "ok" for v in r.calib.values())
        or (r.retarget and any(not v.startswith("ok") for v in r.retarget.values()))
    )
    print(f"\nSummary: {len(reports) - fails}/{len(reports)} pass (load+ik+calib+retarget)")
    print(f"Discovered {len(candidates)} URDF(s) under {len(roots)} root(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
