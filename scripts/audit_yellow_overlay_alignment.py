#!/usr/bin/env python3
"""Audit yellow skeleton overlay vs robot mesh alignment.

Measures per (robot, motion, reference):
  - uniform overlay scale ratio
  - pelvis z_correction (soma scaler vs uniform overlay)
  - yellow foot / head Z after overlay pipeline (frame 0)
  - robot foot / head Z at zero-pose and after retarget frame 0
  - stature ratio (robot standing height vs scaled human stature)
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ROBOTS = ("ultron_evt_v2_0", "t1_29dof", "rp1", "g1_custom_collision_29dof")

MOTION_SAMPLES: list[tuple[str, Path, str]] = [
    ("lafan_bvh", REPO / "assets/motions/mimic/LAFAN/dance1_subject2.bvh", "LAFAN dance"),
    ("lafan_bvh", REPO / "assets/motions/mimic/SOMA/big_light_one_hand_pick_up_front_low_R_005__A508.bvh", "SOMA pick"),
]

# Extra formats outside mimic/ (repo samples)
EXTRA = [
    ("lafan_bvh", REPO / "assets/motions/intermimic/OMOMO/sub12_woodchair_000/sub12_woodchair_000.pkl", "OMOMO chair"),
]
for p in (REPO / "assets/motions/meshmimic/holosoma").glob("parkour_*/clip.npy"):
    EXTRA.append(("lafan_bvh", p, f"holosoma {p.parent.name}"))


def _load_motion(path: Path):
    from hhtools.io import load_motion

    return load_motion(path)


def _robot_foot_head_z(model, joint_q: dict[str, float], ik_map: dict[str, str]):
    from hhtools.retarget.calibration.calibration import _collect_link_transforms_at_q
    from hhtools.robot.standing_height import _trimesh_scene_z_bounds

    saved = model.zero_configuration()
    try:
        model.apply_configuration(joint_q)
        link_T = _collect_link_transforms_at_q(model, joint_q)
        foot_zs: list[float] = []
        head_z = None
        for slot, link in ik_map.items():
            T = link_T.get(link)
            if T is None:
                continue
            z = float(T[2, 3])
            if slot in ("left_ankle", "right_ankle", "left_foot", "right_foot"):
                foot_zs.append(z)
            if slot == "head":
                head_z = z
        mesh_min = mesh_max = None
        try:
            b = _trimesh_scene_z_bounds(model.trimesh_scene(collision=False))
            if b is not None:
                mesh_min, mesh_max = b
        except Exception:
            pass
        return {
            "ankle_min_z": min(foot_zs) if foot_zs else None,
            "head_z": head_z,
            "mesh_min_z": mesh_min,
            "mesh_max_z": mesh_max,
            "mesh_height": (mesh_max - mesh_min) if mesh_min is not None else None,
        }
    finally:
        model.apply_configuration(saved)


def _yellow_overlay_metrics(motion, model, reference: str, human_h: float = 1.65):
    from hhtools.core.grounding import (
        foot_floor_z_in_positions,
        human_source_floor_z_world,
        preferred_floor_contact_bone_indices,
    )
    from hhtools.retarget.calibration import load_calibration, resolve_calibration_file
    from hhtools.retarget.calibration.calibration import (
        uniform_overlay_scale_for_motion,
    )
    from hhtools.retarget.newton_basic.scaler import HumanToRobotScaler
    from hhtools.robot.retarget_profile import resolve_retarget_scaler_config
    from hhtools.robot.standing_height import estimate_robot_standing_height
    from hhtools.web.scaled_preview import (
        _uniform_overlay_z_correction,
        _uniform_scaled_joint_positions,
        resolve_scaled_overlay_z_correction,
    )

    preset = model.preset
    cal_path = resolve_calibration_file(preset.urdf_path.parent, reference)
    cal = load_calibration(cal_path) if cal_path else None
    scaler_cfg = resolve_retarget_scaler_config(
        preset, reference, calibration=cal, model=model, motion=motion, human_height=human_h,
    )
    ik_canons = frozenset(preset.ik_map.keys())
    ratio = float(uniform_overlay_scale_for_motion(
        scaler_cfg, human_h, motion, ik_map_keys=ik_canons,
    ))
    scaler = HumanToRobotScaler(motion.hierarchy, scaler_cfg, human_height=human_h)
    z_corr = float(resolve_scaled_overlay_z_correction(motion, scaler, ratio))
    jn = list(scaler.joint_names)
    pos = _uniform_scaled_joint_positions(
        motion, scaler_cfg, human_h, jn, ik_canons=ik_canons, z_correction=z_corr,
    )
    pos0 = pos[0]

    bone_names = motion.hierarchy.bone_names
    foot_i = preferred_floor_contact_bone_indices(bone_names)
    hname_to_j = {n: i for i, n in enumerate(jn)}
    yellow_foot_zs: list[float] = []
    for hi in foot_i:
        bn = bone_names[int(hi)]
        if bn in hname_to_j:
            yellow_foot_zs.append(float(pos0[hname_to_j[bn], 2]))

    src_foot_z = foot_floor_z_in_positions(motion.positions[0], bone_names)
    z_floor = human_source_floor_z_world(motion)

    # Source stature at frame 0 (foot floor → max joint)
    src_rel = motion.positions[0, :, 2] - z_floor
    src_stature = float(src_rel.max())

    robot_h = estimate_robot_standing_height(model, model.zero_configuration())
    scaled_stature = src_stature * ratio

    root_name = str(scaler_cfg.root_joint)
    pelvis_uniform_z = pelvis_scaler_z = None
    try:
        hi = bone_names.index(root_name)
        pelvis_uniform_z = float((motion.positions[0, hi, 2] - z_floor) * ratio)
        j_root = jn.index(root_name)
        eff = scaler.apply(dataclasses.replace(
            motion, positions=motion.positions[:1], quaternions=motion.quaternions[:1],
        ))
        pelvis_scaler_z = float(eff.transforms[0, j_root, 2])
    except (ValueError, IndexError):
        pass

    return {
        "ratio": ratio,
        "robot_height_cfg": float(scaler_cfg.model_height),
        "robot_height_mesh": robot_h,
        "human_height": human_h,
        "z_correction": z_corr,
        "pelvis_uniform_z": pelvis_uniform_z,
        "pelvis_scaler_z": pelvis_scaler_z,
        "yellow_foot_min_z": min(yellow_foot_zs) if yellow_foot_zs else None,
        "yellow_max_z": float(pos0[:, 2].max()),
        "src_stature_m": src_stature,
        "scaled_stature_m": scaled_stature,
        "stature_ratio_robot_over_scaled": robot_h / max(1e-6, scaled_stature),
        "src_foot_z_world": src_foot_z,
        "z_floor_world": z_floor,
        "z_corr_without_pelvis": float(
            _uniform_overlay_z_correction(motion, scaler, ratio)
        ),
    }


def _retarget_frame0_foot_z(motion, model, reference: str, human_h: float = 1.65):
    from hhtools.retarget.calibration import load_calibration, resolve_calibration_file
    from hhtools.retarget.newton_basic import NewtonBasicPipeline, PipelineConfig
    from hhtools.robot.retarget_profile import resolve_retarget_scaler_config

    preset = model.preset
    cal_path = resolve_calibration_file(preset.urdf_path.parent, reference)
    if cal_path is None:
        return None
    cal = load_calibration(cal_path)
    scaler_cfg = resolve_retarget_scaler_config(
        preset, reference, calibration=cal, model=model, motion=motion, human_height=human_h,
    )
    clip = dataclasses.replace(
        motion,
        positions=motion.positions[: min(30, motion.num_frames)],
        quaternions=motion.quaternions[: min(30, motion.num_frames)],
    )
    pipe = NewtonBasicPipeline(
        model, scaler_config=scaler_cfg,
        pipeline_config=PipelineConfig(ik_iterations=8),
        human_height=human_h, configure_warp=False,
    )
    out = pipe.run(clip)
    root_z = float(out.root_trajectory[0, 2]) if out.root_trajectory is not None else None
    rq = {n: float(out.joint_q[0, i]) for i, n in enumerate(out.dof_names)}
    rz = _robot_foot_head_z(model, rq, dict(preset.ik_map))
    rz["root_z"] = root_z
    return rz


def main() -> int:
    from hhtools.robot.registry import get, refresh
    from hhtools.robot.loader import load_robot

    refresh()
    samples = MOTION_SAMPLES + EXTRA
    print("=" * 100)
    print("YELLOW OVERLAY vs ROBOT ALIGNMENT AUDIT")
    print("=" * 100)

    for robot_name in ROBOTS:
        try:
            preset = get(robot_name)
            model = load_robot(preset, compile_mjcf=False)
        except Exception as exc:
            print(f"\n[{robot_name}] SKIP: {exc}")
            continue

        zero_q = model.zero_configuration()
        zero_z = _robot_foot_head_z(model, zero_q, dict(preset.ik_map))
        print(f"\n{'─' * 100}")
        print(f"ROBOT: {robot_name}")
        print(f"  zero-pose mesh z=[{zero_z['mesh_min_z']}, {zero_z['mesh_max_z']}] h={zero_z['mesh_height']:.3f}m")
        print(f"  zero-pose ankle_min_z={zero_z['ankle_min_z']}")

        for ref, path, label in samples:
            if not path.is_file():
                continue
            try:
                motion = _load_motion(path)
            except Exception as exc:
                print(f"\n  [{label}] load failed: {exc}")
                continue

            try:
                ym = _yellow_overlay_metrics(motion, model, ref)
            except Exception as exc:
                print(f"\n  [{label}] overlay metrics failed: {exc}")
                continue

            foot_gap = (
                ym["yellow_foot_min_z"]
                if ym["yellow_foot_min_z"] is not None
                else float("nan")
            )
            flag = ""
            if ym["yellow_foot_min_z"] is not None and ym["yellow_foot_min_z"] < -0.02:
                flag = " ⚠ FEET BELOW GROUND"
            elif ym["yellow_foot_min_z"] is not None and ym["yellow_foot_min_z"] > 0.05:
                flag = " ⚠ FEET FLOATING"

            print(f"\n  [{label}] ({path.name})")
            print(f"    scale ratio={ym['ratio']:.4f}  robot_h(cfg/mesh)={ym['robot_height_cfg']:.3f}/{ym['robot_height_mesh']:.3f}  human={ym['human_height']:.2f}")
            print(f"    z_correction(pelvis)={ym['z_correction']:+.4f}m  yellow_foot_z={foot_gap:+.4f}m{flag}")
            print(f"    stature: src={ym['src_stature_m']:.3f} → scaled={ym['scaled_stature_m']:.3f}  robot/scaled={ym['stature_ratio_robot_over_scaled']:.3f}")
            print(f"    pelvis uniform={ym['pelvis_uniform_z']}  scaler={ym['pelvis_scaler_z']}")

            try:
                rt = _retarget_frame0_foot_z(motion, model, ref)
                if rt:
                    print(f"    retarget f0: root_z={rt.get('root_z')}  ankle_min={rt.get('ankle_min_z')}  mesh_min={rt.get('mesh_min_z')}")
            except Exception as exc:
                print(f"    retarget f0: failed ({exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
