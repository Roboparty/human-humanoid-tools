"""Preview payload helpers used by the Web application factory."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def _compute_r2r_scaled_preview(
    source_model,
    target_model,
    motion,
    calibrated_joint_q,
) -> dict:
    """Yellow scaled skeleton for R2R - uniform source-to-target scale."""
    from hhtools.analysis.scaled_preview import (
        _uniform_scaled_preview_fallback,
        resolve_scaled_overlay_z_correction,
    )
    from hhtools.retarget import robot_to_robot as r2r
    from hhtools.retarget.calibration.calibration import uniform_overlay_scale_for_motion
    from hhtools.retarget.newton_basic.scaler import HumanToRobotScaler

    cfg, ref = r2r._build_scaler_config(source_model, target_model, calibrated_joint_q)
    human_height = float(ref.height_m)
    ik_canons = (
        frozenset(target_model.preset.ik_map.keys()) if target_model.preset.ik_map else frozenset()
    )
    scaler = HumanToRobotScaler(motion.hierarchy, cfg, human_height=human_height)
    ratio = float(
        uniform_overlay_scale_for_motion(
            cfg,
            human_height,
            motion,
            ik_map_keys=ik_canons,
        )
    )
    z_correction = resolve_scaled_overlay_z_correction(motion, scaler, ratio)
    # Keep R2R ankles at their source mesh-sole-relative height.
    return _uniform_scaled_preview_fallback(
        motion,
        cfg,
        human_height,
        ik_canons,
        z_correction=z_correction,
    )


def _align_scaled_preview_to_robot_playback(
    target_model,
    retargeted,
    scaled_preview: dict,
    trajectory: dict,
) -> dict:
    """Shift yellow overlay Z to the grounded robot sole at playback frame 0."""
    import numpy as np

    from hhtools.io.scene_serialize import _scaled_overlay_foot_z
    from hhtools.robot.foot_geometry import quat_xyzw_to_rotmat, scene_min_mesh_z

    yellow_z = _scaled_overlay_foot_z(scaled_preview, 0)
    if yellow_z is None:
        return _ground_skeleton_preview(scaled_preview)

    frames = trajectory.get("frames") or []
    if not frames:
        return _ground_skeleton_preview(scaled_preview)

    idx = trajectory.get("frame_indices") or [0]
    f0 = int(idx[0]) if idx else 0
    root = np.asarray(retargeted.root_trajectory[f0], dtype=np.float64)
    mesh_lift = float(frames[0].get("mesh_z_lift") or 0.0)
    ret_dof_names = list(retargeted.dof_names)
    dof0 = np.asarray(retargeted.dof_trajectory[f0], dtype=np.float64)
    cfg0 = {ret_dof_names[i]: float(dof0[i]) for i in range(len(ret_dof_names))}
    target_model.apply_configuration(cfg0)
    root_rot = quat_xyzw_to_rotmat(root[3:7])
    min_mesh_z = scene_min_mesh_z(target_model.trimesh_scene(), root_rot)
    robot_ref_z = float(root[2] + mesh_lift + min_mesh_z) if min_mesh_z is not None else 0.0

    dz = robot_ref_z - float(yellow_z)
    if abs(dz) < 1e-5:
        return scaled_preview

    positions = np.asarray(scaled_preview["positions"], dtype=np.float32).copy()
    positions[:, :, 2] += np.float32(dz)
    out = dict(scaled_preview)
    out["positions"] = np.round(positions, 4).tolist()
    return out


def _ground_skeleton_preview(payload: dict) -> dict:
    """Shift skeleton positions so the clip-wide lowest joint rests on z=0."""
    import numpy as np

    from hhtools.core.grounding import clip_floor_z_in_positions

    positions = np.asarray(payload.get("positions") or [], dtype=np.float32)
    if positions.size == 0:
        return payload
    z_ref = float(clip_floor_z_in_positions(positions))
    positions = positions.copy()
    positions[:, :, 2] -= np.float32(z_ref)
    out = dict(payload)
    out["positions"] = np.round(positions, 4).tolist()
    return out


def _compute_scaled_scene(
    model,
    robot_name: str,
    motion,
    reference: str,
    human_height: float,
    *,
    max_frames: int | None = None,
) -> dict | None:
    """Scale terrain and objects into the robot retarget frame."""
    import numpy as np

    if motion.terrain is None and not motion.objects:
        return None
    from hhtools.analysis.scaled_preview import (
        resolve_scaled_overlay_z_correction,
        resolve_web_scaler_config,
    )
    from hhtools.core.grounding import (
        human_source_floor_z_world,
        terrain_heightfield_z_offset_world,
    )
    from hhtools.core.scene import SceneObject
    from hhtools.io.scene_serialize import (
        _MAX_PLAYBACK_FRAMES,
        _downsample_indices,
        _serialize_object_meta,
        _serialize_terrain,
    )
    from hhtools.retarget.calibration.calibration import uniform_overlay_scale_for_motion
    from hhtools.retarget.newton_basic.scaler import HumanToRobotScaler

    try:
        scaler_cfg = resolve_web_scaler_config(
            model,
            motion,
            reference,
            float(human_height),
        )
    except ValueError:
        return None
    scaler = HumanToRobotScaler(
        motion.hierarchy,
        scaler_cfg,
        human_height=float(human_height),
    )
    from hhtools.robot.ik_map_policy import ik_map_canonicals_for_motion

    ik_canons = ik_map_canonicals_for_motion(
        model.preset.name,
        model.preset.ik_map,
        motion,
    )
    ratio = float(
        uniform_overlay_scale_for_motion(
            scaler_cfg,
            float(human_height),
            motion,
            ik_map_keys=ik_canons,
        )
    )

    z_min = float(human_source_floor_z_world(motion))
    z_terrain = float(terrain_heightfield_z_offset_world(motion, z_min))
    z_correction = float(resolve_scaled_overlay_z_correction(motion, scaler, ratio))
    idx = _downsample_indices(
        motion.num_frames,
        _MAX_PLAYBACK_FRAMES if max_frames is None else max_frames,
        motion=motion,
    )

    payload: dict = {"scale_ratio": round(ratio, 5), "objects": [], "terrain": None}
    for i, ob in enumerate(motion.objects):
        op = ob.positions.astype(np.float32, copy=True)
        op[:, 2] -= z_min
        op *= ratio
        if abs(z_correction) > 1e-6:
            op[:, 2] += np.float32(z_correction)
        scaled_ob = SceneObject(
            name=f"scaled_{ob.name}",
            positions=op,
            quaternions=ob.quaternions.copy(),
            extents=ob.extents * ratio,
            mesh_path=ob.mesh_path,
            scale=ob.scale * ratio,
            opacity=ob.opacity,
            color=ob.color,
        )
        meta = _serialize_object_meta(scaled_ob, idx)
        meta["source_index"] = i
        meta["source_scale"] = float(ob.scale)
        payload["objects"].append(meta)

    if motion.terrain is not None:
        hf_robot = motion.terrain.scaled(ratio, z_offset=z_terrain)
        if abs(z_correction) > 1e-6:
            hf_robot = hf_robot.shifted(dz=z_correction)
        if isinstance(motion.meta, dict) and motion.meta.get("dataset") == "parc_ms":
            from hhtools.io.parc_ms_skeleton import PARC_MS_FOOT_CONTACT_OFFSET_M

            hf_robot = hf_robot.shifted(
                dz=float(PARC_MS_FOOT_CONTACT_OFFSET_M) * ratio,
            )
        payload["terrain"] = _serialize_terrain(hf_robot)
    return payload


def _compute_scaled_preview(
    model,
    robot_name: str,
    motion,
    reference: str,
    human_height: float,
    *,
    max_frames: int = 0,
) -> dict:
    """Dense uniform scaled skeleton for the browser."""
    from hhtools.analysis.scaled_preview import compute_web_scaled_preview

    return compute_web_scaled_preview(
        model,
        motion,
        reference,
        human_height,
        max_frames=max_frames,
    )


__all__ = [
    "_align_scaled_preview_to_robot_playback",
    "_compute_r2r_scaled_preview",
    "_compute_scaled_preview",
    "_compute_scaled_scene",
    "_ground_skeleton_preview",
]


def _run_dataset_robot_preview_job(job: Any, body: dict[str, Any], state: Any) -> None:
    from hhtools.services.motion_progress import MotionLoadProgress

    try:
        source_path = Path(str(body["source_path"]))
        robot_name = body.get("robot") or None
        load_prog = MotionLoadProgress(job, base=0.05, span=0.9)
        job.message = "读取机器人轨迹…"
        result = _build_robot_export_playback(
            source_path,
            state,
            robot_name=str(robot_name) if robot_name else None,
            progress=load_prog,
        )
        job.result = result
        job.progress = 1.0
        job.message = "完成"
        job.mark_terminal("done")
    except Exception as err:  # noqa: BLE001
        _log.exception("dataset robot preview failed")
        job.error = str(err)
        job.mark_terminal("error")


def _ensure_robot_model(state: Any, robot_name: str | None):
    """Load a robot preset for FK preview (from CSV meta or G1 default)."""
    from hhtools.robot.loader import load_robot
    from hhtools.robot.registry import get as get_preset
    from hhtools.robot.registry import refresh

    refresh()
    candidates = [robot_name, "unitree_g1__g1_29dof", "unitree_g1"]
    for name in candidates:
        if not name:
            continue
        cached = state.robots.get(name)
        if cached is not None:
            return cached
        try:
            preset = get_preset(name)
            model = load_robot(preset, compile_mjcf=False)
            state.robots[preset.name] = model
            return model
        except Exception as err:  # noqa: BLE001
            _log.debug("robot preset %r unavailable: %s", name, err)
    raise ValueError("无法加载机器人模型以预览轨迹；请先在「机器人」面板加载对应机器人")


def _build_robot_export_playback(
    source_path: Path,
    state: Any,
    *,
    robot_name: str | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Parse a robot export CSV and build mesh playback + optional scene payload."""
    from hhtools.io.r2r_scene import _parse_comment_meta, load_r2r_clip_scene
    from hhtools.io.scene_serialize import serialize_robot_trajectory
    from hhtools.retarget import robot_to_robot as r2r
    from hhtools.retarget.interaction_mesh.heightfield import obj_to_heightfield
    from hhtools.services.r2r_upload_resolve import detect_r2r_profile

    path = Path(source_path).resolve()
    clip_dir = path.parent
    inferred = str(_parse_comment_meta(path).get("robot") or "").strip()
    pick = robot_name or inferred or None
    model = _ensure_robot_model(state, pick)
    actual = model.preset.name

    cb = progress.as_callback() if progress is not None else None
    if cb is not None:
        cb(0.12, f"读取 {path.name}…")

    traj = r2r.load_source_trajectory(path, source_model=model)
    if cb is not None:
        cb(0.45, "生成机器人播放轨迹…")
    num_frames = int(traj.joint_q.shape[0])
    framerate = float(traj.framerate)
    prof = detect_r2r_profile(clip_dir)
    scaled_scene = load_r2r_clip_scene(
        clip_dir,
        profile=prof,
        robot_path=path,
        num_frames=num_frames,
        framerate=framerate,
        terrain_loader=obj_to_heightfield,
    )
    ret_play = r2r.trajectory_to_retargeted_motion(model, traj, name=path.stem)
    playback = serialize_robot_trajectory(
        model,
        ret_play,
        preserve_absolute_z=bool(scaled_scene and scaled_scene.get("terrain")),
    )

    preview_token = uuid.uuid4().hex[:10]
    state.dataset_previews[preview_token] = {
        "clip_dir": str(clip_dir),
        "source_path": str(path),
    }

    if cb is not None:
        cb(1.0, "就绪")

    return {
        "preview_token": preview_token,
        "trajectory": playback,
        "robot": actual,
        "inferred_robot": inferred or actual,
        "num_frames": num_frames,
        "framerate": framerate,
        "has_scene": bool(scaled_scene),
        "scaled_scene": scaled_scene,
        "name": path.stem,
    }


def _load_robot_export_for_web(
    source_path: Path,
    state: Any,
    *,
    progress: Any = None,
):
    """FK a retarget robot CSV export into a :class:`Motion` for 3D preview."""
    from hhtools.io.r2r_scene import (
        _parse_comment_meta,
        attach_r2r_clip_scene_to_motion,
    )
    from hhtools.retarget import robot_to_robot as r2r
    from hhtools.retarget.interaction_mesh.heightfield import obj_to_heightfield
    from hhtools.services.r2r_upload_resolve import detect_r2r_profile

    path = Path(source_path).resolve()
    clip_dir = path.parent
    cb = progress.as_callback() if progress is not None else None
    if cb is not None:
        cb(0.05, f"读取机器人轨迹 {path.name}…")

    robot_name = str(_parse_comment_meta(path).get("robot") or "").strip()
    model = _ensure_robot_model(state, robot_name or None)
    traj = r2r.load_source_trajectory(path, source_model=model)

    def _fk_progress(done: int, total: int) -> None:
        if cb is not None:
            cb(0.15 + 0.55 * (done / max(1, total)), f"正运动学 {done}/{total}")

    motion = r2r.source_trajectory_to_motion(
        model,
        traj.joint_q,
        traj.dof_names,
        framerate=traj.framerate,
        name=path.stem,
        progress_callback=_fk_progress if cb is not None else None,
    )

    prof = detect_r2r_profile(clip_dir)
    try:
        motion = attach_r2r_clip_scene_to_motion(
            motion,
            clip_dir,
            profile=prof,
            robot_path=path,
            terrain_loader=obj_to_heightfield,
        )
    except Exception as err:  # noqa: BLE001 - scene is optional for preview
        _log.warning("robot export scene attach skipped for %s: %s", path, err)

    if cb is not None:
        cb(1.0, "机器人轨迹就绪")
    return motion
