"""Robot-to-robot Web workflow helpers."""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from hhtools.application.export import (
    _batch_export_subdir,
    _parse_csv_header,
    _parse_optional_fps,
    _parse_optional_time,
    _write_r2r_export,
)
from hhtools.application.motions import _motion_for_retarget
from hhtools.application.previews import _ground_skeleton_preview
from hhtools.application.progress import (
    _set_batch_job_progress,
    _set_retarget_job_clip_progress,
)
from hhtools.application.robots import (
    _join_robot_prewarm,
    _require_newton_package,
)
from hhtools.application.state import _snapshot_job_request

if TYPE_CHECKING:
    from hhtools.application.state import Job, SessionState

_log = logging.getLogger(__name__)


def _build_r2r_calibration_session(target_model, source_model) -> dict:
    """Build the calibration payload for aligning a target to a source robot."""
    from hhtools.retarget import robot_to_robot as r2r
    from hhtools.web.analysis.calibration_session import (
        _joint_limits_payload,
        _reference_heading_rad,
        _robot_ground_offset_z,
        joint_world_payload,
        serialize_reference_skeleton,
    )

    joint_order = [
        joint.name for joint in target_model.actuated_joints if joint.joint_type != "fixed"
    ]
    if not joint_order:
        raise ValueError("target robot has no actuated joints; check URDF / upload")
    joint_q = {name: 0.0 for name in joint_order}

    saved: dict[str, float] | None = None
    urdf_path = getattr(target_model.preset, "urdf_path", None)
    if urdf_path is not None:
        saved = r2r.load_r2r_calibration(
            urdf_path.parent,
            source_model.preset.name,
            target_robot=target_model.preset.name,
        )
        if saved:
            for name, value in saved.items():
                if name in joint_q:
                    joint_q[name] = float(value)

    ref = r2r.build_source_reference_pose(source_model)
    target_model.apply_configuration(joint_q)
    ground_z = _robot_ground_offset_z(target_model, joint_q)
    try:
        heading = _reference_heading_rad(
            target_model,
            ref,
            None,
            ref.name,
            current_q=joint_q,
        )
    except Exception:  # noqa: BLE001
        heading = 0.0
    ref_payload = serialize_reference_skeleton(ref, heading_rad=heading)
    return {
        "joint_q": joint_q,
        "joint_limits": _joint_limits_payload(target_model),
        "joint_world": joint_world_payload(target_model),
        "reference": ref_payload,
        "reference_name": ref.name,
        "ground_offset_z": ground_z,
        "has_saved_calibration": bool(saved),
    }


def _run_r2r_source_upload_job(
    job: Job,
    drop: Path,
    source_robot: str,
    profile: str,
    state: SessionState,
    source_fps: float | None = None,
    selected_path: Path | None = None,
) -> None:
    from hhtools.io.r2r_export_bundle import clip_has_export_scene
    from hhtools.io.scene_serialize import (
        serialize_motion_skeleton_preview,
        serialize_robot_trajectory,
    )
    from hhtools.retarget import robot_to_robot as r2r
    from hhtools.services.r2r_upload_resolve import (
        detect_r2r_profile,
        enumerate_r2r_clips,
        r2r_clip_ref_for_path,
        validate_r2r_upload,
    )

    try:
        job.progress = 0.02
        job.message = "正在识别轨迹格式…"
        if selected_path is not None:
            clip_ref = r2r_clip_ref_for_path(selected_path, profile)
            prof = clip_ref.profile
        else:
            validate_r2r_upload(drop, profile)
            prof = (profile or "auto").strip().lower()
            if prof == "auto":
                prof = detect_r2r_profile(drop)
            clips = enumerate_r2r_clips(drop, prof)
            if not clips:
                raise ValueError("no robot trajectory clip found under upload")
            clip_ref = clips[0]
        picked = clip_ref.path
        stem = picked.stem
        clip_dir = picked.parent
        scene_prof = clip_ref.profile or prof

        job.progress = 0.08
        job.message = "正在读取轨迹文件…"
        src_model = state.robots.get(source_robot)
        if src_model is None:
            from hhtools.robot.loader import load_robot
            from hhtools.robot.registry import get as get_preset

            src_model = load_robot(get_preset(source_robot), compile_mjcf=False)
            state.robots[source_robot] = src_model
        traj = r2r.load_source_trajectory(
            picked,
            source_model=src_model,
            source_fps=source_fps,
        )

        def _fk_cb(done: int, total: int) -> None:
            job.progress = 0.1 + 0.55 * (done / max(1, total))
            job.message = f"正运动学还原关键点 {done}/{total}"

        job.message = "正运动学还原关键点…"
        motion = r2r.source_trajectory_to_motion(
            src_model,
            traj.joint_q,
            traj.dof_names,
            framerate=traj.framerate,
            name=stem,
            progress_callback=_fk_cb,
        )

        job.progress = 0.72
        job.message = "正在生成机器人播放轨迹…"
        scaled_scene = None
        src_has_scene = clip_ref.has_scene or clip_has_export_scene(
            clip_dir,
            stem=stem,
            profile=scene_prof,
        )
        if src_has_scene:
            job.progress = 0.88
            job.message = "正在加载地形/物体…"
            from hhtools.io.r2r_scene import load_r2r_clip_scene
            from hhtools.retarget.interaction_mesh.heightfield import obj_to_heightfield

            scaled_scene = load_r2r_clip_scene(
                clip_dir,
                profile=scene_prof,
                robot_path=picked,
                num_frames=int(traj.joint_q.shape[0]),
                framerate=float(traj.framerate),
                terrain_loader=obj_to_heightfield,
            )

        job.progress = 0.9
        job.message = "正在生成机器人播放轨迹…"
        ret_play = r2r.trajectory_to_retargeted_motion(src_model, traj, name=stem)
        playback = serialize_robot_trajectory(
            src_model,
            ret_play,
            preserve_absolute_z=bool(scaled_scene and scaled_scene.get("terrain")),
        )

        job.progress = 0.95
        job.message = "正在生成骨架预览…"
        skel = _ground_skeleton_preview(serialize_motion_skeleton_preview(motion))

        token = uuid.uuid4().hex[:10]
        state.r2r_sources[token] = {
            "source_robot": source_robot,
            "motion": motion,
            "framerate": float(traj.framerate),
            "num_frames": int(traj.joint_q.shape[0]),
            "stem": stem,
            "source_path": str(picked),
            "clip_dir": str(clip_dir),
            "has_scene": bool(src_has_scene),
            "upload_profile": scene_prof,
            "scaled_scene": scaled_scene,
        }
        job.result = {
            "token": token,
            "source_robot": source_robot,
            "num_frames": int(traj.joint_q.shape[0]),
            "framerate": float(traj.framerate),
            "dof_names": list(traj.dof_names),
            "trajectory": playback,
            "skeleton_preview": skel,
            "scaled_scene": scaled_scene,
            "has_scene": bool(src_has_scene),
            "upload_profile": scene_prof,
            "name": stem,
            "suggested_backend": r2r.suggested_r2r_backend(
                scene_prof,
                has_scene=bool(src_has_scene),
            ),
        }
        job.progress = 1.0
        job.message = "done"
        job.mark_terminal("done")
    except Exception as err:  # noqa: BLE001
        _log.exception("r2r source upload job failed")
        job.error = str(err)
        job.mark_terminal("error")


def _r2r_entry_from_upload(drop_dir: Path, ref) -> dict:
    from hhtools.retarget import robot_to_robot as r2r
    from hhtools.services.r2r_upload_resolve import export_subdir_for_r2r_clip

    picked = Path(ref.path).resolve()
    drop_dir = Path(drop_dir).resolve()
    prof = (ref.profile or "mimic").strip().lower()
    folder_by_profile = {
        "intermimic": "intermimic",
        "meshmimic": "meshmimic",
        "mimic": "mimic",
    }
    try:
        rel = picked.relative_to(drop_dir)
        sequence_id = rel.as_posix()
        stem = picked.parent.name if picked.parent.name == picked.stem else picked.stem
    except ValueError:
        sequence_id = picked.name
        stem = picked.stem

    return {
        "dataset": "r2r",
        "asset_kind": "robot_trajectory",
        "folder_label": folder_by_profile.get(prof, "r2r"),
        "sequence_id": sequence_id,
        "source_path": str(picked),
        "clip_dir": str(picked.parent),
        "stem": stem,
        "origin": "upload",
        "export_subdir": export_subdir_for_r2r_clip(drop_dir, picked),
        "upload_profile": prof,
        "clip_kind": ref.clip_kind or "",
        "has_scene": bool(ref.has_scene),
        "upload_drop": str(drop_dir),
        "suggested_backend": r2r.suggested_r2r_backend(
            prof,
            has_scene=bool(ref.has_scene),
        ),
    }


def _run_r2r_basket_upload_job(job: Job, drop: Path, profile: str) -> None:
    from hhtools.services.r2r_upload_resolve import (
        enumerate_r2r_clips,
        validate_r2r_upload,
    )

    try:
        validate_r2r_upload(drop, profile)
        clips = enumerate_r2r_clips(drop, profile)
        entries = [_r2r_entry_from_upload(drop, ref) for ref in clips]
        job.result = {
            "entries": entries,
            "clip_count": len(entries),
            "upload_root": str(drop),
            "profile": profile,
        }
        job.progress = 1.0
        job.message = f"已识别 {len(entries)} 个机器人轨迹 clip"
        job.mark_terminal("done")
    except Exception as err:  # noqa: BLE001
        _log.exception("r2r basket upload failed")
        job.error = str(err)
        job.mark_terminal("error")


def _r2r_prepare_retarget_motion(
    motion,
    *,
    backend: str,
    clip_dir: Path | str | None,
    robot_path: Path | str | None,
    profile: str,
    has_scene: bool,
):
    """Attach terrain/objects when the Interaction-Mesh backend is selected."""
    if (backend or "newton").strip().lower() != "interaction_mesh":
        return motion
    if not has_scene or clip_dir is None or robot_path is None:
        return motion
    from hhtools.io.r2r_scene import attach_r2r_clip_scene_to_motion
    from hhtools.retarget.interaction_mesh.heightfield import obj_to_heightfield

    return attach_r2r_clip_scene_to_motion(
        motion,
        Path(clip_dir),
        profile=profile,
        robot_path=Path(robot_path),
        terrain_loader=obj_to_heightfield,
    )


def _r2r_retarget_progress_cb(
    job: Job | None,
    backend: str,
    *,
    done: int,
    total: int,
) -> None:
    if job is None:
        return
    backend = (backend or "newton").strip().lower()
    if backend == "interaction_mesh":
        if done <= 0:
            _set_retarget_job_clip_progress(
                job,
                0.08,
                "正在构建 Interaction-Mesh 场景…",
            )
        else:
            _set_retarget_job_clip_progress(
                job,
                min(0.98, 0.1 + 0.88 * (done / max(1, total))),
                f"MPC 求解 {done}/{total}",
            )
        return
    if done <= 0:
        _set_retarget_job_clip_progress(job, 0.08, "正在准备逐帧 IK…")
    else:
        _set_retarget_job_clip_progress(
            job,
            min(0.98, 0.1 + 0.88 * (done / max(1, total))),
            f"IK 求解 {done}/{total}",
        )


def _r2r_retarget_from_path(
    source_model,
    target_model,
    traj_path: Path,
    *,
    calibrated_joint_q: dict[str, float],
    retarget_fps: float | None,
    ik_iters: int,
    backend: str = "newton",
    profile: str = "mimic",
    has_scene: bool = False,
    source_fps: float | None = None,
    job: Job | None = None,
):
    from hhtools.retarget import robot_to_robot as r2r

    traj = r2r.load_source_trajectory(
        traj_path,
        source_model=source_model,
        source_fps=source_fps,
    )
    motion_src = r2r.source_trajectory_to_motion(
        source_model,
        traj.joint_q,
        traj.dof_names,
        framerate=traj.framerate,
        name=traj_path.stem,
    )
    motion, _eff_fps = _motion_for_retarget(motion_src, retarget_fps)
    motion = _r2r_prepare_retarget_motion(
        motion,
        backend=backend,
        clip_dir=traj_path.parent,
        robot_path=traj_path,
        profile=profile,
        has_scene=has_scene,
    )

    def _cb(done: int, total: int) -> None:
        _r2r_retarget_progress_cb(job, backend, done=done, total=total)

    ret = r2r.retarget_robot_to_robot(
        source_model,
        target_model,
        calibrated_joint_q=calibrated_joint_q,
        source_motion=motion,
        backend=backend,
        ik_iterations=ik_iters,
        progress_callback=_cb if job is not None else None,
    )
    return ret, motion


def _run_r2r_batch_job(job: Job, body: dict, state: SessionState) -> None:
    try:
        target = body["target"]
        source = body["source"]
        entries = body.get("entries") or []
        if not entries:
            raise ValueError("batch entries list is empty")
        ik_iters = int(body.get("ik_iterations", 24))
        retarget_fps = _parse_optional_fps(body.get("retarget_fps"))
        source_fps = _parse_optional_fps(body.get("source_fps"))
        export_fps = _parse_optional_fps(body.get("export_fps", body.get("fps")))
        export_t_start = _parse_optional_time(body.get("t_start"), name="t_start")
        export_t_end = _parse_optional_time(body.get("t_end"), name="t_end")
        fmt = (body.get("format") or "csv").lower()
        csv_header = _parse_csv_header(body.get("csv_header", True))
        out_name = body.get("out_dir") or "r2r_batch_export"
        backend = (body.get("backend") or "newton").strip().lower()

        job.request = _snapshot_job_request(
            {
                **job.request,
                "target": target,
                "source": source,
                "entries": entries,
                "backend": backend,
                "ik_iterations": ik_iters,
                "retarget_fps": retarget_fps,
                "source_fps": source_fps,
                "export_fps": export_fps,
                "format": fmt,
                "csv_header": csv_header,
                "out_dir": out_name,
            }
        )

        from hhtools.retarget import robot_to_robot as r2r

        tgt = state.robots.get(target)
        if tgt is None:
            from hhtools.robot.loader import load_robot
            from hhtools.robot.registry import get as get_preset

            tgt = load_robot(get_preset(target), compile_mjcf=True)
            state.robots[target] = tgt
        src = state.robots.get(source)
        if src is None:
            from hhtools.robot.loader import load_robot
            from hhtools.robot.registry import get as get_preset

            src = load_robot(get_preset(source), compile_mjcf=False)
            state.robots[source] = src
        calib = r2r.load_r2r_calibration(
            tgt.preset.urdf_path.parent,
            source,
            target_robot=tgt.preset.name,
        )
        if not calib:
            raise ValueError(f"target {target!r} is not calibrated against source {source!r}")

        if backend != "interaction_mesh":
            _require_newton_package()
            _join_robot_prewarm(state, target, job)

        out_dir = state.export_root / f"r2r_batch_{job.id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        errors: list[str] = []
        failures: list[dict] = []
        total = len(entries)
        batch_t0 = time.monotonic()
        _set_batch_job_progress(job, f"R2R 批量开始 · 0/{total}", 0.0, batch_t0)

        for i, entry in enumerate(entries):
            stem = entry.get("stem") or Path(entry.get("source_path", "clip")).stem
            _set_batch_job_progress(
                job,
                f"{i + 1}/{total}: {stem}",
                i / max(1, total),
                batch_t0,
                clip_progress=0.0,
            )
            traj_path = Path(entry["source_path"])
            try:
                ret, motion = _r2r_retarget_from_path(
                    src,
                    tgt,
                    traj_path,
                    calibrated_joint_q=calib,
                    retarget_fps=retarget_fps,
                    ik_iters=ik_iters,
                    backend=backend,
                    profile=str(entry.get("upload_profile") or "mimic"),
                    has_scene=bool(entry.get("has_scene")),
                    source_fps=source_fps,
                    job=job,
                )
            except Exception as err:  # noqa: BLE001
                errors.append(f"{stem}: {err}")
                failures.append({"stem": stem, "stage": "retarget", "reason": str(err)})
                _set_batch_job_progress(
                    job,
                    f"失败 {stem} · {i + 1}/{total}",
                    (i + 1) / max(1, total),
                    batch_t0,
                    clip_progress=1.0,
                )
                continue
            try:
                _set_batch_job_progress(
                    job,
                    f"导出 {stem} · {i + 1}/{total}",
                    i / max(1, total),
                    batch_t0,
                    clip_progress=0.99,
                )
                subdir = _batch_export_subdir(entry)
                out_path = _write_r2r_export(
                    ret,
                    tgt,
                    motion,
                    out_dir,
                    source_model=src,
                    calibrated_joint_q=calib,
                    entry=entry,
                    stem=stem,
                    fps=export_fps,
                    fmt=fmt,
                    subdir=subdir,
                    csv_header=csv_header,
                    t_start=export_t_start,
                    t_end=export_t_end,
                )
                written.append(str(out_path.relative_to(out_dir)))
            except Exception as err:  # noqa: BLE001
                errors.append(f"{stem} export: {err}")
                failures.append({"stem": stem, "stage": "export", "reason": str(err)})
            _set_batch_job_progress(
                job,
                f"完成 {stem} · {i + 1}/{total}",
                (i + 1) / max(1, total),
                batch_t0,
                clip_progress=1.0,
            )

        zip_path = shutil.make_archive(
            str(out_dir.parent / out_name),
            "zip",
            root_dir=str(out_dir),
        )
        shutil.rmtree(out_dir, ignore_errors=True)
        job.result = {
            "written": written,
            "errors": errors,
            "failures": failures,
            "download_name": f"{out_name}.zip",
            "artifact_path": str(zip_path),
            "format": fmt,
        }
        job.progress = 1.0
        job.message = f"完成 {len(written)}/{total}"
        job.mark_terminal("done")
    except Exception as err:  # noqa: BLE001
        _log.exception("r2r batch job failed")
        job.error = str(err)
        job.mark_terminal("error")


__all__ = [
    "_build_r2r_calibration_session",
    "_r2r_entry_from_upload",
    "_r2r_prepare_retarget_motion",
    "_r2r_retarget_from_path",
    "_r2r_retarget_progress_cb",
    "_run_r2r_basket_upload_job",
    "_run_r2r_batch_job",
    "_run_r2r_source_upload_job",
]
