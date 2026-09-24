"""Robot-to-robot source, calibration, retarget, and batch routes."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import Response

from hhtools.application.export import _parse_optional_fps, _write_r2r_export
from hhtools.application.motions import _motion_for_retarget
from hhtools.application.previews import (
    _compute_r2r_scaled_preview,
)
from hhtools.application.robots import _join_robot_prewarm, _require_newton_package
from hhtools.application.state import Job, _snapshot_job_request
from hhtools.io.export_bundle import ensure_export_path, sanitize_export_stem
from hhtools.web.server.r2r_runtime import (
    _build_r2r_calibration_session,
    _r2r_entry_from_upload,
    _r2r_prepare_retarget_motion,
    _r2r_retarget_progress_cb,
    _run_r2r_basket_upload_job,
    _run_r2r_batch_job,
    _run_r2r_source_upload_job,
)
from hhtools.web.server.requests import RobotRetargetRequest

_log = logging.getLogger(__name__)


def register_r2r_routes(app, *, state, jobs, uploads) -> None:
    _schedule_job = jobs.schedule
    _reserve_job_slot = jobs.reserve_slot
    _store_uploads = uploads.store

    def _r2r_get_model(name: str, *, compile_mjcf: bool = True):
        model = state.robots.get(name)
        if model is None:
            from hhtools.robot.loader import load_robot
            from hhtools.robot.registry import get as _get_preset

            model = load_robot(_get_preset(name), compile_mjcf=compile_mjcf)
            state.robots[name] = model
        return model

    @app.post("/api/r2r/source/upload")
    async def r2r_source_upload(
        files: list[UploadFile] = File(...),
        source_robot: str = "",
        profile: str = "auto",
        source_fps: float | None = None,
    ) -> dict:
        """Upload robot trajectory clip(s); FK runs in a background job with progress."""
        if not files:
            raise HTTPException(status_code=400, detail="no trajectory file uploaded")
        if not source_robot:
            raise HTTPException(status_code=400, detail="source_robot is required")
        src_fps = _parse_optional_fps(source_fps)
        admission = _reserve_job_slot()
        drop = state.upload_root / f"r2r_{uuid.uuid4().hex[:8]}"
        scheduled = False
        try:
            drop.mkdir(parents=True, exist_ok=True)
            stored = await _store_uploads(files, drop)
            if not stored:
                raise HTTPException(status_code=400, detail="empty upload")
            job = _schedule_job(
                "r2r_source_upload",
                {
                    "source_robot": source_robot,
                    "profile": profile,
                    "source_fps": src_fps,
                    "file_count": len(stored),
                    "files": [relative.as_posix() for relative, _path in stored],
                },
                _run_r2r_source_upload_job,
                args=(drop, source_robot, profile, state, src_fps),
                reservation=admission,
            )
            scheduled = True
            return {"job_id": job.id}
        finally:
            if not scheduled:
                admission.cancel()
                shutil.rmtree(drop, ignore_errors=True)

    @app.post("/api/r2r/source/library")
    async def r2r_source_library(body: dict) -> dict:
        """Load one existing robot trajectory without crossing into H2R data."""
        from hhtools.services.motion_library_links import library_entry_for_load
        from hhtools.services.r2r_upload_resolve import r2r_clip_ref_for_path

        source_robot = str(body.get("source_robot") or "").strip()
        if not source_robot:
            raise HTTPException(status_code=400, detail="source_robot is required")
        try:
            entry = library_entry_for_load(
                dataset=str(body.get("dataset") or "unknown"),
                folder_label=str(body.get("folder_label") or ""),
                sequence_id=str(body.get("sequence_id") or ""),
                source_path=str(body.get("source_path") or ""),
            )
            profile = str(body.get("upload_profile") or body.get("profile") or "auto")
            clip_ref = r2r_clip_ref_for_path(entry.source_path, profile)
            source_fps = _parse_optional_fps(body.get("source_fps"))
        except (FileNotFoundError, TypeError, ValueError) as err:
            raise HTTPException(status_code=422, detail=str(err)) from err

        job = _schedule_job(
            "r2r_source_library",
            {
                "source_robot": source_robot,
                "profile": clip_ref.profile,
                "source_fps": source_fps,
                "source_path": str(clip_ref.path),
            },
            _run_r2r_source_upload_job,
            args=(
                clip_ref.path.parent,
                source_robot,
                clip_ref.profile,
                state,
                source_fps,
                clip_ref.path,
            ),
        )
        return {"job_id": job.id}

    @app.get("/api/r2r/scene_glb")
    def r2r_scene_glb(token: str, mesh: str, scale: float | None = None) -> Response:
        """Serve an interaction-object mesh from an uploaded R2R clip folder."""
        from types import SimpleNamespace

        from hhtools.io.scene_serialize import object_mesh_glb

        rec = state.r2r_sources.get(token)
        if rec is None:
            raise HTTPException(status_code=404, detail="r2r source token not found")
        clip_dir = Path(rec.get("clip_dir") or Path(rec["source_path"]).parent)
        safe = Path(mesh).name
        path = (clip_dir / safe).resolve()
        if not path.is_file() or clip_dir.resolve() not in path.parents:
            raise HTTPException(status_code=404, detail="mesh not found")
        scale_override = float(scale) if scale is not None and scale > 0 else None
        glb = object_mesh_glb(
            SimpleNamespace(mesh_path=str(path), scale=scale_override or 1.0),
            scale=scale_override,
        )
        if glb is None:
            raise HTTPException(status_code=404, detail="mesh export failed")
        return Response(content=glb, media_type="model/gltf-binary")

    @app.post("/api/r2r/calibration/session")
    async def r2r_calibration_session(body: dict) -> dict:
        target = body.get("target")
        source = body.get("source")
        if not target or not source:
            raise HTTPException(status_code=400, detail="target and source required")
        try:
            tgt = _r2r_get_model(target)
            src = _r2r_get_model(source, compile_mjcf=False)
            return _build_r2r_calibration_session(tgt, src)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        except Exception as err:  # noqa: BLE001
            _log.exception("r2r calibration session failed")
            raise HTTPException(status_code=500, detail=str(err)) from err

    @app.post("/api/r2r/calibration/save")
    async def r2r_calibration_save(body: dict) -> dict:
        from hhtools.retarget import robot_to_robot as r2r

        target = body.get("target")
        source = body.get("source")
        joint_q = {str(k): float(v) for k, v in body.get("joint_q", {}).items()}
        if not target or not source:
            raise HTTPException(status_code=400, detail="target and source required")
        try:
            tgt = _r2r_get_model(target, compile_mjcf=False)
            path = r2r.save_r2r_calibration(
                tgt.preset.urdf_path.parent,
                target_robot=tgt.preset.name,
                source_robot=source,
                calibrated_joint_q=joint_q,
            )
        except Exception as err:  # noqa: BLE001
            raise HTTPException(
                status_code=400,
                detail=f"calibration save failed: {err}",
            ) from err
        return {"ok": True, "path": str(path)}

    @app.get("/api/r2r/calibration/status")
    def r2r_calibration_status(target: str, source: str) -> dict:
        from hhtools.retarget import robot_to_robot as r2r
        from hhtools.robot.registry import get as _get_preset

        try:
            preset = _get_preset(target)
            saved = r2r.load_r2r_calibration(
                preset.urdf_path.parent,
                source,
                target_robot=preset.name,
            )
        except Exception:  # noqa: BLE001
            saved = None
        return {"calibrated": bool(saved)}

    def _run_r2r_retarget_job(job: Job, body: dict) -> None:
        try:
            job.progress = 0.01
            job.message = "正在准备 robot-to-robot retarget…"
            target = body["target"]
            source = body["source"]
            token = body["source_token"]
            ik_iters = int(body.get("ik_iterations", 24))
            retarget_fps = _parse_optional_fps(body.get("retarget_fps"))
            backend = (body.get("backend") or "newton").strip().lower()

            rec = state.r2r_sources.get(token)
            if rec is None:
                raise ValueError("source trajectory expired; re-upload the clip")
            job.request = _snapshot_job_request(
                {
                    **job.request,
                    "target": target,
                    "source": source,
                    "source_path": rec.get("source_path"),
                    "backend": backend,
                    "ik_iterations": ik_iters,
                    "retarget_fps": retarget_fps,
                }
            )

            from hhtools.retarget import robot_to_robot as r2r

            tgt = _r2r_get_model(target)
            src = _r2r_get_model(source, compile_mjcf=False)
            calib = r2r.load_r2r_calibration(
                tgt.preset.urdf_path.parent,
                source,
                target_robot=tgt.preset.name,
            )
            if not calib:
                raise ValueError(
                    "target robot is not calibrated against this source robot; "
                    "run the calibration step first"
                )

            if backend != "interaction_mesh":
                _require_newton_package()
                _join_robot_prewarm(state, target, job)

            motion_src = rec["motion"]
            motion, _eff_fps = _motion_for_retarget(motion_src, retarget_fps)
            motion = _r2r_prepare_retarget_motion(
                motion,
                backend=backend,
                clip_dir=rec.get("clip_dir"),
                robot_path=rec.get("source_path"),
                profile=str(rec.get("upload_profile") or "mimic"),
                has_scene=bool(rec.get("has_scene")),
            )

            def _cb(done: int, total: int) -> None:
                _r2r_retarget_progress_cb(job, backend, done=done, total=total)

            ret = r2r.retarget_robot_to_robot(
                src,
                tgt,
                calibrated_joint_q=calib,
                source_motion=motion,
                backend=backend,
                ik_iterations=ik_iters,
                progress_callback=_cb,
            )
            from hhtools.io.scene_serialize import serialize_robot_trajectory

            scaled = _compute_r2r_scaled_preview(src, tgt, motion, calib)
            traj = serialize_robot_trajectory(
                tgt,
                ret,
                scaled_preview=scaled,
                ground_follow=False,
                yellow_align="ankle",
            )
            from hhtools.analysis.result_diagnostics import build_result_diagnostics

            diagnostics = build_result_diagnostics(
                traj,
                scaled,
                ik_map=tgt.preset.ik_map,
                feet=tgt.preset.feet,
            )
            from hhtools.io.r2r_export_bundle import clip_has_export_scene
            from hhtools.io.r2r_scene import compute_r2r_target_scaled_scene
            from hhtools.io.scene_serialize import _scaled_overlay_foot_z
            from hhtools.retarget.interaction_mesh.heightfield import obj_to_heightfield

            stem = sanitize_export_stem(rec.get("stem") or "r2r")
            clip_dir_path = Path(rec.get("clip_dir") or Path(rec["source_path"]).parent)
            scene_prof = str(rec.get("upload_profile") or "mimic")
            src_has_scene = bool(rec.get("has_scene")) or clip_has_export_scene(
                clip_dir_path,
                stem=stem,
                profile=scene_prof,
            )
            tgt_scene = None
            if src_has_scene and rec.get("clip_dir") and rec.get("source_path"):
                tgt_scene = compute_r2r_target_scaled_scene(
                    motion,
                    scale_ratio=r2r.r2r_scene_scale_ratio(src, tgt, motion, calib),
                    terrain_loader=obj_to_heightfield,
                    clip_dir=Path(rec["clip_dir"]),
                    profile=scene_prof,
                    robot_path=Path(rec["source_path"]),
                    num_frames=int(ret.num_frames),
                    framerate=float(ret.sample_rate),
                )
            export_token = uuid.uuid4().hex[:10]
            has_scene = src_has_scene
            yellow_foot_z = _scaled_overlay_foot_z(scaled, 0)
            r2r_entry = {
                "source_path": rec.get("source_path"),
                "clip_dir": rec.get("clip_dir"),
                "stem": stem,
                "has_scene": has_scene,
                "upload_profile": scene_prof,
            }
            state.motions[f"export::{export_token}"] = {
                "retargeted": ret,
                "robot": target,
                "source_motion": motion,
                "backend": backend,
                "stem": stem,
                "has_scene": has_scene,
                "source_path": rec.get("source_path"),
                "r2r": True,
                "source_robot": source,
                "yellow_foot_z": yellow_foot_z,
                "r2r_entry": r2r_entry,
            }
            artifact_dir = ensure_export_path(
                state.export_root,
                state.export_root / job.id,
            )
            artifact_path = ensure_export_path(
                state.export_root,
                _write_r2r_export(
                    ret,
                    tgt,
                    motion,
                    artifact_dir,
                    source_model=src,
                    calibrated_joint_q=calib,
                    entry=r2r_entry,
                    stem=stem,
                    fps=None,
                    fmt="csv",
                    csv_header=True,
                    yellow_foot_z=yellow_foot_z,
                ),
            )
            from hhtools.services.execution import build_execution_provenance

            job.result = {
                "trajectory": traj,
                "export_token": export_token,
                "stem": stem,
                "num_frames": ret.num_frames,
                "source_fps": float(ret.sample_rate),
                "scaled_preview": scaled,
                "scaled_scene": tgt_scene,
                "diagnostics": diagnostics,
                "has_scene": has_scene,
                "format": "csv",
                "artifact_path": str(artifact_path),
                "download_name": (
                    f"{stem}_export.zip" if artifact_path.suffix == ".zip" else artifact_path.name
                ),
                "execution_provenance": build_execution_provenance(
                    ret,
                    executor="web_r2r_v1",
                    backend=backend,
                ).model_dump(mode="json", exclude_none=True),
            }
            job.progress = 1.0
            job.message = "done"
            job.mark_terminal("done")
        except Exception as err:  # noqa: BLE001
            artifact_dir = state.export_root / job.id
            if artifact_dir.is_symlink() or artifact_dir.is_file():
                artifact_dir.unlink(missing_ok=True)
            else:
                shutil.rmtree(artifact_dir, ignore_errors=True)
            _log.exception("r2r retarget job failed")
            job.error = str(err)
            job.mark_terminal("error")

    @app.post("/api/r2r/retarget")
    async def r2r_retarget(request: RobotRetargetRequest) -> dict:
        from hhtools.retarget import robot_to_robot as r2r

        record = state.r2r_sources.get(request.source_token)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail="source trajectory expired; reload the clip",
            )
        if record.get("source_robot") != request.source:
            raise HTTPException(
                status_code=422,
                detail="source robot does not match the trajectory",
            )
        try:
            target = _r2r_get_model(request.target, compile_mjcf=False)
            _r2r_get_model(request.source, compile_mjcf=False)
            calibration = r2r.load_r2r_calibration(
                target.preset.urdf_path.parent,
                request.source,
                target_robot=target.preset.name,
            )
        except (KeyError, FileNotFoundError) as error:
            raise HTTPException(status_code=404, detail="robot asset not found") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if not calibration:
            raise HTTPException(
                status_code=409,
                detail="target robot is not calibrated; calibrate first",
            )
        body = request.model_dump()
        job = _schedule_job(
            "r2r_retarget",
            body,
            _run_r2r_retarget_job,
            args=(body,),
        )
        return {"job_id": job.id}

    @app.post("/api/r2r/basket/upload")
    async def r2r_basket_upload(
        files: list[UploadFile] = File(...),
        profile: str = "auto",
    ) -> dict:
        admission = _reserve_job_slot()
        drop = state.upload_root / uuid.uuid4().hex[:8]
        scheduled = False
        try:
            drop.mkdir(parents=True, exist_ok=True)
            stored = await _store_uploads(files, drop)
            if not stored:
                raise HTTPException(status_code=400, detail="empty upload")
            job = _schedule_job(
                "r2r_basket_upload",
                {
                    "profile": profile,
                    "file_count": len(stored),
                    "files": [relative.as_posix() for relative, _path in stored],
                },
                _run_r2r_basket_upload_job,
                args=(drop, profile),
                reservation=admission,
            )
            scheduled = True
            return {"job_id": job.id}
        finally:
            if not scheduled:
                admission.cancel()
                shutil.rmtree(drop, ignore_errors=True)

    @app.post("/api/r2r/basket/scan")
    def r2r_basket_scan(body: dict) -> dict:
        """Enumerate R2R clips on a server-local path (no copy)."""
        from hhtools.services.r2r_upload_resolve import enumerate_r2r_clips, validate_r2r_upload

        raw = str(body.get("source") or "").strip()
        profile = str(body.get("profile") or "auto").strip() or "auto"
        if not raw:
            raise HTTPException(status_code=400, detail="请填写本机目录路径")
        root = Path(raw).expanduser()
        if not root.is_dir():
            raise HTTPException(status_code=400, detail=f"目录不存在：{root}")
        root = root.resolve()
        try:
            validate_r2r_upload(root, profile)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        clips = enumerate_r2r_clips(root, profile)
        if not clips:
            raise HTTPException(status_code=400, detail="未找到可识别的机器人轨迹 clip")
        entries = [_r2r_entry_from_upload(root, ref) for ref in clips]
        return {
            "entries": entries,
            "clip_count": len(entries),
            "source": str(root),
            "profile": profile,
        }

    @app.post("/api/r2r/batch/retarget")
    async def r2r_batch_retarget(body: dict) -> dict:
        job = _schedule_job(
            "r2r_batch",
            body,
            _run_r2r_batch_job,
            args=(body, state),
        )
        return {"job_id": job.id}
