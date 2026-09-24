"""Motion loading, basket upload, and video-to-motion routes."""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import Response

from hhtools.application.export import _parse_optional_fps
from hhtools.application.motions import (
    _FORMAT_TO_REFERENCE,
    _ground_motion_for_web,
    _load_motion_file,
    _load_motion_for_web,
    _load_via_adapter,
)
from hhtools.application.previews import _load_robot_export_for_web
from hhtools.application.state import Job, _snapshot_job_request
from hhtools.io.export_bundle import ensure_export_path
from hhtools.web.server.boundary import _safe_upload_directory_name
from hhtools.web.server.library_runtime import (
    _DATASET_TO_REFERENCE,
    _enrich_basket_entry,
    _library_entry_from_link,
    _library_entry_from_upload,
    _matching_materialized_clip,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MotionRouteOperations:
    register_motion: Callable


def register_motion_routes(
    app,
    *,
    state,
    jobs,
    uploads,
    motion_library_publish_lock,
) -> MotionRouteOperations:
    _schedule_job = jobs.schedule
    _reserve_job_slot = jobs.reserve_slot
    _persist_job = jobs.persist
    _store_uploads = uploads.store

    def _suggest_reference(
        motion,
        dataset: str | None,
        *,
        source_path: Path | None = None,
    ) -> str:
        if source_path is not None:
            from hhtools.io.mimic_detect import infer_mimic_dataset

            bone_names = motion.hierarchy.bone_names if str(motion.source_format) == "bvh" else None
            dataset = infer_mimic_dataset(source_path, bone_names=bone_names)
            if dataset == "omnicontact":
                from hhtools.io.bvh_detect import infer_bvh_dataset_from_joints

                detected = infer_bvh_dataset_from_joints(motion.hierarchy.bone_names)
                if detected and detected in _DATASET_TO_REFERENCE:
                    return _DATASET_TO_REFERENCE[detected]
        elif str(motion.source_format) == "bvh":
            from hhtools.io.bvh_detect import infer_bvh_dataset_from_joints

            detected = infer_bvh_dataset_from_joints(motion.hierarchy.bone_names)
            if detected:
                dataset = detected
        if dataset and dataset in _DATASET_TO_REFERENCE:
            return _DATASET_TO_REFERENCE[dataset]
        return _FORMAT_TO_REFERENCE.get(str(motion.source_format), "smpl")

    def _register_motion(
        motion,
        dataset: str | None,
        origin: str,
        *,
        library_entry: dict | None = None,
        job: Job | None = None,
        extra: dict | None = None,
    ) -> dict:
        from hhtools.io.scene_serialize import serialize_motion

        ground_cb = None
        if job is not None:
            from hhtools.services.motion_progress import MotionLoadProgress

            ground_cb = MotionLoadProgress(job, base=0.42, span=0.13).as_callback()
            ground_cb(0.0, "对齐地面与坐标…")

        # Ground + centre the clip ONCE so the visualization, retarget input
        # and any export all share the same source frame (the user wants
        # "保存时以可视化看到的为来源"). Preserves the established preview defaults.
        motion = _ground_motion_for_web(motion)
        if ground_cb is not None:
            ground_cb(1.0, "地面对齐完成")

        token = uuid.uuid4().hex[:12]
        src_path: Path | None = None
        if library_entry is not None and library_entry.get("source_path"):
            src_path = Path(library_entry["source_path"])
        elif extra:
            picked = extra.get("picked") or (extra.get("upload_info") or {}).get("picked")
            if picked:
                src_path = Path(picked)
        ref = _suggest_reference(motion, dataset, source_path=src_path)
        motion_rec: dict = {"motion": motion, "reference": ref, "origin": origin}
        if library_entry is not None and library_entry.get("source_path"):
            motion_rec["source_path"] = library_entry["source_path"]
            motion_rec["library_entry"] = _snapshot_job_request(library_entry)
        state.motions[token] = motion_rec

        ser_cb = None
        if job is not None:
            from hhtools.services.motion_progress import MotionLoadProgress

            ser_cb = MotionLoadProgress(job, base=0.55, span=0.17).as_callback()

        payload = serialize_motion(motion, progress_callback=ser_cb)
        payload["token"] = token
        payload["suggested_reference"] = ref
        payload["dataset"] = dataset
        payload["origin"] = origin
        if library_entry is not None:
            payload["library_entry"] = library_entry
        if extra:
            payload.update(extra)
        # Hint the front-end which retarget backend fits this clip: anything
        # with terrain / interaction objects defaults to interaction-mesh.
        has_scene = bool(motion.terrain is not None or motion.objects)
        payload["suggested_backend"] = "interaction_mesh" if has_scene else "newton"

        if job is not None:
            job.message = "完成"
            job.progress = 1.0
        return payload

    def _run_motion_library_job(job: Job, body: dict) -> None:
        from hhtools.services.motion_progress import MotionLoadProgress
        from hhtools.services.r2r_upload_resolve import _is_robot_export_trajectory

        try:
            from hhtools.services.motion_library_links import library_entry_for_load

            entry = library_entry_for_load(
                dataset=body["dataset"],
                folder_label=body["folder_label"],
                sequence_id=body["sequence_id"],
                source_path=body["source_path"],
            )
            load_prog = MotionLoadProgress(job, base=0.08, span=0.34)
            source_path = entry.source_path
            if body.get("dataset") == "robot" or _is_robot_export_trajectory(source_path):
                motion = _load_robot_export_for_web(
                    source_path,
                    state,
                    progress=load_prog,
                )
                dataset_label = "robot"
            else:
                motion = _load_motion_for_web(
                    entry,
                    state.cache,
                    progress=load_prog,
                )
                dataset_label = entry.dataset
            payload = _register_motion(
                motion,
                dataset_label,
                "library",
                library_entry=_enrich_basket_entry(
                    {
                        "dataset": dataset_label,
                        "folder_label": entry.folder_label,
                        "sequence_id": entry.sequence_id,
                        "source_path": str(entry.source_path),
                        "stem": entry.stem,
                    }
                ),
                job=job,
            )
            job.result = payload
            job.mark_terminal("done")
        except Exception as err:  # noqa: BLE001
            _log.exception("motion library job failed")
            job.error = str(err)
            job.mark_terminal("error")

    def _run_basket_upload_job(job: Job, drop: Path, profile: str) -> None:
        from hhtools.services.upload_resolve import (
            enumerate_upload_clips,
            upload_validation_error,
        )

        try:
            clips = enumerate_upload_clips(drop, profile)
            if not clips:
                raise ValueError(upload_validation_error(profile))
            entries = []
            for i, ref in enumerate(clips):
                job.progress = i / max(1, len(clips))
                job.message = f"解析 {i + 1}/{len(clips)}: {ref.path.name}"
                entry = _library_entry_from_upload(
                    drop,
                    ref.path,
                    ref.dataset,
                    ref.profile,
                    upload_profile=ref.profile,
                    clip_kind=ref.clip_kind,
                )
                entries.append(entry)
            job.result = {
                "entries": entries,
                "clip_count": len(entries),
                "upload_root": str(drop),
            }
            job.progress = 1.0
            job.message = f"已加入 {len(entries)} 个 clip"
            job.mark_terminal("done")
        except Exception as err:  # noqa: BLE001
            _log.exception("basket upload job failed")
            job.error = str(err)
            job.mark_terminal("error")

    def _run_motion_library_dir_job(
        job: Job,
        drop: Path,
        relative_paths: list[str],
        folder_label_hint: str | None,
        profile: str,
        prefer_paths: list[str] | None = None,
    ) -> None:
        from hhtools.services.motion_library_links import materialize_drop
        from hhtools.services.motion_progress import MotionLoadProgress
        from hhtools.services.upload_resolve import resolve_upload_drop

        try:
            load_prog = MotionLoadProgress(job, base=0.08, span=0.34)
            motion, dataset, info = resolve_upload_drop(
                # Never parse through the mutable library label.  A later
                # same-label upload may replace it while this job is pending.
                drop,
                profile,
                load_motion_file=_load_motion_file,
                load_via_adapter=_load_via_adapter,
                progress=load_prog,
                prefer_paths=prefer_paths,
            )
            snapshot_picked = Path(info.get("picked", drop))
            with motion_library_publish_lock:
                lib_dir, folder_label, materialize_mode = materialize_drop(
                    relative_paths,
                    folder_label=folder_label_hint,
                    upload_drop=drop,
                )
                library_picked = _matching_materialized_clip(
                    lib_dir,
                    snapshot_root=drop,
                    snapshot_picked=snapshot_picked,
                    profile=profile,
                )
                library_entry = _library_entry_from_link(
                    folder_label,
                    lib_dir,
                    library_picked,
                    dataset,
                )
                # The request is persisted as soon as it is admitted, when a
                # single-file upload may not yet have an inferred label and no
                # materialisation mode exists.  Replace the snapshot after the
                # atomic publish step so history/restart diagnostics retain the
                # actual library destination instead of the original hint.
                job.request = {
                    **job.request,
                    "folder_label": folder_label,
                    "materialize_mode": materialize_mode,
                }
            # Persist the publication metadata before the potentially long
            # grounding/serialization phase.  A process interruption after
            # library publication must not leave history pointing only at the
            # original label hint.
            _persist_job(job)
            payload = _register_motion(
                motion,
                dataset,
                "link",
                library_entry=library_entry,
                job=job,
                extra={
                    "upload_info": info,
                    "linked_folder": folder_label,
                    "materialize_mode": materialize_mode,
                },
            )
            job.result = payload
            job.mark_terminal("done")
        except Exception as err:  # noqa: BLE001
            _log.exception("motion library dir job failed")
            job.error = str(err)
            job.mark_terminal("error")

    def _run_gvhmr_video_job(
        job: Job,
        drop: Path,
        video_path: Path,
        static_cam: bool,
        f_mm: int | None,
    ) -> None:
        """Convert one uploaded video with the isolated official GVHMR runtime."""

        from hhtools.integrations.gvhmr import GvhmrConfig, run_gvhmr
        from hhtools.services.motion_library_links import materialize_upload_tree
        from hhtools.services.motion_progress import MotionLoadProgress
        from hhtools.services.upload_resolve import load_clip_at_path

        try:
            config = GvhmrConfig.from_environment()
            # GVHMR and hhtools can use the same licensed SMPL-X directory. An
            # explicit hhtools override always wins; this only supplies the
            # integration default for the generated .pt adapter.
            os.environ.setdefault(
                "HHTOOLS_BODY_MODELS",
                str(config.body_models_root),
            )

            def inference_progress(fraction: float, message: str) -> None:
                job.progress = 0.03 + 0.67 * max(0.0, min(1.0, fraction))
                job.message = message
                _persist_job(job)

            result_path = run_gvhmr(
                video_path,
                drop,
                static_cam=static_cam,
                f_mm=f_mm,
                config=config,
                progress=inference_progress,
            )

            job.progress = 0.72
            job.message = "正在转换 GVHMR 参数为 hhtools Motion…"
            load_progress = MotionLoadProgress(job, base=0.72, span=0.13)
            motion, dataset = load_clip_at_path(
                result_path,
                "mimic",
                load_motion_file=_load_motion_file,
                load_via_adapter=_load_via_adapter,
                progress=load_progress,
            )

            folder_label_hint = _safe_upload_directory_name(
                f"gvhmr-{video_path.stem}",
                default=f"gvhmr-{job.id}",
            )
            job.progress = 0.87
            job.message = "正在发布到 Motion Library…"
            with motion_library_publish_lock:
                # Publish only the reusable trajectory. GVHMR's output tree also
                # contains detector/pose feature ``.pt`` files which the generic
                # scanner would otherwise misclassify as independent motions.
                publication_drop = drop / ".hhtools-motion-publication"
                publication_drop.mkdir(parents=True, exist_ok=False)
                publication_result = publication_drop / result_path.name
                shutil.copy2(result_path, publication_result)
                lib_dir = materialize_upload_tree(
                    publication_drop,
                    folder_label_hint,
                )
                folder_label = lib_dir.name
                materialize_mode = "copy"
                library_picked = lib_dir / result_path.name
                library_entry = _library_entry_from_link(
                    folder_label,
                    lib_dir,
                    library_picked,
                    dataset or "gvhmr",
                )

            job.progress = 0.93
            job.message = "正在构建动作预览…"
            payload = _register_motion(
                motion,
                dataset or "gvhmr",
                "gvhmr",
                library_entry=library_entry,
                extra={
                    "video_name": video_path.name,
                    "gvhmr_checkpoint": "official",
                    "gvhmr_static_cam": static_cam,
                    "materialize_mode": materialize_mode,
                    "linked_folder": folder_label,
                },
            )
            artifact_dir = ensure_export_path(
                state.export_root,
                state.export_root / job.id,
            )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = ensure_export_path(
                state.export_root,
                artifact_dir / result_path.name,
            )
            shutil.copy2(result_path, artifact_path)
            payload["artifact_path"] = str(artifact_path)
            payload["download_name"] = result_path.name
            job.result = payload
            job.progress = 1.0
            job.message = "视频动作生成完成"
            job.mark_terminal("done")
        except Exception as err:  # noqa: BLE001
            artifact_dir = state.export_root / job.id
            if artifact_dir.is_symlink() or artifact_dir.is_file():
                artifact_dir.unlink(missing_ok=True)
            else:
                shutil.rmtree(artifact_dir, ignore_errors=True)
            _log.exception("GVHMR video-to-motion job failed")
            job.error = str(err)
            job.mark_terminal("error")

    @app.get("/api/video-to-motion/status")
    def video_to_motion_status() -> dict:
        """Report whether the isolated official GVHMR runtime is ready."""

        from hhtools.integrations.gvhmr import gvhmr_status

        return gvhmr_status()

    @app.post("/api/video-to-motion/upload")
    async def upload_video_to_motion(
        files: list[UploadFile] = File(...),
        static_cam: bool = True,
        f_mm: int | None = None,
    ) -> dict:
        """Upload one video and schedule official GVHMR inference."""

        from hhtools.integrations.gvhmr import gvhmr_status

        if len(files) != 1:
            raise HTTPException(status_code=400, detail="请选择一个视频文件")
        if f_mm is not None and f_mm <= 0:
            raise HTTPException(status_code=400, detail="f_mm 必须为正整数")
        runtime = gvhmr_status()
        if not runtime["ready"]:
            missing = "; ".join(runtime["missing"])
            raise HTTPException(status_code=503, detail=f"GVHMR 尚未就绪：{missing}")

        admission = _reserve_job_slot()
        drop = state.upload_root / f"gvhmr_{uuid.uuid4().hex[:8]}"
        scheduled = False
        try:
            drop.mkdir(parents=True, exist_ok=True)
            stored = await _store_uploads(files, drop)
            if len(stored) != 1:
                raise HTTPException(status_code=400, detail="视频上传为空")
            relative, video_path = stored[0]
            if video_path.suffix.lower() not in {
                ".mp4",
                ".mov",
                ".mkv",
                ".avi",
                ".webm",
                ".m4v",
            }:
                raise HTTPException(
                    status_code=400,
                    detail="支持 MP4、MOV、MKV、AVI、WebM 和 M4V 视频",
                )
            job = _schedule_job(
                "video_to_motion",
                {
                    "file": relative.as_posix(),
                    "static_cam": static_cam,
                    "f_mm": f_mm,
                    "engine": "official_gvhmr",
                    "weights": "official",
                    "training": False,
                },
                _run_gvhmr_video_job,
                args=(drop, video_path, static_cam, f_mm),
                reservation=admission,
            )
            scheduled = True
            return {"job_id": job.id}
        finally:
            if not scheduled:
                admission.cancel()
                shutil.rmtree(drop, ignore_errors=True)

    @app.post("/api/motion/load_library")
    async def load_library(body: dict) -> dict:
        if body.get("usage") == "human_to_robot":
            from hhtools.services.motion_library_links import library_entry_for_load
            from hhtools.services.r2r_upload_resolve import _is_robot_export_trajectory

            try:
                entry = library_entry_for_load(
                    dataset=str(body.get("dataset") or "unknown"),
                    folder_label=str(body.get("folder_label") or ""),
                    sequence_id=str(body.get("sequence_id") or ""),
                    source_path=str(body.get("source_path") or ""),
                )
            except (FileNotFoundError, ValueError) as err:
                raise HTTPException(status_code=422, detail=str(err)) from err
            if str(body.get("dataset") or "").casefold() in {
                "robot",
                "r2r",
            } or _is_robot_export_trajectory(entry.source_path):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "人体到机器人工作流只接受人体动作；机器人关节轨迹请使用"
                        "机器人到机器人工作流。"
                    ),
                )
        job = _schedule_job(
            "motion_load",
            body,
            _run_motion_library_job,
            args=(body,),
        )
        return {"job_id": job.id}

    @app.post("/api/basket/upload")
    async def basket_upload(
        files: list[UploadFile] = File(...),
        profile: str = "auto",
    ) -> dict:
        """Upload external clips into the session cache for batch retarget."""
        admission = _reserve_job_slot()
        drop = state.upload_root / uuid.uuid4().hex[:8]
        scheduled = False
        try:
            drop.mkdir(parents=True, exist_ok=True)
            stored = await _store_uploads(files, drop)
            if not stored:
                raise HTTPException(status_code=400, detail="empty upload")
            job = _schedule_job(
                "basket_upload",
                {
                    "profile": profile,
                    "file_count": len(stored),
                    "files": [relative.as_posix() for relative, _path in stored],
                },
                _run_basket_upload_job,
                args=(drop, profile),
                reservation=admission,
            )
            scheduled = True
            return {"job_id": job.id}
        finally:
            if not scheduled:
                admission.cancel()
                shutil.rmtree(drop, ignore_errors=True)

    @app.post("/api/basket/scan")
    def basket_scan(body: dict) -> dict:
        """Enumerate Human2Robot clips on a server-local path (no copy)."""
        from hhtools.services.upload_resolve import enumerate_upload_clips

        raw = str(body.get("source") or "").strip()
        profile = str(body.get("profile") or "auto").strip() or "auto"
        if not raw:
            raise HTTPException(status_code=400, detail="请填写本机目录路径")
        root = Path(raw).expanduser()
        if not root.is_dir():
            raise HTTPException(status_code=400, detail=f"目录不存在：{root}")
        root = root.resolve()
        clips = enumerate_upload_clips(root, profile)
        if not clips:
            raise HTTPException(
                status_code=400,
                detail="未找到可识别的动作 clip（OmniContact 需要 motion_actor.bvh）",
            )
        entries = [
            _library_entry_from_upload(
                root,
                ref.path,
                ref.dataset,
                ref.profile,
                upload_profile=ref.profile,
                clip_kind=ref.clip_kind,
                origin="local",
            )
            for ref in clips
        ]
        return {
            "entries": entries,
            "clip_count": len(entries),
            "source": str(root),
            "profile": profile,
        }

    @app.post("/api/motion/upload")
    async def upload_motion(
        files: list[UploadFile] = File(...),
        profile: str = "mimic",
        library_folder_label: str | None = None,
    ) -> dict:
        """Upload motion clips; auto-link or copy them into the managed library."""

        from hhtools.services.motion_library_links import motions_library_root

        if not files:
            raise HTTPException(status_code=400, detail="empty upload")

        from hhtools.services.upload_resolve import (
            enumerate_upload_clips,
            upload_validation_error,
        )

        folder_label = str(library_folder_label or "").strip() or None

        # Reserve admission before touching either the upload tree or the user's
        # persistent Motion Library.  A bounded queue therefore rejects cleanly.
        admission = _reserve_job_slot()
        drop = state.upload_root / uuid.uuid4().hex[:8]
        scheduled = False
        try:
            drop.mkdir(parents=True, exist_ok=True)
            # Always buffer browser bytes first so a bad on-disk symlink guess
            # cannot discard the only copy of the clip (see link_to_library).
            stored = await _store_uploads(files, drop)
            rel_paths = [relative.as_posix() for relative, _destination in stored]

            # Validate the isolated drop before materialisation.  Previously an
            # unsupported upload could replace a same-named library folder and
            # only then return HTTP 400.
            if not enumerate_upload_clips(drop, profile):
                raise HTTPException(
                    status_code=400,
                    detail=upload_validation_error(profile),
                )

            job = _schedule_job(
                "motion_link",
                {
                    "profile": profile,
                    "folder_label": folder_label,
                    "file_count": len(rel_paths),
                    "files": rel_paths,
                },
                _run_motion_library_dir_job,
                args=(drop, rel_paths, folder_label, profile),
                kwargs={"prefer_paths": rel_paths},
                reservation=admission,
            )
            scheduled = True
            return {
                "job_id": job.id,
                "linked": False,
                "materialize_mode": "pending",
                "motions_library_root": str(motions_library_root()),
            }
        finally:
            if not scheduled:
                admission.cancel()
                shutil.rmtree(drop, ignore_errors=True)

    @app.get("/api/object_glb")
    def object_glb(token: str, index: int, scale: float | None = None) -> Response:
        rec = state.motions.get(token)
        if not rec:
            raise HTTPException(status_code=404, detail="unknown motion token")
        from hhtools.io.scene_serialize import object_mesh_glb

        objs = rec["motion"].objects
        if index < 0 or index >= len(objs):
            raise HTTPException(status_code=404, detail="object index out of range")
        scale_override = _parse_optional_fps(scale)
        glb = object_mesh_glb(objs[index], scale=scale_override)
        if glb is None:
            raise HTTPException(status_code=404, detail="no mesh for object")
        return Response(content=glb, media_type="model/gltf-binary")

    return MotionRouteOperations(register_motion=_register_motion)
