"""Dataset analysis, upload, export, and preview routes."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import Response

from hhtools.application.previews import _run_dataset_robot_preview_job
from hhtools.application.state import Job

_log = logging.getLogger(__name__)


def register_dataset_routes(app, *, state, jobs, uploads) -> None:
    _schedule_job = jobs.schedule
    _store_uploads = uploads.store

    def _run_dataset_analyze_job(job: Job, body: dict) -> None:
        try:
            from hhtools.web.analysis import dataset_analysis as _da

            root = Path(body.get("source") or state.source_root)
            embedding = str(body.get("embedding") or "handcrafted")
            force = bool(body.get("force", False))

            def cb(frac: float, msg: str) -> None:
                job.progress = float(max(0.0, min(1.0, frac)))
                job.message = msg

            job.message = "扫描数据集…"
            payload = _da.run_analysis(
                root,
                state.save_dir,
                embedding=embedding,
                force=force,
                progress=cb,
            )
            job.result = payload
            job.progress = 1.0
            job.mark_terminal("done")
        except Exception as err:  # noqa: BLE001
            _log.exception("dataset analyze job failed")
            job.error = str(err)
            job.mark_terminal("error")

    @app.post("/api/dataset/analyze")
    async def dataset_analyze(body: dict) -> dict:
        job = _schedule_job(
            "dataset_analyze",
            body,
            _run_dataset_analyze_job,
            args=(body,),
        )
        return {"job_id": job.id}

    @app.get("/api/dataset/result")
    def dataset_result(source: str | None = None, embedding: str = "handcrafted") -> dict:
        from hhtools.web.analysis import dataset_analysis as _da

        root = Path(source) if source else state.source_root
        entries = _da.build_entries(root)
        cached = _da.load_cached(root, state.save_dir, embedding, entries)
        if cached is None:
            return {"available": False, "source_root": str(root)}
        return {"available": True, **cached}

    @app.post("/api/dataset/subset")
    def dataset_subset(body: dict) -> dict:
        from hhtools.web.analysis import dataset_analysis as _da

        clips = body.get("clips") or []
        k = int(body.get("k", 0))
        alpha = float(body.get("alpha", 0.99))
        selected = _da.compute_subset(clips, k, alpha)
        return {"selected": selected, "count": len(selected)}

    @app.get("/api/dataset/catalog")
    def dataset_catalog() -> dict:
        from hhtools.analysis.catalog import load_catalog

        return load_catalog()

    @app.post("/api/dataset/upload")
    async def dataset_upload(
        files: list[UploadFile] = File(...),
        append_to: str | None = None,
        user_source_root: str | None = None,
    ) -> dict:
        """Accept a folder drop for batch analysis (preserves relative paths).

        Pass ``append_to`` (a prior ``source`` path from this endpoint) to merge
        multiple drag-and-drop batches into one analysis basket.
        """
        from hhtools.web.analysis import dataset_analysis as _da

        dataset_root = (state.upload_root / "dataset").resolve()
        dataset_root.mkdir(parents=True, exist_ok=True)
        if append_to:
            drop = Path(append_to).resolve()
            try:
                drop.relative_to(dataset_root)
            except ValueError as err:
                raise HTTPException(status_code=400, detail="invalid append_to") from err
            if not drop.is_dir():
                raise HTTPException(status_code=400, detail="append target missing")
        else:
            drop = dataset_root / uuid.uuid4().hex[:8]
            drop.mkdir(parents=True, exist_ok=True)
        stored = await _store_uploads(files, drop)
        if not stored:
            raise HTTPException(status_code=400, detail="empty upload")
        hint_root = str(user_source_root or "").strip()
        if hint_root:
            _da.save_upload_source_hint(drop, hint_root)
        summary = _da.scan_upload_summary(drop)
        return summary

    @app.post("/api/dataset/scan")
    def dataset_scan(body: dict) -> dict:
        """Scan a server-local directory without copying files into /tmp."""
        from hhtools.web.analysis import dataset_analysis as _da

        raw = str(body.get("source") or "").strip()
        if not raw:
            raise HTTPException(status_code=400, detail="请填写本机目录路径")
        root = Path(raw).expanduser()
        if not root.is_dir():
            raise HTTPException(status_code=400, detail=f"目录不存在：{root}")
        return _da.scan_upload_summary(root.resolve())

    @app.post("/api/dataset/upload/remove")
    async def dataset_upload_remove(body: dict) -> dict:
        from hhtools.web.analysis import dataset_analysis as _da

        source = str(body.get("source") or "").strip()
        folder_label = str(body.get("folder_label") or "").strip()
        if not source:
            raise HTTPException(status_code=400, detail="missing source")
        if not folder_label:
            raise HTTPException(status_code=400, detail="missing folder_label")
        drop = Path(source).resolve()
        dataset_root = (state.upload_root / "dataset").resolve()
        try:
            drop.relative_to(dataset_root)
        except ValueError as err:
            raise HTTPException(status_code=400, detail="invalid source") from err
        try:
            return _da.remove_upload_folder(drop, folder_label)
        except FileNotFoundError as err:
            raise HTTPException(status_code=404, detail=str(err)) from err
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

    @app.post("/api/dataset/export_manifest")
    def dataset_export_manifest(body: dict):
        from fastapi.responses import Response

        from hhtools.web.analysis import dataset_analysis as _da

        clips = body.get("clips") or []
        ids = body.get("ids") or []
        fmt = str(body.get("format") or "json").lower()
        analyze_source = str(body.get("analyze_source") or "").strip() or None
        user_source_root = str(body.get("user_source_root") or "").strip() or None
        if not user_source_root and analyze_source:
            user_source_root = _da.read_upload_source_hint(analyze_source)
        if fmt == "csv":
            text = _da.export_manifest_csv(
                clips,
                ids,
                analyze_source=analyze_source,
                user_source_root=user_source_root,
            )
            return Response(
                content=text,
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": "attachment; filename=dataset_manifest.csv"},
            )
        text = _da.export_manifest(
            clips,
            ids,
            analyze_source=analyze_source,
            user_source_root=user_source_root,
        )
        return Response(
            content=text,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=dataset_manifest.json"},
        )

    @app.post("/api/dataset/export_robot_zip")
    def dataset_export_robot_zip(body: dict):
        """ZIP selected robot clip folders (trajectory CSV + terrain/object sidecars)."""
        from fastapi.responses import FileResponse

        from hhtools.web.analysis import dataset_analysis as _da

        clips = body.get("clips") or []
        ids = body.get("ids") or []
        if not ids:
            raise HTTPException(status_code=400, detail="ids required")
        id_set = set(ids)
        allowed = [
            state.source_root,
            state.upload_root,
            state.upload_root / "dataset",
        ]
        for c in clips:
            if c.get("clip_id") not in id_set:
                continue
            sp = c.get("source_path")
            if sp:
                allowed.append(Path(sp).resolve().parent)
        drop = state.save_dir / "dataset_exports"
        drop.mkdir(parents=True, exist_ok=True)
        try:
            zip_path, stats = _da.export_robot_clips_zip(
                clips,
                ids,
                drop,
                zip_stem="robot_subset_export",
                allowed_roots=allowed,
            )
        except FileNotFoundError as err:
            raise HTTPException(status_code=404, detail=str(err)) from err
        except PermissionError as err:
            raise HTTPException(status_code=403, detail=str(err)) from err
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        return FileResponse(
            zip_path,
            filename=stats["zip_name"],
            media_type="application/zip",
        )

    @app.post("/api/dataset/preview_robot")
    async def dataset_preview_robot(body: dict) -> dict:
        """Load a robot export CSV for mesh playback (dataset viz scatter preview)."""
        if not body.get("source_path"):
            raise HTTPException(status_code=400, detail="source_path required")
        job = _schedule_job(
            "dataset_robot_preview",
            body,
            _run_dataset_robot_preview_job,
            args=(body, state),
        )
        return {"job_id": job.id}

    @app.get("/api/dataset/scene_glb")
    def dataset_scene_glb(token: str, mesh: str) -> Response:
        """Serve object mesh from a dataset robot-preview clip folder."""
        from types import SimpleNamespace

        from hhtools.io.scene_serialize import object_mesh_glb

        rec = state.dataset_previews.get(token)
        if rec is None:
            raise HTTPException(status_code=404, detail="preview token not found")
        clip_dir = Path(rec.get("clip_dir") or Path(rec["source_path"]).parent)
        safe = Path(mesh).name
        path = (clip_dir / safe).resolve()
        if not path.is_file() or clip_dir.resolve() not in path.parents:
            raise HTTPException(status_code=404, detail="mesh not found")
        glb = object_mesh_glb(SimpleNamespace(mesh_path=str(path), scale=1.0))
        if glb is None:
            raise HTTPException(status_code=404, detail="mesh export failed")
        return Response(content=glb, media_type="model/gltf-binary")
