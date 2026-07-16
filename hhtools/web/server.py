"""FastAPI backend for the hhtools web UI.

Single-user, localhost-first.  All heavy lifting (motion IO, URDF loading,
calibration, retargeting) re-uses the existing ``hhtools`` pipeline; the
browser only renders and drives interaction.

Run via ``hhtools web`` (see :mod:`hhtools.cli.web`) or::

    uv run hhtools web
"""

from __future__ import annotations

import logging
import math
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# These names are referenced in route annotations (e.g. ``list[UploadFile]``).
# Because this module uses ``from __future__ import annotations`` *and* imports
# FastAPI lazily inside ``create_app``, FastAPI would resolve those string
# annotations against the *module* globals — where the lazily-imported names are
# absent — and fail.  Importing them here (guarded, so the module still loads
# without the optional ``web`` extra) makes the forward refs resolvable.
try:  # pragma: no cover - depends on optional extra being installed
    from fastapi import UploadFile
except ImportError:  # fastapi not installed; routes are never defined either
    UploadFile = Any  # type: ignore[assignment,misc]

_log = logging.getLogger(__name__)

# Bump when static/ front-end behaviour changes.  Injected into ``index.html``
# at serve time so collaborators only need to pull + restart (no triple-sync).
UI_BUILD_ID = "20260622-v83"

# Datasets whose adapters accept ``with_mesh=True`` (SMPL forward → baked vertices).
# The web UI always requests mesh so AMASS / Motion-X etc. show a real body surface,
# not just a stick skeleton (matches Viser's "Skinned mesh" path).
_SMPL_MESH_DATASETS: frozenset[str] = frozenset(
    {"amass", "motion_x", "phuma", "gvhmr", "kungfu_athlete"}
)

# Map a motion's provenance to the calibration reference pose it needs.  This
# drives the "this format isn't calibrated yet — calibrate first" prompt.
_FORMAT_TO_REFERENCE: dict[str, str] = {
    "smpl": "smpl",
    "smplh": "smpl",
    "smplx": "smplx",
    "bvh": "lafan_bvh",
    "glb": "glb",
    "gltf": "glb",
    "npz": "smpl",
    "csv": "smpl",
    "unknown": "smpl",
}

# Dataset adapter name → reference (more specific than source_format).
_DATASET_TO_REFERENCE: dict[str, str] = {
    "amass": "smpl",
    "motion_x": "smplx",
    "phuma": "smpl",
    "lafan": "lafan_bvh",
    "mocap": "mocap_bvh",
    "soma": "soma_bvh",
    "xsens_mocap": "xsens_mocap",
    "gvhmr": "gvhmr",
    "omomo": "smplx",
    "meshmimic_holosoma": "smplx",
    "glb": "glb",
    "unified_npz": "smpl",
    "parc_ms": "smpl",
}


def _tmpdir(tag: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"hhtools_web_{tag}_"))


def _robot_library_root() -> Path:
    """Persistent per-user robot library (survives ``hhtools web`` restarts)."""
    from hhtools.utils.paths import user_robot_dir

    return user_robot_dir()


def _start_robot_prewarm(state: SessionState, model: Any, name: str) -> None:
    """Background-compile Warp IK kernels after a robot loads (Viser parity)."""

    def _run() -> None:
        try:
            _require_newton_package()
            from hhtools.retarget.newton_basic._warp_config import configure as configure_warp_cache
            from hhtools.retarget.newton_basic.pipeline import NewtonBasicPipeline

            configure_warp_cache()
            NewtonBasicPipeline.prewarm_for_robot(model)
        except Exception:  # noqa: BLE001 — optional GPU / missing newton
            _log.debug("background IK prewarm failed for %r", name, exc_info=True)

    prev = state.robot_prewarm_threads.get(name)
    if isinstance(prev, threading.Thread) and prev.is_alive():
        return
    thread = threading.Thread(
        target=_run, name=f"hhtools-web-prewarm-{name}", daemon=True,
    )
    state.robot_prewarm_threads[name] = thread
    thread.start()


def _join_robot_prewarm(state: SessionState, robot_name: str, job: Job | None) -> None:
    """Wait for background prewarm before the first retarget solve."""
    thread = state.robot_prewarm_threads.get(robot_name)
    if not isinstance(thread, threading.Thread) or not thread.is_alive():
        return
    if job is not None:
        job.progress = max(job.progress, 0.03)
        job.message = "正在预热 IK 内核（新机器人首次 retarget 较慢，请稍候）…"
    thread.join(timeout=180.0)


def _require_newton_package() -> None:
    """Raise a clear error when the optional NVIDIA ``newton`` wheel is missing."""
    try:
        import newton  # noqa: F401
    except ModuleNotFoundError as err:
        raise ValueError(
            "未安装 newton（Newton IK 依赖）。请先安装 retarget 额外依赖：\n"
            "  uv sync --extra web --extra retarget\n"
            "并按 NVIDIA / SOMA-Retargeter 文档安装 newton 包；"
            "仅预览 AMASS/parc_ms 动作不需要 newton，但 Retarget 与部分缩放预览需要。"
        ) from err


@dataclass
class SessionState:
    """In-memory state for the single active browser session."""

    source_root: Path
    save_dir: Path
    cache: Any = None  # EphemeralCache
    motions: dict[str, Any] = field(default_factory=dict)  # token -> (Motion, meta)
    robots: dict[str, Any] = field(default_factory=dict)  # name -> URDFRobotModel
    jobs: dict[str, Job] = field(default_factory=dict)
    # robot-to-robot source trajectories: token -> {source_robot, motion, ...}
    r2r_sources: dict[str, Any] = field(default_factory=dict)
    # dataset viz robot preview: token -> {clip_dir, source_path}
    dataset_previews: dict[str, Any] = field(default_factory=dict)
    basket: list[dict] = field(default_factory=list)  # library entries queued for batch
    upload_root: Path = field(default_factory=lambda: _tmpdir("up"))
    robot_root: Path = field(default_factory=lambda: _robot_library_root())
    export_root: Path = field(default_factory=lambda: _tmpdir("out"))
    robot_prewarm_threads: dict[str, threading.Thread] = field(default_factory=dict)


@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"  # running | done | error
    progress: float = 0.0  # overall job progress (batch: all clips)
    clip_progress: float = 0.0  # batch only: current clip / GPU-chunk progress
    message: str = ""
    result: dict | None = None
    error: str | None = None


def create_app(
    *,
    source_root: Path,
    save_dir: Path,
    cache_dir: Path | None = None,
):
    """Build the FastAPI application."""
    from fastapi import FastAPI, File, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, Response
    from fastapi.staticfiles import StaticFiles

    from hhtools.viewer.cache import EphemeralCache

    app = FastAPI(title="hhtools web", version="0.1")
    static_dir = Path(__file__).parent / "static"

    state = SessionState(source_root=Path(source_root), save_dir=Path(save_dir))
    state.cache = EphemeralCache.create(cache_dir=cache_dir, save_dir=save_dir)

    from hhtools.web.motion_library_links import ensure_motions_library, motions_library_root

    ensure_motions_library()

    def _render_index_html() -> str:
        raw = (static_dir / "index.html").read_text(encoding="utf-8")
        return raw.replace("{{UI_BUILD}}", UI_BUILD_ID)

    @app.get("/")
    @app.get("/index.html")
    def serve_index() -> HTMLResponse:
        return HTMLResponse(
            _render_index_html(),
            headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"},
        )

    @app.middleware("http")
    async def _no_cache_ui_assets(request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response

    # ----------------------------------------------------------------- meta

    @app.get("/api/health")
    def health() -> dict:
        index = static_dir / "index.html"
        index_snip = index.read_text(encoding="utf-8")[:8000] if index.is_file() else ""
        return {
            "ok": True,
            "ui_build": UI_BUILD_ID,
            "static_dir": str(static_dir.resolve()),
            "ui_features": {
                "merged_robot_panel": "data-panel=\"retarget\"" not in index_snip,
                "view_hud": "view-hud" in index_snip,
                "scaled_skeleton_toggle": "tg-scaled" in index_snip,
                "recalib_button": "recalib-btn" in index_snip,
            },
            "source_root": str(state.source_root),
            "save_dir": str(state.save_dir),
            "motions_library_root": str(motions_library_root()),
        }

    @app.get("/api/formats")
    def formats() -> dict:
        from hhtools.io.base import registered_loader_extensions

        exts = registered_loader_extensions()
        # Datasets that require sidecar geometry.
        return {
            "file_formats": [
                {"ext": ".bvh", "label": "BVH mocap", "needs": None},
                {"ext": ".glb", "label": "glTF / GLB (skinned)", "needs": None},
                {"ext": ".gltf", "label": "glTF", "needs": None},
                {"ext": ".npz", "label": "hhtools unified NPZ", "needs": None},
            ],
            "dataset_formats": [
                {"ext": ".npz", "label": "AMASS / SMPL-H,X poses", "needs": "smpl-weights"},
                {"ext": ".npy", "label": "Motion-X / holosoma", "needs": "smpl / terrain.obj"},
                {"ext": ".pkl", "label": "OMOMO (interaction)", "needs": "object .obj sidecar"},
                {"ext": ".pt", "label": "GVHMR", "needs": "smpl-weights"},
            ],
            "registered_loaders": exts,
        }

    # ----------------------------------------------------------------- library

    @app.get("/api/library")
    def library(source: str | None = None) -> dict:
        from hhtools.viewer.library import scan_library
        from hhtools.web.motion_library_links import scan_motions_library

        root = Path(source) if source else state.source_root
        lib_root = motions_library_root()
        merged: list[dict] = []
        seen: set[str] = set()
        for e in scan_library(root):
            row = _enrich_basket_entry({
                "dataset": e.dataset,
                "folder_label": e.folder_label,
                "sequence_id": e.sequence_id,
                "stem": e.stem,
                "source_path": str(e.source_path),
                "label": e.display_label,
                "origin": "assets",
            })
            seen.add(row["source_path"])
            merged.append(row)
        for raw in scan_motions_library():
            sp = str(raw.get("source_path") or "")
            if not sp or sp in seen:
                continue
            seen.add(sp)
            merged.append(_enrich_basket_entry(raw))
        merged.sort(
            key=lambda row: (
                str(row.get("folder_label") or "").lower(),
                str(row.get("stem") or "").lower(),
            ),
        )
        folders: list[str] = []
        for row in merged:
            label = str(row.get("folder_label") or "")
            if label and label not in folders:
                folders.append(label)
        return {
            "source_root": str(root),
            "motions_library_root": str(lib_root),
            "folders": folders,
            "entries": merged,
        }

    @app.post("/api/library/link")
    async def library_link(body: dict) -> dict:
        from hhtools.web.motion_library_links import link_to_library, scan_motions_library

        path = str(body.get("path") or "").strip()
        folder_label = str(body.get("folder_label") or "").strip() or None
        if not path:
            raise HTTPException(status_code=400, detail="需要 path")
        dest = link_to_library(path, folder_label=folder_label)
        entries = [e for e in scan_motions_library() if e.get("folder_label") == dest.name]
        return {
            "folder_label": dest.name,
            "kind": "directory",
            "clip_count": len(entries),
            "path": str(dest),
            "motions_library_root": str(motions_library_root()),
        }

    @app.delete("/api/library/link/{folder_label}")
    def library_unlink(folder_label: str) -> dict:
        from hhtools.web.motion_library_links import remove_library_folder

        if not remove_library_folder(folder_label):
            raise HTTPException(status_code=404, detail="link not found")
        return {"removed": folder_label}

    # --------------------------------------------------- dataset analysis (viz)

    def _run_dataset_analyze_job(job: Job, body: dict) -> None:
        try:
            from hhtools.web import dataset_analysis as _da

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
            job.status = "done"
        except Exception as err:  # noqa: BLE001
            _log.exception("dataset analyze job failed")
            job.status = "error"
            job.error = str(err)

    @app.post("/api/dataset/analyze")
    async def dataset_analyze(body: dict) -> dict:
        job = Job(id=uuid.uuid4().hex[:12], kind="dataset_analyze")
        state.jobs[job.id] = job
        threading.Thread(
            target=_run_dataset_analyze_job, args=(job, body), daemon=True,
        ).start()
        return {"job_id": job.id}

    @app.get("/api/dataset/result")
    def dataset_result(source: str | None = None, embedding: str = "handcrafted") -> dict:
        from hhtools.web import dataset_analysis as _da

        root = Path(source) if source else state.source_root
        entries = _da.build_entries(root)
        cached = _da.load_cached(root, state.save_dir, embedding, entries)
        if cached is None:
            return {"available": False, "source_root": str(root)}
        return {"available": True, **cached}

    @app.post("/api/dataset/subset")
    def dataset_subset(body: dict) -> dict:
        from hhtools.web import dataset_analysis as _da

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
        from hhtools.web import dataset_analysis as _da

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
        wrote = False
        for uf in files:
            rel = Path(uf.filename or "upload.bin")
            dst = drop / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("wb") as fp:
                fp.write(await uf.read())
            wrote = True
        if not wrote:
            raise HTTPException(status_code=400, detail="empty upload")
        hint_root = str(user_source_root or "").strip()
        if hint_root:
            _da.save_upload_source_hint(drop, hint_root)
        summary = _da.scan_upload_summary(drop)
        return summary

    @app.post("/api/dataset/upload/remove")
    async def dataset_upload_remove(body: dict) -> dict:
        from hhtools.web import dataset_analysis as _da

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

        from hhtools.web import dataset_analysis as _da

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
                headers={
                    "Content-Disposition": "attachment; filename=dataset_manifest.csv"
                },
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
            headers={
                "Content-Disposition": "attachment; filename=dataset_manifest.json"
            },
        )

    @app.post("/api/dataset/export_robot_zip")
    def dataset_export_robot_zip(body: dict):
        """ZIP selected robot clip folders (trajectory CSV + terrain/object sidecars)."""
        from fastapi.responses import FileResponse

        from hhtools.web import dataset_analysis as _da

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
        job = Job(id=uuid.uuid4().hex[:12], kind="dataset_robot_preview")
        state.jobs[job.id] = job
        threading.Thread(
            target=_run_dataset_robot_preview_job,
            args=(job, body, state),
            daemon=True,
        ).start()
        return {"job_id": job.id}

    @app.get("/api/dataset/scene_glb")
    def dataset_scene_glb(token: str, mesh: str) -> Response:
        """Serve object mesh from a dataset robot-preview clip folder."""
        from types import SimpleNamespace

        from hhtools.web.serialize import object_mesh_glb

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

    # ----------------------------------------------------------------- motion

    def _suggest_reference(
        motion,
        dataset: str | None,
        *,
        source_path: Path | None = None,
    ) -> str:
        if source_path is not None:
            from hhtools.io.mimic_detect import infer_mimic_dataset

            bone_names = (
                motion.hierarchy.bone_names
                if str(motion.source_format) == "bvh"
                else None
            )
            dataset = infer_mimic_dataset(source_path, bone_names=bone_names)
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
        from hhtools.web.serialize import serialize_motion

        ground_cb = None
        if job is not None:
            from hhtools.web.motion_progress import MotionLoadProgress

            ground_cb = MotionLoadProgress(job, base=0.42, span=0.13).as_callback()
            ground_cb(0.0, "对齐地面与坐标…")

        # Ground + centre the clip ONCE so the visualization, retarget input
        # and any export all share the same source frame (the user wants
        # "保存时以可视化看到的为来源").  Mirrors the Viser viewer defaults.
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
        state.motions[token] = motion_rec

        ser_cb = None
        if job is not None:
            from hhtools.web.motion_progress import MotionLoadProgress

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
        from hhtools.web.motion_progress import MotionLoadProgress
        from hhtools.web.r2r_upload_resolve import _is_robot_export_trajectory

        try:
            from hhtools.web.motion_library_links import library_entry_for_load

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
                    source_path, state, progress=load_prog,
                )
                dataset_label = "robot"
            else:
                motion = _load_motion_for_web(
                    entry, state.cache, progress=load_prog,
                )
                dataset_label = entry.dataset
            payload = _register_motion(
                motion,
                dataset_label,
                "library",
                library_entry=_enrich_basket_entry({
                    "dataset": dataset_label,
                    "folder_label": entry.folder_label,
                    "sequence_id": entry.sequence_id,
                    "source_path": str(entry.source_path),
                    "stem": entry.stem,
                }),
                job=job,
            )
            job.result = payload
            job.status = "done"
        except Exception as err:  # noqa: BLE001
            _log.exception("motion library job failed")
            job.status = "error"
            job.error = str(err)

    def _run_basket_upload_job(job: Job, drop: Path, profile: str) -> None:
        from hhtools.web.upload_resolve import enumerate_upload_clips

        try:
            clips = enumerate_upload_clips(drop, profile)
            if not clips:
                raise ValueError(
                    "未找到可识别的动作 clip（支持 .npz / .pkl / .bvh / .glb …，"
                    "可拖入整个文件夹保留子目录结构）"
                )
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
            job.status = "done"
            job.progress = 1.0
            job.message = f"已加入 {len(entries)} 个 clip"
        except Exception as err:  # noqa: BLE001
            _log.exception("basket upload job failed")
            job.status = "error"
            job.error = str(err)

    def _run_motion_library_dir_job(
        job: Job, lib_dir: Path, folder_label: str, profile: str,
        prefer_paths: list[str] | None = None,
    ) -> None:
        from hhtools.web.motion_progress import MotionLoadProgress
        from hhtools.web.upload_resolve import resolve_upload_drop

        try:
            load_prog = MotionLoadProgress(job, base=0.08, span=0.34)
            motion, dataset, info = resolve_upload_drop(
                lib_dir,
                profile,
                load_motion_file=_load_motion_file,
                load_via_adapter=_load_via_adapter,
                progress=load_prog,
                prefer_paths=prefer_paths,
            )
            picked = Path(info.get("picked", lib_dir))
            library_entry = _library_entry_from_link(
                folder_label, lib_dir, picked, dataset,
            )
            payload = _register_motion(
                motion,
                dataset,
                "link",
                library_entry=library_entry,
                job=job,
                extra={
                    "upload_info": info,
                    "linked_folder": folder_label,
                },
            )
            job.result = payload
            job.status = "done"
        except Exception as err:  # noqa: BLE001
            _log.exception("motion library dir job failed")
            job.status = "error"
            job.error = str(err)

    @app.post("/api/motion/load_library")
    async def load_library(body: dict) -> dict:
        job = Job(id=uuid.uuid4().hex[:12], kind="motion_load")
        state.jobs[job.id] = job
        threading.Thread(
            target=_run_motion_library_job, args=(job, body), daemon=True,
        ).start()
        return {"job_id": job.id}

    @app.post("/api/basket/upload")
    async def basket_upload(
        files: list[UploadFile] = File(...),
        profile: str = "auto",
    ) -> dict:
        """Upload external clips into the session cache for batch retarget."""
        drop = state.upload_root / uuid.uuid4().hex[:8]
        drop.mkdir(parents=True, exist_ok=True)
        wrote = False
        for uf in files:
            rel = Path(uf.filename or "upload.bin")
            dst = drop / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("wb") as fp:
                fp.write(await uf.read())
            wrote = True
        if not wrote:
            raise HTTPException(status_code=400, detail="empty upload")
        job = Job(id=uuid.uuid4().hex[:12], kind="basket_upload")
        state.jobs[job.id] = job
        threading.Thread(
            target=_run_basket_upload_job, args=(job, drop, profile), daemon=True,
        ).start()
        return {"job_id": job.id}

    @app.post("/api/motion/upload")
    async def upload_motion(
        files: list[UploadFile] = File(...),
        profile: str = "mimic",
        library_folder_label: str | None = None,
    ) -> dict:
        """Upload motion clips; auto-symlink or copy into ``~/.config/hhtools/motions``."""

        from hhtools.web.motion_library_links import materialize_drop, motions_library_root

        if not files:
            raise HTTPException(status_code=400, detail="empty upload")

        from hhtools.web.upload_resolve import enumerate_upload_clips

        rel_paths = [str(Path(uf.filename or "")) for uf in files]
        folder_label = str(library_folder_label or "").strip() or None

        # Always buffer browser bytes first so a bad on-disk symlink guess
        # cannot discard the only copy of the clip (see link_to_library).
        drop = state.upload_root / uuid.uuid4().hex[:8]
        drop.mkdir(parents=True, exist_ok=True)
        for uf in files:
            rel = Path(uf.filename or "upload.bin")
            dst = drop / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("wb") as fp:
                fp.write(await uf.read())

        lib_dir, label, materialize_mode = materialize_drop(
            rel_paths,
            folder_label=folder_label,
            upload_drop=drop,
        )
        if not enumerate_upload_clips(lib_dir, profile):
            raise FileNotFoundError(
                "library folder contains no recognizable clips after materialize"
            )

        if not enumerate_upload_clips(lib_dir, profile):
            raise HTTPException(
                status_code=400,
                detail="未找到可识别的动作文件（.npz / .bvh / .glb / .pkl …）",
            )

        job = Job(id=uuid.uuid4().hex[:12], kind="motion_link")
        state.jobs[job.id] = job
        threading.Thread(
            target=_run_motion_library_dir_job,
            args=(job, lib_dir, label, profile),
            kwargs={"prefer_paths": rel_paths},
            daemon=True,
        ).start()
        return {
            "job_id": job.id,
            "linked": True,
            "folder_label": label,
            "materialize_mode": materialize_mode,
            "motions_library_root": str(motions_library_root()),
        }

    @app.get("/api/object_glb")
    def object_glb(token: str, index: int, scale: float | None = None) -> Response:
        rec = state.motions.get(token)
        if not rec:
            raise HTTPException(status_code=404, detail="unknown motion token")
        from hhtools.web.serialize import object_mesh_glb

        objs = rec["motion"].objects
        if index < 0 or index >= len(objs):
            raise HTTPException(status_code=404, detail="object index out of range")
        scale_override = _parse_optional_fps(scale)
        glb = object_mesh_glb(objs[index], scale=scale_override)
        if glb is None:
            raise HTTPException(status_code=404, detail="no mesh for object")
        return Response(content=glb, media_type="model/gltf-binary")

    # ----------------------------------------------------------------- robots

    @app.get("/api/robots")
    def robots() -> dict:
        from hhtools.robot.registry import is_user_installed, list_presets, refresh

        refresh()
        out = []
        for p in list_presets():
            out.append(
                {
                    "name": p.name,
                    "display_name": p.display_name,
                    "has_urdf": p.has_urdf,
                    "num_dof": len(p.dof_order),
                    "deletable": is_user_installed(p, state.robot_root),
                }
            )
        return {
            "robots": out,
            "library_dir": str(state.robot_root.resolve()),
        }

    def _serialize_and_store_robot(name: str) -> dict:
        from hhtools.robot.loader import load_robot
        from hhtools.robot.registry import get as get_preset
        from hhtools.web.serialize import serialize_robot

        preset = get_preset(name)
        model = load_robot(preset, compile_mjcf=True)
        if model.mujoco_model is None:
            raise RuntimeError(
                f"URDF for {name!r} did not compile to a MuJoCo model after mesh "
                f"path repair — upload the full robot folder (URDF + meshes/, and "
                f"any mesh/, convex/, or assets/ sidecars). Collada (.dae) meshes "
                f"are auto-converted to STL at ingest."
            )
        state.robots[name] = model
        _start_robot_prewarm(state, model, name)
        payload = serialize_robot(model, name=name)
        try:
            from hhtools.retarget.newton_basic.pipeline import is_newton_ik_prewarmed

            payload["ik_prewarmed"] = is_newton_ik_prewarmed(name)
        except Exception:  # noqa: BLE001
            payload["ik_prewarmed"] = False
        return payload

    @app.post("/api/robot/select")
    async def robot_select(body: dict) -> dict:
        name = body.get("name", "")
        try:
            return _serialize_and_store_robot(name)
        except Exception as err:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"load robot failed: {err}") from err

    @app.post("/api/robot/upload")
    async def robot_upload(
        files: list[UploadFile] = File(...), name: str | None = None
    ) -> dict:
        """Accept a URDF + mesh files; scaffold + auto-repair, load, serialize."""
        from hhtools.robot.kinematics import prepare_ik_map
        from hhtools.robot.registry import preset_from_dir, refresh
        from hhtools.robot.scaffold import scaffold_yaml_file
        from hhtools.robot.urdf_normalize import (
            ensure_urdf_meshes_resolvable,
            robot_upload_destination,
        )
        from hhtools.robot.yaml_io import update_robot_yaml_ik_map

        urdf_path: Path | None = None
        saved: list[Path] = []
        drop_name = name or "uploaded_robot"
        drop = state.robot_root / drop_name
        # Re-uploading an existing robot rebuilds geometry but must NOT wipe the
        # user's tuned retarget config: keep bundled scalers, calibrations, and
        # the robot.yaml ``retarget.references`` mapping across the rebuild.
        preserved_files: dict[str, bytes] = {}
        preserved_references: dict | None = None
        if drop.exists():
            for pat in ("retarget_calibration_*.yaml", "*scaler_config*.yaml"):
                for f in drop.glob(pat):
                    try:
                        preserved_files[f.name] = f.read_bytes()
                    except OSError:
                        pass
            preserved_references = _read_yaml_retarget_references(drop)
            shutil.rmtree(drop, ignore_errors=True)
        for uf in files:
            rel = uf.filename or "f"
            data = await uf.read()
            is_urdf = rel.lower().endswith(".urdf")
            dst = robot_upload_destination(drop, rel, is_urdf=is_urdf)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            saved.append(dst)
            if is_urdf:
                urdf_path = dst
        if urdf_path is None:
            raise HTTPException(status_code=400, detail="no .urdf file in upload")

        try:
            ensure_urdf_meshes_resolvable(
                urdf_path,
                search_dirs=[drop / "meshes", drop],
                output_path=urdf_path,
            )
            # Restore calibration / bundled scalers before scaffold so
            # retarget_calibration_*.yaml survives the URDF replace.
            for fname, data in preserved_files.items():
                try:
                    (drop / fname).write_bytes(data)
                except OSError:
                    pass
            scaffold_yaml_file(urdf_path, overwrite=True, root_dir=drop)
            try:
                preset = preset_from_dir(drop)
            except FileNotFoundError as err:
                raise HTTPException(
                    status_code=400,
                    detail=f"robot ingest failed: {err}",
                ) from err
            refresh()
            repaired, _changes = prepare_ik_map(urdf_path, dict(preset.ik_map))
            yaml_path = preset.meta.get("yaml_path")
            if yaml_path and repaired != dict(preset.ik_map):
                update_robot_yaml_ik_map(yaml_path, repaired)
                refresh()
            if preserved_references:
                _merge_retarget_references(yaml_path, preserved_references)
            if preserved_references:
                refresh()
            return _serialize_and_store_robot(preset.name)
        except HTTPException:
            raise
        except Exception as err:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"robot ingest failed: {err}") from err

    @app.delete("/api/robot/{name}")
    def robot_delete(name: str) -> dict:
        """Remove a user-installed robot from the persistent library."""
        from hhtools.robot.registry import get as get_preset
        from hhtools.robot.registry import is_user_installed, refresh

        try:
            preset = get_preset(name)
        except KeyError as err:
            raise HTTPException(status_code=404, detail=f"unknown robot: {name}") from err
        if not is_user_installed(preset, state.robot_root):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"robot {name!r} is a built-in preset and cannot be deleted from the UI; "
                    "only robots registered via the web UI (under your user library) are removable"
                ),
            )
        target = preset.root_dir.resolve()
        library = state.robot_root.resolve()
        try:
            if not target.is_relative_to(library):
                raise HTTPException(status_code=403, detail="robot is outside the user library")
        except ValueError as err:
            raise HTTPException(status_code=403, detail="robot is outside the user library") from err
        shutil.rmtree(target, ignore_errors=False)
        state.robots.pop(name, None)
        refresh()
        return {"ok": True, "deleted": name}

    # ----------------------------------------------------------------- calibration

    @app.get("/api/calibration/references")
    def calibration_references() -> dict:
        from hhtools.retarget.calibration import list_reference_names

        return {"references": list(list_reference_names())}

    @app.get("/api/calibration/status")
    def calibration_status(robot: str, reference: str) -> dict:
        from hhtools.retarget.calibration import resolve_calibration_file
        from hhtools.robot.registry import get as get_preset
        from hhtools.robot.retarget_profile import bundled_scaler_path

        try:
            preset = get_preset(robot)
        except Exception as err:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(err)) from err
        if preset.urdf_path is None:
            return {"calibrated": False, "path": None}
        path = resolve_calibration_file(preset.urdf_path.parent, reference)
        bundled = bundled_scaler_path(preset, reference)
        joint_q: dict[str, float] | None = None
        if path is not None:
            from hhtools.retarget.calibration import load_calibration

            cal = load_calibration(path)
            joint_q = {str(k): float(v) for k, v in cal.calibrated_joint_q.items()}
        # Optional per-robot bundled scaler (``robot.yaml`` → ``scaler_config``)
        # also counts as ready; otherwise calibration is required.
        return {
            "calibrated": path is not None or bundled is not None,
            "bundled": bundled is not None,
            "path": str(path) if path else None,
            "joint_q": joint_q,
        }

    @app.post("/api/robot/fk_preview")
    async def robot_fk_preview(body: dict) -> dict:
        """Apply a calibration joint_q on the server and return link transforms."""
        import numpy as np

        from hhtools.web.calibration_session import joint_world_payload

        robot = body.get("robot")
        model = state.robots.get(robot)
        if model is None:
            raise HTTPException(status_code=404, detail="robot not loaded")
        joint_q = {str(k): float(v) for k, v in (body.get("joint_q") or {}).items()}
        try:
            model.apply_configuration(joint_q)
        except Exception as err:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(err)) from err
        from hhtools.web.calibration_session import _robot_ground_offset_z

        ground_z = _robot_ground_offset_z(model)
        links = [link.name for link in model.links]
        link_T: dict[str, list[float]] = {}
        for link in links:
            try:
                T = model.urdf.get_transform(link)
                link_T[link] = np.asarray(T, dtype=np.float32).flatten().tolist()
            except Exception:
                link_T[link] = np.eye(4, dtype=np.float32).flatten().tolist()
        return {
            "links": links,
            "link_transforms": link_T,
            "joint_world": joint_world_payload(model),
            "ground_offset_z": round(ground_z, 5),
        }

    @app.post("/api/calibration/save")
    async def calibration_save(body: dict) -> dict:
        from hhtools.retarget.calibration import (
            RobotRetargetCalibration,
            calibration_path_for,
            derive_calibration_params,
            save_calibration,
        )
        from hhtools.robot.registry import get as get_preset

        robot = body["robot"]
        reference = body["reference"]
        joint_q = {str(k): float(v) for k, v in body.get("joint_q", {}).items()}
        token = body.get("motion_token")
        model = state.robots.get(robot)
        motion = None
        if token:
            rec = state.motions.get(token)
            if rec is not None:
                motion = rec["motion"]
        try:
            preset = get_preset(robot)
            if model is None:
                from hhtools.robot.loader import load_robot

                model = load_robot(preset, compile_mjcf=False)
                state.robots[robot] = model
            cal = RobotRetargetCalibration(
                robot=robot, reference=reference, calibrated_joint_q=joint_q,
                notes="saved from web UI",
            )
            derived = derive_calibration_params(
                cal, model, reference_motion=motion,
            )
            path = calibration_path_for(preset.urdf_path.parent, reference=reference)
            save_calibration(cal, path, derived=derived)
            # Do not sync derived.scales into robot.yaml joint_scale_multipliers:
            # that global table is shared across references and would pollute
            # the next dataset's retarget (see active_joint_scale_overrides).
        except Exception as err:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"calibration save failed: {err}") from err
        return {"ok": True, "path": str(path)}

    @app.post("/api/calibration/session")
    async def calibration_session(body: dict) -> dict:
        """Enter calibration mode: reference T-pose, joint limits, saved joint_q."""
        from hhtools.web.calibration_session import build_calibration_session

        robot = body.get("robot")
        reference = body.get("reference")
        token = body.get("motion_token")
        model = state.robots.get(robot)
        if model is None:
            raise HTTPException(status_code=404, detail="robot not loaded")
        motion = None
        if token:
            rec = state.motions.get(token)
            if rec is not None:
                motion = rec["motion"]
        try:
            return build_calibration_session(
                model, reference=str(reference), motion=motion,
            )
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        except Exception as err:  # noqa: BLE001
            _log.exception("calibration session failed")
            raise HTTPException(status_code=500, detail=str(err)) from err

    # ----------------------------------------------------------------- retarget

    def _run_retarget_job(job: Job, body: dict) -> None:
        try:
            job.progress = 0.01
            job.message = "正在准备 retarget…"
            robot = body["robot"]
            token = body["motion_token"]
            reference = body.get("reference", "smpl")
            backend = body.get("backend", "newton")
            ik_iters = int(body.get("ik_iterations", 24))
            foot_clamp_anti_penetration = bool(
                body.get("foot_clamp_anti_penetration", False)
            )
            from hhtools.robot.registry import get as _get_preset

            human_height = _request_human_height(body, _get_preset(robot), reference)
            limit_frames = body.get("limit_frames")
            retarget_fps = _parse_optional_fps(body.get("retarget_fps"))

            rec = state.motions.get(token)
            if rec is None:
                raise ValueError("motion token expired; reload the clip")
            motion_src = rec["motion"]
            motion_source_fps = float(motion_src.framerate)
            motion, motion_retarget_fps = _motion_for_retarget(motion_src, retarget_fps)
            model = state.robots[robot]
            ret = _retarget_single(
                model, robot, motion, reference, backend,
                ik_iters, human_height, limit_frames, job,
                state=state,
                foot_clamp_anti_penetration=foot_clamp_anti_penetration,
            )
            from hhtools.web.serialize import serialize_robot_trajectory

            scaled = _compute_scaled_preview(
                model, robot, motion, reference, human_height,
            )
            traj = serialize_robot_trajectory(
                model, ret, scaled_preview=scaled,
            )
            scaled = _align_scaled_preview_to_robot_playback(
                model, ret, scaled, traj,
            )
            scaled_scene = _compute_scaled_scene(
                model, robot, motion, reference, human_height,
            )
            # Keep the retarget result + source motion in memory so the export
            # endpoint can render CSV or PKL at any target fps on demand.
            export_token = uuid.uuid4().hex[:10]
            state.motions[f"export::{export_token}"] = {
                "retargeted": ret,
                "robot": robot,
                "source_motion": motion,
                "backend": backend,
                "stem": motion.name or token,
                "has_scene": bool(motion.terrain is not None or motion.objects),
                "source_path": rec.get("source_path"),
            }
            job.result = {
                "trajectory": traj,
                "scaled_preview": scaled,
                "scaled_scene": scaled_scene,
                "export_token": export_token,
                "stem": motion.name or token,
                "motion_source_fps": motion_source_fps,
                "retarget_fps": float(motion_retarget_fps),
                "source_fps": float(ret.sample_rate),
                "has_scene": bool(motion.terrain is not None or motion.objects),
                "num_frames": ret.num_frames,
            }
            job.status = "done"
            job.progress = 1.0
            job.message = "done"
        except Exception as err:  # noqa: BLE001
            _log.exception("retarget job failed")
            job.status = "error"
            job.error = str(err)

    @app.post("/api/retarget")
    async def retarget(body: dict) -> dict:
        job = Job(id=uuid.uuid4().hex[:12], kind="retarget")
        state.jobs[job.id] = job
        threading.Thread(target=_run_retarget_job, args=(job, body), daemon=True).start()
        return {"job_id": job.id}

    @app.post("/api/scaled_preview")
    async def scaled_preview(body: dict) -> dict:
        """Scaled effector skeleton (robot calibration applied, before IK)."""
        robot = body.get("robot")
        token = body.get("motion_token")
        reference = body.get("reference", "smpl")
        rec = state.motions.get(token)
        if rec is None:
            raise HTTPException(status_code=404, detail="motion token expired; reload the clip")
        model = state.robots.get(robot)
        if model is None:
            raise HTTPException(status_code=404, detail="robot not loaded")
        human_height = _request_human_height(body, model.preset, reference)
        try:
            motion = rec["motion"]
            preview = _compute_scaled_preview(
                model, robot, motion, reference, human_height,
            )
            scaled_scene = _compute_scaled_scene(
                model, robot, motion, reference, human_height,
            )
            return {"preview": preview, "scaled_scene": scaled_scene}
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        except Exception as err:  # noqa: BLE001
            _log.exception("scaled preview failed")
            raise HTTPException(status_code=500, detail=str(err)) from err

    @app.get("/api/job/{job_id}")
    def job_status(job_id: str) -> dict:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return {
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "progress": job.progress,
            "clip_progress": job.clip_progress,
            "message": job.message,
            "result": job.result,
            "error": job.error,
        }

    @app.get("/api/job/{job_id}/download")
    def job_download(job_id: str):
        job = state.jobs.get(job_id)
        if job is None or job.status != "done":
            raise HTTPException(status_code=404, detail="job not ready")
        artifact = (job.result or {}).get("artifact_path")
        if not artifact:
            raise HTTPException(status_code=404, detail="no download artifact")
        path = Path(artifact)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="artifact missing")
        name = (job.result or {}).get("download_name") or path.name
        return FileResponse(
            path,
            filename=name,
            media_type="application/zip",
        )

    # ----------------------------------------------------------------- batch

    @app.get("/api/basket")
    def basket_get() -> dict:
        return {"basket": state.basket}

    @app.post("/api/basket/add")
    async def basket_add(body: dict) -> dict:
        fallback = (body.get("reference") or "smpl").strip()
        for e in body.get("entries", []):
            enriched = _enrich_basket_entry(e, fallback)
            if not any(
                x.get("source_path") == enriched.get("source_path")
                for x in state.basket
            ):
                state.basket.append(enriched)
        return {"basket": state.basket}

    @app.post("/api/basket/clear")
    async def basket_clear() -> dict:
        state.basket.clear()
        return {"basket": state.basket}

    def _run_batch_job(job: Job, body: dict) -> None:
        try:
            robot = body["robot"]
            default_reference = body.get("reference", "smpl")
            backend = body.get("backend", "newton")
            ik_iters = int(body.get("ik_iterations", 24))
            from hhtools.robot.registry import get as _get_preset

            human_height = _request_human_height(
                body, _get_preset(robot), default_reference
            )
            out_name = body.get("out_dir") or "batch_export"
            fmt = (body.get("format") or "csv").lower()
            csv_header = _parse_csv_header(body.get("csv_header", True))
            export_fps = _parse_optional_fps(body.get("export_fps", body.get("fps")))
            retarget_fps = _parse_optional_fps(body.get("retarget_fps"))
            limit_frames = body.get("limit_frames")
            foot_clamp_anti_penetration = bool(
                body.get("foot_clamp_anti_penetration", False)
            )
            requested_batch = max(1, min(256, int(body.get("batch_size", 16))))
            batch_size = requested_batch
            entries = [
                _enrich_basket_entry(e, default_reference)
                for e in (body.get("entries") or state.basket)
            ]
            model = state.robots[robot]
            if backend != "interaction_mesh":
                from hhtools.retarget.newton_basic.batch_limits import clamp_gpu_batch_size

                batch_size = clamp_gpu_batch_size(model, requested_batch)
                if batch_size < requested_batch:
                    _log.info(
                        "GPU batch_size clamped %d → %d for robot %r",
                        requested_batch,
                        batch_size,
                        robot,
                    )

            out_dir = state.export_root / job.id
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            total = len(entries)
            written: list[str] = []
            errors: list[str] = []
            failures: list[dict] = []
            failure_log = None
            done_clips = 0
            batch_t0 = time.monotonic()
            clamp_note = ""
            if backend != "interaction_mesh" and batch_size < requested_batch:
                clamp_note = f"（GPU 上限，批量 {requested_batch}→{batch_size}）"
            _set_batch_job_progress(
                job, f"批量开始 · 0/{total}{clamp_note}", 0.0, batch_t0,
                clip_progress=0.0,
            )

            if backend == "interaction_mesh":
                failure_log = _run_batch_entries_sequential(
                    entries, model, robot, default_reference, backend,
                    ik_iters, human_height, limit_frames, retarget_fps,
                    export_fps, fmt, csv_header, out_dir, state,
                    job=job, job_id=job.id, out_name=out_name,
                    written=written, errors=errors, failures=failures,
                    failure_log=failure_log, batch_t0=batch_t0,
                    foot_clamp_anti_penetration=foot_clamp_anti_penetration,
                )
            else:
                from collections import defaultdict

                by_ref: dict[str, list[dict]] = defaultdict(list)
                for e in entries:
                    by_ref[_entry_reference(e, default_reference)].append(e)

                ref_groups = list(by_ref.items())
                for reference, ref_entries in ref_groups:
                    for chunk_start in range(0, len(ref_entries), batch_size):
                        chunk = ref_entries[chunk_start : chunk_start + batch_size]
                        loaded_chunk: list[tuple[dict, object, object]] = []
                        for e in chunk:
                            # ``done_clips`` only advances on export/failure, so
                            # add the count already loaded in this chunk
                            # (``len(loaded_chunk)``) to keep the counter moving
                            # while a large chunk loads clip-by-clip.
                            loading_pos = done_clips + len(loaded_chunk) + 1
                            _set_batch_job_progress(
                                job,
                                f"加载 {e.get('stem', '?')} · {loading_pos}/{total}",
                                (done_clips + len(loaded_chunk)) / max(1, total),
                                batch_t0,
                                clip_progress=0.0,
                            )
                            try:
                                from hhtools.web.motion_library_links import (
                                    library_entry_for_load,
                                )

                                entry = library_entry_for_load(
                                    dataset=e["dataset"],
                                    folder_label=e["folder_label"],
                                    sequence_id=e["sequence_id"],
                                    source_path=e["source_path"],
                                    upload_drop=e.get("upload_drop"),
                                )
                                motion = _load_batch_motion(
                                    e, entry, state.cache,
                                    retarget_fps=retarget_fps,
                                    limit_frames=limit_frames,
                                )
                                loaded_chunk.append((e, motion, entry))
                            except Exception as err:  # noqa: BLE001
                                failure_log = _record_batch_failure(
                                    failure_log, state, job.id, out_name,
                                    e, stage="load", reason=str(err),
                                    reference=reference,
                                    errors=errors, failures=failures,
                                )
                                done_clips += 1
                                _set_batch_job_progress(
                                    job,
                                    f"加载失败 {e.get('stem', '?')} · {done_clips}/{total}",
                                    done_clips / max(1, total),
                                    batch_t0,
                                    clip_progress=1.0,
                                )
                        if not loaded_chunk:
                            continue

                        chunk_label = (
                            f"GPU×{len(loaded_chunk)}"
                            if len(loaded_chunk) > 1
                            else "逐条"
                        )
                        _set_batch_job_progress(
                            job,
                            (
                                f"参考 {reference} · {chunk_label} · "
                                f"clip {done_clips + 1}–"
                                f"{min(done_clips + len(loaded_chunk), total)}/{total}"
                            ),
                            done_clips / max(1, total),
                            batch_t0,
                            clip_progress=0.0,
                        )
                        base_prog = done_clips / max(1, total)
                        span_prog = len(loaded_chunk) / max(1, total)
                        try:
                            exports, failure_log = _retarget_newton_batch_chunk(
                                loaded_chunk,
                                model=model,
                                robot_name=robot,
                                reference=reference,
                                ik_iters=ik_iters,
                                human_height=human_height,
                                state=state,
                                job=job,
                                job_id=job.id,
                                out_name=out_name,
                                failure_log=failure_log,
                                failures=failures,
                                errors=errors,
                                progress_base=base_prog,
                                progress_span=span_prog,
                                batch_t0=batch_t0,
                                chunk_label=chunk_label,
                                foot_clamp_anti_penetration=(
                                    foot_clamp_anti_penetration
                                ),
                            )
                            done_clips, failure_log = _batch_export_retargeted_chunk(
                                exports,
                                model=model,
                                motion_out_dir=out_dir,
                                export_fps=export_fps,
                                fmt=fmt,
                                backend=backend,
                                csv_header=csv_header,
                                base_prog=base_prog,
                                span_prog=span_prog,
                                job=job,
                                batch_t0=batch_t0,
                                done_clips=done_clips,
                                total=total,
                                written=written,
                                failure_log=failure_log,
                                state=state,
                                job_id=job.id,
                                out_name=out_name,
                                reference=reference,
                                errors=errors,
                                failures=failures,
                            )
                        except Exception as err:  # noqa: BLE001
                            for e, _, _ in loaded_chunk:
                                failure_log = _record_batch_failure(
                                    failure_log, state, job.id, out_name,
                                    e, stage="retarget", reason=str(err),
                                    reference=reference,
                                    errors=errors, failures=failures,
                                )
                                done_clips += 1
                            _set_batch_job_progress(
                                job,
                                f"批量失败 · {done_clips}/{total}",
                                done_clips / max(1, total),
                                batch_t0,
                                clip_progress=1.0,
                            )

            if failure_log is not None:
                failure_log.finalize(job_id=job.id, out_name=out_name)

            _set_batch_job_progress(
                job, "正在打包 ZIP…", _BATCH_ZIP_PROGRESS, batch_t0,
                clip_progress=1.0,
            )
            from hhtools.web.export_bundle import zip_directory

            zip_path = zip_directory(out_dir, out_name, compress=False)
            gpu_note = (
                "GPU-parallel Newton"
                if backend != "interaction_mesh" and batch_size > 1
                else "per-clip"
            )
            job.result = {
                "written": written,
                "errors": errors,
                "failures": failures,
                "failure_log": str(failure_log.root) if failure_log else None,
                "format": fmt,
                "download_name": f"{out_name}.zip",
                "artifact_path": str(zip_path),
                "clip_count": len(written),
                "batch_size": batch_size,
                "requested_batch_size": requested_batch,
                "solver_mode": gpu_note,
            }
            job.status = "done"
            job.progress = 1.0
            job.clip_progress = 1.0
            fail_note = f"，{len(failures)} 失败" if failures else ""
            job.message = (
                f"{len(written)} 成功{fail_note}"
                + (f" · {gpu_note}" if backend != "interaction_mesh" else "")
            )
        except Exception as err:  # noqa: BLE001
            _log.exception("batch job failed")
            job.status = "error"
            job.error = str(err)

    @app.post("/api/batch/retarget")
    async def batch_retarget(body: dict) -> dict:
        job = Job(id=uuid.uuid4().hex[:12], kind="batch")
        state.jobs[job.id] = job
        threading.Thread(target=_run_batch_job, args=(job, body), daemon=True).start()
        return {"job_id": job.id}

    # ----------------------------------------------------------------- export

    @app.get("/api/export/{export_token}")
    def export(
        export_token: str,
        fps: float | None = None,
        fmt: str = "csv",
        csv_header: bool = True,
    ):
        rec = state.motions.get(f"export::{export_token}")
        if rec is None:
            raise HTTPException(status_code=404, detail="unknown export token")
        if "path" in rec:
            path = Path(rec["path"])
            media = "application/zip" if path.suffix == ".zip" else "text/csv"
            return FileResponse(path, filename=path.name, media_type=media)

        ret = rec["retargeted"]
        stem = rec["stem"]
        fmt = (fmt or "csv").lower()
        try:
            # The robot may have been unloaded/swapped since this clip was
            # retargeted (``/api/robot`` unload pops ``state.robots``).  A bare
            # ``state.robots[name]`` here used to raise KeyError *outside* this
            # try block → unhandled 500.  Reload the preset on demand; pkl
            # export does not need the model at all, so tolerate its absence.
            model = state.robots.get(rec["robot"])
            if model is None:
                try:
                    from hhtools.robot.loader import load_robot
                    from hhtools.robot.registry import get as _get_preset

                    model = load_robot(_get_preset(rec["robot"]), compile_mjcf=False)
                    state.robots[rec["robot"]] = model
                except Exception as load_err:  # noqa: BLE001
                    if fmt != "pkl":
                        raise RuntimeError(
                            f"robot '{rec['robot']}' is no longer loaded and "
                            f"could not be reloaded for CSV export: {load_err}"
                        ) from load_err
                    model = None  # pkl branch never dereferences the model
            if rec.get("r2r"):
                src_name = rec.get("source_robot")
                src_model = state.robots.get(src_name) if src_name else None
                if src_model is None and src_name:
                    from hhtools.robot.loader import load_robot
                    from hhtools.robot.registry import get as _get_preset

                    src_model = load_robot(_get_preset(src_name), compile_mjcf=False)
                    state.robots[src_name] = src_model
                tgt_name = rec["robot"]
                calib = None
                if src_name and model is not None:
                    from hhtools.retarget import robot_to_robot as r2r

                    calib = r2r.load_r2r_calibration(
                        model.preset.urdf_path.parent, src_name,
                    )
                if src_model is None or not calib:
                    raise RuntimeError(
                        "R2R export needs source robot loaded and calibration saved"
                    )
                path = _write_r2r_export(
                    ret, model, rec["source_motion"], state.export_root,
                    source_model=src_model,
                    calibrated_joint_q=calib,
                    entry=rec.get("r2r_entry") or {
                        "source_path": rec.get("source_path"),
                        "stem": stem,
                        "has_scene": rec.get("has_scene"),
                    },
                    stem=stem, fps=fps, fmt=fmt,
                    csv_header=_parse_csv_header(csv_header),
                )
            else:
                path = _write_export(
                    ret, model, rec["source_motion"], state.export_root,
                    stem=stem, fps=fps, fmt=fmt, backend=rec["backend"],
                    csv_header=_parse_csv_header(csv_header),
                    source_path=rec.get("source_path"),
                )
        except Exception as err:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"export failed: {err}") from err
        if path.suffix == ".zip":
            return FileResponse(
                path,
                filename=f"{stem}_export.zip",
                media_type="application/zip",
            )
        return FileResponse(path, filename=path.name, media_type="text/csv")

    # --------------------------------------------------- robot-to-robot (R2R)

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
    ) -> dict:
        """Upload robot trajectory clip(s); FK runs in a background job with progress."""
        if not files:
            raise HTTPException(status_code=400, detail="no trajectory file uploaded")
        if not source_robot:
            raise HTTPException(status_code=400, detail="source_robot is required")
        drop = state.upload_root / f"r2r_{uuid.uuid4().hex[:8]}"
        drop.mkdir(parents=True, exist_ok=True)
        for uf in files:
            rel = Path(uf.filename or "upload.bin")
            dst = drop / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(await uf.read())
        job = Job(id=uuid.uuid4().hex[:12], kind="r2r_source_upload")
        state.jobs[job.id] = job
        threading.Thread(
            target=_run_r2r_source_upload_job,
            args=(job, drop, source_robot, profile, state),
            daemon=True,
        ).start()
        return {"job_id": job.id}

    @app.get("/api/r2r/scene_glb")
    def r2r_scene_glb(token: str, mesh: str, scale: float | None = None) -> Response:
        """Serve an interaction-object mesh from an uploaded R2R clip folder."""
        from types import SimpleNamespace

        from hhtools.web.serialize import object_mesh_glb

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
                target_robot=target,
                source_robot=source,
                calibrated_joint_q=joint_q,
            )
        except Exception as err:  # noqa: BLE001
            raise HTTPException(
                status_code=400, detail=f"calibration save failed: {err}",
            ) from err
        return {"ok": True, "path": str(path)}

    @app.get("/api/r2r/calibration/status")
    def r2r_calibration_status(target: str, source: str) -> dict:
        from hhtools.retarget import robot_to_robot as r2r
        from hhtools.robot.registry import get as _get_preset

        try:
            preset = _get_preset(target)
            saved = r2r.load_r2r_calibration(preset.urdf_path.parent, source)
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

            from hhtools.retarget import robot_to_robot as r2r

            tgt = _r2r_get_model(target)
            src = _r2r_get_model(source, compile_mjcf=False)
            calib = r2r.load_r2r_calibration(tgt.preset.urdf_path.parent, source)
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
                src, tgt,
                calibrated_joint_q=calib,
                source_motion=motion,
                backend=backend,
                ik_iterations=ik_iters,
                progress_callback=_cb,
            )
            from hhtools.web.serialize import serialize_robot_trajectory

            scaled = _compute_r2r_scaled_preview(src, tgt, motion, calib)
            traj = serialize_robot_trajectory(
                tgt, ret, scaled_preview=scaled, ground_follow=False,
            )
            scaled = _align_scaled_preview_to_robot_playback(tgt, ret, scaled, traj)
            from hhtools.web.r2r_export_bundle import clip_has_export_scene
            from hhtools.web.r2r_scene import compute_r2r_target_scaled_scene

            stem = rec.get("stem") or "r2r"
            clip_dir_path = Path(rec.get("clip_dir") or Path(rec["source_path"]).parent)
            scene_prof = str(rec.get("upload_profile") or "mimic")
            src_has_scene = bool(rec.get("has_scene")) or clip_has_export_scene(
                clip_dir_path, stem=stem, profile=scene_prof,
            )
            tgt_scene = None
            if src_has_scene and rec.get("clip_dir") and rec.get("source_path"):
                tgt_scene = compute_r2r_target_scaled_scene(
                    src,
                    tgt,
                    motion,
                    calib,
                    clip_dir=Path(rec["clip_dir"]),
                    profile=scene_prof,
                    robot_path=Path(rec["source_path"]),
                    num_frames=int(ret.num_frames),
                    framerate=float(ret.sample_rate),
                )
            export_token = uuid.uuid4().hex[:10]
            has_scene = src_has_scene
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
                "r2r_entry": {
                    "source_path": rec.get("source_path"),
                    "clip_dir": rec.get("clip_dir"),
                    "stem": stem,
                    "has_scene": has_scene,
                    "upload_profile": scene_prof,
                },
            }
            job.result = {
                "trajectory": traj,
                "export_token": export_token,
                "stem": rec.get("stem") or "r2r",
                "num_frames": ret.num_frames,
                "source_fps": float(ret.sample_rate),
                "scaled_preview": scaled,
                "scaled_scene": tgt_scene,
                "has_scene": has_scene,
            }
            job.status = "done"
            job.progress = 1.0
            job.message = "done"
        except Exception as err:  # noqa: BLE001
            _log.exception("r2r retarget job failed")
            job.status = "error"
            job.error = str(err)

    @app.post("/api/r2r/retarget")
    async def r2r_retarget(body: dict) -> dict:
        job = Job(id=uuid.uuid4().hex[:12], kind="retarget")
        state.jobs[job.id] = job
        threading.Thread(
            target=_run_r2r_retarget_job, args=(job, body), daemon=True,
        ).start()
        return {"job_id": job.id}

    @app.post("/api/r2r/basket/upload")
    async def r2r_basket_upload(
        files: list[UploadFile] = File(...),
        profile: str = "auto",
    ) -> dict:
        drop = state.upload_root / uuid.uuid4().hex[:8]
        drop.mkdir(parents=True, exist_ok=True)
        wrote = False
        for uf in files:
            rel = Path(uf.filename or "upload.bin")
            dst = drop / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(await uf.read())
            wrote = True
        if not wrote:
            raise HTTPException(status_code=400, detail="empty upload")
        job = Job(id=uuid.uuid4().hex[:12], kind="r2r_basket_upload")
        state.jobs[job.id] = job
        threading.Thread(
            target=_run_r2r_basket_upload_job, args=(job, drop, profile), daemon=True,
        ).start()
        return {"job_id": job.id}

    @app.post("/api/r2r/batch/retarget")
    async def r2r_batch_retarget(body: dict) -> dict:
        job = Job(id=uuid.uuid4().hex[:12], kind="r2r_batch")
        state.jobs[job.id] = job
        threading.Thread(
            target=_run_r2r_batch_job, args=(job, body, state), daemon=True,
        ).start()
        return {"job_id": job.id}

    # ----------------------------------------------------------------- static

    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


def _enrich_basket_entry(entry: dict, fallback: str = "smpl") -> dict:
    """Attach ``reference`` (calibration profile) inferred from dataset / path."""
    out = dict(entry)
    if not (out.get("reference") or "").strip():
        out["reference"] = _entry_reference(out, fallback)
    return out


def _library_entry_from_link(
    folder_label: str,
    lib_dir: Path,
    picked: Path,
    dataset: str | None,
) -> dict:
    """Build a library-shaped entry for a clip under ``~/.config/hhtools/motions``."""
    from hhtools.web.motion_library_links import scan_motions_library

    picked = Path(picked).resolve()
    sp = str(picked)
    for raw in scan_motions_library():
        if raw.get("source_path") == sp:
            return _enrich_basket_entry(raw)

    lib_dir = Path(lib_dir).resolve()
    stem = picked.stem
    sequence_id = picked.name
    try:
        rel = picked.relative_to(lib_dir)
        stem = rel.with_suffix("").as_posix() if rel.parts else picked.stem
    except ValueError:
        pass
    return _enrich_basket_entry({
        "dataset": dataset or "unknown",
        "folder_label": folder_label,
        "sequence_id": sequence_id,
        "source_path": sp,
        "stem": stem,
        "label": f"{folder_label} · {stem}",
        "origin": "link",
    })


def _library_entry_from_upload(
    drop_dir: Path,
    picked: Path,
    dataset: str | None,
    profile: str,
    *,
    upload_profile: str | None = None,
    clip_kind: str = "",
) -> dict:
    """Build a batch-basket / library-shaped entry for an uploaded clip."""
    from hhtools.web.upload_resolve import export_subdir_for_clip

    picked = Path(picked).resolve()
    drop_dir = Path(drop_dir).resolve()
    prof = (upload_profile or profile or "mimic").strip().lower()
    folder_by_profile = {
        "intermimic": "intermimic",
        "meshmimic": "meshmimic",
        "mimic": "mimic",
        "auto": "uploads",
    }
    folder_label = folder_by_profile.get(prof, "uploads")
    try:
        rel = picked.relative_to(drop_dir)
        sequence_id = rel.as_posix()
        stem = picked.parent.name if picked.parent.name == picked.stem else picked.stem
    except ValueError:
        sequence_id = picked.name
        stem = picked.stem
    return _enrich_basket_entry({
        "dataset": dataset or "unknown",
        "folder_label": folder_label,
        "sequence_id": sequence_id,
        "source_path": str(picked),
        "stem": stem,
        "origin": "upload",
        "export_subdir": export_subdir_for_clip(drop_dir, picked),
        "upload_profile": prof,
        "clip_kind": clip_kind,
        "upload_drop": str(drop_dir),
    })


def _load_clip_for_batch(entry_dict: dict, entry, cache):
    """Load a basket clip — uploaded paths bypass adapter-only cache conversion."""
    from hhtools.viewer.cache import _attach_library_folder_label
    from hhtools.web.motion_library_links import resolve_clip_on_disk
    from hhtools.web.upload_resolve import load_clip_at_path

    if entry_dict.get("origin") != "upload":
        entry_dict = dict(entry_dict)
        entry_dict["source_path"] = str(entry.source_path)
        return cache.load_motion(entry)

    resolved = resolve_clip_on_disk(
        entry.source_path,
        extra_names=[entry_dict.get("sequence_id") or ""],
        folder_label=entry_dict.get("folder_label"),
        sequence_id=entry_dict.get("sequence_id"),
        upload_drop=entry_dict.get("upload_drop"),
    )
    entry_dict = dict(entry_dict)
    entry_dict["source_path"] = str(resolved)

    motion, dataset = load_clip_at_path(
        resolved,
        entry_dict.get("upload_profile") or "mimic",
        clip_kind=entry_dict.get("clip_kind") or "",
        load_motion_file=_load_motion_file,
        load_via_adapter=_load_via_adapter,
    )
    if dataset and entry_dict.get("dataset") in (None, "", "unknown"):
        entry_dict["dataset"] = dataset
    _attach_library_folder_label(motion, entry)
    return motion


def _format_duration(seconds: float) -> str:
    """Human-readable duration for batch ETA."""
    if not math.isfinite(seconds) or seconds < 0:
        return "估算中…"
    sec = int(seconds + 0.5)
    if sec < 60:
        return f"{sec} 秒"
    minutes, sec = divmod(sec, 60)
    if minutes < 60:
        return f"{minutes} 分 {sec} 秒"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} 时 {minutes} 分"


# GPU batch: IK frame progress uses only part of each chunk's budget; export + zip
# follow.  Previously IK reached 100% of the chunk span before CSV/ZIP I/O, so
# ETA showed "1 s left" while dozens of large exports still ran.
_BATCH_CHUNK_IK_FRAC = 0.82
_BATCH_CHUNK_EXPORT_FRAC = 0.18
_BATCH_ZIP_PROGRESS = 0.985
_BATCH_EXPORT_WORKERS = 8


def _batch_chunk_ik_progress(
    progress_base: float, progress_span: float, frame_frac: float,
) -> tuple[float, float]:
    ik_clip = 0.05 + 0.95 * min(1.0, max(0.0, frame_frac))
    total = progress_base + progress_span * ik_clip * _BATCH_CHUNK_IK_FRAC
    return total, ik_clip * _BATCH_CHUNK_IK_FRAC


def _batch_chunk_export_progress(
    progress_base: float, progress_span: float, export_frac: float,
) -> tuple[float, float]:
    export_frac = min(1.0, max(0.0, export_frac))
    total = progress_base + progress_span * (
        _BATCH_CHUNK_IK_FRAC + _BATCH_CHUNK_EXPORT_FRAC * export_frac
    )
    clip_p = _BATCH_CHUNK_IK_FRAC + _BATCH_CHUNK_EXPORT_FRAC * export_frac
    return total, clip_p


def _batch_export_retargeted_chunk(
    exports: list[tuple[dict, object, object, object]],
    *,
    model,
    motion_out_dir,
    export_fps,
    fmt: str,
    backend: str,
    csv_header: bool,
    base_prog: float,
    span_prog: float,
    job,
    batch_t0: float,
    done_clips: int,
    total: int,
    written: list[str],
    failure_log,
    state,
    job_id: str,
    out_name: str,
    reference: str,
    errors: list[str],
    failures: list[dict],
):
    """Write retarget results for one GPU chunk (parallel CSV/PKL when >1 clip)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_export = len(exports)
    if n_export == 0:
        return done_clips, failure_log

    workers = 1 if n_export <= 1 else min(_BATCH_EXPORT_WORKERS, n_export)
    prog_lock = threading.Lock()
    export_done = 0

    def _write_one(
        export_i: int,
        e: dict,
        motion: object,
        entry: object,
        ret: object,
    ) -> tuple[int, dict, str | None, str | None]:
        try:
            subdir = _batch_export_subdir(e)
            out_path = _write_export(
                ret, model, motion, motion_out_dir,
                stem=(motion.name or entry.stem),
                fps=export_fps, fmt=fmt, backend=backend,
                subdir=subdir, csv_header=csv_header,
                source_path=e.get("source_path"),
            )
            return export_i, e, str(out_path.relative_to(motion_out_dir)), None
        except Exception as err:  # noqa: BLE001
            return export_i, e, None, str(err)

    def _record_success(rel_path: str) -> None:
        nonlocal export_done, done_clips
        with prog_lock:
            written.append(rel_path)
            export_done += 1
            done_clips += 1
            export_frac = export_done / n_export
            prog, clip_p = _batch_chunk_export_progress(
                base_prog, span_prog, export_frac,
            )
            _set_batch_job_progress(
                job,
                f"导出 · {done_clips}/{total}",
                prog,
                batch_t0,
                clip_progress=clip_p,
            )

    def _record_failure(e: dict, reason: str) -> None:
        nonlocal export_done, done_clips, failure_log
        with prog_lock:
            failure_log = _record_batch_failure(
                failure_log, state, job_id, out_name,
                e, stage="export", reason=reason,
                reference=reference,
                errors=errors, failures=failures,
            )
            export_done += 1
            done_clips += 1
            export_frac = export_done / n_export
            prog, clip_p = _batch_chunk_export_progress(
                base_prog, span_prog, export_frac,
            )
            _set_batch_job_progress(
                job,
                f"导出失败 {e.get('stem', '?')} · {done_clips}/{total}",
                prog,
                batch_t0,
                clip_progress=clip_p,
            )

    if workers == 1:
        for export_i, (e, motion, entry, ret) in enumerate(exports):
            _, _, rel, err = _write_one(export_i, e, motion, entry, ret)
            if err is not None:
                _record_failure(e, err)
            else:
                _record_success(rel)
        return done_clips, failure_log

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [
            pool.submit(_write_one, i, e, motion, entry, ret)
            for i, (e, motion, entry, ret) in enumerate(exports)
        ]
        for fut in as_completed(futs):
            _, e, rel, err = fut.result()
            if err is not None:
                _record_failure(e, err)
            else:
                _record_success(rel)
    return done_clips, failure_log


def _batch_eta_suffix(progress: float, t0: float) -> str:
    """Linear ETA from elapsed time and fractional progress."""
    if progress <= 0.02 or progress >= 0.88:
        return ""
    elapsed = time.monotonic() - t0
    if elapsed <= 0:
        return ""
    remaining = elapsed * (1.0 - progress) / progress
    return f" · 预计剩余 {_format_duration(remaining)}"


def _set_batch_job_progress(
    job: Job | None,
    message: str,
    progress: float,
    t0: float,
    *,
    clip_progress: float | None = None,
) -> None:
    if job is None:
        return
    job.progress = min(0.99, max(0.0, float(progress)))
    if clip_progress is not None:
        job.clip_progress = min(1.0, max(0.0, float(clip_progress)))
    job.message = message + _batch_eta_suffix(job.progress, t0)


def _job_is_batch(job: Job | None) -> bool:
    return job is not None and job.kind in ("batch", "r2r_batch")


def _build_r2r_calibration_session(target_model, source_model) -> dict:
    """Calibration payload for aligning ``target_model`` to a source robot.

    The source robot's forward-kinematics rest pose (canonical joint names) acts
    as the reference skeleton — the robot-to-robot analogue of a human T-pose.
    """
    from hhtools.retarget import robot_to_robot as r2r
    from hhtools.web.calibration_session import (
        _joint_limits_payload,
        _reference_heading_rad,
        _robot_ground_offset_z,
        serialize_reference_skeleton,
    )

    joint_order = [
        j.name for j in target_model.actuated_joints if j.joint_type != "fixed"
    ]
    if not joint_order:
        raise ValueError("target robot has no actuated joints; check URDF / upload")
    joint_q = {n: 0.0 for n in joint_order}

    saved: dict[str, float] | None = None
    urdf_path = getattr(target_model.preset, "urdf_path", None)
    if urdf_path is not None:
        saved = r2r.load_r2r_calibration(urdf_path.parent, source_model.preset.name)
        if saved:
            for name, value in saved.items():
                if name in joint_q:
                    joint_q[name] = float(value)

    ref = r2r.build_source_reference_pose(source_model)
    target_model.apply_configuration(joint_q)
    ground_z = _robot_ground_offset_z(target_model, joint_q)
    try:
        heading = _reference_heading_rad(
            target_model, ref, None, ref.name, current_q=joint_q,
        )
    except Exception:  # noqa: BLE001
        heading = 0.0
    ref_payload = serialize_reference_skeleton(ref, heading_rad=heading)
    return {
        "joint_q": joint_q,
        "joint_limits": _joint_limits_payload(target_model),
        "reference": ref_payload,
        "reference_name": ref.name,
        "ground_offset_z": ground_z,
        "has_saved_calibration": bool(saved),
    }


def _compute_r2r_scaled_preview(source_model, target_model, motion, calibrated_joint_q) -> dict:
    """Yellow scaled skeleton for R2R — uniform overlay + foot grounding (Viser parity)."""
    from hhtools.retarget import robot_to_robot as r2r
    from hhtools.retarget.calibration.calibration import uniform_overlay_scale_for_motion
    from hhtools.retarget.newton_basic.scaler import HumanToRobotScaler
    from hhtools.web.scaled_preview import (
        _uniform_scaled_preview_fallback,
        resolve_scaled_overlay_z_correction,
    )

    cfg, ref = r2r._build_scaler_config(source_model, target_model, calibrated_joint_q)
    human_height = float(ref.height_m)
    ik_canons = (
        frozenset(target_model.preset.ik_map.keys())
        if target_model.preset.ik_map
        else frozenset()
    )
    scaler = HumanToRobotScaler(motion.hierarchy, cfg, human_height=human_height)
    ratio = float(
        uniform_overlay_scale_for_motion(
            cfg, human_height, motion, ik_map_keys=ik_canons,
        )
    )
    z_correction = resolve_scaled_overlay_z_correction(motion, scaler, ratio)
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
    """Shift yellow overlay Z to the grounded robot sole (browser playback frame)."""
    import numpy as np

    from hhtools.web.serialize import (
        _lowest_ankle_z,
        _quat_xyzw_to_rotmat,
        _scaled_overlay_foot_z,
        _scene_min_mesh_z,
    )

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
    ik_map = dict(target_model.preset.ik_map) if target_model.preset.ik_map else {}
    root_rot = _quat_xyzw_to_rotmat(root[3:7])
    ankle_z = _lowest_ankle_z(target_model, ik_map, root_rot)
    if ankle_z is not None:
        # Browser playback: group.z = root.z + mesh_z_lift; ankles ride on the group.
        robot_ref_z = float(root[2] + mesh_lift + ankle_z)
    else:
        min_mesh_z = _scene_min_mesh_z(target_model.trimesh_scene(), root_rot)
        robot_ref_z = (
            float(root[2] + mesh_lift + min_mesh_z) if min_mesh_z is not None else 0.0
        )

    dz = robot_ref_z - float(yellow_z)
    if abs(dz) < 1e-5:
        return scaled_preview

    positions = np.asarray(scaled_preview["positions"], dtype=np.float32).copy()
    positions[:, :, 2] += np.float32(dz)
    out = dict(scaled_preview)
    out["positions"] = np.round(positions, 4).tolist()
    return out


def _run_r2r_source_upload_job(
    job: Job,
    drop: Path,
    source_robot: str,
    profile: str,
    state: SessionState,
) -> None:
    from hhtools.retarget import robot_to_robot as r2r
    from hhtools.web.r2r_upload_resolve import (
        detect_r2r_profile,
        enumerate_r2r_clips,
        validate_r2r_upload,
    )
    from hhtools.web.r2r_export_bundle import clip_has_export_scene
    from hhtools.web.serialize import (
        serialize_motion_skeleton_preview,
        serialize_robot_trajectory,
    )

    try:
        job.progress = 0.02
        job.message = "正在识别轨迹格式…"
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
            from hhtools.robot.registry import get as _get_preset

            src_model = load_robot(_get_preset(source_robot), compile_mjcf=False)
            state.robots[source_robot] = src_model
        traj = r2r.load_source_trajectory(picked, source_model=src_model)

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
            clip_dir, stem=stem, profile=scene_prof,
        )
        if src_has_scene:
            job.progress = 0.88
            job.message = "正在加载地形/物体…"
            from hhtools.web.r2r_scene import load_r2r_clip_scene

            scaled_scene = load_r2r_clip_scene(
                clip_dir,
                profile=scene_prof,
                robot_path=picked,
                num_frames=int(traj.joint_q.shape[0]),
                framerate=float(traj.framerate),
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
                scene_prof, has_scene=bool(src_has_scene),
            ),
        }
        job.status = "done"
        job.progress = 1.0
        job.message = "done"
    except Exception as err:  # noqa: BLE001
        _log.exception("r2r source upload job failed")
        job.status = "error"
        job.error = str(err)


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


def _r2r_entry_from_upload(drop_dir: Path, ref) -> dict:
    from hhtools.web.r2r_upload_resolve import export_subdir_for_r2r_clip

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
    from hhtools.retarget import robot_to_robot as r2r

    return {
        "dataset": "r2r",
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
            prof, has_scene=bool(ref.has_scene),
        ),
    }


def _run_r2r_basket_upload_job(job: Job, drop: Path, profile: str) -> None:
    from hhtools.web.r2r_upload_resolve import enumerate_r2r_clips, validate_r2r_upload

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
        job.status = "done"
        job.progress = 1.0
        job.message = f"已识别 {len(entries)} 个机器人轨迹 clip"
    except Exception as err:  # noqa: BLE001
        _log.exception("r2r basket upload failed")
        job.status = "error"
        job.error = str(err)


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
    from hhtools.web.r2r_scene import attach_r2r_clip_scene_to_motion

    return attach_r2r_clip_scene_to_motion(
        motion,
        Path(clip_dir),
        profile=profile,
        robot_path=Path(robot_path),
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
                job, 0.08, "正在构建 Interaction-Mesh 场景…",
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
    job: Job | None = None,
):
    from hhtools.retarget import robot_to_robot as r2r

    traj = r2r.load_source_trajectory(traj_path, source_model=source_model)
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
        export_fps = _parse_optional_fps(body.get("export_fps", body.get("fps")))
        fmt = (body.get("format") or "csv").lower()
        csv_header = _parse_csv_header(body.get("csv_header", True))
        out_name = body.get("out_dir") or "r2r_batch_export"
        backend = (body.get("backend") or "newton").strip().lower()

        from hhtools.retarget import robot_to_robot as r2r

        tgt = state.robots.get(target)
        if tgt is None:
            from hhtools.robot.loader import load_robot
            from hhtools.robot.registry import get as _get_preset

            tgt = load_robot(_get_preset(target), compile_mjcf=True)
            state.robots[target] = tgt
        src = state.robots.get(source)
        if src is None:
            from hhtools.robot.loader import load_robot
            from hhtools.robot.registry import get as _get_preset

            src = load_robot(_get_preset(source), compile_mjcf=False)
            state.robots[source] = src
        calib = r2r.load_r2r_calibration(tgt.preset.urdf_path.parent, source)
        if not calib:
            raise ValueError(
                f"target {target!r} is not calibrated against source {source!r}"
            )

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

        for i, e in enumerate(entries):
            stem = e.get("stem") or Path(e.get("source_path", "clip")).stem
            _set_batch_job_progress(
                job, f"{i + 1}/{total}: {stem}", i / max(1, total), batch_t0,
                clip_progress=0.0,
            )
            traj_path = Path(e["source_path"])
            try:
                ret, motion = _r2r_retarget_from_path(
                    src, tgt, traj_path,
                    calibrated_joint_q=calib,
                    retarget_fps=retarget_fps,
                    ik_iters=ik_iters,
                    backend=backend,
                    profile=str(e.get("upload_profile") or "mimic"),
                    has_scene=bool(e.get("has_scene")),
                    job=job,
                )
            except Exception as err:  # noqa: BLE001
                errors.append(f"{stem}: {err}")
                failures.append({"stem": stem, "stage": "retarget", "reason": str(err)})
                _set_batch_job_progress(
                    job, f"失败 {stem} · {i + 1}/{total}",
                    (i + 1) / max(1, total), batch_t0, clip_progress=1.0,
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
                subdir = _batch_export_subdir(e)
                out_path = _write_r2r_export(
                    ret, tgt, motion, out_dir,
                    source_model=src,
                    calibrated_joint_q=calib,
                    entry=e,
                    stem=stem, fps=export_fps, fmt=fmt,
                    subdir=subdir, csv_header=csv_header,
                )
                written.append(str(out_path.relative_to(out_dir)))
            except Exception as err:  # noqa: BLE001
                errors.append(f"{stem} export: {err}")
                failures.append({"stem": stem, "stage": "export", "reason": str(err)})
            _set_batch_job_progress(
                job, f"完成 {stem} · {i + 1}/{total}",
                (i + 1) / max(1, total), batch_t0, clip_progress=1.0,
            )

        zip_path = shutil.make_archive(str(out_dir.parent / out_name), "zip", root_dir=str(out_dir))
        shutil.rmtree(out_dir, ignore_errors=True)
        job.result = {
            "written": written,
            "errors": errors,
            "failures": failures,
            "download_name": f"{out_name}.zip",
            "artifact_path": str(zip_path),
            "format": fmt,
        }
        job.status = "done"
        job.progress = 1.0
        job.message = f"完成 {len(written)}/{total}"
    except Exception as err:  # noqa: BLE001
        _log.exception("r2r batch job failed")
        job.status = "error"
        job.error = str(err)


def _set_retarget_job_clip_progress(job: Job | None, value: float, message: str) -> None:
    """Update per-clip progress during batch retarget; otherwise ``job.progress``."""
    if job is None:
        return
    v = min(0.99, max(0.0, float(value)))
    if _job_is_batch(job):
        job.clip_progress = v
    else:
        job.progress = v
    job.message = message


def _batch_export_subdir(entry: dict) -> str | None:
    """Export folder: preserve drag-in tree for uploads, else per-dataset."""
    if entry.get("origin") == "upload":
        sub = (entry.get("export_subdir") or "").strip().replace("\\", "/")
        return sub or None
    return _dataset_subdir(entry)


def _entry_reference(entry: dict, fallback: str) -> str:
    """Map a basket row to the calibration reference it needs."""
    explicit = (entry.get("reference") or "").strip()
    if explicit:
        return explicit
    dataset = (entry.get("dataset") or "").strip()
    if dataset in _DATASET_TO_REFERENCE:
        return _DATASET_TO_REFERENCE[dataset]
    return fallback


def _apply_limit_frames(motion, limit_frames):
    if not limit_frames:
        return motion
    lf = int(limit_frames)
    if motion.num_frames <= lf:
        return motion
    motion.positions = motion.positions[:lf]
    motion.quaternions = motion.quaternions[:lf]
    for o in motion.objects:
        o.positions = o.positions[:lf]
        o.quaternions = o.quaternions[:lf]
    return motion


def _load_batch_motion(entry_dict: dict, entry, cache, *, retarget_fps, limit_frames):
    from hhtools.viewer.library import LibraryEntry

    motion = _load_clip_for_batch(entry_dict, entry, cache)
    motion = _ground_motion_for_web(motion)
    motion, _ = _motion_for_retarget(motion, retarget_fps)
    return _apply_limit_frames(motion, limit_frames)


def _record_batch_failure(
    failure_log,
    state,
    job_id: str,
    out_name: str,
    entry: dict,
    *,
    stage: str,
    reason: str,
    reference: str | None,
    errors: list[str],
    failures: list[dict],
):
    from hhtools.web.batch_failure_log import BatchFailureLog, open_batch_failure_log

    if failure_log is None:
        failure_log = open_batch_failure_log(state.save_dir, job_id, out_name)
    item = failure_log.record(
        entry, stage=stage, reason=reason, reference=reference,
    )
    failures.append(item)
    errors.append(f"{item['stem']} [{stage}]: {reason}")
    return failure_log


def _run_batch_entries_sequential(
    entries,
    model,
    robot_name,
    reference,
    backend,
    ik_iters,
    human_height,
    limit_frames,
    retarget_fps,
    export_fps,
    fmt,
    csv_header,
    out_dir,
    state,
    *,
    job,
    job_id,
    out_name,
    written,
    errors,
    failures,
    failure_log,
    batch_t0: float,
    foot_clamp_anti_penetration: bool = False,
) -> BatchFailureLog | None:
    from hhtools.web.motion_library_links import library_entry_for_load

    total = len(entries)
    for i, e in enumerate(entries):
        _set_batch_job_progress(
            job,
            f"{i + 1}/{total}: {e.get('stem', '?')}",
            i / max(1, total),
            batch_t0,
            clip_progress=0.0,
        )
        ref = _entry_reference(e, reference)
        entry = library_entry_for_load(
            dataset=e["dataset"],
            folder_label=e["folder_label"],
            sequence_id=e["sequence_id"],
            source_path=e["source_path"],
            upload_drop=e.get("upload_drop"),
        )
        try:
            motion = _load_batch_motion(
                e, entry, state.cache,
                retarget_fps=retarget_fps, limit_frames=limit_frames,
            )
        except Exception as err:  # noqa: BLE001
            failure_log = _record_batch_failure(
                failure_log, state, job_id, out_name,
                e, stage="load", reason=str(err), reference=ref,
                errors=errors, failures=failures,
            )
            _set_batch_job_progress(
                job,
                f"加载失败 {e.get('stem', '?')} · {i + 1}/{total}",
                (i + 1) / max(1, total),
                batch_t0,
                clip_progress=1.0,
            )
            continue
        try:
            ret = _retarget_single(
                model, robot_name, motion, ref, backend,
                ik_iters, human_height, limit_frames, job,
                state=state,
                foot_clamp_anti_penetration=foot_clamp_anti_penetration,
            )
        except Exception as err:  # noqa: BLE001
            failure_log = _record_batch_failure(
                failure_log, state, job_id, out_name,
                e, stage="retarget", reason=str(err), reference=ref,
                errors=errors, failures=failures,
            )
            _set_batch_job_progress(
                job,
                f"重定向失败 {e.get('stem', '?')} · {i + 1}/{total}",
                (i + 1) / max(1, total),
                batch_t0,
                clip_progress=1.0,
            )
            continue
        try:
            subdir = _batch_export_subdir(e)
            out_path = _write_export(
                ret, model, motion, out_dir,
                stem=(motion.name or entry.stem), fps=export_fps,
                fmt=fmt, backend=backend, subdir=subdir,
                csv_header=csv_header,
                source_path=e.get("source_path"),
            )
            written.append(str(out_path.relative_to(out_dir)))
            _set_batch_job_progress(
                job,
                f"完成 {e.get('stem', '?')} · {i + 1}/{total}",
                (i + 1) / max(1, total),
                batch_t0,
                clip_progress=1.0,
            )
        except Exception as err:  # noqa: BLE001
            failure_log = _record_batch_failure(
                failure_log, state, job_id, out_name,
                e, stage="export", reason=str(err), reference=ref,
                errors=errors, failures=failures,
            )
            _set_batch_job_progress(
                job,
                f"失败 {e.get('stem', '?')} · {i + 1}/{total}",
                (i + 1) / max(1, total),
                batch_t0,
                clip_progress=1.0,
            )
    _set_batch_job_progress(
        job, f"批量完成 · {total}/{total}", 1.0, batch_t0, clip_progress=1.0,
    )
    return failure_log


def _retarget_newton_batch_chunk(
    loaded: list[tuple[dict, object, object]],
    *,
    model,
    robot_name: str,
    reference: str,
    ik_iters: int,
    human_height: float,
    state,
    job,
    job_id: str,
    out_name: str,
    failure_log,
    failures: list[dict],
    errors: list[str],
    progress_base: float,
    progress_span: float,
    batch_t0: float,
    chunk_label: str,
    foot_clamp_anti_penetration: bool = False,
) -> tuple[list[tuple[dict, object, object, object]], object]:
    """Retarget pre-loaded clips; multi-env GPU when ``len(loaded) > 1``."""
    from hhtools.retarget.calibration import load_calibration, resolve_calibration_file
    from hhtools.retarget.newton_basic import NewtonBasicPipeline
    from hhtools.retarget.newton_basic._warp_config import configure as configure_warp_cache
    from hhtools.robot.registry import get as get_preset

    if not loaded:
        return [], failure_log

    if len(loaded) == 1:
        e, motion, entry = loaded[0]
        ret = _retarget_single(
            model, robot_name, motion, reference, "newton",
            ik_iters, human_height, None, job,
            state=state,
            foot_clamp_anti_penetration=foot_clamp_anti_penetration,
        )
        return [(e, motion, entry, ret)], failure_log

    # Clips that share a calibration ``reference`` can still carry *different
    # source skeletons* — e.g. OMOMO (SMPL-H 24-joint) and holosoma (53-joint)
    # both map to ``smplx``.  A single GPU batch builds **one** ScalerConfig
    # keyed by the first clip's bone names; ``adapt_scaler_config_for_hierarchy``
    # then raises "no joint_scales entries resolvable" on the mismatched clips.
    # Sub-group by skeleton signature so each GPU batch shares one hierarchy.
    # Single-skeleton chunks (the common case) fall through unchanged.
    from collections import defaultdict as _defaultdict

    by_skeleton: dict[tuple, list] = _defaultdict(list)
    for item in loaded:
        sig = tuple(item[1].hierarchy.bone_names)
        by_skeleton[sig].append(item)
    if len(by_skeleton) > 1:
        merged: list[tuple[dict, object, object, object]] = []
        for group in by_skeleton.values():
            sub, failure_log = _retarget_newton_batch_chunk(
                group,
                model=model,
                robot_name=robot_name,
                reference=reference,
                ik_iters=ik_iters,
                human_height=human_height,
                state=state,
                job=job,
                job_id=job_id,
                out_name=out_name,
                failure_log=failure_log,
                failures=failures,
                errors=errors,
                progress_base=progress_base,
                progress_span=progress_span,
                batch_t0=batch_t0,
                chunk_label=chunk_label,
                foot_clamp_anti_penetration=foot_clamp_anti_penetration,
            )
            merged.extend(sub)
        return merged, failure_log

    from hhtools.robot.retarget_profile import (
        build_feet_stabilizer_config,
        build_pipeline_config_for_preset,
        bundled_scaler_path,
        resolve_retarget_scaler_config,
    )

    preset = get_preset(robot_name)
    cal_path = resolve_calibration_file(preset.urdf_path.parent, reference)
    if cal_path is None and bundled_scaler_path(preset, reference) is None:
        raise ValueError(
            f"robot {robot_name!r} not calibrated for reference {reference!r}; calibrate first"
        )

    _join_robot_prewarm(state, robot_name, job)
    configure_warp_cache()
    calibration = load_calibration(cal_path) if cal_path is not None else None
    scaler_cfg = resolve_retarget_scaler_config(
        preset,
        reference,
        calibration=calibration,
        model=model,
        motion=loaded[0][1],
        human_height=human_height,
    )
    feet_cfg = build_feet_stabilizer_config(preset, reference, model=model)
    _set_batch_job_progress(
        job,
        f"并行 IK {chunk_label} · 参考 {reference} · 编译内核…",
        progress_base + 0.02 * progress_span,
        batch_t0,
        clip_progress=0.02,
    )

    pipeline = NewtonBasicPipeline(
        model,
        scaler_config=scaler_cfg,
        pipeline_config=build_pipeline_config_for_preset(
            preset, reference, ik_iterations=ik_iters,
            foot_clamp_anti_penetration=foot_clamp_anti_penetration,
        ),
        feet_stabilizer_config=feet_cfg,
        human_height=human_height,
        configure_warp=False,
    )

    motions = [m for _, m, _ in loaded]
    _ik_total = {"n": 0}

    def _frame_cb(done: int, total: int) -> None:
        if job is None:
            return
        # ``run_batch`` reports IK as ``done/max_frames``, then post-IK as
        # ``max_frames + clip_i / max_frames + n_clips`` (total grows).
        if _ik_total["n"] == 0:
            _ik_total["n"] = max(1, int(total))
        if int(total) > _ik_total["n"]:
            post_done = max(0, int(done) - _ik_total["n"])
            post_tot = max(1, int(total) - _ik_total["n"])
            total_p, clip_p = _batch_chunk_ik_progress(
                progress_base, progress_span, 1.0,
            )
            post_frac = post_done / post_tot
            total_p = progress_base + progress_span * (
                _BATCH_CHUNK_IK_FRAC + 0.5 * _BATCH_CHUNK_EXPORT_FRAC * post_frac
            )
            _set_batch_job_progress(
                job,
                (
                    f"并行 IK {chunk_label} · 参考 {reference} · "
                    f"后处理 {post_done}/{post_tot}"
                ),
                total_p,
                batch_t0,
                clip_progress=clip_p,
            )
            return
        _ik_total["n"] = max(_ik_total["n"], int(total))
        ik_total = _ik_total["n"]
        frac = min(1.0, float(done) / max(1, ik_total))
        total_p, clip_p = _batch_chunk_ik_progress(
            progress_base, progress_span, frac,
        )
        _set_batch_job_progress(
            job,
            (
                f"并行 IK {chunk_label} · 参考 {reference} · "
                f"帧 {min(int(done), ik_total)}/{ik_total}（本批最长 clip）"
            ),
            total_p,
            batch_t0,
            clip_progress=clip_p,
        )

    try:
        results = pipeline.run_batch(motions, progress_callback=_frame_cb)
    except Exception as err:
        from hhtools.retarget.newton_basic.batch_limits import (
            is_ik_shared_memory_error,
            shared_memory_error_hint,
        )

        if not is_ik_shared_memory_error(err):
            raise
        _log.warning(
            "GPU batch IK failed (shared memory), falling back to sequential: %s",
            err,
        )
        hint = shared_memory_error_hint(getattr(pipeline.ctx, "joint_dof_count", None))
        if job is not None:
            _set_batch_job_progress(
                job,
                f"内核共享内存不足，改逐条 IK ×{len(loaded)}（参考 {reference}）…",
                progress_base + 0.05 * progress_span,
                batch_t0,
                clip_progress=0.0,
            )
        out: list[tuple[dict, object, object, object]] = []
        for i, (e, motion, entry) in enumerate(loaded):
            if job is not None:
                _set_batch_job_progress(
                    job,
                    f"逐条 IK {i + 1}/{len(loaded)} · {e.get('stem', '?')}（{hint}）",
                    progress_base + progress_span * (i / max(1, len(loaded))),
                    batch_t0,
                    clip_progress=0.0,
                )
            try:
                ret = _retarget_single(
                    model, robot_name, motion, reference, "newton",
                    ik_iters, human_height, None, job,
                    state=state,
                    foot_clamp_anti_penetration=foot_clamp_anti_penetration,
                )
                out.append((e, motion, entry, ret))
            except Exception as single_err:  # noqa: BLE001
                failure_log = _record_batch_failure(
                    failure_log, state, job_id, out_name,
                    e, stage="retarget", reason=str(single_err),
                    reference=reference,
                    errors=errors, failures=failures,
                )
        if not out:
            raise RuntimeError(
                f"GPU batch IK failed and all {len(loaded)} sequential retries failed "
                f"(first error: {failures[-1]['reason'] if failures else err})"
            ) from err
        return out, failure_log

    if len(results) != len(loaded):
        raise RuntimeError(
            f"run_batch returned {len(results)} results for {len(loaded)} motions"
        )
    return [
        (loaded[i][0], loaded[i][1], loaded[i][2], results[i])
        for i in range(len(loaded))
    ], failure_log


def _ground_motion_for_web(motion):
    """Centre the root at the origin (XY) and snap the lowest point to z=0.

    Matches the Viser viewer's default ``center_motion_root_xy`` +
    ``snap_motion_to_ground`` so terrain, objects and the human all sit on the
    same ground plane the browser draws its grid on.  ``margin=0`` puts the
    lowest foot/terrain point exactly on z=0 (the user's "最低点在水平面上").
    """
    try:
        from hhtools.core.coord import to_up_axis
        from hhtools.retarget.newton_basic.rest_pose import normalize_mocap_bvh_clip
        from hhtools.viewer.anatomy import center_motion_root_xy, snap_motion_to_ground

        motion = normalize_mocap_bvh_clip(motion)
        if motion.up_axis != "Z":
            motion = to_up_axis(motion, "Z")
        motion = center_motion_root_xy(motion)
        motion = snap_motion_to_ground(motion, margin=0.0)
    except Exception:  # noqa: BLE001 — never block loading on grounding
        _log.warning("grounding failed; using raw motion", exc_info=True)
    return motion


def _run_dataset_robot_preview_job(job: Job, body: dict, state: SessionState) -> None:
    from hhtools.web.motion_progress import MotionLoadProgress

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
        job.status = "done"
        job.progress = 1.0
        job.message = "完成"
    except Exception as err:  # noqa: BLE001
        _log.exception("dataset robot preview failed")
        job.status = "error"
        job.error = str(err)


def _ensure_robot_model(state: SessionState, robot_name: str | None):
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
    raise ValueError(
        "无法加载机器人模型以预览轨迹；请先在「机器人」面板加载对应机器人"
    )


def _build_robot_export_playback(
    source_path: Path,
    state: SessionState,
    *,
    robot_name: str | None = None,
    progress=None,
) -> dict[str, Any]:
    """Parse a robot export CSV and build mesh playback + optional scene payload."""
    from hhtools.retarget import robot_to_robot as r2r
    from hhtools.web.r2r_scene import _parse_comment_meta, load_r2r_clip_scene
    from hhtools.web.r2r_upload_resolve import detect_r2r_profile
    from hhtools.web.serialize import serialize_robot_trajectory

    path = Path(source_path).resolve()
    clip_dir = path.parent
    inferred = str(_parse_comment_meta(path).get("robot") or "").strip()
    pick = (robot_name or inferred or None)
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
    state: SessionState,
    *,
    progress=None,
):
    """FK a retarget robot CSV export into a :class:`Motion` for 3D preview."""
    from hhtools.retarget import robot_to_robot as r2r
    from hhtools.web.r2r_scene import (
        _parse_comment_meta,
        attach_r2r_clip_scene_to_motion,
    )
    from hhtools.web.r2r_upload_resolve import detect_r2r_profile

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
        )
    except Exception as err:  # noqa: BLE001 — scene is optional for preview
        _log.warning("robot export scene attach skipped for %s: %s", path, err)

    if cb is not None:
        cb(1.0, "机器人轨迹就绪")
    return motion


def _load_motion_for_web(entry, cache, *, progress=None):
    """Load a library clip with SMPL mesh baking when the dataset supports it."""
    from hhtools.io.datasets import registered_datasets
    from hhtools.viewer.cache import _attach_library_folder_label

    cb = progress.as_callback() if progress is not None else None
    if entry.dataset in _SMPL_MESH_DATASETS:
        adapter_cls = registered_datasets().get(entry.dataset)
        if adapter_cls is not None:
            adapter = adapter_cls(entry.source_path.parent)
            try:
                if cb is not None:
                    cb(0.0, f"读取 {entry.stem}…")
                motion = adapter.load_motion(
                    entry.adapter_sequence_id,
                    with_mesh=True,
                    progress_callback=cb,
                )
                _attach_library_folder_label(motion, entry)
                return motion
            except Exception as err:
                _log.warning(
                    "with_mesh load failed for %s (%s); falling back to cache: %s",
                    entry.stem,
                    entry.dataset,
                    err,
                )
    return cache.load_motion(entry, progress_callback=cb)


def _load_motion_file(path: Path, *, progress=None):
    """Load a motion file with mesh enabled for GLB when possible."""
    cb = progress.as_callback() if progress is not None else None
    suf = path.suffix.lower()
    if suf in (".glb", ".gltf"):
        from hhtools.io.glb import load_glb

        if cb is not None:
            cb(0.1, f"解析 GLB {path.name}…")
        motion = load_glb(path, with_mesh=True)
        if cb is not None:
            cb(1.0, "GLB 解析完成")
        return motion
    if cb is not None:
        cb(0.1, f"读取 {path.name}…")
    from hhtools.io.base import load_motion

    motion = load_motion(path)
    if cb is not None:
        cb(1.0, f"已读取 {path.name}")
    return motion


def _load_via_adapter(path: Path):
    """Best-effort dataset-adapter load for non-io.base extensions."""
    suf = path.suffix.lower()
    try:
        if suf == ".pkl":
            from hhtools.io.datasets.omomo import OmomoAdapter
            from hhtools.io.datasets.parc_ms import ParcMsAdapter

            parent = path.parent
            try:
                if parent.name == path.stem:
                    return (
                        OmomoAdapter(root=parent.parent).load_motion(
                            f"{parent.name}/{path.name}"
                        ),
                        "omomo",
                    )
                return OmomoAdapter(root=parent).load_motion(path.name), "omomo"
            except Exception:
                pass
            try:
                if parent.name == path.stem:
                    return (
                        ParcMsAdapter(root=parent.parent).load_motion(
                            f"{parent.name}/{path.name}"
                        ),
                        "parc_ms",
                    )
                return ParcMsAdapter(root=parent).load_motion(path.name), "parc_ms"
            except Exception:
                return None, None
        if suf == ".npy":
            from hhtools.io.datasets.meshmimic_holosoma import MeshmimicHolosomaAdapter

            parent = path.parent
            if parent.name == path.stem:
                seq = f"{parent.name}/{path.name}"
                m = MeshmimicHolosomaAdapter(root=parent.parent).load_motion(seq)
                return m, "meshmimic_holosoma"
        if suf == ".npz":
            from hhtools.io.datasets.amass import AmassAdapter

            return (
                AmassAdapter(root=path.parent).load_motion(path.name, with_mesh=True),
                "amass",
            )
    except Exception:
        return None, None
    return None, None


def _dataset_subdir(entry: dict) -> str:
    """Per-dataset export subfolder (e.g. ``AMASS``, ``PHUMA``).

    Prefers the dataset adapter name, falling back to the library folder label.
    """
    import re

    raw = entry.get("dataset") or entry.get("folder_label") or "misc"
    name = str(raw).strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "misc"
    return name.upper() if name.islower() and len(name) <= 12 else name


def _parse_csv_header(value) -> bool:
    """Truthy unless the client explicitly disables comments + column headers."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("0", "false", "no", "off", "none", "raw", "numeric"):
        return False
    return True


def _parse_optional_fps(value) -> float | None:
    """Positive target fps from API/JSON, or ``None`` to keep the source rate."""
    if value is None or value == "":
        return None
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return None
    return fps if fps > 0 else None


def _motion_for_retarget(motion, retarget_fps: float | None):
    """Optionally down/up-sample the clip before IK/MPC (fewer frames ⇒ faster).

    Returns ``(motion_for_solver, effective_fps)``.  When the rate is unchanged
    the same ``Motion`` instance is returned (read-only use during retarget).
    """
    from hhtools.core.resample import resample_motion_with_objects

    src = float(motion.framerate)
    target = _parse_optional_fps(retarget_fps)
    if target is None or abs(target - src) < 1e-6:
        return motion, src
    return resample_motion_with_objects(motion, target), float(target)


def _resample_retargeted(retargeted, fps: float | None):
    """Return a (joint_q, sample_rate) pair, optionally resampled to ``fps``."""
    import numpy as np

    from hhtools.web.serialize import resample_joint_q

    src = float(getattr(retargeted, "sample_rate", 30.0))
    if fps is None or fps <= 0 or abs(fps - src) < 1e-6:
        return np.asarray(retargeted.joint_q, dtype=np.float32), src
    rc = int(getattr(retargeted, "root_coord_count", 7))
    jq = resample_joint_q(retargeted.joint_q, src, float(fps), root_coord_count=rc)
    return jq, float(fps)


def _write_r2r_export(
    retargeted,
    target_model,
    source_motion,
    out_root,
    *,
    source_model,
    calibrated_joint_q: dict[str, float],
    entry: dict,
    stem: str,
    fps: float | None,
    fmt: str,
    subdir: str | None = None,
    csv_header: bool = True,
):
    """R2R clip bundle: target robot traj + rescaled terrain/object sidecars."""
    from hhtools.web.export_bundle import resolve_clip_export_dir
    from hhtools.web.r2r_export_bundle import (
        clip_has_export_scene,
        write_r2r_export_bundle,
    )

    out_dir = Path(out_root)
    if subdir:
        out_dir = out_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = write_r2r_export_bundle(
        retargeted,
        target_model,
        source_motion,
        source_model=source_model,
        calibrated_joint_q=calibrated_joint_q,
        entry=entry,
        out_root=out_dir,
        stem=stem,
        fps=fps,
        fmt=fmt,
        resample_fn=_resample_retargeted,
        csv_header=csv_header,
    )
    if subdir is not None and path.suffix == ".zip":
        import shutil

        from hhtools.web.r2r_export_bundle import (
            clip_has_export_scene,
            resolve_r2r_source_clip_dir,
        )

        source_clip_dir = resolve_r2r_source_clip_dir(entry)
        profile = str(entry.get("upload_profile") or "")
        has_scene = bool(entry.get("has_scene")) or (
            clip_has_export_scene(
                source_clip_dir, stem=stem, profile=profile,
            )
            if source_clip_dir is not None
            else False
        )
        clip_dir = resolve_clip_export_dir(
            out_dir, stem, entry.get("source_path"), has_scene=has_scene,
        )
        clip_dir.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(path), str(clip_dir))
        path.unlink(missing_ok=True)
        return clip_dir
    return path


def _write_export(
    retargeted,
    model,
    source_motion,
    out_root,
    *,
    stem: str,
    fps: float | None,
    fmt: str,
    backend: str,
    subdir: str | None = None,
    csv_header: bool = True,
    source_path: str | Path | None = None,
):
    """Write a browser-downloadable CSV/PKL bundle (zip when scene props exist)."""
    from hhtools.web.export_bundle import (
        motion_has_scene,
        resolve_clip_export_dir,
        write_retarget_export_bundle,
    )

    out_dir = Path(out_root)
    if subdir:
        out_dir = out_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = write_retarget_export_bundle(
        retargeted,
        model,
        source_motion,
        out_dir,
        stem=stem,
        fps=fps,
        fmt=fmt,
        backend=backend,
        resample_fn=_resample_retargeted,
        csv_header=csv_header,
        source_path=source_path,
    )
    # Batch jobs unpack per-clip zips into the job tree (final zip later).
    if subdir is not None and path.suffix == ".zip":
        import shutil

        clip_dir = resolve_clip_export_dir(
            out_dir,
            stem,
            source_path,
            has_scene=motion_has_scene(source_motion),
        )
        clip_dir.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(path), str(clip_dir))
        path.unlink(missing_ok=True)
        return clip_dir
    return path


def _read_yaml_retarget_references(drop: Path) -> dict | None:
    """Extract ``retarget.references`` from a robot dir's yaml (pre-rebuild)."""
    import yaml

    for yp in sorted(drop.glob("*.yaml")):
        if yp.name.startswith("retarget_calibration_"):
            continue
        try:
            data = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            continue
        refs = (data.get("retarget") or {}).get("references")
        if isinstance(refs, dict) and refs:
            return refs
    return None


def _merge_retarget_references(yaml_path: str | Path | None, refs: dict) -> None:
    """Re-attach preserved ``retarget.references`` onto a freshly scaffolded yaml."""
    import yaml

    if not yaml_path or not refs:
        return
    p = Path(yaml_path)
    if not p.is_file():
        return
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    rt = data.get("retarget")
    if not isinstance(rt, dict):
        rt = {}
        data["retarget"] = rt
    existing = rt.get("references")
    if not isinstance(existing, dict):
        existing = {}
    existing.update(refs)
    rt["references"] = existing
    p.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _request_human_height(body: dict, preset, reference: str) -> float:
    """Resolve source-human height from the request, with a scaler-aware default.

    Falls back to a reference-family canonical height (1.65 m for SMPL / SOMA /
    LAFAN / GLB) when the UI does not send an explicit height.
    """
    from hhtools.robot.retarget_profile import default_human_height

    raw = body.get("human_height")
    if raw is not None:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = 0.0
        if val > 0.1:
            return val
    return default_human_height(preset, reference)


def _compute_scaled_scene(
    model,
    robot_name: str,
    motion,
    reference: str,
    human_height: float,
) -> dict | None:
    """Scaled terrain + objects in the robot retarget frame (no Viser 5 m offset).

    Mirrors :func:`hhtools.viewer.app._publish_robot_objects` but keeps everything
    co-located with the robot preview in the web UI.
    """
    import numpy as np

    if motion.terrain is None and not motion.objects:
        return None
    from hhtools.core.grounding import (
        human_source_floor_z_world,
        terrain_heightfield_z_offset_world,
    )
    from hhtools.core.scene import SceneObject
    from hhtools.retarget.calibration.calibration import uniform_overlay_scale_for_motion
    from hhtools.retarget.newton_basic.scaler import HumanToRobotScaler
    from hhtools.web.scaled_preview import (
        resolve_scaled_overlay_z_correction,
        resolve_web_scaler_config,
    )
    from hhtools.web.serialize import (
        _MAX_PLAYBACK_FRAMES,
        _downsample_indices,
        _serialize_object_meta,
        _serialize_terrain,
    )

    try:
        scaler_cfg = resolve_web_scaler_config(
            model, motion, reference, float(human_height),
        )
    except ValueError:
        return None
    scaler = HumanToRobotScaler(
        motion.hierarchy, scaler_cfg, human_height=float(human_height),
    )
    ik_canons = frozenset(model.preset.ik_map.keys()) if model.preset.ik_map else frozenset()
    ratio = float(
        uniform_overlay_scale_for_motion(
            scaler_cfg, float(human_height), motion, ik_map_keys=ik_canons,
        )
    )

    z_min = float(human_source_floor_z_world(motion))
    z_terrain = float(terrain_heightfield_z_offset_world(motion, z_min))
    z_correction = float(resolve_scaled_overlay_z_correction(motion, scaler, ratio))
    idx = _downsample_indices(
        motion.num_frames, _MAX_PLAYBACK_FRAMES, motion=motion,
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
        # PARC's ``*_foot`` joints are ankles sitting a fixed offset above the
        # sole contact surface, so ``terrain_data`` (authored to the sole) ends
        # up that much below the foot joints / robot mesh sole the viewer snaps
        # to them.  Lift parc_ms terrain by the scaled offset so the surface,
        # the yellow skeleton foot, and the robot sole coincide.
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
) -> dict:
    """Dense uniform scaled skeleton (Viser ``_compute_scaled_preview`` parity)."""
    from hhtools.web.scaled_preview import compute_web_scaled_preview

    return compute_web_scaled_preview(
        model, motion, reference, human_height,
    )


def _retarget_single(
    model,
    robot_name,
    motion,
    reference,
    backend,
    ik_iters,
    human_height,
    limit_frames,
    job,
    *,
    state: SessionState | None = None,
    foot_clamp_anti_penetration: bool | None = None,
):
    """Run one clip through the requested backend, returning RetargetedMotion."""
    from hhtools.retarget.calibration import resolve_calibration_file
    from hhtools.robot.registry import get as get_preset
    from hhtools.robot.retarget_profile import bundled_scaler_path

    preset = get_preset(robot_name)
    cal_path = resolve_calibration_file(preset.urdf_path.parent, reference)
    if cal_path is None and bundled_scaler_path(preset, reference) is None:
        raise ValueError(
            f"robot {robot_name!r} not calibrated for reference {reference!r}; calibrate first"
        )

    if limit_frames:
        lf = int(limit_frames)
        if motion.num_frames > lf:
            motion.positions = motion.positions[:lf]
            motion.quaternions = motion.quaternions[:lf]
            for o in motion.objects:
                o.positions = o.positions[:lf]
                o.quaternions = o.quaternions[:lf]

    if backend == "interaction_mesh":
        from hhtools.retarget.interaction_mesh.config import (
            InteractionMeshPipelineConfig,
        )
        from hhtools.retarget.interaction_mesh.pipeline import InteractionMeshPipeline

        if job is not None:
            _set_retarget_job_clip_progress(
                job, 0.04, "正在构建 Interaction-Mesh 场景（新机器人首次较慢）…",
            )
        im_cfg = InteractionMeshPipelineConfig()
        # Same UI gate as Newton: unchecked 「脚穿地修正」→ skip mesh foot clamps.
        if foot_clamp_anti_penetration is not None:
            im_cfg.post_mpc_foot_clamps = bool(foot_clamp_anti_penetration)
            if not im_cfg.post_mpc_foot_clamps:
                im_cfg.min_foot_clearance_m = 0.0
        pipe = InteractionMeshPipeline.from_calibration(
            model, motion, str(cal_path), human_height=human_height, cfg=im_cfg,
        )

        def _im_cb(stage: str, cur: int, tot: int) -> None:
            if job is None:
                return
            # precompute is the first ~30%, MPC the remaining ~70%.
            if stage == "precompute":
                frac = 0.3 * (cur / max(1, tot))
                _set_retarget_job_clip_progress(job, frac, f"预处理 {cur}/{tot}")
            else:
                frac = 0.3 + 0.68 * (cur / max(1, tot))
                _set_retarget_job_clip_progress(job, frac, f"MPC 求解 {cur}/{tot}")

        try:
            try:
                return pipe.run(motion, progress_callback=_im_cb)
            except TypeError:
                return pipe.run(motion)
        except ModuleNotFoundError as err:
            if "osqp" in str(err).lower():
                raise ValueError(
                    "interaction-mesh retarget on terrain needs the OSQP solver. "
                    "Install it with `uv pip install osqp` (or re-run "
                    "`uv sync --extra web`)."
                ) from err
            raise

    if backend != "interaction_mesh":
        _require_newton_package()

    # newton
    from hhtools.retarget.calibration import load_calibration
    from hhtools.retarget.newton_basic import NewtonBasicPipeline
    from hhtools.retarget.newton_basic._warp_config import configure as configure_warp_cache
    from hhtools.robot.retarget_profile import (
        build_feet_stabilizer_config,
        build_pipeline_config_for_preset,
        resolve_retarget_scaler_config,
    )

    if job is not None:
        _set_retarget_job_clip_progress(job, 0.03, "正在加载标定与缩放参数…")
    if state is not None:
        _join_robot_prewarm(state, robot_name, job)

    configure_warp_cache()
    calibration = load_calibration(cal_path) if cal_path is not None else None
    scaler_cfg = resolve_retarget_scaler_config(
        preset,
        reference,
        calibration=calibration,
        model=model,
        motion=motion,
        human_height=human_height,
    )
    feet_cfg = build_feet_stabilizer_config(preset, reference, model=model)
    if job is not None:
        # Only advertise kernel compilation when this robot has NOT been
        # prewarmed yet — once Warp's cache is populated (and writable) the
        # init is fast and the old unconditional "compiling kernels" notice was
        # misleading users into thinking every run recompiled.
        try:
            from hhtools.retarget.newton_basic.pipeline import is_newton_ik_prewarmed

            _prewarmed = is_newton_ik_prewarmed(robot_name)
        except Exception:
            _prewarmed = False
        _set_retarget_job_clip_progress(
            job,
            0.06,
            (
                "正在初始化 Newton IK…"
                if _prewarmed
                else "正在初始化 Newton IK（首次会编译 GPU 内核，之后会复用缓存）…"
            ),
        )
    pipeline = NewtonBasicPipeline(
        model,
        scaler_config=scaler_cfg,
        pipeline_config=build_pipeline_config_for_preset(
            preset, reference, ik_iterations=ik_iters,
            foot_clamp_anti_penetration=foot_clamp_anti_penetration,
        ),
        feet_stabilizer_config=feet_cfg,
        human_height=human_height,
        configure_warp=False,
    )

    def _cb(done: int, total: int) -> None:
        if job is None:
            return
        if done <= 0:
            _set_retarget_job_clip_progress(
                job, 0.08, "正在捕获 CUDA 图 / 准备逐帧 IK（首次较慢，请耐心等待）…",
            )
        else:
            _set_retarget_job_clip_progress(
                job,
                min(0.98, 0.1 + 0.88 * (done / max(1, total))),
                f"IK 求解 {done}/{total}",
            )

    try:
        return pipeline.run(motion, progress_callback=_cb)
    except TypeError:
        return pipeline.run(motion)


def run_web(
    *,
    source_root: Path,
    save_dir: Path,
    cache_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8009,
) -> None:
    """Launch the uvicorn server (blocking)."""
    import uvicorn

    app = create_app(source_root=source_root, save_dir=save_dir, cache_dir=cache_dir)
    url = f"http://{host}:{port}"
    static_dir = Path(__file__).parent / "static"
    print(f"\n  hhtools web  →  {url}")
    print(f"  UI build     →  {UI_BUILD_ID}")
    print(f"  static dir   →  {static_dir.resolve()}")
    print(
        "  侧栏应为 3 项（含「机器人 · Retarget」）；舞台左上角有「骨架|身体|机器人」。"
        "\n  git pull 后请在本仓库执行 uv sync 并用 uv run hhtools web 重启（勿用全局旧包）。"
        "\n  若界面异常：确认终端 UI build 与浏览器地址栏端口一致，再 Ctrl+Shift+R。"
        "\n  Retarget 需：uv sync --extra web --extra retarget + NVIDIA newton 包。\n"
    )
    try:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    except Exception:
        pass
    uvicorn.run(app, host=host, port=port, log_level="info")


__all__ = ["create_app", "run_web"]
