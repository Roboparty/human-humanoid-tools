"""Health, format, and process-wide settings routes."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

from hhtools.services.job_scheduler import JobSchedulerClosedError
from hhtools.services.job_settings import JobAdmissionSettings, updated_job_admission_settings
from hhtools.services.motion_library_links import motions_library_root
from hhtools.services.motion_library_settings import (
    effective_motion_library_root,
    updated_motion_library_settings,
)
from hhtools.utils.paths import (
    HHTOOLS_MOTION_LIBRARY_ROOT_ENV,
    user_motion_library_root,
)
from hhtools.web.server.library_runtime import _adopt_motion_library_root

_log = logging.getLogger(__name__)


def register_system_routes(
    app,
    *,
    state,
    static_dir: Path,
    ui_build_id: str,
    scheduler,
    batch_limit_policy,
    jobs,
    job_settings_store,
    job_settings_update_lock,
    motion_library_settings_store,
    motion_library_publish_lock,
) -> None:
    UI_BUILD_ID = ui_build_id
    _scheduler_payload = jobs.scheduler_payload
    _job_settings_editable = jobs.settings_editable

    def _job_admission_payload(*, editable: bool | None = None) -> dict:
        payload = _scheduler_payload(editable=editable)
        batch_limits = batch_limit_policy.snapshot()
        payload.update(
            {
                "max_batch_items": batch_limits.max_batch_items,
                "max_batch_total_frames": batch_limits.max_batch_total_frames,
            }
        )
        return payload

    @app.get("/api/health")
    def health() -> dict:
        index = static_dir / "index.html"
        index_snip = index.read_text(encoding="utf-8")[:8000] if index.is_file() else ""
        vue_renderer = 'id="app-root"' in index_snip
        return {
            "ok": True,
            "ui_build": UI_BUILD_ID,
            "static_dir": str(static_dir.resolve()),
            "ui_features": {
                "merged_robot_panel": vue_renderer or 'data-panel="retarget"' not in index_snip,
                "view_hud": vue_renderer or "view-hud" in index_snip,
                "scaled_skeleton_toggle": vue_renderer or "tg-scaled" in index_snip,
                "recalib_button": vue_renderer or "recalib-btn" in index_snip,
            },
            "source_root": str(state.source_root),
            "save_dir": str(state.save_dir),
            "motions_library_root": str(motions_library_root()),
            "job_scheduler": _job_admission_payload(),
        }

    @app.get("/api/settings/job-admission")
    def get_job_admission_settings(request: Request) -> dict[str, int | bool | str]:
        """Return live scheduler settings, counters, and edit capability."""

        return _job_admission_payload(editable=_job_settings_editable(request))

    @app.patch("/api/settings/job-admission")
    def patch_job_admission_settings(
        payload: dict[str, Any],
        request: Request,
    ) -> dict[str, int | bool | str]:
        """Persist and hot-apply job limits without stopping active work."""

        if not _job_settings_editable(request):
            # Browser mode has no administrator authentication yet.  Keep this
            # persistent, resource-affecting mutation local; SSH loopback tunnels
            # and the authenticated Electron sidecar still satisfy this boundary.
            # Requiring a literal loopback Host also blocks DNS-rebinding pages.
            raise HTTPException(
                status_code=403,
                detail="job admission settings can only be changed from a loopback client",
            )

        # Serialize the complete read/validate/write/apply transaction so two
        # settings tabs cannot leave the JSON file and live scheduler disagreeing.
        with job_settings_update_lock:
            snapshot = scheduler.snapshot()
            batch_limits = batch_limit_policy.snapshot()
            current = JobAdmissionSettings(
                max_running_jobs=snapshot.max_running_jobs,
                max_queued_jobs=snapshot.max_queued_jobs,
                max_batch_items=batch_limits.max_batch_items,
                max_batch_total_frames=batch_limits.max_batch_total_frames,
            )
            try:
                updated = updated_job_admission_settings(current, payload)
            except ValueError as err:
                raise HTTPException(status_code=422, detail=str(err)) from err

            # Persist first: if the filesystem rejects the write, the live runtime
            # remains unchanged and the UI can truthfully report that Save failed.
            if job_settings_store is not None:
                try:
                    job_settings_store.save(updated)
                except OSError as err:
                    _log.exception("failed to persist Web job admission settings")
                    raise HTTPException(
                        status_code=500,
                        detail="failed to persist job admission settings",
                    ) from err
            try:
                scheduler.reconfigure(
                    max_running_jobs=updated.max_running_jobs,
                    max_queued_jobs=updated.max_queued_jobs,
                )
                batch_limit_policy.reconfigure(
                    max_batch_items=updated.max_batch_items,
                    max_batch_total_frames=updated.max_batch_total_frames,
                )
            except JobSchedulerClosedError as err:
                raise HTTPException(status_code=503, detail=str(err)) from err
            return _job_admission_payload(editable=True)

    def _motion_library_settings_payload(
        request: Request,
    ) -> dict[str, str | bool | None]:
        settings = motion_library_settings_store.load()
        environment_override = bool(os.environ.get(HHTOOLS_MOTION_LIBRARY_ROOT_ENV))
        local_request = _job_settings_editable(request)
        editable = local_request and not environment_override
        readonly_reason: str | None = None
        if not editable:
            readonly_reason = "environment_override" if environment_override else "remote"
        return {
            "root": str(effective_motion_library_root(settings)),
            "default_root": str(user_motion_library_root().expanduser().resolve(strict=False)),
            # An environment override is an administrator-owned launch setting;
            # writing a lower-priority JSON value would misleadingly appear to
            # succeed while leaving the effective root unchanged.
            "editable": editable,
            "readonly_reason": readonly_reason,
        }

    @app.get("/api/settings/motion-library")
    def get_motion_library_settings(request: Request) -> dict[str, str | bool | None]:
        """Return the effective server-side Motion Library root."""

        return _motion_library_settings_payload(request)

    @app.patch("/api/settings/motion-library")
    def patch_motion_library_settings(
        payload: dict[str, Any],
        request: Request,
    ) -> dict[str, str | bool | None]:
        """Persist and hot-apply a dedicated managed library directory."""

        if not _job_settings_editable(request):
            raise HTTPException(
                status_code=403,
                detail="motion library settings can only be changed from a loopback client",
            )
        if os.environ.get(HHTOOLS_MOTION_LIBRARY_ROOT_ENV):
            raise HTTPException(
                status_code=409,
                detail=(
                    "HHTOOLS_MOTION_LIBRARY_ROOT overrides the saved directory; "
                    "change or remove that environment setting first"
                ),
            )

        # Switching the resolver while another thread publishes or scans a
        # library would split one operation across two roots. The same lock used
        # for materialization makes validation, persistence, and the next scan
        # observe one complete root selection.
        with motion_library_publish_lock:
            current = motion_library_settings_store.load()
            try:
                updated = updated_motion_library_settings(current, payload)
                selected_root = effective_motion_library_root(updated)
                current_root = motions_library_root()
                default_root = user_motion_library_root().expanduser().resolve(strict=False)
                if selected_root != current_root:
                    # Mark the root we are leaving before changing the resolver.
                    # This preserves a safe return path even if platform-default
                    # discovery changes while the custom root is active.
                    _adopt_motion_library_root(
                        current_root,
                        current_root=current_root,
                    )
                _adopt_motion_library_root(
                    selected_root,
                    current_root=current_root,
                    # The default/legacy location is owned by hhtools even when
                    # it predates ownership markers. Trust it once so a populated
                    # library can safely round-trip default -> custom -> default;
                    # adoption writes the canonical marker before saving.
                    trusted_roots=(default_root,),
                )
                motion_library_settings_store.save(updated)
            except ValueError as err:
                raise HTTPException(status_code=422, detail=str(err)) from err
            except OSError as err:
                _log.exception("failed to persist Motion Library settings")
                raise HTTPException(
                    status_code=500,
                    detail="failed to persist Motion Library settings",
                ) from err
        return _motion_library_settings_payload(request)

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
                {
                    "ext": ".bvh",
                    "label": "OmniContact (HOI mocap)",
                    "needs": "object CSV + optional assets/",
                },
                {"ext": ".pt", "label": "GVHMR", "needs": "smpl-weights"},
            ],
            "registered_loaders": exts,
        }
