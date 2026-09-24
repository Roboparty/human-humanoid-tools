"""Motion loading, registration, and robot-export playback helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# Datasets whose adapters accept ``with_mesh=True`` (SMPL forward -> baked vertices).
_SMPL_MESH_DATASETS: frozenset[str] = frozenset(
    {"amass", "motion_x", "phuma", "gvhmr", "kungfu_athlete"}
)

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


def _load_clip_for_batch(entry_dict: dict[str, Any], entry: Any, cache: Any):
    """Load uploaded/local paths directly; managed-library entries use cache."""
    from hhtools.services.motion_cache import _attach_library_folder_label
    from hhtools.services.motion_library_links import resolve_clip_on_disk
    from hhtools.services.upload_resolve import load_clip_at_path

    if entry_dict.get("origin") not in {"upload", "local"}:
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


def _apply_limit_frames(motion: Any, limit_frames: Any):
    if not limit_frames:
        return motion
    lf = int(limit_frames)
    if motion.num_frames <= lf:
        return motion
    motion.positions = motion.positions[:lf]
    motion.quaternions = motion.quaternions[:lf]
    for obj in motion.objects:
        obj.positions = obj.positions[:lf]
        obj.quaternions = obj.quaternions[:lf]
    return motion


def _load_batch_motion(
    entry_dict: dict[str, Any],
    entry: Any,
    cache: Any,
    *,
    retarget_fps: float | None,
    limit_frames: Any,
):
    motion = _load_clip_for_batch(entry_dict, entry, cache)
    motion = _ground_motion_for_web(motion)
    motion, _ = _motion_for_retarget(motion, retarget_fps)
    return _apply_limit_frames(motion, limit_frames)


def _ground_motion_for_web(motion: Any):
    """Centre the root at the origin (XY) and snap the lowest point to z=0."""
    try:
        from hhtools.core.anatomy import center_motion_root_xy, snap_motion_to_ground
        from hhtools.core.coord import to_up_axis
        from hhtools.retarget.newton_basic.rest_pose import normalize_mocap_bvh_clip

        motion = normalize_mocap_bvh_clip(motion)
        if motion.up_axis != "Z":
            motion = to_up_axis(motion, "Z")
        motion = center_motion_root_xy(motion)
        motion = snap_motion_to_ground(motion, margin=0.0)
    except Exception:  # noqa: BLE001 - never block loading on grounding
        _log.warning("grounding failed; using raw motion", exc_info=True)
    return motion


def _load_motion_for_web(entry: Any, cache: Any, *, progress: Any = None):
    """Load a library clip with SMPL mesh baking when the dataset supports it."""
    from hhtools.io.datasets import registered_datasets
    from hhtools.services.motion_cache import _attach_library_folder_label

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


def _load_motion_file(path: Path, *, progress: Any = None):
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


def _load_via_adapter(path: Path):  # noqa: PLR0911 - one explicit branch per format
    """Best-effort dataset-adapter load for non-io.base extensions."""
    suf = path.suffix.lower()
    try:
        if suf == ".bvh":
            from hhtools.io.mimic_detect import is_omnicontact_capture

            if is_omnicontact_capture(path):
                from hhtools.io.datasets.omnicontact import OmniContactAdapter

                return (
                    OmniContactAdapter(root=path.parent).load_motion(path.name),
                    "omnicontact",
                )
        if suf == ".pkl":
            from hhtools.io.datasets.omomo import OmomoAdapter
            from hhtools.io.datasets.parc_ms import ParcMsAdapter

            parent = path.parent
            try:
                if parent.name == path.stem:
                    return (
                        OmomoAdapter(root=parent.parent).load_motion(f"{parent.name}/{path.name}"),
                        "omomo",
                    )
                return OmomoAdapter(root=parent).load_motion(path.name), "omomo"
            except Exception:
                pass
            try:
                if parent.name == path.stem:
                    return (
                        ParcMsAdapter(root=parent.parent).load_motion(f"{parent.name}/{path.name}"),
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
                motion = MeshmimicHolosomaAdapter(root=parent.parent).load_motion(seq)
                return motion, "meshmimic_holosoma"
        if suf == ".npz":
            from hhtools.io.datasets.amass import AmassAdapter

            return (
                AmassAdapter(root=path.parent).load_motion(path.name, with_mesh=True),
                "amass",
            )
    except Exception:
        return None, None
    return None, None


def _motion_for_retarget(motion: Any, retarget_fps: float | None):
    """Optionally resample the clip before IK/MPC.

    Returns ``(motion_for_solver, effective_fps)``. When the rate is unchanged,
    the same ``Motion`` instance is returned.
    """
    from hhtools.application.export import _parse_optional_fps
    from hhtools.core.resample import resample_motion_with_objects

    src = float(motion.framerate)
    target = _parse_optional_fps(retarget_fps)
    if target is None or abs(target - src) < 1e-6:
        return motion, src
    return resample_motion_with_objects(motion, target), float(target)
