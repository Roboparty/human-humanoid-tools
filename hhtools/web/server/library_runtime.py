"""Motion Library entry and local-batch helpers used by Web runtimes."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

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
    "omnicontact": "lafan_bvh",
    "meshmimic_holosoma": "smplx",
    "glb": "glb",
    "unified_npz": "smpl",
    "parc_ms": "smpl",
}

_ROBOT_RESULT_JOB_KINDS = frozenset({"retarget", "r2r_retarget"})


def _source_robot_trajectory_entries(source_root: Path) -> list[dict[str, Any]]:
    """Return structurally validated robot trajectories below ``source_root``.

    The legacy source-library scanner only knows human dataset adapters and
    therefore cannot surface CSV robot exports (including MotionDecode).  Keep
    those files on the same Library boundary, but mark them explicitly so H2R
    and R2R clients can retain their separate input contracts.
    """
    from hhtools.io.robot_trajectory_detect import trajectory_robot_name
    from hhtools.services.r2r_upload_resolve import enumerate_r2r_clips

    root = Path(source_root).expanduser().resolve()
    entries: list[dict[str, Any]] = []
    for reference in enumerate_r2r_clips(root, profile="auto"):
        path = Path(reference.path).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = Path(path.name)
        parent = relative.parent
        folder_label = parent.as_posix() if parent != Path(".") else "Robot trajectories"
        stem = path.parent.name if path.parent.name == path.stem else path.stem
        entry: dict[str, Any] = {
            "dataset": "robot",
            "folder_label": folder_label,
            "sequence_id": relative.as_posix(),
            "source_path": str(path),
            "stem": stem,
            "label": f"{folder_label} · {stem}",
            "origin": "assets",
            "upload_profile": reference.profile,
            "motion_category": (
                "terrain"
                if reference.profile == "meshmimic"
                else "object"
                if reference.profile == "intermimic"
                else "motion"
            ),
            "asset_kind": "robot_trajectory",
        }
        source_robot = trajectory_robot_name(path)
        if source_robot:
            entry["source_robot"] = source_robot
        entries.append(entry)
    return entries


def _retained_robot_trajectory_entries(job_history: Any) -> list[dict[str, Any]]:
    """Expose retained H2R/R2R files as reusable robot trajectories.

    Workflow artifacts already live in the bounded, persistent job-history
    store.  Referencing them avoids a second copy while allowing a completed
    H2R result to become the source of a later R2R workflow.  ZIP bundles are
    intentionally omitted because the single-source loader requires one
    concrete trajectory file.
    """
    from hhtools.io.robot_trajectory_detect import is_robot_export_trajectory

    entries: list[dict[str, Any]] = []
    for record in job_history.list_records():
        if record.get("status") != "done" or record.get("kind") not in _ROBOT_RESULT_JOB_KINDS:
            continue
        path = job_history.artifact_path(record)
        if path is None or not is_robot_export_trajectory(path):
            continue
        request = record.get("request")
        request = request if isinstance(request, dict) else {}
        source_robot = str(
            (request.get("robot") if record.get("kind") == "retarget" else request.get("target"))
            or ""
        ).strip()
        folder_label = "Generated trajectories"
        entry: dict[str, Any] = {
            "dataset": "robot",
            "folder_label": folder_label,
            "sequence_id": path.name,
            "source_path": str(path),
            "stem": path.stem,
            "label": f"{folder_label} · {path.stem}",
            "origin": "job",
            "job_id": str(record.get("id") or ""),
            "motion_category": "motion",
            "asset_kind": "robot_trajectory",
        }
        if source_robot:
            entry["source_robot"] = source_robot
        entries.append(entry)
    return entries


def _adopt_motion_library_root(
    target: Path,
    *,
    current_root: Path | None = None,
    trusted_roots: tuple[Path, ...] = (),
) -> Path:
    """Create or explicitly adopt one dedicated managed library container."""

    from hhtools.services.motion_library_settings import (
        motion_library_marker_path,
        motion_library_marker_payload,
        validate_motion_library_marker,
    )

    root = Path(target).expanduser().resolve(strict=False)
    if root.parent == root or root == Path.home().resolve(strict=False):
        raise ValueError("请选择专用的资源库目录，不能使用文件系统根目录或用户主目录")
    if root.exists() and not root.is_dir():
        raise ValueError("资源库路径必须是目录")

    marker = motion_library_marker_path(root)
    known_roots = trusted_roots + ((current_root,) if current_root is not None else ())
    already_owned = any(
        root == Path(candidate).expanduser().resolve(strict=False) for candidate in known_roots
    )
    marker_is_valid = validate_motion_library_marker(root)
    if root.is_dir() and not marker_is_valid and not already_owned:
        try:
            has_existing_content = next(root.iterdir(), None) is not None
        except OSError as err:
            raise ValueError(f"无法读取资源库目录：{err}") from err
        if has_existing_content:
            raise ValueError(
                "所选目录不是空目录。请选择一个空的专用目录；已有数据集请使用“链接目录”接入。"
            )

    try:
        root.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_name(f".{marker.name}.{time.time_ns()}.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    motion_library_marker_payload(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(marker)
        finally:
            temporary.unlink(missing_ok=True)
    except OSError as err:
        raise ValueError(f"无法写入资源库目录：{err}") from err
    return root


def _entry_reference(entry: dict[str, Any], fallback: str) -> str:
    """Map a basket row to the calibration reference it needs."""
    explicit = (entry.get("reference") or "").strip()
    if explicit:
        return explicit
    dataset = (entry.get("dataset") or "").strip()
    if dataset in _DATASET_TO_REFERENCE:
        return _DATASET_TO_REFERENCE[dataset]
    return fallback


def _enrich_basket_entry(entry: dict[str, Any], fallback: str = "smpl") -> dict[str, Any]:
    """Attach stable calibration and Motion Library UX metadata."""
    from hhtools.web.library.motion_library_categories import infer_motion_category

    out = dict(entry)
    if not (out.get("reference") or "").strip():
        out["reference"] = _entry_reference(out, fallback)
    # Keep category semantics on the API boundary. The renderer must not infer
    # object/terrain workflows from volatile adapter or folder display names.
    out["motion_category"] = infer_motion_category(out)
    explicit_kind = str(out.get("asset_kind") or "").strip().casefold()
    if explicit_kind in {"human_motion", "robot_trajectory"}:
        out["asset_kind"] = explicit_kind
    else:
        dataset = str(out.get("dataset") or "").strip().casefold()
        out["asset_kind"] = "robot_trajectory" if dataset in {"robot", "r2r"} else "human_motion"
    return out


def _matching_materialized_clip(
    library_root: Path,
    *,
    snapshot_root: Path,
    snapshot_picked: Path,
    profile: str,
) -> Path:
    """Match a loaded snapshot clip to its newly materialized library path."""
    from hhtools.services.upload_resolve import enumerate_upload_clips

    library_root = Path(library_root).resolve()
    snapshot_root = Path(snapshot_root).resolve()
    snapshot_picked = Path(snapshot_picked).resolve()
    candidates = enumerate_upload_clips(library_root, profile)
    if not candidates:
        raise ValueError("动作已解析，但发布后的 Motion Library 中没有可识别文件。")
    try:
        snapshot_parts = snapshot_picked.relative_to(snapshot_root).parts
    except ValueError:
        snapshot_parts = (snapshot_picked.name,)

    def score(candidate: Any) -> tuple[int, bool]:
        # Keep the candidate lexical while deriving its library-relative
        # suffix. ``resolve()`` here would follow file/directory symlinks out
        # of the library and collapse distinct paths such as ``a/clip.bvh``
        # and ``b/clip.bvh`` to their external targets.
        candidate_path = Path(candidate.path)
        try:
            candidate_parts = candidate_path.relative_to(library_root).parts
        except ValueError:
            try:
                candidate_parts = candidate_path.absolute().relative_to(library_root).parts
            except ValueError:
                candidate_parts = (candidate_path.name,)
        matching_suffix = 0
        for left, right in zip(
            reversed(snapshot_parts),
            reversed(candidate_parts),
            strict=False,
        ):
            if left != right:
                break
            matching_suffix += 1
        return matching_suffix, candidate_path.name == snapshot_picked.name

    return Path(max(candidates, key=score).path).resolve()


def _library_entry_from_link(
    folder_label: str,
    lib_dir: Path,
    picked: Path,
    dataset: str | None,
) -> dict[str, Any]:
    """Build a library-shaped entry for a clip under the managed library root."""
    from hhtools.services.motion_library_links import scan_motions_library

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
    return _enrich_basket_entry(
        {
            "dataset": dataset or "unknown",
            "folder_label": folder_label,
            "sequence_id": sequence_id,
            "source_path": sp,
            "stem": stem,
            "label": f"{folder_label} · {stem}",
            "origin": "link",
        }
    )


def _library_entry_from_upload(
    drop_dir: Path,
    picked: Path,
    dataset: str | None,
    profile: str,
    *,
    upload_profile: str | None = None,
    clip_kind: str = "",
    origin: str = "upload",
) -> dict[str, Any]:
    """Build a batch entry for a direct-path clip, uploaded or server-local."""
    from hhtools.services.upload_resolve import export_subdir_for_clip

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
        if picked.stem.lower() == "motion_actor":
            stem = picked.parent.name or stem
    except ValueError:
        sequence_id = picked.name
        stem = picked.parent.name if picked.stem.lower() == "motion_actor" else picked.stem
    return _enrich_basket_entry(
        {
            "dataset": dataset or "unknown",
            "folder_label": folder_label,
            "sequence_id": sequence_id,
            "source_path": str(picked),
            "stem": stem,
            "origin": origin,
            "export_subdir": export_subdir_for_clip(drop_dir, picked),
            "upload_profile": prof,
            "clip_kind": clip_kind,
            "upload_drop": str(drop_dir),
        }
    )


def _normalise_batch_profile(profile: object) -> str:
    return str(profile or "auto").strip().lower() or "auto"


def _resolve_batch_source(source: object) -> Path:
    raw = str(source or "").strip()
    if not raw:
        raise ValueError("请填写本机目录路径")
    root = Path(raw).expanduser()
    if not root.is_dir():
        raise ValueError(f"目录不存在：{root}")
    return root.resolve()


def _entries_for_batch_source(
    source: object,
    profile: object = "auto",
) -> tuple[Path, str, list[dict[str, Any]]]:
    """Enumerate a local batch directory while leaving every clip in place."""
    from hhtools.services.upload_resolve import (
        enumerate_upload_clips,
        upload_validation_error,
    )

    root = _resolve_batch_source(source)
    normalized_profile = _normalise_batch_profile(profile)
    clips = enumerate_upload_clips(root, normalized_profile)
    if not clips:
        raise ValueError(upload_validation_error(normalized_profile))
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
    return root, normalized_profile, entries
