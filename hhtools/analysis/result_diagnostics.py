"""Lightweight result diagnostics for Web H2R and R2R previews.

The diagnostics operate on the same downsampled payloads rendered by Three.js.
They are intentionally solver-independent: no IK, simulation, or model rebuild is
performed after a retarget job completes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from hhtools.retarget.newton_basic.human_aliases import auto_source_to_canonical

_FOOT_CANONICALS = {
    "left": ("left_ankle", "left_foot", "left_toe"),
    "right": ("right_ankle", "right_foot", "right_toe"),
}


def _target_link(spec: object, canonical: str) -> str:
    if isinstance(spec, Mapping):
        return str(spec.get("t_body") or spec.get("link") or canonical)
    return str(spec or canonical)


def _mapping_pairs(ik_map: Mapping[str, object] | None) -> list[tuple[str, str]]:
    return [
        (str(canonical), _target_link(spec, str(canonical)))
        for canonical, spec in (ik_map or {}).items()
    ]


def _canonical_name_to_index(names: Sequence[str]) -> dict[str, int]:
    """Resolve source-rig bone names to the canonical names used by ``ik_map``."""
    source_to_canonical = auto_source_to_canonical(tuple(str(name) for name in names))
    canonical_to_index: dict[str, int] = {}
    for index, source_name in enumerate(names):
        canonical = str(source_to_canonical.get(str(source_name), source_name))
        # Some rigs collapse multiple spine bones onto one canonical joint. The
        # first matching row is deterministic and mirrors the retarget pipeline.
        canonical_to_index.setdefault(canonical, index)
    return canonical_to_index


def _quat_rotate_xyzw(vector: np.ndarray, quaternion: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if q.shape != (4,) or norm < 1e-9:
        return vector
    q = q / norm
    xyz = q[:3]
    return vector + 2.0 * np.cross(xyz, np.cross(xyz, vector) + q[3] * vector)


def _robot_link_position(frame: Mapping[str, Any], link: str) -> np.ndarray | None:
    flat = (frame.get("links") or {}).get(link)
    if not isinstance(flat, Sequence) or len(flat) != 16:
        return None
    matrix = np.asarray(flat, dtype=np.float64).reshape(4, 4)
    local = matrix[:3, 3]
    root = frame.get("root")
    if not isinstance(root, Sequence) or len(root) < 7:
        return local
    translation = np.asarray(root[:3], dtype=np.float64)
    translation[2] += float(frame.get("mesh_z_lift") or 0.0)
    return translation + _quat_rotate_xyzw(local, root[3:7])


def _frame_pairs(
    trajectory: Mapping[str, Any],
    scaled_preview: Mapping[str, Any],
) -> list[tuple[int, Sequence[Sequence[float]], Mapping[str, Any]]]:
    robot_frames = trajectory.get("frames") or []
    scaled_frames = scaled_preview.get("positions") or []
    if not robot_frames or not scaled_frames:
        return []

    robot_indices = trajectory.get("frame_indices") or list(range(len(robot_frames)))
    scaled_indices = scaled_preview.get("frame_indices") or list(range(len(scaled_frames)))
    scaled_by_index = {
        int(frame_index): scaled_frames[index]
        for index, frame_index in enumerate(scaled_indices[: len(scaled_frames)])
    }
    pairs: list[tuple[int, Sequence[Sequence[float]], Mapping[str, Any]]] = []
    for index, robot_frame in enumerate(robot_frames):
        frame_index = int(robot_indices[index]) if index < len(robot_indices) else index
        scaled_frame = scaled_by_index.get(frame_index)
        if scaled_frame is not None:
            pairs.append((frame_index, scaled_frame, robot_frame))
    return pairs


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _contact_flags(
    points: np.ndarray,
    frame_indices: np.ndarray,
    fps: float,
    *,
    height_tolerance_m: float = 0.05,
    speed_tolerance_mps: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate rendered contact from low height and low planar speed.

    This is deliberately a viewer-side quality heuristic, not a physics contact
    query: diagnostics run on downsampled preview payloads after retargeting and
    may not have a simulator, collision shapes, forces, or the original frames.
    """
    floor_z = float(np.percentile(points[:, 2], 5.0))
    speed = np.zeros(points.shape[0], dtype=np.float64)
    if points.shape[0] > 1:
        dt = np.diff(frame_indices) / max(fps, 1e-6)
        dt = np.maximum(dt, 1.0 / max(fps, 1e-6))
        segment_speed = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1) / dt
        speed[1:] = segment_speed
        speed[0] = segment_speed[0]
    contact = (points[:, 2] <= floor_z + height_tolerance_m) & (speed <= speed_tolerance_mps)
    return contact, speed


def _contact_diagnostics(
    frame_pairs: Sequence[tuple[int, Sequence[Sequence[float]], Mapping[str, Any]]],
    names: Sequence[str],
    ik_map: Mapping[str, object] | None,
    feet: Mapping[str, object] | None,
    fps: float,
) -> tuple[dict[str, Any], dict[int, tuple[int, int]]]:
    name_to_index = _canonical_name_to_index(names)
    links = dict(_mapping_pairs(ik_map))
    per_foot: list[dict[str, Any]] = []
    contacts_by_frame: dict[int, list[int]] = {
        frame_index: [0, 0] for frame_index, _, _ in frame_pairs
    }

    for side in ("left", "right"):
        canonical = next(
            (name for name in _FOOT_CANONICALS[side] if name in name_to_index),
            None,
        )
        if canonical is None:
            continue
        configured_link = (feet or {}).get(f"{side}_contact_link")
        target_link = str(configured_link or links.get(canonical) or "")
        if not target_link:
            continue

        frame_ids: list[int] = []
        source_points: list[np.ndarray] = []
        target_points: list[np.ndarray] = []
        for frame_index, scaled_frame, robot_frame in frame_pairs:
            source_index = name_to_index[canonical]
            if source_index >= len(scaled_frame):
                continue
            target_point = _robot_link_position(robot_frame, target_link)
            if target_point is None:
                continue
            source_point = np.asarray(scaled_frame[source_index], dtype=np.float64)
            if source_point.shape != (3,) or not np.all(np.isfinite(source_point)):
                continue
            frame_ids.append(frame_index)
            source_points.append(source_point)
            target_points.append(target_point)

        if len(frame_ids) < 2:
            continue
        indices = np.asarray(frame_ids, dtype=np.float64)
        source_contact, _ = _contact_flags(np.vstack(source_points), indices, fps)
        target_contact, target_speed = _contact_flags(np.vstack(target_points), indices, fps)
        agreement = float(np.mean(source_contact == target_contact))
        source_count = int(np.count_nonzero(source_contact))
        true_positive = int(np.count_nonzero(source_contact & target_contact))
        recall = float(true_positive / source_count) if source_count else 1.0
        slide = target_speed[target_contact]

        for index, frame_id in enumerate(frame_ids):
            counts = contacts_by_frame[frame_id]
            counts[0] += int(source_contact[index])
            counts[1] += int(target_contact[index])

        per_foot.append(
            {
                "side": side,
                "canonical": canonical,
                "target_link": target_link,
                "agreement_ratio": round(agreement, 4),
                "recall_ratio": round(recall, 4),
                "source_contact_ratio": round(float(np.mean(source_contact)), 4),
                "target_contact_ratio": round(float(np.mean(target_contact)), 4),
                "target_slide_mean_mps": round(float(np.mean(slide)) if slide.size else 0.0, 4),
                "target_slide_p95_mps": round(
                    float(np.percentile(slide, 95.0)) if slide.size else 0.0,
                    4,
                ),
            }
        )

    if not per_foot:
        return (
            {
                "available": False,
                "reason": "foot mappings are unavailable in the result payload",
                "feet": [],
            },
            {frame: (values[0], values[1]) for frame, values in contacts_by_frame.items()},
        )

    return (
        {
            "available": True,
            "agreement_ratio": round(
                float(np.mean([foot["agreement_ratio"] for foot in per_foot])),
                4,
            ),
            "recall_ratio": round(
                float(np.mean([foot["recall_ratio"] for foot in per_foot])),
                4,
            ),
            "target_slide_mean_mps": round(
                float(np.mean([foot["target_slide_mean_mps"] for foot in per_foot])),
                4,
            ),
            "target_slide_p95_mps": round(
                float(np.max([foot["target_slide_p95_mps"] for foot in per_foot])),
                4,
            ),
            "feet": per_foot,
        },
        {frame: (values[0], values[1]) for frame, values in contacts_by_frame.items()},
    )


def build_result_diagnostics(
    trajectory: Mapping[str, Any],
    scaled_preview: Mapping[str, Any] | None,
    *,
    ik_map: Mapping[str, object] | None,
    feet: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Compare rendered IK targets with rendered robot link trajectories."""
    if not scaled_preview:
        return {
            "schema_version": 1,
            "available": False,
            "reason": "scaled target preview is unavailable",
        }

    names = [str(name) for name in (scaled_preview.get("bone_names") or [])]
    name_to_index = _canonical_name_to_index(names)
    frame_pairs = _frame_pairs(trajectory, scaled_preview)
    mappings = [
        (canonical, link)
        for canonical, link in _mapping_pairs(ik_map)
        if canonical in name_to_index
    ]
    if not frame_pairs or not mappings:
        return {
            "schema_version": 1,
            "available": False,
            "reason": "no mapped target frames are available for comparison",
        }

    per_effector_errors: dict[tuple[str, str], list[float]] = {mapping: [] for mapping in mappings}
    frame_errors: list[tuple[int, list[float]]] = []
    for frame_index, scaled_frame, robot_frame in frame_pairs:
        errors: list[float] = []
        for canonical, link in mappings:
            source_index = name_to_index[canonical]
            if source_index >= len(scaled_frame):
                continue
            target = _robot_link_position(robot_frame, link)
            if target is None:
                continue
            source = np.asarray(scaled_frame[source_index], dtype=np.float64)
            if source.shape != (3,) or not np.all(np.isfinite(source)):
                continue
            error = float(np.linalg.norm(target - source))
            errors.append(error)
            per_effector_errors[(canonical, link)].append(error)
        if errors:
            frame_errors.append((frame_index, errors))

    all_errors = [error for _, errors in frame_errors for error in errors]
    if not all_errors:
        return {
            "schema_version": 1,
            "available": False,
            "reason": "mapped robot links are missing from the trajectory",
        }

    fps = float(
        trajectory.get("framerate")
        or trajectory.get("sample_rate")
        or scaled_preview.get("framerate")
        or 30.0
    )
    contact, contacts_by_frame = _contact_diagnostics(
        frame_pairs,
        names,
        ik_map,
        feet,
        fps,
    )
    effectors = []
    for (canonical, link), errors in per_effector_errors.items():
        if not errors:
            continue
        effectors.append(
            {
                "canonical": canonical,
                "target_link": link,
                "sample_count": len(errors),
                "mean_error_m": round(float(np.mean(errors)), 5),
                "p95_error_m": round(_percentile(errors, 95.0), 5),
                "max_error_m": round(float(np.max(errors)), 5),
            }
        )
    effectors.sort(key=lambda item: item["p95_error_m"], reverse=True)

    series = []
    for frame_index, errors in frame_errors:
        source_contacts, target_contacts = contacts_by_frame.get(frame_index, (0, 0))
        series.append(
            {
                "frame": frame_index,
                "time_s": round(frame_index / max(fps, 1e-6), 4),
                "mean_error_m": round(float(np.mean(errors)), 5),
                "max_error_m": round(float(np.max(errors)), 5),
                "source_contacts": source_contacts,
                "target_contacts": target_contacts,
            }
        )

    return {
        "schema_version": 1,
        "available": True,
        "frame_count": len(frame_errors),
        "mapped_effectors": len(effectors),
        "requested_effectors": len(mappings),
        "tracking": {
            "mean_error_m": round(float(np.mean(all_errors)), 5),
            "p95_error_m": round(_percentile(all_errors, 95.0), 5),
            "max_error_m": round(float(np.max(all_errors)), 5),
            "effectors": effectors,
            "series": series,
        },
        "contact": contact,
    }


__all__ = ["build_result_diagnostics"]
