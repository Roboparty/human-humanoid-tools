# SPDX-License-Identifier: Apache-2.0
"""``AnalyzableClip`` — the unified per-clip analysis record + orchestration.

One clip == one library primary file (a ``.bvh`` / ``.npz`` / ``.pkl`` / ``.npy``
human source, or a retargeted robot ``.csv``).  Sidecar meshes / terrain are
folded into the loaded :class:`~hhtools.core.motion.Motion` by the dataset
adapter and surface here as L1 / L2 metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hhtools.analysis.canonical import project_to_canonical
from hhtools.analysis.config import load_config
from hhtools.analysis import metrics as _metrics
from hhtools.analysis import scene_metrics as _scene
from hhtools.analysis import tags as _tags


@dataclass
class AnalyzableClip:
    """All analysis outputs for a single clip."""

    clip_id: str
    source_kind: str  # "human" | "robot"
    source_path: str
    dataset: str
    folder_label: str
    metrics: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    scatter: tuple[float, float] | None = None
    cluster_id: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "source_kind": self.source_kind,
            "source_path": self.source_path,
            "dataset": self.dataset,
            "folder_label": self.folder_label,
            "metrics": self.metrics,
            "tags": list(self.tags),
            "embedding": self.embedding,
            "scatter": list(self.scatter) if self.scatter is not None else None,
            "cluster_id": self.cluster_id,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnalyzableClip":
        scatter = d.get("scatter")
        return cls(
            clip_id=d["clip_id"],
            source_kind=d.get("source_kind", "human"),
            source_path=d.get("source_path", ""),
            dataset=d.get("dataset", ""),
            folder_label=d.get("folder_label", ""),
            metrics=d.get("metrics", {}),
            tags=list(d.get("tags", [])),
            embedding=d.get("embedding"),
            scatter=tuple(scatter) if scatter else None,
            cluster_id=d.get("cluster_id"),
            error=d.get("error"),
        )


def _load_source(source_path: Path, dataset: str = ""):
    """Load a clip path into a ``Motion`` or ``RobotCSV``.

    Mirrors the web library loading path: robot trajectory CSVs go through the
    robot reader; every other clip is dispatched to its registered dataset
    adapter (``amass`` / ``omomo`` / ``parc_ms`` / ``holosoma`` / ``lafan`` /
    ``glb`` / ...), exactly as :meth:`hhtools.viewer.cache.EphemeralCache._convert`
    does, so heterogeneous formats (``.pkl`` / ``.npy`` / ``.bvh`` / ...) all load.
    Falls back to the plain loader registry when no dataset name is supplied.
    """
    from hhtools.web.r2r_upload_resolve import _is_robot_export_trajectory

    suffix = source_path.suffix.lower()
    is_robot = dataset == "robot" or _is_robot_export_trajectory(source_path)
    if is_robot and suffix in (".csv", ".pkl", ".pickle", ".npz"):
        from hhtools.io.robot_csv import RobotCSV, load_robot_csv
        from hhtools.retarget.robot_to_robot import load_source_trajectory

        if suffix == ".csv":
            return load_robot_csv(source_path), "robot"
        traj = load_source_trajectory(source_path, source_model=None)
        n = int(traj.joint_q.shape[0])
        fps = float(traj.framerate)
        if n > 1:
            times = np.arange(n, dtype=np.float64) / max(fps, 1e-6)
        else:
            times = np.zeros((0,), dtype=np.float64)
        return (
            RobotCSV(
                times=times,
                joint_q=np.asarray(traj.joint_q, dtype=np.float32),
                dof_names=tuple(traj.dof_names),
                sample_rate=fps,
                meta=dict(traj.meta),
            ),
            "robot",
        )

    if dataset:
        from hhtools.io.datasets import registered_datasets

        adapter_cls = registered_datasets().get(dataset)
        if adapter_cls is not None:
            adapter = adapter_cls(source_path.parent)
            return adapter.load_motion(source_path.name), "human"

    from hhtools.io.base import load_motion

    return load_motion(source_path), "human"


def analyze_clip(
    obj_or_path: Any,
    *,
    clip_id: str,
    source_path: str,
    dataset: str = "",
    folder_label: str = "",
    cfg: dict[str, Any] | None = None,
) -> AnalyzableClip:
    """Compute metrics + per-clip tags for one clip.

    ``obj_or_path`` may be a pre-loaded ``Motion`` / ``RobotCSV`` or a path/str to
    load.  Embedding, dataset-relative tags, scatter and cluster id are filled in
    later at the collection level (see :mod:`hhtools.analysis.collection`).
    """
    cfg = cfg or load_config()
    clip = AnalyzableClip(
        clip_id=clip_id,
        source_kind="human",
        source_path=str(source_path),
        dataset=dataset,
        folder_label=folder_label,
    )
    try:
        if isinstance(obj_or_path, (str, Path)):
            loaded, kind = _load_source(Path(obj_or_path), dataset)
        else:
            from hhtools.io.robot_csv import RobotCSV

            loaded = obj_or_path
            kind = "robot" if isinstance(obj_or_path, RobotCSV) else "human"
        clip.source_kind = kind

        feat = project_to_canonical(loaded)
        m: dict[str, Any] = {}
        m.update(_metrics.compute_dynamics(feat, cfg))
        quality = _metrics.compute_quality(feat, cfg)

        # L1 objects (human Motion only carries objects/terrain).
        if kind == "human":
            l1 = _scene.compute_l1_objects(loaded, feat, cfg)
            if l1 is not None:
                m.update(l1)
            l2 = _scene.compute_l2_terrain(loaded, feat, cfg)
            if l2 is not None:
                m["has_terrain"] = 1.0
                # Splice terrain-corrected severities back into the quality score.
                for key in ("sev_floating", "sev_penetration", "sev_foot_slide"):
                    if key in l2:
                        quality[key] = l2[key]
                quality = _metrics.finalize_quality(quality, cfg)
                # Keep terrain descriptors (drop the sev_* duplicates we merged).
                for key, val in l2.items():
                    if not key.startswith("sev_"):
                        m[key] = val

        m.update(quality)
        if kind == "robot":
            from hhtools.io.robot_csv import RobotCSV

            if isinstance(loaded, RobotCSV):
                preset = str(loaded.meta.get("robot") or "").strip()
                if preset:
                    m["robot_preset"] = preset
        clip.metrics = m
        clip.tags = _tags.assign_clip_tags(m, cfg)
    except Exception as exc:  # noqa: BLE001 - record per-clip failure, keep going
        clip.error = f"{type(exc).__name__}: {exc}"
    return clip


# ----------------------------------------------------------------- cache IO

def write_manifest(path: Path, clips: list[AnalyzableClip], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "clips": [c.to_dict() for c in clips]}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


__all__ = ["AnalyzableClip", "analyze_clip", "read_manifest", "write_manifest"]
