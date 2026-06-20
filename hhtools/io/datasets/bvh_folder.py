"""Generic adapter for BVH-based datasets (LAFAN, SOMA, Xsens mocap, etc.)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hhtools.core.motion import Motion
from hhtools.io.bvh import load_bvh
from hhtools.io.datasets.base import DatasetAdapter, register_dataset


class _BvhFolderAdapter(DatasetAdapter):
    requires = "bvh"
    file_patterns = ("*.bvh",)

    def list_sequences(self) -> Iterator[str]:
        if not self.root.exists():
            return
        for p in sorted(self.root.rglob("*.bvh")):
            if p.is_file():
                yield str(p.relative_to(self.root))

    def _resolve(self, sequence_id: str) -> Path:
        p = (self.root / sequence_id).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"BVH sequence not found: {p}")
        return p

    def load_motion(self, sequence_id: str, **kwargs: Any) -> Motion:
        path = self._resolve(sequence_id)
        m = load_bvh(path, **kwargs)
        m.meta["dataset"] = self.name
        m.meta["sequence_id"] = sequence_id
        return m


@register_dataset
class LafanAdapter(_BvhFolderAdapter):
    name = "lafan"
    display_name = "LAFAN1"


@register_dataset
class SomaAdapter(_BvhFolderAdapter):
    name = "soma"
    display_name = "SOMA Retargeter test clips"


@register_dataset
class XsensMocapAdapter(_BvhFolderAdapter):
    name = "xsens_mocap"
    display_name = "Xsens MVN / biomechanics BVH"


__all__ = ["LafanAdapter", "SomaAdapter", "XsensMocapAdapter"]
