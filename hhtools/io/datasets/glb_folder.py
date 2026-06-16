"""Generic adapter for flat directories of authored rigs (GLB / glTF).

These rigs ship as single-clip files authored in Blender / Maya / Mixamo.  Unlike
AMASS / OMOMO / Motion-X which need dataset-specific decoding, a ``.glb`` holds
its own skeleton, animation tracks, and optional skinned mesh already, so the
"adapter" here is just a thin indexer over a folder of such files.

Why we keep these as *adapters* rather than directly listing via the library's
format dispatch:

* The library UI is organised around dataset folders.  Routing ``assets/motions/mimic/GLB/``
  through a registered adapter keeps the folder visible in the folder-picker dropdown
  alongside AMASS / Motion-X, rather than being rendered as an ad-hoc "misc" bucket.
* It gives us one place to decide "the viewer should load this with_mesh=True" —
  the :meth:`load_motion` override below sets that flag by default so any viewer
  session picking up a GLB clip gets skinned-mesh rendering for free.

The adapter is intentionally tiny: it delegates actual parsing to
:func:`hhtools.io.glb.load_glb`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hhtools.core.motion import Motion
from hhtools.io.datasets.base import DatasetAdapter, register_dataset


class _FlatAuthoredClipAdapter(DatasetAdapter):
    """Shared logic for flat folders of authored skeletal clips."""

    file_patterns: tuple[str, ...] = ()  # filled by subclasses
    _with_mesh_default: bool = True

    def list_sequences(self) -> Iterator[str]:
        if not self.root.exists():
            return
        for pat in self.file_patterns:
            for p in sorted(self.root.rglob(pat)):
                if p.is_file():
                    yield str(p.relative_to(self.root))

    def _resolve(self, sequence_id: str) -> Path:
        p = (self.root / sequence_id).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"{self.name} clip not found: {p}")
        return p

    def load_motion(self, sequence_id: str, **kwargs: Any) -> Motion:
        path = self._resolve(sequence_id)
        m = self._load(path, **kwargs)
        m.meta["dataset"] = self.name
        m.meta["sequence_id"] = sequence_id
        return m

    # Subclasses override this to delegate to the right loader.  We keep it on the
    # instance (not a module-level reference) so tests can monkey-patch it when
    # exercising the adapter-level glue without pulling in pygltflib.
    def _load(self, path: Path, **kwargs: Any) -> Motion:  # pragma: no cover - stub
        raise NotImplementedError


@register_dataset
class GlbFolderAdapter(_FlatAuthoredClipAdapter):
    """Browse a flat directory of ``.glb`` / ``.gltf`` rigs."""

    name = "glb"
    display_name = "GLB/glTF authored rigs"
    requires = "formats"
    file_patterns = ("*.glb", "*.gltf")

    def _load(self, path: Path, **kwargs: Any) -> Motion:
        from hhtools.io.glb import load_glb  # local import: pygltflib is optional

        kwargs.setdefault("with_mesh", self._with_mesh_default)
        return load_glb(path, **kwargs)


__all__ = ["GlbFolderAdapter"]
