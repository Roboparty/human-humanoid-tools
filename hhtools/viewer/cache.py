"""Per-session ephemeral NPZ cache with opt-in persistence.

Philosophy: conversion from raw dataset files (``.npy`` / ``.pkl`` / ``.bvh`` / ...)
to the unified NPZ schema is expensive. The viewer performs conversion **lazily**
on first play, writes the NPZ into a session-owned cache directory, and wipes the
cache when the session ends (normal quit or ``Ctrl+C``). This holds regardless of
whether the user saved anything: save and cache are independent side effects.

Users who want to keep a result click the ``Save`` buttons in the UI, which call
:meth:`save_clip` / :meth:`save_folder`. Those copy the NPZ into
``assets/save_npz/<FolderLabel>/`` — a location that is *outside* the cache dir,
so the shutdown wipe never touches saved files.

Cache directory rules:

- If the caller passes ``cache_dir=None`` (the default), we mint a brand-new
  ``tempfile.mkdtemp("hhtools_cache_")`` for the session. On :meth:`cleanup` that
  directory is ``shutil.rmtree``'d in full.
- If the caller passes an explicit ``cache_dir``, we treat it as *shared* — we
  only remove files this instance itself wrote (tracked in :attr:`written`).
  Any other content the user left behind stays untouched.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from hhtools.core.motion import Motion
from hhtools.core.scene import SceneObject
from hhtools.io import npz as _npz
from hhtools.io.datasets import registered_datasets
from hhtools.viewer.library import LibraryEntry


def _attach_library_folder_label(motion: Motion, entry: LibraryEntry) -> None:
    """Tag ``motion.meta`` so :func:`hhtools.core.grounding.use_split_terrain_grounding` can route."""

    motion.meta["library_folder_label"] = entry.folder_label


@dataclass
class EphemeralCache:
    """Manage lazy on-disk NPZs for a viewer session.

    Attributes:
        cache_dir: Where freshly-converted NPZs land this session. Auto-created
            under ``/tmp`` when the default ``create(cache_dir=None)`` is used.
        save_dir: Destination for user-persisted NPZs (``assets/save_npz``).
        owns_cache_dir: ``True`` if ``cache_dir`` was minted by :meth:`create`
            itself, in which case :meth:`cleanup` rmtrees the whole directory.
            ``False`` when the caller supplied a directory, in which case
            :meth:`cleanup` only unlinks files this instance wrote.
        written: Basenames of NPZs this session wrote. Populated by :meth:`get`
            and unchanged by :meth:`save_clip` (saves live outside cache_dir).
    """

    cache_dir: Path
    save_dir: Path
    owns_cache_dir: bool = False
    written: set[str] = field(default_factory=set)

    @classmethod
    def create(
        cls,
        cache_dir: str | Path | None,
        save_dir: str | Path,
    ) -> EphemeralCache:
        """Construct the cache.

        If ``cache_dir`` is ``None`` we allocate a fresh ``tempfile.mkdtemp`` and
        record ``owns_cache_dir=True`` so :meth:`cleanup` can rmtree the whole
        thing. Otherwise we use the supplied path and only remove entries we
        wrote ourselves on cleanup.
        """
        if cache_dir is None:
            cache = Path(tempfile.mkdtemp(prefix="hhtools_cache_"))
            owns = True
        else:
            cache = Path(cache_dir)
            cache.mkdir(parents=True, exist_ok=True)
            owns = False
        save = Path(save_dir)
        save.mkdir(parents=True, exist_ok=True)
        return cls(cache_dir=cache, save_dir=save, owns_cache_dir=owns, written=set())

    # ------------------------------------------------------------------ conversion

    def get(
        self,
        entry: LibraryEntry,
        progress_callback=None,
    ) -> Path:
        """Return the NPZ path for ``entry``, converting the raw file on demand.

        Within a session we reuse an existing NPZ as long as the source file
        hasn't been re-saved underneath us (mtime check). The converted file's
        basename is recorded in :attr:`written` so :meth:`cleanup` can find it
        when the cache dir is shared with other tooling.

        :func:`hhtools.io.npz.save_npz` may also drop a sidecar ``.pkl``
        next to the NPZ when the motion carries a heightfield terrain
        (this is what makes ``Motion.terrain`` survive a cache round
        trip).  We track that sidecar in :attr:`written` too so the
        shared-cache branch of :meth:`cleanup` collects it.
        """
        dst = self.cache_dir / entry.cache_name

        if dst.exists() and dst.stat().st_mtime >= entry.source_path.stat().st_mtime:
            if progress_callback is not None:
                progress_callback(0.92, f"使用缓存 {entry.stem}")
            return dst

        if progress_callback is not None:
            progress_callback(0.05, f"解析 {entry.stem}…")
        motion = self._convert(entry, progress_callback=progress_callback)
        if progress_callback is not None:
            progress_callback(0.88, "写入缓存…")
        _npz.save_npz(motion, dst)
        self.written.add(dst.name)
        sidecar = dst.with_suffix(".pkl")
        if sidecar.is_file():
            self.written.add(sidecar.name)
        return dst

    def load_motion(
        self,
        entry: LibraryEntry,
        progress_callback=None,
    ) -> Motion:
        """Shortcut: convert if needed, then return the loaded :class:`Motion`."""
        if progress_callback is not None:
            progress_callback(0.0, f"读取 {entry.stem}…")
        m = _npz.load_npz(self.get(entry, progress_callback=progress_callback))
        if progress_callback is not None:
            progress_callback(1.0, f"已加载 {entry.stem}")
        _attach_library_folder_label(m, entry)
        return m

    def _convert(self, entry: LibraryEntry, progress_callback=None) -> Motion:
        registry = registered_datasets()
        adapter_cls = registry.get(entry.dataset)
        if adapter_cls is None:
            raise KeyError(
                f"No adapter registered for dataset '{entry.dataset}'. "
                f"Known: {sorted(registry)}"
            )
        adapter = adapter_cls(entry.source_path.parent)
        motion = adapter.load_motion(
            entry.adapter_sequence_id,
            progress_callback=progress_callback,
        )
        _attach_library_folder_label(motion, entry)
        return motion

    # ---------------------------------------------------------------------- save

    def save_clip(self, entry: LibraryEntry) -> Path:
        """Persist ``entry`` into ``save_dir`` as a self-contained per-clip bundle.

        Output layout mirrors the source-tree convention already used under
        ``assets/motions/<category>/<source>/<clip>/``::

            save_dir/
              <folder_label>/                # e.g. OMOMO, holosoma
                <clip_stem>/                 # e.g. sub10_largebox_000, parkour_1
                  <clip_stem>.npz            # body + scene-object trajectories
                  <mesh_basename>.obj        # one file per referenced mesh

        The returned path is the NPZ itself. Sibling ``.obj`` files carry the
        referenced :attr:`SceneObject.mesh_path` geometry copied out of the
        source tree, and the NPZ's ``objects_mesh_paths`` field is rewritten to
        hold just the basenames — relative-path resolution in
        :func:`hhtools.io.npz.load_npz` then points back at those siblings
        regardless of where the bundle is moved to.

        Implementation notes:

        * We go load → mutate → save rather than ``shutil.copy2`` the cache
          NPZ because the cache version stores absolute mesh paths (those are
          correct for live playback but defeat portability).
        * Mesh de-duplication keys on the *source* path so two scene objects
          sharing a mesh copy it once; basename collisions from two different
          sources get a numeric suffix so nothing gets clobbered.
        * Empty ``mesh_path`` (cuboid placeholder) is preserved as an empty
          string in the saved NPZ — the viewer falls back to ``extents`` just
          as it does today for OMOMO clips without captured meshes.
        """
        src_npz = self.get(entry)
        motion = _npz.load_npz(src_npz)
        _attach_library_folder_label(motion, entry)

        dst_dir = self.save_dir / entry.folder_label / entry.stem
        dst_dir.mkdir(parents=True, exist_ok=True)

        new_objects: list[SceneObject] = []
        # Map: absolute source mesh path → basename we ended up using in dst_dir.
        # Used for both intra-clip de-dup and inter-object collision detection.
        mesh_src_to_basename: dict[str, str] = {}
        used_basenames: set[str] = set()
        for obj in motion.objects:
            relative_mesh = ""
            if obj.mesh_path:
                src_mesh = Path(obj.mesh_path)
                if src_mesh.is_file():
                    src_key = str(src_mesh.resolve())
                    if src_key in mesh_src_to_basename:
                        relative_mesh = mesh_src_to_basename[src_key]
                    else:
                        relative_mesh = _unique_basename(src_mesh.name, used_basenames)
                        shutil.copy2(src_mesh, dst_dir / relative_mesh)
                        mesh_src_to_basename[src_key] = relative_mesh
                        used_basenames.add(relative_mesh)
            new_objects.append(
                SceneObject(
                    name=obj.name,
                    positions=obj.positions,
                    quaternions=obj.quaternions,
                    extents=obj.extents,
                    mesh_path=relative_mesh,
                    scale=obj.scale,
                    opacity=obj.opacity,
                    color=obj.color,
                )
            )

        relocatable = Motion(
            name=motion.name,
            hierarchy=motion.hierarchy,
            positions=motion.positions,
            quaternions=motion.quaternions,
            framerate=motion.framerate,
            up_axis=motion.up_axis,
            source_format=motion.source_format,
            meta=motion.meta,
            objects=new_objects,
            # Preserve terrain so saved bundles carry their own heightfield
            # sidecar (save_npz writes ``<stem>.pkl`` automatically).  Without
            # this, ``Save folder`` bundles would silently drop terrain even
            # though the in-memory motion still has it.
            terrain=motion.terrain,
        )

        dst_npz = dst_dir / f"{entry.stem}.npz"
        _npz.save_npz(relocatable, dst_npz)
        return dst_npz

    def save_folder(self, entries: list[LibraryEntry]) -> list[Path]:
        """Persist every entry in ``entries`` (typically an entire folder bucket)."""
        saved: list[Path] = []
        for e in entries:
            saved.append(self.save_clip(e))
        return saved

    # ----------------------------------------------------------------- cleanup

    def cleanup(self) -> None:
        """Remove every NPZ this session created. Idempotent.

        When we own the cache directory (``owns_cache_dir=True``, the default
        tempfile case) the entire directory is rmtree'd so Linux reclaims the
        inode. When the caller supplied the path, we only unlink files listed
        in :attr:`written` — anything else the user left behind is untouched.
        """
        if self.owns_cache_dir and self.cache_dir.exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.written.clear()
            return
        for name in list(self.written):
            p = self.cache_dir / name
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
            self.written.discard(name)

    # ------------------------------------------------------------ introspection

    def is_saved(self, entry: LibraryEntry) -> bool:
        """Return ``True`` if a persisted copy exists under ``save_dir``.

        Checks the per-clip layout produced by :meth:`save_clip`; legacy flat
        saves (``save_dir/<folder>/<dataset>__<stem>.npz``) are intentionally
        ignored so the UI prompts users to re-save into the new layout.
        """
        dst = self.save_dir / entry.folder_label / entry.stem / f"{entry.stem}.npz"
        return dst.exists()

    def summary(self) -> dict[str, int]:
        """Small status dict used by the viewer's status label."""
        return {"written": len(self.written)}


def _unique_basename(candidate: str, taken: set[str]) -> str:
    """Return a filename that doesn't clash with any name already in ``taken``.

    ``save_clip`` de-duplicates by *source* path first (two scene objects sharing
    one mesh → one copy), but when two distinct source meshes happen to have the
    same basename (``box.obj`` from two different clips would never collide
    because each bundle has its own folder, but a single clip can in principle
    reference two different files named the same), we fall back to this numeric
    suffix scheme. Extensions are preserved; ``terrain.obj`` → ``terrain_1.obj``
    → ``terrain_2.obj`` …
    """
    if candidate not in taken:
        return candidate
    stem, suffix = Path(candidate).stem, Path(candidate).suffix
    i = 1
    while True:
        alt = f"{stem}_{i}{suffix}"
        if alt not in taken:
            return alt
        i += 1


__all__ = ["EphemeralCache"]
