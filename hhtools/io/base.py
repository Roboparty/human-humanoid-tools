"""Format-dispatch helpers for motion IO.

Loaders and savers are registered against lowercase file extensions. Third-party code (or
downstream projects) can plug in new formats by calling :func:`register_loader` /
:func:`register_saver`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from hhtools.core.motion import Motion


class MotionLoader(Protocol):
    """Callable that reads a file path into a :class:`Motion`.

    Implementations must accept an optional ``progress_callback(frac, message)``
    keyword (see :mod:`hhtools.io.loader_progress`) even when they ignore it.
    """

    def __call__(self, path: Path, **kwargs) -> Motion: ...


class MotionSaver(Protocol):
    """Callable that writes a :class:`Motion` to a file path."""

    def __call__(self, motion: Motion, path: Path, **kwargs) -> None: ...


_LOADERS: dict[str, MotionLoader] = {}
_SAVERS: dict[str, MotionSaver] = {}


def register_loader(extension: str, loader: MotionLoader) -> None:
    """Register a loader for ``extension`` (with or without the leading dot, case-insensitive)."""
    _LOADERS[_norm(extension)] = loader


def register_saver(extension: str, saver: MotionSaver) -> None:
    _SAVERS[_norm(extension)] = saver


def unregister_loader(extension: str) -> None:
    _LOADERS.pop(_norm(extension), None)


def unregister_saver(extension: str) -> None:
    _SAVERS.pop(_norm(extension), None)


def registered_loader_extensions() -> list[str]:
    return sorted(_LOADERS)


def registered_saver_extensions() -> list[str]:
    return sorted(_SAVERS)


def load_motion(path: str | Path, **kwargs) -> Motion:
    """Dispatch to the registered loader for ``path``'s extension."""
    p = Path(path)
    ext = _norm(p.suffix)
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"No loader registered for extension {ext!r}; "
            f"supported: {registered_loader_extensions()}"
        )
    return loader(p, **kwargs)


def save_motion(motion: Motion, path: str | Path, **kwargs) -> None:
    """Dispatch to the registered saver for ``path``'s extension."""
    p = Path(path)
    ext = _norm(p.suffix)
    saver = _SAVERS.get(ext)
    if saver is None:
        raise ValueError(
            f"No saver registered for extension {ext!r}; supported: {registered_saver_extensions()}"
        )
    saver(motion, p, **kwargs)


def _norm(ext: str) -> str:
    ext = ext.lower()
    if not ext.startswith("."):
        ext = "." + ext
    return ext


# ---------------------------------------------------------------------- auto registration


def _install_default_handlers() -> None:
    # Imported here to avoid circular imports during package init.
    from hhtools.io import bvh as _bvh
    from hhtools.io import glb as _glb
    from hhtools.io import npz as _npz

    register_loader(".bvh", _bvh.load_bvh)
    register_saver(".bvh", _bvh.save_bvh)
    register_loader(".npz", _npz.load_npz)
    register_saver(".npz", _npz.save_npz)
    # GLB / glTF share the same importer — both extensions point at the same callable
    # and the function handles JSON vs binary GLB transparently via pygltflib. pygltflib
    # itself is an optional dep (``hhtools[formats]`` extra); ``load_glb`` raises a
    # friendly ModuleNotFoundError if the user didn't install it.
    register_loader(".glb", _glb.load_glb)
    register_loader(".gltf", _glb.load_glb)


_install_default_handlers()


# Convenience: expose as module-level attributes (``hhtools.io.base._register``)
__all__ = [
    "MotionLoader",
    "MotionSaver",
    "load_motion",
    "register_loader",
    "register_saver",
    "registered_loader_extensions",
    "registered_saver_extensions",
    "save_motion",
    "unregister_loader",
    "unregister_saver",
]


# ---------------------------------------------------------------------- typing
_Fn = Callable[..., object]
