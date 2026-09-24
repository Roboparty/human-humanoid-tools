# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workspace-local Warp kernel cache configuration.

Warp writes JIT-compiled kernels to ``~/.cache/warp/<version>/`` by default.
Many CI / sandboxed environments can't write there (e.g. read-only home,
permission-restricted execution).  This module lets callers (CLI, UI, tests)
opt-in to a workspace-local cache location without touching Warp directly.

Invariants:

* Called at most once per process — subsequent imports are a no-op.
* Honours ``WARP_CACHE_DIR`` if the user already set it.
* Falls back to ``<workspace>/.hhtools/warp_cache`` when we can detect a
  workspace (detected by walking up for ``pyproject.toml``) and the default
  Warp location isn't writable.

The import side-effect happens only when callers explicitly invoke
:func:`configure` (we deliberately avoid import-time magic so ``import
hhtools.retarget.newton_basic`` stays side-effect-free).
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

__all__ = ["configure", "is_configured", "resolved_cache_dir", "is_cache_persistent"]

_log = logging.getLogger("hhtools.warp_cache")

_configured = False
_resolved: Path | None = None
_persistent = False


def is_configured() -> bool:
    return _configured


def resolved_cache_dir() -> Path | None:
    """The Warp kernel-cache directory chosen by the last :func:`configure`."""
    return _resolved


def is_cache_persistent() -> bool:
    """Whether the resolved cache survives across processes.

    ``False`` means we fell back to a throwaway temp dir, so kernels will be
    JIT-recompiled on every run — usually a symptom of a non-writable
    ``~/.cache/warp`` or a stale root-owned ``.hhtools/warp_cache``.
    """
    return _persistent


def configure(
    explicit: str | os.PathLike[str] | None = None,
    *,
    quiet: bool | None = None,
) -> Path:
    """Point Warp at a writable kernel cache directory.

    Must be called *before* the first ``import warp``; we rely on the
    ``WARP_CACHE_DIR`` env var rather than ``warp.config.kernel_cache_dir``
    because Warp reads the cache path at ``init`` time and subsequent
    attribute writes are not honoured for already-initialised modules.

    Args:
        explicit: If given, use this path unconditionally.  Directory is
            created on demand.  Otherwise we honour ``WARP_CACHE_DIR`` /
            default, and fall back to the workspace-local cache when the
            default isn't writable.
        quiet: Optional process-local Warp diagnostic setting.  ``True`` is
            used by stdio protocol hosts so Warp cannot write its first-init
            banner into the wire.  The default ``None`` preserves Warp's
            existing behavior for the CLI and WebUI.

    Returns:
        The resolved cache directory.
    """

    def _commit(path: Path, *, persistent: bool) -> Path:
        global _configured, _resolved, _persistent  # noqa: PLW0603 - module config snapshot
        path.mkdir(parents=True, exist_ok=True)
        os.environ["WARP_CACHE_DIR"] = str(path)
        # Also update config for already-loaded ``warp`` modules; this is
        # effective for the *next* compile in the same process, but the
        # env var is the authoritative setting.
        try:
            import warp as wp  # local import — may not be imported yet

            # Importing Warp itself is silent; its first ``wp.init()`` emits
            # the device banner.  Set this before any Newton import can cause
            # that initialization.  Do not redirect process-wide stdout:
            # concurrent stdio traffic could otherwise be captured with it.
            if quiet is not None:
                wp.config.quiet = quiet
            wp.config.kernel_cache_dir = str(path)
        except ImportError:
            pass
        _configured = True
        _resolved = path
        _persistent = persistent
        if persistent:
            _log.info("Warp kernel cache: %s (persistent)", path)
        else:
            _log.warning(
                "Warp kernel cache fell back to a non-persistent temp dir: %s. "
                "GPU kernels will be RECOMPILED every run. Make a stable cache "
                "writable, e.g.:\n"
                '  sudo chown -R "$USER" ~/.cache/warp .hhtools/warp_cache 2>/dev/null; '
                "rm -rf ~/.cache/warp/* .hhtools/warp_cache/*\n"
                "or set WARP_CACHE_DIR to a writable, persistent path.",
                path,
            )
        return path

    if explicit is not None:
        cache_dir = Path(explicit).expanduser().resolve()
        return _commit(cache_dir, persistent=_dir_writable(cache_dir))

    env = os.environ.get("WARP_CACHE_DIR")
    if env:
        cache_dir = Path(env).expanduser().resolve()
        if _dir_writable(cache_dir):
            return _commit(cache_dir, persistent=True)
        _log.warning(
            "WARP_CACHE_DIR=%s is not writable; falling back to a writable "
            "location so kernels persist across runs.",
            cache_dir,
        )

    # Probe each candidate for writability without importing warp (which would
    # lock in the default).  The first WRITABLE candidate wins; a dir that
    # merely *exists* but is owned by another user (the classic stale
    # root-owned cache) would otherwise silently force a recompile each run.
    candidates: list[Path] = [Path.home() / ".cache" / "warp"]
    workspace = _find_workspace()
    if workspace is not None:
        candidates.append(workspace / ".hhtools" / "warp_cache")
    for cand in candidates:
        if _dir_writable(cand):
            return _commit(cand.resolve(), persistent=True)

    # Stable per-user temp dir: still persists across runs for the same user
    # (unlike a random ``mkdtemp``), so a non-writable home/workspace does not
    # condemn every run to a full JIT recompile.
    try:
        uid = os.getuid()  # type: ignore[attr-defined]
    except AttributeError:
        uid = os.environ.get("USERNAME") or os.environ.get("USER") or "user"
    stable_tmp = Path(tempfile.gettempdir()) / f"hhtools_warp_cache_{uid}"
    if _dir_writable(stable_tmp):
        return _commit(stable_tmp.resolve(), persistent=True)

    tmp = Path(tempfile.mkdtemp(prefix="hhtools_warp_cache_"))
    return _commit(tmp, persistent=False)


def _dir_writable(path: Path) -> bool:
    """Check writability by actually creating + deleting a temp file.

    ``os.access`` is not reliable under bind-mounted filesystems or
    sandboxes where permission bits say yes but the write-time kernel
    policy says no.  A real write probe is the only reliable test.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    try:
        with tempfile.NamedTemporaryFile(dir=str(path), prefix=".hhtools_writetest_", delete=True):
            pass
    except OSError:
        return False
    return True


def _find_workspace() -> Path | None:
    here = Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / "pyproject.toml").is_file():
            return ancestor
    return None
