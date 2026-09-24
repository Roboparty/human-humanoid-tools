# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared progress-hook contract for motion file loaders.

The web UI and :class:`~hhtools.services.motion_cache.EphemeralCache` pass an optional
``progress_callback(frac, message)`` through dataset adapters into format
loaders (:func:`~hhtools.io.bvh.load_bvh`, :func:`~hhtools.io.npz.load_npz`,
etc.).  Every registered :class:`~hhtools.io.base.MotionLoader` **must**
accept this keyword (and may ignore it when parsing is instantaneous).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ProgressCallback = Callable[[float, str], None]


def pop_progress_callback(kwargs: dict[str, Any]) -> ProgressCallback | None:
    """Remove and return ``progress_callback`` from loader kwargs."""
    cb = kwargs.pop("progress_callback", None)
    return cb if cb is not None else None


def report_progress(
    callback: ProgressCallback | None,
    frac: float,
    message: str,
) -> None:
    """Invoke ``callback`` when present; swallow caller exceptions."""
    if callback is None:
        return
    try:
        callback(max(0.0, min(1.0, float(frac))), str(message))
    except Exception:
        pass


__all__ = ["ProgressCallback", "pop_progress_callback", "report_progress"]
