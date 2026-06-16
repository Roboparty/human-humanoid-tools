# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Bridge motion-load sub-stages into the web UI job progress bar."""

from __future__ import annotations

from typing import Any, Callable

ProgressFn = Callable[[float, str], None]


class MotionLoadProgress:
    """Map a sub-stage fraction ``[0, 1]`` into a global job progress slice."""

    def __init__(self, job: Any, *, base: float, span: float) -> None:
        self._job = job
        self._base = float(base)
        self._span = float(span)

    def report(self, frac: float, message: str) -> None:
        f = max(0.0, min(1.0, float(frac)))
        self._job.message = str(message)
        self._job.progress = self._base + self._span * f

    def as_callback(self) -> ProgressFn:
        return self.report


__all__ = ["MotionLoadProgress", "ProgressFn"]
