"""Hot-reconfigurable optional limits for Agent batch composition."""

from __future__ import annotations

import threading
from dataclasses import dataclass


def _limit(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class BatchLimitSnapshot:
    """Zero means unlimited; positive values are administrator-selected caps."""

    max_batch_items: int = 0
    max_batch_total_frames: int = 0


class BatchLimitPolicy:
    """Keep live batch caps independent from scheduler concurrency limits."""

    def __init__(
        self,
        *,
        max_batch_items: int = 0,
        max_batch_total_frames: int = 0,
    ) -> None:
        self._lock = threading.RLock()
        self._snapshot = BatchLimitSnapshot(
            max_batch_items=_limit(max_batch_items, name="max_batch_items"),
            max_batch_total_frames=_limit(
                max_batch_total_frames,
                name="max_batch_total_frames",
            ),
        )

    def snapshot(self) -> BatchLimitSnapshot:
        with self._lock:
            return self._snapshot

    def reconfigure(
        self,
        *,
        max_batch_items: int,
        max_batch_total_frames: int,
    ) -> BatchLimitSnapshot:
        updated = BatchLimitSnapshot(
            max_batch_items=_limit(max_batch_items, name="max_batch_items"),
            max_batch_total_frames=_limit(
                max_batch_total_frames,
                name="max_batch_total_frames",
            ),
        )
        with self._lock:
            self._snapshot = updated
            return updated


__all__ = ["BatchLimitPolicy", "BatchLimitSnapshot"]
