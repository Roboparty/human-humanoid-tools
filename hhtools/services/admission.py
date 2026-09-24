"""Transport-neutral protocols for shared background-job admission.

The Web scheduler implements these small interfaces, while JobManager depends
only on the application-service boundary.  This avoids making REST, CLI, or
MCP adapters part of the execution semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class AdmissionQueueFullError(RuntimeError):
    """Raised before durable work when the configured waiting queue is full."""


class AdmissionClosedError(RuntimeError):
    """Raised when the execution scheduler no longer accepts work."""


class ScheduledHandle(Protocol):
    """Opaque identity for one admitted callable."""

    def cancel(self) -> bool: ...

    def queue_position(self) -> int | None: ...


class AdmissionReservation(Protocol):
    """One capacity token reserved before a durable job is created."""

    def submit(
        self,
        run: Callable[[], object],
        *,
        on_cancel: Callable[[str], None] | None = None,
    ) -> ScheduledHandle: ...

    def cancel(self) -> None: ...


class AdmissionScheduler(Protocol):
    """Shared scheduler surface consumed by the application service."""

    def reserve(self) -> AdmissionReservation: ...

    def snapshot(self) -> object: ...


__all__ = [
    "AdmissionClosedError",
    "AdmissionQueueFullError",
    "AdmissionReservation",
    "AdmissionScheduler",
    "ScheduledHandle",
]
