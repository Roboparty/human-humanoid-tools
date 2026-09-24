"""Small daemon-thread scheduler for Web jobs.

The default mode deliberately preserves the historical behaviour: every
accepted job starts immediately.  Deployments that need GPU admission control
can set a positive running limit; only then does the FIFO waiting queue exist.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from hhtools.services.admission import AdmissionClosedError, AdmissionQueueFullError

_log = logging.getLogger(__name__)


class JobQueueFullError(AdmissionQueueFullError):
    """Raised when a bounded waiting queue has no admission slot left."""


class JobSchedulerClosedError(AdmissionClosedError):
    """Raised when work is submitted after scheduler shutdown begins."""


@dataclass(frozen=True)
class SchedulerSnapshot:
    """Thread-safe scheduler counters for diagnostics and tests."""

    max_running_jobs: int
    max_queued_jobs: int
    running_jobs: int
    queued_jobs: int
    reserved_jobs: int
    cancelling_jobs: int
    closed: bool


@dataclass(frozen=True)
class _ScheduledCall:
    token: int
    run: Callable[[], object]
    on_cancel: Callable[[str], None] | None = None


class ScheduledJobHandle:
    """Opaque identity for one submitted call.

    The handle can remove a call while it is still waiting in the FIFO queue.
    Once execution has started, Python cannot safely stop the worker thread and
    ``cancel`` returns ``False``; a higher-level JobManager must then use a
    cooperative cancellation token.
    """

    def __init__(self, scheduler: JobScheduler, token: int) -> None:
        self._scheduler = scheduler
        self._token = token

    def cancel(self) -> bool:
        """Cancel exactly this queued call, returning whether it was removed."""

        return self._scheduler._cancel_scheduled(self._token)  # noqa: SLF001

    def queue_position(self) -> int | None:
        """Return the current one-based FIFO position, if still queued."""

        return self._scheduler._queue_position(self._token)  # noqa: SLF001


class JobReservation:
    """Admission token reserved before an upload performs filesystem writes."""

    def __init__(self, scheduler: JobScheduler, token: int) -> None:
        self._scheduler = scheduler
        self._token = token

    def submit(
        self,
        run: Callable[[], object],
        *,
        on_cancel: Callable[[str], None] | None = None,
    ) -> ScheduledJobHandle:
        """Consume this reservation and submit its work exactly once."""

        return self._scheduler._submit_reserved(  # noqa: SLF001 - paired internal type
            self._token,
            run,
            on_cancel=on_cancel,
        )

    def cancel(self) -> None:
        """Release an unused reservation; repeated calls are harmless."""

        self._scheduler._cancel_reservation(self._token)  # noqa: SLF001


class JobScheduler:
    """Run jobs immediately or gate them behind a configurable FIFO queue.

    ``max_running_jobs == 0`` means unlimited concurrency and makes the queue
    setting irrelevant.  With a positive running limit, ``max_queued_jobs ==
    0`` means an unbounded waiting queue; a positive value bounds only waiting
    jobs, not jobs that are already running.
    """

    def __init__(self, *, max_running_jobs: int = 0, max_queued_jobs: int = 0) -> None:
        if int(max_running_jobs) < 0 or int(max_queued_jobs) < 0:
            raise ValueError("job scheduler limits must be non-negative")
        self.max_running_jobs = int(max_running_jobs)
        self.max_queued_jobs = int(max_queued_jobs)
        self._condition = threading.Condition(threading.Lock())
        self._pending: deque[_ScheduledCall] = deque()
        self._reservations: set[int] = set()
        self._next_token = 1
        self._running = 0
        self._cancelling = 0
        self._closed = False

    def reserve(self) -> JobReservation:
        """Reserve admission, allowing upload routes to fail before writing data."""

        with self._condition:
            if self._closed:
                raise JobSchedulerClosedError("job scheduler is shutting down")
            if self.max_running_jobs > 0 and self.max_queued_jobs > 0:
                admitted = self._running + len(self._pending) + len(self._reservations)
                capacity = self.max_running_jobs + self.max_queued_jobs
                if admitted >= capacity:
                    raise JobQueueFullError(
                        f"job queue is full (waiting limit: {self.max_queued_jobs})"
                    )
            token = self._next_token
            self._next_token += 1
            self._reservations.add(token)
        return JobReservation(self, token)

    def submit(
        self,
        run: Callable[[], object],
        *,
        on_cancel: Callable[[str], None] | None = None,
    ) -> ScheduledJobHandle:
        """Reserve and submit a call in one operation."""

        reservation = self.reserve()
        try:
            return reservation.submit(run, on_cancel=on_cancel)
        except BaseException:
            reservation.cancel()
            raise

    def snapshot(self) -> SchedulerSnapshot:
        with self._condition:
            return SchedulerSnapshot(
                max_running_jobs=self.max_running_jobs,
                max_queued_jobs=self.max_queued_jobs,
                running_jobs=self._running,
                queued_jobs=len(self._pending),
                reserved_jobs=len(self._reservations),
                cancelling_jobs=self._cancelling,
                closed=self._closed,
            )

    def reconfigure(
        self,
        *,
        max_running_jobs: int,
        max_queued_jobs: int,
    ) -> SchedulerSnapshot:
        """Apply new admission limits without interrupting active work.

        Lowering the running limit grandfathers calls that already hold a slot;
        later completions stop promoting queued work until the new limit has
        room.  Raising the limit (or switching to unlimited mode) promotes the
        oldest waiting calls immediately.  A smaller queue limit never cancels
        work that was already admitted; it only affects future reservations.
        """

        running_limit = int(max_running_jobs)
        queued_limit = int(max_queued_jobs)
        if running_limit < 0 or queued_limit < 0:
            raise ValueError("job scheduler limits must be non-negative")

        promoted: list[_ScheduledCall] = []
        with self._condition:
            if self._closed:
                raise JobSchedulerClosedError("job scheduler is shutting down")
            self.max_running_jobs = running_limit
            self.max_queued_jobs = queued_limit

            available = (
                len(self._pending) if running_limit == 0 else max(0, running_limit - self._running)
            )
            for _ in range(min(available, len(self._pending))):
                call = self._pending.popleft()
                promoted.append(call)
                self._running += 1
            self._condition.notify_all()

        # Starting threads outside the scheduler lock keeps submit/snapshot
        # responsive.  The slots are already counted, so concurrent completions
        # cannot over-promote beyond the new limit.
        for call in promoted:
            self._start(call, propagate_failure=False)
        return self.snapshot()

    def shutdown(
        self,
        *,
        wait: bool = False,
        timeout: float | None = None,
    ) -> bool:
        """Reject new work, cancel pending calls, and optionally wait for runners.

        Python threads cannot safely be force-stopped.  The return value reports
        whether all running calls and cancellation callbacks finished before
        ``timeout``.
        """

        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()
            # Reservations have no Job object yet.  Invalidating the tokens makes
            # their eventual submit fail cleanly with JobSchedulerClosedError.
            self._reservations.clear()
            self._cancelling += len(pending)
            self._condition.notify_all()

        if pending:
            cancellation_thread = threading.Thread(
                target=self._cancel_pending,
                args=(pending,),
                name="hhtools-web-job-cancellation",
                daemon=True,
            )
            try:
                cancellation_thread.start()
            except RuntimeError:
                # Thread creation failure is exceptional; preserve correctness
                # by completing callbacks in the caller before returning.
                self._cancel_pending(pending)

        if not wait:
            snapshot = self.snapshot()
            return snapshot.running_jobs == 0 and snapshot.cancelling_jobs == 0

        with self._condition:
            while self._running or self._cancelling:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            return self._running == 0 and self._cancelling == 0

    def _submit_reserved(
        self,
        token: int,
        run: Callable[[], object],
        *,
        on_cancel: Callable[[str], None] | None,
    ) -> ScheduledJobHandle:
        call = _ScheduledCall(token=token, run=run, on_cancel=on_cancel)
        with self._condition:
            if token not in self._reservations:
                if self._closed:
                    raise JobSchedulerClosedError("job scheduler is shutting down")
                raise RuntimeError("job reservation is no longer valid")
            self._reservations.remove(token)
            if self._closed:
                raise JobSchedulerClosedError("job scheduler is shutting down")

            starts_now = self.max_running_jobs == 0 or self._running < self.max_running_jobs
            if starts_now:
                self._running += 1
            else:
                self._pending.append(call)

        if starts_now:
            self._start(call, propagate_failure=True)
        return ScheduledJobHandle(self, token)

    def _cancel_reservation(self, token: int) -> None:
        with self._condition:
            self._reservations.discard(token)
            self._condition.notify_all()

    def _queue_position(self, token: int) -> int | None:
        with self._condition:
            for position, call in enumerate(self._pending, start=1):
                if call.token == token:
                    return position
        return None

    def _cancel_scheduled(self, token: int) -> bool:
        cancelled: _ScheduledCall | None = None
        with self._condition:
            for index, call in enumerate(self._pending):
                if call.token == token:
                    cancelled = call
                    del self._pending[index]
                    self._cancelling += 1
                    self._condition.notify_all()
                    break
            if cancelled is None:
                return False

        try:
            self._notify_cancel(cancelled, "任务在等待队列中被取消。")
        finally:
            with self._condition:
                self._cancelling -= 1
                self._condition.notify_all()
        return True

    def _start(self, call: _ScheduledCall, *, propagate_failure: bool) -> None:
        """Start one reserved call, iteratively draining calls if starts fail."""

        current: _ScheduledCall | None = call
        first_error: BaseException | None = None
        while current is not None:
            thread = threading.Thread(
                target=self._execute,
                args=(current,),
                name="hhtools-web-job",
                daemon=True,
            )
            start_error: BaseException | None = None
            cancelled_by_shutdown = False
            with self._condition:
                if self._closed:
                    # A call can be claimed for immediate or promoted execution
                    # just before shutdown closes admission.  Return its slot and
                    # cancel it instead of starting new work after that boundary.
                    self._running -= 1
                    self._cancelling += 1
                    cancelled_by_shutdown = True
                    self._condition.notify_all()
                else:
                    try:
                        # Starting while holding the condition linearizes this
                        # transition against shutdown.  The new worker can run
                        # user code immediately, but its final accounting waits
                        # for this short critical section to finish.
                        thread.start()
                    except BaseException as err:
                        start_error = err
                        self._running -= 1
                        # Terminal persistence performed by on_cancel is part of
                        # shutdown.  Track it before exposing running == 0 so a
                        # concurrent cleanup cannot delete files beneath it.
                        self._cancelling += 1
                        self._condition.notify_all()

            if start_error is None and not cancelled_by_shutdown:
                if first_error is not None and propagate_failure:
                    raise first_error
                if first_error is not None:
                    _log.error(
                        "one or more queued Web job threads failed to start: %s",
                        first_error,
                    )
                return

            if cancelled_by_shutdown:
                current_error: BaseException = JobSchedulerClosedError(
                    "job scheduler is shutting down",
                )
                reason = "Web 服务关闭，任务尚未开始。"
            else:
                assert start_error is not None
                current_error = start_error
                reason = "无法启动后台任务线程。"
            if first_error is None:
                first_error = current_error
            try:
                self._notify_cancel(current, reason)
            finally:
                with self._condition:
                    self._cancelling -= 1
                    # Choose the replacement only after cancellation.  If
                    # shutdown began meanwhile, pending calls have already
                    # been removed and no new worker may start.
                    next_call = self._take_next_locked()
                    self._condition.notify_all()
            current = next_call
            continue

        if first_error is not None and propagate_failure:
            raise first_error
        if first_error is not None:
            _log.error(
                "all promoted Web job threads failed to start: %s",
                first_error,
            )

    def _execute(self, call: _ScheduledCall) -> None:
        try:
            call.run()
        finally:
            with self._condition:
                self._running -= 1
                next_call = self._take_next_locked()
                self._condition.notify_all()
            if next_call is not None:
                self._start(next_call, propagate_failure=False)

    def _take_next_locked(self) -> _ScheduledCall | None:
        if self._closed or not self._pending:
            return None
        if self.max_running_jobs > 0 and self._running >= self.max_running_jobs:
            return None
        call = self._pending.popleft()
        self._running += 1
        return call

    def _cancel_pending(self, pending: list[_ScheduledCall]) -> None:
        try:
            for call in pending:
                self._notify_cancel(call, "Web 服务关闭，排队任务尚未开始。")
        finally:
            with self._condition:
                self._cancelling -= len(pending)
                self._condition.notify_all()

    @staticmethod
    def _notify_cancel(call: _ScheduledCall, reason: str) -> None:
        if call.on_cancel is None:
            return
        try:
            call.on_cancel(reason)
        except Exception:  # noqa: BLE001 - one callback must not block shutdown
            _log.exception("failed to cancel queued Web job")


__all__ = [
    "JobQueueFullError",
    "JobReservation",
    "JobScheduler",
    "JobSchedulerClosedError",
    "ScheduledJobHandle",
    "SchedulerSnapshot",
]
