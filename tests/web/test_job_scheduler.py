from __future__ import annotations

import threading
import time
from functools import partial

import pytest

from hhtools.web.jobs.job_scheduler import (
    JobQueueFullError,
    JobScheduler,
    JobSchedulerClosedError,
)


def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


def test_default_scheduler_starts_every_job_without_a_queue() -> None:
    scheduler = JobScheduler()
    release = threading.Event()
    lock = threading.Lock()
    started = 0

    def run() -> None:
        nonlocal started
        with lock:
            started += 1
        release.wait(timeout=2.0)

    for _ in range(4):
        scheduler.submit(run)

    _wait_until(lambda: started == 4)
    snapshot = scheduler.snapshot()
    assert snapshot.max_running_jobs == 0
    assert snapshot.running_jobs == 4
    assert snapshot.queued_jobs == 0

    release.set()
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_queue_limit_is_ignored_without_a_running_cap() -> None:
    scheduler = JobScheduler(max_running_jobs=0, max_queued_jobs=1)
    release = threading.Event()

    for _ in range(3):
        scheduler.submit(lambda: release.wait(timeout=2.0))

    _wait_until(lambda: scheduler.snapshot().running_jobs == 3)
    assert scheduler.snapshot().queued_jobs == 0
    release.set()
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_limited_scheduler_runs_fifo_and_reports_pending_jobs() -> None:
    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=2)
    releases = [threading.Event() for _ in range(3)]
    started: list[int] = []
    lock = threading.Lock()

    def run(index: int) -> None:
        with lock:
            started.append(index)
        releases[index].wait(timeout=2.0)

    for index in range(3):
        scheduler.submit(partial(run, index))

    _wait_until(lambda: started == [0])
    assert scheduler.snapshot().queued_jobs == 2

    releases[0].set()
    _wait_until(lambda: started == [0, 1])
    releases[1].set()
    _wait_until(lambda: started == [0, 1, 2])
    releases[2].set()

    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_bounded_queue_rejects_before_reserved_upload_work_starts() -> None:
    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=1)
    first = scheduler.reserve()
    second = scheduler.reserve()

    with pytest.raises(JobQueueFullError, match="waiting limit: 1"):
        scheduler.reserve()

    first.cancel()
    replacement = scheduler.reserve()
    second.cancel()
    replacement.cancel()
    assert scheduler.shutdown(wait=True, timeout=1.0)


def test_zero_queue_limit_means_unbounded_waiting_when_running_is_limited() -> None:
    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=0)
    release = threading.Event()

    scheduler.submit(lambda: release.wait(timeout=2.0))
    for _ in range(5):
        scheduler.submit(lambda: None)

    _wait_until(lambda: scheduler.snapshot().queued_jobs == 5)
    release.set()
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_handle_reports_fifo_position_and_cancels_exact_waiter() -> None:
    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=3)
    release = threading.Event()
    ran: list[str] = []
    cancelled: list[str] = []
    scheduler.submit(lambda: release.wait(timeout=2.0))
    first = scheduler.submit(lambda: ran.append("first"))
    middle = scheduler.submit(
        lambda: ran.append("middle"),
        on_cancel=cancelled.append,
    )
    last = scheduler.submit(lambda: ran.append("last"))
    _wait_until(lambda: scheduler.snapshot().queued_jobs == 3)

    assert first.queue_position() == 1
    assert middle.queue_position() == 2
    assert last.queue_position() == 3
    assert middle.cancel() is True
    assert middle.cancel() is False
    assert middle.queue_position() is None
    assert first.queue_position() == 1
    assert last.queue_position() == 2
    assert len(cancelled) == 1

    release.set()
    _wait_until(lambda: ran == ["first", "last"])
    assert scheduler.shutdown(wait=True, timeout=2.0)
    assert ran == ["first", "last"]


def test_handle_cannot_claim_a_running_thread_was_cancelled() -> None:
    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=1)
    started = threading.Event()
    release = threading.Event()

    def run() -> None:
        started.set()
        release.wait(timeout=2.0)

    handle = scheduler.submit(run)
    assert started.wait(timeout=1.0)

    assert handle.queue_position() is None
    assert handle.cancel() is False
    assert scheduler.snapshot().running_jobs == 1
    release.set()
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_cancelled_waiter_immediately_releases_bounded_queue_capacity() -> None:
    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=1)
    release = threading.Event()
    scheduler.submit(lambda: release.wait(timeout=2.0))
    waiter = scheduler.submit(lambda: None)
    _wait_until(lambda: scheduler.snapshot().queued_jobs == 1)
    with pytest.raises(JobQueueFullError):
        scheduler.reserve()

    assert waiter.cancel() is True
    replacement = scheduler.reserve()
    replacement.cancel()
    release.set()
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_reserved_submission_returns_a_cancellable_handle() -> None:
    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=1)
    release = threading.Event()
    scheduler.submit(lambda: release.wait(timeout=2.0))
    reservation = scheduler.reserve()
    handle = reservation.submit(lambda: None)
    _wait_until(lambda: scheduler.snapshot().queued_jobs == 1)

    assert handle.queue_position() == 1
    assert handle.cancel() is True
    release.set()
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_shutdown_cancels_pending_and_rejects_new_work() -> None:
    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=2)
    release = threading.Event()
    cancelled: list[str] = []

    scheduler.submit(lambda: release.wait(timeout=2.0))
    scheduler.submit(lambda: None, on_cancel=cancelled.append)
    scheduler.submit(lambda: None, on_cancel=cancelled.append)
    _wait_until(lambda: scheduler.snapshot().queued_jobs == 2)

    assert not scheduler.shutdown(wait=False)
    _wait_until(lambda: len(cancelled) == 2)
    assert len(cancelled) == 2
    assert all("尚未开始" in reason for reason in cancelled)
    with pytest.raises(JobSchedulerClosedError):
        scheduler.submit(lambda: None)

    release.set()
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_shutdown_timeout_includes_slow_pending_cancellation() -> None:
    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=2)
    release = threading.Event()

    def slow_cancel(_reason: str) -> None:
        time.sleep(0.15)

    scheduler.submit(lambda: release.wait(timeout=2.0))
    scheduler.submit(lambda: None, on_cancel=slow_cancel)
    scheduler.submit(lambda: None, on_cancel=slow_cancel)
    _wait_until(lambda: scheduler.snapshot().queued_jobs == 2)

    started = time.monotonic()
    assert not scheduler.shutdown(wait=True, timeout=0.05)
    assert time.monotonic() - started < 0.2

    release.set()
    assert scheduler.shutdown(wait=True, timeout=1.0)


def test_repeated_thread_start_failures_drain_without_recursion(
    monkeypatch,
) -> None:
    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=0)
    release = threading.Event()
    cancelled: list[str] = []
    scheduler.submit(lambda: release.wait(timeout=2.0))
    for _ in range(1_200):
        scheduler.submit(lambda: None, on_cancel=cancelled.append)
    _wait_until(lambda: scheduler.snapshot().queued_jobs == 1_200)

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("simulated thread start failure")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    release.set()

    _wait_until(
        lambda: scheduler.snapshot().running_jobs == 0 and scheduler.snapshot().queued_jobs == 0,
    )
    assert len(cancelled) == 1_200
    assert scheduler.shutdown(wait=True, timeout=1.0)


def test_thread_start_failure_cancellation_is_part_of_shutdown(
    monkeypatch,
) -> None:
    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=0)
    release_running = threading.Event()
    cancel_started = threading.Event()
    release_cancel = threading.Event()

    scheduler.submit(lambda: release_running.wait(timeout=2.0))

    def slow_cancel(_reason: str) -> None:
        cancel_started.set()
        release_cancel.wait(timeout=2.0)

    scheduler.submit(lambda: None, on_cancel=slow_cancel)
    _wait_until(lambda: scheduler.snapshot().queued_jobs == 1)

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("simulated promoted-thread start failure")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    release_running.set()
    assert cancel_started.wait(timeout=1.0)

    assert not scheduler.shutdown(wait=True, timeout=0.05)
    assert scheduler.snapshot().cancelling_jobs == 1

    release_cancel.set()
    assert scheduler.shutdown(wait=True, timeout=1.0)


def test_reconfigure_higher_running_limit_promotes_fifo_immediately() -> None:
    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=3)
    releases = [threading.Event() for _ in range(3)]
    started: list[int] = []
    lock = threading.Lock()

    def run(index: int) -> None:
        with lock:
            started.append(index)
        releases[index].wait(timeout=2.0)

    for index in range(3):
        scheduler.submit(partial(run, index))
    _wait_until(lambda: started == [0])

    snapshot = scheduler.reconfigure(max_running_jobs=2, max_queued_jobs=3)

    _wait_until(lambda: started == [0, 1])
    assert snapshot.max_running_jobs == 2
    assert scheduler.snapshot().queued_jobs == 1
    for release in releases:
        release.set()
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_reconfigure_lower_limit_grandfathers_running_jobs() -> None:
    scheduler = JobScheduler(max_running_jobs=3, max_queued_jobs=2)
    releases = [threading.Event() for _ in range(4)]
    started: list[int] = []
    lock = threading.Lock()

    def run(index: int) -> None:
        with lock:
            started.append(index)
        releases[index].wait(timeout=2.0)

    for index in range(4):
        scheduler.submit(partial(run, index))
    _wait_until(lambda: started == [0, 1, 2])

    scheduler.reconfigure(max_running_jobs=1, max_queued_jobs=2)
    releases[0].set()
    _wait_until(lambda: scheduler.snapshot().running_jobs == 2)
    assert started == [0, 1, 2]
    releases[1].set()
    _wait_until(lambda: scheduler.snapshot().running_jobs == 1)
    assert started == [0, 1, 2]
    releases[2].set()
    _wait_until(lambda: started == [0, 1, 2, 3])
    releases[3].set()
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_reconfigure_unlimited_starts_every_waiting_job() -> None:
    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=3)
    release = threading.Event()
    started = 0
    lock = threading.Lock()

    def run() -> None:
        nonlocal started
        with lock:
            started += 1
        release.wait(timeout=2.0)

    for _ in range(4):
        scheduler.submit(run)
    _wait_until(lambda: started == 1)

    scheduler.reconfigure(max_running_jobs=0, max_queued_jobs=1)

    _wait_until(lambda: started == 4)
    assert scheduler.snapshot().queued_jobs == 0
    release.set()
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_reconfigure_smaller_queue_keeps_admitted_waiters() -> None:
    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=3)
    release = threading.Event()
    scheduler.submit(lambda: release.wait(timeout=2.0))
    for _ in range(3):
        scheduler.submit(lambda: None)
    _wait_until(lambda: scheduler.snapshot().queued_jobs == 3)

    scheduler.reconfigure(max_running_jobs=1, max_queued_jobs=1)

    assert scheduler.snapshot().queued_jobs == 3
    with pytest.raises(JobQueueFullError, match="waiting limit: 1"):
        scheduler.reserve()
    release.set()
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_reconfigure_does_not_start_a_promoted_call_after_shutdown(
    monkeypatch,
) -> None:
    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=1)
    release_running = threading.Event()
    promotion_claimed = threading.Event()
    allow_start = threading.Event()
    promoted_ran = threading.Event()
    promoted_cancelled = threading.Event()

    scheduler.submit(lambda: release_running.wait(timeout=2.0))
    scheduler.submit(
        promoted_ran.set,
        on_cancel=lambda _reason: promoted_cancelled.set(),
    )
    _wait_until(lambda: scheduler.snapshot().queued_jobs == 1)

    original_start = scheduler._start

    def gated_start(call, *, propagate_failure: bool) -> None:
        promotion_claimed.set()
        allow_start.wait(timeout=2.0)
        original_start(call, propagate_failure=propagate_failure)

    monkeypatch.setattr(scheduler, "_start", gated_start)
    reconfigure_thread = threading.Thread(
        target=lambda: scheduler.reconfigure(
            max_running_jobs=2,
            max_queued_jobs=1,
        ),
    )
    reconfigure_thread.start()
    assert promotion_claimed.wait(timeout=1.0)

    assert not scheduler.shutdown(wait=False)
    allow_start.set()
    reconfigure_thread.join(timeout=1.0)

    assert promoted_cancelled.wait(timeout=1.0)
    assert not promoted_ran.is_set()
    release_running.set()
    assert scheduler.shutdown(wait=True, timeout=2.0)
