"""Exactly-once cleanup for one Web application runtime."""

from __future__ import annotations

import atexit
import logging
import threading
from contextlib import asynccontextmanager

from hhtools.application.state import SessionState, _cleanup_session_state

_log = logging.getLogger(__name__)


class RuntimeLifecycle:
    """Drain jobs, release the Agent lease, and remove only temporary roots."""

    def __init__(self, state: SessionState, scheduler, job_settings_lock, agent_lease) -> None:
        self.state = state
        self.scheduler = scheduler
        self.job_settings_lock = job_settings_lock
        self.agent_lease = agent_lease
        self._cleanup_lock = threading.Lock()
        self._cleanup_complete = False
        atexit.register(self._cleanup_once)

    def _cleanup_once(self) -> None:
        with self._cleanup_lock:
            if self._cleanup_complete:
                return
            self._cleanup_complete = True
        _cleanup_session_state(self.state)
        atexit.unregister(self._cleanup_once)

    def _deferred_cleanup(self) -> None:
        self.scheduler.shutdown(wait=True)
        try:
            self._cleanup_once()
        finally:
            self.agent_lease.release()

    @asynccontextmanager
    async def lifespan(self, _app):  # type: ignore[no-untyped-def]
        try:
            yield
        finally:
            # A saved admission setting and its live scheduler update form one
            # operation, so shutdown shares their lock.
            with self.job_settings_lock:
                drained = self.scheduler.shutdown(wait=True, timeout=5.0)
            if drained:
                try:
                    self._cleanup_once()
                finally:
                    self.agent_lease.release()
            else:
                # Python cannot force-cancel a running thread. Keep its
                # temporary inputs alive until the worker drains, then release
                # ownership.
                _log.warning("Web shutdown timed out with active jobs; deferring session cleanup")
                threading.Thread(
                    target=self._deferred_cleanup,
                    name="hhtools-web-deferred-cleanup",
                    daemon=True,
                ).start()
