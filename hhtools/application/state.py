"""Process-local state owned by the Web application runtime."""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def _tmpdir(tag: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"hhtools_web_{tag}_"))


def _robot_library_root() -> Path:
    """Persistent per-user robot library (survives ``hhtools web`` restarts)."""
    from hhtools.utils.paths import user_robot_dir

    return user_robot_dir()


def _default_robot_library_root() -> Path:
    """Resolve the user root when each session is constructed."""

    return _robot_library_root()


@dataclass
class SessionState:
    """In-memory state for the single active browser session."""

    source_root: Path
    save_dir: Path
    cache: Any = None  # EphemeralCache
    job_history: Any = None  # JobHistoryStore
    motions: dict[str, Any] = field(default_factory=dict)  # token -> (Motion, meta)
    robots: dict[str, Any] = field(default_factory=dict)  # name -> URDFRobotModel
    jobs: dict[str, Job] = field(default_factory=dict)
    # robot-to-robot source trajectories: token -> {source_robot, motion, ...}
    r2r_sources: dict[str, Any] = field(default_factory=dict)
    # dataset viz robot preview: token -> {clip_dir, source_path}
    dataset_previews: dict[str, Any] = field(default_factory=dict)
    basket: list[dict] = field(default_factory=list)  # library entries queued for batch
    upload_root: Path = field(default_factory=lambda: _tmpdir("up"))
    robot_root: Path = field(default_factory=_default_robot_library_root)
    export_root: Path = field(default_factory=lambda: _tmpdir("out"))
    robot_prewarm_threads: dict[str, threading.Thread] = field(default_factory=dict)
    job_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class Job:
    id: str
    kind: str
    request: dict[str, Any] = field(default_factory=dict)
    status: str = "running"  # pending | running | done | error
    progress: float = 0.0  # overall job progress (batch: all clips)
    clip_progress: float = 0.0  # batch only: current clip / GPU-chunk progress
    message: str = ""
    result: dict | None = None
    error: str | None = None
    parent_job_id: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    created_wall_time: float = field(default_factory=time.time)
    finished_wall_time: float | None = None
    terminal_since: float | None = None
    last_accessed_at: float = field(default_factory=time.monotonic)
    on_terminal: Callable[[Job, str], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def mark_running(self) -> None:
        """Publish the transition from the FIFO queue into an execution slot."""

        self.status = "running"

    def _publish_terminal(self, status: str) -> None:
        self.terminal_since = time.monotonic()
        self.finished_wall_time = time.time()
        self.status = status

    def mark_terminal(self, status: str) -> None:
        """Let the runtime finalize durable results before publishing terminal status."""
        if self.on_terminal is not None:
            try:
                self.on_terminal(self, status)
            except Exception:  # noqa: BLE001 - history must not fail the actual job
                _log.exception("failed to persist terminal Web job %s", self.id)
        if self.status != status:
            self._publish_terminal(status)


def _snapshot_job_request(value: Any) -> Any:
    """Copy JSON-like request data so later UI mutations cannot alter job history."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _snapshot_job_request(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot_job_request(item) for item in value]
    return str(value)


def _cleanup_session_state(state: SessionState) -> None:
    """Release resources owned by one application runtime without touching user data."""
    try:
        if state.cache is not None:
            state.cache.cleanup()
    except Exception:  # noqa: BLE001 - shutdown cleanup must remain best-effort
        _log.warning("failed to clean the web motion cache", exc_info=True)

    # Robot presets and save_dir are persistent user data. Only roots minted by
    # SessionState are safe to remove when this server instance shuts down.
    for root in (state.upload_root, state.export_root):
        shutil.rmtree(root, ignore_errors=True)

    with state.job_lock:
        state.jobs.clear()
    state.motions.clear()
    state.r2r_sources.clear()
    state.dataset_previews.clear()
    state.basket.clear()
    state.robot_prewarm_threads.clear()
