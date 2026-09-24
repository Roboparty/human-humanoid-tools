"""Cross-process ownership lease for one Agent data directory.

The job scheduler and interrupted-job recovery are process-local.  Two live
runtimes must therefore never construct independent managers over the same
data directory.  This module provides the small ownership primitive used by
composition roots before they create any stores or recover active jobs.

The lock file is deliberately retained after release.  Removing an advisory
lock file creates an inode/handle race in which two processes can each lock a
different file with the same name.  Only the operating-system lock is the
lease; closing the descriptor, including at process termination, releases it.
"""

from __future__ import annotations

import errno
import importlib
import os
import threading
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Self

from hhtools.contracts import ApiError, ErrorStage

_LOCK_FILENAME = ".agent-runtime.lock"
_CONTENTION_ERRNOS = frozenset({errno.EACCES, errno.EAGAIN, errno.EDEADLK})
_WINDOWS_CONTENTION_ERRORS = frozenset({32, 33})


class RuntimeLeaseError(RuntimeError):
    """Expected, transport-neutral failure to acquire runtime ownership."""

    def __init__(self, error: ApiError) -> None:
        self.error = error
        super().__init__(f"{error.code}: {error.message}")

    @property
    def api_error(self) -> ApiError:
        """Return the versioned public error without exposing the lock path."""

        return self.error

    @property
    def code(self) -> str:
        """Return the stable machine code without requiring message parsing."""

        return self.error.code


def _error(code: str, message: str, *, retryable: bool) -> RuntimeLeaseError:
    return RuntimeLeaseError(
        ApiError(
            code=code,
            message=message,
            retryable=retryable,
            stage=ErrorStage.ADMISSION,
        )
    )


def _already_active_error() -> RuntimeLeaseError:
    return _error(
        "RUNTIME_ALREADY_ACTIVE",
        "Another HHTools runtime already owns the Agent data directory.",
        retryable=True,
    )


def _unavailable_error() -> RuntimeLeaseError:
    return _error(
        "RUNTIME_LEASE_UNAVAILABLE",
        "HHTools could not establish exclusive runtime ownership.",
        retryable=False,
    )


def _is_contention(error: OSError) -> bool:
    return error.errno in _CONTENTION_ERRNOS or getattr(error, "winerror", None) in (
        _WINDOWS_CONTENTION_ERRORS
    )


def _fcntl_module() -> Any:
    """Load the POSIX-only module without applying Windows stub types."""

    return importlib.import_module("fcntl")


def _lock(stream: BinaryIO) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        return

    fcntl = _fcntl_module()
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(stream: BinaryIO) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return

    fcntl = _fcntl_module()
    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _open_lock_file(data_dir: Path) -> BinaryIO:
    data_dir.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOINHERIT", 0)
    # Refuse a pre-existing symlink where the platform can enforce this in the
    # same system call.  The file never contains host paths or process data.
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(data_dir / _LOCK_FILENAME, flags, 0o600)
    try:
        return os.fdopen(descriptor, "r+b", buffering=0)
    except Exception:
        os.close(descriptor)
        raise


class AgentRuntimeLease:
    """One held advisory lease over an Agent data directory.

    Acquire this before constructing ``JobManager`` so a competing process
    cannot run interrupted-job recovery or create a second GPU scheduler.
    Instances are context managers and ``release`` is idempotent.
    """

    def __init__(self, stream: BinaryIO) -> None:
        self._stream: BinaryIO | None = stream
        self._state_lock = threading.Lock()

    @classmethod
    def acquire(cls, data_dir: str | os.PathLike[str]) -> Self:
        """Acquire exclusive ownership or raise a sanitized service error."""

        try:
            stream = _open_lock_file(Path(data_dir))
        except OSError as error:
            raise _unavailable_error() from error

        try:
            _lock(stream)
        except OSError as error:
            try:
                stream.close()
            except OSError:
                pass
            if _is_contention(error):
                raise _already_active_error() from error
            raise _unavailable_error() from error
        return cls(stream)

    @property
    def held(self) -> bool:
        """Whether this object still holds its operating-system descriptor."""

        with self._state_lock:
            return self._stream is not None

    def release(self) -> None:
        """Release ownership; repeated calls are safe."""

        with self._state_lock:
            stream = self._stream
            self._stream = None
        if stream is None:
            return
        try:
            _unlock(stream)
        except OSError:
            # Closing the descriptor is the authoritative OS release path.
            # An unlock failure must not leave a live handle behind.
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


__all__ = ["AgentRuntimeLease", "RuntimeLeaseError"]
