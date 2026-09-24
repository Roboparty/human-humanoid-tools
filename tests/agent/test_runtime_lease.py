from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from hhtools.contracts import ErrorStage
from hhtools.services.runtime_lease import AgentRuntimeLease, RuntimeLeaseError


def _assert_sanitized_contention(error: RuntimeLeaseError, data_dir: Path) -> None:
    assert error.code == "RUNTIME_ALREADY_ACTIVE"
    assert error.api_error.stage is ErrorStage.ADMISSION
    assert error.api_error.retryable is True
    assert str(data_dir) not in str(error)
    assert str(data_dir) not in error.api_error.model_dump_json()


def test_same_process_handles_contend_and_release_is_reacquirable(tmp_path: Path) -> None:
    data_dir = tmp_path / "agent-data"
    first = AgentRuntimeLease.acquire(data_dir)
    assert first.held is True

    with pytest.raises(RuntimeLeaseError) as captured:
        AgentRuntimeLease.acquire(data_dir)
    _assert_sanitized_contention(captured.value, data_dir)

    first.release()
    first.release()
    assert first.held is False

    with AgentRuntimeLease.acquire(data_dir) as second:
        assert second.held is True
    assert second.held is False


def test_child_process_crash_releases_runtime_lease(tmp_path: Path) -> None:
    data_dir = tmp_path / "agent-data"
    ready_file = tmp_path / "child-ready"
    child_program = """
import sys
import time
from pathlib import Path

from hhtools.services.runtime_lease import AgentRuntimeLease

lease = AgentRuntimeLease.acquire(Path(sys.argv[1]))
Path(sys.argv[2]).write_text("ready", encoding="utf-8")
while True:
    time.sleep(1)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", child_program, str(data_dir), str(ready_file)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready_deadline = time.monotonic() + 10.0
        while not ready_file.is_file():
            if process.poll() is not None:
                assert process.stderr is not None
                raise AssertionError(
                    f"lease child exited before readiness: {process.stderr.read()}"
                )
            if time.monotonic() >= ready_deadline:
                raise AssertionError("lease child did not become ready")
            time.sleep(0.05)

        with pytest.raises(RuntimeLeaseError) as captured:
            AgentRuntimeLease.acquire(data_dir)
        _assert_sanitized_contention(captured.value, data_dir)

        process.kill()
        process.wait(timeout=10)

        deadline = time.monotonic() + 5.0
        while True:
            try:
                recovered = AgentRuntimeLease.acquire(data_dir)
                break
            except RuntimeLeaseError as error:
                if error.code != "RUNTIME_ALREADY_ACTIVE" or time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        recovered.release()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        if process.stderr is not None:
            process.stderr.close()
