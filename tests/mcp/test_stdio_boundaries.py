"""Subprocess regressions for stdout framing and startup diagnostics."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hhtools.services.runtime_lease import AgentRuntimeLease

_REPOSITORY_ROOT = Path(__file__).parents[2]
_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2026-07-28",
        "capabilities": {},
        "clientInfo": {"name": "stdio-boundary-test", "version": "1"},
    },
}


def _wire_input(*extra_messages: dict[str, object]) -> str:
    messages = [
        _INITIALIZE,
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        *extra_messages,
    ]
    return "".join(json.dumps(message, separators=(",", ":")) + "\n" for message in messages)


@pytest.mark.skipif(importlib.util.find_spec("warp") is None, reason="Warp is optional")
def test_first_warp_initialization_never_writes_into_stdio_frames() -> None:
    fixture = Path(__file__).with_name("stdio_warp_fixture_server.py")
    result = subprocess.run(
        [sys.executable, str(fixture)],
        input=_wire_input({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        text=True,
        capture_output=True,
        timeout=120,
        cwd=_REPOSITORY_ROOT,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    # Validate every physical stdout line.  A tolerant MCP client can recover
    # after a prose line, so a successful client round-trip alone is not proof
    # that the stdio wire stayed clean.
    documents = [json.loads(line) for line in result.stdout.splitlines()]
    # Lifespan (and therefore the real ``wp.init()`` above) completes before
    # the initialize response is emitted.  Do not require the pipelined
    # ``tools/list`` response: stdin reaches EOF immediately after the request,
    # so the stdio transport may close cleanly before dispatching that second
    # frame.  The boundary under test is that every frame which *is* emitted is
    # JSON and that first Warp initialization never inserts a prose line.
    assert 1 in {document.get("id") for document in documents}
    assert "Warp " not in result.stdout


def test_runtime_lease_conflict_has_one_sanitized_actionable_stderr_line(
    tmp_path: Path,
) -> None:
    fixture = Path(__file__).with_name("stdio_lease_fixture_server.py")
    save_dir = tmp_path / "private-save-directory"
    lease = AgentRuntimeLease.acquire(save_dir / ".hhtools-agent")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(fixture),
                "--save-dir",
                str(save_dir),
            ],
            input=_wire_input(),
            text=True,
            capture_output=True,
            timeout=30,
            cwd=_REPOSITORY_ROOT,
            check=False,
        )
    finally:
        lease.release()

    assert result.returncode == 3
    assert result.stdout == ""
    lines = result.stderr.splitlines()
    assert len(lines) == 1
    assert "RUNTIME_ALREADY_ACTIVE" in lines[0]
    assert "Close the existing WebUI" in lines[0]
    assert "same --save-dir" in lines[0]
    assert str(save_dir) not in result.stderr
    assert "Traceback" not in result.stderr
    assert "ExceptionGroup" not in result.stderr
