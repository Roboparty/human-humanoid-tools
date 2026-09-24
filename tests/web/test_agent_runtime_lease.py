"""Composition-root coverage for exclusive Agent runtime ownership."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hhtools.services import RuntimeLeaseError
from hhtools.web import server
from hhtools.web.server import state as server_state


def _app(tmp_path: Path, *, cache_name: str):
    return server.create_app(
        source_root=tmp_path / "motions",
        save_dir=tmp_path / "save",
        cache_dir=tmp_path / cache_name,
        job_history_dir=tmp_path / f"history-{cache_name}",
    )


def test_create_app_holds_one_runtime_until_its_workers_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "HHTOOLS_MOTION_LIBRARY_ROOT",
        str(tmp_path / "motion-library"),
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(server_state, "_robot_library_root", lambda: tmp_path / "robots")

    first = _app(tmp_path, cache_name="cache-first")
    assert first.state.agent_runtime_lease.held is True

    with pytest.raises(RuntimeLeaseError) as captured:
        _app(tmp_path, cache_name="cache-contender")
    assert captured.value.code == "RUNTIME_ALREADY_ACTIVE"

    # TestClient drives the same lifespan used by Web, Electron, and the
    # in-process MCP composition bridge.  The lease is released only after the
    # scheduler has drained.
    with TestClient(first):
        assert first.state.agent_runtime_lease.held is True
    assert first.state.agent_runtime_lease.held is False

    second = _app(tmp_path, cache_name="cache-second")
    with TestClient(second):
        assert second.state.agent_runtime_lease.held is True
    assert second.state.agent_runtime_lease.held is False
