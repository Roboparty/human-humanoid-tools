from __future__ import annotations

from pathlib import Path

import pytest

from hhtools.web.jobs.job_specs import (
    JobSpecError,
    build_job_spec,
    normalize_job_spec,
    replay_capability,
)


def test_build_job_spec_removes_session_only_tokens() -> None:
    spec = build_job_spec(
        "retarget",
        {
            "motion_token": "expired",
            "source_path": "/motions/walk.npz",
            "robot": "unitree_g1",
            "nested": {"export_token": "temporary", "keep": True},
        },
    )

    assert spec == {
        "schema_version": 1,
        "kind": "retarget",
        "request": {
            "source_path": "/motions/walk.npz",
            "robot": "unitree_g1",
            "nested": {"keep": True},
        },
    }


def test_normalize_accepts_downloaded_job_config() -> None:
    nested = {
        "schema_version": 1,
        "kind": "batch",
        "request": {"robot": "unitree_g1", "entries": []},
    }

    assert normalize_job_spec({"job_id": "old", "spec": nested}) == nested


def test_normalize_rejects_unknown_schema() -> None:
    with pytest.raises(JobSpecError, match="不支持 JobSpec v2"):
        normalize_job_spec({"schema_version": 2, "kind": "retarget", "request": {}})


def test_replay_capability_checks_sources_and_ephemeral_root(tmp_path: Path) -> None:
    source = tmp_path / "library" / "walk.npz"
    source.parent.mkdir()
    source.write_bytes(b"motion")
    spec = build_job_spec(
        "retarget",
        {"robot": "unitree_g1", "source_path": str(source)},
    )

    assert replay_capability(spec, ephemeral_root=tmp_path / "uploads") == {
        "available": True,
        "reason": None,
        "source_count": 1,
    }
    blocked = replay_capability(spec, ephemeral_root=source.parent)
    assert blocked["available"] is False
    assert "临时上传目录" in blocked["reason"]


def test_replay_capability_explains_session_only_job() -> None:
    capability = replay_capability(build_job_spec("r2r_retarget", {"source_token": "gone"}))

    assert capability["available"] is False
    assert "会话内对象" in capability["reason"]
