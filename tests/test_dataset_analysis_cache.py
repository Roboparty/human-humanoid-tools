import json
import os
from pathlib import Path

import pytest

from hhtools.analysis import dataset_analysis as analysis
from hhtools.analysis.config import load_config


def _fixture(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    clip = root / "walk.bvh"
    clip.write_text("original")
    entries = [{"source_path": str(clip), "clip_id": "walk", "dataset": "lafan"}]
    return root, clip, entries


def _cache(root, entries, cfg=None):
    data = {
        "meta": {
            "fingerprint": analysis._fingerprint(entries, source_root=root, cfg=cfg),
            "metric_schema": analysis._METRIC_CACHE_SCHEMA,
        },
        "clips": entries,
    }
    analysis._manifest_path(root, root, "handcrafted").write_text(json.dumps(data))
    return data


@pytest.mark.parametrize(
    "mutation", ["rename", "same_second", "preserved_mtime", "replace", "sidecar", "delete"]
)
def test_analysis_cache_invalidates_changed_inputs(tmp_path: Path, mutation: str):
    root, clip, entries = _fixture(tmp_path)
    _cache(root, entries)
    assert analysis.load_cached(root, root, "handcrafted", entries) is not None
    before = clip.stat()
    if mutation == "rename":
        changed = clip.with_name("renamed.bvh")
        clip.rename(changed)
        entries[0]["source_path"] = str(changed)
    elif mutation == "delete":
        clip.unlink()
    elif mutation == "sidecar":
        (root / "scene.json").write_text('{"height":2}')
    elif mutation == "replace":
        replacement = root / "replacement"
        replacement.write_text("modified")
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(clip)
    else:
        clip.write_text("modified")
        mtime = before.st_mtime_ns + 1 if mutation == "same_second" else before.st_mtime_ns
        os.utime(clip, ns=(before.st_atime_ns, mtime))
    assert analysis.load_cached(root, root, "handcrafted", entries) is None


def test_fingerprint_is_order_independent_and_ignores_generated_cache(tmp_path: Path):
    root, _, entries = _fixture(tmp_path)
    second = root / "run.bvh"
    second.write_text("run")
    entries.append({"source_path": str(second), "clip_id": "run"})
    before = analysis._fingerprint(entries, source_root=root)
    _cache(root, entries)
    assert analysis._fingerprint(list(reversed(entries)), source_root=root) == before


def test_content_change_invalidates_even_when_all_file_metadata_is_preserved(
    tmp_path: Path, monkeypatch
):
    root, clip, entries = _fixture(tmp_path)
    snapshot = clip.stat()
    original_stat = Path.stat
    monkeypatch.setattr(
        Path,
        "stat",
        lambda path, *args, **kwargs: (
            snapshot if path == clip else original_stat(path, *args, **kwargs)
        ),
    )
    _cache(root, entries)
    clip.write_text("modified")
    assert analysis.load_cached(root, root, "handcrafted", entries) is None


def test_effective_config_and_algorithm_changes_invalidate_cache(tmp_path: Path, monkeypatch):
    root, _, entries = _fixture(tmp_path)
    _cache(root, entries)
    changed = load_config({"thresholds": {"contact_height_m": 9}})
    assert analysis.load_cached(root, root, "handcrafted", entries, cfg=changed) is None
    monkeypatch.setattr(analysis, "_METRIC_CACHE_SCHEMA", 999)
    assert analysis.load_cached(root, root, "handcrafted", entries) is None


def test_default_config_file_changes_are_seen_without_restart(tmp_path: Path, monkeypatch):
    from hhtools.analysis import config

    root, _, entries = _fixture(tmp_path)
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text("thresholds:\n  contact_height_m: 0.05\n")
    monkeypatch.setattr(config, "_config_path", lambda: defaults)
    _cache(root, entries)
    assert analysis.load_cached(root, root, "handcrafted", entries) is not None
    defaults.write_text("thresholds:\n  contact_height_m: 0.5\n")
    assert analysis.load_cached(root, root, "handcrafted", entries) is None


def test_run_analysis_uses_effective_config_before_cache_lookup(tmp_path: Path, monkeypatch):
    from hhtools.analysis import collection

    root, _, entries = _fixture(tmp_path)
    calls = []
    monkeypatch.setattr(analysis, "build_entries", lambda _: entries)
    monkeypatch.setattr(
        collection, "analyze_entries", lambda *args, **kwargs: calls.append(kwargs["cfg"]) or []
    )
    monkeypatch.setattr(collection, "build_summary", lambda *args: {})
    analysis.run_analysis(root, root)
    analysis.run_analysis(root, root)
    assert len(calls) == 1
    analysis.run_analysis(root, root, cfg_override={"thresholds": {"contact_height_m": 9}})
    assert len(calls) == 2


def test_inputs_changed_during_analysis_are_not_cached(tmp_path: Path, monkeypatch):
    from hhtools.analysis import collection

    root, clip, entries = _fixture(tmp_path)
    monkeypatch.setattr(analysis, "build_entries", lambda _: entries)

    def mutate(*args, **kwargs):
        clip.write_text("changed during analysis")
        return []

    monkeypatch.setattr(collection, "analyze_entries", mutate)
    monkeypatch.setattr(collection, "build_summary", lambda *args: {})
    result = analysis.run_analysis(root, root)
    assert result["meta"]["fingerprint"] is None
    assert analysis.load_cached(root, root, "handcrafted", entries) is None
