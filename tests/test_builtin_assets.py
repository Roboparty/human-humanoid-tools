from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_builtin_assets import (
    DEFAULT_MANIFEST,
    EXPECTED_MOTION_ENTRIES,
    EXPECTED_MOTION_FILES,
    EXPECTED_ROBOTS,
    BuiltinAssetManifestError,
    motion_packaging_paths,
    validate_manifest,
)


def test_builtin_asset_manifest_is_complete_and_matches_baseline() -> None:
    summary = validate_manifest()

    assert summary == {
        "motion_entries": EXPECTED_MOTION_ENTRIES,
        "motion_files": EXPECTED_MOTION_FILES,
        "robots": EXPECTED_ROBOTS,
    }


def test_builtin_motion_allowlist_excludes_generated_and_user_assets() -> None:
    payload = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    paths = set(motion_packaging_paths(payload))

    assert "assets/motions/mimic/GVHMR/hmr4d_results.pt" not in paths
    assert not any(".hhtools_analysis" in path for path in paths)
    assert not any("Robot-Unitree-G1" in path or "Robot-AgiBot-X2" in path for path in paths)
    assert payload["baseline_exclusions"] == [
        {
            "path": "assets/motions/mimic/GVHMR/hmr4d_results.pt",
            "reason": (
                "Generated GVHMR output with a known motion-loading problem; "
                "not a stable built-in sample."
            ),
        }
    ]


def test_builtin_asset_manifest_rejects_unreviewed_baseline_paths(tmp_path: Path, monkeypatch) -> None:
    payload = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    # Source archives and shallow clones may have no historical Git object.
    # Exercise the comparison itself with an explicit baseline in every runner.
    baseline = set(motion_packaging_paths(payload)) | {
        "assets/motions/mimic/GVHMR/hmr4d_results.pt",
    }
    monkeypatch.setattr("scripts.verify_builtin_assets._git_paths", lambda *_: baseline)
    payload["baseline_exclusions"] = []
    changed = tmp_path / "builtin-assets.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BuiltinAssetManifestError, match="reviewed motion paths differ"):
        validate_manifest(changed)


@pytest.mark.parametrize("field", ["repository", "commit", "main_urdf", "license_paths"])
def test_builtin_manifest_rejects_unreviewed_rp1_source(tmp_path: Path, field: str) -> None:
    payload = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    robot = next(entry for entry in payload["robots"] if entry["preset"] == "rp1")
    robot[field] = ["LICENSE"] if field == "license_paths" else "unreviewed-source"
    changed = tmp_path / "builtin-assets.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BuiltinAssetManifestError, match=rf"rp1\.{field} differs"):
        validate_manifest(changed)
