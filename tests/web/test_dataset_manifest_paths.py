from __future__ import annotations

import json
from pathlib import Path

from hhtools.web.analysis.dataset_analysis import export_manifest, resolve_manifest_source_path


def test_manifest_uses_analysis_relative_paths_without_exposing_host_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    clip = root / "subject" / "walk.bvh"
    clip.parent.mkdir(parents=True)
    clip.write_text("HIERARCHY\n", encoding="utf-8")

    relative = resolve_manifest_source_path(clip, analyze_source=root)
    document = json.loads(
        export_manifest(
            [
                {
                    "clip_id": "subject/walk",
                    "source_path": str(clip),
                    "folder_label": "subject",
                }
            ],
            ["subject/walk"],
            analyze_source=str(root),
        )
    )

    assert relative == "subject/walk.bvh"
    assert document["meta"] == {"path_basis": "analysis_relative"}
    assert document["clips"][0]["source_path"] == relative
    assert str(tmp_path) not in json.dumps(document)
