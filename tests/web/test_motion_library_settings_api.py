from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from hhtools.web import server
from hhtools.web.library.motion_library_settings import (
    MOTION_LIBRARY_MARKER_FILENAME,
    MotionLibrarySettings,
    motion_library_marker_path,
    validate_motion_library_marker,
)
from hhtools.web.server import state as server_state


def _create_app(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HHTOOLS_MOTION_LIBRARY_ROOT", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv(
        "HHTOOLS_MOTION_LIBRARY_SETTINGS_PATH",
        str(tmp_path / "motion-library-settings.json"),
    )
    monkeypatch.setattr(server_state, "_robot_library_root", lambda: tmp_path / "robots")
    return server.create_app(
        source_root=tmp_path / "assets" / "motions",
        save_dir=tmp_path / "save",
        cache_dir=tmp_path / "cache",
    )


def _local_client(app) -> TestClient:
    return TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    )


def test_motion_library_root_switch_persists_and_applies_without_restart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_app(tmp_path, monkeypatch)
    selected = tmp_path / "managed-motion-library"
    app_identity = id(app)

    with _local_client(app) as client:
        before = client.get("/api/settings/motion-library")
        changed = client.patch(
            "/api/settings/motion-library",
            json={"root": str(selected)},
        )
        library = client.get("/api/library")

    assert before.status_code == 200
    assert before.json()["editable"] is True
    assert before.json()["readonly_reason"] is None
    assert changed.status_code == 200
    assert changed.json()["root"] == str(selected.resolve())
    assert library.json()["motions_library_root"] == str(selected.resolve())
    assert id(app) == app_identity
    assert (selected / MOTION_LIBRARY_MARKER_FILENAME).is_file()
    persisted = json.loads((tmp_path / "motion-library-settings.json").read_text())
    assert persisted["root"] == str(selected.resolve())


def test_populated_default_library_can_round_trip_through_custom_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_app(tmp_path, monkeypatch)

    with _local_client(app) as client:
        default_root = Path(client.get("/api/settings/motion-library").json()["root"])
        clip = default_root / "existing-dataset" / "walk.bvh"
        clip.parent.mkdir(parents=True)
        clip.write_bytes(b"motion")

        custom_root = tmp_path / "custom-library"
        switched_away = client.patch(
            "/api/settings/motion-library",
            json={"root": str(custom_root)},
        )
        assert validate_motion_library_marker(default_root) is True
        switched_back = client.patch(
            "/api/settings/motion-library",
            json={"root": str(default_root)},
        )

    assert switched_away.status_code == 200
    assert switched_back.status_code == 200
    assert switched_back.json()["root"] == str(default_root.resolve())
    assert clip.read_bytes() == b"motion"
    assert validate_motion_library_marker(default_root) is True


def test_motion_library_root_rejects_populated_unmanaged_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_app(tmp_path, monkeypatch)
    selected = tmp_path / "existing-dataset"
    selected.mkdir()
    (selected / "do-not-touch.bvh").write_bytes(b"motion")

    with _local_client(app) as client:
        before = client.get("/api/settings/motion-library").json()["root"]
        response = client.patch(
            "/api/settings/motion-library",
            json={"root": str(selected)},
        )
        after = client.get("/api/settings/motion-library").json()["root"]

    assert response.status_code == 422
    assert "空" in response.json()["detail"]
    assert before == after
    assert (selected / "do-not-touch.bvh").read_bytes() == b"motion"
    assert not (selected / MOTION_LIBRARY_MARKER_FILENAME).exists()


def test_motion_library_root_rejects_forged_ownership_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_app(tmp_path, monkeypatch)
    selected = tmp_path / "forged-library"
    selected.mkdir()
    clip = selected / "do-not-touch.bvh"
    clip.write_bytes(b"motion")
    motion_library_marker_path(selected).write_text(
        '{"schema_version": 1, "managed_by": "another-tool"}',
        encoding="utf-8",
    )

    with _local_client(app) as client:
        response = client.patch(
            "/api/settings/motion-library",
            json={"root": str(selected)},
        )

    assert response.status_code == 422
    assert "ownership marker" in response.json()["detail"]
    assert clip.read_bytes() == b"motion"


def test_motion_library_root_change_uses_local_admin_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_app(tmp_path, monkeypatch)
    selected = tmp_path / "remote-cannot-select"

    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("192.0.2.10", 50000),
    ) as client:
        settings = client.get("/api/settings/motion-library")
        response = client.patch(
            "/api/settings/motion-library",
            json={"root": str(selected)},
        )

    assert settings.json()["editable"] is False
    assert settings.json()["readonly_reason"] == "remote"
    assert response.status_code == 403
    assert not selected.exists()


def test_environment_override_reports_specific_readonly_reason(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_app(tmp_path, monkeypatch)
    override = tmp_path / "environment-library"
    monkeypatch.setenv("HHTOOLS_MOTION_LIBRARY_ROOT", str(override))

    with _local_client(app) as client:
        settings = client.get("/api/settings/motion-library")
        response = client.patch(
            "/api/settings/motion-library",
            json={"root": str(tmp_path / "ignored")},
        )

    assert settings.status_code == 200
    assert settings.json()["editable"] is False
    assert settings.json()["readonly_reason"] == "environment_override"
    assert settings.json()["root"] == str(override.resolve())
    assert response.status_code == 409


def test_library_api_reads_root_and_entries_from_one_locked_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_app(tmp_path, monkeypatch)
    switched_root = (tmp_path / "switched-during-assets-scan").resolve()
    switched_root.mkdir()
    seen_scan_roots: list[Path | None] = []

    from hhtools.viewer import library as viewer_library
    from hhtools.web.library import motion_library_links

    def switch_root_during_assets_scan(_root: Path) -> list:
        app.state.motion_library_settings_store.save(
            MotionLibrarySettings(root=switched_root),
        )
        return []

    def scan_selected_root(root: str | Path | None = None) -> list[dict]:
        seen_scan_roots.append(Path(root).resolve() if root is not None else None)
        return []

    monkeypatch.setattr(viewer_library, "scan_library", switch_root_during_assets_scan)
    monkeypatch.setattr(motion_library_links, "scan_motions_library", scan_selected_root)

    with _local_client(app) as client:
        response = client.get("/api/library")

    assert response.status_code == 200
    assert response.json()["motions_library_root"] == str(switched_root)
    assert seen_scan_roots == [switched_root]


def test_library_link_uses_one_explicit_root_for_link_scan_and_response(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_app(tmp_path, monkeypatch)
    seen: dict[str, Path | None] = {}

    from hhtools.web.library import motion_library_links

    def link_with_root(
        _path: str,
        *,
        folder_label: str | None = None,
        library_root: str | Path | None = None,
    ) -> Path:
        del folder_label
        root = Path(library_root).resolve() if library_root is not None else None
        seen["link"] = root
        assert root is not None
        return root / "linked-dataset"

    def scan_with_root(root: str | Path | None = None) -> list[dict]:
        seen["scan"] = Path(root).resolve() if root is not None else None
        return []

    monkeypatch.setattr(motion_library_links, "link_to_library", link_with_root)
    monkeypatch.setattr(motion_library_links, "scan_motions_library", scan_with_root)

    with _local_client(app) as client:
        response = client.post("/api/library/link", json={"path": str(tmp_path)})

    response_root = Path(response.json()["motions_library_root"]).resolve()
    assert response.status_code == 200
    assert seen == {"link": response_root, "scan": response_root}
    assert Path(response.json()["path"]).parent == response_root


def test_library_api_exposes_stable_motion_category(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_app(tmp_path, monkeypatch)
    clip = tmp_path / "assets" / "motions" / "omomo" / "carry.pkl"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"clip")

    from hhtools.viewer import library as viewer_library
    from hhtools.web.library import motion_library_links

    monkeypatch.setattr(
        viewer_library,
        "scan_library",
        lambda _root: [
            SimpleNamespace(
                dataset="omomo",
                folder_label="opaque-folder-name",
                sequence_id="carry.pkl",
                stem="carry",
                source_path=clip,
                display_label="Carry",
            )
        ],
    )
    monkeypatch.setattr(
        motion_library_links,
        "scan_motions_library",
        lambda _root=None: [],
    )

    with _local_client(app) as client:
        response = client.get("/api/library")

    assert response.status_code == 200
    assert response.json()["entries"][0]["motion_category"] == "object"
    assert response.json()["entries"][0]["asset_kind"] == "human_motion"


def test_library_api_exposes_robot_trajectory_asset_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_app(tmp_path, monkeypatch)
    clip = tmp_path / "robot-exports" / "walk.csv"
    clip.parent.mkdir(parents=True)
    clip.write_text("root_x,root_y,root_z,dof_0\n0,0,1,0\n", encoding="utf-8")

    from hhtools.viewer import library as viewer_library
    from hhtools.web.library import motion_library_links

    monkeypatch.setattr(viewer_library, "scan_library", lambda _root: [])
    monkeypatch.setattr(
        motion_library_links,
        "scan_motions_library",
        lambda _root=None: [
            {
                "dataset": "robot",
                "folder_label": "G1 exports",
                "sequence_id": "walk.csv",
                "stem": "walk",
                "source_path": str(clip),
            }
        ],
    )

    with _local_client(app) as client:
        response = client.get("/api/library")

    assert response.status_code == 200
    assert response.json()["entries"][0]["asset_kind"] == "robot_trajectory"


def test_r2r_library_discovers_and_loads_robot_pkl_before_human_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from hhtools.web.server.routes import r2r as r2r_routes

    def complete_source_job(
        job,
        _drop,
        source_robot,
        _profile,
        _state,
        _source_fps=None,
        selected_path=None,
    ) -> None:
        job.result = {
            "token": "source-token",
            "source_robot": source_robot,
            "name": Path(selected_path).stem,
        }
        job.mark_terminal("done")

    monkeypatch.setattr(
        r2r_routes,
        "_run_r2r_source_upload_job",
        complete_source_job,
    )
    app = _create_app(tmp_path, monkeypatch)
    selected_root = tmp_path / "managed-motion-library"

    with _local_client(app) as client:
        switched = client.patch(
            "/api/settings/motion-library",
            json={"root": str(selected_root)},
        )
        assert switched.status_code == 200

        clip = selected_root / "G1 exports" / "walk.pkl"
        clip.parent.mkdir(parents=True)
        with clip.open("wb") as stream:
            pickle.dump(
                {
                    "hhtools_export": "r2r_v1",
                    "robot": {"joint_q": [[0.0] * 8]},
                },
                stream,
            )

        library = client.get("/api/library")
        assert library.status_code == 200
        entries = library.json()["entries"]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["dataset"] == "robot"
        assert entry["asset_kind"] == "robot_trajectory"
        assert entry["motion_category"] == "motion"

        started = client.post(
            "/api/r2r/source/library",
            json={**entry, "source_robot": "g1_29dof"},
        )
        assert started.status_code == 200
        job_id = started.json()["job_id"]
        for _ in range(100):
            completed = client.get(f"/api/job/{job_id}").json()
            if completed["status"] in {"done", "error"}:
                break
            time.sleep(0.01)

    assert completed["status"] == "done"
    assert completed["kind"] == "r2r_source_library"
    assert completed["result"]["name"] == "walk"


def test_library_api_includes_robot_trajectories_from_source_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clip = tmp_path / "assets" / "motions" / "r2r" / "samples" / "walk.csv"
    clip.parent.mkdir(parents=True)
    clip.write_text(
        "# robot: g1_29dof\n"
        "root_x,root_y,root_z,root_qx,root_qy,root_qz,root_qw,dof_0\n"
        "0,0,1,0,0,0,1,0\n",
        encoding="utf-8",
    )
    app = _create_app(tmp_path, monkeypatch)

    with _local_client(app) as client:
        response = client.get("/api/library")

    assert response.status_code == 200
    entry = next(row for row in response.json()["entries"] if row["source_path"] == str(clip))
    assert entry["dataset"] == "robot"
    assert entry["asset_kind"] == "robot_trajectory"
    assert entry["origin"] == "assets"
    assert entry["upload_profile"] == "mimic"
    assert entry["source_robot"] == "g1_29dof"


def test_managed_robot_library_preserves_declared_source_robot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_app(tmp_path, monkeypatch)
    selected_root = tmp_path / "managed-motion-library"

    with _local_client(app) as client:
        switched = client.patch(
            "/api/settings/motion-library",
            json={"root": str(selected_root)},
        )
        assert switched.status_code == 200
        clip = selected_root / "AgiBot X2" / "reach.csv"
        clip.parent.mkdir(parents=True)
        clip.write_text(
            "# robot: agibot_x2_ultra\n"
            "root_x,root_y,root_z,root_qx,root_qy,root_qz,root_qw,dof_0\n"
            "0,0,1,0,0,0,1,0\n",
            encoding="utf-8",
        )

        response = client.get("/api/library")

    assert response.status_code == 200
    entry = next(row for row in response.json()["entries"] if row["source_path"] == str(clip))
    assert entry["asset_kind"] == "robot_trajectory"
    assert entry["source_robot"] == "agibot_x2_ultra"


def test_library_api_includes_retained_h2r_result_as_r2r_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_app(tmp_path, monkeypatch)
    history = app.state.session_state.job_history
    generated = tmp_path / "runtime" / "walk.csv"
    generated.parent.mkdir()
    generated.write_text(
        "root_x,root_y,root_z,root_qx,root_qy,root_qz,root_qw,dof_0\n0,0,1,0,0,0,1,0\n",
        encoding="utf-8",
    )
    retained = history.adopt_artifact(
        "h2r-result",
        generated,
        download_name="walk.csv",
    )
    history.put(
        {
            "id": "h2r-result",
            "kind": "retarget",
            "status": "done",
            "created_at": 10.0,
            "finished_at": 11.0,
            "request": {"robot": "g1_29dof"},
            "artifact_path": str(retained),
            "download_name": retained.name,
        }
    )

    with _local_client(app) as client:
        response = client.get("/api/library")

    assert response.status_code == 200
    entry = next(row for row in response.json()["entries"] if row.get("job_id") == "h2r-result")
    assert entry["asset_kind"] == "robot_trajectory"
    assert entry["origin"] == "job"
    assert entry["source_robot"] == "g1_29dof"
    assert entry["source_path"] == str(retained)
