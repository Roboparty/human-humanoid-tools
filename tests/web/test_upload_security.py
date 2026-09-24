from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hhtools.web import server
from hhtools.web.library.motion_library_links import _safe_folder_name, motions_library_root
from hhtools.web.server import boundary as server_boundary
from hhtools.web.server import library_runtime
from hhtools.web.server import state as server_state


@pytest.fixture()
def web_client(tmp_path: Path, monkeypatch) -> TestClient:
    def local_tmpdir(tag: str) -> Path:
        path = tmp_path / f"runtime-{tag}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(server_state, "_tmpdir", local_tmpdir)
    monkeypatch.setattr(server_state, "_robot_library_root", lambda: tmp_path / "robots")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    app = server.create_app(
        source_root=tmp_path / "motions",
        save_dir=tmp_path / "save",
        cache_dir=tmp_path / "cache",
    )
    return TestClient(app)


@pytest.mark.parametrize(
    "filename",
    [
        "../escaped.bin",
        "..\\escaped.bin",
        "/escaped.bin",
        "C:\\escaped.bin",
        "C:/escaped.bin",
        "C:escaped.bin",
        "\\\\server\\share\\escaped.bin",
        "safe\\..\\escaped.bin",
    ],
)
def test_upload_relative_path_rejects_escape_forms(filename: str) -> None:
    with pytest.raises(ValueError):
        server_boundary._safe_upload_relative_path(filename)


def test_upload_relative_path_allows_safe_nested_path() -> None:
    relative = server_boundary._safe_upload_relative_path("dataset/session/clip.bvh")

    assert relative.parts == ("dataset", "session", "clip.bvh")


def test_resolved_upload_destination_must_stay_below_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        server_boundary._ensure_path_within(tmp_path / "root", tmp_path / "escaped.bin")


def test_materialized_clip_matches_an_unwrapped_upload_tree(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "drop"
    snapshot_clip = snapshot_root / "wrapper" / "nested" / "clip.bvh"
    library_root = tmp_path / "library"
    library_clip = library_root / "nested" / "clip.bvh"
    snapshot_clip.parent.mkdir(parents=True)
    library_clip.parent.mkdir(parents=True)
    snapshot_clip.write_bytes(b"snapshot")
    library_clip.write_bytes(b"materialized")

    matched = library_runtime._matching_materialized_clip(
        library_root,
        snapshot_root=snapshot_root,
        snapshot_picked=snapshot_clip,
        profile="mimic",
    )

    assert matched == library_clip.resolve()


@pytest.mark.parametrize(
    ("url", "data"),
    [
        ("/api/dataset/upload", {}),
        ("/api/basket/upload", {}),
        ("/api/motion/upload", {}),
        ("/api/robot/upload", {"name": "safe-robot"}),
        ("/api/r2r/source/upload", {"source_robot": "unit-test"}),
        ("/api/r2r/basket/upload", {}),
    ],
)
def test_all_upload_routes_reject_parent_directory_escape(
    web_client: TestClient,
    tmp_path: Path,
    url: str,
    data: dict[str, str],
) -> None:
    response = web_client.post(
        url,
        data=data,
        files={"files": ("../escaped.bin", b"outside", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert not list(tmp_path.rglob("escaped.bin"))


def test_robot_upload_rejects_escaping_robot_name(
    web_client: TestClient,
    tmp_path: Path,
) -> None:
    response = web_client.post(
        "/api/robot/upload",
        data={"name": "../escaped-robot"},
        files={"files": ("robot.urdf", b"<robot/>", "application/xml")},
    )

    assert response.status_code == 400
    assert not (tmp_path / "escaped-robot").exists()


def test_invalid_motion_upload_returns_400_without_publishing_library_data(
    web_client: TestClient,
) -> None:
    library_dir = motions_library_root() / "invalid-motion-test"
    library_dir.mkdir(parents=True, exist_ok=True)
    marker = library_dir / "keep.bvh"
    marker.write_bytes(b"existing library data")

    response = web_client.post(
        "/api/motion/upload",
        params={"library_folder_label": "invalid-motion-test"},
        files={
            "files": (
                "readme.definitely-not-motion",
                b"not a motion clip",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400
    assert "未找到可识别的动作文件" in response.json()["detail"]
    assert marker.read_bytes() == b"existing library data"
    assert not (library_dir / "readme.definitely-not-motion").exists()
    assert not list(
        web_client.app.state.session_state.upload_root.rglob("readme.definitely-not-motion")
    )


@pytest.mark.parametrize(
    ("profile", "expected_fragments"),
    [
        (
            "intermimic",
            ("intermimic", "OMOMO", "<clip>/<clip>.pkl", "*_cleaned_simplified.obj"),
        ),
        (
            "meshmimic",
            ("meshmimic", "parc_ms", "<clip>/<clip>.pkl", "*_terrain.obj"),
        ),
        (
            "mimic",
            ("动作文件（mimic）", ".bvh", ".gltf", ".pt"),
        ),
    ],
)
def test_invalid_motion_upload_explains_selected_profile(
    web_client: TestClient,
    profile: str,
    expected_fragments: tuple[str, ...],
) -> None:
    response = web_client.post(
        "/api/motion/upload",
        params={"profile": profile},
        files={
            "files": (
                "readme.definitely-not-motion",
                b"not a motion clip",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    for fragment in expected_fragments:
        assert fragment in detail


def test_invalid_auto_basket_upload_reports_generic_requirements(
    web_client: TestClient,
) -> None:
    response = web_client.post(
        "/api/basket/upload",
        files={
            "files": (
                "readme.definitely-not-motion",
                b"not a motion clip",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    deadline = time.monotonic() + 2.0
    payload: dict = {}
    while time.monotonic() < deadline:
        payload = web_client.get(f"/api/job/{job_id}").json()
        if payload.get("status") == "error":
            break
        time.sleep(0.005)

    assert payload.get("status") == "error"
    assert "intermimic / meshmimic" in payload["error"]
    assert "（mimic）" not in payload["error"]


def test_motion_library_label_is_one_directory_name() -> None:
    label = _safe_folder_name("nested/../../escaped-library")

    assert "/" not in label
    assert "\\" not in label
    assert Path(label).name == label
