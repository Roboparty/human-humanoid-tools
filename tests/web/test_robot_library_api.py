from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hhtools.robot import registry
from hhtools.web import server
from hhtools.web.server import state as server_state

CURATED_ROBOT_NAMES = (
    "g1_29dof",
    "roboto_origin",
    "agibot_x2_ultra",
    "asimov_1",
    "fourier_gr2",
    "berkeley_humanoid_lite",
)


@pytest.fixture()
def robot_library_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    robot_root = tmp_path / "robots"
    robot_root.mkdir()
    monkeypatch.setattr(server_state, "_robot_library_root", lambda: robot_root)
    app = server.create_app(
        source_root=tmp_path / "motions",
        save_dir=tmp_path / "save",
        cache_dir=tmp_path / "cache",
    )
    return TestClient(app)


def _preset(name: str, display_name: str, dof: int) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        display_name=display_name,
        has_urdf=True,
        dof_order=tuple(f"joint_{index}" for index in range(dof)),
        root_dir=Path("robots") / name,
    )


def test_robot_list_marks_curated_models_as_builtin(
    robot_library_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curated = [_preset(name, name.replace("_", " ").title(), 29) for name in CURATED_ROBOT_NAMES]
    custom = _preset("custom_bot", "Custom Bot", 12)
    monkeypatch.setattr(registry, "refresh", lambda: [*curated, custom])
    monkeypatch.setattr(registry, "list_presets", lambda: [*curated, custom])
    monkeypatch.setattr(registry, "is_user_installed", lambda *_args: True)

    response = robot_library_client.get("/api/robots")

    assert response.status_code == 200
    robots = {item["name"]: item for item in response.json()["robots"]}
    for name in CURATED_ROBOT_NAMES:
        assert robots[name]["builtin"] is True
        assert robots[name]["deletable"] is False
    assert robots["custom_bot"]["builtin"] is False
    assert robots["custom_bot"]["deletable"] is True


@pytest.mark.parametrize("name", CURATED_ROBOT_NAMES)
def test_builtin_robot_cannot_be_deleted(
    robot_library_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    curated = _preset(name, name.replace("_", " ").title(), 29)
    monkeypatch.setattr(registry, "get", lambda _name: curated)

    response = robot_library_client.delete(f"/api/robot/{name}")

    assert response.status_code == 403
    assert "built-in preset" in response.json()["detail"]


@pytest.mark.parametrize("name", CURATED_ROBOT_NAMES)
def test_builtin_robot_cannot_be_overwritten_by_upload(
    robot_library_client: TestClient,
    name: str,
) -> None:
    robot_root = robot_library_client.app.state.session_state.robot_root
    robot_dir = robot_root / name
    robot_dir.mkdir(parents=True)
    marker = robot_dir / "original.txt"
    marker.write_text("curated robot data", encoding="utf-8")

    response = robot_library_client.post(
        "/api/robot/upload",
        params={"name": name},
        files={
            "files": (
                "replacement.urdf",
                b'<robot name="replacement"/>',
                "application/xml",
            )
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        f"robot {name!r} is a built-in preset and cannot be overwritten via upload"
    )
    assert marker.read_text(encoding="utf-8") == "curated robot data"


def test_custom_robot_upload_is_not_blocked_as_builtin(
    robot_library_client: TestClient,
) -> None:
    response = robot_library_client.post(
        "/api/robot/upload",
        params={"name": "custom_bot"},
        files={"files": ("readme.txt", b"custom robot", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "no .urdf file in upload"
