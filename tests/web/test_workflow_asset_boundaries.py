from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hhtools.web import server
from hhtools.web.library.r2r_upload_resolve import r2r_clip_ref_for_path
from hhtools.web.server import state as server_state


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv(
        "HHTOOLS_MOTION_LIBRARY_SETTINGS_PATH",
        str(tmp_path / "motion-library-settings.json"),
    )
    monkeypatch.setattr(server_state, "_robot_library_root", lambda: tmp_path / "robots")
    app = server.create_app(
        source_root=tmp_path / "motions",
        save_dir=tmp_path / "save",
        cache_dir=tmp_path / "cache",
    )
    return TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    )


def test_h2r_rejects_robot_joint_trajectory(tmp_path: Path, monkeypatch) -> None:
    trajectory = tmp_path / "robot.csv"
    trajectory.write_text(
        "root_x,root_y,root_z,root_qx,root_qy,root_qz,root_qw,dof_0\n0,0,1,0,0,0,1,0\n",
        encoding="utf-8",
    )

    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/motion/load_library",
            json={
                "usage": "human_to_robot",
                "dataset": "robot",
                "folder_label": "exports",
                "sequence_id": trajectory.name,
                "source_path": str(trajectory),
            },
        )

    assert response.status_code == 422
    assert "只接受人体动作" in response.json()["detail"]


def test_r2r_rejects_human_motion(tmp_path: Path, monkeypatch) -> None:
    motion = tmp_path / "walk.bvh"
    motion.write_text("HIERARCHY\nMOTION\nFrames: 0\n", encoding="utf-8")

    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/r2r/source/library",
            json={
                "source_robot": "g1_29dof",
                "dataset": "lafan",
                "folder_label": "LAFAN",
                "sequence_id": motion.name,
                "source_path": str(motion),
            },
        )

    assert response.status_code == 422
    assert "不是机器人轨迹" in response.json()["detail"]


def test_r2r_library_selection_keeps_the_exact_selected_clip(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    selected = tmp_path / "selected.csv"
    robot_header = "root_x,root_y,root_z,root_qx,root_qy,root_qz,root_qw,dof_0\n"
    first.write_text(f"{robot_header}0,0,1,0,0,0,1,0\n", encoding="utf-8")
    selected.write_text(f"{robot_header}0,0,1,0,0,0,1,0.2\n", encoding="utf-8")

    clip = r2r_clip_ref_for_path(selected)

    assert clip.path == selected.resolve()
    assert clip.profile == "mimic"
