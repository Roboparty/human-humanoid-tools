from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from hhtools.retarget import calibration as calibration_api
from hhtools.retarget.calibration import calibration as calibration_impl
from hhtools.robot import registry, retarget_profile
from hhtools.web import server


def test_calibration_save_uses_user_overlay_for_packaged_robot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The Web endpoint must never require write access to packaged assets."""

    bundled = tmp_path / "opt" / "Human-Humanoid Tools" / "robots" / "g1_29dof"
    bundled.mkdir(parents=True)
    urdf = bundled / "g1.urdf"
    urdf.write_text('<robot name="g1"/>', encoding="utf-8")
    user_robot_root = tmp_path / "user-config" / "robots"
    preset = SimpleNamespace(
        name="g1_29dof",
        root_dir=bundled,
        urdf_path=urdf,
    )

    monkeypatch.setenv("HHTOOLS_ROBOT_DIR", str(user_robot_root))
    monkeypatch.setattr(registry, "get", lambda name: preset)
    monkeypatch.setattr(
        calibration_api,
        "derive_calibration_params",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(retarget_profile, "bundled_scaler_path", lambda *_args: None)
    # chmod is not a reliable read-only simulation on Windows or when tests run
    # as root.  Force the same branch an installed /opt directory takes.
    monkeypatch.setattr(calibration_impl, "_path_appears_writable", lambda _path: False)

    app = server.create_app(
        source_root=tmp_path / "motions",
        save_dir=tmp_path / "save",
        cache_dir=tmp_path / "cache",
    )
    app.state.session_state.robots[preset.name] = object()

    response = TestClient(app).post(
        "/api/calibration/save",
        json={
            "robot": preset.name,
            "reference": "lafan_bvh",
            "joint_q": {"left_knee_joint": 0.25},
        },
    )

    assert response.status_code == 200, response.text
    expected = (
        user_robot_root
        / preset.name
        / "retarget_calibration_lafan_bvh.yaml"
    ).resolve()
    assert Path(response.json()["path"]).resolve() == expected
    assert expected.is_file()
    assert not (bundled / expected.name).exists()
    saved = calibration_api.load_calibration(expected)
    assert saved.robot == preset.name
    assert saved.reference == "lafan_bvh"
    assert saved.calibrated_joint_q == {"left_knee_joint": 0.25}

    status = TestClient(app).get(
        "/api/calibration/status",
        params={"robot": preset.name, "reference": "lafan_bvh"},
    )
    assert status.status_code == 200, status.text
    assert status.json() == {
        "calibrated": True,
        "bundled": False,
        "path": str(expected),
        "joint_q": {"left_knee_joint": 0.25},
    }
