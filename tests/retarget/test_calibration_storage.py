from __future__ import annotations

import errno
from pathlib import Path

import pytest

from hhtools.retarget.calibration import (
    RobotRetargetCalibration,
    load_calibration,
    resolve_preset_calibration_file,
    save_calibration,
    save_calibration_for_preset,
)
from hhtools.retarget.calibration import calibration as calibration_module
from hhtools.robot.base import RobotPreset


def _preset(tmp_path: Path, *, name: str = "test_robot") -> RobotPreset:
    root = tmp_path / "bundled" / name
    description = root / "description"
    description.mkdir(parents=True)
    urdf = description / "robot.urdf"
    urdf.write_text("<robot name='test_robot'/>", encoding="utf-8")
    return RobotPreset(
        name=name,
        display_name=name,
        root_dir=root,
        urdf_path=urdf,
        dof_order=("hip_joint",),
    )


def _calibration(
    preset: RobotPreset,
    *,
    reference: str = "smpl",
    robot: str | None = None,
) -> RobotRetargetCalibration:
    return RobotRetargetCalibration(
        robot=robot if robot is not None else preset.name,
        reference=reference,  # type: ignore[arg-type]
        calibrated_joint_q={"hip_joint": 0.25},
    )


def test_user_calibration_overrides_bundled_file(tmp_path: Path) -> None:
    preset = _preset(tmp_path)
    user_root = tmp_path / "user-robots"
    bundled = preset.urdf_path.parent / "retarget_calibration_smpl.yaml"  # type: ignore[union-attr]
    override = user_root / preset.name / "retarget_calibration_smpl.yaml"
    save_calibration(_calibration(preset), bundled)
    user_cal = _calibration(preset)
    user_cal.calibrated_joint_q["hip_joint"] = 0.75
    save_calibration(user_cal, override)

    resolved = resolve_preset_calibration_file(preset, "smpl", user_root)

    assert resolved == override.resolve()
    assert load_calibration(resolved).calibrated_joint_q["hip_joint"] == 0.75


def test_library_overlay_preserves_bundle_and_receives_later_ui_saves(tmp_path: Path) -> None:
    preset = _preset(tmp_path)
    user_root = preset.root_dir.parent
    bundled = preset.urdf_path.parent / "retarget_calibration_smpl.yaml"
    save_calibration(_calibration(preset), bundled)
    previous = bundled.read_bytes()
    calibration = _calibration(preset)
    calibration.calibrated_joint_q["hip_joint"] = 0.5
    saved = save_calibration_for_preset(
        calibration,
        preset,
        user_robot_root=user_root,
        prefer_user_overlay=True,
    )
    assert saved == user_root / ".calibration-overlays" / preset.name / bundled.name
    assert resolve_preset_calibration_file(preset, "smpl", user_root) == saved
    calibration.calibrated_joint_q["hip_joint"] = 0.75
    assert save_calibration_for_preset(calibration, preset, user_robot_root=user_root) == saved
    assert load_calibration(saved).calibrated_joint_q["hip_joint"] == 0.75
    assert bundled.read_bytes() == previous
    saved.write_text("invalid: [")
    with pytest.raises(ValueError, match="invalid retarget calibration"):
        resolve_preset_calibration_file(preset, "smpl", user_root)


@pytest.mark.parametrize("link_layer", [True, False])
def test_separate_overlay_cannot_link_back_into_robot_bundle(
    tmp_path: Path,
    link_layer: bool,
) -> None:
    from hhtools.utils.paths import user_calibration_overlay_dir

    preset = _preset(tmp_path)
    user_root = preset.root_dir.parent
    layer = user_root / ".calibration-overlays"
    if link_layer:
        layer.symlink_to(user_root, target_is_directory=True)
    else:
        layer.mkdir()
        (layer / preset.name).symlink_to(preset.root_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be symlinks"):
        user_calibration_overlay_dir(preset.name, user_root=user_root)


def test_bundled_file_is_read_only_fallback_when_user_override_is_absent(
    tmp_path: Path,
) -> None:
    preset = _preset(tmp_path)
    bundled = preset.urdf_path.parent / "retarget_calibration_smpl.yaml"  # type: ignore[union-attr]
    save_calibration(_calibration(preset), bundled)

    resolved = resolve_preset_calibration_file(
        preset,
        "smpl",
        tmp_path / "missing-user-root",
    )

    assert resolved == bundled.resolve()
    assert not (tmp_path / "missing-user-root").exists()


def test_malformed_user_override_does_not_silently_fall_back(
    tmp_path: Path,
) -> None:
    preset = _preset(tmp_path)
    user_root = tmp_path / "user-robots"
    override = user_root / preset.name / "retarget_calibration_smpl.yaml"
    override.parent.mkdir(parents=True)
    override.write_text("calibrated_joint_q: [", encoding="utf-8")
    save_calibration(
        _calibration(preset),
        preset.urdf_path.parent / "retarget_calibration_smpl.yaml",  # type: ignore[union-attr]
    )

    with pytest.raises(ValueError, match="invalid retarget calibration"):
        resolve_preset_calibration_file(preset, "smpl", user_root)


def test_identity_mismatched_user_override_does_not_silently_fall_back(
    tmp_path: Path,
) -> None:
    preset = _preset(tmp_path)
    user_root = tmp_path / "user-robots"
    override = user_root / preset.name / "retarget_calibration_smpl.yaml"
    save_calibration(_calibration(preset, robot="different_robot"), override)

    with pytest.raises(ValueError, match="does not match preset"):
        resolve_preset_calibration_file(preset, "smpl", user_root)


def test_legacy_user_file_for_another_reference_allows_bundled_fallback(
    tmp_path: Path,
) -> None:
    preset = _preset(tmp_path)
    user_root = tmp_path / "user-robots"
    save_calibration(
        _calibration(preset, reference="smplx"),
        user_root / preset.name / "retarget_calibration.yaml",
    )
    bundled = preset.urdf_path.parent / "retarget_calibration_smpl.yaml"  # type: ignore[union-attr]
    save_calibration(_calibration(preset), bundled)

    assert resolve_preset_calibration_file(preset, "smpl", user_root) == bundled.resolve()


def test_reference_alias_resolves_canonical_filename(tmp_path: Path) -> None:
    preset = _preset(tmp_path)
    user_root = tmp_path / "user-robots"
    canonical = user_root / preset.name / "retarget_calibration_lafan_bvh.yaml"
    save_calibration(_calibration(preset, reference="lafan_bvh"), canonical)

    assert resolve_preset_calibration_file(preset, "mixamo_bvh", user_root) == canonical.resolve()


@pytest.mark.parametrize("name", ("../escape", "nested/robot", r"nested\robot", "C:robot"))
def test_user_overlay_rejects_unsafe_preset_names(tmp_path: Path, name: str) -> None:
    preset = _preset(tmp_path, name="safe")
    preset.name = name

    with pytest.raises(ValueError, match="unsafe robot preset name"):
        resolve_preset_calibration_file(preset, "smpl", tmp_path / "user-robots")


def test_save_keeps_writable_source_tree_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _preset(tmp_path)
    user_root = tmp_path / "user-robots"
    monkeypatch.setattr(calibration_module, "_path_appears_writable", lambda _path: True)

    result = save_calibration_for_preset(
        _calibration(preset),
        preset,
        user_robot_root=user_root,
    )

    assert result == preset.urdf_path.parent / "retarget_calibration_smpl.yaml"  # type: ignore[union-attr]
    assert not (user_root / preset.name).exists()


def test_save_stays_in_user_layer_after_an_override_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _preset(tmp_path)
    user_root = tmp_path / "user-robots"
    override = user_root / preset.name / "retarget_calibration_smpl.yaml"
    save_calibration(_calibration(preset), override)
    monkeypatch.setattr(calibration_module, "_path_appears_writable", lambda _path: True)

    updated = _calibration(preset)
    updated.calibrated_joint_q["hip_joint"] = 0.9
    result = save_calibration_for_preset(
        updated,
        preset,
        user_robot_root=user_root,
    )

    assert result == override.resolve()
    assert load_calibration(result).calibrated_joint_q["hip_joint"] == 0.9


def test_save_uses_user_layer_when_bundled_directory_is_not_writable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _preset(tmp_path)
    user_root = tmp_path / "user-robots"
    monkeypatch.setattr(calibration_module, "_path_appears_writable", lambda _path: False)

    result = save_calibration_for_preset(
        _calibration(preset),
        preset,
        user_robot_root=user_root,
    )

    assert result == (user_root / preset.name / "retarget_calibration_smpl.yaml").resolve()
    assert result.is_file()


@pytest.mark.parametrize("failure_errno", (errno.EACCES, errno.EPERM, errno.EROFS))
def test_save_retries_in_user_layer_after_authoritative_write_denial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_errno: int,
) -> None:
    preset = _preset(tmp_path)
    user_root = tmp_path / "user-robots"
    bundled = preset.urdf_path.parent / "retarget_calibration_smpl.yaml"  # type: ignore[union-attr]
    real_save = calibration_module.save_calibration
    monkeypatch.setattr(calibration_module, "_path_appears_writable", lambda _path: True)

    def denied_once(calibration, path, *, derived=None):
        if Path(path) == bundled:
            raise OSError(failure_errno, "read-only packaged preset")
        return real_save(calibration, path, derived=derived)

    monkeypatch.setattr(calibration_module, "save_calibration", denied_once)

    result = save_calibration_for_preset(
        _calibration(preset),
        preset,
        user_robot_root=user_root,
    )

    assert result == (user_root / preset.name / "retarget_calibration_smpl.yaml").resolve()


def test_save_rejects_calibration_for_another_robot(tmp_path: Path) -> None:
    preset = _preset(tmp_path)

    with pytest.raises(ValueError, match="does not match preset"):
        save_calibration_for_preset(
            _calibration(preset, robot="different_robot"),
            preset,
            user_robot_root=tmp_path / "user-robots",
        )
