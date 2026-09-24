from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hhtools.retarget import robot_to_robot as r2r


def _payload(
    *,
    target: str = "target_bot",
    source: str = "source_bot",
    joint_q: object | None = None,
) -> dict[str, object]:
    return {
        "kind": "robot_to_robot",
        "target_robot": target,
        "source_robot": source,
        "calibrated_joint_q": joint_q if joint_q is not None else {"hip": 0.25},
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_writable_checkout_keeps_legacy_sibling_location(tmp_path: Path) -> None:
    target_dir = tmp_path / "target_bot"
    target_dir.mkdir()
    user_root = tmp_path / "user-robots"

    saved = r2r.save_r2r_calibration(
        target_dir,
        target_robot="target_bot",
        source_robot="source_bot",
        calibrated_joint_q={"hip": 0.25},
        user_root=user_root,
    )

    assert saved == target_dir / "r2r_calibration_source_bot.yaml"
    assert (
        r2r.resolve_r2r_calibration_file(
            target_dir,
            "source_bot",
            target_robot="target_bot",
            user_root=user_root,
        )
        == saved
    )
    assert r2r.load_r2r_calibration(
        target_dir,
        "source_bot",
        target_robot="target_bot",
        user_root=user_root,
    ) == {"hip": 0.25}


def test_readonly_sibling_falls_back_to_user_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_dir = tmp_path / "bundled" / "target_bot"
    target_dir.mkdir(parents=True)
    user_root = tmp_path / "user-robots"
    original_write = r2r._atomic_write_r2r_payload

    def reject_bundled(path: Path, payload: dict[str, object]) -> None:
        if path.parent == target_dir:
            raise PermissionError("read-only bundled runtime")
        original_write(path, payload)

    monkeypatch.setattr(r2r, "_atomic_write_r2r_payload", reject_bundled)

    saved = r2r.save_r2r_calibration(
        target_dir,
        target_robot="target_bot",
        source_robot="source_bot",
        calibrated_joint_q={"hip": -0.5},
        user_root=user_root,
    )

    assert saved == user_root / "target_bot" / "r2r_calibration_source_bot.yaml"
    assert saved.is_file()
    assert not (target_dir / "r2r_calibration_source_bot.yaml").exists()
    assert r2r.load_r2r_calibration(
        target_dir,
        "source_bot",
        target_robot="target_bot",
        user_root=user_root,
    ) == {"hip": -0.5}


def test_agent_save_can_force_the_user_overlay_in_a_writable_checkout(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "target_bot"
    target_dir.mkdir()
    user_root = tmp_path / "user-robots"

    saved = r2r.save_r2r_calibration(
        target_dir,
        target_robot="target_bot",
        source_robot="source_bot",
        calibrated_joint_q={"hip": 0.25},
        user_root=user_root,
        prefer_user_overlay=True,
    )

    assert saved == user_root / "target_bot" / "r2r_calibration_source_bot.yaml"
    assert not (target_dir / saved.name).exists()


def test_library_r2r_overlay_is_authoritative_for_subsequent_ui_saves(tmp_path: Path) -> None:
    user_root = tmp_path / "robots"
    target_dir = user_root / "target_bot"
    bundled = target_dir / "r2r_calibration_source_bot.yaml"
    _write(bundled, _payload())
    previous = bundled.read_bytes()
    request = dict(
        target_robot="target_bot",
        source_robot="source_bot",
        calibrated_joint_q={"hip": 0.75},
        user_root=user_root,
    )
    saved = r2r.save_r2r_calibration(target_dir, **request, prefer_user_overlay=True)
    assert saved == user_root / ".calibration-overlays" / "target_bot" / bundled.name
    assert r2r.save_r2r_calibration(target_dir, **request) == saved
    assert r2r.load_r2r_calibration(
        target_dir,
        "source_bot",
        target_robot="target_bot",
        user_root=user_root,
    ) == {"hip": 0.75}
    assert bundled.read_bytes() == previous
    saved.write_text("invalid: [")
    with pytest.raises(ValueError):
        r2r.load_r2r_calibration(
            target_dir,
            "source_bot",
            target_robot="target_bot",
            user_root=user_root,
        )


def test_existing_user_override_wins_and_receives_later_saves(tmp_path: Path) -> None:
    target_dir = tmp_path / "target_bot"
    target_dir.mkdir()
    user_root = tmp_path / "user-robots"
    bundled = r2r.r2r_calibration_path(target_dir, "source_bot")
    override = r2r.r2r_user_calibration_path(
        "target_bot",
        "source_bot",
        user_root=user_root,
    )
    _write(bundled, _payload(joint_q={"hip": 0.1}))
    _write(override, _payload(joint_q={"hip": 0.8}))

    assert r2r.load_r2r_calibration(
        target_dir,
        "source_bot",
        target_robot="target_bot",
        user_root=user_root,
    ) == {"hip": 0.8}

    saved = r2r.save_r2r_calibration(
        target_dir,
        target_robot="target_bot",
        source_robot="source_bot",
        calibrated_joint_q={"hip": 0.9},
        user_root=user_root,
    )
    assert saved == override
    assert r2r.load_r2r_calibration(
        target_dir,
        "source_bot",
        target_robot="target_bot",
        user_root=user_root,
    ) == {"hip": 0.9}
    assert yaml.safe_load(bundled.read_text(encoding="utf-8"))["calibrated_joint_q"] == {
        "hip": 0.1,
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"kind": "human_to_robot"}, "kind"),
        ({"target_robot": "another_target"}, "target"),
        ({"source_robot": "another_source"}, "source"),
        ({"calibrated_joint_q": {"hip": ".nan"}}, "numeric"),
        ({"calibrated_joint_q": {"hip": float("inf")}}, "non-finite"),
        ({"calibrated_joint_q": {}}, "non-empty"),
        ({"notes": 42}, "notes"),
    ],
)
def test_canonical_user_override_is_strictly_validated(
    tmp_path: Path,
    change: dict[str, object],
    message: str,
) -> None:
    target_dir = tmp_path / "bundled" / "target_bot"
    target_dir.mkdir(parents=True)
    user_root = tmp_path / "user-robots"
    override = r2r.r2r_user_calibration_path(
        "target_bot",
        "source_bot",
        user_root=user_root,
    )
    payload = _payload()
    payload.update(change)
    _write(override, payload)

    with pytest.raises(ValueError, match=message):
        r2r.load_r2r_calibration(
            target_dir,
            "source_bot",
            target_robot="target_bot",
            user_root=user_root,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_save_rejects_non_finite_or_boolean_joint_values(
    tmp_path: Path,
    value: object,
) -> None:
    target_dir = tmp_path / "target_bot"
    target_dir.mkdir()

    with pytest.raises(ValueError):
        r2r.save_r2r_calibration(
            target_dir,
            target_robot="target_bot",
            source_robot="source_bot",
            calibrated_joint_q={"hip": value},  # type: ignore[dict-item]
            user_root=tmp_path / "user-robots",
        )
    assert list(tmp_path.rglob("r2r_calibration_*.yaml")) == []


def test_unsafe_robot_names_cannot_escape_or_collide(tmp_path: Path) -> None:
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    first = r2r.r2r_calibration_path(target_dir, "../source:robot")
    second = r2r.r2r_calibration_path(target_dir, "..\\source/robot")
    reserved = r2r.r2r_calibration_path(target_dir, "CON.txt")

    assert first.parent == target_dir.resolve()
    assert second.parent == target_dir.resolve()
    assert first != second
    assert first.name.startswith("r2r_calibration_id-")
    assert second.name.startswith("r2r_calibration_id-")
    assert reserved.name.startswith("r2r_calibration_id-")


@pytest.mark.parametrize("target", ["../../target", "CON", "CON.txt", "target."])
def test_user_overlay_rejects_unsafe_target(
    tmp_path: Path,
    target: str,
) -> None:
    user_root = tmp_path / "user-robots"

    with pytest.raises(ValueError, match="unsafe target_robot"):
        r2r.r2r_user_calibration_path(
            target,
            "source_bot",
            user_root=user_root,
        )


def test_user_overlay_preserves_preset_namespace(tmp_path: Path) -> None:
    user_root = tmp_path / "user-robots"

    user_path = r2r.r2r_user_calibration_path(
        "机器人",
        "C:\\source\\robot",
        user_root=user_root,
    )
    assert user_path.resolve().is_relative_to(user_root.resolve())
    assert user_path.parent == user_root.resolve() / "机器人"
    assert user_path.name.startswith("r2r_calibration_id-")


def test_sibling_path_preserves_relative_target_directory() -> None:
    path = r2r.r2r_calibration_path(Path("robots") / "target_bot", "source_bot")

    assert path == Path("robots/target_bot/r2r_calibration_source_bot.yaml")
    assert not path.is_absolute()


def test_old_lossy_sibling_filename_remains_readable(tmp_path: Path) -> None:
    target_dir = tmp_path / "target_bot"
    target_dir.mkdir()
    user_root = tmp_path / "user-robots"
    source = "source:robot/variant"
    legacy = target_dir / "r2r_calibration_source_robot_variant.yaml"
    _write(legacy, _payload(source=source))

    assert r2r.r2r_calibration_path(target_dir, source) != legacy
    assert (
        r2r.resolve_r2r_calibration_file(
            target_dir,
            source,
            target_robot="target_bot",
            user_root=user_root,
        )
        == legacy.resolve()
    )
    assert r2r.load_r2r_calibration(
        target_dir,
        source,
        target_robot="target_bot",
        user_root=user_root,
    ) == {"hip": 0.25}


def test_old_lossy_user_override_is_strictly_validated(tmp_path: Path) -> None:
    target_dir = tmp_path / "bundled" / "target_bot"
    target_dir.mkdir(parents=True)
    user_root = tmp_path / "user-robots"
    source = "source:robot/variant"
    legacy = user_root / "target_bot" / "r2r_calibration_source_robot_variant.yaml"
    _write(legacy, _payload(source=source, target="another_target"))

    with pytest.raises(ValueError, match="target"):
        r2r.load_r2r_calibration(
            target_dir,
            source,
            target_robot="target_bot",
            user_root=user_root,
        )


def test_existing_legacy_user_layer_receives_new_saves(tmp_path: Path) -> None:
    target_dir = tmp_path / "target_bot"
    target_dir.mkdir()
    user_root = tmp_path / "user-robots"
    source = "source:robot/variant"
    legacy = user_root / "target_bot" / "r2r_calibration_source_robot_variant.yaml"
    _write(legacy, _payload(source=source))

    saved = r2r.save_r2r_calibration(
        target_dir,
        target_robot="target_bot",
        source_robot=source,
        calibrated_joint_q={"hip": 0.75},
        user_root=user_root,
    )

    assert saved == r2r.r2r_user_calibration_path(
        "target_bot",
        source,
        user_root=user_root,
    )
    assert r2r.load_r2r_calibration(
        target_dir,
        source,
        target_robot="target_bot",
        user_root=user_root,
    ) == {"hip": 0.75}
    assert list(target_dir.glob("r2r_calibration_*.yaml")) == []


def test_target_argument_remains_optional_for_legacy_callers(tmp_path: Path) -> None:
    target_dir = tmp_path / "target_bot"
    target_dir.mkdir()
    sibling = r2r.r2r_calibration_path(target_dir, "source_bot")
    _write(sibling, _payload())

    assert r2r.load_r2r_calibration(
        target_dir,
        "source_bot",
        user_root=tmp_path / "user-robots",
    ) == {"hip": 0.25}


def test_user_calibration_symlink_is_rejected(tmp_path: Path) -> None:
    target_dir = tmp_path / "bundled" / "target_bot"
    target_dir.mkdir(parents=True)
    user_root = tmp_path / "user-robots"
    outside = tmp_path / "outside.yaml"
    _write(outside, _payload())
    override = r2r.r2r_user_calibration_path(
        "target_bot",
        "source_bot",
        user_root=user_root,
    )
    override.parent.mkdir(parents=True)
    try:
        override.symlink_to(outside)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable on this host")

    with pytest.raises(ValueError, match="calibration path escapes its storage root"):
        r2r.load_r2r_calibration(
            target_dir,
            "source_bot",
            target_robot="target_bot",
            user_root=user_root,
        )
