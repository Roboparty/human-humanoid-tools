"""Export bundle includes interaction-object retarget tracks for OMOMO-style clips."""

from __future__ import annotations

import csv
import io
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

from hhtools.core.motion import Motion
from hhtools.io.datasets.omomo import OmomoAdapter
from hhtools.retarget.retarget_result import RetargetedMotion
from hhtools.web.output import r2r_export_bundle
from hhtools.web.output.export_bundle import (
    OBJECT_CSV_HEADER,
    _resolve_export_scene_params,
    resolve_clip_export_dir,
    sanitize_export_stem,
    write_retarget_export_bundle,
    zip_directory,
)


@pytest.fixture()
def woodchair_motion() -> Motion:
    root = Path("assets/motions/intermimic/OMOMO")
    pkl = root / "sub12_woodchair_000" / "sub12_woodchair_000.pkl"
    if not pkl.is_file():
        pytest.skip(f"OMOMO demo missing: {pkl}")
    return OmomoAdapter(root=root).load_motion(
        "sub12_woodchair_000/sub12_woodchair_000.pkl",
    )


def test_omomo_pkl_carries_object_track(woodchair_motion: Motion) -> None:
    assert woodchair_motion.objects
    ob = woodchair_motion.objects[0]
    assert ob.positions.shape[0] == woodchair_motion.num_frames
    assert ob.quaternions.shape[0] == woodchair_motion.num_frames
    assert ob.name == "woodchair"


def test_export_zip_includes_object_pkl(tmp_path: Path, woodchair_motion: Motion) -> None:
    ret = RetargetedMotion(
        name="sub12_woodchair_000",
        joint_q=np.zeros((woodchair_motion.num_frames, 10), dtype=np.float32),
        sample_rate=float(woodchair_motion.framerate),
        dof_names=("j1", "j2", "j3"),
        meta={"smpl_scale": 0.85, "source_z_min": 0.12},
    )

    class _DummyModel:
        preset = type("P", (), {"name": "test_robot"})()

    zip_path = write_retarget_export_bundle(
        ret,
        _DummyModel(),
        woodchair_motion,
        tmp_path,
        stem="sub12_woodchair_000",
        fps=None,
        fmt="pkl",
        backend="interaction_mesh",
        resample_fn=lambda r, fps: (r.joint_q, r.sample_rate),
    )
    assert zip_path.suffix == ".zip"
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert any(n.endswith("object_0_woodchair.pkl") for n in names)
    assert any(n.endswith("woodchair_cleaned_simplified.obj") for n in names)

    with zipfile.ZipFile(zip_path) as zf:
        obj_name = next(n for n in zf.namelist() if n.endswith("object_0_woodchair.pkl"))
        with zf.open(obj_name) as fh:
            blob = pickle.load(fh)
    assert blob["name"] == "woodchair"
    assert blob["positions"].shape == (woodchair_motion.num_frames, 3)
    assert blob["quat_format"] == "wxyz"
    assert blob["frame"] == "retarget_robot"
    assert "mesh_scale" not in blob


def test_export_csv_includes_object_csv(tmp_path: Path, woodchair_motion: Motion) -> None:
    ret = RetargetedMotion(
        name="sub12_woodchair_000",
        joint_q=np.zeros((woodchair_motion.num_frames, 10), dtype=np.float32),
        sample_rate=float(woodchair_motion.framerate),
        dof_names=("j1", "j2", "j3"),
        meta={"smpl_scale": 0.85, "source_z_min": 0.12},
    )

    class _DummyModel:
        preset = type("P", (), {"name": "test_robot"})()

        def dof_names(self):
            return ("j1", "j2", "j3")

    zip_path = write_retarget_export_bundle(
        ret,
        _DummyModel(),
        woodchair_motion,
        tmp_path,
        stem="sub12_woodchair_000",
        fps=None,
        fmt="csv",
        backend="interaction_mesh",
        resample_fn=lambda r, fps: (r.joint_q, r.sample_rate),
    )
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert any(n.endswith("object_0_woodchair.csv") for n in names)
        assert any(n.endswith("sub12_woodchair_000.csv") for n in names)
        csv_name = next(n for n in names if n.endswith("object_0_woodchair.csv"))
        text = zf.read(csv_name).decode("utf-8")

    rows = list(csv.reader(io.StringIO(text)))
    data_rows = [r for r in rows if r and r[0] != "time" and not r[0].startswith("#")]
    assert len(data_rows) == woodchair_motion.num_frames
    header_row = next(r for r in rows if r and r[0] == "time")
    assert header_row == list(OBJECT_CSV_HEADER)

    ob = woodchair_motion.objects[0]
    src_pos = np.asarray(ob.positions[0], dtype=np.float64)
    scaled_pos = (src_pos - np.array([0.0, 0.0, 0.12])) * 0.85
    row0 = [float(x) for x in data_rows[0]]
    assert row0[1:4] == pytest.approx(scaled_pos.tolist(), abs=1e-4)

    extents = (
        np.asarray(ob.extents, dtype=np.float64).reshape(3)
        * float(ob.scale)
        * 0.85
    )
    assert row0[8:11] == pytest.approx(extents.tolist(), abs=1e-4)
    assert len(row0) == len(OBJECT_CSV_HEADER)


def test_flat_amass_csv_writes_without_extra_stem_folder(tmp_path: Path) -> None:
    amass_dir = tmp_path / "AMASS"
    amass_dir.mkdir()
    source = amass_dir / "B10_demo.npz"
    source.touch()

    class _DummyModel:
        preset = type("P", (), {"name": "test_robot"})()

        def dof_names(self):
            return ("j1", "j2", "j3")

    ret = RetargetedMotion(
        name="B10_demo",
        joint_q=np.zeros((3, 10), dtype=np.float32),
        sample_rate=30.0,
        dof_names=("j1", "j2", "j3"),
        meta={},
    )
    out = write_retarget_export_bundle(
        ret,
        _DummyModel(),
        type("M", (), {"terrain": None, "objects": []})(),
        amass_dir,
        stem="B10_demo",
        fps=None,
        fmt="csv",
        backend="newton",
        resample_fn=lambda r, fps: (r.joint_q, r.sample_rate),
        source_path=source,
    )
    assert out == amass_dir / "B10_demo.csv"
    assert out.is_file()
    assert not (amass_dir / "B10_demo" / "B10_demo.csv").exists()


def test_flat_batch_exports_accumulate_in_shared_dir(tmp_path: Path) -> None:
    """Multiple flat clips in one folder must not delete each other's CSV."""
    lafan_dir = tmp_path / "lafan1"
    lafan_dir.mkdir()

    class _DummyModel:
        preset = type("P", (), {"name": "test_robot"})()

        def dof_names(self):
            return ("j1", "j2", "j3")

    motion = type("M", (), {"terrain": None, "objects": []})()
    for stem in ("walk1_subject1", "walk2_subject1", "walk4_subject1"):
        source = lafan_dir / f"{stem}.bvh"
        source.touch()
        ret = RetargetedMotion(
            name=stem,
            joint_q=np.zeros((3, 10), dtype=np.float32),
            sample_rate=30.0,
            dof_names=("j1", "j2", "j3"),
            meta={},
        )
        out = write_retarget_export_bundle(
            ret,
            _DummyModel(),
            motion,
            lafan_dir,
            stem=stem,
            fps=None,
            fmt="csv",
            backend="newton",
            resample_fn=lambda r, fps: (r.joint_q, r.sample_rate),
            source_path=source,
        )
        assert out == lafan_dir / f"{stem}.csv"
    assert sorted(p.name for p in lafan_dir.glob("*.csv")) == [
        "walk1_subject1.csv",
        "walk2_subject1.csv",
        "walk4_subject1.csv",
    ]


def test_export_stem_sanitization_preserves_unicode_and_removes_path_syntax() -> None:
    assert sanitize_export_stem("舞蹈_01") == "舞蹈_01"
    assert sanitize_export_stem("../outside") == "outside"
    assert sanitize_export_stem(r"C:\outside\clip") == "clip"
    assert len(sanitize_export_stem("舞" * 100).encode("utf-8")) <= 120


@pytest.mark.parametrize("stem", ["../escaped", "/tmp/absolute", r"C:\outside\clip"])
def test_flat_export_with_untrusted_stem_stays_inside_root(
    tmp_path: Path,
    stem: str,
) -> None:
    out_root = tmp_path / "exports"

    class _DummyModel:
        preset = type("P", (), {"name": "test_robot"})()

        def dof_names(self):
            return ("j1", "j2", "j3")

    ret = RetargetedMotion(
        name="clip",
        joint_q=np.zeros((3, 10), dtype=np.float32),
        sample_rate=30.0,
        dof_names=("j1", "j2", "j3"),
        meta={},
    )
    out = write_retarget_export_bundle(
        ret,
        _DummyModel(),
        type("M", (), {"terrain": None, "objects": []})(),
        out_root,
        stem=stem,
        fps=None,
        fmt="csv",
        backend="newton",
        resample_fn=lambda r, fps: (r.joint_q, r.sample_rate),
    )

    out.resolve().relative_to(out_root.resolve())
    assert out.name == f"{sanitize_export_stem(stem)}.csv"
    assert out.is_file()
    assert not (tmp_path / "escaped.csv").exists()


def test_flat_export_rejects_output_symlink_escape(tmp_path: Path) -> None:
    out_root = tmp_path / "exports"
    out_root.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("do not replace", encoding="utf-8")
    (out_root / "walk.csv").symlink_to(outside)

    class _DummyModel:
        preset = type("P", (), {"name": "test_robot"})()

        def dof_names(self):
            return ("j1", "j2", "j3")

    ret = RetargetedMotion(
        name="walk",
        joint_q=np.zeros((3, 10), dtype=np.float32),
        sample_rate=30.0,
        dof_names=("j1", "j2", "j3"),
        meta={},
    )
    with pytest.raises(ValueError, match="escapes output root"):
        write_retarget_export_bundle(
            ret,
            _DummyModel(),
            type("M", (), {"terrain": None, "objects": []})(),
            out_root,
            stem="walk",
            fps=None,
            fmt="csv",
            backend="newton",
            resample_fn=lambda r, fps: (r.joint_q, r.sample_rate),
        )

    assert outside.read_text(encoding="utf-8") == "do not replace"


def test_zip_export_sanitizes_name_and_rejects_symlink_members(tmp_path: Path) -> None:
    clip_dir = tmp_path / "clip"
    clip_dir.mkdir()
    (clip_dir / "walk.csv").write_text("trajectory", encoding="utf-8")

    archive = zip_directory(clip_dir, "../outside")
    assert archive == tmp_path / "outside.zip"
    assert archive.is_file()
    assert not (tmp_path.parent / "outside.zip").exists()

    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    (clip_dir / "linked.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="contains a symlink"):
        zip_directory(clip_dir, "linked")


def test_r2r_export_reuses_safe_stem_policy(tmp_path: Path, monkeypatch) -> None:
    out_root = tmp_path / "exports"
    source_path = tmp_path / "source.csv"
    source_path.touch()

    class _DummyModel:
        preset = type("P", (), {"name": "test_robot"})()

        def dof_names(self):
            return ("j1", "j2", "j3")

    ret = RetargetedMotion(
        name="walk",
        joint_q=np.zeros((3, 10), dtype=np.float32),
        sample_rate=30.0,
        dof_names=("j1", "j2", "j3"),
        meta={},
    )
    motion = type("M", (), {"terrain": None, "objects": []})()
    monkeypatch.setattr(
        r2r_export_bundle,
        "_bake_export_joint_q",
        lambda _model, _ret, joint_q, *_args, **_kwargs: (joint_q, 0.0),
    )

    out = r2r_export_bundle.write_r2r_export_bundle(
        ret,
        _DummyModel(),
        motion,
        scene_scale_ratio=1.0,
        entry={"source_path": str(source_path)},
        out_root=out_root,
        stem="../../机器人动作",
        fps=None,
        fmt="csv",
        resample_fn=lambda r, fps: (r.joint_q, r.sample_rate),
        yellow_foot_z=0.0,
    )

    assert out.name == "机器人动作.csv"
    out.resolve().relative_to(out_root.resolve())
    assert out.is_file()


def test_resolve_clip_export_dir_matches_source_layout() -> None:
    flat = Path("/data/mimic/AMASS/clip.npz")
    assert resolve_clip_export_dir("/out/AMASS", "clip", flat) == Path("/out/AMASS")

    folder = Path("/data/OMOMO/sub10/sub10.pkl")
    assert resolve_clip_export_dir("/out/OMOMO", "sub10", folder) == Path("/out/OMOMO/sub10")

    upload = Path("/tmp/up/parc_ms/BOXES/BOXES.pkl")
    assert resolve_clip_export_dir("/out/parc_ms/BOXES", "BOXES", upload) == Path(
        "/out/parc_ms/BOXES",
    )


def test_resolve_export_scene_params_uses_terrain_z_offset() -> None:
    from hhtools.core.scene import TerrainHeightfield

    terrain = TerrainHeightfield(
        hf=np.zeros((4, 4), dtype=np.float32),
        hf_maxmin=np.zeros((4, 4, 2), dtype=np.float32),
        min_point=np.zeros(2, dtype=np.float32),
        dx=0.1,
    )
    motion = type("M", (), {"terrain": terrain})()
    meta = {
        "smpl_scale": 0.9,
        "source_z_min": 0.2,
        "source_terrain_z_offset": 0.15,
    }
    smpl, z_sk, z_terr = _resolve_export_scene_params(meta, motion)
    assert smpl == pytest.approx(0.9)
    assert z_sk == pytest.approx(0.2)
    assert z_terr == pytest.approx(0.15)
