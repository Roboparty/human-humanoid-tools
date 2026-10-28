"""Install the public RP1 layout without bundling upstream model files in tests."""

from __future__ import annotations

import io
import json
import shutil
import struct
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from scripts import install_builtin_robots as installer


@pytest.fixture
def rp1_archive(tmp_path: Path) -> tuple[installer.RobotSource, Path, bytes, bytes]:
    source = next(item for item in installer.SOURCES if item.name == "rp1")
    robot = ET.Element("robot", name="rp1")
    parent = "base_link"
    ET.SubElement(robot, "link", name=parent)
    # Synthetic 28-DoF model uses the official archive's URDF/relative-mesh layout.
    for index in range(28):
        child = f"link_{index}"
        link = ET.SubElement(robot, "link", name=child)
        for kind in ("visual", "collision"):
            geometry = ET.SubElement(ET.SubElement(link, kind), "geometry")
            ET.SubElement(geometry, "mesh", filename="../meshes/shared.STL")
        joint = ET.SubElement(robot, "joint", name=f"joint_{index}", type="revolute")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(joint, "origin", xyz="0 0 0.1", rpy="0 0 0")
        ET.SubElement(joint, "axis", xyz="0 1 0")
        ET.SubElement(joint, "limit", lower="-1.5", upper="1.5", effort="10", velocity="2")
        parent = child
    urdf = ET.tostring(robot, encoding="utf-8")
    # One valid binary STL triangle; mesh decoding is covered by the real-source smoke check.
    mesh = bytes(80) + struct.pack("<I12fH", 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)
    archive_path = tmp_path / "rp1.tar.gz"
    files = {
        source.main_urdf: urdf,
        "meshes/shared.STL": mesh,
        "urdf/rp1_24dof.urdf": b"not the selected model",
        "meshes/unused.STL": b"not referenced by the selected model",
    }
    with tarfile.open(archive_path, "w:gz") as archive:
        for relative, data in files.items():
            info = tarfile.TarInfo(f"rp1_description-{source.commit}/{relative}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return source, archive_path, urdf, mesh


def _use_archive(monkeypatch: pytest.MonkeyPatch, archive: Path) -> list[str]:
    downloads: list[str] = []

    def download(source: installer.RobotSource, output: Path) -> int:
        downloads.append(source.commit)
        shutil.copyfile(archive, output)
        return output.stat().st_size

    monkeypatch.setattr(installer, "_download_archive", download)
    return downloads


def test_install_rp1_preserves_dofs_and_records_actual_public_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rp1_archive: tuple[installer.RobotSource, Path, bytes, bytes],
) -> None:
    source, archive, source_urdf, mesh = rp1_archive
    downloads = _use_archive(monkeypatch, archive)
    library = tmp_path / "robots"
    library.mkdir()

    installer._install(source, library, replace=False)

    installed = library / "rp1"
    preset = yaml.safe_load((installed / "robot.yaml").read_text())
    assert preset["name"] == "rp1"
    assert preset["display_name"] == "ROBOTO 01"
    assert preset["dof_order"] == [f"joint_{index}" for index in range(28)]
    model = ET.parse(installed / "rp1.urdf").getroot()
    assert len(model.findall("link")) == 29
    assert [ET.tostring(joint) for joint in model.findall("joint")] == [
        ET.tostring(joint) for joint in ET.fromstring(source_urdf).findall("joint")
    ]
    assert {element.get("filename") for element in model.iter("mesh")} == {"meshes/shared.STL"}
    assert (installed / "meshes" / "shared.STL").read_bytes() == mesh
    assert not (installed / "rp1_24dof.urdf").exists()
    assert not (installed / "meshes" / "unused.STL").exists()
    manifest = json.loads((installed / "SOURCE.json").read_text())
    assert manifest["source"]["repository"] == "https://github.com/Roboparty/rp1_description.git"
    assert manifest["source"]["commit"] == source.commit
    assert manifest["source"]["main_urdf"] == "urdf/rp1.urdf"
    assert manifest["source"]["license_paths"] == []
    assert {record["source_path"] for record in manifest["source_files"]} == {
        "urdf/rp1.urdf", "meshes/shared.STL",
    }
    assert installer._is_up_to_date(source, installed)
    assert installer._install(source, library, replace=False) == "up to date: rp1"
    assert downloads == [source.commit]


def test_install_rp1_does_not_replace_user_changes_without_explicit_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rp1_archive: tuple[installer.RobotSource, Path, bytes, bytes],
) -> None:
    source, archive, _, _ = rp1_archive
    _use_archive(monkeypatch, archive)
    library = tmp_path / "robots"
    library.mkdir()
    installer._install(source, library, replace=False)
    preset = library / "rp1" / "robot.yaml"
    modified = preset.read_bytes() + b"\n# User calibration and mapping edits\n"
    preset.write_bytes(modified)

    with pytest.raises(FileExistsError, match="pass --replace"):
        installer._install(source, library, replace=False)

    assert preset.read_bytes() == modified
