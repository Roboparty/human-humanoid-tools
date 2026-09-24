"""Regression tests for Agent RobotBundle identities in robot-side caches."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from hhtools.robot import foot_geometry, joint_scales, retarget_profile
from hhtools.robot.base import RobotPreset


def _preset(tmp_path, *, asset_id: str) -> RobotPreset:
    return RobotPreset(
        name="shared_robot_name",
        display_name="Shared robot name",
        root_dir=tmp_path,
        urdf_path=None,
        meta={"_agent_asset_id": asset_id},
    )


class _SceneGraph:
    def __init__(self, node: str, geom_name: str) -> None:
        self.nodes_geometry = (node,)
        self._entry = (np.eye(4, dtype=np.float64), geom_name)

    def __getitem__(self, node: str):
        assert node in self.nodes_geometry
        return self._entry


def _model(tmp_path, *, asset_id: str, node: str, vertices: np.ndarray):
    geom_name = "shared_foot_geometry"
    scene = SimpleNamespace(
        graph=_SceneGraph(node, geom_name),
        geometry={geom_name: trimesh.Trimesh(vertices=vertices, faces=())},
    )
    return SimpleNamespace(
        preset=_preset(tmp_path, asset_id=asset_id),
        urdf=SimpleNamespace(scene=scene),
    )


@pytest.fixture(autouse=True)
def _isolated_foot_geometry_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(foot_geometry, "_FOOT_MESH_NODE_CACHE", {})
    monkeypatch.setattr(foot_geometry, "_GEOM_VERTEX_CACHE", {})


def test_foot_geometry_caches_are_isolated_by_agent_asset_id(tmp_path) -> None:
    vertices_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    vertices_b = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    model_a = _model(
        tmp_path,
        asset_id="robot_asset_a",
        node="left_foot_link_asset_a",
        vertices=vertices_a,
    )
    model_b = _model(
        tmp_path,
        asset_id="robot_asset_b",
        node="left_foot_link_asset_b",
        vertices=vertices_b,
    )

    parts_a = foot_geometry._foot_mesh_node_parts(model_a, "left_foot_link")
    cached_vertices_a = foot_geometry._cached_geom_vertices(model_a, "shared_foot_geometry")
    parts_b = foot_geometry._foot_mesh_node_parts(model_b, "left_foot_link")
    cached_vertices_b = foot_geometry._cached_geom_vertices(model_b, "shared_foot_geometry")

    assert parts_a == (("left_foot_link_asset_a", "shared_foot_geometry"),)
    assert parts_b == (("left_foot_link_asset_b", "shared_foot_geometry"),)
    np.testing.assert_array_equal(cached_vertices_a, vertices_a)
    np.testing.assert_array_equal(cached_vertices_b, vertices_b)
    assert cached_vertices_a is not cached_vertices_b
    assert set(foot_geometry._FOOT_MESH_NODE_CACHE) == {
        ("robot_asset_a", "left_foot_link"),
        ("robot_asset_b", "left_foot_link"),
    }
    assert set(foot_geometry._GEOM_VERTEX_CACHE) == {
        ("robot_asset_a", "shared_foot_geometry"),
        ("robot_asset_b", "shared_foot_geometry"),
    }


def test_joint_scale_cache_key_uses_stable_agent_asset_id(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(joint_scales, "_newest_calibration_mtime", lambda _path: 0.0)
    asset_a = _preset(tmp_path, asset_id="robot_asset_a")
    same_asset_a = _preset(tmp_path, asset_id="robot_asset_a")
    asset_b = _preset(tmp_path, asset_id="robot_asset_b")

    key_a = joint_scales.scale_cache_key(asset_a)

    assert joint_scales.scale_cache_key(asset_a) == key_a
    assert joint_scales.scale_cache_key(same_asset_a) == key_a
    assert joint_scales.scale_cache_key(asset_b) != key_a


def test_agent_scaler_lookup_cannot_fall_back_to_same_named_workspace_robot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_root = tmp_path / "snapshot" / "shared_robot_name"
    workspace_root = tmp_path / "workspace" / "shared_robot_name"
    snapshot_root.mkdir(parents=True)
    workspace_root.mkdir(parents=True)
    (workspace_root / "robot.yaml").write_text(
        "retarget:\n  references:\n    smpl:\n      scaler_config: workspace-only.yaml\n",
        encoding="utf-8",
    )
    preset = _preset(snapshot_root, asset_id="robot_asset_a")
    preset.meta["retarget"] = {"references": {"smpl": {"scaler_config": "snapshot.yaml"}}}
    monkeypatch.setattr(
        retarget_profile,
        "_workspace_robot_dir",
        lambda _name: workspace_root,
    )

    assert retarget_profile._scaler_search_roots(preset) == [snapshot_root.resolve()]
    assert retarget_profile._scaler_rel_candidates(preset, "smpl") == ["snapshot.yaml"]
