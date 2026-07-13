# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Import PARC ``MSFileData`` pickles (``dec_release`` layout) into hhtools assets.

``dec_release`` ships ``motion_data`` + ``terrain_data`` blobs compatible with
``PARC.util.file_io.load_ms_file``.  This module reconstructs a unified-schema
:class:`~hhtools.core.motion.Motion` (NPZ) plus a triangulated ``*_terrain.obj``
and copies the source ``.pkl`` so the clip folder matches
``meshmimic/parc_ms/<clip>/`` (same layout as hand-authored parc_ms samples).

Motion **positions** are recovered by forward kinematics using **rest bone offsets**
(in each parent's local frame) taken from frame 0 of a **reference** parc_ms NPZ
with the same skeleton topology — identical to the meshmimic 15-bone rigs shipped
under ``assets/motions/meshmimic/parc_ms/``.
"""

from __future__ import annotations

import logging
import pickle
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hhtools.core.hierarchy import Hierarchy
from hhtools.core.math import quaternion as Q
from hhtools.core.motion import Motion
from hhtools.core.scene import SceneObject, TerrainHeightfield

_log = logging.getLogger(__name__)

_MOTION_DATA_KEY = "motion_data"
_TERRAIN_DATA_KEY = "terrain_data"
_MISC_DATA_KEY = "misc_data"


def load_ms_pickle_container(path: str | Path) -> tuple[dict[str, Any] | None, Any, Any]:
    """Load a top-level MS container and unpickle inner blobs.

    Returns
    -------
    motion_dict, terrain_dict, misc_dict
        Inner dicts (or ``None`` / raw ``None`` when a blob is missing / empty).
    """
    p = Path(path)
    with open(p, "rb") as f:
        container = pickle.load(f)
    if not isinstance(container, dict):
        raise ValueError(f"{p}: expected dict MS container, got {type(container)}")

    def _maybe_blob(key: str) -> Any:
        raw = container.get(key)
        if raw is None:
            return None
        return pickle.loads(raw)

    motion = _maybe_blob(_MOTION_DATA_KEY)
    terrain = _maybe_blob(_TERRAIN_DATA_KEY)
    misc = _maybe_blob(_MISC_DATA_KEY)
    return motion, terrain, misc


def rest_offsets_local_from_reference_npz(
    reference_npz: str | Path,
    *,
    frame: int = 0,
) -> tuple[list[str], NDArray[np.int32], NDArray[np.float32]]:
    """Bone names, ``parent_indices``, and per-bone rest offsets for FK.

    For each bone ``j > 0`` with parent ``p``, the offset is the vector from
    parent to child expressed in the **parent's** local frame at ``frame``:

        ``offset[j] = R(parent)^T @ (pos[j] - pos[p])``
    """
    ref = Path(reference_npz)
    with np.load(ref, allow_pickle=False) as data:
        bone_names = [str(x) for x in data["bone_names"].tolist()]
        parent_indices = np.asarray(data["parent_indices"], dtype=np.int32)
        positions = np.asarray(data["positions"], dtype=np.float64)
        quats = np.asarray(data["quaternions"], dtype=np.float64)

    t = int(np.clip(frame, 0, positions.shape[0] - 1))
    pos0 = positions[t]
    q0 = quats[t]
    jn = len(bone_names)
    offsets = np.zeros((jn, 3), dtype=np.float64)
    for j in range(1, jn):
        p = int(parent_indices[j])
        diff = pos0[j] - pos0[p]
        inv_p = Q.conjugate(Q.normalize(q0[p]))
        offsets[j] = Q.rotate(inv_p, diff)
    return bone_names, parent_indices, offsets.astype(np.float32, copy=False)


def rest_bind_pose_from_reference_npz(
    reference_npz: str | Path,
    *,
    frame: int = 0,
) -> tuple[list[str], NDArray[np.int32], NDArray[np.float32], NDArray[np.float32]]:
    """Bone names, parents, and PARC bind pose (local translation + rotation per body).

    Matches ``KinCharModel.forward_kinematics`` in the upstream PARC codebase:
    each child body carries a constant bind rotation ``local_rotation[j]`` in addition
    to the time-varying ``joint_rot[..., j-1]``.
    """
    ref = Path(reference_npz)
    with np.load(ref, allow_pickle=False) as data:
        bone_names = [str(x) for x in data["bone_names"].tolist()]
        parent_indices = np.asarray(data["parent_indices"], dtype=np.int32)
        positions = np.asarray(data["positions"], dtype=np.float64)
        quats = np.asarray(data["quaternions"], dtype=np.float64)

    t = int(np.clip(frame, 0, positions.shape[0] - 1))
    _, parents, local_trans = rest_offsets_local_from_reference_npz(ref, frame=t)
    q0 = Q.normalize(quats[t])
    jn = len(bone_names)
    local_rot = np.zeros((jn, 4), dtype=np.float64)
    local_rot[:, 3] = 1.0
    for j in range(1, jn):
        p = int(parent_indices[j])
        local_rot[j] = Q.multiply(Q.conjugate(q0[p]), q0[j])
    local_rot = Q.normalize(local_rot)
    return bone_names, parents, local_trans, local_rot.astype(np.float32, copy=False)


def _normalize_ms_joint_rot(
    joint_rot: NDArray[np.floating],
    num_bodies: int,
) -> NDArray[np.float64]:
    """Return ``(T, num_bodies - 1, 4)`` joint rotations in PARC storage order."""
    jr = np.asarray(joint_rot, dtype=np.float64)
    if jr.ndim != 3 or jr.shape[-1] != 4:
        raise ValueError(f"joint_rot must be (T, J, 4); got {jr.shape}")
    t, j, _ = jr.shape
    need = num_bodies - 1
    if j == need:
        return jr
    if j == num_bodies:
        # Common export: extra leading column (identity on the root body).
        return jr[:, :need, :]
    raise ValueError(
        f"joint_rot has {j} joints but skeleton has {num_bodies} bodies "
        f"(expected {need} or {num_bodies} rotation rows)"
    )


def fk_parc_ms(
    root_pos: NDArray[np.floating],
    root_rot: NDArray[np.floating],
    joint_rot: NDArray[np.floating],
    parent_indices: NDArray[np.integer],
    local_translation: NDArray[np.floating],
    local_rotation: NDArray[np.floating],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """PARC ``KinCharModel.forward_kinematics`` (world xyzw + positions)."""
    rp = np.asarray(root_pos, dtype=np.float64)
    rr = Q.normalize(np.asarray(root_rot, dtype=np.float64))
    parents = np.asarray(parent_indices, dtype=np.int32)
    local_trans = np.asarray(local_translation, dtype=np.float64)
    local_rot = Q.normalize(np.asarray(local_rotation, dtype=np.float64))
    num_bodies = int(parents.shape[0])
    jr = _normalize_ms_joint_rot(joint_rot, num_bodies)
    t = int(rp.shape[0])
    if rr.shape != (t, 4):
        raise ValueError(f"root_rot must be (T, 4); got {rr.shape}")
    if local_trans.shape != (num_bodies, 3):
        raise ValueError(f"local_translation must be (J, 3); got {local_trans.shape}")
    if local_rot.shape != (num_bodies, 4):
        raise ValueError(f"local_rotation must be (J, 4); got {local_rot.shape}")

    pos = np.zeros((t, num_bodies, 3), dtype=np.float64)
    world = np.zeros((t, num_bodies, 4), dtype=np.float64)
    pos[:, 0] = rp
    world[:, 0] = rr
    for j in range(1, num_bodies):
        p = int(parents[j])
        j_rot = Q.normalize(jr[:, j - 1, :])
        bind_rot = np.broadcast_to(local_rot[j], (t, 4))
        world[:, j] = Q.multiply(Q.multiply(world[:, p], bind_rot), j_rot)
        pos[:, j] = pos[:, p] + Q.rotate(world[:, p], local_trans[j])
    return pos.astype(np.float32, copy=False), Q.normalize(world).astype(np.float32, copy=False)


def ms_motion_to_world_quaternions_with_parents(
    ms_motion: Mapping[str, Any],
    parent_indices: NDArray[np.integer],
    *,
    local_translation: NDArray[np.floating] | None = None,
    local_rotation: NDArray[np.floating] | None = None,
) -> NDArray[np.float32]:
    """Rebuild per-body world xyzw quaternions from MS ``motion_data``.

    When ``local_translation`` / ``local_rotation`` are omitted, falls back to the
    legacy hhtools FK (parent world quat × joint_rot[bone], no bind rotation).
    """
    if local_translation is not None and local_rotation is not None:
        _, world = fk_parc_ms(
            ms_motion["root_pos"],
            ms_motion["root_rot"],
            ms_motion["joint_rot"],
            parent_indices,
            local_translation,
            local_rotation,
        )
        return world

    root_rot = np.asarray(ms_motion["root_rot"], dtype=np.float64)
    joint_rot = np.asarray(ms_motion["joint_rot"], dtype=np.float64)
    parents = np.asarray(parent_indices, dtype=np.int32)
    num_bodies = int(parents.shape[0])
    jr = _normalize_ms_joint_rot(joint_rot, num_bodies)
    t = int(jr.shape[0])
    if root_rot.shape[0] != t:
        raise ValueError("root_rot / joint_rot frame count mismatch")

    world = np.zeros((t, num_bodies, 4), dtype=np.float64)
    world[:, 0] = Q.normalize(root_rot)
    for bone in range(1, num_bodies):
        p = int(parents[bone])
        world[:, bone] = Q.multiply(world[:, p], jr[:, bone - 1, :])
    return Q.normalize(world).astype(np.float32, copy=False)


def fk_global_positions(
    root_pos: NDArray[np.floating],
    world_quats: NDArray[np.floating],
    parent_indices: NDArray[np.integer],
    offsets_local: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Forward kinematics: root translation + world rotations + rest offsets."""
    rp = np.asarray(root_pos, dtype=np.float64)
    wq = np.asarray(world_quats, dtype=np.float64)
    parents = np.asarray(parent_indices, dtype=np.int32)
    off = np.asarray(offsets_local, dtype=np.float64)
    t, j, _ = wq.shape
    if rp.shape != (t, 3):
        raise ValueError(f"root_pos must be (T,3); got {rp.shape}")
    if off.shape != (j, 3):
        raise ValueError(f"offsets_local must be (J,3); got {off.shape}")

    pos = np.zeros((t, j, 3), dtype=np.float64)
    pos[:, 0] = rp
    for bone in range(1, j):
        p = int(parents[bone])
        pos[:, bone] = pos[:, p] + Q.rotate(wq[:, p], off[bone])
    return pos.astype(np.float32, copy=False)


def heightfield_to_wavefront_obj(hf: TerrainHeightfield, path: str | Path) -> None:
    """Write a triangle mesh approximating ``hf`` (same grid as Viser uses)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    verts, faces = hf.triangulate()
    lines: list[str] = []
    for v in verts:
        lines.append(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}\n")
    for f in faces:
        lines.append(
            f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}\n"
        )
    out.write_text("".join(lines), encoding="utf-8")


def _skeleton_bundle_from_reference(
    reference_npz: str | Path,
) -> tuple[list[str], NDArray[np.int32], NDArray[np.float32], NDArray[np.float32]]:
    return rest_bind_pose_from_reference_npz(reference_npz)


def _unpack_skeleton_bundle(
    bundle: tuple[
        list[str],
        NDArray[np.int32],
        NDArray[np.float32],
        NDArray[np.float32] | None,
    ]
    | tuple[list[str], NDArray[np.int32], NDArray[np.float32]],
) -> tuple[list[str], NDArray[np.int32], NDArray[np.float32], NDArray[np.float32]]:
    if len(bundle) == 4:
        return bundle  # type: ignore[return-value]
    bone_names, parents, offsets = bundle
    jn = len(bone_names)
    local_rot = np.zeros((jn, 4), dtype=np.float32)
    local_rot[:, 3] = 1.0
    return bone_names, parents, offsets, local_rot


def motion_from_ms_pickle(
    src_pkl: str | Path,
    *,
    reference_npz: str | Path | None = None,
    skeleton_bundle: (
        tuple[list[str], NDArray[np.int32], NDArray[np.float32]]
        | tuple[list[str], NDArray[np.int32], NDArray[np.float32], NDArray[np.float32]]
        | None
    ) = None,
    attach_terrain_heightfield: bool = True,
) -> Motion:
    """Build a :class:`~hhtools.core.motion.Motion` from a PARC ``MSFileData`` pickle (no NPZ IO).

    Sets ``Motion.terrain`` from the MS ``terrain_data`` blob when present so the
    interaction-mesh retargeter does not require an on-disk NPZ sidecar.
    """
    src = Path(src_pkl).resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    stem = src.stem

    motion_ms, terrain_ms, misc_ms = load_ms_pickle_container(src)
    if motion_ms is None:
        raise ValueError(f"{src}: missing motion_data blob")

    if skeleton_bundle is not None:
        bone_names, parent_indices, offsets_local, local_rot = _unpack_skeleton_bundle(
            skeleton_bundle
        )
    elif reference_npz is not None:
        bone_names, parent_indices, offsets_local, local_rot = (
            _skeleton_bundle_from_reference(reference_npz)
        )
    else:
        raise ValueError("pass reference_npz or skeleton_bundle")

    root_pos = np.asarray(motion_ms["root_pos"], dtype=np.float32)
    root_rot = np.asarray(motion_ms["root_rot"], dtype=np.float32)
    joint_rot = np.asarray(motion_ms["joint_rot"], dtype=np.float32)
    # Maya / MotionBuilder clips often park root XY at the origin on frame 0
    # (bind pose) then teleport to the real start.  Drop that lead-in so
    # retarget does not crawl from the origin under the SQP trust region.
    _trim_eps = 1e-3
    _t0 = 0
    for _t in range(int(root_pos.shape[0])):
        if float(np.linalg.norm(root_pos[_t, :2])) > _trim_eps:
            _t0 = int(_t)
            break
    else:
        _t0 = int(min(1, max(0, root_pos.shape[0] - 1)))
    if _t0 > 0:
        root_pos = root_pos[_t0:]
        root_rot = root_rot[_t0:]
        joint_rot = joint_rot[_t0:]
    positions, world_q = fk_parc_ms(
        root_pos,
        root_rot,
        joint_rot,
        parent_indices,
        offsets_local,
        local_rot,
    )

    fps = float(np.asarray(motion_ms["fps"]).item())
    loop_mode = str(motion_ms.get("loop_mode", "CLAMP"))

    meta: dict[str, Any] = {
        "dataset": "parc_ms",
        "source_format": "parc_ms_pkl",
        "source_pkl": str(src),
        "loop_mode": loop_mode,
        "misc_data_present": misc_ms is not None,
    }
    if _t0 > 0:
        meta["trimmed_leading_bind_frames"] = int(_t0)
    if isinstance(misc_ms, dict):
        for key in (
            "library_folder_label",
            "mocap_source_take_dir",
            "source_skeleton_bvh",
            "terrain_heightfield_frame",
        ):
            if key in misc_ms and misc_ms[key] is not None:
                meta[key] = misc_ms[key]

    hierarchy = Hierarchy.from_parent_indices(bone_names, parent_indices)

    terrain_hf: TerrainHeightfield | None = None
    if attach_terrain_heightfield and terrain_ms is not None:
        terrain_hf = TerrainHeightfield.from_ms_terrain_data_dict(terrain_ms)

    return Motion(
        name=stem,
        hierarchy=hierarchy,
        positions=positions,
        quaternions=world_q,
        framerate=fps,
        up_axis="Z",
        source_format="npz",
        meta=meta,
        objects=[],
        terrain=terrain_hf,
    )


def export_ms_pkl_to_meshmimic_clip_dir(
    src_pkl: str | Path,
    out_parent: str | Path,
    *,
    reference_npz: str | Path,
    copy_source_pkl: bool = True,
    overwrite: bool = False,
) -> Path:
    """Convert one ``dec_release`` / MS pickle into ``parc_ms/<stem>/`` bundle.

    Writes:

    * ``out_parent/<stem>/<stem>.npz`` — unified schema (objects_* reference terrain OBJ).
    * ``out_parent/<stem>/<stem>_terrain.obj`` — triangulated heightfield.
    * ``out_parent/<stem>/<stem>.pkl`` — copy of ``src_pkl`` when ``copy_source_pkl``.

    Parameters
    ----------
    src_pkl
        Path to the MS ``MSFileData`` pickle.
    out_parent
        Typically ``assets/motions/meshmimic/parc_ms`` — **not** the per-clip folder.
    reference_npz
        Existing parc_ms NPZ with the same bone count / topology (e.g. BOXES_7_1).
    """
    src = Path(src_pkl).resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    stem = src.stem
    clip_dir = Path(out_parent) / stem
    if clip_dir.is_dir():
        if not overwrite and any(clip_dir.iterdir()):
            raise FileExistsError(
                f"{clip_dir} exists and is non-empty; pass overwrite=True to replace"
            )
        if overwrite:
            shutil.rmtree(clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)

    motion_ms, terrain_ms, misc_ms = load_ms_pickle_container(src)
    if motion_ms is None:
        raise ValueError(f"{src}: missing motion_data blob")

    bone_names, parent_indices, offsets_local, local_rot = (
        rest_bind_pose_from_reference_npz(reference_npz)
    )

    root_pos = np.asarray(motion_ms["root_pos"], dtype=np.float32)
    root_rot = np.asarray(motion_ms["root_rot"], dtype=np.float32)
    joint_rot = np.asarray(motion_ms["joint_rot"], dtype=np.float32)
    positions, world_q = fk_parc_ms(
        root_pos,
        root_rot,
        joint_rot,
        parent_indices,
        offsets_local,
        local_rot,
    )

    fps = float(np.asarray(motion_ms["fps"]).item())
    loop_mode = str(motion_ms.get("loop_mode", "CLAMP"))

    meta: dict[str, Any] = {
        "dataset": "parc_ms",
        "source_format": "parc_ms_pkl",
        "source_pkl": str(src),
        "loop_mode": loop_mode,
        "misc_data_present": misc_ms is not None,
    }

    hierarchy = Hierarchy.from_parent_indices(bone_names, parent_indices)
    num_frames = positions.shape[0]
    z0 = np.zeros((num_frames, 3), dtype=np.float32)
    id_quat = np.zeros((num_frames, 4), dtype=np.float32)
    id_quat[:, 3] = 1.0

    terrain_obj_rel = f"{stem}_terrain.obj"
    terrain_obj = SceneObject(
        name="terrain",
        positions=z0,
        quaternions=id_quat,
        extents=np.array([80.0, 80.0, 2.0], dtype=np.float32),
        mesh_path=terrain_obj_rel,
        scale=1.0,
        opacity=1.0,
        color=(140, 140, 145),
    )

    mot = Motion(
        name=stem,
        hierarchy=hierarchy,
        positions=positions,
        quaternions=world_q,
        framerate=fps,
        up_axis="Z",
        source_format="npz",
        meta=meta,
        objects=[terrain_obj],
        terrain=None,
    )

    npz_path = clip_dir / f"{stem}.npz"
    from hhtools.io.npz import save_npz

    save_npz(mot, npz_path, compressed=True)

    if terrain_ms is None:
        _log.warning("%s: no terrain_data — writing flat placeholder OBJ", src)
        hf = TerrainHeightfield(
            hf=np.zeros((2, 2), dtype=np.float32),
            hf_maxmin=np.zeros((2, 2, 2), dtype=np.float32),
            min_point=np.array([0.0, 0.0], dtype=np.float32),
            dx=1.0,
        )
    else:
        hf = TerrainHeightfield.from_ms_terrain_data_dict(terrain_ms)

    obj_path = clip_dir / f"{stem}_terrain.obj"
    heightfield_to_wavefront_obj(hf, obj_path)

    if copy_source_pkl:
        shutil.copy2(src, clip_dir / f"{stem}.pkl")

    return clip_dir


def export_ms_pkl_to_parc_ms_clip_dir(
    src_pkl: str | Path,
    out_parent: str | Path,
    *,
    clip_name: str | None = None,
    copy_source_pkl: bool = True,
    overwrite: bool = False,
) -> Path:
    """Convert one ``dec_release`` MS pickle into the lean ``parc_ms`` layout.

    Writes only the artefacts shipped by hand-authored parc_ms demos:

    * ``out_parent/<clip_name>/<clip_name>.pkl`` — MS ``MSFileData`` (copy of source).
    * ``out_parent/<clip_name>/<clip_name>_terrain.obj`` — triangulated heightfield.

    No ``.npz`` is emitted; :class:`~hhtools.io.datasets.parc_ms.ParcMsAdapter`
    reconstructs skeleton poses from ``motion_data`` via FK.
    """
    src = Path(src_pkl).resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    name = str(clip_name or src.stem)
    clip_dir = Path(out_parent) / name
    pkl_out = clip_dir / f"{name}.pkl"
    obj_out = clip_dir / f"{name}_terrain.obj"

    motion_ms, terrain_ms, _misc_ms = load_ms_pickle_container(src)
    if motion_ms is None:
        raise ValueError(f"{src}: missing motion_data blob")

    if clip_dir.is_dir() and not overwrite:
        if pkl_out.is_file() and obj_out.is_file():
            return clip_dir
        if any(clip_dir.iterdir()):
            raise FileExistsError(
                f"{clip_dir} exists and is incomplete; pass overwrite=True to replace"
            )

    if overwrite and clip_dir.is_dir():
        shutil.rmtree(clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)

    if terrain_ms is None:
        _log.warning("%s: no terrain_data — writing flat placeholder OBJ", src)
        hf = TerrainHeightfield(
            hf=np.zeros((2, 2), dtype=np.float32),
            hf_maxmin=np.zeros((2, 2, 2), dtype=np.float32),
            min_point=np.array([0.0, 0.0], dtype=np.float32),
            dx=1.0,
        )
    else:
        hf = TerrainHeightfield.from_ms_terrain_data_dict(terrain_ms)

    heightfield_to_wavefront_obj(hf, obj_out)

    if copy_source_pkl:
        shutil.copy2(src, pkl_out)
    else:
        with open(pkl_out, "wb") as f:
            pickle.dump(
                {
                    _MOTION_DATA_KEY: pickle.dumps(motion_ms),
                    _TERRAIN_DATA_KEY: (
                        None if terrain_ms is None else pickle.dumps(terrain_ms)
                    ),
                    _MISC_DATA_KEY: None,
                },
                f,
            )

    return clip_dir


__all__ = [
    "export_ms_pkl_to_meshmimic_clip_dir",
    "export_ms_pkl_to_parc_ms_clip_dir",
    "fk_global_positions",
    "fk_parc_ms",
    "rest_bind_pose_from_reference_npz",
    "heightfield_to_wavefront_obj",
    "load_ms_pickle_container",
    "motion_from_ms_pickle",
    "ms_motion_to_world_quaternions_with_parents",
    "rest_offsets_local_from_reference_npz",
]
