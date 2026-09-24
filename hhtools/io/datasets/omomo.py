"""OMOMO dataset adapter.

OMOMO (Li, Wu & Liu, SIGGRAPH Asia 2023) pairs a human with a rigid object the
subject manipulates. The official release (https://github.com/lijiaman/omomo_release)
ships two relevant pickles:

* ``train_diffusion_manip_seq_joints24.p`` — raw long SMPL-H sequences.
* ``train_diffusion_manip_window_120_cano_joints24.p`` — pre-processed 120-frame
  windows, each canonicalised so the first frame sits at a reference orientation.

We target the *window* format since it is the one most users actually have: the
master file contains ~16k overlapping windows. To keep the demo tree manageable
we ship a single small ``.pkl`` per demo window (see
``scripts/extract_omomo_demos.py``), each containing just the fields the viewer
needs:

    seq_name        str                            e.g. "sub10_largebox_000"
    gender          str                            "male" | "female"
    betas           (1, 16) float32                SMPL-H body shape
    trans2joint     (3,)    float32                root→pelvis offset
    motion          (T, 276) float32               packed features
    obj_trans       (T, 3)   float32/float64       world position of the mesh's
                                                   *raw* local origin (NOT the
                                                   geometric centre — see below)
    obj_rot_mat     (T, 3, 3) float32              object rotation matrix
    obj_scale       (T,)    float32                mesh uniform scale

OMOMO's geometry contract is:

    world_vert = obj_scale[t] * (obj_rot_mat[t] · raw_vert) + obj_trans[t]

i.e. ``obj_trans`` is the **raw mesh origin's** world position, not the
mesh centroid.  Some OMOMO captures ship a mesh whose raw vertices are
already centred (``largebox`` raw centroid ≈ origin) and some don't
(``woodchair`` raw centroid sits at +(16, -9, +16) cm).  The latter
make ``obj_trans`` look "off by ~scale·centroid" when interpreted as a
generic object world position — for ``woodchair`` that's a 0.6 m offset
and the chair appears nowhere near the actor's hands even though the
upstream geometry is internally consistent.

To present a consistent contract downstream (``SceneObject.positions[t]
== world position of the mesh's geometric centre``, which is what both
the Web renderer and the interaction-mesh / MPC retargeter assume),
the adapter folds the centroid offset into ``positions`` at load time
and pairs that with :func:`hhtools.io.scene_serialize.object_mesh_glb`, which
centres the loaded mesh on its raw centroid before scaling. The
two steps together preserve the upstream world-vertex formula bit-for-bit
while giving ``SceneObject.positions`` an unambiguous "object centre"
meaning.

Layout of the 276-dim ``motion`` vector, cross-checked against
``manip/data/hand_foot_dataset.py::cal_normalize_data_input`` in the upstream
repository:

    [0  .. 72)   = global joint positions   (24 joints × 3)
    [72 .. 144)  = global joint velocities  (24 joints × 3)
    [144 .. 276) = global joint rotations   (22 body joints × 6D)

We only need the first 72 dims for skeleton / capsule rendering; rotations can be
revisited when we add SMPL-H mesh forward. Global quaternions are set to identity
here — every downstream consumer (SkeletonRenderer, CapsuleMeshRenderer,
analytics) uses positions only, so this loses no information for visualisation.

The 24 joints follow OMOMO's ``use_joints24=True`` convention: the first 22 are
the standard SMPL-H body joints, and the last two are ``l_index1`` / ``r_index1``
(the first knuckle of each hand). Parent indices are taken directly from the
SMPL-H ``kintree_table`` so the skeleton connects cleanly.

Object mesh rendering: the upstream ``captured_objects/*.obj`` files are not
redistributable with this repo, but if the user drops the matching
``<object_name>_cleaned_simplified.obj`` **next to the ``.pkl``** (so each clip
directory is self-contained — mirroring the layout of ``meshmimic`` terrain
clips) the adapter fills :attr:`SceneObject.mesh_path` and
:attr:`SceneObject.scale`, and the viewer renders the real mesh instead of the
placeholder cuboid. When no mesh is present we fall back to a per-name cuboid
placeholder (see :data:`_OBJECT_EXTENTS_METRES`).
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from hhtools.core.hierarchy import Hierarchy
from hhtools.core.motion import Motion
from hhtools.core.scene import SceneObject
from hhtools.io.datasets.base import DatasetAdapter, register_dataset

# ----------------------------------------------------------------- constants

# Official OMOMO 24-joint parents (see get_smpl_parents(use_joints24=True) upstream).
# Joints 0..21 are the standard SMPL-H body; joints 22, 23 are l_index1 / r_index1
# (the hand-tip proxy), both parented to their respective wrists (20, 21).
_OMOMO_PARENTS: tuple[int, ...] = (
    -1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,
     9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
)
_OMOMO_NAMES: tuple[str, ...] = (
    "pelvis",
    "l_hip", "r_hip", "spine1",
    "l_knee", "r_knee", "spine2",
    "l_ankle", "r_ankle", "spine3",
    "l_foot", "r_foot", "neck",
    "l_collar", "r_collar", "head",
    "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow",
    "l_wrist", "r_wrist",
    "l_hand", "r_hand",
)

_NUM_JOINTS = 24
_MOTION_FEATURE_DIM = 276
_JPOS_DIMS = _NUM_JOINTS * 3  # 72 — first 72 dims of `motion`

# Known object names → coarse cuboid extents (width, depth, height) in metres. These
# are *rough* visual placeholders — OMOMO's captured mesh archive has the
# authoritative geometry. Extents live here (rather than a YAML config) because
# they ship with the adapter and need no extra file to work out-of-the-box.
_OBJECT_EXTENTS_METRES: dict[str, tuple[float, float, float]] = {
    "largebox": (0.60, 0.40, 0.35),
    "smallbox": (0.35, 0.25, 0.20),
    "plasticbox": (0.45, 0.32, 0.28),
    "trashcan": (0.35, 0.35, 0.55),
    "largetable": (1.20, 0.70, 0.75),
    "smalltable": (0.60, 0.40, 0.50),
    "woodchair": (0.50, 0.50, 0.90),
    "whitechair": (0.50, 0.50, 0.90),
    "suitcase": (0.55, 0.35, 0.20),
    "monitor": (0.55, 0.18, 0.40),
    "tripod": (0.25, 0.25, 1.20),
    "clothesstand": (0.45, 0.45, 1.60),
    "floorlamp": (0.30, 0.30, 1.50),
    "mop": (0.25, 0.25, 1.30),
    "vacuum": (0.25, 0.25, 1.10),
}
_DEFAULT_OBJECT_EXTENTS = (0.40, 0.30, 0.30)

# OMOMO captures at 30 Hz (confirmed in the upstream paper and loader).
_DEFAULT_FRAMERATE = 30.0


# ----------------------------------------------------------------- helpers


def _object_name_from_stem(stem: str) -> str:
    """``sub10_largebox_000`` → ``largebox``.

    File names follow ``sub<subject-id>_<object-name>_<clip-id>``; we split on the
    underscore and pick the second token. Robust to object names that themselves
    contain digits (``sub3_clothesstand_002`` still yields ``clothesstand``).
    """
    parts = stem.split("_")
    return parts[1] if len(parts) >= 2 else "object"


def _lookup_extents(object_name: str) -> NDArray:
    extents = _OBJECT_EXTENTS_METRES.get(object_name.lower(), _DEFAULT_OBJECT_EXTENTS)
    return np.asarray(extents, dtype=np.float32)


def _load_window_pickle(path: Path) -> dict[str, Any]:
    """Open a demo ``.pkl`` produced by ``scripts/extract_omomo_demos.py``.

    We use plain ``pickle`` rather than ``joblib`` so the adapter works without an
    extra dep at load time (the extractor script pickles with the highest
    protocol, which stock ``pickle`` on Python ≥3.8 reads fine).
    """
    with open(path, "rb") as fh:
        window = pickle.load(fh)
    if not isinstance(window, dict):
        raise ValueError(
            f"Expected a dict in {path}; got {type(window).__name__}. Re-run "
            "scripts/extract_omomo_demos.py to regenerate the demo file."
        )
    return window


def _extract_joint_positions(motion_feat: NDArray) -> NDArray:
    """Pull the 24-joint global positions out of the 276-dim ``motion`` vector."""
    m = np.asarray(motion_feat, dtype=np.float32)
    if m.ndim != 2 or m.shape[1] != _MOTION_FEATURE_DIM:
        raise ValueError(
            f"OMOMO 'motion' must be (T, {_MOTION_FEATURE_DIM}); got {m.shape}"
        )
    return m[:, :_JPOS_DIMS].reshape(m.shape[0], _NUM_JOINTS, 3).astype(np.float32)


def _rotmat_to_xyzw_quat(rot_mats: NDArray) -> NDArray:
    """Convert ``(T, 3, 3)`` rotation matrices to ``(T, 4)`` xyzw quaternions.

    ``scipy.spatial.transform.Rotation.as_quat`` already returns xyzw, which
    matches our internal convention. Input is cast to float32 up front so the
    output array stays in a single dtype through the pipeline.
    """
    mats = np.asarray(rot_mats, dtype=np.float32)
    if mats.ndim != 3 or mats.shape[1:] != (3, 3):
        raise ValueError(f"Expected (T, 3, 3) rotation matrices; got {mats.shape}")
    quats = R.from_matrix(mats).as_quat()  # xyzw
    return quats.astype(np.float32)


def _find_captured_mesh(clip_dir: Path, object_name: str) -> str:
    """Return the absolute path to ``<object_name>_cleaned_simplified.obj`` if it
    sits next to the pickle, else ``""``.

    OMOMO's ``captured_objects`` archive uses the ``_cleaned_simplified.obj``
    suffix (confirmed against ``manip/data/hand_foot_dataset.py`` upstream).
    Clips of articulated props (``mop``, ``vacuum``) split the mesh into
    ``_top`` / ``_bottom`` parts; we ignore those for now and fall back to the
    cuboid placeholder — the viewer doesn't animate sub-parts yet.
    """
    candidate = clip_dir / f"{object_name}_cleaned_simplified.obj"
    if candidate.is_file():
        return str(candidate.resolve())
    return ""


def _mesh_raw_centroid(mesh_path: str) -> NDArray | None:
    """Read the raw mesh's geometric centroid in its native coordinate frame.

    Used to fold OMOMO's "raw mesh origin → world" translation contract
    into a "geometric centre → world" contract that downstream
    consumers (Web renderer, interaction-mesh / MPC retargeter)
    expect.  Returns ``None`` when the mesh is unavailable or
    unreadable; callers then fall back to the raw ``obj_trans`` and
    accept the (small) cuboid-placeholder offset.

    Trimesh is loaded lazily so the adapter still works in stripped-
    down sandboxes that haven't installed it.
    """
    if not mesh_path:
        return None
    try:
        import trimesh  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None
    try:
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
    except Exception:
        return None
    verts = np.asarray(getattr(mesh, "vertices", np.zeros((0, 3))), dtype=np.float64)
    if verts.size == 0:
        return None
    return verts.mean(axis=0).astype(np.float32)


def _extract_uniform_scale(obj_scale: Any) -> float:
    """Collapse OMOMO's per-frame ``obj_scale`` array to a single float.

    OMOMO records ``obj_scale`` as ``(T,)`` but within a window it's effectively
    constant (the capture rig doesn't animate the calibration factor). We take
    the frame-0 value and fall back to 1.0 if the field is missing or empty —
    consistent with the official loader's
    ``obj_scale[:, None, None] * verts + trans`` pattern at frame 0.
    """
    if obj_scale is None:
        return 1.0
    arr = np.asarray(obj_scale, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 1.0
    return float(arr[0])


def _build_scene_object(
    window: dict[str, Any],
    *,
    clip_dir: Path,
    object_extents: tuple[float, float, float] | None,
) -> SceneObject:
    """Assemble the object track from a window dict, with optional mesh path.

    OMOMO's raw ``obj_trans`` is the world position of the mesh's
    *raw local origin* — not its geometric centre.  When the mesh ships
    with a non-centred raw frame (e.g. ``woodchair`` raw centroid at
    ~25 cm from origin in cm-units, becoming ~62 cm of offset after
    ``obj_scale``), naive use of ``obj_trans`` drifts the rendered /
    retargeted object away from the actor's hands.  We fold the mesh
    centroid offset into ``positions`` here so :class:`SceneObject` carries
    a single, unambiguous "object centre in world" trajectory:

        positions[t] = obj_trans[t] + obj_scale · (R_obj[t] · raw_centroid)

    Pair with :func:`hhtools.io.scene_serialize.object_mesh_glb`, which
    centres the loaded mesh on the same raw centroid before
    scaling — together they preserve OMOMO's upstream world-vertex
    formula bit-for-bit while making downstream consumers (Web
    renderer, IM / MPC retargeter) see object positions that actually
    align with the actor's hands.

    When the captured mesh is not co-located with the pickle we fall
    back to ``obj_trans`` directly; the cuboid placeholder is then
    centred on the raw origin (small offset for most props, only
    visible on outliers like ``woodchair``) — fixing that requires
    either downloading the mesh or extending the catalog with per-
    object ``raw_centroid`` measurements.
    """
    obj_trans = np.asarray(window["obj_trans"], dtype=np.float32)
    if obj_trans.ndim != 2 or obj_trans.shape[1] != 3:
        raise ValueError(f"obj_trans must be (T, 3); got {obj_trans.shape}")

    obj_quat = _rotmat_to_xyzw_quat(window["obj_rot_mat"])
    if obj_quat.shape[0] != obj_trans.shape[0]:
        raise ValueError(
            f"obj_trans and obj_rot_mat disagree on frame count: "
            f"{obj_trans.shape[0]} vs {obj_quat.shape[0]}"
        )

    object_name = _object_name_from_stem(str(window.get("seq_name", "")))
    extents_arr = (
        np.asarray(object_extents, dtype=np.float32)
        if object_extents is not None
        else _lookup_extents(object_name)
    )

    mesh_path = _find_captured_mesh(clip_dir, object_name)
    scale = _extract_uniform_scale(window.get("obj_scale"))

    raw_centroid = _mesh_raw_centroid(mesh_path) if mesh_path else None
    if raw_centroid is not None:
        # Per-frame world centroid: obj_trans[t] + scale · R[t] · raw_centroid
        rot_mats = np.asarray(window["obj_rot_mat"], dtype=np.float32)
        rotated = np.einsum("tij,j->ti", rot_mats, raw_centroid)
        positions = (obj_trans + np.float32(scale) * rotated).astype(np.float32)
    else:
        positions = obj_trans

    return SceneObject(
        name=object_name,
        positions=positions,
        quaternions=obj_quat,
        extents=extents_arr,
        mesh_path=mesh_path,
        scale=scale,
    )


# ----------------------------------------------------------------- adapter


@register_dataset
class OmomoAdapter(DatasetAdapter):
    """Adapter for pre-extracted OMOMO ``.pkl`` demo windows.

    Each file is a single 120-frame window plus the interacting object's 6-DoF
    track. ``scripts/extract_omomo_demos.py`` produces these from the upstream
    master pickle so users never need to redistribute the 2 GB original.
    """

    name = "omomo"
    display_name = "OMOMO"
    requires = "skeleton"
    file_patterns = ("*.pkl",)

    def list_sequences(self) -> Iterator[str]:
        if not self.root.exists():
            return
        for p in sorted(self.root.rglob("*.pkl")):
            if p.is_file():
                yield str(p.relative_to(self.root))

    def _resolve(self, sequence_id: str) -> Path:
        p = (self.root / sequence_id).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"OMOMO sequence not found: {p}")
        return p

    def load_motion(
        self,
        sequence_id: str,
        *,
        framerate: float | None = None,
        object_extents: tuple[float, float, float] | None = None,
        **_kwargs: Any,
    ) -> Motion:
        path = self._resolve(sequence_id)
        window = _load_window_pickle(path)

        positions = _extract_joint_positions(window["motion"])  # (T, 24, 3) Z-up metres
        num_frames = int(positions.shape[0])

        # Global joint quaternions are not reconstructed here; keeping them at
        # identity keeps SkeletonRenderer / CapsuleMeshRenderer happy without a
        # full SMPL forward pass. SMPL-H mesh support can populate these later.
        quaternions = np.zeros((num_frames, _NUM_JOINTS, 4), dtype=np.float32)
        quaternions[..., 3] = 1.0  # xyzw identity

        hierarchy = Hierarchy.from_parent_indices(
            list(_OMOMO_NAMES), np.asarray(_OMOMO_PARENTS)
        )

        scene_obj = _build_scene_object(
            window, clip_dir=path.parent, object_extents=object_extents
        )

        return Motion(
            name=Path(sequence_id).stem,
            hierarchy=hierarchy,
            positions=positions,
            quaternions=quaternions,
            framerate=_DEFAULT_FRAMERATE if framerate is None else float(framerate),
            up_axis="Z",
            source_format="smplh",
            meta={
                "dataset": "omomo",
                "sequence_id": sequence_id,
                "seq_name": str(window.get("seq_name", "")),
                "object_name": scene_obj.name,
                "gender": str(window.get("gender", "")),
                "source_repo": "https://github.com/lijiaman/omomo_release",
                "notes": (
                    "Joint positions decoded from OMOMO 276-dim window feature "
                    "(first 72 dims). Global rotations left at identity — "
                    "skeleton/capsule rendering only needs positions. Object mesh "
                    "is a placeholder cuboid; drop captured_objects/<name>.obj "
                    "alongside for a future mesh-aware iteration."
                ),
            },
            objects=[scene_obj],
        )


__all__ = [
    "OmomoAdapter",
    "_build_scene_object",
    "_extract_joint_positions",
    "_extract_uniform_scale",
    "_find_captured_mesh",
    "_object_name_from_stem",
    "_rotmat_to_xyzw_quat",
]
