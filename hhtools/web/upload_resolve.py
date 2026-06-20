# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Resolve drag-and-drop uploads into a :class:`~hhtools.core.motion.Motion`.

Three UI profiles mirror ``assets/motions`` layout:

* **intermimic** — OMOMO-style ``<clip>/<clip>.pkl`` + ``*_cleaned_simplified.obj``
* **meshmimic** — parc_ms ``<clip>/<clip>.pkl`` (+ ``*_terrain.obj``) or legacy ``.npz``
* **mimic** — AMASS / BVH / GLB / … (supports nested folders)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_log = logging.getLogger(__name__)

_MOTION_EXTS = (".bvh", ".glb", ".gltf", ".npz", ".npy", ".pkl", ".pt")
_MIMIC_PRIORITY = (".npz", ".bvh", ".glb", ".gltf", ".npy", ".pkl", ".pt")


def _is_sidecar_pkl(pkl: Path) -> bool:
    parent = pkl.parent
    stem = pkl.stem
    return any((parent / f"{stem}{ext}").is_file() for ext in (".npz", ".npy", ".bvh", ".glb", ".gltf"))


def _is_omomo_pkl(pkl: Path) -> bool:
    """OMOMO / intermimic clip — not parc_ms meshmimic."""
    parent = pkl.parent
    stem = pkl.stem
    if (parent / f"{stem}_cleaned_simplified.obj").is_file():
        return True
    return any(parent.glob("*_cleaned_simplified.obj"))


def _is_parc_ms_pkl(pkl: Path) -> bool:
    parent = pkl.parent
    stem = pkl.stem
    if (parent / f"{stem}_terrain.obj").is_file():
        return True
    return parent.name == stem


def _is_parc_ms_npz(npz: Path) -> bool:
    parent = npz.parent
    stem = npz.stem
    if (parent / f"{stem}_terrain.obj").is_file():
        return True
    return parent.name == stem


def _find_intermimic_pkls(drop_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(drop_dir.rglob("*.pkl")):
        if p.is_file() and not _is_sidecar_pkl(p):
            out.append(p)
    return out


def _find_meshmimic_primaries(drop_dir: Path) -> list[tuple[str, Path]]:
    """Return ``('npz'|'pkl', path)`` clip primaries (deduped by parent folder)."""
    by_dir: dict[Path, tuple[str, Path]] = {}
    for npz in sorted(drop_dir.rglob("*.npz")):
        if npz.is_file() and _is_parc_ms_npz(npz):
            by_dir[npz.parent] = ("npz", npz)
    for pkl in sorted(drop_dir.rglob("*.pkl")):
        if not pkl.is_file() or _is_sidecar_pkl(pkl) or _is_omomo_pkl(pkl):
            continue
        if not _is_parc_ms_pkl(pkl):
            continue
        if pkl.parent not in by_dir:
            by_dir[pkl.parent] = ("pkl", pkl)
    return list(by_dir.values())


def _find_mimic_primaries(drop_dir: Path) -> list[Path]:
    from hhtools.io.datasets.amass import is_amass_motion_file

    found: list[Path] = []
    for ext in _MIMIC_PRIORITY:
        for p in sorted(drop_dir.rglob(f"*{ext}")):
            if not p.is_file():
                continue
            if ext == ".pkl" and _is_sidecar_pkl(p):
                continue
            if ext == ".npz" and not is_amass_motion_file(p):
                continue
            found.append(p)
    # Prefer shallowest paths (clip at drop root) then alphabetical.
    found.sort(key=lambda p: (len(p.parts), str(p)))
    return found


def _pick_primary_clip(
    primaries: list[Path],
    prefer_paths: list[str] | None,
) -> Path:
    """Pick the clip the user explicitly dropped, not an arbitrary sibling."""

    if not primaries:
        raise ValueError("no primary clips")
    if not prefer_paths:
        return primaries[0]

    rels = [
        str(r).replace("\\", "/").lstrip("/")
        for r in prefer_paths
        if str(r or "").strip()
    ]
    for rel in rels:
        name = PurePosixPath(rel).name
        for p in primaries:
            if p.name == name:
                return p
        for p in primaries:
            pos = p.as_posix()
            if pos.endswith(f"/{rel}") or pos.endswith(rel):
                return p
    return primaries[0]


def _load_intermimic(pkl: Path):
    from hhtools.io.datasets.omomo import OmomoAdapter

    root = pkl.parent
    if root.name == pkl.stem:
        adapter = OmomoAdapter(root=root.parent)
        seq = f"{root.name}/{pkl.name}"
    else:
        adapter = OmomoAdapter(root=root)
        seq = pkl.name
    return adapter.load_motion(seq), "omomo"


def _load_meshmimic(kind: str, path: Path):
    if kind == "npz":
        from hhtools.io.npz import load_npz

        return load_npz(path), "parc_ms"
    from hhtools.io.datasets.parc_ms import ParcMsAdapter

    root = path.parent
    if root.name == path.stem:
        adapter = ParcMsAdapter(root=root.parent)
        seq = f"{root.name}/{path.name}"
    else:
        adapter = ParcMsAdapter(root=root)
        seq = path.name
    return adapter.load_motion(seq), "parc_ms"


def _adapter_seq(path: Path) -> tuple[object, str]:
    """Instantiate a dataset adapter for ``path`` and return ``(adapter, sequence_id)``."""
    parent = path.parent
    if parent.name == path.stem:
        seq = f"{parent.name}/{path.name}"
        root = parent.parent
    else:
        seq = path.name
        root = parent
    return root, seq


def _load_via_dataset_adapter(path: Path, dataset: str, *, progress=None):
    """Load ``path`` with the adapter implied by :func:`infer_mimic_dataset`."""
    cb = progress.as_callback() if progress is not None else None
    if cb is not None:
        cb(0.0, f"读取 {path.name}…")

    if dataset == "unified_npz":
        from hhtools.io.npz import load_npz

        return load_npz(path, progress_callback=cb), dataset

    adapter_factories: dict[str, tuple] = {
        "amass": (_amass_adapter, True),
        "phuma": (_phuma_adapter, True),
        "motion_x": (_motion_x_adapter, True),
        "meshmimic_holosoma": (
            lambda: __import__(
                "hhtools.io.datasets.meshmimic_holosoma",
                fromlist=["MeshmimicHolosomaAdapter"],
            ).MeshmimicHolosomaAdapter,
            False,
        ),
        "gvhmr": (
            lambda: __import__(
                "hhtools.io.datasets.hmr4d", fromlist=["GvhmrAdapter"]
            ).GvhmrAdapter,
            False,
        ),
        "kungfu_athlete": (
            lambda: __import__(
                "hhtools.io.datasets.hmr4d", fromlist=["KungFuAthleteAdapter"]
            ).KungFuAthleteAdapter,
            False,
        ),
    }

    if dataset not in adapter_factories:
        raise ValueError(f"no loader for dataset {dataset!r}")

    factory, with_mesh = adapter_factories[dataset]
    adapter_cls = factory()
    if adapter_cls is None:
        raise ValueError(f"adapter {dataset!r} unavailable")
    root, seq = _adapter_seq(path)
    kwargs: dict = {}
    if with_mesh:
        kwargs["with_mesh"] = True
    if cb is not None:
        kwargs["progress_callback"] = cb
    motion = adapter_cls(root=root).load_motion(seq, **kwargs)
    return motion, dataset


def _load_mimic(path: Path, load_motion_file, load_via_adapter, *, progress=None):
    from hhtools.io.mimic_detect import infer_mimic_dataset

    suf = path.suffix.lower()
    dataset = infer_mimic_dataset(path)

    if suf == ".npz":
        try:
            return _load_via_dataset_adapter(path, dataset, progress=progress)
        except Exception:
            pass
        for fallback in ("unified_npz", "amass"):
            if fallback == dataset:
                continue
            try:
                return _load_via_dataset_adapter(path, fallback, progress=progress)
            except Exception:
                continue

    if suf == ".npy":
        for candidate in (dataset, "motion_x", "phuma", "meshmimic_holosoma"):
            try:
                return _load_via_dataset_adapter(path, candidate, progress=progress)
            except Exception:
                continue

    if suf in (".pt", ".pth"):
        for candidate in (dataset, "gvhmr", "kungfu_athlete"):
            try:
                return _load_via_dataset_adapter(path, candidate, progress=progress)
            except Exception:
                continue

    try:
        motion = load_motion_file(path, progress=progress)
        if suf == ".bvh":
            return motion, infer_mimic_dataset(
                path, bone_names=motion.hierarchy.bone_names,
            )
        return motion, dataset
    except Exception:
        pass
    motion, loaded_dataset = load_via_adapter(path)
    if motion is not None:
        return motion, loaded_dataset or dataset
    raise ValueError(f"could not load {path.name}")


def _amass_adapter():
    try:
        from hhtools.io.datasets.amass import AmassAdapter

        return AmassAdapter
    except Exception:
        return None


def _phuma_adapter():
    try:
        from hhtools.io.datasets.phuma import PhumaAdapter

        return PhumaAdapter
    except Exception:
        return None


def _motion_x_adapter():
    try:
        from hhtools.io.datasets.motion_x import MotionXAdapter

        return MotionXAdapter
    except Exception:
        return None


@dataclass(frozen=True)
class UploadClipRef:
    """A single clip discovered under an upload drop directory."""

    path: Path
    profile: str
    clip_kind: str  # meshmimic: "npz"|"pkl"; otherwise ""
    dataset: str | None


def _infer_dataset_from_path(path: Path, profile: str, *, clip_kind: str = "") -> str | None:
    """Best-effort adapter name for a clip path (no I/O beyond existence checks)."""
    from hhtools.viewer.library import _DIR_TO_ADAPTER, _normalise_dirname

    profile = (profile or "mimic").strip().lower()
    if profile == "intermimic":
        return "omomo"
    if profile == "meshmimic":
        if clip_kind == "npz" or path.suffix.lower() == ".npz":
            for parent in (path.parent, path.parent.parent):
                adapter = _DIR_TO_ADAPTER.get(_normalise_dirname(parent.name))
                if adapter is not None:
                    return adapter
            return "parc_ms" if _is_parc_ms_npz(path) else "unified_npz"
        return "parc_ms"

    from hhtools.io.mimic_detect import infer_mimic_dataset

    try:
        return infer_mimic_dataset(path)
    except Exception:
        return "amass"


def detect_upload_profile(drop_dir: Path) -> str:
    """Guess intermimic / meshmimic / mimic from files under ``drop_dir``."""
    if _find_meshmimic_primaries(drop_dir):
        return "meshmimic"
    inter = _find_intermimic_pkls(drop_dir)
    if inter and any(_is_omomo_pkl(p) for p in inter):
        return "intermimic"
    if inter:
        return "intermimic"
    return "mimic"


def enumerate_upload_clips(drop_dir: Path, profile: str = "auto") -> list[UploadClipRef]:
    """List every primary clip under an upload drop (for batch basket)."""
    drop_dir = Path(drop_dir).resolve()
    profile = (profile or "auto").strip().lower()
    if profile == "auto":
        profile = detect_upload_profile(drop_dir)

    out: list[UploadClipRef] = []
    seen: set[str] = set()

    def _add(path: Path, prof: str, kind: str = "") -> None:
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        out.append(
            UploadClipRef(
                path=path,
                profile=prof,
                clip_kind=kind,
                dataset=_infer_dataset_from_path(path, prof, clip_kind=kind),
            )
        )

    if profile == "auto":
        for kind, path in _find_meshmimic_primaries(drop_dir):
            _add(path, "meshmimic", kind)
        for pkl in _find_intermimic_pkls(drop_dir):
            _add(pkl, "intermimic", "pkl")
        for path in _find_mimic_primaries(drop_dir):
            _add(path, "mimic")
        return out

    if profile == "intermimic":
        for pkl in _find_intermimic_pkls(drop_dir):
            _add(pkl, profile, "pkl")
        return out

    if profile == "meshmimic":
        for kind, path in _find_meshmimic_primaries(drop_dir):
            _add(path, profile, kind)
        return out

    for path in _find_mimic_primaries(drop_dir):
        _add(path, profile)
    return out


def load_clip_at_path(
    path: Path,
    profile: str,
    *,
    clip_kind: str = "",
    load_motion_file,
    load_via_adapter,
    progress=None,
) -> tuple[object, str | None]:
    """Load one clip by path (batch basket / preview)."""
    profile = (profile or "mimic").strip().lower()
    path = Path(path)
    if profile == "intermimic":
        return _load_intermimic(path)
    if profile == "meshmimic":
        kind = clip_kind or ("npz" if path.suffix.lower() == ".npz" else "pkl")
        return _load_meshmimic(kind, path)
    return _load_mimic(path, load_motion_file, load_via_adapter, progress=progress)


def export_subdir_for_clip(drop_dir: Path, picked: Path) -> str:
    """ZIP subfolder mirroring the drag-in tree (relative to drop root)."""
    drop_dir = Path(drop_dir).resolve()
    picked = Path(picked).resolve()
    try:
        rel = picked.relative_to(drop_dir)
        parent = rel.parent
        return parent.as_posix() if parent != Path(".") else ""
    except ValueError:
        return ""


def resolve_upload_drop(
    drop_dir: Path,
    profile: str,
    *,
    load_motion_file,
    load_via_adapter,
    progress=None,
    prefer_paths: list[str] | None = None,
) -> tuple[object, str | None, dict]:
    """Pick a primary clip under ``drop_dir`` and load it.

    Returns ``(motion, dataset, info)`` where ``info`` may contain
    ``skipped_clips`` / ``picked`` for UI toasts.

    When ``prefer_paths`` is set (browser upload relative paths), load the
    matching clip instead of the first file in alphabetical order.
    """
    profile = (profile or "mimic").strip().lower()
    info: dict = {"profile": profile}

    if profile == "intermimic":
        pkls = _find_intermimic_pkls(drop_dir)
        if not pkls:
            raise ValueError(
                "未找到 intermimic/OMOMO 风格 clip（需要 <clip>/<clip>.pkl，"
                "可连同 *_cleaned_simplified.obj 一起拖入）"
            )
        picked = _pick_primary_clip(pkls, prefer_paths)
        info["picked"] = str(picked)
        info["skipped_clips"] = max(0, len(pkls) - 1)
        motion, dataset = _load_intermimic(picked)
        return motion, dataset, info

    if profile == "meshmimic":
        clips = _find_meshmimic_primaries(drop_dir)
        if not clips:
            raise ValueError(
                "未找到 meshmimic/parc_ms 风格 clip（需要 <clip>/<clip>.pkl 或 .npz，"
                "可连同 *_terrain.obj 一起拖入）"
            )
        paths = [p for _, p in clips]
        picked_path = _pick_primary_clip(paths, prefer_paths)
        kind = next(k for k, p in clips if p == picked_path)
        info["picked"] = str(picked_path)
        info["skipped_clips"] = max(0, len(clips) - 1)
        motion, dataset = _load_meshmimic(kind, picked_path)
        return motion, dataset, info

    # mimic — any supported motion file
    primaries = _find_mimic_primaries(drop_dir)
    if not primaries:
        raise ValueError(
            "未找到可识别的动作文件（.npz / .bvh / .glb / .pkl …）"
        )
    path = _pick_primary_clip(primaries, prefer_paths)
    info["picked"] = str(path)
    info["skipped_clips"] = max(0, len(primaries) - 1)
    motion, dataset = _load_mimic(
        path, load_motion_file, load_via_adapter, progress=progress,
    )
    return motion, dataset, info
