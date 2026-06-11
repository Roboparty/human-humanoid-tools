"""URDF normalisation helpers for vendor drops with inconsistent mesh paths.

MuJoCo resolves ``<mesh filename="meshes/foo.stl"/>`` relative to the URDF
directory (and may also honour ``<compiler meshdir="meshes"/>``).  Vendor
URDFs often reference bare ``link.STL`` while meshes live under ``meshes/``,
or combine ``meshdir`` with filenames that already include ``meshes/``.  This
module rewrites mesh paths in-place so :func:`hhtools.robot.loader.load_robot`
and MuJoCo compilation succeed without manual editing.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

__all__ = [
    "detect_mesh_path_issues",
    "detect_redundant_floating_base_root",
    "detect_urdf_inertial_issues",
    "ensure_mujoco_compiler_block",
    "ensure_urdf_meshes_resolvable",
    "normalize_urdf_meshes",
    "repair_urdf_mesh_paths",
    "repair_urdf_xml_structure",
    "robot_upload_destination",
    "strip_redundant_floating_base_root",
    "urdf_needs_persisted_fixes",
    "writable_urdf_copy",
]

# Names commonly used for the *fixed inertial frame* anchor that SolidWorks /
# ROS exporters bolt on top of the real base link.  hhtools adds its own
# floating base, so any of these as the URDF root is redundant.
_DUMMY_ROOT_LINK_NAMES = frozenset(
    {"world", "base_footprint", "odom", "map", "dummy_root"}
)
# Only a ``floating`` anchor is harmful: ``add_urdf(floating=True)`` already adds
# a 6-DoF root, so a second ``type="floating"`` joint stacks a *duplicate*
# floating base (the retargeter drives one, the other drifts → robot floats).
#
# A ``fixed`` anchor (``dummy_link``/``world`` welded to the base) adds **zero**
# DoF, so it yields a single, correct floating base — stripping it would only
# rename the root body and could break ik_map slots that legitimately point at
# the dummy root (e.g. Kuavo ``spine: dummy_link``).  We therefore leave ``fixed``
# anchors untouched so existing, working robots are never disturbed.
_ANCHOR_JOINT_TYPES = frozenset({"floating"})

# MuJoCo URDF import decodes STL/OBJ reliably; Collada (.dae) is common in GMR /
# Unitree drops but must be converted at ingest.
_MUJOCO_NATIVE_MESH_EXTS = frozenset({".stl", ".obj"})
_CONVERT_TO_STL_EXTS = frozenset({".dae"})

_INERTIA_TRIANGLE_TOL = 1e-9


def _mesh_filename_prefix(filename: str) -> str | None:
    """Return a normalised directory prefix from a mesh filename, if any."""
    fname = _strip_package_uri(filename)
    if fname.startswith("./"):
        fname = fname[2:]
    if "/" not in fname:
        return None
    return fname.rsplit("/", 1)[0]


def _strip_package_uri(filename: str) -> str:
    fname = filename.replace("\\", "/").strip()
    if fname.startswith("package://"):
        rest = fname[len("package://") :]
        # ``package:///meshes/foo.stl`` (ROS / Onshape) — no package name.
        if rest.startswith("/"):
            rest = rest.lstrip("/")
        elif "/" in rest:
            rest = rest.split("/", 1)[1]
        return rest
    return fname


def _mesh_path_resolved_from_urdf(filename: str, urdf_dir: Path) -> Path:
    """Resolve a mesh path exactly as MuJoCo does — relative to the URDF file.

    Do **not** use ``str.lstrip("./")`` here: ``"../meshes/foo.stl".lstrip("./")``
    becomes ``"meshes/foo.stl"``, which falsely looks resolvable when the URDF
    was flattened from ``urdf/bot.urdf`` to ``bot.urdf`` beside ``meshes/``.
    """
    fname = _strip_package_uri(filename).replace("\\", "/")
    if not fname:
        return urdf_dir / "__missing__"
    if fname.startswith("/"):
        return Path(fname)
    return (urdf_dir / fname).resolve()


def _normalize_meshdir(meshdir: str | None) -> str | None:
    if not meshdir:
        return None
    norm = meshdir.replace("\\", "/").strip().lstrip("./").rstrip("/")
    return norm or "."


def _meshdir_doubles_filename_prefix(
    compiler_meshdir: str | None,
    prefixes: set[str],
) -> bool:
    """True when ``meshdir="meshes"`` and filenames already start with ``meshes/``."""
    mdir = _normalize_meshdir(compiler_meshdir)
    if not mdir or mdir == ".":
        return False
    for prefix in prefixes:
        norm = prefix.replace("\\", "/").lstrip("./").rstrip("/")
        if norm == mdir:
            return True
    return False


def default_mesh_search_dirs(urdf_path: Path) -> list[Path]:
    """Directories to search when URDF mesh paths do not resolve as written."""
    urdf_dir = urdf_path.parent
    candidates = [
        urdf_dir,
        urdf_dir / "meshes",
        urdf_dir.parent,
        urdf_dir.parent / "meshes",
        urdf_dir.parent / "mesh",
        urdf_dir.parent / "convex",
        urdf_dir / "assets",
        urdf_dir.parent / "assets",
        urdf_dir / "visual",
        urdf_dir / "collision",
    ]
    out: list[Path] = []
    seen: set[Path] = set()
    for c in candidates:
        try:
            r = c.resolve()
        except OSError:
            continue
        if r in seen or not r.is_dir():
            continue
        seen.add(r)
        out.append(r)
    return out


def _resolve_mesh_on_disk(
    filename: str,
    urdf_dir: Path,
    search_dirs: Sequence[Path],
) -> Path | None:
    """Locate a mesh file; return an absolute path when found."""
    fname = _strip_package_uri(filename)
    if not fname or fname.startswith("/"):
        return None

    direct = _mesh_path_resolved_from_urdf(fname, urdf_dir)
    if direct.is_file():
        return direct

    base = Path(fname).name
    if not base:
        return None

    for directory in search_dirs:
        try:
            d = directory.resolve()
        except OSError:
            continue
        if not d.is_dir():
            continue
        candidate = (d / base).resolve()
        if candidate.is_file():
            return candidate
        # Case-insensitive match (vendor URDFs vary .STL vs .stl).
        try:
            for entry in d.iterdir():
                if entry.is_file() and entry.name.lower() == base.lower():
                    return entry.resolve()
        except OSError:
            continue
    return None


def detect_mesh_path_issues(urdf_path: Path) -> list[str]:
    """Return human-readable issues with mesh path conventions."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = urdf_path.parent
    search_dirs = default_mesh_search_dirs(urdf_path)
    issues: list[str] = []

    inline_meshdir: str | None = None
    for child in root:
        if child.tag != "mujoco":
            continue
        compiler = child.find("compiler")
        if compiler is not None:
            inline_meshdir = compiler.get("meshdir")
            break

    prefixes: set[str] = set()
    for mesh in root.iter("mesh"):
        prefix = _mesh_filename_prefix(mesh.get("filename", ""))
        if prefix:
            prefixes.add(prefix)

    if inline_meshdir and prefixes:
        if _meshdir_doubles_filename_prefix(inline_meshdir, prefixes):
            issues.append(
                f"mesh filenames use directory prefix(es) {sorted(prefixes)!r} "
                f"while <compiler meshdir> is {inline_meshdir!r} — MuJoCo will "
                f"double the path (meshes/meshes/...)"
            )

    for mesh in root.iter("mesh"):
        fname = mesh.get("filename", "")
        if not fname:
            continue
        if fname.strip().startswith("package://"):
            issues.append(f"ROS package mesh URI must be rewritten: {fname}")
            continue
        located = _resolve_mesh_on_disk(fname, urdf_dir, search_dirs)
        if located is None:
            issues.append(f"missing mesh on disk: {fname}")
            continue
        stripped = _strip_package_uri(fname).replace("\\", "/")
        direct = _mesh_path_resolved_from_urdf(fname, urdf_dir)
        if direct.is_file():
            continue
        # G1 / Fourier: bare ``link.STL`` + ``<compiler meshdir="meshes"/>``.
        mdir = _normalize_meshdir(inline_meshdir)
        if mdir and mdir != "." and "/" not in stripped:
            via_meshdir = (urdf_dir / mdir / Path(stripped).name).resolve()
            if via_meshdir.is_file():
                continue
        try:
            rel = os.path.relpath(located, urdf_dir).replace("\\", "/")
        except ValueError:
            rel = str(located)
        if rel != stripped:
            issues.append(
                f"mesh path not MuJoCo-resolvable: {fname!r} "
                f"(found at {rel!r}; will auto-rewrite on ingest)"
            )

    return issues


def repair_urdf_mesh_paths(
    urdf_path: Path,
    *,
    search_dirs: Sequence[Path] | None = None,
    output_path: Path | None = None,
) -> Path:
    """Rewrite ``<mesh filename>`` attributes to paths that resolve from the URDF.

    Typical fix: ``pelvis.STL`` → ``meshes/pelvis.STL`` when the STL lives under
    ``meshes/`` but the URDF references a bare basename (X2, Booster, Fourier, …).
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = urdf_path.parent
    dirs = list(search_dirs) if search_dirs is not None else default_mesh_search_dirs(urdf_path)

    for mesh in root.iter("mesh"):
        fname = mesh.get("filename")
        if not fname:
            continue
        located = _resolve_mesh_on_disk(fname, urdf_dir, dirs)
        if located is None:
            continue
        try:
            rel = os.path.relpath(located, urdf_dir)
        except ValueError:
            rel = str(located)
        mesh.set("filename", rel.replace("\\", "/"))

    dest = output_path
    if dest is None:
        fd, tmp = tempfile.mkstemp(
            prefix=f"hhtools_urdf_{urdf_path.stem}_",
            suffix=".urdf",
        )
        os.close(fd)
        dest = Path(tmp)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)

    tree.write(str(dest), encoding="utf-8", xml_declaration=True)
    return dest


def normalize_urdf_meshes(
    urdf_path: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    """Rewrite mesh filenames so MuJoCo can resolve them.

    * Strips redundant ``./`` / ``meshes/`` prefixes when ``meshdir`` duplicates
      them.
    * Removes the inline ``<mujoco><compiler meshdir=...>`` block when filenames
      already carry the same directory prefix.

    Returns the path to the normalised URDF (``output_path`` or a temp file).
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    inline_meshdir: str | None = None
    mujoco_el: ET.Element | None = None
    for child in root:
        if child.tag == "mujoco":
            mujoco_el = child
            compiler = child.find("compiler")
            if compiler is not None:
                inline_meshdir = compiler.get("meshdir")
            break

    prefixes: set[str] = set()
    for mesh in root.iter("mesh"):
        prefix = _mesh_filename_prefix(mesh.get("filename", ""))
        if prefix:
            prefixes.add(prefix)

    strip_prefix: str | None = None
    norm_meshdir = _normalize_meshdir(inline_meshdir)
    if norm_meshdir and norm_meshdir != ".":
        for prefix in prefixes:
            norm = prefix.replace("\\", "/").lstrip("./").rstrip("/")
            if norm == norm_meshdir:
                strip_prefix = prefix
                break

    if strip_prefix and mujoco_el is not None:
        root.remove(mujoco_el)
        mujoco_el = None
        inline_meshdir = None

    if strip_prefix:
        pattern = re.compile(r"^\.?/?" + re.escape(strip_prefix) + r"/", re.I)
        for mesh in root.iter("mesh"):
            fname = mesh.get("filename")
            if not fname:
                continue
            cleaned = pattern.sub("", fname.replace("\\", "/"))
            mesh.set("filename", cleaned)

    dest = output_path
    if dest is None:
        fd, tmp = tempfile.mkstemp(
            prefix=f"hhtools_urdf_{urdf_path.stem}_",
            suffix=".urdf",
        )
        os.close(fd)
        dest = Path(tmp)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)

    tree.write(str(dest), encoding="utf-8", xml_declaration=True)
    return dest


def _inertia_attr(inertia_el: ET.Element | None, name: str, default: float = 0.0) -> float:
    if inertia_el is None:
        return default
    raw = inertia_el.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _inertia_matrix_from_urdf(inertial_el: ET.Element) -> np.ndarray | None:
    """Build the symmetric 3×3 inertia tensor from a URDF ``<inertial>`` block."""
    inertia_el = inertial_el.find("inertia")
    if inertia_el is None:
        return None
    ixx = _inertia_attr(inertia_el, "ixx")
    ixy = _inertia_attr(inertia_el, "ixy")
    ixz = _inertia_attr(inertia_el, "ixz")
    iyy = _inertia_attr(inertia_el, "iyy")
    iyz = _inertia_attr(inertia_el, "iyz")
    izz = _inertia_attr(inertia_el, "izz")
    return np.array(
        [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
        dtype=np.float64,
    )


def _principal_moments_violate_triangle(
    moments: np.ndarray,
    *,
    tol: float = _INERTIA_TRIANGLE_TOL,
) -> bool:
    """Return True when principal moments break MuJoCo's A+B≥C rule."""
    m = np.sort(np.asarray(moments, dtype=np.float64).reshape(3))
    if np.any(m < -tol):
        return True
    return bool(m[0] + m[1] < m[2] - tol)


def _inertial_block_issue(link_name: str, inertial_el: ET.Element) -> str | None:
    """Human-readable reason when a link inertia may fail MuJoCo compile."""
    inertia_el_inner = inertial_el.find("inertia")
    if inertia_el_inner is None:
        return None

    mat = _inertia_matrix_from_urdf(inertial_el)
    if mat is None:
        return None

    if not np.allclose(mat, mat.T, atol=1e-12, rtol=0.0):
        return f"link {link_name!r}: inertia tensor is not symmetric"

    try:
        evals = np.linalg.eigvalsh(mat)
    except np.linalg.LinAlgError:
        return f"link {link_name!r}: inertia tensor is not numerically valid"

    if _principal_moments_violate_triangle(evals):
        ixx = _inertia_attr(inertia_el_inner, "ixx")
        iyy = _inertia_attr(inertia_el_inner, "iyy")
        izz = _inertia_attr(inertia_el_inner, "izz")
        return (
            f"link {link_name!r}: inertia violates MuJoCo triangle inequality "
            f"(principal moments {evals.tolist()}, diagonal "
            f"ixx={ixx}, iyy={iyy}, izz={izz})"
        )
    return None


def detect_urdf_inertial_issues_from_root(root: ET.Element) -> list[str]:
    issues: list[str] = []
    for link in root.iter("link"):
        name = link.get("name", "<unnamed>")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        issue = _inertial_block_issue(name, inertial)
        if issue:
            issues.append(issue)
    return issues


def detect_urdf_inertial_issues(urdf_path: Path) -> list[str]:
    """List link inertias that are likely to fail MuJoCo URDF compilation."""
    return detect_urdf_inertial_issues_from_root(ET.parse(urdf_path).getroot())


def _mujoco_compiler_element(root: ET.Element) -> ET.Element | None:
    mujoco_el = next((child for child in root if child.tag == "mujoco"), None)
    if mujoco_el is None:
        return None
    return mujoco_el.find("compiler")


def _has_mujoco_compiler_block(urdf_path: Path) -> bool:
    """True when the URDF already carries a ``<mujoco><compiler>`` stanza."""
    return _mujoco_compiler_element(ET.parse(urdf_path).getroot()) is not None


def _mujoco_balanceinertia_enabled(urdf_path: Path) -> bool:
    compiler = _mujoco_compiler_element(ET.parse(urdf_path).getroot())
    if compiler is None:
        return False
    return (compiler.get("balanceinertia") or "").lower() == "true"


def _mesh_issues_need_repair(issues: Sequence[str]) -> bool:
    """True only for mesh paths that cannot load without rewriting the URDF."""
    return any(
        issue.startswith("missing mesh")
        or "not MuJoCo-resolvable" in issue
        or "package mesh URI" in issue
        for issue in issues
    )


def _meshdir_doubling_detected(issues: Sequence[str]) -> bool:
    """True when ``meshdir`` and filename prefixes both carry ``meshes/``."""
    return any("double the path" in issue for issue in issues)


def _compiler_meshdir_from_root(root: ET.Element) -> str | None:
    for child in root:
        if child.tag != "mujoco":
            continue
        compiler = child.find("compiler")
        if compiler is not None:
            return compiler.get("meshdir")
        break
    return None


def convert_unsupported_meshes_for_mujoco(
    urdf_path: Path,
    *,
    search_dirs: Sequence[Path] | None = None,
    output_path: Path | None = None,
) -> Path:
    """Convert meshes MuJoCo cannot decode (e.g. Collada ``.dae``) to ``.stl``."""
    import trimesh

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = urdf_path.parent
    dirs = list(search_dirs) if search_dirs is not None else default_mesh_search_dirs(urdf_path)
    changed = False

    for mesh in root.iter("mesh"):
        fname = mesh.get("filename")
        if not fname:
            continue
        ext = Path(_strip_package_uri(fname)).suffix.lower()
        if ext not in _CONVERT_TO_STL_EXTS:
            continue
        located = _resolve_mesh_on_disk(fname, urdf_dir, dirs)
        if located is None:
            continue
        stl_path = located.with_suffix(".stl")
        if not stl_path.is_file() or stl_path.stat().st_mtime < located.stat().st_mtime:
            loaded = trimesh.load(str(located), force="mesh")
            if isinstance(loaded, trimesh.Scene):
                geoms = [g for g in loaded.geometry.values() if hasattr(g, "vertices")]
                if not geoms:
                    continue
                loaded = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
            loaded.export(stl_path)
        try:
            rel = os.path.relpath(stl_path, urdf_dir).replace("\\", "/")
        except ValueError:
            rel = str(stl_path)
        mesh.set("filename", rel)
        changed = True

    dest = output_path or urdf_path
    if not changed:
        if dest != urdf_path:
            shutil.copy2(urdf_path, dest)
        return dest
    if dest != urdf_path:
        dest.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(dest), encoding="utf-8", xml_declaration=True)
    return dest


def prune_unresolvable_mesh_geometries(
    urdf_path: Path,
    *,
    search_dirs: Sequence[Path] | None = None,
    output_path: Path | None = None,
) -> tuple[Path, int]:
    """Drop ``visual``/``collision`` elements whose ``<mesh>`` file is missing.

    Keeps inertial + joint kinematics so MuJoCo can still compile for IK when
    a vendor drop omitted mesh assets (e.g. incomplete assembly URDFs).
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = urdf_path.parent
    dirs = list(search_dirs) if search_dirs is not None else default_mesh_search_dirs(urdf_path)
    removed = 0

    for link in root.iter("link"):
        for tag in ("visual", "collision"):
            for geom_parent in list(link.findall(tag)):
                mesh_el = geom_parent.find(".//mesh")
                if mesh_el is None:
                    continue
                fname = mesh_el.get("filename", "")
                if not fname:
                    continue
                if _resolve_mesh_on_disk(fname, urdf_dir, dirs) is None:
                    link.remove(geom_parent)
                    removed += 1

    dest = output_path or urdf_path
    if removed:
        if dest != urdf_path:
            dest.parent.mkdir(parents=True, exist_ok=True)
        tree.write(str(dest), encoding="utf-8", xml_declaration=True)
        ensure_mujoco_compiler_block(dest, output_path=dest)
    elif dest != urdf_path:
        shutil.copy2(urdf_path, dest)
    return dest, removed


def _link_is_dummy(link_el: ET.Element) -> bool:
    """True when a link carries no inertia, no visual and no collision geometry.

    Such links are pure kinematic placeholders (the ``world`` inertial frame of
    SolidWorks/ROS exports) — they never represent a physical body.
    """
    if link_el.find("inertial") is not None:
        return False
    if link_el.find("visual") is not None:
        return False
    if link_el.find("collision") is not None:
        return False
    return True


def detect_redundant_floating_base_root(
    root: ET.Element,
) -> tuple[str, str, str] | None:
    """Detect a dummy ``world``-style root anchored to the real base by a
    *floating* joint.

    Returns ``(root_link_name, anchor_joint_name, base_link_name)`` when the
    URDF root is a geometry-less placeholder (named ``world`` / ``base_footprint``
    / … *or* carrying no inertial/visual/collision) connected to exactly one
    child by a single ``type="floating"`` joint; otherwise ``None``.

    Why this matters: the Newton IK model is built with
    ``add_urdf(floating=True)``, which adds its own 6-DoF free joint at the URDF
    root.  When the URDF *also* ships ``<link name="world"/>`` plus a
    ``type="floating"`` joint to the real base, the articulation ends up with
    **two stacked floating bases**.  The retargeter only drives one of them, so
    the uncontrolled second 6-DoF leaves the robot drifting/floating above the
    ground regardless of motion format.  Stripping the anchor makes the real
    base the single floating root.

    A ``fixed`` anchor is intentionally *not* matched: it adds no DoF (single,
    correct floating base) and may host valid ik_map slots, so removing it would
    risk disturbing already-working robots.  See ``_ANCHOR_JOINT_TYPES``.
    """
    links = {
        link.get("name"): link
        for link in root.findall("link")
        if link.get("name")
    }
    if not links:
        return None

    joints = []
    child_links: set[str] = set()
    for joint in root.findall("joint"):
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.get("link")
        child = child_el.get("link")
        if not parent or not child:
            continue
        joints.append((joint.get("name") or "", parent, child, joint.get("type", "fixed")))
        child_links.add(child)

    roots = [name for name in links if name not in child_links]
    for root_name in roots:
        link_el = links[root_name]
        is_dummy = (
            (root_name or "").lower() in _DUMMY_ROOT_LINK_NAMES
            or _link_is_dummy(link_el)
        )
        if not is_dummy:
            continue
        outgoing = [j for j in joints if j[1] == root_name]
        if len(outgoing) != 1:
            continue
        jname, _parent, base_child, jtype = outgoing[0]
        if (jtype or "fixed").lower() not in _ANCHOR_JOINT_TYPES:
            continue
        if base_child not in links or base_child == root_name:
            continue
        return root_name, jname, base_child
    return None


def strip_redundant_floating_base_root(
    urdf_path: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    """Remove a redundant ``world``/floating-base anchor from a URDF.

    See :func:`detect_redundant_floating_base_root` for the rationale.  When no
    such anchor exists the file is returned/copied unchanged.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    detected = detect_redundant_floating_base_root(root)

    dest = output_path or urdf_path
    if detected is None:
        if dest != urdf_path:
            shutil.copy2(urdf_path, dest)
        return dest

    root_name, joint_name, base_child = detected
    for joint in list(root.findall("joint")):
        if (joint.get("name") or "") == joint_name:
            root.remove(joint)
    for link in list(root.findall("link")):
        if link.get("name") == root_name:
            root.remove(link)

    if dest != urdf_path:
        dest.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(dest), encoding="utf-8", xml_declaration=True)
    _log.info(
        "stripped redundant floating-base anchor %r (link %r → base %r) from %s; "
        "Newton add_urdf(floating=True) now adds a single floating root",
        joint_name, root_name, base_child, urdf_path.name,
    )
    return dest


def robot_upload_destination(drop: Path, relpath: str | Path, *, is_urdf: bool) -> Path:
    """Map a browser upload path into the robot library folder layout."""
    rel = Path(str(relpath).replace("\\", "/"))
    parts = rel.parts
    # Folder drops often prefix everything with the robot directory name.
    if len(parts) >= 2 and parts[0] not in (
        "meshes", "mesh", "convex", "assets", "urdf",
    ):
        rel = Path(*parts[1:])
        parts = rel.parts
    if is_urdf:
        # Folder drags often prefix ``urdf/foo.urdf``.  The robot registry only
        # scans ``<drop>/robot.yaml`` at the drop root — never nested folders —
        # so always flatten the URDF beside that yaml.
        return drop / Path(rel).name
    if len(parts) == 1:
        return drop / "meshes" / parts[0]
    return drop / rel


def urdf_needs_persisted_fixes(urdf_path: Path) -> bool:
    """Return whether ingest should rewrite the on-disk URDF."""
    if detect_redundant_floating_base_root(ET.parse(urdf_path).getroot()) is not None:
        return True
    mesh_issues = detect_mesh_path_issues(urdf_path)
    if _mesh_issues_need_repair(mesh_issues):
        return True
    if _meshdir_doubling_detected(mesh_issues):
        return True
    if not _has_mujoco_compiler_block(urdf_path):
        return True
    if detect_urdf_inertial_issues(urdf_path) and not _mujoco_balanceinertia_enabled(urdf_path):
        return True
    return False


def _mesh_filename_prefixes(root: ET.Element) -> set[str]:
    prefixes: set[str] = set()
    for mesh in root.iter("mesh"):
        prefix = _mesh_filename_prefix(mesh.get("filename", ""))
        if prefix:
            prefixes.add(prefix.replace("\\", "/").lstrip("./").rstrip("/"))
    return prefixes


def _infer_mujoco_meshdir(urdf_path: Path, root: ET.Element | None = None) -> str:
    """Guess ``<compiler meshdir>`` from the preset folder layout.

    Booster / SolidWorks exports often reference ``meshes/Foo.STL`` while a
    ``meshes/`` directory sits next to the URDF.  Setting ``meshdir="meshes"``
    in that case makes MuJoCo look for ``meshes/meshes/Foo.STL`` — use ``.``
    instead so the filename's directory prefix is honoured as-is.
    """
    urdf_dir = urdf_path.parent
    if not (urdf_dir / "meshes").is_dir():
        return "."
    if root is None:
        root = ET.parse(urdf_path).getroot()
    prefixes = _mesh_filename_prefixes(root)
    norm_prefixes = {p.replace("\\", "/").lstrip("./").rstrip("/") for p in prefixes}
    if norm_prefixes and norm_prefixes <= {"meshes"}:
        return "."
    return "meshes"


def ensure_mujoco_compiler_block(
    urdf_path: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    """Ensure a MuJoCo ``<compiler>`` block exists with sane defaults.

    Vendor URDFs without the inline ``<mujoco>`` stanza fail MuJoCo compilation
    even when mesh paths are otherwise correct.  Inserts the same minimal block
    used by bundled presets (e.g. Unitree G1) without changing joint/link
    kinematics or numeric attributes.

    When any link inertia violates MuJoCo's principal-moment triangle inequality,
    sets ``balanceinertia="true"`` so compilation can proceed (MuJoCo will
    adjust invalid diagonal/principal inertias).
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    meshdir = _infer_mujoco_meshdir(urdf_path, root)
    needs_balance = bool(detect_urdf_inertial_issues_from_root(root))
    changed = False

    mujoco_el: ET.Element | None = None
    for child in root:
        if child.tag == "mujoco":
            mujoco_el = child
            break

    if mujoco_el is None:
        mujoco_el = ET.Element("mujoco")
        root.insert(0, mujoco_el)
        changed = True

    compiler = mujoco_el.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_el, "compiler")
        changed = True
    current_meshdir = _normalize_meshdir(compiler.get("meshdir"))
    target_meshdir = _normalize_meshdir(meshdir) or "."
    prefixes = _mesh_filename_prefixes(root)
    if _meshdir_doubles_filename_prefix(compiler.get("meshdir"), prefixes):
        target_meshdir = "."
    if current_meshdir != target_meshdir:
        compiler.set("meshdir", target_meshdir)
        changed = True
    if compiler.get("discardvisual") is None:
        compiler.set("discardvisual", "false")
        changed = True
    if needs_balance and compiler.get("balanceinertia") is None:
        compiler.set("balanceinertia", "true")
        changed = True

    dest = output_path or urdf_path
    if dest != urdf_path:
        dest.parent.mkdir(parents=True, exist_ok=True)

    if not changed and dest == urdf_path:
        return urdf_path
    if not changed and dest != urdf_path:
        shutil.copy2(urdf_path, dest)
        return dest

    tree.write(str(dest), encoding="utf-8", xml_declaration=True)
    return dest


def repair_urdf_xml_structure(
    urdf_path: Path,
    *,
    output_path: Path | None = None,
) -> bool:
    """Fix common vendor XML mistakes (trailing junk after ``</robot>``).

    Returns True when the file was rewritten.
    """
    text = urdf_path.read_text(encoding="utf-8", errors="replace")
    close = "</robot>"
    idx = text.find(close)
    if idx < 0:
        return False
    end = idx + len(close)
    tail = text[end:].strip()
    if not tail:
        return False
    fixed = text[:end] + "\n"
    dest = output_path or urdf_path
    dest.write_text(fixed, encoding="utf-8")
    _log.info("repaired URDF XML structure (truncated junk after </robot>) in %s", urdf_path)
    return True


def ensure_urdf_meshes_resolvable(
    urdf_path: Path,
    *,
    search_dirs: Sequence[Path] | None = None,
    output_path: Path | None = None,
) -> Path:
    """Prepare a URDF for MuJoCo compile with the smallest possible edit.

    * Already-valid presets (meshes resolve, compiler block present, inertias
      OK) are left untouched — no rewrite on every reload.
    * Missing only the MuJoCo ``<compiler>`` stanza → insert the minimal G1-style
      block (``meshdir`` + ``discardvisual``); mesh paths and ``<inertia>``
      values are not rewritten.
    * Broken mesh paths → rewrite to paths relative to the URDF directory.
    * ``meshdir`` doubling (``meshes/foo`` + ``meshdir="meshes"``) → sync ``meshdir``.
    * Invalid inertias → add ``balanceinertia="true"`` on the compiler (MuJoCo
      adjusts at compile time; URDF inertia numbers stay as written).
    * Redundant ``world``/floating-base anchor (``<link name="world"/>`` +
      ``type="floating"`` joint) → strip it so the real base is the single
      root; otherwise ``add_urdf(floating=True)`` stacks a *second* floating
      base and the retargeted robot floats.
    """
    dest = output_path or urdf_path
    repaired_xml = repair_urdf_xml_structure(
        urdf_path, output_path=dest if dest != urdf_path else urdf_path
    )
    # ``repair_urdf_xml_structure`` only writes ``dest`` when it truncates junk.
    # ``dest`` may be a freshly-created empty temp file (``_compile_mjcf``), so
    # only treat it as the working copy once something has actually been written
    # to it — otherwise keep reading the canonical source.
    work_urdf = dest if (dest != urdf_path and repaired_xml) else urdf_path

    # Strip a redundant world/floating-base anchor *first* so every downstream
    # pass — and Newton's ``add_urdf(floating=True)`` — sees the real base as the
    # single root.  This is what fixes the "retarget result floats" symptom.
    if detect_redundant_floating_base_root(ET.parse(work_urdf).getroot()) is not None:
        strip_redundant_floating_base_root(work_urdf, output_path=dest)
        work_urdf = dest

    dirs = list(search_dirs) if search_dirs is not None else default_mesh_search_dirs(work_urdf)
    mesh_issues = detect_mesh_path_issues(work_urdf)
    needs_mesh_repair = _mesh_issues_need_repair(mesh_issues)
    needs_meshdir_fix = _meshdir_doubling_detected(mesh_issues)
    needs_compiler = not _has_mujoco_compiler_block(work_urdf)
    needs_balance_flag = (
        bool(detect_urdf_inertial_issues(work_urdf))
        and not _mujoco_balanceinertia_enabled(work_urdf)
    )
    needs_compiler_pass = (
        needs_compiler or needs_balance_flag or needs_meshdir_fix or needs_mesh_repair
    )

    if not needs_mesh_repair and not needs_compiler_pass:
        if dest != urdf_path and work_urdf != dest:
            shutil.copy2(work_urdf, dest)
        return dest

    if needs_mesh_repair:
        repair_urdf_mesh_paths(work_urdf, search_dirs=dirs, output_path=dest)
        work_path = dest
    else:
        work_path = work_urdf

    if needs_compiler_pass:
        ensure_mujoco_compiler_block(work_path, output_path=dest)
    elif dest != urdf_path and work_path != dest:
        shutil.copy2(work_path, dest)

    convert_unsupported_meshes_for_mujoco(dest, search_dirs=dirs, output_path=dest)

    missing = [
        i for i in detect_mesh_path_issues(dest) if i.startswith("missing mesh")
    ]
    if missing:
        dest, pruned = prune_unresolvable_mesh_geometries(
            dest, search_dirs=dirs, output_path=dest,
        )
        if pruned:
            _log.info(
                "Pruned %d unresolvable mesh visual/collision geoms from %s",
                pruned,
                urdf_path.name,
            )
        missing = [
            i for i in detect_mesh_path_issues(dest) if i.startswith("missing mesh")
        ]
    if missing:
        sample = ", ".join(missing[:5])
        extra = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise ValueError(
            f"URDF {urdf_path.name} still references {len(missing)} missing mesh "
            f"file(s) after auto-repair ({sample}{extra}).  Upload the full robot "
            f"folder (URDF + meshes/, mesh/, convex/, or assets/) and re-import."
        )
    return dest


def writable_urdf_copy(urdf_path: Path) -> Path:
    """Return ``urdf_path`` if writable, else a normalised temp copy."""
    try:
        test = urdf_path.parent / ".hhtools_write_test"
        test.write_text("", encoding="utf-8")
        test.unlink()
        return urdf_path
    except OSError:
        return ensure_urdf_meshes_resolvable(urdf_path)
