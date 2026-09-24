"""Registry that scans ``configs/robots/*/robot.yaml`` and exposes presets.

Usage:

    from hhtools.robot.registry import list_presets, get

    for preset in list_presets():
        print(preset.name, preset.has_urdf)

    rp1 = get("roboparty_rp1")

The registry is a thin cache around filesystem scanning.  It's deliberately
*stateful but idempotent*: calling :func:`refresh` on the process-wide
singleton re-scans the config directories and rebuilds the dict.  Callers who
need a fresh view (e.g. the "+ add robot" workflow that just dropped a new
preset on disk) should call :func:`refresh` explicitly.

Design notes:

* **Presets with missing URDFs are still registered.**  The UI renders them
  greyed-out with a "URDF missing — drop a file at ``<path>``" hint.  This
  matches the user's preference for ``unitree_g1``: we ship the
  ``robot.yaml`` scaffold and the user drops the URDF in later.
* **Directory name is the truth.**  The name field inside the yaml is
  validated against the enclosing directory — if they disagree, we trust the
  directory and log a warning.  This prevents "I renamed the folder but not
  the yaml" bugs.
* **Discovery roots**:
    1. ``<workspace>/configs/robots/`` (source tree; primary for dev).
    2. ``~/.config/hhtools/robots/`` (user-installed robots — the target for
       ``hhtools add-robot``).
    3. Anything on the ``HHTOOLS_ROBOT_PATH`` env var (colon-separated).
  We iterate in that order; **later** roots win on name collisions so a
  user-installed robot under ``~/.config/hhtools/robots/`` overrides a
  same-named built-in from the workspace (e.g. uploading ``rpo`` when the
  repo already ships ``configs/robots/rpo``).
* **Zero-config robots** (user dropped a URDF + mesh/ but didn't write any
  yaml): we auto-synthesise a ``robot.yaml`` via
  :mod:`hhtools.robot.scaffold` the first time we see the directory, then
  treat the written yaml as source of truth on subsequent scans.  Directories
  holding more than one ``.urdf`` file produce one
  ``robot.<urdf_stem>.yaml`` + ``<dirname>__<stem>`` preset per URDF.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from hhtools.robot.base import RobotPreset
from hhtools.robot.scaffold import scaffold_yaml_file
from hhtools.utils.paths import user_robot_dir

_log = logging.getLogger(__name__)

# Module-level cache.  The singleton flavour is simpler than a class given
# how rarely callers need more than one registry — and it matches how the
# skeleton-preset / body-model registries in the rest of hhtools work.
_CACHE: dict[str, RobotPreset] | None = None


# --------------------------------------------------------------------------- public API


def list_presets() -> list[RobotPreset]:
    """All known robot presets, alphabetically by name."""
    _ensure_loaded()
    assert _CACHE is not None
    return sorted(_CACHE.values(), key=lambda p: p.name)


def list_presets_readonly() -> list[RobotPreset]:
    """Discover existing YAML presets without scaffolding or mutating cache.

    Capability discovery and preflight use this path because a read-only query
    must not turn an orphan URDF into a newly written ``robot.yaml``.  Normal
    UI/CLI discovery keeps the historical zero-config scaffolding behaviour in
    :func:`list_presets` and :func:`refresh`.
    """

    if _CACHE is not None:
        return sorted(_CACHE.values(), key=lambda preset: preset.name)
    discovered: dict[str, RobotPreset] = {}
    for root in _discovery_roots():
        for preset in _scan_root(root, scaffold_missing=False):
            discovered[preset.name] = preset
    return sorted(discovered.values(), key=lambda preset: preset.name)


def list_presets_in_root_readonly(root: Path) -> list[RobotPreset]:
    """Discover existing presets below one trusted root without scaffolding."""

    discovered = {
        preset.name: preset for preset in _scan_root(Path(root), scaffold_missing=False)
    }
    return sorted(discovered.values(), key=lambda preset: preset.name)


def get(name: str) -> RobotPreset:
    """Look up a preset by name.  Raises :class:`KeyError` if unknown."""
    _ensure_loaded()
    assert _CACHE is not None
    try:
        return _CACHE[name]
    except KeyError as err:
        known = ", ".join(sorted(_CACHE)) or "<none>"
        raise KeyError(
            f"no robot preset named {name!r}; known: {known}"
        ) from err


def refresh() -> list[RobotPreset]:
    """Re-scan the discovery roots and rebuild the cache.

    Call this after ``hhtools add-robot`` writes a new preset, or in tests
    that synthesise fake presets on disk.
    """
    global _CACHE
    _CACHE = {}
    for root in _discovery_roots():
        for preset in _scan_root(root):
            # Later discovery roots override earlier ones on duplicate names so
            # Web UI uploads in ``~/.config/hhtools/robots/`` take precedence
            # over same-named repo presets (``configs/robots/<name>``).
            _CACHE[preset.name] = preset
    return sorted(_CACHE.values(), key=lambda p: p.name)


def preset_from_dir(drop: Path) -> RobotPreset:
    """Load the primary preset from a robot library folder.

    Used immediately after Web upload / ``hhtools robot add`` writes files to
    disk, before relying on the global cache scan.
    """
    drop = drop.resolve()
    yaml_paths = _collect_yaml_paths(drop)
    if not yaml_paths:
        raise FileNotFoundError(f"no robot.yaml under {drop}")
    preset = _load_preset(yaml_paths[0], drop)
    if not preset.has_urdf:
        raise FileNotFoundError(
            f"robot.yaml under {drop} references missing URDF {preset.urdf_path}"
        )
    return preset


def preset_from_yaml(yaml_path: Path) -> RobotPreset:
    """Load one exact existing preset YAML without caching or scaffolding.

    Agent preflight uses this after binding the YAML hash to a RobotBundle so
    a previously populated process cache cannot supply stale ``dof_order`` or
    ``ik_map`` values.  The function is read-only and never synthesizes files.
    """

    path = Path(yaml_path).resolve(strict=True)
    if not path.is_file() or not (
        path.name == "robot.yaml"
        or (path.name.startswith("robot.") and path.name.endswith(".yaml"))
    ):
        raise FileNotFoundError(f"no robot preset YAML at {path}")
    return _load_preset(path, path.parent)


def clear_cache() -> None:
    """Wipe the cache so the next access re-scans.  Used in tests."""
    global _CACHE
    _CACHE = None


def is_user_installed(
    preset: RobotPreset,
    user_root: Path | None = None,
) -> bool:
    """True when ``preset`` lives in the per-user robot library (deletable via UI)."""
    root = (user_root or user_robot_dir()).resolve()
    try:
        return preset.root_dir.resolve().is_relative_to(root)
    except (ValueError, OSError):
        return False


# --------------------------------------------------------------------------- internals


def _ensure_loaded() -> None:
    if _CACHE is None:
        refresh()


def _discovery_roots() -> list[Path]:
    """Ordered list of directories to scan for ``<name>/robot.yaml`` files."""
    roots: list[Path] = []

    # 1. Workspace.  We walk up from this file until we find ``configs/robots``.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs" / "robots"
        if candidate.is_dir():
            roots.append(candidate)
            break

    # 2. User-installed (web UI uploads, ``hhtools robot add`` to user dir).
    user_root = user_robot_dir()
    if user_root.is_dir():
        roots.append(user_root)

    # 3. Explicit env override.
    env = os.environ.get("HHTOOLS_ROBOT_PATH", "")
    for part in env.split(os.pathsep):
        if not part:
            continue
        p = Path(part).expanduser()
        if p.is_dir():
            roots.append(p)

    return roots


def _scan_root(root: Path, *, scaffold_missing: bool = True) -> list[RobotPreset]:
    """Scan one discovery root for ``<child>/robot*.yaml`` files.

    Per directory we first collect all ``robot.yaml`` + ``robot.<stem>.yaml``
    files that are already on disk (those are authoritative — users may have
    edited them).  If the directory holds URDFs that *aren't* covered by any
    of those yamls we scaffold the missing ones via
    :func:`hhtools.robot.scaffold.scaffold_yaml_file`, which writes a
    ``# auto-generated`` yaml next to each orphaned URDF.  This is how a user
    can add a new robot by just dropping a URDF + meshes into the tree — no
    yaml editing required.
    """
    out: list[RobotPreset] = []
    for child in sorted(root.iterdir()):
        if child.name.startswith(("_", ".")):
            # Templates and hidden runtime folders stay invisible.
            continue
        if child.is_symlink() or not child.is_dir():
            continue

        if scaffold_missing:
            _autoscaffold_missing_yaml(child)

        yaml_paths = _collect_yaml_paths(child)
        if not yaml_paths:
            continue
        for yaml_path in yaml_paths:
            try:
                preset = _load_preset(yaml_path, enclosing_dir=child)
            except Exception as err:
                _log.warning("skipping %s: %s", yaml_path, err)
                continue
            out.append(preset)
    return out


def _collect_yaml_paths(child: Path) -> list[Path]:
    """Return every ``robot*.yaml`` file in ``child`` (``robot.yaml`` first).

    Order matters: the plain ``robot.yaml`` wins over ``robot.<stem>.yaml``
    when both exist for the same preset *name*, but that collision is only
    possible if the user manually created a ``robot.yaml`` *and* stem-suffixed
    variants in the same directory — in which case the warning in
    :func:`_load_preset` will flag the duplicate and the first one wins.
    """
    paths: list[Path] = []
    canonical = child / "robot.yaml"
    if canonical.is_file():
        paths.append(canonical)
    for extra in sorted(child.glob("robot.*.yaml")):
        # Skip things like ``robot.yaml.bak`` (glob above only matches
        # ``robot.<stem>.yaml`` but defensive filtering is cheap).
        if extra.suffix == ".yaml" and extra.name != "robot.yaml":
            paths.append(extra)
    return paths


def _autoscaffold_missing_yaml(child: Path) -> None:
    """Write ``robot.yaml`` / ``robot.<stem>.yaml`` for orphaned URDFs.

    Invariants:
    * If the directory already has a yaml that points at a given URDF, we
      leave it alone — the user may have tuned ``ik_map`` or ``dof_order``.
    * If the directory has a single URDF and no yaml, we produce
      ``robot.yaml`` (preset name = directory name).
    * If the directory has multiple URDFs, each URDF gets its own
      ``robot.<stem>.yaml`` / ``<dirname>__<stem>`` preset.  Any URDF that
      already has a matching yaml on disk is left alone.
    """
    urdfs = sorted(child.glob("*.urdf"))
    if not urdfs:
        return

    # Find URDF paths already referenced by existing yamls so we don't
    # re-scaffold them.
    covered: set[Path] = set()
    for yaml_path in _collect_yaml_paths(child):
        try:
            with yaml_path.open("r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
        except Exception:
            continue
        urdf_rel = data.get("urdf") if isinstance(data, dict) else None
        if not urdf_rel:
            continue
        try:
            covered.add((child / str(urdf_rel)).resolve())
        except OSError:
            pass

    for urdf_path in urdfs:
        if urdf_path.resolve() in covered:
            continue
        try:
            scaffold_yaml_file(urdf_path)
        except Exception as err:
            _log.warning("auto-scaffold failed for %s: %s", urdf_path, err)


def _load_preset(yaml_path: Path, enclosing_dir: Path) -> RobotPreset:
    """Parse a ``robot*.yaml`` into a :class:`RobotPreset`.

    Validation is deliberately lenient at registration time — only the
    *structural* bits (name, URDF reachability) are checked.  Anything opaque
    to this layer (``ik_map``, ``weights`` etc.) is preserved as-is so the
    retargeter can validate it later with domain-specific rules.

    The *expected* preset name is derived from the yaml filename:
    * ``robot.yaml``            → ``<dir_name>``
    * ``robot.<stem>.yaml``     → ``<dir_name>__<stem>``

    The yaml's ``name:`` key must agree with this expectation (directory + file
    layout wins, yaml gets a warning).
    """
    with yaml_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}

    if not isinstance(data, dict):
        raise ValueError(f"top level of {yaml_path} must be a mapping")

    dir_name = enclosing_dir.name
    if yaml_path.name == "robot.yaml":
        expected_name = dir_name
    else:
        # ``robot.<stem>.yaml`` → drop leading ``robot.`` and trailing ``.yaml``.
        stem = yaml_path.name[len("robot."):-len(".yaml")]
        expected_name = f"{dir_name}__{stem}"

    name = str(data.get("name") or expected_name)
    if name != expected_name:
        _log.warning(
            "yaml %s has name=%r but directory+filename expects %r; "
            "using %r (on-disk layout wins)",
            yaml_path, name, expected_name, expected_name,
        )
        name = expected_name

    display_name = str(data.get("display_name") or name)

    # URDF resolution — relative paths anchor at the yaml's directory.
    root_dir = enclosing_dir
    urdf_path: Path | None = None
    urdf_value = data.get("urdf")
    if urdf_value:
        urdf_path = (root_dir / urdf_value).resolve()
        if not urdf_path.is_file():
            # Leave ``urdf_path`` pointing at the expected location so the UI
            # can tell users "drop the URDF here".  ``preset.has_urdf`` is
            # the canonical way to check availability downstream.
            pass

    mesh_search_paths: list[Path] = []
    for raw in data.get("mesh_search_paths") or []:
        p = (root_dir / raw).resolve()
        if p.is_dir():
            mesh_search_paths.append(p)

    # Everything below is opaque to the registry — the retarget layer owns
    # the schema of these fields.  We pass them through so one yaml is the
    # single source of truth.
    ik_map = dict(data.get("ik_map") or {})
    weights = dict(data.get("weights") or {})
    rest_offsets = {
        k: tuple(float(x) for x in (v or (0.0, 0.0, 0.0)))
        for k, v in (data.get("rest_offsets") or {}).items()
    }
    smooth_joint_filter_masks = {
        str(k): float(v)
        for k, v in (data.get("smooth_joint_filter_masks") or {}).items()
    }
    feet = dict(data.get("feet") or {})
    dof_order = tuple(str(x) for x in (data.get("dof_order") or []))
    length_scale = float(data.get("length_scale") or 1.0)
    up_axis = str(data.get("up_axis") or "Z").upper()
    forward_axis = str(data.get("forward_axis") or "X").upper()

    known_keys = {
        "name", "display_name", "urdf", "mesh_search_paths", "ik_map",
        "weights", "rest_offsets", "smooth_joint_filter_masks", "feet",
        "length_scale", "up_axis", "forward_axis", "dof_order",
    }
    meta = {k: v for k, v in data.items() if k not in known_keys}
    # Auto-generated yamls (written by :mod:`hhtools.robot.scaffold`) start
    # with a ``# ... auto-generated by hhtools.robot.scaffold.`` banner.
    # That's opaque to YAML, so we peek at the raw file to set the flag —
    # which surfaces in ``hhtools robot list`` / UI so users can tell which
    # presets were synthesised vs hand-tuned.
    try:
        first_line = yaml_path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        first_line = ""
    meta["auto_generated"] = "auto-generated by hhtools.robot.scaffold" in first_line
    meta["yaml_path"] = str(yaml_path)

    return RobotPreset(
        name=name,
        display_name=display_name,
        root_dir=root_dir,
        urdf_path=urdf_path,
        mesh_search_paths=tuple(mesh_search_paths),
        ik_map=ik_map,  # type: ignore[arg-type]
        weights=weights,  # type: ignore[arg-type]
        rest_offsets=rest_offsets,  # type: ignore[arg-type]
        smooth_joint_filter_masks=smooth_joint_filter_masks,
        feet=feet,
        length_scale=length_scale,
        up_axis=up_axis,  # type: ignore[arg-type]
        forward_axis=forward_axis,  # type: ignore[arg-type]
        dof_order=dof_order,
        meta=meta,
    )
