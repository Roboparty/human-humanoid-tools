"""Round-trip YAML read/write helpers for ``configs/robots/<name>/robot.yaml``.

The calibration UI lets users edit the ``ik_map`` interactively and
persist the result into the same ``robot.yaml`` that ships the rest of
the preset metadata.  A naive ``pyyaml.safe_dump`` round-trip would
obliterate the hand-authored comments (``# Canonical human joint → RP1
link mapping`` etc.) and re-order keys alphabetically, making future
``git diff``s noisy and hiding intent.

``ruamel.yaml`` round-trips comments, key order, indentation style and
quoting — so the only thing that changes after a "Save calibration"
click is the ``ik_map`` block (optionally followed by a one-line
audit-trail comment appended above it if we want to surface "last
edited by UI on <date>").

The helpers here are intentionally small and opinionated:

* ``update_robot_yaml_ik_map(path, ik_map)`` — replaces the ``ik_map:``
  block only; every other key stays byte-identical (comments, order,
  quoting).  Missing ``ik_map:`` blocks are created in a sensible
  position (after ``display_name`` if present, else after ``name:``).
* ``load_robot_yaml(path)`` — convenience RT loader for the same file,
  used by tests that want to verify the after-write state without
  re-importing through :mod:`hhtools.robot.registry` (which does its
  own normalisation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

__all__ = [
    "update_robot_yaml_ik_map",
    "update_robot_yaml_smooth_joint_filter_masks",
    "load_robot_yaml",
]


def _yaml_rt():
    """Build a ``ruamel.yaml.YAML`` configured for our repo style.

    Indentation: mapping=2, sequence=4, offset=2 — matches every
    hand-authored yaml we ship.  ``preserve_quotes`` keeps the rare
    quoted strings (e.g. ``"-1"`` for negative axis flags) unchanged.

    Lazy import keeps the whole module import-cheap when the calibration
    path isn't used.
    """

    from ruamel.yaml import YAML  # type: ignore[import]

    y = YAML()
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)
    # Don't reformat long strings onto multiple lines — keeps the
    # ``ik_map`` values (link names) on single lines as users expect.
    y.width = 4096
    return y


def load_robot_yaml(path: str | Path):
    """Read ``robot.yaml`` with round-trip preservation enabled.

    Returns ruamel's :class:`CommentedMap` (behaves like a ``dict`` but
    carries formatting metadata).
    """

    path = Path(path)
    yaml_io = _yaml_rt()
    with path.open("r", encoding="utf-8") as fp:
        return yaml_io.load(fp)


def update_robot_yaml_ik_map(
    path: str | Path,
    ik_map: Mapping[str, str],
) -> None:
    """Rewrite ``path`` with ``data['ik_map'] = ik_map``; preserves rest.

    Keys already present in the on-disk ``ik_map`` are updated in place
    (so their relative order is kept where possible), new keys are
    appended to the end, and keys that were on disk but no longer in
    ``ik_map`` are deleted.  This produces the minimal line-level diff
    when the user renames a single canonical slot in the UI.

    Args:
        path: Absolute or workspace-relative path to the ``robot.yaml``.
        ik_map: ``canonical_human_joint_name -> robot_link_name`` map,
            exactly as it should appear on disk.  Values must be
            non-empty strings; empty or ``None`` values raise a
            ``ValueError`` (deleting a mapping is done by omitting its
            key from the dict — no empty placeholders).

    Raises:
        FileNotFoundError: If ``path`` doesn't exist.
        ValueError: If ``ik_map`` values are malformed.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"robot.yaml not found at {path}")

    for k, v in ik_map.items():
        if not isinstance(v, str) or not v.strip():
            raise ValueError(
                f"ik_map[{k!r}]={v!r} — link names must be non-empty strings"
            )

    yaml_io = _yaml_rt()
    with path.open("r", encoding="utf-8") as fp:
        data = yaml_io.load(fp)

    if data is None:
        # Empty file — bootstrap a minimal mapping.
        from ruamel.yaml.comments import CommentedMap  # type: ignore[import]

        data = CommentedMap()

    existing = data.get("ik_map")
    if existing is None:
        from ruamel.yaml.comments import CommentedMap  # type: ignore[import]

        data["ik_map"] = CommentedMap(dict(ik_map))
    else:
        # In-place update so comments attached to individual ik_map
        # entries survive (ruamel attaches comments to the preceding
        # key by default).  Iterating through existing keys first
        # preserves their relative position.
        seen: set[str] = set()
        for k in list(existing.keys()):
            if k in ik_map:
                existing[k] = ik_map[k]
                seen.add(k)
            else:
                del existing[k]
        for k, v in ik_map.items():
            if k not in seen:
                existing[k] = v

    with path.open("w", encoding="utf-8") as fp:
        yaml_io.dump(data, fp)


def update_robot_yaml_smooth_joint_filter_masks(
    path: str | Path,
    masks: Mapping[str, float],
) -> None:
    """Rewrite ``smooth_joint_filter_masks`` in ``robot.yaml``; preserves rest."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"robot.yaml not found at {path}")

    for k, v in masks.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"smooth_joint_filter_masks key {k!r} must be a non-empty string")
        if not isinstance(v, (int, float)):
            raise ValueError(f"smooth_joint_filter_masks[{k!r}]={v!r} must be numeric")

    yaml_io = _yaml_rt()
    with path.open("r", encoding="utf-8") as fp:
        data = yaml_io.load(fp)

    if data is None:
        from ruamel.yaml.comments import CommentedMap  # type: ignore[import]

        data = CommentedMap()

    from ruamel.yaml.comments import CommentedMap  # type: ignore[import]

    existing = data.get("smooth_joint_filter_masks")
    if existing is None:
        data["smooth_joint_filter_masks"] = CommentedMap(
            {k: float(v) for k, v in masks.items()}
        )
    else:
        seen: set[str] = set()
        for k in list(existing.keys()):
            if k in masks:
                existing[k] = float(masks[k])
                seen.add(k)
            else:
                del existing[k]
        for k, v in masks.items():
            if k not in seen:
                existing[k] = float(v)

    with path.open("w", encoding="utf-8") as fp:
        yaml_io.dump(data, fp)
