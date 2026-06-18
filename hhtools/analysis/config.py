# SPDX-License-Identifier: Apache-2.0
"""Loader for the dataset-analysis configuration (thresholds + weights).

The single source of truth is ``configs/analysis/default.yaml``.  Values are read
once and cached; callers may pass an explicit override dict (e.g. from the web UI)
that is deep-merged on top of the defaults.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any


def _config_path() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent.parent / "configs" / "analysis" / "default.yaml"


@lru_cache(maxsize=1)
def _load_default() -> dict[str, Any]:
    path = _config_path()
    if not path.is_file():
        return dict(_BUILTIN_DEFAULTS)
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return dict(_BUILTIN_DEFAULTS)
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = deepcopy(val)
    return out


def load_config(override: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the merged analysis config (defaults + optional override)."""
    cfg = deepcopy(_load_default())
    if override:
        cfg = _deep_merge(cfg, override)
    return cfg


# Fallback used only if the YAML file is missing; kept in sync with default.yaml.
_BUILTIN_DEFAULTS: dict[str, Any] = {
    "complexity": {"lambda_accel": 0.1},
    "thresholds": {
        "floating_height_m": 0.15,
        "contact_height_m": 0.05,
        "foot_slide_speed_mps": 0.1,
        "static_root_speed_mps": 0.15,
        "inverted_angle_deg": 90.0,
        "jump_airborne_ratio": 0.12,
        "leg_static_quantile": 0.15,
        "dynamics_quantiles": [0.25, 0.75, 0.95],
    },
    "quality": {
        "weights": {
            "floating": 45.0,
            "penetration": 15.0,
            "vel_violation": 22.0,
            "foot_slide": 8.0,
            "self_collision": 5.0,
            "jerk": 5.0,
        },
        "warn_below": 90.0,
        "bad_below": 70.0,
        # LIMMT Table 3 raw calibrated weights (reference only; scaled to the
        # paper's metric units, not directly comparable to the normalised
        # severities above).
        "limmt_raw_weights": {
            "floating": 24.19,
            "foot_slide": 1.70,
            "penetration": 216.62,
            "jerk": 0.28,
            "vel_violation": 44.22,
            "self_collision": 0.17,
        },
    },
    "embedding": {"backend": "handcrafted", "pae": {"window_s": 4.0, "k": 8}},
    "subset": {"alpha": 0.99, "default_ratio": 0.1},
    # ``workers: 0`` (or omitted) = auto, capped by ``max_workers``; ``1`` = sequential.
    "parallel": {"workers": 0, "max_workers": 8},
}


__all__ = ["load_config"]
