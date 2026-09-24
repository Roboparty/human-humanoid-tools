"""Dataset directory aliases shared by discovery and format sniffers."""

from __future__ import annotations

import re

# Directory-name -> registered adapter name. Keys are normalized with
# ``normalize_dataset_dirname`` so discovery remains insensitive to spacing,
# punctuation, and capitalization.
DATASET_DIR_TO_ADAPTER: dict[str, str] = {
    "amass": "amass",
    "accad": "amass",
    "cmu": "amass",
    "motionx": "motion_x",
    "phuma": "phuma",
    "lafan": "lafan",
    "lafan1": "lafan",
    "mocap": "mocap",
    "soma": "soma",
    "xsens": "xsens_mocap",
    "xsensmocap": "xsens_mocap",
    "origin_data": "xsens_mocap",
    "100style": "xsens_mocap",
    "style100": "xsens_mocap",
    "gvhmr": "gvhmr",
    "kungfu": "kungfu_athlete",
    "kungfuathlete": "kungfu_athlete",
    "omomo": "omomo",
    "omnicontact": "omnicontact",
    "omnicontactdataset": "omnicontact",
    "raw_mocap": "omnicontact",
    "holosoma": "meshmimic_holosoma",
    "glb": "glb",
    "gltf": "glb",
    "parc_ms": "parc_ms",
    "parcms": "parc_ms",
    "unified_npz": "unified_npz",
    "unifiednpz": "unified_npz",
    "20260429mocap": "parc_ms",
}


def normalize_dataset_dirname(name: str) -> str:
    """Normalize a dataset folder name for alias lookup."""

    return re.sub(r"[^a-z0-9]", "", name.lower())


# Compatibility aliases for callers migrated from the old viewer module.
_DIR_TO_ADAPTER = DATASET_DIR_TO_ADAPTER
_normalise_dirname = normalize_dataset_dirname


__all__ = ["DATASET_DIR_TO_ADAPTER", "normalize_dataset_dirname"]
