"""Core data structures for the hhtools motion pipeline.

This subpackage is intentionally free of optional dependencies
(no torch, UI frameworks, or newton).
so that headless pipelines remain lightweight.
"""

from __future__ import annotations

from hhtools.core.animation_buffer import AnimationBuffer
from hhtools.core.hierarchy import Hierarchy
from hhtools.core.motion import Motion
from hhtools.core.resample import resample_motion, resample_motion_with_objects
from hhtools.core.skeleton import Skeleton

__all__ = [
    "AnimationBuffer",
    "Hierarchy",
    "Motion",
    "Skeleton",
    "resample_motion",
    "resample_motion_with_objects",
]
