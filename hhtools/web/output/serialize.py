"""Compatibility import for :mod:`hhtools.io.scene_serialize`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("hhtools.io.scene_serialize")
