"""Compatibility import for :mod:`hhtools.services.motion_library_settings`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("hhtools.services.motion_library_settings")
