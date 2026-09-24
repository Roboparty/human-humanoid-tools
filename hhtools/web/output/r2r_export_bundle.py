"""Compatibility import for :mod:`hhtools.io.r2r_export_bundle`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("hhtools.io.r2r_export_bundle")
