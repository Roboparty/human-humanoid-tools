"""Compatibility import for :mod:`hhtools.analysis.scaled_preview`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("hhtools.analysis.scaled_preview")
