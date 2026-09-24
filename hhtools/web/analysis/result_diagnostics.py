"""Compatibility import for :mod:`hhtools.analysis.result_diagnostics`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("hhtools.analysis.result_diagnostics")
