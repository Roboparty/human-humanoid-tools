"""Compatibility import for :mod:`hhtools.analysis.dataset_analysis`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("hhtools.analysis.dataset_analysis")
