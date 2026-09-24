"""Compatibility import for :mod:`hhtools.core.anatomy`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("hhtools.core.anatomy")
