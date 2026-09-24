"""Compatibility import for :mod:`hhtools.application.export`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("hhtools.application.export")
