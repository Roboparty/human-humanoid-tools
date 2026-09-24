"""Compatibility import for :mod:`hhtools.application.progress`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("hhtools.application.progress")
