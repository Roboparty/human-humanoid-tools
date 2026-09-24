"""Compatibility import for :mod:`hhtools.core.human_anatomy`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("hhtools.core.human_anatomy")
