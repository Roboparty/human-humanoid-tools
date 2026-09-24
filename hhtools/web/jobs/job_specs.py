"""Compatibility import for :mod:`hhtools.contracts.legacy_jobs`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("hhtools.contracts.legacy_jobs")
