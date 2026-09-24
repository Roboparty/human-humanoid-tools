"""Compatibility import for :mod:`hhtools.services.job_settings`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("hhtools.services.job_settings")
