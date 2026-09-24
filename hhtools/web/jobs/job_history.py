"""Compatibility import for :mod:`hhtools.services.web_job_history`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("hhtools.services.web_job_history")
