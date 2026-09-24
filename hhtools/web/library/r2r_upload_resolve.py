"""Compatibility import for :mod:`hhtools.services.r2r_upload_resolve`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("hhtools.services.r2r_upload_resolve")
