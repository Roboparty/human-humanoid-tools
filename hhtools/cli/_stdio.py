"""Cross-platform standard-stream configuration for the command-line app."""

from __future__ import annotations

import sys
from typing import TextIO


def _reconfigure_utf8(stream: TextIO | None) -> None:
    """Use UTF-8 when the active stream supports runtime reconfiguration."""
    if stream is None:
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):
        # Test runners and embedded hosts commonly replace stdio with StringIO.
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        # A closed or host-owned stream cannot be reconfigured safely.
        return


def configure_utf8_stdio() -> None:
    """Make CLI output deterministic on Windows and when redirected to a pipe."""
    _reconfigure_utf8(sys.stdout)
    _reconfigure_utf8(sys.stderr)


__all__ = ["configure_utf8_stdio"]
