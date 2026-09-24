"""Small, responsive landing page for the human-facing ``hhtools`` CLI."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal, TextIO

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_BRAND = "#62a0ff"
_BRAND_SOFT = "#9fc2ff"
_ROBOPARTY_BLUE = "#0a3c96"
_MUTED = "#8b98ab"
_BORDER = "#34445c"
_MAX_PAGE_WIDTH = 104


@dataclass(frozen=True, slots=True)
class HomeEntry:
    """One stable destination shown on the CLI landing page."""

    title: str
    description: str
    command: str
    mode: Literal["CLI", "WEB", "APP"]


_WORKFLOWS = (
    HomeEntry(
        "Human → Robot",
        "Retarget human motion to a humanoid.",
        "hhtools retarget run",
        "CLI",
    ),
    HomeEntry(
        "Robot → Robot",
        "Open the source-to-target robot workspace.",
        "hhtools web",
        "WEB",
    ),
    HomeEntry(
        "Batch",
        "Run H2R or R2R batches from the workspace.",
        "hhtools web",
        "WEB",
    ),
)

_TOOLS = (
    HomeEntry(
        "Convert Motion",
        "Convert BVH or GLB into unified NPZ.",
        "hhtools convert run",
        "CLI",
    ),
    HomeEntry(
        "Robots",
        "List, inspect, or add robot presets.",
        "hhtools robot list",
        "CLI",
    ),
    HomeEntry(
        "System Check",
        "Check local features and dependencies.",
        "hhtools doctor",
        "CLI",
    ),
)

_OPEN = (
    HomeEntry(
        "WebUI",
        "Launch the browser workspace.",
        "hhtools web",
        "WEB",
    ),
    HomeEntry(
        "Desktop GUI",
        "Launch the installed Linux desktop app.",
        "hhtools-desktop",
        "APP",
    ),
)


def _badge(mode: str) -> Text:
    return Text(f" {mode} ", style=f"bold black on {_BRAND_SOFT}")


def _wide_section(entries: tuple[HomeEntry, ...], *, width: int) -> Table:
    table = Table.grid(expand=True, padding=(0, 1))
    table.width = width
    table.add_column(width=3, justify="right", style=_MUTED)
    table.add_column(width=18, style="bold")
    table.add_column(ratio=1, style=_MUTED)
    table.add_column(justify="right", style=_BRAND_SOFT, no_wrap=True)
    table.add_column(width=7, justify="right")
    for index, entry in enumerate(entries, start=1):
        table.add_row(
            f"{index:02d}",
            entry.title,
            entry.description,
            f"$ {entry.command}",
            _badge(entry.mode),
        )
    return table


def _compact_section(entries: tuple[HomeEntry, ...], *, width: int) -> Table:
    table = Table.grid(expand=True, padding=(0, 1))
    table.width = width
    table.add_column(width=3, justify="right", style=_MUTED)
    table.add_column(ratio=1)
    table.add_column(width=7, justify="right")
    for index, entry in enumerate(entries, start=1):
        table.add_row(f"{index:02d}", Text(entry.title, style="bold"), _badge(entry.mode))
        table.add_row("", Text(entry.description, style=_MUTED), "")
        table.add_row("", Text(f"$ {entry.command}", style=_BRAND_SOFT), "")
        if index != len(entries):
            table.add_row("", "", "")
    return table


def _section(
    title: str,
    entries: tuple[HomeEntry, ...],
    *,
    compact: bool,
    width: int,
) -> Group:
    heading = Text(title.upper(), style=f"bold {_BRAND}")
    body = (
        _compact_section(entries, width=width) if compact else _wide_section(entries, width=width)
    )
    return Group(heading, Text("─" * max(8, len(title) + 2), style=_BORDER), body)


def _identity(version: str) -> Group:
    title = Text.assemble(
        ("HHTOOLS", "bold"),
        (f"   v{version}", _MUTED),
    )
    subtitle = Text("Human motion → humanoid robots, in one toolkit.", style=_MUTED)
    attribution = Text.assemble(
        ("by ", _MUTED),
        ("Jagger Shen, Nora Sun and hhtools contributors", _ROBOPARTY_BLUE),
    )
    return Group(title, subtitle, attribution)


def _header(version: str, *, width: int) -> Panel:
    return Panel(
        _identity(version),
        border_style=_BORDER,
        padding=(1, 2),
        width=width,
    )


def render_homepage(console: Console, *, version: str) -> None:
    """Render the landing page without reading input or probing optional features."""

    page_width = min(console.width, _MAX_PAGE_WIDTH)
    compact = page_width < 88
    sections: tuple[RenderableType, ...] = (
        _section("Workflows", _WORKFLOWS, compact=compact, width=page_width),
        Text(),
        _section("Tools", _TOOLS, compact=compact, width=page_width),
        Text(),
        _section("Open", _OPEN, compact=compact, width=page_width),
    )
    console.print(_header(version, width=page_width))
    console.print()
    for section in sections:
        console.print(section)
    console.print()
    console.print(
        Text.assemble(
            ("Full command reference  ", _MUTED),
            ("$ hhtools --help", _BRAND_SOFT),
        )
    )


def print_homepage(
    *,
    version: str,
    file: TextIO | None = None,
) -> None:
    """Print one terminal-aware homepage to stdout by default."""

    console = Console(
        file=file or sys.stdout,
        highlight=False,
        soft_wrap=False,
    )
    render_homepage(console, version=version)


__all__ = ["HomeEntry", "print_homepage", "render_homepage"]
