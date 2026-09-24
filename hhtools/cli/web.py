"""``hhtools web`` — launch the HTML / three.js web UI (FastAPI backend).

The browser performs 3D rendering while the backend reuses the hhtools
application and retargeting services.
"""

from __future__ import annotations

from pathlib import Path

import typer

from hhtools.web.dependencies import MissingWebDependenciesError

app = typer.Typer(help="Launch the HTML web UI (Apple-styled three.js front-end).")


@app.callback(invoke_without_command=True)
def launch(
    ctx: typer.Context,
    source: Path = typer.Option(
        Path("assets/motions"),
        "--source",
        "-s",
        envvar="HHTOOLS_SOURCE_ROOT",
        show_envvar=True,
        help="Raw-dataset root scanned recursively for the motion library.",
    ),
    save_dir: Path = typer.Option(
        Path("assets/save_npz"),
        "--save-dir",
        envvar="HHTOOLS_SAVE_DIR",
        show_envvar=True,
        help="Optional persisted NPZ cache (web exports download via the browser).",
    ),
    cache: Path | None = typer.Option(
        None,
        "--cache",
        envvar="HHTOOLS_CACHE_DIR",
        show_envvar=True,
        help="Per-session NPZ cache dir (defaults to a tempdir).",
    ),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8009, "--port"),
    max_running_jobs: int | None = typer.Option(
        None,
        "--max-running-jobs",
        min=0,
        envvar="HHTOOLS_MAX_RUNNING_JOBS",
        show_envvar=True,
        help="Concurrent background jobs; 0 selects unlimited mode.",
    ),
    max_queued_jobs: int | None = typer.Option(
        None,
        "--max-queued-jobs",
        min=0,
        envvar="HHTOOLS_MAX_QUEUED_JOBS",
        show_envvar=True,
        help=(
            "Waiting jobs when concurrency is limited; 0 means an unlimited queue."
        ),
    ),
    max_batch_items: int | None = typer.Option(
        None,
        "--max-batch-items",
        min=0,
        envvar="HHTOOLS_MAX_BATCH_ITEMS",
        show_envvar=True,
        help="Maximum items in one Agent batch; 0 selects unlimited mode.",
    ),
    max_batch_total_frames: int | None = typer.Option(
        None,
        "--max-batch-total-frames",
        min=0,
        envvar="HHTOOLS_MAX_BATCH_TOTAL_FRAMES",
        show_envvar=True,
        help="Maximum estimated frames in one Agent batch; 0 selects unlimited mode.",
    ),
) -> None:
    """Start the web UI on ``host:port`` and open a browser."""
    if ctx.invoked_subcommand is not None:
        return
    from hhtools.web.server import run_web

    try:
        run_web(
            source_root=source,
            save_dir=save_dir,
            cache_dir=cache,
            host=host,
            port=port,
            max_running_jobs=max_running_jobs,
            max_queued_jobs=max_queued_jobs,
            max_batch_items=max_batch_items,
            max_batch_total_frames=max_batch_total_frames,
        )
    except MissingWebDependenciesError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None


__all__ = ["app"]
