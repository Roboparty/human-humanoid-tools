"""Secured FastAPI sidecar entry point for the Electron desktop shell."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hhtools.web.dependencies import MissingWebDependenciesError


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the hhtools Electron sidecar")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--session-secret")
    parser.add_argument(
        "--max-running-jobs",
        type=_non_negative_int,
        default=os.environ.get("HHTOOLS_MAX_RUNNING_JOBS"),
        help="Concurrent jobs; 0 means unlimited.",
    )
    parser.add_argument(
        "--max-queued-jobs",
        type=_non_negative_int,
        default=os.environ.get("HHTOOLS_MAX_QUEUED_JOBS"),
        help="Waiting jobs under a concurrency cap; 0 means unlimited.",
    )
    parser.add_argument(
        "--max-batch-items",
        type=_non_negative_int,
        default=os.environ.get("HHTOOLS_MAX_BATCH_ITEMS"),
        help="Items in one Agent batch; 0 means unlimited.",
    )
    parser.add_argument(
        "--max-batch-total-frames",
        type=_non_negative_int,
        default=os.environ.get("HHTOOLS_MAX_BATCH_TOTAL_FRAMES"),
        help="Estimated frames in one Agent batch; 0 means unlimited.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    args.source.mkdir(parents=True, exist_ok=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    # Electron normally uses the environment so the secret is not exposed in process listings.
    session_secret = args.session_secret or os.environ.get("HHTOOLS_DESKTOP_SESSION_SECRET")
    if not session_secret:
        parser.error(
            "a session secret is required via --session-secret or HHTOOLS_DESKTOP_SESSION_SECRET"
        )

    # Delay the heavier web imports until arguments and writable directories are valid.
    from hhtools.web.server import run_desktop_sidecar

    try:
        run_desktop_sidecar(
            source_root=args.source,
            save_dir=args.save_dir,
            cache_dir=args.cache,
            host=args.host,
            port=args.port,
            session_secret=session_secret,
            max_running_jobs=args.max_running_jobs,
            max_queued_jobs=args.max_queued_jobs,
            max_batch_items=args.max_batch_items,
            max_batch_total_frames=args.max_batch_total_frames,
        )
    except MissingWebDependenciesError as exc:
        parser.exit(status=1, message=f"{exc}\n")


if __name__ == "__main__":
    main()


__all__ = ["main"]
