"""Stdio fixture that initializes real Warp through the MCP runtime path."""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from hhtools.application.runtime import ApplicationPaths
from hhtools.mcp.runtime import LocalRuntimeConfig, local_agent_runtime
from hhtools.mcp.server import create_mcp_server

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="hhtools-stdio-warp-") as temporary:
        config = LocalRuntimeConfig(paths=ApplicationPaths.isolated(Path(temporary)))

        @asynccontextmanager
        async def runtime_with_initialized_warp():
            async with local_agent_runtime(config) as runtime:
                import warp as wp

                wp.init()
                yield runtime

        create_mcp_server(
            config,
            runtime_factory=runtime_with_initialized_warp,
        ).run("stdio")
