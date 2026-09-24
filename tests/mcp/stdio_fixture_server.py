"""Deterministic stdio fixture used by the MCP subprocess smoke test."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from hhtools.contracts import CapabilityResponse, SchedulerCapability
from hhtools.mcp.runtime import AgentRuntime
from hhtools.mcp.server import create_mcp_server


class _Capabilities:
    def get_capabilities(self) -> CapabilityResponse:
        return CapabilityResponse(
            service_version="stdio-test",
            scheduler=SchedulerCapability(
                max_running_jobs=0,
                max_queued_jobs=0,
                mode="unlimited",
            ),
            supported_input_formats=["bvh"],
            supported_output_formats=["csv"],
            features={"agent_rest": True, "json_cli": True, "mcp": True},
        )


_UNUSED = cast(Any, object())
_RUNTIME = AgentRuntime(
    capabilities=cast(Any, _Capabilities()),
    assets=_UNUSED,
    available_assets=_UNUSED,
    preflight=_UNUSED,
    r2r_preflight=_UNUSED,
    batch_preflight=_UNUSED,
    plans=_UNUSED,
    jobs=_UNUSED,
    exports=_UNUSED,
)


@asynccontextmanager
async def _runtime_factory() -> AsyncIterator[AgentRuntime]:
    yield _RUNTIME


if __name__ == "__main__":
    # stdout is exclusively the MCP wire.  This fixture intentionally performs
    # no import-time or pre-run printing.
    create_mcp_server(runtime_factory=_runtime_factory).run("stdio")
