"""Normalization of solver-observed execution provenance."""

from __future__ import annotations

from typing import Any

from hhtools.contracts.execution import ExecutionProvenance


def build_execution_provenance(
    retargeted: Any,
    *,
    executor: str,
    backend: str,
    **identity: Any,
) -> ExecutionProvenance:
    """Combine solver metadata with adapter-owned execution identity."""

    metadata = getattr(retargeted, "meta", {})
    observed = metadata.get("execution_provenance", {}) if isinstance(metadata, dict) else {}
    if not isinstance(observed, dict):
        observed = {}
    payload = {
        **observed,
        "executor": executor,
        "backend": backend,
        **{key: value for key, value in identity.items() if value is not None},
    }
    return ExecutionProvenance.model_validate(payload)


__all__ = ["build_execution_provenance"]
