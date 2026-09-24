"""Validated execution inputs for browser adapters."""

from __future__ import annotations

from hhtools.contracts.execution import (
    ExecutionIdentifier,
    HumanRetargetExecutionOptions,
    SolverExecutionOptions,
)


class HumanRetargetRequest(HumanRetargetExecutionOptions):
    robot: ExecutionIdentifier
    motion_token: ExecutionIdentifier


class RobotRetargetRequest(SolverExecutionOptions):
    target: ExecutionIdentifier
    source: ExecutionIdentifier
    source_token: ExecutionIdentifier
