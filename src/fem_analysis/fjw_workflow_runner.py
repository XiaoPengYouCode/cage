from __future__ import annotations

from .fjw_workflow_driver import (
    FJWWorkflowDriverRequest,
    FJWWorkflowDriverResult,
    run_fjw_workflow_iteration,
)
from .fjw_workflow_solver_adapters import (
    FJWAbaqusWorkflowSolverConfig,
    build_fjw_abaqus_solver_adapters,
)
from .fjw_workflow_sfepy_solver_adapters import (
    FJWSfePyWorkflowSolverConfig,
    build_fjw_sfepy_solver_adapters,
)


def run_fjw_abaqus_workflow_iteration(
    *,
    driver_request: FJWWorkflowDriverRequest,
    solver_config: FJWAbaqusWorkflowSolverConfig | None = None,
) -> FJWWorkflowDriverResult:
    """Run one FJW workflow iteration with Abaqus-backed solver adapters."""

    solver_adapters = build_fjw_abaqus_solver_adapters(config=solver_config)
    return run_fjw_workflow_iteration(
        driver_request,
        forward_solver=solver_adapters.forward_solver,
        adjoint_solver=solver_adapters.adjoint_solver,
    )


def run_fjw_sfepy_workflow_iteration(
    *,
    driver_request: FJWWorkflowDriverRequest,
    solver_config: FJWSfePyWorkflowSolverConfig | None = None,
) -> FJWWorkflowDriverResult:
    """Run one FJW workflow iteration with SfePy direct-solver adapters."""

    solver_adapters = build_fjw_sfepy_solver_adapters(config=solver_config)
    return run_fjw_workflow_iteration(
        driver_request,
        forward_solver=solver_adapters.forward_solver,
        adjoint_solver=solver_adapters.adjoint_solver,
    )


__all__ = [
    "run_fjw_abaqus_workflow_iteration",
    "run_fjw_sfepy_workflow_iteration",
]
