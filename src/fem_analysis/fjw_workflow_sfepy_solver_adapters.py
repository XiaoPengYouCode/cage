from __future__ import annotations

from dataclasses import dataclass, field

from .fjw_direct_solver import (
    FJWDirectSolverConfig,
    solve_fjw_direct_adjoint_case,
    solve_fjw_direct_case,
)
from .fjw_workflow_single_case import (
    FJWAdjointSolveRequest,
    FJWAdjointSolver,
    FJWElementSolveResult,
    FJWForwardSolveRequest,
    FJWForwardSolver,
)
from .fjw_workflow_solver_adapters import _build_step_workflow_state
from .fjw_workflow_vectors import build_element_displacement_cache


@dataclass(frozen=True, slots=True)
class FJWSfePyWorkflowSolverConfig:
    direct_solver_config: FJWDirectSolverConfig = field(default_factory=FJWDirectSolverConfig)
    metadata_prefix: str = "sfepy_direct"


@dataclass(frozen=True, slots=True)
class FJWSfePyWorkflowSolverBackend:
    config: FJWSfePyWorkflowSolverConfig

    def solve_forward(self, request: FJWForwardSolveRequest) -> FJWElementSolveResult:
        step_state = _build_step_workflow_state(
            request.workflow_state,
            design_cage=request.design_cage,
            obj_bo=request.obj_bo,
        )
        result = solve_fjw_direct_case(
            step_state,
            load_case_name=request.load_case.name,
            config=self.config.direct_solver_config,
        )
        cache = build_element_displacement_cache(
            result.nodal_displacements,
            step_state,
            cache_name=f"{self.config.metadata_prefix}_forward_{request.load_case.name}_t{int(request.time_index)}",
        )
        return FJWElementSolveResult(
            element_displacements=cache.vectors_2d,
            source_path=cache.source_result_path,
            metadata={
                "solver_backend": "sfepy_direct",
                "solve_kind": "forward",
                "load_case_name": request.load_case.name,
                "time_index": int(request.time_index),
                "max_displacement_mm": result.max_displacement_mm,
            },
        )

    def solve_adjoint(self, request: FJWAdjointSolveRequest) -> FJWElementSolveResult:
        step_state = _build_step_workflow_state(
            request.workflow_state,
            design_cage=request.design_cage,
            obj_bo=request.obj_bo,
        )
        result = solve_fjw_direct_adjoint_case(
            step_state,
            load_case_name=request.load_case.name,
            nodal_forces_flat=request.load_vector.nodal_forces_flat,
            config=self.config.direct_solver_config,
        )
        cache = build_element_displacement_cache(
            result.nodal_displacements,
            step_state,
            cache_name=f"{self.config.metadata_prefix}_adjoint_{request.load_case.name}_t{int(request.time_index)}",
        )
        return FJWElementSolveResult(
            element_displacements=cache.vectors_2d,
            source_path=cache.source_result_path,
            metadata={
                "solver_backend": "sfepy_direct",
                "solve_kind": "adjoint",
                "load_case_name": request.load_case.name,
                "time_index": int(request.time_index),
                "active_load_node_count": int(request.load_vector.active_node_ids.size),
                "max_displacement_mm": result.max_displacement_mm,
            },
        )


@dataclass(frozen=True, slots=True)
class FJWSfePyForwardSolverAdapter(FJWForwardSolver):
    backend: FJWSfePyWorkflowSolverBackend

    def solve_forward(self, request: FJWForwardSolveRequest) -> FJWElementSolveResult:
        return self.backend.solve_forward(request)


@dataclass(frozen=True, slots=True)
class FJWSfePyAdjointSolverAdapter(FJWAdjointSolver):
    backend: FJWSfePyWorkflowSolverBackend

    def solve_adjoint(self, request: FJWAdjointSolveRequest) -> FJWElementSolveResult:
        return self.backend.solve_adjoint(request)


@dataclass(frozen=True, slots=True)
class FJWSfePyWorkflowSolverAdapters:
    backend: FJWSfePyWorkflowSolverBackend
    forward_solver: FJWSfePyForwardSolverAdapter
    adjoint_solver: FJWSfePyAdjointSolverAdapter


def build_fjw_sfepy_solver_adapters(
    config: FJWSfePyWorkflowSolverConfig | None = None,
) -> FJWSfePyWorkflowSolverAdapters:
    resolved_config = config or FJWSfePyWorkflowSolverConfig()
    backend = FJWSfePyWorkflowSolverBackend(config=resolved_config)
    return FJWSfePyWorkflowSolverAdapters(
        backend=backend,
        forward_solver=FJWSfePyForwardSolverAdapter(backend=backend),
        adjoint_solver=FJWSfePyAdjointSolverAdapter(backend=backend),
    )


__all__ = [
    "FJWSfePyAdjointSolverAdapter",
    "FJWSfePyForwardSolverAdapter",
    "FJWSfePyWorkflowSolverAdapters",
    "FJWSfePyWorkflowSolverBackend",
    "FJWSfePyWorkflowSolverConfig",
    "build_fjw_sfepy_solver_adapters",
]
