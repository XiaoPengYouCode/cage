from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

import numpy as np

from .fjw_direct_solver import (
    FJWDirectSolverConfig,
    FJWDirectProblemSetup,
    build_fjw_direct_problem_setup,
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
    enable_setup_cache: bool = True


@dataclass(slots=True)
class FJWSfePyWorkflowSolverBackend:
    config: FJWSfePyWorkflowSolverConfig
    _setup_cache: dict[tuple[int, str, bytes, bytes], FJWDirectProblemSetup] = field(default_factory=dict, init=False)
    _cache_lock: Lock = field(default_factory=Lock, init=False)

    def solve_forward(self, request: FJWForwardSolveRequest) -> FJWElementSolveResult:
        step_state = _build_step_workflow_state(
            request.workflow_state,
            design_cage=request.design_cage,
            obj_bo=request.obj_bo,
        )
        setup, cache_hit = self._get_setup(
            step_state,
            load_case_name=request.load_case.name,
            design_cage=request.design_cage,
            obj_bo=request.obj_bo,
        )
        result = solve_fjw_direct_case(
            step_state,
            load_case_name=request.load_case.name,
            config=self.config.direct_solver_config,
            setup=setup,
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
                "setup_cache_hit": cache_hit,
            },
        )

    def solve_adjoint(self, request: FJWAdjointSolveRequest) -> FJWElementSolveResult:
        step_state = _build_step_workflow_state(
            request.workflow_state,
            design_cage=request.design_cage,
            obj_bo=request.obj_bo,
        )
        setup, cache_hit = self._get_setup(
            step_state,
            load_case_name=request.load_case.name,
            design_cage=request.design_cage,
            obj_bo=request.obj_bo,
        )
        result = solve_fjw_direct_adjoint_case(
            step_state,
            load_case_name=request.load_case.name,
            nodal_forces_flat=request.load_vector.nodal_forces_flat,
            config=self.config.direct_solver_config,
            setup=setup,
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
                "setup_cache_hit": cache_hit,
            },
        )

    def _get_setup(
        self,
        workflow_state,
        *,
        load_case_name: str,
        design_cage: np.ndarray,
        obj_bo: np.ndarray,
    ) -> tuple[FJWDirectProblemSetup | None, bool]:
        if not self.config.enable_setup_cache:
            return None, False

        key = (
            id(workflow_state.mesh),
            str(load_case_name),
            np.ascontiguousarray(design_cage, dtype=np.float64).tobytes(),
            np.ascontiguousarray(obj_bo, dtype=np.float64).tobytes(),
        )
        with self._cache_lock:
            setup = self._setup_cache.get(key)
            if setup is not None:
                return setup, True
            setup = build_fjw_direct_problem_setup(
                workflow_state,
                load_case_name=load_case_name,
                design_cage=design_cage,
                obj_bo=obj_bo,
            )
            self._setup_cache[key] = setup
            return setup, False


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
