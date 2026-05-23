from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from .fjw_workflow_adjoint import (
    FJWAdjointLoadVector,
    FJWAdjointStepState,
    build_adjoint_step_state,
    build_fv_load_vector_from_forward_step,
    build_terminal_fai,
)
from .fjw_workflow_forward import FJWSingleLoadTimeStepResult, build_single_load_time_step_result
from .fjw_workflow_models import FJWLoadCase, FJWWorkflowState
from .fjw_workflow_three_force import FJWAdjointStepFields, FJWThreeForceCaseResult


@dataclass(frozen=True, slots=True)
class FJWForwardSolveRequest:
    workflow_state: FJWWorkflowState
    load_case: FJWLoadCase
    time_index: int
    design_cage: np.ndarray
    obj_bo: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "design_cage", np.asarray(self.design_cage, dtype=np.float64).reshape(-1))
        object.__setattr__(self, "obj_bo", np.asarray(self.obj_bo, dtype=np.float64).reshape(-1))


@dataclass(frozen=True, slots=True)
class FJWAdjointSolveRequest:
    workflow_state: FJWWorkflowState
    load_case: FJWLoadCase
    time_index: int
    design_cage: np.ndarray
    obj_bo: np.ndarray
    load_vector: FJWAdjointLoadVector

    def __post_init__(self) -> None:
        object.__setattr__(self, "design_cage", np.asarray(self.design_cage, dtype=np.float64).reshape(-1))
        object.__setattr__(self, "obj_bo", np.asarray(self.obj_bo, dtype=np.float64).reshape(-1))


@dataclass(frozen=True, slots=True)
class FJWElementSolveResult:
    element_displacements: np.ndarray
    source_path: Path | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        element_displacements = np.asarray(self.element_displacements, dtype=np.float64)
        if element_displacements.ndim == 1:
            if element_displacements.size % 24 != 0:
                raise ValueError("Flattened element_displacements length must be divisible by 24.")
            element_displacements = element_displacements.reshape(-1, 24)
        if element_displacements.ndim != 2 or element_displacements.shape[1] != 24:
            raise ValueError("element_displacements must have shape (n_elements, 24).")
        object.__setattr__(self, "element_displacements", element_displacements)


class FJWForwardSolver(Protocol):
    def solve_forward(self, request: FJWForwardSolveRequest) -> FJWElementSolveResult:
        ...


class FJWAdjointSolver(Protocol):
    def solve_adjoint(self, request: FJWAdjointSolveRequest) -> FJWElementSolveResult:
        ...


@dataclass(frozen=True, slots=True)
class FJWSingleCaseAdjointStep:
    state: FJWAdjointStepState
    forward_element_displacements: np.ndarray
    solve_request: FJWAdjointSolveRequest
    solve_result: FJWElementSolveResult

    def __post_init__(self) -> None:
        forward_element_displacements = np.asarray(self.forward_element_displacements, dtype=np.float64)
        if forward_element_displacements.ndim == 1:
            if forward_element_displacements.size % 24 != 0:
                raise ValueError("Flattened forward_element_displacements length must be divisible by 24.")
            forward_element_displacements = forward_element_displacements.reshape(-1, 24)
        if forward_element_displacements.ndim != 2 or forward_element_displacements.shape[1] != 24:
            raise ValueError("forward_element_displacements must have shape (n_elements, 24).")
        if forward_element_displacements.shape != self.solve_result.element_displacements.shape:
            raise ValueError("forward_element_displacements must match adjoint solve_result element shape.")
        object.__setattr__(self, "forward_element_displacements", forward_element_displacements)

    @property
    def time_index(self) -> int:
        return self.state.time_index

    @property
    def step_fields(self) -> FJWAdjointStepFields:
        forward_field = self.forward_element_displacements.reshape(-1)
        adjoint_field = self.solve_result.element_displacements.reshape(-1)
        return FJWAdjointStepFields(
            U1_ele_nod_dir_P=forward_field,
            U1_ele_nod_dir_Fv=adjoint_field,
        )


@dataclass(frozen=True, slots=True)
class FJWSingleCaseResult:
    load_case_name: str
    design_cage: np.ndarray
    obj_bo_history: np.ndarray
    bo_sum_history: np.ndarray
    forward_steps: tuple[FJWSingleLoadTimeStepResult, ...]
    adjoint_steps: tuple[FJWSingleCaseAdjointStep, ...]
    fai_history: np.ndarray
    initial_compliance: float
    initial_design_sensitivity: np.ndarray
    bone_reference_compliance: float

    def __post_init__(self) -> None:
        design_cage = np.asarray(self.design_cage, dtype=np.float64).reshape(-1)
        obj_bo_history = np.asarray(self.obj_bo_history, dtype=np.float64)
        bo_sum_history = np.asarray(self.bo_sum_history, dtype=np.float64).reshape(-1)
        fai_history = np.asarray(self.fai_history, dtype=np.float64)
        initial_design_sensitivity = np.asarray(self.initial_design_sensitivity, dtype=np.float64).reshape(-1)

        if obj_bo_history.ndim != 2:
            raise ValueError("obj_bo_history must have shape (time+1, obj_num).")
        if bo_sum_history.size != obj_bo_history.shape[0]:
            raise ValueError("bo_sum_history length must match obj_bo_history time dimension.")
        if fai_history.ndim != 3 or fai_history.shape[2] != 24:
            raise ValueError("fai_history must have shape (time+1, obj_num, 24).")
        if fai_history.shape[:2] != obj_bo_history.shape:
            raise ValueError("fai_history leading dimensions must match obj_bo_history shape.")
        if initial_design_sensitivity.size != design_cage.size:
            raise ValueError("initial_design_sensitivity size must match design_cage size.")

        object.__setattr__(self, "design_cage", design_cage)
        object.__setattr__(self, "obj_bo_history", obj_bo_history)
        object.__setattr__(self, "bo_sum_history", bo_sum_history)
        object.__setattr__(self, "fai_history", fai_history)
        object.__setattr__(self, "initial_design_sensitivity", initial_design_sensitivity)

    @property
    def terminal_bo_sum(self) -> float:
        return float(self.bo_sum_history[-1])

    def to_three_force_case_result(self) -> FJWThreeForceCaseResult:
        ordered_adjoint_steps = tuple(sorted(self.adjoint_steps, key=lambda step: step.time_index))
        return FJWThreeForceCaseResult(
            load_case=self.load_case_name,
            bo_sum=self.bo_sum_history[1:].copy(),
            adjoint_steps=tuple(step.step_fields for step in ordered_adjoint_steps),
            obj_bo=self.obj_bo_history[-1].copy(),
        )


def _coerce_load_case(workflow_state: FJWWorkflowState, load_case_name: str) -> FJWLoadCase:
    for load_case in workflow_state.load_cases:
        if load_case.name == load_case_name:
            return load_case
    available = ", ".join(load_case.name for load_case in workflow_state.load_cases)
    raise ValueError(f"Unknown load_case_name {load_case_name!r}. Available: {available}.")


def _initial_design_cage(
    workflow_state: FJWWorkflowState,
    design_cage: np.ndarray | Sequence[float] | None,
) -> np.ndarray:
    if design_cage is None:
        return np.asarray(workflow_state.initial_state.design_cage, dtype=np.float64).copy()
    return np.asarray(design_cage, dtype=np.float64).reshape(-1).copy()


def _initial_obj_bo(
    workflow_state: FJWWorkflowState,
    obj_bo: np.ndarray | Sequence[float] | None,
) -> np.ndarray:
    if obj_bo is None:
        return np.asarray(workflow_state.initial_state.obj_bo, dtype=np.float64).copy()
    return np.asarray(obj_bo, dtype=np.float64).reshape(-1).copy()


def run_single_case_workflow(
    *,
    workflow_state: FJWWorkflowState,
    load_case_name: str,
    forward_solver: FJWForwardSolver,
    adjoint_solver: FJWAdjointSolver,
    design_cage: np.ndarray | Sequence[float] | None = None,
    initial_obj_bo: np.ndarray | Sequence[float] | None = None,
    num_time_steps: int | None = None,
) -> FJWSingleCaseResult:
    load_case = _coerce_load_case(workflow_state, load_case_name)
    material_constants = workflow_state.material_constants
    mesh = workflow_state.mesh
    design_density = _initial_design_cage(workflow_state, design_cage)
    current_obj_bo = _initial_obj_bo(workflow_state, initial_obj_bo)
    step_count = (
        int(material_constants.num_time_steps)
        if num_time_steps is None
        else int(num_time_steps)
    )
    if step_count <= 0:
        raise ValueError("num_time_steps must be positive.")
    objective_count = current_obj_bo.size

    obj_bo_history = np.zeros((step_count + 1, objective_count), dtype=np.float64)
    bo_sum_history = np.zeros(step_count + 1, dtype=np.float64)
    obj_bo_history[0] = current_obj_bo
    bo_sum_history[0] = float(np.sum(current_obj_bo, dtype=np.float64))

    forward_steps: list[FJWSingleLoadTimeStepResult] = []
    for time_index in range(step_count):
        forward_request = FJWForwardSolveRequest(
            workflow_state=workflow_state,
            load_case=load_case,
            time_index=time_index,
            design_cage=design_density,
            obj_bo=current_obj_bo,
        )
        forward_solve_result = forward_solver.solve_forward(forward_request)
        step_result = build_single_load_time_step_result(
            load_case_name=load_case_name,
            time_index=time_index,
            element_displacements=forward_solve_result.element_displacements,
            mesh=mesh,
            material_constants=material_constants,
            design_cage=design_density,
            obj_bo=current_obj_bo,
        )
        forward_steps.append(step_result)
        current_obj_bo = step_result.obj_bo_next.copy()
        obj_bo_history[time_index + 1] = current_obj_bo
        bo_sum_history[time_index + 1] = step_result.bo_sum_next

    fai_history = np.zeros((step_count + 1, objective_count, 24), dtype=np.float64)
    fai_history[step_count] = build_terminal_fai(objective_count)
    adjoint_steps: list[FJWSingleCaseAdjointStep] = []

    for time_index in range(step_count - 1, -1, -1):
        forward_step = forward_steps[time_index]
        adjoint_load_vector = build_fv_load_vector_from_forward_step(
            forward_step,
            fai_next=fai_history[time_index + 1],
            mesh=mesh,
            material_constants=material_constants,
        )
        adjoint_request = FJWAdjointSolveRequest(
            workflow_state=workflow_state,
            load_case=load_case,
            time_index=time_index,
            design_cage=design_density,
            obj_bo=forward_step.obj_bo_previous,
            load_vector=adjoint_load_vector,
        )
        adjoint_solve_result = adjoint_solver.solve_adjoint(adjoint_request)
        adjoint_state = build_adjoint_step_state(
            forward_step=forward_step,
            adjoint_element_displacements=adjoint_solve_result.element_displacements,
            fai_next=fai_history[time_index + 1],
            mesh=mesh,
            material_constants=material_constants,
        )
        fai_history[time_index] = adjoint_state.fai_current
        adjoint_steps.append(
            FJWSingleCaseAdjointStep(
                state=adjoint_state,
                forward_element_displacements=forward_step.full_element_displacements,
                solve_request=adjoint_request,
                solve_result=adjoint_solve_result,
            )
        )

    ordered_adjoint_steps = tuple(sorted(adjoint_steps, key=lambda step: step.time_index))
    return FJWSingleCaseResult(
        load_case_name=load_case_name,
        design_cage=design_density,
        obj_bo_history=obj_bo_history,
        bo_sum_history=bo_sum_history,
        forward_steps=tuple(forward_steps),
        adjoint_steps=ordered_adjoint_steps,
        fai_history=fai_history,
        initial_compliance=forward_steps[0].compliance,
        initial_design_sensitivity=forward_steps[0].design_sensitivity,
        bone_reference_compliance=forward_steps[0].bone_reference_compliance,
    )


__all__ = [
    "FJWAdjointSolveRequest",
    "FJWAdjointSolver",
    "FJWElementSolveResult",
    "FJWForwardSolveRequest",
    "FJWForwardSolver",
    "FJWSingleCaseAdjointStep",
    "FJWSingleCaseResult",
    "run_single_case_workflow",
]
