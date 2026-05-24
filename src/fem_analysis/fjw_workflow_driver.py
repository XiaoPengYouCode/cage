from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .fjw_reference import DEFAULT_REFERENCE_DIR
from .fjw_workflow_iteration import run_iteration_from_histories
from .fjw_workflow_iteration_state import FJWWorkflowIterationState
from .fjw_workflow_loaders import (
    DEFAULT_ABAQUS_INPUTS_PATH,
    DEFAULT_END1_TEMPLATE_PATH,
    DEFAULT_INPUT_INVENTORY_PATH,
    load_fjw_workflow_state,
)
from .fjw_workflow_models import FJWWorkflowState
from .fjw_workflow_optimizer import FJWMMAState, FJWOptimizer
from .fjw_workflow_runtime import FJWSingleLoadCaseHistory
from .fjw_workflow_single_case import (
    FJWAdjointSolver,
    FJWForwardSolver,
    FJWSingleCaseResult,
    run_single_case_workflow,
)
from .fjw_workflow_timing import FJWTimingRecorder, maybe_measure
from .fjw_workflow_three_force import FORCE_CASE_ORDER, FJWAdjointStepFields


@dataclass(frozen=True, slots=True)
class FJWWorkflowDriverRequest:
    workflow_state: FJWWorkflowState | None = None
    reference_dir: Path = DEFAULT_REFERENCE_DIR
    abaqus_inputs_path: Path = DEFAULT_ABAQUS_INPUTS_PATH
    input_inventory_path: Path = DEFAULT_INPUT_INVENTORY_PATH
    end1_template_path: Path = DEFAULT_END1_TEMPLATE_PATH
    initial_design_mode: str = "three_load"
    load_case_names: tuple[str, ...] = FORCE_CASE_ORDER
    design: np.ndarray | Sequence[float] | None = None
    initial_obj_bo: np.ndarray | Sequence[float] | None = None
    initial_obj_bo_by_case: Mapping[str, np.ndarray | Sequence[float]] | None = None
    num_time_steps: int | None = None
    optimizer: FJWOptimizer | None = None
    mma_state: FJWMMAState | None = None
    previous_iteration_state: FJWWorkflowIterationState | None = None
    case_parallelism: int = 1
    timing_recorder: FJWTimingRecorder | None = None

    def __post_init__(self) -> None:
        if self.initial_obj_bo is not None and self.initial_obj_bo_by_case is not None:
            raise ValueError("Provide either initial_obj_bo or initial_obj_bo_by_case, not both.")

        load_case_names = tuple(str(name) for name in self.load_case_names)
        if len(load_case_names) != len(set(load_case_names)):
            raise ValueError("load_case_names must be unique.")
        if self.case_parallelism <= 0:
            raise ValueError("case_parallelism must be positive.")

        object.__setattr__(self, "reference_dir", Path(self.reference_dir))
        object.__setattr__(self, "abaqus_inputs_path", Path(self.abaqus_inputs_path))
        object.__setattr__(self, "input_inventory_path", Path(self.input_inventory_path))
        object.__setattr__(self, "end1_template_path", Path(self.end1_template_path))
        object.__setattr__(self, "load_case_names", load_case_names)


@dataclass(frozen=True, slots=True)
class FJWWorkflowDriverResult:
    workflow_state: FJWWorkflowState
    load_case_names: tuple[str, ...]
    design: np.ndarray
    initial_obj_bo_by_case: Mapping[str, np.ndarray]
    single_case_results: tuple[FJWSingleCaseResult, ...]
    iteration_state: FJWWorkflowIterationState
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        load_case_names = tuple(str(name) for name in self.load_case_names)
        design = np.asarray(self.design, dtype=np.float64).reshape(-1)
        initial_obj_bo_by_case = {
            str(name): np.asarray(obj_bo, dtype=np.float64).reshape(-1).copy()
            for name, obj_bo in self.initial_obj_bo_by_case.items()
        }
        single_case_results = tuple(self.single_case_results)

        if len(single_case_results) != len(load_case_names):
            raise ValueError("single_case_results length must match load_case_names length.")

        expected_names = set(load_case_names)
        if set(initial_obj_bo_by_case) != expected_names:
            raise ValueError("initial_obj_bo_by_case keys must match load_case_names.")

        result_names = tuple(result.load_case_name for result in single_case_results)
        if result_names != load_case_names:
            raise ValueError("single_case_results must follow load_case_names order.")

        object.__setattr__(self, "load_case_names", load_case_names)
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "initial_obj_bo_by_case", initial_obj_bo_by_case)
        object.__setattr__(self, "single_case_results", single_case_results)

    @property
    def case_results_by_name(self) -> dict[str, FJWSingleCaseResult]:
        return {result.load_case_name: result for result in self.single_case_results}

    @property
    def case_histories(self) -> tuple[FJWSingleLoadCaseHistory, ...]:
        return tuple(
            FJWSingleLoadCaseHistory(
                load_case_name=result.load_case_name,
                time_steps=result.forward_steps,
            )
            for result in self.single_case_results
        )

    @property
    def adjoint_steps_by_case(self) -> dict[str, tuple[FJWAdjointStepFields, ...]]:
        return {
            result.load_case_name: tuple(step.step_fields for step in result.adjoint_steps)
            for result in self.single_case_results
        }


def _resolve_workflow_state(request: FJWWorkflowDriverRequest) -> FJWWorkflowState:
    if request.workflow_state is not None:
        return request.workflow_state
    return load_fjw_workflow_state(
        reference_dir=request.reference_dir,
        abaqus_inputs_path=request.abaqus_inputs_path,
        input_inventory_path=request.input_inventory_path,
        end1_template_path=request.end1_template_path,
        initial_design_mode=request.initial_design_mode,
    )


def _normalize_load_case_names(
    workflow_state: FJWWorkflowState,
    load_case_names: Sequence[str],
) -> tuple[str, str, str]:
    requested = tuple(str(name) for name in load_case_names)
    if set(requested) != set(FORCE_CASE_ORDER) or len(requested) != len(FORCE_CASE_ORDER):
        raise ValueError(
            "FJW workflow driver currently requires the archived three-force load set: "
            f"{FORCE_CASE_ORDER}."
        )

    available = {load_case.name for load_case in workflow_state.load_cases}
    missing = [name for name in FORCE_CASE_ORDER if name not in available]
    if missing:
        raise ValueError(f"workflow_state is missing load cases required by the driver: {missing}.")

    return tuple(FORCE_CASE_ORDER)


def _resolve_design(
    workflow_state: FJWWorkflowState,
    request: FJWWorkflowDriverRequest,
) -> np.ndarray:
    if request.design is not None:
        design = np.asarray(request.design, dtype=np.float64).reshape(-1).copy()
    elif request.previous_iteration_state is not None and request.previous_iteration_state.next_design is not None:
        design = request.previous_iteration_state.next_design.copy()
    elif request.previous_iteration_state is not None:
        design = request.previous_iteration_state.design.copy()
    else:
        design = workflow_state.initial_state.design_cage.copy()

    expected_size = int(workflow_state.initial_state.design_cage.size)
    if design.size != expected_size:
        raise ValueError(
            f"design size mismatch for workflow driver: {design.size} != {expected_size}."
        )
    return design


def _normalize_obj_bo(
    obj_bo: np.ndarray | Sequence[float],
    *,
    expected_size: int,
    load_case_name: str,
) -> np.ndarray:
    normalized = np.asarray(obj_bo, dtype=np.float64).reshape(-1).copy()
    if normalized.size != expected_size:
        raise ValueError(
            f"initial objective bone density size mismatch for {load_case_name}: "
            f"{normalized.size} != {expected_size}."
        )
    return normalized


def _resolve_initial_obj_bo_by_case(
    workflow_state: FJWWorkflowState,
    request: FJWWorkflowDriverRequest,
    *,
    load_case_names: Sequence[str],
) -> dict[str, np.ndarray]:
    expected_size = int(workflow_state.initial_state.obj_bo.size)

    if request.initial_obj_bo_by_case is not None:
        missing = [name for name in load_case_names if name not in request.initial_obj_bo_by_case]
        extra = [name for name in request.initial_obj_bo_by_case if name not in load_case_names]
        if missing or extra:
            raise ValueError(
                "initial_obj_bo_by_case keys must match load_case_names exactly. "
                f"missing={missing}, extra={extra}."
            )
        return {
            load_case_name: _normalize_obj_bo(
                request.initial_obj_bo_by_case[load_case_name],
                expected_size=expected_size,
                load_case_name=load_case_name,
            )
            for load_case_name in load_case_names
        }

    shared = workflow_state.initial_state.obj_bo if request.initial_obj_bo is None else request.initial_obj_bo
    shared_obj_bo = _normalize_obj_bo(
        shared,
        expected_size=expected_size,
        load_case_name="shared",
    )
    return {
        load_case_name: shared_obj_bo.copy()
        for load_case_name in load_case_names
    }


def run_fjw_workflow_iteration(
    request: FJWWorkflowDriverRequest,
    *,
    forward_solver: FJWForwardSolver,
    adjoint_solver: FJWAdjointSolver,
) -> FJWWorkflowDriverResult:
    workflow_state = _resolve_workflow_state(request)
    load_case_names = _normalize_load_case_names(workflow_state, request.load_case_names)
    design = _resolve_design(workflow_state, request)
    initial_obj_bo_by_case = _resolve_initial_obj_bo_by_case(
        workflow_state,
        request,
        load_case_names=load_case_names,
    )

    with maybe_measure(
        request.timing_recorder,
        "case_batch",
        case_parallelism=min(int(request.case_parallelism), len(load_case_names)),
    ):
        single_case_results = _run_single_case_batch(
            workflow_state=workflow_state,
            load_case_names=load_case_names,
            forward_solver=forward_solver,
            adjoint_solver=adjoint_solver,
            design=design,
            initial_obj_bo_by_case=initial_obj_bo_by_case,
            num_time_steps=request.num_time_steps,
            case_parallelism=request.case_parallelism,
        )

    case_histories = tuple(
        FJWSingleLoadCaseHistory(
            load_case_name=case_result.load_case_name,
            time_steps=case_result.forward_steps,
        )
        for case_result in single_case_results
    )
    adjoint_steps_by_case = {
        case_result.load_case_name: tuple(
            step.step_fields for step in case_result.adjoint_steps
        )
        for case_result in single_case_results
    }

    with maybe_measure(request.timing_recorder, "iteration_aggregate"):
        iteration_state = run_iteration_from_histories(
            case_histories=case_histories,
            workflow_state=workflow_state,
            design=design,
            mma_state=request.mma_state,
            optimizer=request.optimizer,
            adjoint_steps_by_case=adjoint_steps_by_case,
            previous_iteration_state=request.previous_iteration_state,
        )

    result = FJWWorkflowDriverResult(
        workflow_state=workflow_state,
        load_case_names=load_case_names,
        design=design,
        initial_obj_bo_by_case=initial_obj_bo_by_case,
        single_case_results=tuple(single_case_results),
        iteration_state=iteration_state,
        metadata={
            "workflow_state_source": "provided" if request.workflow_state is not None else "loaded",
            "initial_design_mode": request.initial_design_mode,
            "num_time_steps": request.num_time_steps or workflow_state.material_constants.num_time_steps,
            "case_parallelism": int(request.case_parallelism),
            "timing": None if request.timing_recorder is None else request.timing_recorder.as_jsonable(),
        },
    )
    return result


def _run_single_case_batch(
    *,
    workflow_state: FJWWorkflowState,
    load_case_names: Sequence[str],
    forward_solver: FJWForwardSolver,
    adjoint_solver: FJWAdjointSolver,
    design: np.ndarray,
    initial_obj_bo_by_case: Mapping[str, np.ndarray],
    num_time_steps: int | None,
    case_parallelism: int,
) -> list[FJWSingleCaseResult]:
    parallelism = min(int(case_parallelism), len(load_case_names))

    def run_case(load_case_name: str) -> FJWSingleCaseResult:
        case_timing = FJWTimingRecorder(root_name=f"case:{load_case_name}")
        with case_timing.measure("single_case", load_case_name=load_case_name):
            result = run_single_case_workflow(
                workflow_state=workflow_state,
                load_case_name=load_case_name,
                forward_solver=forward_solver,
                adjoint_solver=adjoint_solver,
                design_cage=design,
                initial_obj_bo=initial_obj_bo_by_case[load_case_name],
                num_time_steps=num_time_steps,
                timing_recorder=case_timing,
            )
        result.metadata["timing"] = case_timing.as_jsonable()
        return result

    if parallelism <= 1:
        return [run_case(load_case_name) for load_case_name in load_case_names]

    by_name: dict[str, FJWSingleCaseResult] = {}
    with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="fjw-case") as executor:
        futures = {
            executor.submit(run_case, load_case_name): load_case_name
            for load_case_name in load_case_names
        }
        for future, load_case_name in futures.items():
            by_name[load_case_name] = future.result()
    return [by_name[load_case_name] for load_case_name in load_case_names]


__all__ = [
    "FJWWorkflowDriverRequest",
    "FJWWorkflowDriverResult",
    "run_fjw_workflow_iteration",
]
