from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

from .fjw_workflow_iteration_state import (
    FJWAdjointReservation,
    FJWIterationCaseRecord,
    FJWWorkflowIterationState,
)
from .fjw_workflow_models import FJWWorkflowState
from .fjw_workflow_optimizer import (
    FJWMMAOptimizer,
    FJWMMAState,
    FJWOptimizationTerms,
    FJWOptimizer,
    build_initial_mma_state,
)
from .fjw_workflow_runtime import FJWSingleLoadCaseHistory
from .fjw_workflow_three_force import (
    FORCE_CASE_ORDER,
    FJWAdjointStepFields,
    FJWThreeForceAggregate,
    FJWThreeForceCaseResult,
    build_three_force_aggregate,
)


@dataclass(frozen=True, slots=True)
class FJWAdjointResolution:
    load_case_name: str
    steps: tuple[FJWAdjointStepFields, ...]
    reservation: FJWAdjointReservation


class FJWAdjointStepProvider(Protocol):
    provider_name: str

    def build_steps(
        self,
        *,
        load_case_name: str,
        history: FJWSingleLoadCaseHistory,
        workflow_state: FJWWorkflowState,
        iteration_state: FJWWorkflowIterationState | None = None,
    ) -> FJWAdjointResolution:
        ...


def _coerce_design(
    design: np.ndarray | None,
    workflow_state: FJWWorkflowState,
    iteration_state: FJWWorkflowIterationState | None,
) -> np.ndarray:
    if design is not None:
        return np.asarray(design, dtype=np.float64).reshape(-1).copy()
    if iteration_state is not None and iteration_state.next_design is not None:
        return iteration_state.next_design.copy()
    if iteration_state is not None:
        return iteration_state.design.copy()
    return workflow_state.initial_state.design_cage.copy()


def _coerce_mma_state(
    mma_state: FJWMMAState | None,
    *,
    design: np.ndarray,
    iteration_state: FJWWorkflowIterationState | None,
) -> FJWMMAState:
    if mma_state is not None:
        return mma_state
    if iteration_state is not None:
        return iteration_state.mma_state
    return build_initial_mma_state(design)


def _normalize_case_histories(
    case_histories: Mapping[str, FJWSingleLoadCaseHistory] | Sequence[FJWSingleLoadCaseHistory],
) -> tuple[FJWSingleLoadCaseHistory, FJWSingleLoadCaseHistory, FJWSingleLoadCaseHistory]:
    if isinstance(case_histories, Mapping):
        by_name = {str(name): history for name, history in case_histories.items()}
    else:
        by_name = {history.load_case_name: history for history in case_histories}

    missing = [name for name in FORCE_CASE_ORDER if name not in by_name]
    if missing:
        raise ValueError(f"Missing force-case histories: {missing}.")
    return tuple(by_name[name] for name in FORCE_CASE_ORDER)


def _normalize_three_force_cases(
    three_force_cases: Mapping[str, FJWThreeForceCaseResult] | Sequence[FJWThreeForceCaseResult],
) -> tuple[FJWThreeForceCaseResult, FJWThreeForceCaseResult, FJWThreeForceCaseResult]:
    if isinstance(three_force_cases, Mapping):
        by_name = {str(name): case for name, case in three_force_cases.items()}
    else:
        by_name = {case.load_case: case for case in three_force_cases}

    missing = [name for name in FORCE_CASE_ORDER if name not in by_name]
    if missing:
        raise ValueError(f"Missing three-force case results: {missing}.")
    return tuple(by_name[name] for name in FORCE_CASE_ORDER)


def _normalize_adjoint_steps(
    *,
    history: FJWSingleLoadCaseHistory,
    steps: Sequence[FJWAdjointStepFields],
    load_case_name: str,
    provider_name: str,
    workflow_state: FJWWorkflowState,
) -> FJWAdjointResolution:
    normalized_steps = tuple(steps)
    expected_step_count = len(history.time_steps)
    if len(normalized_steps) != expected_step_count:
        raise ValueError(
            f"Adjoint step count mismatch for {load_case_name}: "
            f"{len(normalized_steps)} != {expected_step_count}."
        )
    expected_field_size = int(workflow_state.mesh.element_nodes.shape[0] * 24)
    for step in normalized_steps:
        if step.U1_ele_nod_dir_P.size != expected_field_size:
            raise ValueError(
                f"Adjoint field size mismatch for {load_case_name}: "
                f"{step.U1_ele_nod_dir_P.size} != {expected_field_size}."
            )
    reservation = FJWAdjointReservation(
        load_case_name=load_case_name,
        expected_step_count=expected_step_count,
        expected_field_size=expected_field_size,
        provider_name=provider_name,
        status="resolved",
        metadata={"resolved": True},
    )
    return FJWAdjointResolution(
        load_case_name=load_case_name,
        steps=normalized_steps,
        reservation=reservation,
    )


def build_three_force_case_result_from_history(
    *,
    history: FJWSingleLoadCaseHistory,
    workflow_state: FJWWorkflowState,
    adjoint_steps: Sequence[FJWAdjointStepFields] | None = None,
    adjoint_provider: FJWAdjointStepProvider | None = None,
    iteration_state: FJWWorkflowIterationState | None = None,
) -> tuple[FJWThreeForceCaseResult, FJWAdjointReservation, str]:
    if adjoint_steps is None:
        if adjoint_provider is None:
            raise RuntimeError(
                "Missing adjoint step fields for production iteration. "
                "Run the dynamic forward/adjoint workflow or pass an explicit "
                "FJWAdjointStepProvider; static zero-Fv templates are not valid "
                "optimization inputs."
            )
        provider = adjoint_provider
        resolution = provider.build_steps(
            load_case_name=history.load_case_name,
            history=history,
            workflow_state=workflow_state,
            iteration_state=iteration_state,
        )
    else:
        resolution = _normalize_adjoint_steps(
            history=history,
            steps=adjoint_steps,
            load_case_name=history.load_case_name,
            provider_name="manual",
            workflow_state=workflow_state,
        )

    case_result = FJWThreeForceCaseResult(
        load_case=history.load_case_name,
        bo_sum=np.array([step.bo_sum_next for step in history.time_steps], dtype=np.float64),
        adjoint_steps=resolution.steps,
        obj_bo=history.terminal_obj_bo,
    )
    return case_result, resolution.reservation, resolution.reservation.provider_name


def build_iteration_case_records(
    *,
    case_histories: Mapping[str, FJWSingleLoadCaseHistory] | Sequence[FJWSingleLoadCaseHistory],
    workflow_state: FJWWorkflowState,
    adjoint_steps_by_case: Mapping[str, Sequence[FJWAdjointStepFields]] | None = None,
    adjoint_provider: FJWAdjointStepProvider | None = None,
    iteration_state: FJWWorkflowIterationState | None = None,
) -> tuple[tuple[FJWIterationCaseRecord, ...], tuple[FJWAdjointReservation, ...]]:
    ordered_histories = _normalize_case_histories(case_histories)
    steps_by_case = {} if adjoint_steps_by_case is None else {str(name): tuple(steps) for name, steps in adjoint_steps_by_case.items()}

    records: list[FJWIterationCaseRecord] = []
    reservations: list[FJWAdjointReservation] = []
    for history in ordered_histories:
        case_result, reservation, adjoint_source = build_three_force_case_result_from_history(
            history=history,
            workflow_state=workflow_state,
            adjoint_steps=steps_by_case.get(history.load_case_name),
            adjoint_provider=adjoint_provider,
            iteration_state=iteration_state,
        )
        records.append(
            FJWIterationCaseRecord(
                load_case_name=history.load_case_name,
                history=history,
                three_force_case=case_result,
                adjoint_source=adjoint_source,
                metadata={"terminal_bo_sum": history.terminal_bo_sum},
            )
        )
        reservations.append(reservation)
    return tuple(records), tuple(reservations)


def build_iteration_aggregate_terms(
    *,
    case_records: Sequence[FJWIterationCaseRecord],
    workflow_state: FJWWorkflowState,
    design: np.ndarray,
) -> FJWThreeForceAggregate:
    return build_three_force_aggregate(
        tuple(record.three_force_case for record in case_records),
        design_cage=design,
        design_element_ids=workflow_state.mesh.design_elements,
        strain_displacement_matrix=workflow_state.mesh.strain_displacement_matrix,
        constitutive_matrix=workflow_state.mesh.constitutive_matrix,
        cage_modulus_0=workflow_state.material_constants.cage_modulus_0,
        initial_design_total=workflow_state.initial_state.initial_design_total,
        voxel_volume=workflow_state.material_constants.voxel_volume,
    )


def build_iteration_optimization_terms(
    *,
    case_records: Sequence[FJWIterationCaseRecord],
    workflow_state: FJWWorkflowState,
    design: np.ndarray,
) -> FJWOptimizationTerms:
    aggregate = build_iteration_aggregate_terms(
        case_records=case_records,
        workflow_state=workflow_state,
        design=design,
    )
    return aggregate.as_optimization_terms()


def build_iteration_case_records_from_results(
    *,
    case_histories: Mapping[str, FJWSingleLoadCaseHistory] | Sequence[FJWSingleLoadCaseHistory],
    three_force_cases: Mapping[str, FJWThreeForceCaseResult] | Sequence[FJWThreeForceCaseResult],
    adjoint_reservations: Sequence[FJWAdjointReservation] = (),
) -> tuple[tuple[FJWIterationCaseRecord, ...], tuple[FJWAdjointReservation, ...]]:
    ordered_histories = _normalize_case_histories(case_histories)
    ordered_cases = _normalize_three_force_cases(three_force_cases)
    history_by_name = {history.load_case_name: history for history in ordered_histories}
    reservation_by_name = {reservation.load_case_name: reservation for reservation in adjoint_reservations}

    records: list[FJWIterationCaseRecord] = []
    reservations: list[FJWAdjointReservation] = []
    for case_result in ordered_cases:
        history = history_by_name[case_result.load_case]
        reservation = reservation_by_name.get(
            case_result.load_case,
            FJWAdjointReservation(
                load_case_name=case_result.load_case,
                expected_step_count=len(case_result.adjoint_steps),
                expected_field_size=case_result.adjoint_steps[0].U1_ele_nod_dir_P.size if case_result.adjoint_steps else 0,
                provider_name="external",
                status="resolved",
                metadata={"resolved": True},
            ),
        )
        records.append(
            FJWIterationCaseRecord(
                load_case_name=case_result.load_case,
                history=history,
                three_force_case=case_result,
                adjoint_source=reservation.provider_name,
                metadata={"terminal_bo_sum": history.terminal_bo_sum},
            )
        )
        reservations.append(reservation)
    return tuple(records), tuple(reservations)


def run_iteration_from_case_results(
    *,
    case_records: Sequence[FJWIterationCaseRecord],
    workflow_state: FJWWorkflowState,
    design: np.ndarray | None = None,
    mma_state: FJWMMAState | None = None,
    optimizer: FJWOptimizer | None = None,
    previous_iteration_state: FJWWorkflowIterationState | None = None,
    adjoint_reservations: Sequence[FJWAdjointReservation] = (),
) -> FJWWorkflowIterationState:
    current_design = _coerce_design(design, workflow_state, previous_iteration_state)
    current_mma_state = _coerce_mma_state(
        mma_state,
        design=current_design,
        iteration_state=previous_iteration_state,
    )
    aggregate = build_iteration_aggregate_terms(
        case_records=case_records,
        workflow_state=workflow_state,
        design=current_design,
    )
    optimization_terms = aggregate.as_optimization_terms()
    optimizer_impl = optimizer or FJWMMAOptimizer()
    optimizer_step = optimizer_impl.step(current_design, optimization_terms, current_mma_state)

    return FJWWorkflowIterationState(
        iteration_index=optimizer_step.state.iteration,
        design=current_design,
        mma_state=optimizer_step.state,
        case_records=tuple(case_records),
        aggregate_terms=aggregate,
        optimization_terms=optimization_terms,
        optimizer_step=optimizer_step,
        next_design=optimizer_step.design.copy(),
        adjoint_reservations=tuple(adjoint_reservations),
        metadata={
            "input_iteration": current_mma_state.iteration,
            "optimizer": optimizer_impl.__class__.__name__,
        },
    )


def run_iteration_from_three_force_cases(
    *,
    case_histories: Mapping[str, FJWSingleLoadCaseHistory] | Sequence[FJWSingleLoadCaseHistory],
    three_force_cases: Mapping[str, FJWThreeForceCaseResult] | Sequence[FJWThreeForceCaseResult],
    workflow_state: FJWWorkflowState,
    design: np.ndarray | None = None,
    mma_state: FJWMMAState | None = None,
    optimizer: FJWOptimizer | None = None,
    previous_iteration_state: FJWWorkflowIterationState | None = None,
    adjoint_reservations: Sequence[FJWAdjointReservation] = (),
) -> FJWWorkflowIterationState:
    case_records, reservations = build_iteration_case_records_from_results(
        case_histories=case_histories,
        three_force_cases=three_force_cases,
        adjoint_reservations=adjoint_reservations,
    )
    return run_iteration_from_case_results(
        case_records=case_records,
        workflow_state=workflow_state,
        design=design,
        mma_state=mma_state,
        optimizer=optimizer,
        previous_iteration_state=previous_iteration_state,
        adjoint_reservations=reservations,
    )


def run_iteration_from_histories(
    *,
    case_histories: Mapping[str, FJWSingleLoadCaseHistory] | Sequence[FJWSingleLoadCaseHistory],
    workflow_state: FJWWorkflowState,
    design: np.ndarray | None = None,
    mma_state: FJWMMAState | None = None,
    optimizer: FJWOptimizer | None = None,
    adjoint_steps_by_case: Mapping[str, Sequence[FJWAdjointStepFields]] | None = None,
    adjoint_provider: FJWAdjointStepProvider | None = None,
    previous_iteration_state: FJWWorkflowIterationState | None = None,
) -> FJWWorkflowIterationState:
    case_records, reservations = build_iteration_case_records(
        case_histories=case_histories,
        workflow_state=workflow_state,
        adjoint_steps_by_case=adjoint_steps_by_case,
        adjoint_provider=adjoint_provider,
        iteration_state=previous_iteration_state,
    )
    return run_iteration_from_case_results(
        case_records=case_records,
        workflow_state=workflow_state,
        design=design,
        mma_state=mma_state,
        optimizer=optimizer,
        previous_iteration_state=previous_iteration_state,
        adjoint_reservations=reservations,
    )


__all__ = [
    "FJWAdjointResolution",
    "FJWAdjointStepProvider",
    "build_iteration_aggregate_terms",
    "build_iteration_case_records",
    "build_iteration_case_records_from_results",
    "build_iteration_optimization_terms",
    "build_three_force_case_result_from_history",
    "run_iteration_from_case_results",
    "run_iteration_from_histories",
    "run_iteration_from_three_force_cases",
]
