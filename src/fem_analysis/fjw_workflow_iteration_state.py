from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fjw_workflow_optimizer import FJWMMAState, FJWOptimizationTerms, FJWOptimizerStepResult
from .fjw_workflow_runtime import FJWSingleLoadCaseHistory
from .fjw_workflow_three_force import FJWThreeForceAggregate, FJWThreeForceCaseResult


@dataclass(frozen=True, slots=True)
class FJWAdjointReservation:
    load_case_name: str
    expected_step_count: int
    expected_field_size: int
    provider_name: str
    status: str = "pending"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FJWIterationCaseRecord:
    load_case_name: str
    history: FJWSingleLoadCaseHistory
    three_force_case: FJWThreeForceCaseResult
    adjoint_source: str = "unresolved"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.load_case_name != self.history.load_case_name:
            raise ValueError("case record load_case_name must match history.load_case_name.")
        if self.load_case_name != self.three_force_case.load_case:
            raise ValueError("case record load_case_name must match three_force_case.load_case.")

    @property
    def terminal_bo_sum(self) -> float:
        return self.history.terminal_bo_sum


@dataclass(frozen=True, slots=True)
class FJWWorkflowIterationState:
    iteration_index: int
    design: np.ndarray
    mma_state: FJWMMAState
    case_records: tuple[FJWIterationCaseRecord, ...]
    aggregate_terms: FJWThreeForceAggregate | None = None
    optimization_terms: FJWOptimizationTerms | None = None
    optimizer_step: FJWOptimizerStepResult | None = None
    next_design: np.ndarray | None = None
    adjoint_reservations: tuple[FJWAdjointReservation, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        design = np.asarray(self.design, dtype=np.float64).reshape(-1)
        if design.size == 0:
            raise ValueError("iteration design must not be empty.")
        if design.size != self.mma_state.xold1.size:
            raise ValueError("iteration design size must match MMA state size.")

        next_design = None
        if self.next_design is not None:
            next_design = np.asarray(self.next_design, dtype=np.float64).reshape(-1)
            if next_design.size != design.size:
                raise ValueError("next_design size must match design size.")

        if self.optimization_terms is not None and self.optimization_terms.design_size != design.size:
            raise ValueError("optimization_terms design size must match iteration design size.")
        if self.aggregate_terms is not None and self.aggregate_terms.d_ob.size != design.size:
            raise ValueError("aggregate_terms design size must match iteration design size.")
        if self.optimizer_step is not None and self.optimizer_step.design.size != design.size:
            raise ValueError("optimizer_step design size must match iteration design size.")

        case_names = tuple(record.load_case_name for record in self.case_records)
        if len(case_names) != len(set(case_names)):
            raise ValueError("iteration case_records must have unique load_case_name values.")

        object.__setattr__(self, "design", design)
        object.__setattr__(self, "next_design", next_design)

    @property
    def case_histories(self) -> tuple[FJWSingleLoadCaseHistory, ...]:
        return tuple(record.history for record in self.case_records)

    @property
    def three_force_cases(self) -> tuple[FJWThreeForceCaseResult, ...]:
        return tuple(record.three_force_case for record in self.case_records)

    @property
    def has_placeholder_adjoint(self) -> bool:
        return any(record.adjoint_source in {"placeholder", "unresolved"} for record in self.case_records)


__all__ = [
    "FJWAdjointReservation",
    "FJWIterationCaseRecord",
    "FJWWorkflowIterationState",
]
