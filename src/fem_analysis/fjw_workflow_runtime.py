from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .fjw_workflow_forward import FJWSingleLoadTimeStepResult, build_single_load_time_step_result
from .fjw_workflow_models import FJWWorkflowState
from .fjw_workflow_vectors import FJWElementDisplacementVectorCache


@dataclass(frozen=True, slots=True)
class FJWSingleLoadCaseHistory:
    load_case_name: str
    time_steps: tuple[FJWSingleLoadTimeStepResult, ...]

    @property
    def terminal_bo_sum(self) -> float:
        if not self.time_steps:
            raise ValueError("Single-load history is empty.")
        return float(self.time_steps[-1].bo_sum_next)

    @property
    def terminal_obj_bo(self) -> np.ndarray:
        if not self.time_steps:
            raise ValueError("Single-load history is empty.")
        return self.time_steps[-1].obj_bo_next.copy()


def _coerce_cache_sequence(
    caches_or_vectors: Sequence[FJWElementDisplacementVectorCache | np.ndarray],
) -> tuple[FJWElementDisplacementVectorCache | np.ndarray, ...]:
    if not caches_or_vectors:
        raise ValueError("At least one time-step displacement cache is required.")
    return tuple(caches_or_vectors)


def _element_displacements_2d(
    cache_or_vectors: FJWElementDisplacementVectorCache | np.ndarray,
) -> np.ndarray:
    if isinstance(cache_or_vectors, FJWElementDisplacementVectorCache):
        return cache_or_vectors.vectors_2d
    vectors = np.asarray(cache_or_vectors, dtype=np.float64)
    if vectors.ndim == 1:
        if vectors.size % 24 != 0:
            raise ValueError("Flattened element displacement length must be divisible by 24.")
        return vectors.reshape(-1, 24)
    if vectors.ndim != 2 or vectors.shape[1] != 24:
        raise ValueError("Element displacement vectors must have shape (N, 24).")
    return vectors


def build_single_load_case_history(
    *,
    load_case_name: str,
    time_step_caches: Sequence[FJWElementDisplacementVectorCache | np.ndarray],
    workflow_state: FJWWorkflowState,
    initial_obj_bo: np.ndarray | None = None,
) -> FJWSingleLoadCaseHistory:
    caches = _coerce_cache_sequence(time_step_caches)
    obj_bo = (
        workflow_state.initial_state.obj_bo.copy()
        if initial_obj_bo is None
        else np.asarray(initial_obj_bo, dtype=np.float64).reshape(-1).copy()
    )
    design_cage = workflow_state.initial_state.design_cage.copy()

    time_steps: list[FJWSingleLoadTimeStepResult] = []
    for time_index, cache in enumerate(caches):
        step_result = build_single_load_time_step_result(
            load_case_name=load_case_name,
            time_index=time_index,
            element_displacements=_element_displacements_2d(cache),
            mesh=workflow_state.mesh,
            material_constants=workflow_state.material_constants,
            design_cage=design_cage,
            obj_bo=obj_bo,
        )
        time_steps.append(step_result)
        obj_bo = step_result.obj_bo_next.copy()

    return FJWSingleLoadCaseHistory(
        load_case_name=load_case_name,
        time_steps=tuple(time_steps),
    )


__all__ = [
    "FJWSingleLoadCaseHistory",
    "build_single_load_case_history",
]
