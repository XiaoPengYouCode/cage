from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .fjw_workflow_optimizer import FJWOptimizationTerms


FORCE_CASE_ORDER = ("force_1", "force_2", "force_3")


@dataclass(frozen=True, slots=True)
class FJWAdjointStepFields:
    U1_ele_nod_dir_P: np.ndarray
    U1_ele_nod_dir_Fv: np.ndarray

    def __post_init__(self) -> None:
        p_field = np.asarray(self.U1_ele_nod_dir_P, dtype=np.float64).reshape(-1)
        fv_field = np.asarray(self.U1_ele_nod_dir_Fv, dtype=np.float64).reshape(-1)
        if p_field.size != fv_field.size:
            raise ValueError("U1_ele_nod_dir_P and U1_ele_nod_dir_Fv must have the same size.")
        if p_field.size % 24 != 0:
            raise ValueError("Adjoint fields must be divisible into 24-DOF hexahedral element blocks.")
        object.__setattr__(self, "U1_ele_nod_dir_P", p_field)
        object.__setattr__(self, "U1_ele_nod_dir_Fv", fv_field)


@dataclass(frozen=True, slots=True)
class FJWThreeForceCaseResult:
    load_case: str
    bo_sum: np.ndarray
    adjoint_steps: tuple[FJWAdjointStepFields, ...]
    obj_bo: np.ndarray | None = None

    def __post_init__(self) -> None:
        bo_sum = np.asarray(self.bo_sum, dtype=np.float64).reshape(-1)
        if bo_sum.size == 0:
            raise ValueError("bo_sum must not be empty.")
        if self.obj_bo is None:
            obj_bo = None
        else:
            obj_bo = np.asarray(self.obj_bo, dtype=np.float64)
        object.__setattr__(self, "bo_sum", bo_sum)
        object.__setattr__(self, "obj_bo", obj_bo)

    @property
    def terminal_bo_sum(self) -> float:
        return float(self.bo_sum[-1])


@dataclass(frozen=True, slots=True)
class FJWThreeForceAggregate:
    objective: float
    d_ob: np.ndarray
    g2: float
    d_g2: np.ndarray
    bo_sum_by_case: Mapping[str, float]
    per_step_case_energy: tuple[dict[str, np.ndarray], ...]

    def __post_init__(self) -> None:
        d_ob = np.asarray(self.d_ob, dtype=np.float64).reshape(-1)
        d_g2 = np.asarray(self.d_g2, dtype=np.float64).reshape(-1)
        if d_ob.size != d_g2.size:
            raise ValueError("d_ob and d_g2 must have the same size.")
        object.__setattr__(self, "d_ob", d_ob)
        object.__setattr__(self, "d_g2", d_g2)

    def as_optimization_terms(self) -> FJWOptimizationTerms:
        return FJWOptimizationTerms(
            objective=float(self.objective),
            objective_gradient=self.d_ob.copy(),
            constraints=np.array([self.g2], dtype=np.float64),
            constraint_gradients=self.d_g2.reshape(1, -1),
            constraint_names=("g2",),
            metadata={
                "bo_sum_by_case": dict(self.bo_sum_by_case),
                "per_step_case_energy": [
                    {key: value.copy() for key, value in step.items()}
                    for step in self.per_step_case_energy
                ],
            },
        )


def _validate_force_cases(
    force_cases: Sequence[FJWThreeForceCaseResult],
) -> tuple[FJWThreeForceCaseResult, FJWThreeForceCaseResult, FJWThreeForceCaseResult]:
    if len(force_cases) != 3:
        raise ValueError("Three-force aggregation expects exactly 3 force cases.")
    by_name = {case.load_case: case for case in force_cases}
    missing = [name for name in FORCE_CASE_ORDER if name not in by_name]
    if missing:
        raise ValueError(f"Missing force cases: {missing}.")
    return tuple(by_name[name] for name in FORCE_CASE_ORDER)


def _build_element_kernel(
    strain_displacement_matrix: np.ndarray,
    constitutive_matrix: np.ndarray,
) -> np.ndarray:
    B = np.asarray(strain_displacement_matrix, dtype=np.float64)
    D = np.asarray(constitutive_matrix, dtype=np.float64)
    return B.T @ D @ B


def _extract_element_blocks(
    field: np.ndarray,
    element_ids_1_based: np.ndarray,
) -> np.ndarray:
    blocks = np.asarray(field, dtype=np.float64).reshape(-1, 24)
    element_ids = np.asarray(element_ids_1_based, dtype=np.int64).reshape(-1)
    if np.any(element_ids < 1) or np.any(element_ids > blocks.shape[0]):
        raise ValueError("element_ids contain out-of-range 1-based ids.")
    return blocks[element_ids - 1]


def compute_three_force_objective(
    force_cases: Sequence[FJWThreeForceCaseResult],
) -> float:
    ordered_cases = _validate_force_cases(force_cases)
    return -sum(case.terminal_bo_sum for case in ordered_cases)


def compute_three_force_d_ob(
    force_cases: Sequence[FJWThreeForceCaseResult],
    *,
    design_cage: np.ndarray,
    design_element_ids: np.ndarray,
    strain_displacement_matrix: np.ndarray,
    constitutive_matrix: np.ndarray,
    cage_modulus_0: float,
) -> tuple[np.ndarray, tuple[dict[str, np.ndarray], ...]]:
    ordered_cases = _validate_force_cases(force_cases)
    design_cage = np.asarray(design_cage, dtype=np.float64).reshape(-1)
    design_element_ids = np.asarray(design_element_ids, dtype=np.int64).reshape(-1)
    if design_cage.size != design_element_ids.size:
        raise ValueError("design_cage and design_element_ids must have the same size.")

    step_count = len(ordered_cases[0].adjoint_steps)
    for case in ordered_cases[1:]:
        if len(case.adjoint_steps) != step_count:
            raise ValueError("All force cases must carry the same number of adjoint steps.")

    element_kernel = _build_element_kernel(strain_displacement_matrix, constitutive_matrix)
    modulus_derivative = 3.0 * float(cage_modulus_0) * np.square(design_cage)
    d_ob = np.zeros_like(design_cage)
    per_step_case_energy: list[dict[str, np.ndarray]] = []

    for step_index in range(step_count):
        step_energy: dict[str, np.ndarray] = {}
        for case in ordered_cases:
            step_fields = case.adjoint_steps[step_index]
            p_blocks = _extract_element_blocks(step_fields.U1_ele_nod_dir_P, design_element_ids)
            fv_blocks = _extract_element_blocks(step_fields.U1_ele_nod_dir_Fv, design_element_ids)
            elemental_energy = np.einsum("bi,ij,bj->b", fv_blocks, element_kernel, p_blocks, optimize=True)
            step_energy[case.load_case] = elemental_energy
            d_ob += elemental_energy * modulus_derivative
        per_step_case_energy.append(step_energy)

    return d_ob, tuple(per_step_case_energy)


def compute_g2_terms(
    design_cage: np.ndarray,
    *,
    initial_design_total: float,
    voxel_volume: float,
) -> tuple[float, np.ndarray]:
    design_cage = np.asarray(design_cage, dtype=np.float64).reshape(-1)
    g2 = float(np.sum(design_cage) - float(initial_design_total))
    d_g2 = np.full(design_cage.shape, float(voxel_volume), dtype=np.float64)
    return g2, d_g2


def build_three_force_aggregate(
    force_cases: Sequence[FJWThreeForceCaseResult],
    *,
    design_cage: np.ndarray,
    design_element_ids: np.ndarray,
    strain_displacement_matrix: np.ndarray,
    constitutive_matrix: np.ndarray,
    cage_modulus_0: float,
    initial_design_total: float,
    voxel_volume: float,
) -> FJWThreeForceAggregate:
    ordered_cases = _validate_force_cases(force_cases)
    objective = compute_three_force_objective(ordered_cases)
    d_ob, per_step_case_energy = compute_three_force_d_ob(
        ordered_cases,
        design_cage=design_cage,
        design_element_ids=design_element_ids,
        strain_displacement_matrix=strain_displacement_matrix,
        constitutive_matrix=constitutive_matrix,
        cage_modulus_0=cage_modulus_0,
    )
    g2, d_g2 = compute_g2_terms(
        design_cage,
        initial_design_total=initial_design_total,
        voxel_volume=voxel_volume,
    )
    return FJWThreeForceAggregate(
        objective=objective,
        d_ob=d_ob,
        g2=g2,
        d_g2=d_g2,
        bo_sum_by_case={case.load_case: case.terminal_bo_sum for case in ordered_cases},
        per_step_case_energy=per_step_case_energy,
    )


__all__ = [
    "FJWAdjointStepFields",
    "FJWThreeForceAggregate",
    "FJWThreeForceCaseResult",
    "FORCE_CASE_ORDER",
    "build_three_force_aggregate",
    "compute_g2_terms",
    "compute_three_force_d_ob",
    "compute_three_force_objective",
]
