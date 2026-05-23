from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fjw_workflow_biology import (
    FJWBoneBiologyStep,
    MIN_BONE_DENSITY,
    advance_bone_density,
)
from .fjw_workflow_models import FJWMaterialConstants, FJWReferenceMeshContext


@dataclass(frozen=True, slots=True)
class FJWInitialComplianceResult:
    compliance: float
    design_sensitivity: np.ndarray
    bone_reference_compliance: float


@dataclass(frozen=True, slots=True)
class FJWSingleLoadTimeStepResult:
    load_case_name: str
    time_index: int
    full_element_displacements: np.ndarray
    design_element_displacements: np.ndarray
    objective_element_displacements: np.ndarray
    cage_modulus: np.ndarray
    objective_modulus: np.ndarray
    design_quadratic_terms: np.ndarray
    objective_quadratic_terms: np.ndarray
    compliance: float
    design_sensitivity: np.ndarray
    bone_reference_compliance: float
    bone_s: np.ndarray
    bone_density_delta: np.ndarray
    obj_bo_previous: np.ndarray
    obj_bo_next: np.ndarray
    bo_sum_previous: float
    bo_sum_next: float


def _as_element_displacement_matrix(
    element_displacements: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    displacement_array = np.asarray(element_displacements, dtype=np.float64)
    if displacement_array.ndim == 1:
        if displacement_array.size % 24 != 0:
            raise ValueError("Flattened element_displacements length must be divisible by 24.")
        return displacement_array.reshape(-1, 24)
    if displacement_array.ndim == 2 and displacement_array.shape[1] == 24:
        return displacement_array
    raise ValueError("element_displacements must have shape (n_elements, 24) or (n_elements * 24,).")


def _element_quadratic_terms(
    element_displacements: np.ndarray,
    strain_displacement_matrix: np.ndarray,
    constitutive_matrix: np.ndarray,
) -> np.ndarray:
    element_matrix = np.asarray(element_displacements, dtype=np.float64)
    if element_matrix.ndim != 2 or element_matrix.shape[1] != 24:
        raise ValueError("element_displacements must have shape (n_elements, 24).")

    B = np.asarray(strain_displacement_matrix, dtype=np.float64)
    D = np.asarray(constitutive_matrix, dtype=np.float64)
    if B.shape != (6, 24):
        raise ValueError(f"Expected B to have shape (6, 24), got {B.shape}.")
    if D.shape != (6, 6):
        raise ValueError(f"Expected D to have shape (6, 6), got {D.shape}.")

    stiffness_like = B.T @ D @ B
    return 0.5 * np.einsum("bi,ij,bj->b", element_matrix, stiffness_like, element_matrix)


def cage_objective_modulus(
    design_cage: np.ndarray | list[float] | tuple[float, ...],
    material_constants: FJWMaterialConstants,
) -> np.ndarray:
    design = np.clip(np.asarray(design_cage, dtype=np.float64).reshape(-1), MIN_BONE_DENSITY, 1.0)
    modulus = material_constants.cage_modulus_min + (
        material_constants.cage_modulus_0 * np.power(design, 3)
    )
    return np.minimum(modulus, material_constants.cage_modulus_0)


def compute_initial_compliance(
    design_element_displacements: np.ndarray | list[float] | tuple[float, ...],
    objective_element_displacements: np.ndarray | list[float] | tuple[float, ...],
    design_cage: np.ndarray | list[float] | tuple[float, ...],
    obj_bo: np.ndarray | list[float] | tuple[float, ...],
    *,
    strain_displacement_matrix: np.ndarray,
    constitutive_matrix: np.ndarray,
    material_constants: FJWMaterialConstants,
) -> FJWInitialComplianceResult:
    design_displacements = _as_element_displacement_matrix(design_element_displacements)
    objective_displacements = _as_element_displacement_matrix(objective_element_displacements)
    design_density = np.clip(np.asarray(design_cage, dtype=np.float64).reshape(-1), MIN_BONE_DENSITY, 1.0)
    objective_density = np.clip(
        np.asarray(obj_bo, dtype=np.float64).reshape(-1),
        MIN_BONE_DENSITY,
        material_constants.bone_density_upper_bound,
    )

    if design_displacements.shape[0] != design_density.size:
        raise ValueError("design_element_displacements and design_cage size do not match.")
    if objective_displacements.shape[0] != objective_density.size:
        raise ValueError("objective_element_displacements and obj_bo size do not match.")

    design_quadratic_terms = _element_quadratic_terms(
        design_displacements,
        strain_displacement_matrix,
        constitutive_matrix,
    )
    objective_quadratic_terms = _element_quadratic_terms(
        objective_displacements,
        strain_displacement_matrix,
        constitutive_matrix,
    )

    cage_modulus = cage_objective_modulus(design_density, material_constants)
    objective_modulus = (
        material_constants.bone_modulus_min
        + material_constants.bone_modulus_0
        * np.power(objective_density / material_constants.bone_density_upper_bound, 3)
    )
    objective_modulus = np.minimum(objective_modulus, material_constants.bone_modulus_0)

    compliance = float(np.sum(design_quadratic_terms * cage_modulus, dtype=np.float64))
    design_sensitivity = -design_quadratic_terms * (
        3.0 * material_constants.cage_modulus_0 * np.power(design_density, 2)
    )
    bone_reference_compliance = float(
        1.5 * np.sum(objective_quadratic_terms * objective_modulus, dtype=np.float64)
    )
    return FJWInitialComplianceResult(
        compliance=compliance,
        design_sensitivity=design_sensitivity,
        bone_reference_compliance=bone_reference_compliance,
    )


def build_single_load_time_step_result(
    *,
    load_case_name: str,
    time_index: int,
    element_displacements: np.ndarray | list[float] | tuple[float, ...],
    mesh: FJWReferenceMeshContext,
    material_constants: FJWMaterialConstants,
    design_cage: np.ndarray | list[float] | tuple[float, ...],
    obj_bo: np.ndarray | list[float] | tuple[float, ...],
) -> FJWSingleLoadTimeStepResult:
    full_element_displacements = _as_element_displacement_matrix(element_displacements)
    num_elements = mesh.element_nodes.shape[0]
    if full_element_displacements.shape[0] != num_elements:
        raise ValueError(
            "element_displacements element count does not match mesh.element_nodes: "
            f"{full_element_displacements.shape[0]} != {num_elements}."
        )

    design_ids = np.asarray(mesh.design_elements, dtype=np.int64).reshape(-1) - 1
    objective_ids = np.asarray(mesh.objective_elements, dtype=np.int64).reshape(-1) - 1
    design_element_displacements = full_element_displacements[design_ids]
    objective_element_displacements = full_element_displacements[objective_ids]

    compliance_result = compute_initial_compliance(
        design_element_displacements,
        objective_element_displacements,
        design_cage,
        obj_bo,
        strain_displacement_matrix=mesh.strain_displacement_matrix,
        constitutive_matrix=mesh.constitutive_matrix,
        material_constants=material_constants,
    )

    objective_quadratic_terms = _element_quadratic_terms(
        objective_element_displacements,
        mesh.strain_displacement_matrix,
        mesh.constitutive_matrix,
    )
    biology_step: FJWBoneBiologyStep = advance_bone_density(
        objective_quadratic_terms,
        obj_bo,
        bone_modulus_0=material_constants.bone_modulus_0,
        bone_modulus_min=material_constants.bone_modulus_min,
        bone_density_upper_bound=material_constants.bone_density_upper_bound,
        time_step_dt=material_constants.time_step_dt,
    )
    design_density = np.clip(np.asarray(design_cage, dtype=np.float64).reshape(-1), MIN_BONE_DENSITY, 1.0)
    design_quadratic_terms = _element_quadratic_terms(
        design_element_displacements,
        mesh.strain_displacement_matrix,
        mesh.constitutive_matrix,
    )

    previous_density = np.clip(
        np.asarray(obj_bo, dtype=np.float64).reshape(-1),
        MIN_BONE_DENSITY,
        material_constants.bone_density_upper_bound,
    )

    return FJWSingleLoadTimeStepResult(
        load_case_name=str(load_case_name),
        time_index=int(time_index),
        full_element_displacements=full_element_displacements,
        design_element_displacements=design_element_displacements,
        objective_element_displacements=objective_element_displacements,
        cage_modulus=cage_objective_modulus(design_density, material_constants),
        objective_modulus=biology_step.objective_modulus,
        design_quadratic_terms=design_quadratic_terms,
        objective_quadratic_terms=objective_quadratic_terms,
        compliance=compliance_result.compliance,
        design_sensitivity=compliance_result.design_sensitivity,
        bone_reference_compliance=compliance_result.bone_reference_compliance,
        bone_s=biology_step.stimulus,
        bone_density_delta=biology_step.density_delta,
        obj_bo_previous=previous_density,
        obj_bo_next=biology_step.next_density,
        bo_sum_previous=float(np.sum(previous_density, dtype=np.float64)),
        bo_sum_next=biology_step.bo_sum,
    )


__all__ = [
    "FJWInitialComplianceResult",
    "FJWSingleLoadTimeStepResult",
    "build_single_load_time_step_result",
    "cage_objective_modulus",
    "compute_initial_compliance",
]
