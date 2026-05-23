from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fjw_workflow_biology import MIN_BONE_DENSITY, bone_objective_modulus, d_bone_delta
from .fjw_workflow_forward import FJWSingleLoadTimeStepResult
from .fjw_workflow_models import FJWMaterialConstants, FJWReferenceMeshContext


@dataclass(frozen=True, slots=True)
class FJWAdjointLoadVector:
    time_index: int
    load_case_name: str
    nodal_forces_flat: np.ndarray
    active_node_ids: np.ndarray
    active_forces_xyz: np.ndarray

    def __post_init__(self) -> None:
        nodal_forces_flat = np.asarray(self.nodal_forces_flat, dtype=np.float64).reshape(-1)
        active_node_ids = np.asarray(self.active_node_ids, dtype=np.int32).reshape(-1)
        active_forces_xyz = np.asarray(self.active_forces_xyz, dtype=np.float64)

        if nodal_forces_flat.size % 3 != 0:
            raise ValueError("nodal_forces_flat must contain 3 DOFs per node.")
        if active_forces_xyz.ndim != 2 or active_forces_xyz.shape[1] != 3:
            raise ValueError("active_forces_xyz must have shape (N, 3).")
        if active_node_ids.shape[0] != active_forces_xyz.shape[0]:
            raise ValueError("active_node_ids and active_forces_xyz must have the same length.")
        if active_node_ids.size and np.any(active_node_ids <= 0):
            raise ValueError("active_node_ids must contain positive 1-based node ids.")

        object.__setattr__(self, "nodal_forces_flat", nodal_forces_flat)
        object.__setattr__(self, "active_node_ids", active_node_ids)
        object.__setattr__(self, "active_forces_xyz", active_forces_xyz)

    @property
    def nodal_forces_xyz(self) -> np.ndarray:
        return self.nodal_forces_flat.reshape(-1, 3)

    def to_abaqus_table(self) -> np.ndarray:
        if self.active_node_ids.size == 0:
            return np.zeros((0, 4), dtype=np.float64)
        return np.column_stack((self.active_node_ids, self.active_forces_xyz))


@dataclass(frozen=True, slots=True)
class FJWAdjointStepState:
    time_index: int
    load_case_name: str
    objective_bone_density: np.ndarray
    fai_next: np.ndarray
    load_vector: FJWAdjointLoadVector
    adjoint_element_displacements: np.ndarray
    adjoint_objective_displacements: np.ndarray
    fai_current: np.ndarray
    stimulus: np.ndarray
    stimulus_derivative: np.ndarray
    scalar_gain: np.ndarray
    interaction_term: np.ndarray

    def __post_init__(self) -> None:
        objective_bone_density = np.asarray(self.objective_bone_density, dtype=np.float64).reshape(-1)
        fai_next = _as_objective_block_matrix(self.fai_next, label="fai_next")
        adjoint_element_displacements = _as_element_block_matrix(
            self.adjoint_element_displacements,
            label="adjoint_element_displacements",
        )
        adjoint_objective_displacements = _as_objective_block_matrix(
            self.adjoint_objective_displacements,
            label="adjoint_objective_displacements",
        )
        fai_current = _as_objective_block_matrix(self.fai_current, label="fai_current")
        stimulus = np.asarray(self.stimulus, dtype=np.float64).reshape(-1)
        stimulus_derivative = np.asarray(self.stimulus_derivative, dtype=np.float64).reshape(-1)
        scalar_gain = np.asarray(self.scalar_gain, dtype=np.float64).reshape(-1)
        interaction_term = np.asarray(self.interaction_term, dtype=np.float64).reshape(-1)

        objective_count = objective_bone_density.size
        for name, value in (
            ("fai_next", fai_next),
            ("adjoint_objective_displacements", adjoint_objective_displacements),
            ("fai_current", fai_current),
        ):
            if value.shape[0] != objective_count:
                raise ValueError(f"{name} row count must match objective_bone_density size.")
        for name, value in (
            ("stimulus", stimulus),
            ("stimulus_derivative", stimulus_derivative),
            ("scalar_gain", scalar_gain),
            ("interaction_term", interaction_term),
        ):
            if value.size != objective_count:
                raise ValueError(f"{name} size must match objective_bone_density size.")

        object.__setattr__(self, "objective_bone_density", objective_bone_density)
        object.__setattr__(self, "fai_next", fai_next)
        object.__setattr__(self, "adjoint_element_displacements", adjoint_element_displacements)
        object.__setattr__(self, "adjoint_objective_displacements", adjoint_objective_displacements)
        object.__setattr__(self, "fai_current", fai_current)
        object.__setattr__(self, "stimulus", stimulus)
        object.__setattr__(self, "stimulus_derivative", stimulus_derivative)
        object.__setattr__(self, "scalar_gain", scalar_gain)
        object.__setattr__(self, "interaction_term", interaction_term)


def _as_element_block_matrix(
    element_displacements: np.ndarray | list[float] | tuple[float, ...],
    *,
    label: str,
) -> np.ndarray:
    displacement_array = np.asarray(element_displacements, dtype=np.float64)
    if displacement_array.ndim == 1:
        if displacement_array.size % 24 != 0:
            raise ValueError(f"{label} length must be divisible by 24.")
        return displacement_array.reshape(-1, 24)
    if displacement_array.ndim == 2 and displacement_array.shape[1] == 24:
        return displacement_array
    raise ValueError(f"{label} must have shape (n, 24) or flattened length n*24.")


def _as_objective_block_matrix(
    values: np.ndarray | list[float] | tuple[float, ...],
    *,
    label: str,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        if matrix.size % 24 != 0:
            raise ValueError(f"{label} length must be divisible by 24.")
        return matrix.reshape(-1, 24)
    if matrix.ndim == 2 and matrix.shape[1] == 24:
        return matrix
    raise ValueError(f"{label} must have shape (n, 24) or flattened length n*24.")


def _build_element_kernel(
    strain_displacement_matrix: np.ndarray,
    constitutive_matrix: np.ndarray,
) -> np.ndarray:
    B = np.asarray(strain_displacement_matrix, dtype=np.float64)
    D = np.asarray(constitutive_matrix, dtype=np.float64)
    if B.shape != (6, 24):
        raise ValueError(f"Expected B to have shape (6, 24), got {B.shape}.")
    if D.shape != (6, 6):
        raise ValueError(f"Expected D to have shape (6, 6), got {D.shape}.")
    return B.T @ D @ B


def _objective_density(
    objective_bone_density: np.ndarray | list[float] | tuple[float, ...],
    *,
    material_constants: FJWMaterialConstants,
) -> np.ndarray:
    return np.clip(
        np.asarray(objective_bone_density, dtype=np.float64).reshape(-1),
        MIN_BONE_DENSITY,
        material_constants.bone_density_upper_bound,
    )


def _objective_step_terms(
    objective_element_displacements: np.ndarray,
    objective_bone_density: np.ndarray,
    *,
    strain_displacement_matrix: np.ndarray,
    constitutive_matrix: np.ndarray,
    material_constants: FJWMaterialConstants,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    element_kernel = _build_element_kernel(strain_displacement_matrix, constitutive_matrix)
    objective_modulus = bone_objective_modulus(
        objective_bone_density,
        bone_modulus_0=material_constants.bone_modulus_0,
        bone_modulus_min=material_constants.bone_modulus_min,
        bone_density_upper_bound=material_constants.bone_density_upper_bound,
    )
    kernel_times_u = objective_element_displacements @ element_kernel.T
    quadratic_terms = 0.5 * np.einsum(
        "bi,bi->b",
        objective_element_displacements,
        kernel_times_u,
        optimize=True,
    )
    stimulus = quadratic_terms * objective_modulus / objective_bone_density
    stimulus_derivative = d_bone_delta(stimulus)
    return element_kernel, objective_modulus, kernel_times_u, stimulus, stimulus_derivative


def build_adjoint_load_vector(
    *,
    load_case_name: str,
    time_index: int,
    objective_element_displacements: np.ndarray | list[float] | tuple[float, ...],
    objective_bone_density: np.ndarray | list[float] | tuple[float, ...],
    fai_next: np.ndarray | list[float] | tuple[float, ...],
    objective_element_nodes: np.ndarray,
    num_nodes: int,
    strain_displacement_matrix: np.ndarray,
    constitutive_matrix: np.ndarray,
    material_constants: FJWMaterialConstants,
) -> FJWAdjointLoadVector:
    objective_displacements = _as_objective_block_matrix(
        objective_element_displacements,
        label="objective_element_displacements",
    )
    density = _objective_density(objective_bone_density, material_constants=material_constants)
    fai_next_blocks = _as_objective_block_matrix(fai_next, label="fai_next")
    objective_element_nodes = np.asarray(objective_element_nodes, dtype=np.int32)

    if objective_displacements.shape[0] != density.size:
        raise ValueError("objective_element_displacements row count must match objective_bone_density size.")
    if fai_next_blocks.shape != objective_displacements.shape:
        raise ValueError("fai_next must have the same shape as objective_element_displacements.")
    if objective_element_nodes.shape != (density.size, 8):
        raise ValueError("objective_element_nodes must have shape (objective_count, 8).")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")

    _, objective_modulus, kernel_times_u, stimulus, stimulus_derivative = _objective_step_terms(
        objective_displacements,
        density,
        strain_displacement_matrix=strain_displacement_matrix,
        constitutive_matrix=constitutive_matrix,
        material_constants=material_constants,
    )
    scalar_gain = stimulus_derivative / (material_constants.voxel_volume * density)
    objective_force_blocks = fai_next_blocks * (
        scalar_gain[:, None] * objective_modulus[:, None] * kernel_times_u
    )

    nodal_forces_xyz = np.zeros((num_nodes, 3), dtype=np.float64)
    for local_node_index in range(8):
        node_ids = objective_element_nodes[:, local_node_index] - 1
        nodal_forces_xyz[node_ids] += objective_force_blocks[
            :,
            local_node_index * 3 : (local_node_index + 1) * 3,
        ]

    active_mask = np.any(np.abs(nodal_forces_xyz) > 0.0, axis=1)
    return FJWAdjointLoadVector(
        time_index=int(time_index),
        load_case_name=str(load_case_name),
        nodal_forces_flat=nodal_forces_xyz.reshape(-1),
        active_node_ids=np.flatnonzero(active_mask).astype(np.int32) + 1,
        active_forces_xyz=nodal_forces_xyz[active_mask],
    )


def build_adjoint_load_vector_from_forward_step(
    forward_step: FJWSingleLoadTimeStepResult,
    *,
    fai_next: np.ndarray | list[float] | tuple[float, ...],
    mesh: FJWReferenceMeshContext,
    material_constants: FJWMaterialConstants,
) -> FJWAdjointLoadVector:
    objective_element_ids = np.asarray(mesh.objective_elements, dtype=np.int32).reshape(-1) - 1
    objective_element_nodes = np.asarray(mesh.element_nodes, dtype=np.int32)[objective_element_ids]
    return build_adjoint_load_vector(
        load_case_name=forward_step.load_case_name,
        time_index=forward_step.time_index,
        objective_element_displacements=forward_step.objective_element_displacements,
        objective_bone_density=forward_step.obj_bo_previous,
        fai_next=fai_next,
        objective_element_nodes=objective_element_nodes,
        num_nodes=mesh.node_coordinates.shape[0],
        strain_displacement_matrix=mesh.strain_displacement_matrix,
        constitutive_matrix=mesh.constitutive_matrix,
        material_constants=material_constants,
    )


def build_fv_load_vector_from_forward_step(
    forward_step: FJWSingleLoadTimeStepResult,
    *,
    fai_next: np.ndarray | list[float] | tuple[float, ...],
    mesh: FJWReferenceMeshContext,
    material_constants: FJWMaterialConstants,
) -> FJWAdjointLoadVector:
    return build_adjoint_load_vector_from_forward_step(
        forward_step,
        fai_next=fai_next,
        mesh=mesh,
        material_constants=material_constants,
    )


def build_terminal_fai(objective_count: int) -> np.ndarray:
    objective_count = int(objective_count)
    if objective_count <= 0:
        raise ValueError("objective_count must be positive.")
    return np.ones((objective_count, 24), dtype=np.float64)


def update_fai_state(
    *,
    objective_element_displacements: np.ndarray | list[float] | tuple[float, ...],
    adjoint_objective_displacements: np.ndarray | list[float] | tuple[float, ...],
    objective_bone_density: np.ndarray | list[float] | tuple[float, ...],
    fai_next: np.ndarray | list[float] | tuple[float, ...],
    strain_displacement_matrix: np.ndarray,
    constitutive_matrix: np.ndarray,
    material_constants: FJWMaterialConstants,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    objective_displacements = _as_objective_block_matrix(
        objective_element_displacements,
        label="objective_element_displacements",
    )
    adjoint_displacements = _as_objective_block_matrix(
        adjoint_objective_displacements,
        label="adjoint_objective_displacements",
    )
    density = _objective_density(objective_bone_density, material_constants=material_constants)
    fai_next_blocks = _as_objective_block_matrix(fai_next, label="fai_next")

    if adjoint_displacements.shape != objective_displacements.shape:
        raise ValueError("adjoint_objective_displacements must match objective_element_displacements shape.")
    if fai_next_blocks.shape != objective_displacements.shape:
        raise ValueError("fai_next must match objective_element_displacements shape.")
    if objective_displacements.shape[0] != density.size:
        raise ValueError("objective_element_displacements row count must match objective_bone_density size.")

    element_kernel, objective_modulus, _, stimulus, stimulus_derivative = _objective_step_terms(
        objective_displacements,
        density,
        strain_displacement_matrix=strain_displacement_matrix,
        constitutive_matrix=constitutive_matrix,
        material_constants=material_constants,
    )
    scalar_gain = (
        2.0
        * stimulus_derivative
        * (stimulus * density)
        / (material_constants.voxel_volume * np.square(density))
    )
    interaction_term = np.einsum(
        "bi,ij,bj->b",
        adjoint_displacements,
        element_kernel,
        objective_displacements,
        optimize=True,
    )
    stiffness_gradient_term = (
        interaction_term
        * 3.0
        * np.square(density)
        * material_constants.bone_modulus_0
        / (material_constants.bone_density_upper_bound**3)
    )
    fai_current = ((1.0 + scalar_gain)[:, None] * fai_next_blocks) - stiffness_gradient_term[:, None]
    return fai_current, stimulus, stimulus_derivative, scalar_gain


def build_adjoint_step_state(
    *,
    forward_step: FJWSingleLoadTimeStepResult,
    adjoint_element_displacements: np.ndarray | list[float] | tuple[float, ...],
    fai_next: np.ndarray | list[float] | tuple[float, ...],
    mesh: FJWReferenceMeshContext,
    material_constants: FJWMaterialConstants,
) -> FJWAdjointStepState:
    adjoint_element_matrix = _as_element_block_matrix(
        adjoint_element_displacements,
        label="adjoint_element_displacements",
    )
    num_elements = mesh.element_nodes.shape[0]
    if adjoint_element_matrix.shape[0] != num_elements:
        raise ValueError(
            "adjoint_element_displacements element count does not match mesh.element_nodes: "
            f"{adjoint_element_matrix.shape[0]} != {num_elements}."
        )

    objective_element_ids = np.asarray(mesh.objective_elements, dtype=np.int32).reshape(-1) - 1
    adjoint_objective_displacements = adjoint_element_matrix[objective_element_ids]
    load_vector = build_adjoint_load_vector_from_forward_step(
        forward_step,
        fai_next=fai_next,
        mesh=mesh,
        material_constants=material_constants,
    )
    fai_current, stimulus, stimulus_derivative, scalar_gain = update_fai_state(
        objective_element_displacements=forward_step.objective_element_displacements,
        adjoint_objective_displacements=adjoint_objective_displacements,
        objective_bone_density=forward_step.obj_bo_previous,
        fai_next=fai_next,
        strain_displacement_matrix=mesh.strain_displacement_matrix,
        constitutive_matrix=mesh.constitutive_matrix,
        material_constants=material_constants,
    )
    interaction_term = np.einsum(
        "bi,ij,bj->b",
        adjoint_objective_displacements,
        _build_element_kernel(mesh.strain_displacement_matrix, mesh.constitutive_matrix),
        forward_step.objective_element_displacements,
        optimize=True,
    )
    return FJWAdjointStepState(
        time_index=forward_step.time_index,
        load_case_name=forward_step.load_case_name,
        objective_bone_density=np.asarray(forward_step.obj_bo_previous, dtype=np.float64),
        fai_next=_as_objective_block_matrix(fai_next, label="fai_next"),
        load_vector=load_vector,
        adjoint_element_displacements=adjoint_element_matrix,
        adjoint_objective_displacements=adjoint_objective_displacements,
        fai_current=fai_current,
        stimulus=stimulus,
        stimulus_derivative=stimulus_derivative,
        scalar_gain=scalar_gain,
        interaction_term=interaction_term,
    )


__all__ = [
    "FJWAdjointLoadVector",
    "FJWAdjointStepState",
    "build_adjoint_load_vector",
    "build_adjoint_load_vector_from_forward_step",
    "build_fv_load_vector_from_forward_step",
    "build_adjoint_step_state",
    "build_terminal_fai",
    "update_fai_state",
]
