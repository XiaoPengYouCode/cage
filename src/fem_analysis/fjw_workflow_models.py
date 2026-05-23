from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .fjw_reference import FJWReferenceModel


@dataclass(frozen=True, slots=True)
class FJWBoundaryCondition:
    target: str
    constraint: str | None = None
    dof_start: int | None = None
    dof_end: int | None = None
    value: float | None = None


@dataclass(frozen=True, slots=True)
class FJWLoad:
    target: str
    dof: int
    magnitude: float
    op: str = "NEW"


@dataclass(frozen=True, slots=True)
class FJWLoadCase:
    name: str
    template_path: Path
    template_lines: tuple[str, ...]
    boundary_conditions: tuple[FJWBoundaryCondition, ...]
    loads: tuple[FJWLoad, ...]


@dataclass(frozen=True, slots=True)
class FJWMaterialBucket:
    index: int
    material_name: str
    section_elset: str
    youngs_modulus: float
    poisson_ratio: float
    density: float


@dataclass(frozen=True, slots=True)
class FJWMaterialConstants:
    voxel_volume: float
    time_step_dt: float
    num_time_steps: int
    bone_density_upper_bound: float
    bone_modulus_0: float
    bone_modulus_min: float
    cage_modulus_0: float
    cage_modulus_min: float
    initial_bone_density: float
    single_load_initial_design_cage: float
    three_load_initial_design_cage: float
    cage_bucket_max_index: int
    bone_bucket_max_index: int

    def initial_design_value(self, mode: str) -> float:
        if mode == "single_load":
            return self.single_load_initial_design_cage
        if mode == "three_load":
            return self.three_load_initial_design_cage
        raise ValueError(f"Unsupported FJW initial-design mode: {mode!r}.")


@dataclass(frozen=True, slots=True)
class FJWModulusBuckets:
    design_cage_clipped: np.ndarray
    obj_bo_clipped: np.ndarray
    E_cage: np.ndarray
    E_obj: np.ndarray
    cage_bucket_indices: np.ndarray
    obj_bucket_indices: np.ndarray
    cage_bucket_moduli: np.ndarray
    obj_bucket_moduli: np.ndarray
    cage_material_names: np.ndarray
    obj_material_names: np.ndarray


@dataclass(frozen=True, slots=True)
class FJWInitialState:
    mode: str
    cage_3d: np.ndarray
    bone_3d: np.ndarray
    design_cage: np.ndarray
    obj_bo: np.ndarray
    initial_design_total: float
    xold1: np.ndarray
    xold2: np.ndarray
    modulus_buckets: FJWModulusBuckets


@dataclass(frozen=True, slots=True)
class FJWReferenceMeshContext:
    reference_model: FJWReferenceModel
    strain_displacement_matrix: np.ndarray
    constitutive_matrix: np.ndarray
    grid_shape_xyz: tuple[int, int, int]
    top_node_ids: np.ndarray
    bottom_node_ids: np.ndarray
    element_anchor_indices: np.ndarray
    design_anchor_indices: np.ndarray
    objective_anchor_indices: np.ndarray

    @property
    def node_coordinates(self) -> np.ndarray:
        return self.reference_model.node_coordinates

    @property
    def element_nodes(self) -> np.ndarray:
        return self.reference_model.element_nodes

    @property
    def cor_elements(self) -> np.ndarray:
        return self.reference_model.cor_elements

    @property
    def tra_elements(self) -> np.ndarray:
        return self.reference_model.tra_elements

    @property
    def cage_elements(self) -> np.ndarray:
        return self.reference_model.cage_elements

    @property
    def design_elements(self) -> np.ndarray:
        return self.reference_model.design_elements

    @property
    def objective_elements(self) -> np.ndarray:
        return self.reference_model.objective_elements


@dataclass(frozen=True, slots=True)
class FJWWorkflowState:
    reference_dir: Path
    abaqus_inputs_path: Path
    input_inventory_path: Path
    end1_template_path: Path
    mesh: FJWReferenceMeshContext
    material_constants: FJWMaterialConstants
    load_cases: tuple[FJWLoadCase, ...]
    cage_material_buckets: tuple[FJWMaterialBucket, ...]
    bone_material_buckets: tuple[FJWMaterialBucket, ...]
    background_material_buckets: tuple[FJWMaterialBucket, ...]
    initial_state: FJWInitialState
    assembly_controls: Mapping[str, Any]
    adjoint_load_template: Mapping[str, Any]
    structured_inputs: Mapping[str, Any]
    input_inventory: Mapping[str, Any]


__all__ = [
    "FJWBoundaryCondition",
    "FJWInitialState",
    "FJWLoad",
    "FJWLoadCase",
    "FJWMaterialBucket",
    "FJWMaterialConstants",
    "FJWModulusBuckets",
    "FJWReferenceMeshContext",
    "FJWWorkflowState",
]
