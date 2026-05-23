from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from fem_analysis.fjw_reference import FJWReferenceModel
from fem_analysis.fjw_workflow_driver import FJWWorkflowDriverRequest
from fem_analysis.fjw_workflow_models import (
    FJWInitialState,
    FJWLoad,
    FJWLoadCase,
    FJWMaterialConstants,
    FJWModulusBuckets,
    FJWReferenceMeshContext,
    FJWWorkflowState,
)
from fem_analysis.fjw_workflow_runner import run_fjw_sfepy_workflow_iteration
from fem_analysis.fjw_workflow_three_force import FORCE_CASE_ORDER


def build_minimal_sfepy_iteration_state() -> FJWWorkflowState:
    node_coordinates = np.array(
        [
            [1, 1, 1],
            [2, 1, 1],
            [3, 1, 1],
            [1, 2, 1],
            [2, 2, 1],
            [3, 2, 1],
            [1, 1, 2],
            [2, 1, 2],
            [3, 1, 2],
            [1, 2, 2],
            [2, 2, 2],
            [3, 2, 2],
        ],
        dtype=np.int32,
    )
    element_nodes = np.array(
        [
            [1, 2, 5, 4, 7, 8, 11, 10],
            [2, 3, 6, 5, 8, 9, 12, 11],
        ],
        dtype=np.int32,
    )
    reference_model = FJWReferenceModel(
        node_coordinates=node_coordinates,
        element_nodes=element_nodes,
        cor_elements=np.zeros(0, dtype=np.int32),
        tra_elements=np.zeros(0, dtype=np.int32),
        cage_elements=np.array([2], dtype=np.int32),
        design_elements=np.array([2], dtype=np.int32),
        objective_elements=np.array([1], dtype=np.int32),
    )
    strain_displacement_matrix = np.zeros((6, 24), dtype=np.float64)
    strain_displacement_matrix[:, :6] = np.eye(6, dtype=np.float64)
    constitutive_matrix = np.eye(6, dtype=np.float64)
    mesh = FJWReferenceMeshContext(
        reference_model=reference_model,
        strain_displacement_matrix=strain_displacement_matrix,
        constitutive_matrix=constitutive_matrix,
        grid_shape_xyz=(2, 1, 1),
        top_node_ids=np.array([7, 8, 9, 10, 11, 12], dtype=np.int32),
        bottom_node_ids=np.array([1, 2, 3, 4, 5, 6], dtype=np.int32),
        element_anchor_indices=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32),
        design_anchor_indices=np.array([[1, 0, 0]], dtype=np.int32),
        objective_anchor_indices=np.array([[0, 0, 0]], dtype=np.int32),
    )
    material_constants = FJWMaterialConstants(
        voxel_volume=1.0,
        time_step_dt=1.0,
        num_time_steps=1,
        bone_density_upper_bound=1.0,
        bone_modulus_0=1.0,
        bone_modulus_min=0.1,
        cage_modulus_0=2.0,
        cage_modulus_min=0.2,
        initial_bone_density=0.36,
        single_load_initial_design_cage=0.2,
        three_load_initial_design_cage=0.3,
        cage_bucket_max_index=10,
        bone_bucket_max_index=10,
    )
    design = np.array([0.3], dtype=np.float64)
    obj_bo = np.array([0.36], dtype=np.float64)
    modulus_buckets = FJWModulusBuckets(
        design_cage_clipped=design.copy(),
        obj_bo_clipped=obj_bo.copy(),
        E_cage=np.array([0.254], dtype=np.float64),
        E_obj=np.array([0.146656], dtype=np.float64),
        cage_bucket_indices=np.array([3], dtype=np.int32),
        obj_bucket_indices=np.array([4], dtype=np.int32),
        cage_bucket_moduli=np.array([0.254], dtype=np.float64),
        obj_bucket_moduli=np.array([0.146656], dtype=np.float64),
        cage_material_names=np.array(["CAGE_3"], dtype=object),
        obj_material_names=np.array(["BONE_4"], dtype=object),
    )
    initial_state = FJWInitialState(
        mode="three_load",
        cage_3d=np.array([[[0.0]], [[0.3]]], dtype=np.float64),
        bone_3d=np.array([[[0.36]], [[0.36]]], dtype=np.float64),
        design_cage=design.copy(),
        obj_bo=obj_bo.copy(),
        initial_design_total=float(np.sum(design, dtype=np.float64)),
        xold1=design.copy(),
        xold2=design.copy(),
        modulus_buckets=modulus_buckets,
    )
    load_specs = {
        "force_1": (FJWLoad(target="M_SET-2", dof=3, magnitude=-1200.0),),
        "force_2": (FJWLoad(target="M_SET-2", dof=4, magnitude=250.0),),
        "force_3": (FJWLoad(target="M_SET-2", dof=5, magnitude=-180.0),),
    }
    load_cases = tuple(
        FJWLoadCase(
            name=name,
            template_path=Path(f"{name}.inp"),
            template_lines=("*Heading",),
            boundary_conditions=(),
            loads=load_specs[name],
        )
        for name in FORCE_CASE_ORDER
    )
    structured_inputs = {
        "assembly_controls": {
            "reference_nodes": [
                {"coordinates": [0.0, 0.0, 0.0]},
                {"coordinates": [1.0, 1.0, 1.0]},
            ]
        }
    }
    return FJWWorkflowState(
        reference_dir=Path("references/fjw_work"),
        abaqus_inputs_path=Path("datasets/fjw_abaqus_inputs.json"),
        input_inventory_path=Path("datasets/fjw_input_inventory.json"),
        end1_template_path=Path("references/fjw_work/end1.inp"),
        mesh=mesh,
        material_constants=material_constants,
        load_cases=load_cases,
        cage_material_buckets=(),
        bone_material_buckets=(),
        background_material_buckets=(),
        initial_state=initial_state,
        assembly_controls=structured_inputs["assembly_controls"],
        adjoint_load_template={},
        structured_inputs=structured_inputs,
        input_inventory={},
    )


class FJWSfePyWorkflowRunnerTest(unittest.TestCase):
    def test_run_fjw_sfepy_workflow_iteration_smoke(self) -> None:
        workflow_state = build_minimal_sfepy_iteration_state()

        result = run_fjw_sfepy_workflow_iteration(
            driver_request=FJWWorkflowDriverRequest(
                workflow_state=workflow_state,
                num_time_steps=1,
            )
        )

        self.assertEqual(result.load_case_names, FORCE_CASE_ORDER)
        self.assertEqual(len(result.single_case_results), 3)
        self.assertEqual(result.iteration_state.iteration_index, 1)
        self.assertFalse(result.iteration_state.has_placeholder_adjoint)
        self.assertIsNotNone(result.iteration_state.aggregate_terms)
        self.assertIsNotNone(result.iteration_state.optimization_terms)
        self.assertIsNotNone(result.iteration_state.next_design)
        self.assertEqual(result.iteration_state.next_design.shape, (1,))
        self.assertEqual(
            [record.adjoint_source for record in result.iteration_state.case_records],
            ["manual", "manual", "manual"],
        )

        terminal_bo_sum_by_case = {
            case_result.load_case_name: case_result.terminal_bo_sum
            for case_result in result.single_case_results
        }
        self.assertEqual(set(terminal_bo_sum_by_case), set(FORCE_CASE_ORDER))
        self.assertTrue(all(np.isfinite(value) for value in terminal_bo_sum_by_case.values()))
        self.assertTrue(np.isfinite(result.iteration_state.aggregate_terms.objective))


if __name__ == "__main__":
    unittest.main()
