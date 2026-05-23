from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from fem_analysis.fjw_direct_solver import (
    FJWDirectSolverConfig,
    build_fjw_direct_problem_setup,
    solve_fjw_direct_adjoint_case,
    solve_fjw_direct_case,
)
from fem_analysis.fjw_reference import FJWReferenceModel
from fem_analysis.fjw_workflow_models import (
    FJWInitialState,
    FJWLoad,
    FJWLoadCase,
    FJWMaterialConstants,
    FJWModulusBuckets,
    FJWReferenceMeshContext,
    FJWWorkflowState,
)


def build_minimal_direct_workflow_state(loads: tuple[FJWLoad, ...]) -> FJWWorkflowState:
    nx = ny = nz = 2
    node_coordinates = []
    for kk in range(1, nz + 2):
        for jj in range(1, ny + 2):
            for ii in range(1, nx + 2):
                node_coordinates.append([ii, jj, kk])
    node_coordinates = np.asarray(node_coordinates, dtype=np.int32)

    def node_id(ii: int, jj: int, kk: int) -> int:
        return kk * (ny + 1) * (nx + 1) + jj * (nx + 1) + ii + 1

    element_nodes = []
    for kk in range(nz):
        for jj in range(ny):
            for ii in range(nx):
                element_nodes.append(
                    [
                        node_id(ii, jj, kk),
                        node_id(ii + 1, jj, kk),
                        node_id(ii + 1, jj + 1, kk),
                        node_id(ii, jj + 1, kk),
                        node_id(ii, jj, kk + 1),
                        node_id(ii + 1, jj, kk + 1),
                        node_id(ii + 1, jj + 1, kk + 1),
                        node_id(ii, jj + 1, kk + 1),
                    ]
                )
    element_nodes = np.asarray(element_nodes, dtype=np.int32)
    element_ids = np.arange(1, element_nodes.shape[0] + 1, dtype=np.int32)
    reference_model = FJWReferenceModel(
        node_coordinates=node_coordinates,
        element_nodes=element_nodes,
        cor_elements=np.array([], dtype=np.int32),
        tra_elements=np.array([], dtype=np.int32),
        cage_elements=element_ids.copy(),
        design_elements=element_ids.copy(),
        objective_elements=np.array([], dtype=np.int32),
    )
    mesh = FJWReferenceMeshContext(
        reference_model=reference_model,
        strain_displacement_matrix=np.zeros((6, 24), dtype=np.float64),
        constitutive_matrix=np.eye(6, dtype=np.float64),
        grid_shape_xyz=(1, 1, 1),
        top_node_ids=np.where(node_coordinates[:, 2] == nz + 1)[0].astype(np.int32) + 1,
        bottom_node_ids=np.where(node_coordinates[:, 2] == 1)[0].astype(np.int32) + 1,
        element_anchor_indices=np.asarray(
            [[ii, jj, kk] for kk in range(nz) for jj in range(ny) for ii in range(nx)],
            dtype=np.int32,
        ),
        design_anchor_indices=np.asarray(
            [[ii, jj, kk] for kk in range(nz) for jj in range(ny) for ii in range(nx)],
            dtype=np.int32,
        ),
        objective_anchor_indices=np.zeros((0, 3), dtype=np.int32),
    )
    material_constants = FJWMaterialConstants(
        voxel_volume=0.216,
        time_step_dt=1.0,
        num_time_steps=1,
        bone_density_upper_bound=1.86,
        bone_modulus_0=12000.0,
        bone_modulus_min=1.2,
        cage_modulus_0=110000.0,
        cage_modulus_min=11.0,
        initial_bone_density=0.36,
        single_load_initial_design_cage=0.2,
        three_load_initial_design_cage=0.3,
        cage_bucket_max_index=100,
        bone_bucket_max_index=10,
    )
    initial_state = FJWInitialState(
        mode="three_load",
        cage_3d=np.full((nx, ny, nz), 0.3, dtype=np.float64),
        bone_3d=np.array([[[0.36]]], dtype=np.float64),
        design_cage=np.full(element_nodes.shape[0], 0.3, dtype=np.float64),
        obj_bo=np.zeros(0, dtype=np.float64),
        initial_design_total=float(element_nodes.shape[0]) * 0.3,
        xold1=np.full(element_nodes.shape[0], 0.3, dtype=np.float64),
        xold2=np.full(element_nodes.shape[0], 0.3, dtype=np.float64),
        modulus_buckets=FJWModulusBuckets(
            design_cage_clipped=np.full(element_nodes.shape[0], 0.3, dtype=np.float64),
            obj_bo_clipped=np.zeros(0, dtype=np.float64),
            E_cage=np.full(element_nodes.shape[0], 2981.0, dtype=np.float64),
            E_obj=np.zeros(0, dtype=np.float64),
            cage_bucket_indices=np.full(element_nodes.shape[0], 3, dtype=np.int32),
            obj_bucket_indices=np.zeros(0, dtype=np.int32),
            cage_bucket_moduli=np.full(element_nodes.shape[0], 2981.0, dtype=np.float64),
            obj_bucket_moduli=np.zeros(0, dtype=np.float64),
            cage_material_names=np.array(["CAGE_3"] * element_nodes.shape[0], dtype=object),
            obj_material_names=np.array([], dtype=object),
        ),
    )
    load_case = FJWLoadCase(
        name="force_test",
        template_path=Path("force_test.inp"),
        template_lines=(),
        boundary_conditions=(),
        loads=loads,
    )
    return FJWWorkflowState(
        reference_dir=Path("references/fjw_work"),
        abaqus_inputs_path=Path("datasets/fjw_abaqus_inputs.json"),
        input_inventory_path=Path("datasets/fjw_input_inventory.json"),
        end1_template_path=Path("references/fjw_work/end1.inp"),
        mesh=mesh,
        material_constants=material_constants,
        load_cases=(load_case,),
        cage_material_buckets=(),
        bone_material_buckets=(),
        background_material_buckets=(),
        initial_state=initial_state,
        assembly_controls={},
        adjoint_load_template={},
        structured_inputs={
            "assembly_controls": {
                "reference_nodes": [
                    {"coordinates": [0.0, 0.0, 0.0]},
                    {"coordinates": [0.6, 0.6, 1.2]},
                ]
            }
        },
        input_inventory={},
    )


class FJWDirectSolverTest(unittest.TestCase):
    def test_build_setup_maps_rigid_dofs(self) -> None:
        workflow_state = build_minimal_direct_workflow_state(
            (
                FJWLoad(target="M_SET-2", dof=4, magnitude=-7500.0),
                FJWLoad(target="M_SET-2", dof=5, magnitude=125.0),
            )
        )
        setup = build_fjw_direct_problem_setup(workflow_state, load_case_name="force_test")
        np.testing.assert_allclose(
            setup.load_vector,
            np.array([[0.0, 0.0, 0.0, -7500.0, 125.0, 0.0]], dtype=np.float64),
        )
        self.assertEqual(setup.top_rp_vertex_id, 22)
        self.assertEqual(setup.bottom_rp_vertex_id, 0)

    def test_solve_force_case_runs_on_single_hex_mesh(self) -> None:
        workflow_state = build_minimal_direct_workflow_state(
            (FJWLoad(target="M_SET-2", dof=3, magnitude=-1200.0),)
        )
        result = solve_fjw_direct_case(workflow_state, load_case_name="force_test")
        self.assertGreater(result.max_displacement_mm, 0.0)
        self.assertLess(result.top_rp_displacement[2], 0.0)
        self.assertTrue(np.allclose(result.top_rp_rotation, 0.0, atol=1e-10))
        self.assertEqual(result.setup.material_groups[0].element_count, 8)

    def test_scipy_iterative_solver_runs_on_single_hex_mesh(self) -> None:
        workflow_state = build_minimal_direct_workflow_state(
            (FJWLoad(target="M_SET-2", dof=3, magnitude=-1200.0),)
        )
        result = solve_fjw_direct_case(
            workflow_state,
            load_case_name="force_test",
            config=FJWDirectSolverConfig(linear_solver_kind="scipy_iterative"),
        )
        self.assertGreater(result.max_displacement_mm, 0.0)
        self.assertLess(result.top_rp_displacement[2], 0.0)
        self.assertTrue(np.all(np.isfinite(result.nodal_displacements)))

    def test_solve_adjoint_case_accepts_dense_nodal_loads(self) -> None:
        workflow_state = build_minimal_direct_workflow_state(
            (FJWLoad(target="M_SET-2", dof=3, magnitude=-1200.0),)
        )
        nodal_forces_flat = np.zeros(workflow_state.mesh.node_coordinates.shape[0] * 3, dtype=np.float64)
        top_corner_node_id = 27
        nodal_forces_flat[(top_corner_node_id - 1) * 3 + 2] = -500.0

        result = solve_fjw_direct_adjoint_case(
            workflow_state,
            load_case_name="force_test",
            nodal_forces_flat=nodal_forces_flat,
        )

        self.assertGreater(result.max_displacement_mm, 0.0)
        self.assertLess(result.top_rp_displacement[2], 0.0)


if __name__ == "__main__":
    unittest.main()
