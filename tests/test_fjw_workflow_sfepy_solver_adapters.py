from __future__ import annotations

import unittest

import numpy as np

from fem_analysis.fjw_workflow_adjoint import FJWAdjointLoadVector
from fem_analysis.fjw_workflow_sfepy_solver_adapters import build_fjw_sfepy_solver_adapters
from fem_analysis.fjw_workflow_single_case import FJWAdjointSolveRequest, FJWForwardSolveRequest
from fem_analysis.fjw_workflow_models import FJWLoad

from tests.test_fjw_direct_solver import build_minimal_direct_workflow_state


class FJWSfePyWorkflowSolverAdaptersTest(unittest.TestCase):
    def test_forward_solver_returns_full_element_vectors(self) -> None:
        workflow_state = build_minimal_direct_workflow_state(
            (FJWLoad(target="M_SET-2", dof=3, magnitude=-1200.0),)
        )
        load_case = workflow_state.load_cases[0]
        adapters = build_fjw_sfepy_solver_adapters()

        result = adapters.forward_solver.solve_forward(
            FJWForwardSolveRequest(
                workflow_state=workflow_state,
                load_case=load_case,
                time_index=0,
                design_cage=workflow_state.initial_state.design_cage,
                obj_bo=workflow_state.initial_state.obj_bo,
            )
        )

        self.assertEqual(result.element_displacements.shape, (8, 24))
        self.assertEqual(result.metadata["solver_backend"], "sfepy_direct")
        self.assertEqual(result.metadata["solve_kind"], "forward")
        self.assertGreater(float(result.metadata["max_displacement_mm"]), 0.0)

    def test_adjoint_solver_accepts_sparse_nodal_load_vector(self) -> None:
        workflow_state = build_minimal_direct_workflow_state(
            (FJWLoad(target="M_SET-2", dof=3, magnitude=-1200.0),)
        )
        load_case = workflow_state.load_cases[0]
        adapters = build_fjw_sfepy_solver_adapters()

        nodal_forces_flat = np.zeros(workflow_state.mesh.node_coordinates.shape[0] * 3, dtype=np.float64)
        top_corner_node_id = 27
        nodal_forces_flat[(top_corner_node_id - 1) * 3 + 2] = -500.0
        load_vector = FJWAdjointLoadVector(
            time_index=0,
            load_case_name=load_case.name,
            nodal_forces_flat=nodal_forces_flat,
            active_node_ids=np.array([top_corner_node_id], dtype=np.int32),
            active_forces_xyz=np.array([[0.0, 0.0, -500.0]], dtype=np.float64),
        )

        result = adapters.adjoint_solver.solve_adjoint(
            FJWAdjointSolveRequest(
                workflow_state=workflow_state,
                load_case=load_case,
                time_index=0,
                design_cage=workflow_state.initial_state.design_cage,
                obj_bo=workflow_state.initial_state.obj_bo,
                load_vector=load_vector,
            )
        )

        self.assertEqual(result.element_displacements.shape, (8, 24))
        self.assertEqual(result.metadata["solver_backend"], "sfepy_direct")
        self.assertEqual(result.metadata["solve_kind"], "adjoint")
        self.assertEqual(result.metadata["active_load_node_count"], 1)
        self.assertGreater(float(result.metadata["max_displacement_mm"]), 0.0)


if __name__ == "__main__":
    unittest.main()
