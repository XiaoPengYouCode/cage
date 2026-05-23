from __future__ import annotations

import unittest

import numpy as np

from fem_analysis.fjw_workflow_single_case import run_single_case_workflow
from tests.test_fjw_workflow_regression import (
    RecordingAdjointSolver,
    RecordingForwardSolver,
    build_minimal_workflow_state,
)


class FJWDynamicAdjointPipelineTest(unittest.TestCase):
    def test_adjoint_requests_use_runtime_fv_from_forward_steps(self) -> None:
        workflow_state = build_minimal_workflow_state(load_case_names=("force_1",), num_time_steps=2)
        forward_solver = RecordingForwardSolver()
        adjoint_solver = RecordingAdjointSolver()

        result = run_single_case_workflow(
            workflow_state=workflow_state,
            load_case_name="force_1",
            forward_solver=forward_solver,
            adjoint_solver=adjoint_solver,
            num_time_steps=2,
        )

        self.assertEqual([request.time_index for request in adjoint_solver.requests], [1, 0])
        self.assertEqual(len(result.adjoint_steps), 2)
        self.assertTrue(
            all(step.solve_request.load_vector.nodal_forces_flat.size == workflow_state.mesh.node_coordinates.shape[0] * 3 for step in result.adjoint_steps)
        )
        self.assertTrue(
            all(np.isfinite(step.solve_request.load_vector.nodal_forces_flat).all() for step in result.adjoint_steps)
        )


if __name__ == "__main__":
    unittest.main()
