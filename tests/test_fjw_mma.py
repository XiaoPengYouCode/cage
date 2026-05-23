from __future__ import annotations

import unittest

import numpy as np

from fem_analysis.fjw_workflow_iteration import run_iteration_from_histories
from fem_analysis.fjw_workflow_optimizer import FJWMMAOptimizer, FJWOptimizationTerms, build_initial_mma_state
from fem_analysis.fjw_workflow_runtime import FJWSingleLoadCaseHistory
from fem_analysis.fjw_workflow_single_case import run_single_case_workflow

from tests.test_fjw_workflow_regression import (
    RecordingAdjointSolver,
    RecordingForwardSolver,
    build_minimal_workflow_state,
)


class FJWMMAOptimizerTest(unittest.TestCase):
    def test_mma_step_matches_archived_small_vector_fixture(self) -> None:
        design = np.array([0.3, 0.45, 0.6], dtype=np.float64)
        terms = FJWOptimizationTerms(
            objective=-1.25,
            objective_gradient=np.array([-0.9, 0.2, -0.35], dtype=np.float64),
            constraints=np.array([0.05], dtype=np.float64),
            constraint_gradients=np.array([[1.0, 1.0, 1.0]], dtype=np.float64),
        )
        result = FJWMMAOptimizer().step(design, terms, build_initial_mma_state(design))

        np.testing.assert_allclose(
            result.design,
            np.array([0.42677313265564956, 0.001001091740939041, 0.6119509389187682], dtype=np.float64),
            rtol=1e-7,
            atol=1e-9,
        )
        self.assertEqual(result.diagnostics["solver"], "mmasub_subsolv")
        self.assertEqual(result.state.iteration, 1)
        np.testing.assert_allclose(result.state.xold1, design)
        np.testing.assert_allclose(result.state.low, np.array([-0.1995, -0.0495, 0.1005]))
        self.assertAlmostEqual(float(result.diagnostics["delta"]), 0.19590766110514726)

    def test_iteration_without_dynamic_adjoint_fields_fails_explicitly(self) -> None:
        workflow_state = build_minimal_workflow_state(load_case_names=("force_1", "force_2", "force_3"))
        histories = []
        for load_case_name in ("force_1", "force_2", "force_3"):
            case_result = run_single_case_workflow(
                workflow_state=workflow_state,
                load_case_name=load_case_name,
                forward_solver=RecordingForwardSolver(),
                adjoint_solver=RecordingAdjointSolver(),
                num_time_steps=1,
            )
            histories.append(
                FJWSingleLoadCaseHistory(
                    load_case_name=load_case_name,
                    time_steps=case_result.forward_steps,
                )
            )

        with self.assertRaisesRegex(RuntimeError, "Missing adjoint step fields"):
            run_iteration_from_histories(
                case_histories=tuple(histories),
                workflow_state=workflow_state,
            )


if __name__ == "__main__":
    unittest.main()
