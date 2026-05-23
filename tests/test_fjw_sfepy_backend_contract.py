from __future__ import annotations

import unittest

from fem_analysis.fjw_direct_solver import (
    FJWDirectSolverConfig,
    build_fjw_direct_problem,
    build_fjw_direct_problem_setup,
)
from tests.test_fjw_direct_solver import build_minimal_direct_workflow_state
from fem_analysis.fjw_workflow_models import FJWLoad


class FJWSfePyBackendContractTest(unittest.TestCase):
    def test_petsc_mumps_backend_reports_missing_runtime_stack(self) -> None:
        workflow_state = build_minimal_direct_workflow_state(
            (FJWLoad(target="M_SET-2", dof=3, magnitude=-1200.0),)
        )
        setup = build_fjw_direct_problem_setup(
            workflow_state,
            load_case_name="force_test",
        )

        with self.assertRaisesRegex(RuntimeError, "petsc4py"):
            build_fjw_direct_problem(
                setup,
                config=FJWDirectSolverConfig(linear_solver_kind="petsc_mumps"),
            )


if __name__ == "__main__":
    unittest.main()
