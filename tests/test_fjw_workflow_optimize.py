from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fem_analysis.fjw_workflow_checkpoint_io import load_resume_state
from fem_analysis.fjw_workflow_optimize import FJWOptimizationConfig, run_fjw_optimization

from tests.test_fjw_workflow_sfepy_runner import build_minimal_sfepy_iteration_state


class FJWWorkflowOptimizeTest(unittest.TestCase):
    def test_sfepy_optimizer_writes_checkpoint_and_can_resume(self) -> None:
        workflow_state = build_minimal_sfepy_iteration_state()

        with tempfile.TemporaryDirectory() as temp_dir:
            run_directory = Path(temp_dir) / "fjw_optimize"
            with mock.patch(
                "fem_analysis.fjw_workflow_optimize.load_fjw_workflow_state",
                return_value=workflow_state,
            ):
                first = run_fjw_optimization(
                    FJWOptimizationConfig(
                        run_directory=run_directory,
                        max_iterations=1,
                        num_time_steps=1,
                        runtime_profile="local",
                    )
                )
                resumed = run_fjw_optimization(
                    FJWOptimizationConfig(
                        run_directory=run_directory,
                        max_iterations=1,
                        num_time_steps=1,
                        resume=True,
                        runtime_profile="local",
                    )
                )

            self.assertEqual(len(first.iterations), 1)
            self.assertEqual(len(resumed.iterations), 1)
            self.assertTrue((run_directory / "workflow_manifest.json").exists())
            self.assertTrue((run_directory / "iter_000" / "design_cage.npz").exists())
            self.assertTrue((run_directory / "iter_001" / "aggregate_terms.npz").exists())
            self.assertTrue((run_directory / "iter_001" / "force_1" / "adjoint_t0" / "fv_manifest.json").exists())
            resume_state = load_resume_state(run_directory)
            self.assertEqual(resume_state.iteration_index, 2)
            self.assertEqual(resume_state.design.shape, workflow_state.initial_state.design_cage.shape)

            payload = json.loads((run_directory / "iter_001" / "iteration_state.json").read_text(encoding="utf-8"))
            self.assertFalse(payload["has_placeholder_adjoint"])
            self.assertEqual(payload["optimizer_diagnostics"]["solver"], "mmasub_subsolv")
            self.assertTrue((run_directory / "iter_001" / "timing.json").exists())

    def test_abaqus_optimizer_rejects_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "requires --real-run"):
                run_fjw_optimization(
                    FJWOptimizationConfig(
                        backend="abaqus",
                        real_run=False,
                        run_directory=Path(temp_dir),
                        runtime_profile="local",
                    )
                )


if __name__ == "__main__":
    unittest.main()
