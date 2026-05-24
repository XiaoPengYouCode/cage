from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest import mock

from fem_analysis.cli import main, parse_args

from tests.test_fjw_workflow_sfepy_runner import build_minimal_sfepy_iteration_state


class FemAnalysisCliTest(unittest.TestCase):
    def test_fjw_sfepy_commands_default_to_python_only_iterative_solver(self) -> None:
        for command in ("fjw-direct",):
            args = parse_args([command])
            self.assertEqual(args.sfepy_linear_solver, "scipy_iterative")
        for command in ("fjw-sfepy-iterate", "fjw-optimize"):
            args = parse_args([command])
            self.assertIsNone(args.sfepy_linear_solver)

    def test_fjw_sfepy_iterate_command_runs_with_minimal_workflow_state(self) -> None:
        workflow_state = build_minimal_sfepy_iteration_state()
        stdout = io.StringIO()

        with (
            mock.patch("fem_analysis.cli.load_fjw_workflow_state", return_value=workflow_state),
            contextlib.redirect_stdout(stdout),
        ):
            main(["fjw-sfepy-iterate", "--num-time-steps", "1", "--runtime-profile", "local"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["backend"], "sfepy_direct")
        self.assertEqual(payload["runtime_profile"], "local")
        self.assertEqual(payload["sfepy_linear_solver"], "scipy_iterative")
        self.assertEqual(payload["case_parallelism"], 1)
        self.assertEqual(payload["load_cases"], ["force_1", "force_2", "force_3"])
        self.assertFalse(payload["has_placeholder_adjoint"])
        self.assertEqual(payload["num_time_steps"], 1)
        self.assertIn("objective", payload)
        self.assertIn("terminal_bo_sum_by_case", payload)


if __name__ == "__main__":
    unittest.main()
