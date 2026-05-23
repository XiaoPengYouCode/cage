from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fem_analysis.fjw_workflow_artifacts import build_job_artifacts
from fem_analysis.fjw_workflow_abaqus import build_standard_job_command


class FJWAbaqusBackendContractTest(unittest.TestCase):
    def test_job_artifacts_are_isolated_per_job_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifacts = build_job_artifacts(root, "solver_forward_force_1_t0")
            command = build_standard_job_command(
                job_name=artifacts.job_name,
                cpus=8,
                workdir=artifacts.run_directory,
            )

            self.assertEqual(artifacts.run_directory, root / "solver_forward_force_1_t0")
            self.assertEqual(artifacts.inp_path.parent, artifacts.run_directory)
            self.assertEqual(command.lock_file, artifacts.lock_path)


if __name__ == "__main__":
    unittest.main()
