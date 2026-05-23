from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fem_analysis.fjw_workflow_execution import execute_job_and_collect


class FJWAbaqusRealRunContractTest(unittest.TestCase):
    def test_real_run_fails_before_creating_partial_solver_result_without_abaqus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_dir = root / "contract_job"
            job_dir.mkdir()
            (job_dir / "contract_job.inp").write_text("*Heading\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "was not found on PATH"):
                execute_job_and_collect(
                    run_directory=root,
                    job_name="contract_job",
                    workflow_or_mesh=object(),
                    abaqus_executable="definitely_missing_abaqus_executable",
                    dry_run=False,
                )

            self.assertFalse((job_dir / "contract_job_U1_vectors.npz").exists())


if __name__ == "__main__":
    unittest.main()
