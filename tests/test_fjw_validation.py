from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from fem_analysis.fjw_workflow_optimize import FJWOptimizationConfig, run_fjw_optimization
from fem_analysis.fjw_validation import (
    capture_fjw_golden_run,
    compare_golden_manifest,
    compare_npz_arrays,
    normalize_inp_text,
    validate_run_directory,
    write_validation_report,
)
from tests.test_fjw_workflow_sfepy_runner import build_minimal_sfepy_iteration_state


class FJWValidationTest(unittest.TestCase):
    def test_normalize_inp_text_removes_spacing_and_case_noise(self) -> None:
        self.assertEqual(
            normalize_inp_text("*Cload, op=NEW\r\n  VERT-1.1, 1,  2.0\r\n"),
            "*cload,op=new\nvert-1.1,1,2.0",
        )

    def test_compare_npz_arrays_reports_max_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            actual = root / "actual.npz"
            expected = root / "expected.npz"
            np.savez_compressed(actual, values=np.array([1.0, 2.0]))
            np.savez_compressed(expected, values=np.array([1.0, 2.1]))

            checks = compare_npz_arrays(actual, expected, atol=1e-12, rtol=1e-12)

            self.assertEqual(checks[0].status, "fail")
            self.assertAlmostEqual(checks[0].metadata["max_error"], 0.1)

    def test_validate_run_directory_warns_without_golden(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "workflow_manifest.json").write_text("{}", encoding="utf-8")
            checkpoint = root / "iter_000"
            checkpoint.mkdir()
            (checkpoint / "iteration_state.json").write_text(
                '{"iteration_index": 0, "checkpoint_kind": "initial"}',
                encoding="utf-8",
            )
            np.savez_compressed(checkpoint / "design_cage.npz", design_cage=np.array([0.3]))
            np.savez_compressed(
                checkpoint / "mma_state.npz",
                iteration=np.array(0),
                xold1=np.array([0.3]),
                xold2=np.array([0.3]),
                xmin=np.array([0.001]),
                xmax=np.array([1.0]),
                low=np.array([0.0]),
                up=np.array([0.0]),
                a0=np.array(1.0),
                a=np.array([0.0]),
                c=np.array([1000.0]),
                d=np.array([1.0]),
            )

            report = validate_run_directory(root)
            report_path = write_validation_report(report)

            self.assertEqual(report.status, "warn")
            self.assertEqual(
                report.historical_equivalence_claim,
                "cannot_prove_historical_equivalence_without_matching_golden_outputs",
            )
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "warn")

    def test_capture_golden_run_writes_manifest_and_copies_small_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            golden_dir = root / "golden"
            checkpoint = run_dir / "iter_000"
            checkpoint.mkdir(parents=True)
            (run_dir / "workflow_manifest.json").write_text('{"ok": true}', encoding="utf-8")
            (checkpoint / "iteration_state.json").write_text(
                '{"iteration_index": 0, "checkpoint_kind": "initial"}',
                encoding="utf-8",
            )
            np.savez_compressed(checkpoint / "design_cage.npz", design_cage=np.array([0.3, 0.4]))
            np.savez_compressed(
                checkpoint / "mma_state.npz",
                iteration=np.array(0),
                xold1=np.array([0.3, 0.4]),
                xold2=np.array([0.3, 0.4]),
                xmin=np.array([0.001, 0.001]),
                xmax=np.array([1.0, 1.0]),
                low=np.array([0.0, 0.0]),
                up=np.array([0.0, 0.0]),
                a0=np.array(1.0),
                a=np.array([0.0]),
                c=np.array([1000.0]),
                d=np.array([1.0]),
            )
            (checkpoint / "large_U1_vectors.npz").write_bytes(b"large-vector-cache" * 10)

            report = capture_fjw_golden_run(run_dir, golden_dir, copy_max_bytes=50)
            manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["source_validation_status"], "warn")
            self.assertTrue((golden_dir / "workflow_manifest.json").exists())
            self.assertFalse((golden_dir / "iter_000" / "large_U1_vectors.npz").exists())
            by_path = {record["relative_path"]: record for record in manifest["files"]}
            self.assertFalse(by_path["iter_000/large_U1_vectors.npz"]["copied"])
            self.assertEqual(
                compare_golden_manifest(run_dir, report.manifest_path)[0].status,
                "pass",
            )

    def test_capture_golden_run_rejects_invalid_run_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "workflow_manifest.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid run directory"):
                capture_fjw_golden_run(run_dir, root / "golden")

            report = capture_fjw_golden_run(
                run_dir,
                root / "golden",
                require_valid_run=False,
            )
            self.assertTrue(report.manifest_path.exists())

    def test_validate_run_directory_compares_golden_manifest_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            golden_dir = root / "golden"
            checkpoint = run_dir / "iter_000"
            checkpoint.mkdir(parents=True)
            (run_dir / "workflow_manifest.json").write_text("{}", encoding="utf-8")
            (checkpoint / "iteration_state.json").write_text(
                '{"iteration_index": 0, "checkpoint_kind": "initial"}',
                encoding="utf-8",
            )
            np.savez_compressed(checkpoint / "design_cage.npz", design_cage=np.array([0.3]))
            np.savez_compressed(
                checkpoint / "mma_state.npz",
                iteration=np.array(0),
                xold1=np.array([0.3]),
                xold2=np.array([0.3]),
                xmin=np.array([0.001]),
                xmax=np.array([1.0]),
                low=np.array([0.0]),
                up=np.array([0.0]),
                a0=np.array(1.0),
                a=np.array([0.0]),
                c=np.array([1000.0]),
                d=np.array([1.0]),
            )
            capture_fjw_golden_run(run_dir, golden_dir)

            passing_report = validate_run_directory(run_dir, golden_directory=golden_dir)
            self.assertEqual(passing_report.status, "pass")

            (run_dir / "workflow_manifest.json").write_text('{"changed": true}', encoding="utf-8")
            failing_report = validate_run_directory(run_dir, golden_directory=golden_dir)
            self.assertEqual(failing_report.status, "fail")
            self.assertTrue(
                any(
                    check.name == "golden_manifest:workflow_manifest.json" and check.status == "fail"
                    for check in failing_report.checks
                )
            )

    def test_validate_run_directory_checks_solver_and_adjoint_artifacts(self) -> None:
        workflow_state = build_minimal_sfepy_iteration_state()
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            golden_dir = Path(temp_dir) / "golden"
            with mock.patch(
                "fem_analysis.fjw_workflow_optimize.load_fjw_workflow_state",
                return_value=workflow_state,
            ):
                run_fjw_optimization(
                    FJWOptimizationConfig(
                        run_directory=run_dir,
                        max_iterations=1,
                        num_time_steps=1,
                    )
                )

            report = validate_run_directory(run_dir)

            self.assertEqual(report.status, "warn")
            self.assertFalse([check for check in report.checks if check.status == "fail"])
            by_name = {check.name: check for check in report.checks}
            self.assertEqual(
                by_name["iter_001:has_placeholder_adjoint"].status,
                "pass",
            )
            self.assertEqual(
                by_name[
                    "iter_001:force_1:forward_t0:forward_step.npz:full_element_displacements:nonzero"
                ].status,
                "pass",
            )
            self.assertEqual(
                by_name[
                    "iter_001:force_1:adjoint_t0:fai.npz:adjoint_element_displacements:nonzero"
                ].status,
                "warn",
            )
            self.assertEqual(
                by_name["iter_001:force_1:adjoint_t0:fv_manifest:dense_l2_norm"].status,
                "warn",
            )

            capture_report = capture_fjw_golden_run(run_dir, golden_dir)
            manifest = json.loads(capture_report.manifest_path.read_text(encoding="utf-8"))
            paths = {record["relative_path"] for record in manifest["files"]}
            self.assertIn("iter_001/force_1/adjoint_t0/fv_manifest.json", paths)


if __name__ == "__main__":
    unittest.main()
