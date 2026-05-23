from __future__ import annotations

import contextlib
import io
import json
import importlib.machinery
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fem_analysis.cli import main
from fem_analysis.fjw_environment import (
    check_fjw_runtime_environment,
    find_fjw_runtime_golden_outputs,
)


class FJWEnvironmentTest(unittest.TestCase):
    def test_preflight_default_is_python_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = check_fjw_runtime_environment(
                reference_dir=Path(temp_dir),
                abaqus_executable="definitely-missing-abaqus",
            )

        self.assertEqual(report.status, "pass")
        by_name = {check.name: check for check in report.checks}
        self.assertEqual(by_name["reference_dir"].status, "pass")
        self.assertEqual(by_name["python_dependency:numpy"].status, "pass")
        self.assertEqual(by_name["python_dependency:scipy"].status, "pass")
        self.assertEqual(by_name["python_dependency:sfepy"].status, "pass")
        self.assertNotIn("abaqus_executable", by_name)
        self.assertNotIn("petsc_mumps_runtime", by_name)
        self.assertNotIn("runtime_golden_outputs", by_name)

    def test_preflight_fails_when_required_runtime_dependencies_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                mock.patch("fem_analysis.fjw_environment.shutil.which", return_value=None),
                mock.patch(
                    "fem_analysis.fjw_environment.importlib.util.find_spec",
                    side_effect=_fake_find_spec(missing={"petsc4py"}),
                ),
            ):
                report = check_fjw_runtime_environment(
                    reference_dir=Path(temp_dir),
                    abaqus_executable="definitely-missing-abaqus",
                    require_abaqus=True,
                    require_petsc_mumps=True,
                    require_golden=True,
                )

        self.assertEqual(report.status, "fail")
        failed = {check.name for check in report.checks if check.status == "fail"}
        self.assertEqual(
            failed,
            {
                "abaqus_executable",
                "petsc_mumps_runtime",
                "runtime_golden_outputs",
            },
        )

    def test_preflight_accepts_captured_golden_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            golden_dir = root / "captured"
            golden_dir.mkdir()
            (golden_dir / "golden_manifest.json").write_text('{"files": []}', encoding="utf-8")
            with (
                mock.patch("fem_analysis.fjw_environment.shutil.which", return_value=None),
            ):
                report = check_fjw_runtime_environment(
                    reference_dir=root,
                    golden_directory=golden_dir,
                    require_golden=True,
                )

        by_name = {check.name: check for check in report.checks}
        self.assertEqual(by_name["runtime_golden_outputs"].status, "pass")
        self.assertEqual(report.status, "pass")

    def test_preflight_requires_petsc_mumps_not_only_petsc4py(self) -> None:
        class FakeSys:
            @staticmethod
            def hasExternalPackage(name: str) -> bool:
                return False

        class FakePETSc:
            Sys = FakeSys

        with tempfile.TemporaryDirectory() as temp_dir:
            spec = importlib.machinery.ModuleSpec("petsc4py", loader=None)
            spec.origin = "/fake/petsc4py"
            with (
                mock.patch("fem_analysis.fjw_environment.shutil.which", return_value=None),
                mock.patch(
                    "fem_analysis.fjw_environment.importlib.util.find_spec",
                    side_effect=_fake_find_spec(overrides={"petsc4py": spec}),
                ),
                mock.patch("fem_analysis.fjw_environment.importlib.import_module", return_value=FakePETSc),
            ):
                report = check_fjw_runtime_environment(
                    reference_dir=Path(temp_dir),
                    require_petsc_mumps=True,
                )

        by_name = {check.name: check for check in report.checks}
        self.assertEqual(by_name["petsc_mumps_runtime"].status, "fail")
        self.assertFalse(by_name["petsc_mumps_runtime"].metadata["has_mumps"])

    def test_preflight_passes_petsc_mumps_when_petsc_reports_mumps(self) -> None:
        class FakeSys:
            @staticmethod
            def hasExternalPackage(name: str) -> bool:
                return name == "mumps"

        class FakePETSc:
            Sys = FakeSys

        with tempfile.TemporaryDirectory() as temp_dir:
            spec = importlib.machinery.ModuleSpec("petsc4py", loader=None)
            spec.origin = "/fake/petsc4py"
            with (
                mock.patch("fem_analysis.fjw_environment.shutil.which", return_value=None),
                mock.patch(
                    "fem_analysis.fjw_environment.importlib.util.find_spec",
                    side_effect=_fake_find_spec(overrides={"petsc4py": spec}),
                ),
                mock.patch("fem_analysis.fjw_environment.importlib.import_module", return_value=FakePETSc),
            ):
                report = check_fjw_runtime_environment(
                    reference_dir=Path(temp_dir),
                    require_petsc_mumps=True,
                )

        by_name = {check.name: check for check in report.checks}
        self.assertEqual(by_name["petsc_mumps_runtime"].status, "pass")
        self.assertTrue(by_name["petsc_mumps_runtime"].metadata["has_mumps"])

    def test_runtime_golden_scan_ignores_static_obj_ele_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "obj_ele.mat").write_bytes(b"static")
            self.assertEqual(find_fjw_runtime_golden_outputs(root), ())

            force = root / "Force_1.mat"
            force.write_bytes(b"runtime")
            self.assertEqual(find_fjw_runtime_golden_outputs(root), (force,))

    def test_fjw_preflight_cli_writes_json_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = io.StringIO()
            with (
                mock.patch("fem_analysis.fjw_environment.shutil.which", return_value=None),
                contextlib.redirect_stdout(stdout),
            ):
                main(
                    [
                        "fjw-preflight",
                        "--reference-dir",
                        temp_dir,
                        "--golden-directory",
                        temp_dir,
                        "--abaqus-executable",
                        "definitely-missing-abaqus",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "pass")
        self.assertEqual(
            [check["name"] for check in payload["checks"]],
            [
                "reference_dir",
                "python_dependency:numpy",
                "python_dependency:scipy",
                "python_dependency:sfepy",
            ],
        )

    def test_fjw_preflight_cli_writes_report_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "preflight.json"
            stdout = io.StringIO()
            with (
                mock.patch("fem_analysis.fjw_environment.shutil.which", return_value=None),
                contextlib.redirect_stdout(stdout),
            ):
                main(
                    [
                        "fjw-preflight",
                        "--reference-dir",
                        temp_dir,
                        "--output",
                        str(output),
                    ]
                )

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(json.loads(stdout.getvalue())["report_path"], str(output.resolve()))

    def test_fjw_preflight_cli_exits_nonzero_for_required_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = io.StringIO()
            with (
                mock.patch("fem_analysis.fjw_environment.shutil.which", return_value=None),
                contextlib.redirect_stdout(stdout),
                self.assertRaises(SystemExit) as raised,
            ):
                main(
                    [
                        "fjw-preflight",
                        "--reference-dir",
                        temp_dir,
                        "--abaqus-executable",
                        "definitely-missing-abaqus",
                        "--require-abaqus",
                    ]
                )

        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "fail")


def _fake_find_spec(*, missing: set[str] | None = None, overrides: dict[str, object] | None = None):
    missing_names = set() if missing is None else set(missing)
    override_specs = {} if overrides is None else dict(overrides)
    real_find_spec = importlib.machinery.PathFinder.find_spec

    def find_spec(name: str):
        if name in override_specs:
            return override_specs[name]
        if name in missing_names:
            return None
        return real_find_spec(name)

    return find_spec


if __name__ == "__main__":
    unittest.main()
