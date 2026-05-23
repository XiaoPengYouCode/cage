from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fem_analysis.fjw_workflow_adjoint import FJWAdjointLoadVector
from fem_analysis.fjw_workflow_artifacts import build_job_artifacts
from fem_analysis.fjw_workflow_execution import FJWExecutionResult
from fem_analysis.fjw_workflow_inp import render_adjoint_input, render_forward_input
from fem_analysis.fjw_workflow_loaders import load_fjw_workflow_state
from fem_analysis.fjw_workflow_single_case import FJWAdjointSolveRequest, FJWForwardSolveRequest
from fem_analysis.fjw_workflow_solver_adapters import (
    FJWAbaqusWorkflowSolverConfig,
    _build_step_workflow_state,
    _execution_job_location,
    build_fjw_abaqus_solver_adapters,
)
from fem_analysis.fjw_workflow_vectors import (
    FJWElementDisplacementVectorCache,
    save_element_displacement_cache,
)


def _make_dense_vector_cache(
    workflow_state,
    *,
    fill_value: float,
) -> FJWElementDisplacementVectorCache:
    num_elements = int(workflow_state.mesh.element_nodes.shape[0])
    vectors = np.full((num_elements, 24), fill_value, dtype=np.float64)
    return FJWElementDisplacementVectorCache(
        vectors_2d=vectors,
        element_ids=np.arange(1, num_elements + 1, dtype=np.int32),
    )


class FJWAbaqusWorkflowSolverAdaptersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow_state = load_fjw_workflow_state(initial_design_mode="single_load")
        cls.load_case = cls.workflow_state.load_cases[0]

    def test_forward_solver_stages_step_specific_input_and_returns_executor_vectors(self) -> None:
        workflow_state = self.workflow_state
        load_case = self.load_case
        design_cage = workflow_state.initial_state.design_cage.copy()
        obj_bo = workflow_state.initial_state.obj_bo.copy()
        design_cage[:3] = np.array([0.13, 0.57, 0.91], dtype=np.float64)
        obj_bo[:3] = np.array([0.08, 0.19, 0.28], dtype=np.float64)
        expected_step_state = _build_step_workflow_state(
            workflow_state,
            design_cage=design_cage,
            obj_bo=obj_bo,
        )
        expected_text = render_forward_input(expected_step_state, load_case_name=load_case.name)
        expected_cache = _make_dense_vector_cache(workflow_state, fill_value=1.25)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_directory = Path(temp_dir)

            def fake_execute_job(**kwargs) -> FJWExecutionResult:
                artifacts = build_job_artifacts(kwargs["run_directory"], kwargs["job_name"])
                return FJWExecutionResult(
                    artifacts=artifacts,
                    dry_run=False,
                    vector_cache=expected_cache,
                    abaqus_elapsed_seconds=2.5,
                )

            adapters = build_fjw_abaqus_solver_adapters(
                FJWAbaqusWorkflowSolverConfig(
                    run_directory=run_directory,
                    dry_run=False,
                    job_prefix="adapter",
                    execute_job=fake_execute_job,
                )
            )
            result = adapters.forward_solver.solve_forward(
                FJWForwardSolveRequest(
                    workflow_state=workflow_state,
                    load_case=load_case,
                    time_index=0,
                    design_cage=design_cage,
                    obj_bo=obj_bo,
                )
            )

            artifacts = build_job_artifacts(run_directory, "adapter_forward_force_1_t0")
            self.assertEqual(
                self._normalized_text(artifacts.inp_path.read_text(encoding="utf-8")),
                self._normalized_text(expected_text),
            )
            self.assertEqual(result.metadata["job_name"], "adapter_forward_force_1_t0")
            self.assertFalse(bool(result.metadata["dry_run"]))
            self.assertEqual(result.metadata["load_case_name"], "force_1")
            np.testing.assert_allclose(result.element_displacements, expected_cache.vectors_2d)

    def test_adjoint_solver_reads_cached_vectors_when_executor_runs_dry(self) -> None:
        workflow_state = self.workflow_state
        load_case = self.load_case
        design_cage = workflow_state.initial_state.design_cage.copy()
        obj_bo = workflow_state.initial_state.obj_bo.copy()
        design_cage[:2] = np.array([0.21, 0.79], dtype=np.float64)
        obj_bo[:2] = np.array([0.11, 0.27], dtype=np.float64)

        nodal_forces_flat = np.zeros(workflow_state.mesh.node_coordinates.shape[0] * 3, dtype=np.float64)
        nodal_forces_flat[:6] = np.array([1.5, 0.0, -0.5, 0.0, -2.0, 0.75], dtype=np.float64)
        load_vector = FJWAdjointLoadVector(
            time_index=1,
            load_case_name=load_case.name,
            nodal_forces_flat=nodal_forces_flat,
            active_node_ids=np.array([1, 2], dtype=np.int32),
            active_forces_xyz=np.array(
                [
                    [1.5, 0.0, -0.5],
                    [0.0, -2.0, 0.75],
                ],
                dtype=np.float64,
            ),
        )
        expected_step_state = _build_step_workflow_state(
            workflow_state,
            design_cage=design_cage,
            obj_bo=obj_bo,
        )
        expected_text = render_adjoint_input(
            expected_step_state,
            fv_vector=nodal_forces_flat,
        )
        expected_cache = _make_dense_vector_cache(workflow_state, fill_value=3.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_directory = Path(temp_dir)

            def fake_execute_job(**kwargs) -> FJWExecutionResult:
                artifacts = build_job_artifacts(kwargs["run_directory"], kwargs["job_name"])
                save_element_displacement_cache(artifacts.vector_cache_path, expected_cache)
                return FJWExecutionResult(
                    artifacts=artifacts,
                    dry_run=True,
                    vector_cache=None,
                    abaqus_elapsed_seconds=None,
                )

            adapters = build_fjw_abaqus_solver_adapters(
                FJWAbaqusWorkflowSolverConfig(
                    run_directory=run_directory,
                    dry_run=True,
                    job_prefix="adapter",
                    execute_job=fake_execute_job,
                )
            )
            result = adapters.adjoint_solver.solve_adjoint(
                FJWAdjointSolveRequest(
                    workflow_state=workflow_state,
                    load_case=load_case,
                    time_index=1,
                    design_cage=design_cage,
                    obj_bo=obj_bo,
                    load_vector=load_vector,
                )
            )

            artifacts = build_job_artifacts(run_directory, "adapter_adjoint_force_1_t1")
            self.assertEqual(
                self._normalized_text(artifacts.inp_path.read_text(encoding="utf-8")),
                self._normalized_text(expected_text),
            )
            self.assertTrue(bool(result.metadata["dry_run"]))
            self.assertEqual(result.source_path, artifacts.vector_cache_path)
            np.testing.assert_allclose(result.element_displacements, expected_cache.vectors_2d)

    def test_legacy_job_naming_runs_vert_inside_isolated_logical_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_directory = Path(temp_dir)
            config = FJWAbaqusWorkflowSolverConfig(
                run_directory=run_directory,
                job_prefix="iter_001",
                job_naming="legacy",
            )

            job_id = "iter_001_forward_force_1_t0"
            execution_directory, abaqus_job_name = _execution_job_location(config, job_id)

            self.assertEqual(execution_directory, run_directory / job_id)
            self.assertEqual(abaqus_job_name, "vert")
            self.assertEqual(
                build_job_artifacts(execution_directory, abaqus_job_name).odb_path,
                run_directory / job_id / "vert" / "vert.odb",
            )

    @staticmethod
    def _normalized_text(value: str) -> str:
        return value.replace("\r\n", "\n")


if __name__ == "__main__":
    unittest.main()
