from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from fem_analysis.fjw_reference import FJWReferenceModel
from fem_analysis.fjw_workflow_driver import (
    FJWWorkflowDriverRequest,
    run_fjw_workflow_iteration,
)
from fem_analysis.fjw_workflow_models import (
    FJWInitialState,
    FJWLoadCase,
    FJWMaterialConstants,
    FJWModulusBuckets,
    FJWReferenceMeshContext,
    FJWWorkflowState,
)
from fem_analysis.fjw_workflow_optimizer import FJWMMAState, FJWOptimizationTerms, FJWOptimizerStepResult
from fem_analysis.fjw_workflow_pipeline import (
    FJWAbaqusWorkflowConfig,
    execute_workflow_jobs,
    prepare_job_specs,
    prepare_workflow,
)
from fem_analysis.fjw_workflow_single_case import (
    FJWAdjointSolveRequest,
    FJWElementSolveResult,
    FJWForwardSolveRequest,
    run_single_case_workflow,
)
from fem_analysis.fjw_workflow_three_force import FORCE_CASE_ORDER


def build_minimal_workflow_state(
    *,
    load_case_names: tuple[str, ...],
    num_time_steps: int = 2,
    design_value: float = 0.3,
    bone_density: float = 0.36,
) -> FJWWorkflowState:
    node_coordinates = np.array(
        [
            [1, 1, 1],
            [2, 1, 1],
            [2, 2, 1],
            [1, 2, 1],
            [1, 1, 2],
            [2, 1, 2],
            [2, 2, 2],
            [1, 2, 2],
        ],
        dtype=np.int32,
    )
    element_nodes = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
    reference_model = FJWReferenceModel(
        node_coordinates=node_coordinates,
        element_nodes=element_nodes,
        cor_elements=np.zeros(0, dtype=np.int32),
        tra_elements=np.zeros(0, dtype=np.int32),
        cage_elements=np.array([1], dtype=np.int32),
        design_elements=np.array([1], dtype=np.int32),
        objective_elements=np.array([1], dtype=np.int32),
    )

    strain_displacement_matrix = np.zeros((6, 24), dtype=np.float64)
    strain_displacement_matrix[:, :6] = np.eye(6, dtype=np.float64)
    constitutive_matrix = np.eye(6, dtype=np.float64)

    mesh = FJWReferenceMeshContext(
        reference_model=reference_model,
        strain_displacement_matrix=strain_displacement_matrix,
        constitutive_matrix=constitutive_matrix,
        grid_shape_xyz=(1, 1, 1),
        top_node_ids=np.array([5, 6, 7, 8], dtype=np.int32),
        bottom_node_ids=np.array([1, 2, 3, 4], dtype=np.int32),
        element_anchor_indices=np.array([[0, 0, 0]], dtype=np.int32),
        design_anchor_indices=np.array([[0, 0, 0]], dtype=np.int32),
        objective_anchor_indices=np.array([[0, 0, 0]], dtype=np.int32),
    )

    material_constants = FJWMaterialConstants(
        voxel_volume=1.0,
        time_step_dt=1.0,
        num_time_steps=num_time_steps,
        bone_density_upper_bound=1.0,
        bone_modulus_0=1.0,
        bone_modulus_min=0.1,
        cage_modulus_0=2.0,
        cage_modulus_min=0.2,
        initial_bone_density=bone_density,
        single_load_initial_design_cage=0.2,
        three_load_initial_design_cage=design_value,
        cage_bucket_max_index=1,
        bone_bucket_max_index=1,
    )

    design = np.array([design_value], dtype=np.float64)
    obj_bo = np.array([bone_density], dtype=np.float64)
    modulus_buckets = FJWModulusBuckets(
        design_cage_clipped=design.copy(),
        obj_bo_clipped=obj_bo.copy(),
        E_cage=np.array([0.0], dtype=np.float64),
        E_obj=np.array([0.0], dtype=np.float64),
        cage_bucket_indices=np.array([0], dtype=np.int32),
        obj_bucket_indices=np.array([0], dtype=np.int32),
        cage_bucket_moduli=np.array([0.0], dtype=np.float64),
        obj_bucket_moduli=np.array([0.0], dtype=np.float64),
        cage_material_names=np.array(["cage"], dtype=object),
        obj_material_names=np.array(["bone"], dtype=object),
    )
    initial_state = FJWInitialState(
        mode="three_load",
        cage_3d=np.full((1, 1, 1), design_value, dtype=np.float64),
        bone_3d=np.full((1, 1, 1), bone_density, dtype=np.float64),
        design_cage=design.copy(),
        obj_bo=obj_bo.copy(),
        initial_design_total=float(np.sum(design, dtype=np.float64)),
        xold1=design.copy(),
        xold2=design.copy(),
        modulus_buckets=modulus_buckets,
    )

    load_cases = tuple(
        FJWLoadCase(
            name=name,
            template_path=Path(f"{name}.inp"),
            template_lines=("*Heading",),
            boundary_conditions=(),
            loads=(),
        )
        for name in load_case_names
    )

    return FJWWorkflowState(
        reference_dir=Path("references/fjw_work"),
        abaqus_inputs_path=Path("references/fjw_work/abaqus_inputs"),
        input_inventory_path=Path("references/fjw_work/input_inventory.json"),
        end1_template_path=Path("references/fjw_work/end1.inp"),
        mesh=mesh,
        material_constants=material_constants,
        load_cases=load_cases,
        cage_material_buckets=(),
        bone_material_buckets=(),
        background_material_buckets=(),
        initial_state=initial_state,
        assembly_controls={},
        adjoint_load_template={},
        structured_inputs={},
        input_inventory={},
    )


class RecordingForwardSolver:
    def __init__(self, case_scale: dict[str, float] | None = None) -> None:
        self.case_scale = {} if case_scale is None else dict(case_scale)
        self.requests: list[FJWForwardSolveRequest] = []

    def solve_forward(self, request: FJWForwardSolveRequest) -> FJWElementSolveResult:
        self.requests.append(request)
        scale = self.case_scale.get(request.load_case.name, 1.0)
        displacement = np.zeros((request.workflow_state.mesh.element_nodes.shape[0], 24), dtype=np.float64)
        displacement[:, :6] = scale * float(request.time_index + 1)
        return FJWElementSolveResult(
            element_displacements=displacement,
            metadata={
                "solver": "recording_forward",
                "load_case": request.load_case.name,
                "time_index": request.time_index,
            },
        )


class RecordingAdjointSolver:
    def __init__(self, case_scale: dict[str, float] | None = None) -> None:
        self.case_scale = {} if case_scale is None else dict(case_scale)
        self.requests: list[FJWAdjointSolveRequest] = []

    def solve_adjoint(self, request: FJWAdjointSolveRequest) -> FJWElementSolveResult:
        self.requests.append(request)
        scale = self.case_scale.get(request.load_case.name, 1.0)
        displacement = np.zeros((request.workflow_state.mesh.element_nodes.shape[0], 24), dtype=np.float64)
        displacement[:, :6] = scale * float(request.time_index + 1) * 0.5
        return FJWElementSolveResult(
            element_displacements=displacement,
            metadata={
                "solver": "recording_adjoint",
                "load_case": request.load_case.name,
                "time_index": request.time_index,
            },
        )


class RecordingOptimizer:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, FJWOptimizationTerms, FJWMMAState]] = []

    def step(
        self,
        design: np.ndarray,
        terms: FJWOptimizationTerms,
        state: FJWMMAState,
    ) -> FJWOptimizerStepResult:
        design_array = np.asarray(design, dtype=np.float64).reshape(-1)
        self.calls.append((design_array.copy(), terms, state))
        next_design = np.clip(design_array + 0.05, state.xmin, state.xmax)
        next_state = FJWMMAState(
            iteration=state.iteration + 1,
            xold1=next_design.copy(),
            xold2=state.xold1.copy(),
            xmin=state.xmin.copy(),
            xmax=state.xmax.copy(),
            low=state.low.copy(),
            up=state.up.copy(),
            a0=state.a0,
            a=state.a.copy(),
            c=state.c.copy(),
            d=state.d.copy(),
        )
        return FJWOptimizerStepResult(
            design=next_design,
            state=next_state,
            diagnostics={"solver": "recording_optimizer"},
        )


class SingleCaseWorkflowRegressionTest(unittest.TestCase):
    def test_run_single_case_workflow_closes_forward_and_adjoint_loop(self) -> None:
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

        self.assertEqual(result.load_case_name, "force_1")
        self.assertEqual(len(result.forward_steps), 2)
        self.assertEqual(len(result.adjoint_steps), 2)
        self.assertEqual([request.time_index for request in forward_solver.requests], [0, 1])
        self.assertEqual([request.time_index for request in adjoint_solver.requests], [1, 0])
        self.assertEqual(result.obj_bo_history.shape, (3, 1))
        np.testing.assert_allclose(result.bo_sum_history, result.obj_bo_history[:, 0])
        np.testing.assert_allclose(
            adjoint_solver.requests[0].obj_bo,
            result.forward_steps[1].obj_bo_previous,
        )
        np.testing.assert_allclose(
            adjoint_solver.requests[1].obj_bo,
            result.forward_steps[0].obj_bo_previous,
        )

        three_force_case = result.to_three_force_case_result()
        np.testing.assert_allclose(three_force_case.bo_sum, result.bo_sum_history[1:])
        self.assertEqual(len(three_force_case.adjoint_steps), 2)
        self.assertTrue(np.all(result.fai_history[-1] == 1.0))


class WorkflowDriverRegressionTest(unittest.TestCase):
    def test_run_fjw_workflow_iteration_runs_three_force_driver_loop(self) -> None:
        workflow_state = build_minimal_workflow_state(
            load_case_names=FORCE_CASE_ORDER,
            num_time_steps=2,
        )
        forward_solver = RecordingForwardSolver(
            case_scale={"force_1": 1.0, "force_2": 2.0, "force_3": 3.0},
        )
        adjoint_solver = RecordingAdjointSolver(
            case_scale={"force_1": 1.0, "force_2": 1.5, "force_3": 2.0},
        )
        optimizer = RecordingOptimizer()

        result = run_fjw_workflow_iteration(
            FJWWorkflowDriverRequest(
                workflow_state=workflow_state,
                num_time_steps=2,
                optimizer=optimizer,
            ),
            forward_solver=forward_solver,
            adjoint_solver=adjoint_solver,
        )

        self.assertEqual(result.load_case_names, FORCE_CASE_ORDER)
        self.assertEqual(
            [case_result.load_case_name for case_result in result.single_case_results],
            list(FORCE_CASE_ORDER),
        )
        self.assertEqual(len(forward_solver.requests), 6)
        self.assertEqual(len(adjoint_solver.requests), 6)
        self.assertEqual(len(optimizer.calls), 1)
        self.assertEqual(result.iteration_state.iteration_index, 1)
        self.assertFalse(result.iteration_state.has_placeholder_adjoint)
        self.assertEqual(
            [record.load_case_name for record in result.iteration_state.case_records],
            list(FORCE_CASE_ORDER),
        )
        self.assertTrue(
            all(record.adjoint_source == "manual" for record in result.iteration_state.case_records)
        )
        np.testing.assert_allclose(
            result.iteration_state.next_design,
            workflow_state.initial_state.design_cage + 0.05,
        )
        self.assertEqual(result.metadata["workflow_state_source"], "provided")
        self.assertEqual(result.metadata["num_time_steps"], 2)

        optimizer_design, optimizer_terms, optimizer_state = optimizer.calls[0]
        np.testing.assert_allclose(optimizer_design, workflow_state.initial_state.design_cage)
        self.assertEqual(optimizer_terms.design_size, workflow_state.initial_state.design_cage.size)
        self.assertLess(optimizer_terms.objective, 0.0)
        self.assertEqual(optimizer_state.iteration, 0)

class WorkflowPipelineRegressionTest(unittest.TestCase):
    def test_prepare_and_execute_workflow_jobs_write_dry_run_manifests(self) -> None:
        workflow_state = build_minimal_workflow_state(
            load_case_names=FORCE_CASE_ORDER,
            num_time_steps=2,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_directory = temp_path / "runs" / "fjw_workflow"
            run_directory.mkdir(parents=True, exist_ok=True)
            stale_lock = run_directory / "vert_force_1.lck"
            stale_lock.write_text("stale", encoding="utf-8")

            config = FJWAbaqusWorkflowConfig(
                reference_dir=temp_path / "references" / "fjw_work",
                run_directory=run_directory,
                mode="three-force",
                time_steps=2,
                dry_run=True,
            )

            def fake_generate_workflow_input_files(
                _workflow_state: FJWWorkflowState,
                *,
                run_directory: Path,
                mode: str,
                time_steps: int,
            ) -> None:
                force_cases = ["force_1"] if mode == "single-force" else list(FORCE_CASE_ORDER)
                for load_case in force_cases:
                    (run_directory / f"vert_{load_case}.inp").write_text("*HEADING\n", encoding="utf-8")
                    for time_index in range(time_steps - 1, -1, -1):
                        (run_directory / f"adjoint_{load_case}_t{time_index}.inp").write_text(
                            "*HEADING\n",
                            encoding="utf-8",
                        )

            with mock.patch(
                "fem_analysis.fjw_workflow_pipeline.load_fjw_workflow_state",
                return_value=workflow_state,
            ) as load_state_mock, mock.patch(
                "fem_analysis.fjw_workflow_pipeline.generate_workflow_input_files",
                side_effect=fake_generate_workflow_input_files,
            ) as generate_inputs_mock:
                prepared = prepare_workflow(config)
                execution_manifest = execute_workflow_jobs(prepared)

            self.assertEqual(len(prepared.jobs), len(prepare_job_specs(config)))
            self.assertEqual(len(execution_manifest.jobs), len(prepared.jobs))
            self.assertTrue(all(job["dry_run"] for job in execution_manifest.jobs))
            self.assertTrue(all("vector_cache" not in job for job in execution_manifest.jobs))
            self.assertTrue((run_directory / "workflow_manifest.json").exists())
            self.assertTrue((run_directory / "workflow_execution_manifest.json").exists())
            self.assertTrue((run_directory / "odbFieldOutput1.py").exists())

            with (run_directory / "workflow_manifest.json").open("r", encoding="utf-8") as handle:
                workflow_manifest_payload = json.load(handle)
            with (run_directory / "workflow_execution_manifest.json").open("r", encoding="utf-8") as handle:
                execution_manifest_payload = json.load(handle)

            self.assertEqual(len(workflow_manifest_payload["generated_jobs"]), len(prepared.jobs))
            self.assertEqual(
                workflow_manifest_payload["removed_stale_locks"],
                [str(stale_lock)],
            )
            self.assertTrue(execution_manifest_payload["config"]["dry_run"])
            self.assertTrue(execution_manifest_payload["config"]["include_forward"])
            self.assertTrue(execution_manifest_payload["config"]["include_adjoint"])

            for job in execution_manifest.jobs:
                metadata_path = Path(job["artifacts"]["metadata_path"])
                export_script_path = Path(job["artifacts"]["odb_export_script_path"])
                self.assertTrue(metadata_path.exists())
                self.assertTrue(export_script_path.exists())

            load_state_mock.assert_called_once_with(
                reference_dir=config.reference_dir,
                initial_design_mode="three_load",
            )
            generate_inputs_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
