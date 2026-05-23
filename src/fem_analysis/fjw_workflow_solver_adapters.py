from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from .fjw_workflow_abaqus import build_standard_job_command, remove_stale_lock
from .fjw_workflow_artifacts import build_job_artifacts
from .fjw_workflow_execution import FJWExecutionResult, execute_job_and_collect
from .fjw_workflow_inp import render_adjoint_input, render_forward_input
from .fjw_workflow_loaders import compute_modulus_buckets
from .fjw_workflow_models import FJWInitialState, FJWWorkflowState
from .fjw_workflow_single_case import (
    FJWAdjointSolveRequest,
    FJWAdjointSolver,
    FJWElementSolveResult,
    FJWForwardSolveRequest,
    FJWForwardSolver,
)
from .fjw_workflow_vectors import (
    FJWElementDisplacementVectorCache,
    build_element_displacement_cache_from_u1,
    load_element_displacement_cache,
)


FJWAbaqusJobExecutor = Callable[..., FJWExecutionResult]


@dataclass(frozen=True, slots=True)
class FJWAbaqusWorkflowSolverConfig:
    run_directory: Path = Path("runs/fjw_workflow_solver")
    abaqus_executable: str = "abaqus"
    cpus: int = 8
    dry_run: bool = True
    job_prefix: str = "solver"
    job_naming: Literal["safe", "legacy"] = "safe"
    remove_stale_locks: bool = True
    execute_job: FJWAbaqusJobExecutor = execute_job_and_collect

    def __post_init__(self) -> None:
        if self.job_naming not in ("safe", "legacy"):
            raise ValueError("job_naming must be 'safe' or 'legacy'.")


@dataclass(frozen=True, slots=True)
class FJWAbaqusWorkflowSolverBackend:
    config: FJWAbaqusWorkflowSolverConfig

    def solve_forward(self, request: FJWForwardSolveRequest) -> FJWElementSolveResult:
        step_state = _build_step_workflow_state(
            request.workflow_state,
            design_cage=request.design_cage,
            obj_bo=request.obj_bo,
        )
        job_id = _compose_job_name(
            prefix=self.config.job_prefix,
            solver_kind="forward",
            load_case_name=request.load_case.name,
            time_index=request.time_index,
        )
        self._stage_input_file(
            job_id,
            render_forward_input(step_state, load_case_name=request.load_case.name),
        )
        execution_result = self._execute_job(job_id=job_id, workflow_state=step_state)
        vector_cache = _resolve_vector_cache(execution_result, step_state)
        return _build_element_solve_result(
            solver_kind="forward",
            job_name=job_id,
            execution_result=execution_result,
            vector_cache=vector_cache,
            load_case_name=request.load_case.name,
            time_index=request.time_index,
        )

    def solve_adjoint(self, request: FJWAdjointSolveRequest) -> FJWElementSolveResult:
        step_state = _build_step_workflow_state(
            request.workflow_state,
            design_cage=request.design_cage,
            obj_bo=request.obj_bo,
        )
        job_id = _compose_job_name(
            prefix=self.config.job_prefix,
            solver_kind="adjoint",
            load_case_name=request.load_case.name,
            time_index=request.time_index,
        )
        self._stage_input_file(
            job_id,
            render_adjoint_input(
                step_state,
                fv_vector=request.load_vector.nodal_forces_flat,
            ),
        )
        execution_result = self._execute_job(job_id=job_id, workflow_state=step_state)
        vector_cache = _resolve_vector_cache(execution_result, step_state)
        return _build_element_solve_result(
            solver_kind="adjoint",
            job_name=job_id,
            execution_result=execution_result,
            vector_cache=vector_cache,
            load_case_name=request.load_case.name,
            time_index=request.time_index,
        )

    def _stage_input_file(self, job_id: str, content: str) -> Path:
        run_directory, abaqus_job_name = _execution_job_location(self.config, job_id)
        artifacts = build_job_artifacts(run_directory, abaqus_job_name)
        artifacts.inp_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts.inp_path.write_text(content, encoding="utf-8")
        return artifacts.inp_path

    def _execute_job(self, *, job_id: str, workflow_state: FJWWorkflowState) -> FJWExecutionResult:
        run_directory, abaqus_job_name = _execution_job_location(self.config, job_id)
        run_directory.mkdir(parents=True, exist_ok=True)

        if self.config.remove_stale_locks:
            artifacts = build_job_artifacts(run_directory, abaqus_job_name)
            remove_stale_lock(artifacts.lock_path)
            remove_stale_lock(run_directory / f"{abaqus_job_name}.lck")

        return self.config.execute_job(
            run_directory=run_directory,
            job_name=abaqus_job_name,
            workflow_or_mesh=workflow_state,
            abaqus_executable=self.config.abaqus_executable,
            cpus=self.config.cpus,
            dry_run=self.config.dry_run,
        )


@dataclass(frozen=True, slots=True)
class FJWAbaqusForwardSolverAdapter(FJWForwardSolver):
    backend: FJWAbaqusWorkflowSolverBackend

    def solve_forward(self, request: FJWForwardSolveRequest) -> FJWElementSolveResult:
        return self.backend.solve_forward(request)


@dataclass(frozen=True, slots=True)
class FJWAbaqusAdjointSolverAdapter(FJWAdjointSolver):
    backend: FJWAbaqusWorkflowSolverBackend

    def solve_adjoint(self, request: FJWAdjointSolveRequest) -> FJWElementSolveResult:
        return self.backend.solve_adjoint(request)


@dataclass(frozen=True, slots=True)
class FJWAbaqusWorkflowSolverAdapters:
    backend: FJWAbaqusWorkflowSolverBackend
    forward_solver: FJWAbaqusForwardSolverAdapter
    adjoint_solver: FJWAbaqusAdjointSolverAdapter


def build_fjw_abaqus_solver_adapters(
    config: FJWAbaqusWorkflowSolverConfig | None = None,
) -> FJWAbaqusWorkflowSolverAdapters:
    resolved_config = config or FJWAbaqusWorkflowSolverConfig()
    backend = FJWAbaqusWorkflowSolverBackend(config=resolved_config)
    return FJWAbaqusWorkflowSolverAdapters(
        backend=backend,
        forward_solver=FJWAbaqusForwardSolverAdapter(backend=backend),
        adjoint_solver=FJWAbaqusAdjointSolverAdapter(backend=backend),
    )


def _compose_job_name(
    *,
    prefix: str,
    solver_kind: str,
    load_case_name: str,
    time_index: int,
) -> str:
    base_name = f"{solver_kind}_{load_case_name}_t{int(time_index)}"
    if prefix.strip():
        base_name = f"{prefix}_{base_name}"
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_name)
    return sanitized.strip("._-") or "solver_job"


def _execution_job_location(
    config: FJWAbaqusWorkflowSolverConfig,
    job_id: str,
) -> tuple[Path, str]:
    if config.job_naming == "legacy":
        return Path(config.run_directory) / job_id, "vert"
    return Path(config.run_directory), job_id


def _build_step_workflow_state(
    workflow_state: FJWWorkflowState,
    *,
    design_cage: np.ndarray,
    obj_bo: np.ndarray,
) -> FJWWorkflowState:
    return replace(
        workflow_state,
        initial_state=_build_step_initial_state(
            workflow_state,
            design_cage=design_cage,
            obj_bo=obj_bo,
        ),
    )


def _build_step_initial_state(
    workflow_state: FJWWorkflowState,
    *,
    design_cage: np.ndarray,
    obj_bo: np.ndarray,
) -> FJWInitialState:
    design_values = np.asarray(design_cage, dtype=np.float64).reshape(-1).copy()
    obj_values = np.asarray(obj_bo, dtype=np.float64).reshape(-1).copy()

    expected_design_size = int(workflow_state.mesh.design_elements.shape[0])
    expected_objective_size = int(workflow_state.mesh.objective_elements.shape[0])
    if design_values.size != expected_design_size:
        raise ValueError(
            "design_cage size does not match workflow_state.mesh.design_elements: "
            f"{design_values.size} != {expected_design_size}."
        )
    if obj_values.size != expected_objective_size:
        raise ValueError(
            "obj_bo size does not match workflow_state.mesh.objective_elements: "
            f"{obj_values.size} != {expected_objective_size}."
        )

    modulus_buckets = compute_modulus_buckets(
        design_values,
        obj_values,
        workflow_state.material_constants,
        cage_material_buckets=workflow_state.cage_material_buckets,
        bone_material_buckets=workflow_state.bone_material_buckets,
    )
    cage_3d = _embed_field_values(
        workflow_state.initial_state.cage_3d,
        workflow_state.mesh.design_anchor_indices,
        design_values,
    )
    bone_3d = _embed_field_values(
        workflow_state.initial_state.bone_3d,
        workflow_state.mesh.objective_anchor_indices,
        obj_values,
    )

    return replace(
        workflow_state.initial_state,
        cage_3d=cage_3d,
        bone_3d=bone_3d,
        design_cage=design_values,
        obj_bo=obj_values,
        initial_design_total=float(np.sum(design_values, dtype=np.float64)),
        xold1=design_values.copy(),
        xold2=design_values.copy(),
        modulus_buckets=modulus_buckets,
    )


def _embed_field_values(
    grid: np.ndarray,
    anchor_indices: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    embedded = np.asarray(grid, dtype=np.float64).copy()
    anchors = np.asarray(anchor_indices, dtype=np.int32)
    embedded[
        anchors[:, 0],
        anchors[:, 1],
        anchors[:, 2],
    ] = np.asarray(values, dtype=np.float64).reshape(-1)
    return embedded


def _resolve_vector_cache(
    execution_result: FJWExecutionResult,
    workflow_state: FJWWorkflowState,
) -> FJWElementDisplacementVectorCache:
    if execution_result.vector_cache is not None:
        return execution_result.vector_cache

    artifacts = execution_result.artifacts
    if artifacts.vector_cache_path.exists():
        return load_element_displacement_cache(artifacts.vector_cache_path)
    if artifacts.u1_path.exists():
        return build_element_displacement_cache_from_u1(
            artifacts.u1_path,
            workflow_state,
            cache_name=artifacts.job_name,
            strict=False,
        )

    raise RuntimeError(
        "Abaqus solver adapter could not resolve element displacements for "
        f"{artifacts.job_name!r}. Expected one of "
        f"{artifacts.vector_cache_path} or {artifacts.u1_path} to exist, "
        f"or execute_job to return vector_cache directly."
    )


def _build_element_solve_result(
    *,
    solver_kind: str,
    job_name: str,
    execution_result: FJWExecutionResult,
    vector_cache: FJWElementDisplacementVectorCache,
    load_case_name: str,
    time_index: int,
) -> FJWElementSolveResult:
    source_path = vector_cache.source_result_path
    if source_path is None and execution_result.artifacts.vector_cache_path.exists():
        source_path = execution_result.artifacts.vector_cache_path
    if source_path is None and execution_result.artifacts.u1_path.exists():
        source_path = execution_result.artifacts.u1_path

    return FJWElementSolveResult(
        element_displacements=vector_cache.vectors_2d,
        source_path=source_path,
        metadata={
            "solver_kind": solver_kind,
            "job_name": job_name,
            "load_case_name": load_case_name,
            "time_index": int(time_index),
            "dry_run": execution_result.dry_run,
            "abaqus_elapsed_seconds": execution_result.abaqus_elapsed_seconds,
            "artifacts": execution_result.artifacts.as_jsonable(),
        },
    )


__all__ = [
    "FJWAbaqusAdjointSolverAdapter",
    "FJWAbaqusForwardSolverAdapter",
    "FJWAbaqusWorkflowSolverAdapters",
    "FJWAbaqusWorkflowSolverBackend",
    "FJWAbaqusWorkflowSolverConfig",
    "_execution_job_location",
    "build_fjw_abaqus_solver_adapters",
]
