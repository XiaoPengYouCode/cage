from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from fem_analysis.fjw_workflow_abaqus import (
    AbaqusCommand,
    build_standard_job_command,
    remove_stale_lock,
)
from fem_analysis.fjw_workflow_artifacts import build_job_artifacts
from fem_analysis.fjw_workflow_execution import FJWExecutionResult, execute_job_and_collect
from fem_analysis.fjw_workflow_inp import generate_workflow_input_files
from fem_analysis.fjw_workflow_loaders import load_fjw_workflow_state
from fem_analysis.fjw_workflow_models import FJWWorkflowState
from fem_analysis.fjw_workflow_odb import write_abaqus_odb_export_script


WorkflowMode = Literal["single-force", "three-force"]


@dataclass(frozen=True)
class FJWAbaqusWorkflowConfig:
    reference_dir: Path = Path("references/fjw_work")
    run_directory: Path = Path("runs/fjw_workflow")
    abaqus_executable: str = "abaqus"
    cpus: int = 8
    mode: WorkflowMode = "three-force"
    time_steps: int = 3
    initial_bone_density: float = 0.36
    initial_single_force_design_density: float = 0.2
    initial_three_force_design_density: float = 0.3
    dry_run: bool = True


@dataclass(frozen=True)
class FJWJobSpec:
    name: str
    inp_path: Path
    kind: Literal["forward", "adjoint"]
    load_case: str


@dataclass(frozen=True)
class FJWWorkflowManifest:
    config: dict[str, object]
    generated_jobs: list[dict[str, object]]
    removed_stale_locks: list[str]


@dataclass(frozen=True)
class FJWPreparedWorkflow:
    config: FJWAbaqusWorkflowConfig
    workflow_state: FJWWorkflowState
    manifest: FJWWorkflowManifest
    jobs: tuple[FJWJobSpec, ...]


@dataclass(frozen=True)
class FJWWorkflowExecutionManifest:
    config: dict[str, object]
    jobs: list[dict[str, object]]


def _config_as_jsonable(config: FJWAbaqusWorkflowConfig) -> dict[str, object]:
    payload = asdict(config)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _ensure_run_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_manifest(path: Path, manifest: FJWWorkflowManifest) -> None:
    path.write_text(
        json.dumps(
            {
                "config": manifest.config,
                "generated_jobs": manifest.generated_jobs,
                "removed_stale_locks": manifest.removed_stale_locks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_execution_manifest(path: Path, manifest: FJWWorkflowExecutionManifest) -> None:
    path.write_text(
        json.dumps(
            {
                "config": manifest.config,
                "jobs": manifest.jobs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _build_forward_job_specs(run_directory: Path, mode: WorkflowMode) -> list[FJWJobSpec]:
    force_cases = ["force_1"] if mode == "single-force" else ["force_1", "force_2", "force_3"]
    specs: list[FJWJobSpec] = []
    for load_case in force_cases:
        name = f"vert_{load_case}"
        specs.append(
            FJWJobSpec(
                name=name,
                inp_path=build_job_artifacts(run_directory, name).inp_path,
                kind="forward",
                load_case=load_case,
            )
        )
    return specs


def _build_adjoint_job_specs(
    run_directory: Path,
    mode: WorkflowMode,
    time_steps: int,
) -> list[FJWJobSpec]:
    force_cases = ["force_1"] if mode == "single-force" else ["force_1", "force_2", "force_3"]
    specs: list[FJWJobSpec] = []
    for load_case in force_cases:
        for time_index in range(time_steps - 1, -1, -1):
            name = f"adjoint_{load_case}_t{time_index}"
            specs.append(
                FJWJobSpec(
                    name=name,
                    inp_path=build_job_artifacts(run_directory, name).inp_path,
                    kind="adjoint",
                    load_case=load_case,
                )
            )
    return specs


def prepare_job_specs(config: FJWAbaqusWorkflowConfig) -> list[FJWJobSpec]:
    return _build_forward_job_specs(config.run_directory, config.mode) + _build_adjoint_job_specs(
        config.run_directory,
        config.mode,
        config.time_steps,
    )


def build_job_commands(
    config: FJWAbaqusWorkflowConfig,
    jobs: list[FJWJobSpec],
) -> list[AbaqusCommand]:
    return [
        build_standard_job_command(
            job_name=job.name,
            cpus=config.cpus,
            workdir=build_job_artifacts(config.run_directory, job.name).run_directory,
            abaqus_executable=config.abaqus_executable,
        )
        for job in jobs
    ]


def render_reference_odb_export_script(reference_dir: Path, destination: Path) -> Path:
    _ = reference_dir
    return write_abaqus_odb_export_script(
        destination,
        odb_filename="vert.odb",
        output_filename="U1.txt",
        field_name="U",
        step_name="Load",
    )


def prepare_workflow_manifest(config: FJWAbaqusWorkflowConfig) -> FJWWorkflowManifest:
    _ensure_run_directory(config.run_directory)

    removed_stale_locks: list[str] = []
    jobs = prepare_job_specs(config)
    for job in jobs:
        artifacts = build_job_artifacts(config.run_directory, job.name)
        if remove_stale_lock(artifacts.lock_path):
            removed_stale_locks.append(str(artifacts.lock_path))
        legacy_lock_path = config.run_directory / f"{job.name}.lck"
        if remove_stale_lock(legacy_lock_path):
            removed_stale_locks.append(str(legacy_lock_path))

    manifest = FJWWorkflowManifest(
        config=_config_as_jsonable(config),
        generated_jobs=[
            {
                "name": job.name,
                "inp_path": str(job.inp_path),
                "kind": job.kind,
                "load_case": job.load_case,
                "artifacts": build_job_artifacts(config.run_directory, job.name).as_jsonable(),
            }
            for job in jobs
        ],
        removed_stale_locks=removed_stale_locks,
    )
    _write_manifest(config.run_directory / "workflow_manifest.json", manifest)
    return manifest


def run_workflow_dry_run(config: FJWAbaqusWorkflowConfig) -> FJWWorkflowManifest:
    prepared = prepare_workflow(config)
    return prepared.manifest


def prepare_workflow(config: FJWAbaqusWorkflowConfig) -> FJWPreparedWorkflow:
    manifest = prepare_workflow_manifest(config)
    initial_design_mode = "single_load" if config.mode == "single-force" else "three_load"
    workflow_state = load_fjw_workflow_state(
        reference_dir=config.reference_dir,
        initial_design_mode=initial_design_mode,
    )
    generate_workflow_input_files(
        workflow_state,
        run_directory=config.run_directory,
        mode=config.mode,
        time_steps=config.time_steps,
    )
    render_reference_odb_export_script(
        config.reference_dir,
        config.run_directory / "odbFieldOutput1.py",
    )
    return FJWPreparedWorkflow(
        config=config,
        workflow_state=workflow_state,
        manifest=manifest,
        jobs=tuple(prepare_job_specs(config)),
    )


def execute_workflow_jobs(
    prepared: FJWPreparedWorkflow,
    *,
    include_forward: bool = True,
    include_adjoint: bool = True,
    dry_run: bool | None = None,
) -> FJWWorkflowExecutionManifest:
    selected_jobs = [
        job
        for job in prepared.jobs
        if (include_forward and job.kind == "forward") or (include_adjoint and job.kind == "adjoint")
    ]
    resolved_dry_run = prepared.config.dry_run if dry_run is None else bool(dry_run)

    execution_jobs: list[dict[str, object]] = []
    for job in selected_jobs:
        result = execute_job_and_collect(
            run_directory=prepared.config.run_directory,
            job_name=job.name,
            workflow_or_mesh=prepared.workflow_state,
            abaqus_executable=prepared.config.abaqus_executable,
            cpus=prepared.config.cpus,
            dry_run=resolved_dry_run,
        )
        execution_jobs.append(_execution_result_as_jsonable(job, result))

    execution_manifest = FJWWorkflowExecutionManifest(
        config={
            **prepared.manifest.config,
            "include_forward": include_forward,
            "include_adjoint": include_adjoint,
            "dry_run": resolved_dry_run,
        },
        jobs=execution_jobs,
    )
    _write_execution_manifest(
        prepared.config.run_directory / "workflow_execution_manifest.json",
        execution_manifest,
    )
    return execution_manifest


def _execution_result_as_jsonable(
    job: FJWJobSpec,
    result: FJWExecutionResult,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": job.name,
        "kind": job.kind,
        "load_case": job.load_case,
        "dry_run": result.dry_run,
        "artifacts": result.artifacts.as_jsonable(),
        "abaqus_elapsed_seconds": result.abaqus_elapsed_seconds,
    }
    if result.vector_cache is not None:
        payload["vector_cache"] = {
            "cache_name": result.vector_cache.cache_name,
            "num_elements": result.vector_cache.num_elements,
            "source_result_path": (
                None
                if result.vector_cache.source_result_path is None
                else str(result.vector_cache.source_result_path)
            ),
        }
    return payload


__all__ = [
    "FJWAbaqusWorkflowConfig",
    "FJWJobSpec",
    "FJWPreparedWorkflow",
    "FJWWorkflowExecutionManifest",
    "FJWWorkflowManifest",
    "build_job_commands",
    "execute_workflow_jobs",
    "prepare_job_specs",
    "prepare_workflow",
    "prepare_workflow_manifest",
    "render_reference_odb_export_script",
    "run_workflow_dry_run",
]
