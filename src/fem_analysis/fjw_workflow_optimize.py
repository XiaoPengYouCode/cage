from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .fjw_runtime_config import configure_numeric_runtime_threads, get_fjw_runtime_config
from .fjw_workflow_checkpoint_io import (
    load_resume_state,
    write_initial_checkpoint,
    write_iteration_checkpoint,
    write_iteration_timing_report,
    write_workflow_manifest,
)
from .fjw_workflow_driver import FJWWorkflowDriverRequest, FJWWorkflowDriverResult
from .fjw_workflow_loaders import (
    DEFAULT_ABAQUS_INPUTS_PATH,
    DEFAULT_END1_TEMPLATE_PATH,
    DEFAULT_INPUT_INVENTORY_PATH,
    load_fjw_workflow_state,
)
from .fjw_workflow_optimizer import build_initial_mma_state
from .fjw_workflow_runner import run_fjw_abaqus_workflow_iteration, run_fjw_sfepy_workflow_iteration
from .fjw_workflow_solver_adapters import FJWAbaqusWorkflowSolverConfig
from .fjw_workflow_sfepy_solver_adapters import FJWSfePyWorkflowSolverConfig
from .fjw_workflow_timing import FJWHeartbeatWriter, FJWTimingRecorder
from .fjw_direct_solver import FJWDirectSolverConfig


FJWBackendName = Literal["abaqus", "sfepy"]
FJWOptimizationMode = Literal["three-force"]


@dataclass(frozen=True, slots=True)
class FJWOptimizationConfig:
    reference_dir: Path = Path("references/fjw_work")
    abaqus_inputs_path: Path = DEFAULT_ABAQUS_INPUTS_PATH
    input_inventory_path: Path = DEFAULT_INPUT_INVENTORY_PATH
    end1_template_path: Path = DEFAULT_END1_TEMPLATE_PATH
    backend: FJWBackendName = "sfepy"
    mode: FJWOptimizationMode = "three-force"
    max_iterations: int = 1
    delta_tol: float = 1.0e-4
    num_time_steps: int = 3
    run_directory: Path = Path("runs/fjw_optimize")
    resume: bool = False
    checkpoint_every: int = 1
    abaqus_executable: str = "abaqus"
    cpus: int = 8
    real_run: bool = False
    sfepy_linear_solver: str | None = None
    runtime_profile: str = "wuyinyun"
    case_parallelism: int | None = None
    solver_threads: int | None = None
    enable_sfepy_setup_cache: bool = True
    enable_heartbeat: bool = True
    heartbeat_interval_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.backend not in ("abaqus", "sfepy"):
            raise ValueError("backend must be 'abaqus' or 'sfepy'.")
        if self.mode != "three-force":
            raise ValueError("The production FJW optimizer currently implements the archived three-force workflow.")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if self.delta_tol <= 0.0:
            raise ValueError("delta_tol must be positive.")
        if self.num_time_steps <= 0:
            raise ValueError("num_time_steps must be positive.")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive.")
        if self.heartbeat_interval_seconds <= 0.0:
            raise ValueError("heartbeat_interval_seconds must be positive.")

        runtime_config = get_fjw_runtime_config(self.runtime_profile)
        sfepy_linear_solver = self.sfepy_linear_solver or runtime_config.sfepy_linear_solver
        case_parallelism = runtime_config.case_parallelism if self.case_parallelism is None else int(self.case_parallelism)
        solver_threads = runtime_config.solver_threads if self.solver_threads is None else int(self.solver_threads)
        if case_parallelism <= 0:
            raise ValueError("case_parallelism must be positive.")
        if solver_threads <= 0:
            raise ValueError("solver_threads must be positive.")

        object.__setattr__(self, "reference_dir", Path(self.reference_dir))
        object.__setattr__(self, "abaqus_inputs_path", Path(self.abaqus_inputs_path))
        object.__setattr__(self, "input_inventory_path", Path(self.input_inventory_path))
        object.__setattr__(self, "end1_template_path", Path(self.end1_template_path))
        object.__setattr__(self, "run_directory", Path(self.run_directory))
        object.__setattr__(self, "sfepy_linear_solver", sfepy_linear_solver)
        object.__setattr__(self, "case_parallelism", case_parallelism)
        object.__setattr__(self, "solver_threads", solver_threads)


@dataclass(frozen=True, slots=True)
class FJWOptimizationRunResult:
    config: FJWOptimizationConfig
    iterations: tuple[FJWWorkflowDriverResult, ...]
    final_design: np.ndarray
    final_delta: float
    stopped_reason: str
    manifest_path: Path


def run_fjw_optimization(config: FJWOptimizationConfig) -> FJWOptimizationRunResult:
    if config.backend == "abaqus" and not config.real_run:
        raise RuntimeError(
            "Abaqus optimization requires --real-run. Dry-run Abaqus jobs do not produce "
            "forward or adjoint displacement vectors and cannot be used as optimization input."
        )

    config.run_directory.mkdir(parents=True, exist_ok=True)
    numeric_thread_env = configure_numeric_runtime_threads(config.solver_threads or 1)
    heartbeat_writer = None
    if config.enable_heartbeat:
        heartbeat_writer = FJWHeartbeatWriter(
            status_path=config.run_directory / "runtime_status.json",
            events_path=config.run_directory / "runtime_events.jsonl",
            interval_seconds=config.heartbeat_interval_seconds,
            run_metadata={
                "backend": config.backend,
                "mode": config.mode,
                "runtime_profile": config.runtime_profile,
                "sfepy_linear_solver": config.sfepy_linear_solver,
                "case_parallelism": config.case_parallelism,
                "solver_threads": config.solver_threads,
                "run_directory": str(config.run_directory),
            },
        )
        heartbeat_writer.mark_run_state("loading_workflow")
    try:
        workflow_state = load_fjw_workflow_state(
            reference_dir=config.reference_dir,
            abaqus_inputs_path=config.abaqus_inputs_path,
            input_inventory_path=config.input_inventory_path,
            end1_template_path=config.end1_template_path,
            initial_design_mode="three_load",
        )
        if config.resume:
            resume_state = load_resume_state(config.run_directory)
            design = resume_state.design.copy()
            mma_state = resume_state.mma_state
            start_iteration = resume_state.iteration_index
        else:
            design = workflow_state.initial_state.design_cage.copy()
            mma_state = build_initial_mma_state(design)
            start_iteration = 0
            write_initial_checkpoint(
                run_directory=config.run_directory,
                workflow_state=workflow_state,
                design=design,
                mma_state=mma_state,
            )

        manifest_path = write_workflow_manifest(
            run_directory=config.run_directory,
            payload={
                "backend": config.backend,
                "mode": config.mode,
                "max_iterations": config.max_iterations,
                "delta_tol": config.delta_tol,
                "num_time_steps": config.num_time_steps,
                "runtime_profile": config.runtime_profile,
                "sfepy_linear_solver": config.sfepy_linear_solver,
                "case_parallelism": config.case_parallelism,
                "solver_threads": config.solver_threads,
                "numeric_thread_environment": numeric_thread_env,
                "enable_sfepy_setup_cache": config.enable_sfepy_setup_cache,
                "enable_heartbeat": config.enable_heartbeat,
                "heartbeat_interval_seconds": config.heartbeat_interval_seconds,
                "resume": config.resume,
                "start_iteration": start_iteration,
                "run_directory": str(config.run_directory),
                "reference_dir": str(config.reference_dir),
                "runtime_artifact_policy": "Large Abaqus/SfePy runtime artifacts stay under runs/ and are not committed by default.",
            },
        )

        iterations: list[FJWWorkflowDriverResult] = []
        final_delta = float("inf")
        stopped_reason = "max_iterations"
        current_design = design
        current_mma_state = mma_state
        if heartbeat_writer is not None:
            heartbeat_writer.mark_run_state(
                "optimizing",
                start_iteration=start_iteration,
                max_iterations=config.max_iterations,
            )

        for _ in range(config.max_iterations):
            iteration_index = current_mma_state.iteration + 1
            if heartbeat_writer is not None:
                heartbeat_writer.mark_run_state("iteration", iteration_index=iteration_index)
            timing_recorder = FJWTimingRecorder(
                root_name=f"iteration:{iteration_index:03d}",
                heartbeat_writer=heartbeat_writer,
            )
            driver_request = FJWWorkflowDriverRequest(
                workflow_state=workflow_state,
                initial_design_mode="three_load",
                design=current_design,
                mma_state=current_mma_state,
                num_time_steps=config.num_time_steps,
                case_parallelism=config.case_parallelism or 1,
                timing_recorder=timing_recorder,
            )
            with timing_recorder.measure("outer_iteration", iteration_index=iteration_index):
                result = _run_one_iteration(config, driver_request)
            result.metadata["timing"] = timing_recorder.as_jsonable()
            iterations.append(result)

            next_design = result.iteration_state.next_design
            if next_design is None:
                raise RuntimeError("FJW optimizer did not return next_design.")
            final_delta = _iteration_delta(result)
            if result.iteration_state.iteration_index % config.checkpoint_every == 0:
                with timing_recorder.measure("checkpoint_write", iteration_index=result.iteration_state.iteration_index):
                    write_iteration_checkpoint(
                        run_directory=config.run_directory,
                        result=result,
                        delta=final_delta,
                    )
                result.metadata["timing"] = timing_recorder.as_jsonable()
                write_iteration_timing_report(
                    run_directory=config.run_directory,
                    iteration_index=result.iteration_state.iteration_index,
                    timing=result.metadata["timing"],
                )

            current_design = next_design.copy()
            current_mma_state = result.iteration_state.mma_state
            if final_delta <= config.delta_tol:
                stopped_reason = "delta_tol"
                break

        if heartbeat_writer is not None:
            heartbeat_writer.mark_run_state(
                "completed",
                final_delta=float(final_delta),
                stopped_reason=stopped_reason,
            )
        return FJWOptimizationRunResult(
            config=config,
            iterations=tuple(iterations),
            final_design=current_design,
            final_delta=float(final_delta),
            stopped_reason=stopped_reason,
            manifest_path=manifest_path,
        )
    except BaseException as exc:
        if heartbeat_writer is not None:
            heartbeat_writer.mark_run_state("failed", error=f"{type(exc).__name__}: {exc}"[:500])
        raise
    finally:
        if heartbeat_writer is not None:
            heartbeat_writer.close()


def _run_one_iteration(
    config: FJWOptimizationConfig,
    driver_request: FJWWorkflowDriverRequest,
) -> FJWWorkflowDriverResult:
    if config.backend == "abaqus":
        return run_fjw_abaqus_workflow_iteration(
            driver_request=driver_request,
            solver_config=FJWAbaqusWorkflowSolverConfig(
                run_directory=config.run_directory / "abaqus_jobs",
                abaqus_executable=config.abaqus_executable,
                cpus=config.cpus,
                dry_run=False,
                job_prefix=f"iter_{driver_request.mma_state.iteration + 1:03d}",
            ),
        )
    return run_fjw_sfepy_workflow_iteration(
        driver_request=driver_request,
        solver_config=FJWSfePyWorkflowSolverConfig(
            direct_solver_config=FJWDirectSolverConfig(
                linear_solver_kind=config.sfepy_linear_solver,
            ),
            enable_setup_cache=config.enable_sfepy_setup_cache,
        ),
    )


def _iteration_delta(result: FJWWorkflowDriverResult) -> float:
    diagnostics = (
        {} if result.iteration_state.optimizer_step is None else result.iteration_state.optimizer_step.diagnostics
    )
    if "delta" in diagnostics:
        return float(diagnostics["delta"])
    next_design = result.iteration_state.next_design
    if next_design is None:
        return float("inf")
    return float(np.mean(np.abs(next_design - result.iteration_state.design)))


__all__ = [
    "FJWOptimizationConfig",
    "FJWOptimizationRunResult",
    "run_fjw_optimization",
]
