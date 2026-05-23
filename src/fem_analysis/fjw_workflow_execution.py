from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .fjw_workflow_abaqus import (
    build_standard_job_command,
    run_abaqus_command,
    run_odb_export,
)
from .fjw_workflow_artifacts import FJWJobArtifacts, build_job_artifacts
from .fjw_workflow_odb import write_abaqus_odb_export_script
from .fjw_workflow_results import load_abaqus_u1_result
from .fjw_workflow_vectors import (
    FJWElementDisplacementVectorCache,
    build_element_displacement_cache,
    save_element_displacement_cache,
)


@dataclass(frozen=True, slots=True)
class FJWExecutionResult:
    artifacts: FJWJobArtifacts
    dry_run: bool
    vector_cache: FJWElementDisplacementVectorCache | None = None
    abaqus_elapsed_seconds: float | None = None


def _write_artifact_metadata(
    path: Path,
    *,
    artifacts: FJWJobArtifacts,
    dry_run: bool,
    abaqus_elapsed_seconds: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "artifacts": artifacts.as_jsonable(),
                "dry_run": dry_run,
                "abaqus_elapsed_seconds": abaqus_elapsed_seconds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _check_real_run_environment(
    *,
    artifacts: FJWJobArtifacts,
    abaqus_executable: str,
) -> None:
    resolved = shutil.which(abaqus_executable)
    if resolved is None:
        raise RuntimeError(
            f"Abaqus executable {abaqus_executable!r} was not found on PATH. "
            "Install Abaqus or pass --abaqus-executable to a valid executable."
        )
    artifacts.run_directory.mkdir(parents=True, exist_ok=True)
    if not artifacts.inp_path.exists():
        raise RuntimeError(f"Cannot run Abaqus job {artifacts.job_name!r}: missing input {artifacts.inp_path}.")
    probe = artifacts.run_directory / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
    finally:
        if probe.exists():
            probe.unlink()


def execute_job_and_collect(
    *,
    run_directory: Path,
    job_name: str,
    workflow_or_mesh,
    abaqus_executable: str = "abaqus",
    cpus: int = 8,
    dry_run: bool = True,
) -> FJWExecutionResult:
    artifacts = build_job_artifacts(run_directory, job_name)
    artifacts.run_directory.mkdir(parents=True, exist_ok=True)
    write_abaqus_odb_export_script(
        artifacts.odb_export_script_path,
        odb_filename=artifacts.odb_path.name,
        output_filename=artifacts.u1_path.name,
        field_name="U",
        step_name="Load",
    )

    if dry_run:
        _write_artifact_metadata(
            artifacts.metadata_path,
            artifacts=artifacts,
            dry_run=True,
            abaqus_elapsed_seconds=None,
        )
        return FJWExecutionResult(artifacts=artifacts, dry_run=True)

    _check_real_run_environment(
        artifacts=artifacts,
        abaqus_executable=abaqus_executable,
    )
    command = build_standard_job_command(
        job_name=job_name,
        cpus=cpus,
        workdir=artifacts.run_directory,
        abaqus_executable=abaqus_executable,
    )
    run_result = run_abaqus_command(command)
    if not artifacts.odb_path.exists():
        raise RuntimeError(
            f"Abaqus job {job_name!r} finished but did not produce {artifacts.odb_path}. "
            f"Debug files: log={artifacts.log_path}, dat={artifacts.dat_path}, "
            f"sta={artifacts.sta_path}, msg={artifacts.msg_path}."
        )
    run_odb_export(
        script_path=artifacts.odb_export_script_path,
        workdir=artifacts.run_directory,
        abaqus_executable=abaqus_executable,
    )
    if not artifacts.u1_path.exists():
        raise RuntimeError(
            f"ODB export for job {job_name!r} did not produce {artifacts.u1_path}. "
            f"Debug files: odb={artifacts.odb_path}, log={artifacts.log_path}, "
            f"dat={artifacts.dat_path}, sta={artifacts.sta_path}, msg={artifacts.msg_path}."
        )

    u1_result = load_abaqus_u1_result(artifacts.u1_path)
    vector_cache = build_element_displacement_cache(
        u1_result,
        workflow_or_mesh,
        cache_name=job_name,
        source_result_path=artifacts.u1_path,
        strict=False,
    )
    save_element_displacement_cache(artifacts.vector_cache_path, vector_cache)
    _write_artifact_metadata(
        artifacts.metadata_path,
        artifacts=artifacts,
        dry_run=False,
        abaqus_elapsed_seconds=run_result.elapsed_seconds,
    )
    return FJWExecutionResult(
        artifacts=artifacts,
        dry_run=False,
        vector_cache=vector_cache,
        abaqus_elapsed_seconds=run_result.elapsed_seconds,
    )


__all__ = [
    "FJWExecutionResult",
    "execute_job_and_collect",
]
