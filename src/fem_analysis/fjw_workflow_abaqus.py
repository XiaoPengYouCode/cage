from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AbaqusCommand:
    command: list[str]
    workdir: Path
    lock_file: Path


@dataclass(frozen=True)
class AbaqusRunResult:
    command: list[str]
    return_code: int
    elapsed_seconds: float
    lock_wait_seconds: float


def build_standard_job_command(
    *,
    job_name: str,
    cpus: int,
    workdir: Path,
    abaqus_executable: str = "abaqus",
) -> AbaqusCommand:
    workdir = Path(workdir)
    return AbaqusCommand(
        command=[abaqus_executable, f"job={job_name}", f"cpus={cpus}"],
        workdir=workdir,
        lock_file=workdir / f"{job_name}.lck",
    )


def build_odb_export_command(
    *,
    script_path: Path,
    workdir: Path,
    abaqus_executable: str = "abaqus",
) -> list[str]:
    return [abaqus_executable, "cae", f"noGUI={Path(script_path)}"]


def remove_stale_lock(lock_file: Path) -> bool:
    lock_file = Path(lock_file)
    if lock_file.exists():
        lock_file.unlink()
        return True
    return False


def run_abaqus_command(
    command: AbaqusCommand,
    *,
    poll_interval_seconds: float = 0.5,
    startup_delay_seconds: float = 5.0,
) -> AbaqusRunResult:
    started_at = time.perf_counter()
    subprocess.run(
        command.command,
        cwd=command.workdir,
        check=True,
    )
    if startup_delay_seconds > 0.0:
        time.sleep(startup_delay_seconds)

    lock_started_at = time.perf_counter()
    while command.lock_file.exists():
        time.sleep(poll_interval_seconds)

    finished_at = time.perf_counter()
    return AbaqusRunResult(
        command=command.command,
        return_code=0,
        elapsed_seconds=finished_at - started_at,
        lock_wait_seconds=finished_at - lock_started_at,
    )


def run_odb_export(
    *,
    script_path: Path,
    workdir: Path,
    abaqus_executable: str = "abaqus",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        build_odb_export_command(
            script_path=script_path,
            workdir=workdir,
            abaqus_executable=abaqus_executable,
        ),
        cwd=workdir,
        check=True,
        text=True,
        capture_output=True,
    )
