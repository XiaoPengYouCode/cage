from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FJWJobArtifacts:
    job_name: str
    run_directory: Path
    inp_path: Path
    odb_path: Path
    u1_path: Path
    odb_export_script_path: Path
    vector_cache_path: Path
    metadata_path: Path
    lock_path: Path
    log_path: Path
    dat_path: Path
    sta_path: Path
    msg_path: Path

    def as_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


def build_job_artifacts(run_directory: Path, job_name: str) -> FJWJobArtifacts:
    job_directory = Path(run_directory) / job_name
    return FJWJobArtifacts(
        job_name=job_name,
        run_directory=job_directory,
        inp_path=job_directory / f"{job_name}.inp",
        odb_path=job_directory / f"{job_name}.odb",
        u1_path=job_directory / f"{job_name}_U1.txt",
        odb_export_script_path=job_directory / f"{job_name}_odb_export.py",
        vector_cache_path=job_directory / f"{job_name}_U1_vectors.npz",
        metadata_path=job_directory / f"{job_name}_artifacts.json",
        lock_path=job_directory / f"{job_name}.lck",
        log_path=job_directory / f"{job_name}.log",
        dat_path=job_directory / f"{job_name}.dat",
        sta_path=job_directory / f"{job_name}.sta",
        msg_path=job_directory / f"{job_name}.msg",
    )


__all__ = [
    "FJWJobArtifacts",
    "build_job_artifacts",
]
