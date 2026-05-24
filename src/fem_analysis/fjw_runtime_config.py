from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


FJWRuntimeProfileName = Literal["local", "wuyinyun"]


@dataclass(frozen=True, slots=True)
class FJWRuntimeConfig:
    name: FJWRuntimeProfileName
    sfepy_linear_solver: str
    case_parallelism: int
    solver_threads: int
    memory_gb: int


_PROFILES: dict[str, FJWRuntimeConfig] = {
    "local": FJWRuntimeConfig(
        name="local",
        sfepy_linear_solver="scipy_iterative",
        case_parallelism=1,
        solver_threads=1,
        memory_gb=0,
    ),
    "wuyinyun": FJWRuntimeConfig(
        name="wuyinyun",
        sfepy_linear_solver="petsc_mumps",
        case_parallelism=2,
        solver_threads=12,
        memory_gb=48,
    ),
}


def get_fjw_runtime_config(name: str) -> FJWRuntimeConfig:
    try:
        return _PROFILES[str(name)]
    except KeyError as exc:
        available = ", ".join(sorted(_PROFILES))
        raise ValueError(f"Unknown FJW runtime profile {name!r}. Available: {available}.") from exc


def fjw_runtime_profile_names() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))


def configure_numeric_runtime_threads(thread_count: int) -> dict[str, str]:
    if int(thread_count) <= 0:
        raise ValueError("thread_count must be positive.")
    value = str(int(thread_count))
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    applied: dict[str, str] = {}
    for key in keys:
        os.environ.setdefault(key, value)
        applied[key] = os.environ[key]
    return applied


__all__ = [
    "FJWRuntimeConfig",
    "configure_numeric_runtime_threads",
    "fjw_runtime_profile_names",
    "get_fjw_runtime_config",
]
