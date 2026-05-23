from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


FJWSfePyLinearSolverKind = Literal["scipy_direct", "scipy_iterative", "petsc_mumps"]


@dataclass(frozen=True, slots=True)
class FJWSfePyLinearSolverProfile:
    kind: FJWSfePyLinearSolverKind = "scipy_iterative"
    options: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("scipy_direct", "scipy_iterative", "petsc_mumps"):
            raise ValueError("SfePy linear solver kind must be scipy_direct, scipy_iterative, or petsc_mumps.")


def scipy_iterative_options() -> dict[str, object]:
    return {
        "method": "cg",
        "eps_r": 1.0e-8,
        "i_max": 10_000,
    }


def petsc_mumps_options() -> dict[str, object]:
    return {
        "method": "preonly",
        "precond": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }


__all__ = [
    "FJWSfePyLinearSolverKind",
    "FJWSfePyLinearSolverProfile",
    "petsc_mumps_options",
    "scipy_iterative_options",
]
