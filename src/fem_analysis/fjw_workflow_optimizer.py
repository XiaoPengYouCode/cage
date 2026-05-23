from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .fjw_mma import mmasub


@dataclass(frozen=True, slots=True)
class FJWOptimizationTerms:
    objective: float
    objective_gradient: np.ndarray
    constraints: np.ndarray
    constraint_gradients: np.ndarray
    constraint_names: tuple[str, ...] = ("g2",)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        objective_gradient = np.asarray(self.objective_gradient, dtype=np.float64).reshape(-1)
        constraints = np.asarray(self.constraints, dtype=np.float64).reshape(-1)
        constraint_gradients = np.asarray(self.constraint_gradients, dtype=np.float64)

        if constraint_gradients.ndim != 2:
            raise ValueError("constraint_gradients must have shape (m, n).")
        if constraint_gradients.shape[1] != objective_gradient.size:
            raise ValueError("constraint_gradients column count must match objective_gradient size.")
        if constraints.size != constraint_gradients.shape[0]:
            raise ValueError("constraints size must match constraint_gradients row count.")
        if len(self.constraint_names) != constraints.size:
            raise ValueError("constraint_names size must match constraints size.")

        object.__setattr__(self, "objective_gradient", objective_gradient)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "constraint_gradients", constraint_gradients)

    @property
    def design_size(self) -> int:
        return int(self.objective_gradient.size)

    @property
    def constraint_count(self) -> int:
        return int(self.constraints.size)


@dataclass(frozen=True, slots=True)
class FJWMMAState:
    iteration: int
    xold1: np.ndarray
    xold2: np.ndarray
    xmin: np.ndarray
    xmax: np.ndarray
    low: np.ndarray
    up: np.ndarray
    a0: float = 1.0
    a: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    c: np.ndarray = field(default_factory=lambda: np.array([1000.0], dtype=np.float64))
    d: np.ndarray = field(default_factory=lambda: np.array([1.0], dtype=np.float64))

    def __post_init__(self) -> None:
        xold1 = np.asarray(self.xold1, dtype=np.float64).reshape(-1)
        xold2 = np.asarray(self.xold2, dtype=np.float64).reshape(-1)
        xmin = np.asarray(self.xmin, dtype=np.float64).reshape(-1)
        xmax = np.asarray(self.xmax, dtype=np.float64).reshape(-1)
        low = np.asarray(self.low, dtype=np.float64).reshape(-1)
        up = np.asarray(self.up, dtype=np.float64).reshape(-1)
        a = np.asarray(self.a, dtype=np.float64).reshape(-1)
        c = np.asarray(self.c, dtype=np.float64).reshape(-1)
        d = np.asarray(self.d, dtype=np.float64).reshape(-1)

        design_size = xold1.size
        for name, value in (
            ("xold2", xold2),
            ("xmin", xmin),
            ("xmax", xmax),
            ("low", low),
            ("up", up),
        ):
            if value.size != design_size:
                raise ValueError(f"{name} size must match xold1 size.")

        object.__setattr__(self, "xold1", xold1)
        object.__setattr__(self, "xold2", xold2)
        object.__setattr__(self, "xmin", xmin)
        object.__setattr__(self, "xmax", xmax)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "up", up)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "d", d)


@dataclass(frozen=True, slots=True)
class FJWOptimizerStepResult:
    design: np.ndarray
    state: FJWMMAState
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        design = np.asarray(self.design, dtype=np.float64).reshape(-1)
        if design.size != self.state.xold1.size:
            raise ValueError("design size must match MMA state size.")
        object.__setattr__(self, "design", design)


class FJWOptimizer(Protocol):
    def step(
        self,
        design: np.ndarray,
        terms: FJWOptimizationTerms,
        state: FJWMMAState,
    ) -> FJWOptimizerStepResult:
        ...


def build_initial_mma_state(
    design: np.ndarray,
    *,
    xmin: float | np.ndarray = 0.001,
    xmax: float | np.ndarray = 1.0,
    constraint_count: int = 1,
    a0: float = 1.0,
    c_value: float = 1000.0,
    d_value: float = 1.0,
) -> FJWMMAState:
    design = np.asarray(design, dtype=np.float64).reshape(-1)
    if design.size == 0:
        raise ValueError("design must not be empty.")

    xmin_array = np.broadcast_to(np.asarray(xmin, dtype=np.float64), design.shape).copy()
    xmax_array = np.broadcast_to(np.asarray(xmax, dtype=np.float64), design.shape).copy()
    if np.any(xmin_array > xmax_array):
        raise ValueError("xmin must be <= xmax elementwise.")

    return FJWMMAState(
        iteration=0,
        xold1=design.copy(),
        xold2=design.copy(),
        xmin=xmin_array,
        xmax=xmax_array,
        low=np.zeros_like(design),
        up=np.zeros_like(design),
        a0=float(a0),
        a=np.zeros(int(constraint_count), dtype=np.float64),
        c=np.full(int(constraint_count), float(c_value), dtype=np.float64),
        d=np.full(int(constraint_count), float(d_value), dtype=np.float64),
    )


class FJWMMAOptimizer:
    """Production MMA optimizer matching the archived `mmasub.m` call shape."""

    def step(
        self,
        design: np.ndarray,
        terms: FJWOptimizationTerms,
        state: FJWMMAState,
    ) -> FJWOptimizerStepResult:
        design = np.asarray(design, dtype=np.float64).reshape(-1)
        if design.size != state.xold1.size:
            raise ValueError("design size must match MMA state size.")
        if terms.design_size != design.size:
            raise ValueError("terms design size must match design size.")

        if terms.constraint_count != state.a.size:
            raise ValueError("MMA state constraint count must match optimization terms.")

        mma_result = mmasub(
            m=terms.constraint_count,
            n=design.size,
            iteration=state.iteration + 1,
            xval=design,
            xmin=state.xmin,
            xmax=state.xmax,
            xold1=state.xold1,
            xold2=state.xold2,
            f0val=terms.objective,
            df0dx=terms.objective_gradient,
            fval=terms.constraints,
            dfdx=terms.constraint_gradients,
            low=state.low,
            upp=state.up,
            a0=state.a0,
            a=state.a,
            c=state.c,
            d=state.d,
        )
        xmma = np.clip(mma_result.xmma, state.xmin, state.xmax)
        delta = float(np.mean(np.abs(xmma - design)))
        next_state = FJWMMAState(
            iteration=state.iteration + 1,
            xold1=design.copy(),
            xold2=state.xold1.copy(),
            xmin=state.xmin.copy(),
            xmax=state.xmax.copy(),
            low=mma_result.low.copy(),
            up=mma_result.upp.copy(),
            a0=state.a0,
            a=state.a.copy(),
            c=state.c.copy(),
            d=state.d.copy(),
        )
        return FJWOptimizerStepResult(
            design=xmma,
            state=next_state,
            diagnostics={
                "solver": "mmasub_subsolv",
                "status": "updated",
                "objective": float(terms.objective),
                "constraints": terms.constraints.copy(),
                "delta": delta,
                "mma_iteration": int(state.iteration + 1),
                "ymma": mma_result.ymma.copy(),
                "zmma": float(mma_result.zmma),
                "lam": mma_result.lam.copy(),
                "low": mma_result.low.copy(),
                "upp": mma_result.upp.copy(),
                "alfa": mma_result.alfa.copy(),
                "beta": mma_result.beta.copy(),
            },
        )


__all__ = [
    "FJWMMAState",
    "FJWMMAOptimizer",
    "FJWOptimizationTerms",
    "FJWOptimizer",
    "FJWOptimizerStepResult",
    "build_initial_mma_state",
]
