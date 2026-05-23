from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MIN_BONE_DENSITY = 0.001
_BONE_DELTA_K = 0.000485
_BONE_DELTA_SS = 0.1
_BONE_DELTA_SAT = 7e-3
_BONE_DELTA_ALPHA = 2.0
_BONE_DELTA_FORMATION_THRESHOLD = 3e-2
_BONE_DELTA_OVERSHOOT_SATURATION = -0.05
_BONE_DELTA_OVERSHOOT_MIDPOINT = 0.1
_BONE_DELTA_THETA1 = _BONE_DELTA_K * (1.0 - _BONE_DELTA_SS)
_BONE_DELTA_THETA2 = _BONE_DELTA_K * (1.0 + _BONE_DELTA_SS)
_BONE_DELTA_C1 = _BONE_DELTA_SAT / _BONE_DELTA_THETA1
_BONE_DELTA_C2 = (
    -4.0
    * _BONE_DELTA_SAT
    * _BONE_DELTA_ALPHA
    / (_BONE_DELTA_THETA2 - _BONE_DELTA_FORMATION_THRESHOLD) ** 2
)
_BONE_DELTA_C3 = abs(_BONE_DELTA_OVERSHOOT_SATURATION) / (
    _BONE_DELTA_FORMATION_THRESHOLD - _BONE_DELTA_OVERSHOOT_MIDPOINT
) ** 2


@dataclass(frozen=True, slots=True)
class FJWBoneBiologyStep:
    objective_modulus: np.ndarray
    stimulus: np.ndarray
    density_delta: np.ndarray
    next_density: np.ndarray
    bo_sum: float


def _as_flat_float64(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def bone_delta(stimulus: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    stimulus_array = _as_flat_float64(stimulus)
    density_delta = np.zeros_like(stimulus_array)

    low_mask = stimulus_array < _BONE_DELTA_THETA1
    formation_mask = (
        (stimulus_array > _BONE_DELTA_THETA2)
        & (stimulus_array <= _BONE_DELTA_FORMATION_THRESHOLD)
    )
    overshoot_mask = (
        (stimulus_array > _BONE_DELTA_FORMATION_THRESHOLD)
        & (stimulus_array <= _BONE_DELTA_OVERSHOOT_MIDPOINT)
    )
    plateau_mask = stimulus_array > _BONE_DELTA_OVERSHOOT_MIDPOINT

    density_delta[low_mask] = _BONE_DELTA_C1 * (stimulus_array[low_mask] - _BONE_DELTA_THETA1)
    density_delta[formation_mask] = (
        _BONE_DELTA_C2
        * (
            stimulus_array[formation_mask]
            - (_BONE_DELTA_THETA2 + _BONE_DELTA_FORMATION_THRESHOLD) / 2.0
        )
        ** 2
        + _BONE_DELTA_SAT * _BONE_DELTA_ALPHA
    )
    density_delta[overshoot_mask] = (
        _BONE_DELTA_C3
        * (stimulus_array[overshoot_mask] - _BONE_DELTA_OVERSHOOT_MIDPOINT) ** 2
        + _BONE_DELTA_OVERSHOOT_SATURATION
    )
    density_delta[plateau_mask] = _BONE_DELTA_OVERSHOOT_SATURATION
    return density_delta


def d_bone_delta(stimulus: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    stimulus_array = _as_flat_float64(stimulus)
    derivative = np.zeros_like(stimulus_array)

    low_mask = stimulus_array < _BONE_DELTA_THETA1
    formation_mask = (
        (stimulus_array > _BONE_DELTA_THETA2)
        & (stimulus_array <= _BONE_DELTA_FORMATION_THRESHOLD)
    )
    overshoot_mask = (
        (stimulus_array > _BONE_DELTA_FORMATION_THRESHOLD)
        & (stimulus_array <= _BONE_DELTA_OVERSHOOT_MIDPOINT)
    )

    derivative[low_mask] = _BONE_DELTA_C1
    derivative[formation_mask] = 2.0 * _BONE_DELTA_C2 * (
        stimulus_array[formation_mask]
        - (_BONE_DELTA_THETA2 + _BONE_DELTA_FORMATION_THRESHOLD) / 2.0
    )
    derivative[overshoot_mask] = 2.0 * _BONE_DELTA_C3 * (
        stimulus_array[overshoot_mask] - _BONE_DELTA_OVERSHOOT_MIDPOINT
    )
    return derivative


def bone_objective_modulus(
    bone_density: np.ndarray | list[float] | tuple[float, ...],
    *,
    bone_modulus_0: float,
    bone_modulus_min: float,
    bone_density_upper_bound: float,
) -> np.ndarray:
    density = np.clip(
        _as_flat_float64(bone_density),
        MIN_BONE_DENSITY,
        float(bone_density_upper_bound),
    )
    modulus = bone_modulus_min + bone_modulus_0 * np.power(
        density / bone_density_upper_bound,
        3,
    )
    return np.minimum(modulus, bone_modulus_0)


def compute_bone_stimulus(
    element_quadratic_terms: np.ndarray | list[float] | tuple[float, ...],
    bone_density: np.ndarray | list[float] | tuple[float, ...],
    *,
    bone_modulus_0: float,
    bone_modulus_min: float,
    bone_density_upper_bound: float,
) -> tuple[np.ndarray, np.ndarray]:
    quadratic_terms = _as_flat_float64(element_quadratic_terms)
    density = np.clip(
        _as_flat_float64(bone_density),
        MIN_BONE_DENSITY,
        float(bone_density_upper_bound),
    )
    if quadratic_terms.shape != density.shape:
        raise ValueError(
            "element_quadratic_terms and bone_density must have the same flattened shape."
        )

    modulus = bone_objective_modulus(
        density,
        bone_modulus_0=bone_modulus_0,
        bone_modulus_min=bone_modulus_min,
        bone_density_upper_bound=bone_density_upper_bound,
    )
    stimulus = quadratic_terms * modulus / density
    return stimulus, modulus


def advance_bone_density(
    element_quadratic_terms: np.ndarray | list[float] | tuple[float, ...],
    bone_density: np.ndarray | list[float] | tuple[float, ...],
    *,
    bone_modulus_0: float,
    bone_modulus_min: float,
    bone_density_upper_bound: float,
    time_step_dt: float,
) -> FJWBoneBiologyStep:
    density = np.clip(
        _as_flat_float64(bone_density),
        MIN_BONE_DENSITY,
        float(bone_density_upper_bound),
    )
    stimulus, modulus = compute_bone_stimulus(
        element_quadratic_terms,
        density,
        bone_modulus_0=bone_modulus_0,
        bone_modulus_min=bone_modulus_min,
        bone_density_upper_bound=bone_density_upper_bound,
    )
    density_delta = bone_delta(stimulus)
    next_density = np.clip(
        density + density_delta * float(time_step_dt),
        MIN_BONE_DENSITY,
        float(bone_density_upper_bound),
    )
    return FJWBoneBiologyStep(
        objective_modulus=modulus,
        stimulus=stimulus,
        density_delta=density_delta,
        next_density=next_density,
        bo_sum=float(np.sum(next_density, dtype=np.float64)),
    )


__all__ = [
    "FJWBoneBiologyStep",
    "MIN_BONE_DENSITY",
    "advance_bone_density",
    "bone_delta",
    "bone_objective_modulus",
    "compute_bone_stimulus",
    "d_bone_delta",
]
