from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from topopt_sampling.probability import (
    density_to_probability_intensity,
    load_density_input,
    sample_seed_points,
    save_seed_mapping_result,
)


@dataclass(frozen=True)
class SeedMappingResult:
    input_npz: Path
    output_npz: Path
    seed_points: np.ndarray
    probability: np.ndarray


def map_density_to_seed_mapping(
    input_npz: Path,
    output_npz: Path,
    *,
    num_seeds: int = 2_000,
    gamma: float = 1.0,
    rng_seed: int = 42,
    progress: bool = False,
) -> SeedMappingResult:
    density_result = load_density_input(input_npz)
    density_milli = density_result["density_milli"]
    density = density_milli.astype(np.float32) / 1000.0
    probability = density_to_probability_intensity(
        density=density,
        gamma=gamma,
    )
    seed_points = sample_seed_points(
        density_milli=density_milli,
        num_seeds=num_seeds,
        gamma=gamma,
        rng_seed=rng_seed,
        progress=progress,
    )
    save_seed_mapping_result(
        output_npz=output_npz,
        seed_points=seed_points,
        original_shape=density_milli.shape,
        gamma=gamma,
        num_seeds=num_seeds,
        input_npz=input_npz,
    )
    return SeedMappingResult(
        input_npz=input_npz,
        output_npz=output_npz,
        seed_points=seed_points,
        probability=probability,
    )
