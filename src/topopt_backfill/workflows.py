from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from topopt_backfill.probability import (
    density_to_probability_intensity,
    downsample_density,
    load_npz,
    sample_seed_points,
    save_seed_mapping_result,
)
from topopt_backfill.selection import (
    build_candidate_pool,
    farthest_point_sampling,
    normalize_seed_points,
)
from topopt_backfill.templates import build_backfilled_templates, save_backfilled_templates


@dataclass(frozen=True)
class SeedMappingResult:
    input_npz: Path
    output_npz: Path
    seed_points: np.ndarray
    display_density: np.ndarray
    display_probability: np.ndarray
    display_steps: tuple[int, int, int]


@dataclass(frozen=True)
class BackfillResult:
    input_npz: Path
    output_npz: Path
    representative_seeds: np.ndarray
    candidate_points: np.ndarray
    block_density: np.ndarray
    block_active: np.ndarray
    template_ids: np.ndarray
    template_seed_counts: np.ndarray


def map_density_to_seed_mapping(
    input_npz: Path,
    output_npz: Path,
    *,
    num_seeds: int = 100_000,
    gamma: float = 1.8,
    chunk_depth: int = 8,
    max_display_size: int = 84,
    rng_seed: int = 42,
    progress: bool = False,
) -> SeedMappingResult:
    result = load_npz(input_npz)
    density_milli = result["density_milli"]
    _, display_density, display_steps = downsample_density(
        density_milli=density_milli,
        max_display_size=max_display_size,
    )
    display_probability = density_to_probability_intensity(
        density=display_density,
        gamma=gamma,
    )
    seed_points = sample_seed_points(
        density_milli=density_milli,
        num_seeds=num_seeds,
        gamma=gamma,
        chunk_depth=chunk_depth,
        rng_seed=rng_seed,
        progress=progress,
    )
    save_seed_mapping_result(
        output_npz=output_npz,
        seed_points=seed_points,
        original_shape=density_milli.shape,
        gamma=gamma,
        num_seeds=num_seeds,
        display_density=display_density,
        display_probability=display_probability,
        display_steps=display_steps,
        input_npz=input_npz,
    )
    return SeedMappingResult(
        input_npz=input_npz,
        output_npz=output_npz,
        seed_points=seed_points,
        display_density=display_density,
        display_probability=display_probability,
        display_steps=display_steps,
    )


def build_template_backfill(
    density_npz: Path,
    seed_mapping_npz: Path,
    output_npz: Path,
    *,
    representative_seeds: int = 81,
    candidate_limit: int = 15_000,
    candidate_voxel_bins: tuple[int, int, int] = (48, 48, 24),
    target_shape: tuple[int, int, int] = (999, 999, 399),
    template_seed_base: int = 1000,
    rng_seed: int = 42,
    progress: bool = False,
) -> BackfillResult:
    density_result = load_npz(density_npz)
    seed_mapping = load_npz(seed_mapping_npz)

    density_milli = density_result["density_milli"]
    full_seed_points = seed_mapping["seed_points"].astype(np.float32)
    normalized_seed_points = normalize_seed_points(full_seed_points, target_shape)
    candidate_points = build_candidate_pool(
        normalized_seed_points,
        candidate_limit=candidate_limit,
        voxel_bins=candidate_voxel_bins,
        rng_seed=rng_seed,
        progress=progress,
    )
    selected_representatives = farthest_point_sampling(
        candidate_points,
        count=representative_seeds,
        progress=progress,
    )
    (
        block_density,
        block_active,
        template_ids,
        template_seed_counts,
        templates,
    ) = build_backfilled_templates(
        density_milli,
        template_seed_base=template_seed_base,
        progress=progress,
    )
    save_backfilled_templates(
        output_npz,
        block_density=block_density,
        block_active=block_active,
        template_ids=template_ids,
        seed_counts=template_seed_counts,
        templates=templates,
        representative_seeds=selected_representatives,
        candidate_points=candidate_points,
        target_shape=target_shape,
    )
    return BackfillResult(
        input_npz=density_npz,
        output_npz=output_npz,
        representative_seeds=selected_representatives,
        candidate_points=candidate_points,
        block_density=block_density,
        block_active=block_active,
        template_ids=template_ids,
        template_seed_counts=template_seed_counts,
    )
