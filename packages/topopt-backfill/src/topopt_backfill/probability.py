from __future__ import annotations

from pathlib import Path

import numpy as np


def load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def downsample_density(
    density_milli: np.ndarray,
    max_display_size: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    steps = tuple(max(1, int(np.ceil(size / max_display_size))) for size in density_milli.shape)
    sampled_density = density_milli[:: steps[0], :: steps[1], :: steps[2]].astype(np.float32) / 1000.0
    occupancy = sampled_density > 0.0
    return occupancy, sampled_density, steps


def density_to_probability_intensity(
    density: np.ndarray,
    gamma: float,
) -> np.ndarray:
    weights = np.where(density > 0.0, density**gamma, 0.0)
    max_weight = float(weights.max()) if weights.size else 0.0
    if max_weight <= 0.0:
        return np.zeros_like(density, dtype=np.float32)
    return (weights / max_weight).astype(np.float32)


def compute_chunk_weight_sums(
    density_milli: np.ndarray,
    gamma: float,
    chunk_depth: int,
    progress: bool = False,
) -> np.ndarray:
    z_size = density_milli.shape[2]
    chunk_sums = []
    for z_start in range(0, z_size, chunk_depth):
        z_stop = min(z_start + chunk_depth, z_size)
        chunk = density_milli[:, :, z_start:z_stop].astype(np.float32) / 1000.0
        weight_sum = float(np.sum(np.where(chunk > 0.0, chunk**gamma, 0.0), dtype=np.float64))
        chunk_sums.append(weight_sum)
        if progress:
            print(f"computed probability mass for z-slices [{z_start}, {z_stop})")
    return np.asarray(chunk_sums, dtype=np.float64)


def sample_seed_points(
    density_milli: np.ndarray,
    num_seeds: int,
    gamma: float,
    chunk_depth: int,
    rng_seed: int,
    progress: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    z_size = density_milli.shape[2]
    chunk_sums = compute_chunk_weight_sums(
        density_milli=density_milli,
        gamma=gamma,
        chunk_depth=chunk_depth,
        progress=progress,
    )
    total_mass = float(chunk_sums.sum())
    if total_mass <= 0.0:
        raise ValueError("No positive density found, cannot sample seed points.")

    chunk_probs = chunk_sums / total_mass
    chunk_counts = rng.multinomial(num_seeds, chunk_probs)
    sampled_points: list[np.ndarray] = []

    for chunk_index, z_start in enumerate(range(0, z_size, chunk_depth)):
        count = int(chunk_counts[chunk_index])
        z_stop = min(z_start + chunk_depth, z_size)
        if count == 0:
            continue

        chunk = density_milli[:, :, z_start:z_stop].astype(np.float32) / 1000.0
        weights = np.where(chunk > 0.0, chunk**gamma, 0.0)
        flat_weights = weights.ravel().astype(np.float64)
        cdf = np.cumsum(flat_weights)
        if cdf[-1] <= 0.0:
            continue

        random_values = np.sort(rng.random(count) * cdf[-1])
        flat_indices = np.searchsorted(cdf, random_values, side="right")
        x_idx, y_idx, z_idx_local = np.unravel_index(flat_indices, chunk.shape)
        points = np.column_stack(
            (
                x_idx.astype(np.float32) + rng.random(count, dtype=np.float32),
                y_idx.astype(np.float32) + rng.random(count, dtype=np.float32),
                (z_start + z_idx_local).astype(np.float32) + rng.random(count, dtype=np.float32),
            )
        )
        sampled_points.append(points)
        if progress:
            print(f"sampled {count} seeds from z-slices [{z_start}, {z_stop})")

    if not sampled_points:
        raise ValueError("Sampling failed, no seed points were generated.")
    return np.vstack(sampled_points).astype(np.float32)


def save_seed_mapping_result(
    output_npz: Path,
    seed_points: np.ndarray,
    original_shape: tuple[int, int, int],
    gamma: float,
    num_seeds: int,
    display_density: np.ndarray,
    display_probability: np.ndarray,
    display_steps: tuple[int, int, int],
    input_npz: Path,
) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        seed_points=seed_points,
        num_seeds=np.array(num_seeds, dtype=np.int32),
        gamma=np.array(gamma, dtype=np.float32),
        original_shape=np.array(original_shape, dtype=np.int32),
        display_density=np.round(display_density, 3).astype(np.float32),
        display_probability=np.round(display_probability, 6).astype(np.float32),
        display_steps=np.array(display_steps, dtype=np.int32),
        input_result_npz=np.array(str(input_npz)),
        mapping_type=np.array("density_power_probability"),
    )
