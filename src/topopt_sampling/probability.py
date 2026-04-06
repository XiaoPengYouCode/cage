from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def _load_mat(mat_path: Path) -> dict[str, np.ndarray]:
    raw = loadmat(mat_path, struct_as_record=False, squeeze_me=False)
    return {key: np.asarray(value) for key, value in raw.items() if not key.startswith("__")}


def _to_uint16(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.size == 0:
        return arr.astype(np.uint16)

    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.max(arr))
        if max_value <= 1.5:
            arr = arr * 1000.0
        arr = np.rint(arr)

    arr = np.asarray(arr, dtype=np.int64)
    arr = np.clip(arr, 0, 65535)
    return arr.astype(np.uint16)


def _ensure_density_milli(data: dict[str, np.ndarray]) -> np.ndarray:
    if "density_milli" in data:
        return _to_uint16(data["density_milli"])

    for candidate in ("density", "density_field", "density_matrix"):
        if candidate in data:
            return _to_uint16(data[candidate])

    raise ValueError(
        "Density array not found in input file, expected one of 'density_milli', 'density', 'density_field', or 'density_matrix'."
    )


def load_density_input(input_path: Path) -> dict[str, np.ndarray]:
    suffix = input_path.suffix.lower()
    if suffix == ".npz":
        return load_npz(input_path)
    if suffix == ".mat":
        data = _load_mat(input_path)
        data["density_milli"] = _ensure_density_milli(data)
        return data
    raise ValueError(
        f"Unsupported topology input format '{input_path.suffix}'. Only .npz and .mat are supported."
    )


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


def sample_seed_points(
    density_milli: np.ndarray,
    num_seeds: int,
    gamma: float,
    chunk_depth: int | None = None,
    rng_seed: int = 42,
    progress: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    density = density_milli.astype(np.float32) / 1000.0
    weights = np.where(density > 0.0, density**gamma, 0.0).astype(np.float64, copy=False)
    flat_weights = weights.ravel()
    total_mass = float(flat_weights.sum(dtype=np.float64))
    if total_mass <= 0.0 or not np.isfinite(total_mass):
        raise ValueError("No positive density found, cannot sample seed points.")

    cdf = np.cumsum(flat_weights, dtype=np.float64)
    random_values = np.sort(rng.random(num_seeds) * total_mass)
    flat_indices = np.searchsorted(cdf, random_values, side="right")
    x_idx, y_idx, z_idx = np.unravel_index(flat_indices, density_milli.shape)
    sampled_points = np.column_stack(
        (
            x_idx.astype(np.float32) + rng.random(num_seeds, dtype=np.float32),
            y_idx.astype(np.float32) + rng.random(num_seeds, dtype=np.float32),
            z_idx.astype(np.float32) + rng.random(num_seeds, dtype=np.float32),
        )
    )
    if progress:
        print(f"sampled {num_seeds} seeds from full density grid {density_milli.shape}")
    return sampled_points.astype(np.float32, copy=False)


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
        target_shape=np.array(original_shape, dtype=np.int32),
        display_density=np.round(display_density, 3).astype(np.float32),
        display_probability=np.round(display_probability, 6).astype(np.float32),
        display_steps=np.array(display_steps, dtype=np.int32),
        input_result_npz=np.array(str(input_npz)),
        mapping_type=np.array("density_power_probability"),
    )
