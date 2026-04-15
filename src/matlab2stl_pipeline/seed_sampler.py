"""Step 6 — Sample seed points from the aligned probability density field.

200 seeds, gamma=1.0 (configurable).
Uses inverse-transform sampling on the flattened probability field,
then converts flat indices back to 3-D continuous coordinates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def sample_seeds(
    aligned_npz_path: Path,
    output_path: Path,
    num_seeds: int = 200,
    gamma: float = 1.0,
    rng_seed: int = 42,
) -> np.ndarray:
    """Sample *num_seeds* seed points from the aligned density field.

    Parameters
    ----------
    aligned_npz_path : NPZ produced by Step 5 (contains ``probability_field``
                       or ``density_milli`` if probability was not pre-saved).
    output_path      : Destination ``.npz``.
    num_seeds        : Number of seed points (default 200).
    gamma            : Power-law exponent for probability (default 1.0).
                       Only used if ``probability_field`` is absent from NPZ.
    rng_seed         : Random seed for reproducibility.

    Returns
    -------
    seed_points : float32 (num_seeds, 3) in voxel-index continuous coordinates.
    """
    data = np.load(str(aligned_npz_path))

    if "probability_field" in data:
        prob: np.ndarray = data["probability_field"].astype(np.float64)
    else:
        density = data["density_milli"].astype(np.float64) / 1000.0
        powered = np.where(density > 0, density ** gamma, 0.0)
        total = powered.sum()
        prob = powered / total if total > 0 else powered

    grid_shape = prob.shape  # (nx, ny, nz)

    # Flatten and build CDF
    flat_prob = prob.ravel()
    flat_prob = np.maximum(flat_prob, 0.0)
    cdf = np.cumsum(flat_prob)
    cdf /= cdf[-1]  # ensure exactly 1.0 at end

    rng = np.random.default_rng(rng_seed)
    u = rng.random(num_seeds)
    flat_indices = np.searchsorted(cdf, u).clip(0, len(cdf) - 1)

    # Convert flat → 3-D index and add sub-voxel jitter
    idx_3d = np.stack(np.unravel_index(flat_indices, grid_shape), axis=1).astype(np.float32)
    jitter = rng.uniform(-0.5, 0.5, size=idx_3d.shape).astype(np.float32)
    seed_points = idx_3d + jitter

    # Clip seeds strictly inside the box so no seed lands outside the domain.
    # Sub-voxel jitter on boundary voxels can produce slightly negative coords
    # (e.g. y=-0.22) which causes HalfspaceIntersection to fail for that cell.
    box_min = np.zeros(3, dtype=np.float32)
    box_max = np.array([s - 1 for s in grid_shape], dtype=np.float32)
    seed_points = np.clip(seed_points, box_min + 1e-3, box_max - 1e-3)

    # Convert to metres using voxel_size_xyz_m
    voxel_size_xyz_m: np.ndarray = data["voxel_size_xyz_m"]  # (3,)
    origin_m: np.ndarray = data["origin_m"]                  # (3,)
    seed_points_m = origin_m + seed_points * voxel_size_xyz_m

    payload = {
        "seed_points": seed_points,          # voxel-index coords (used for Voronoi)
        "seed_points_m": seed_points_m,      # physical coords (metres)
        "num_seeds": np.int32(num_seeds),
        "gamma": np.float32(gamma),
        "grid_shape_xyz": np.array(list(grid_shape), dtype=np.int32),
        "voxel_size_xyz_m": voxel_size_xyz_m,
        "origin_m": origin_m,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **payload)
    return seed_points
