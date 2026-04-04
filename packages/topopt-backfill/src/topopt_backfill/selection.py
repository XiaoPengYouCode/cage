from __future__ import annotations

import numpy as np


def normalize_seed_points(
    seed_points: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    scale = np.array(target_shape, dtype=np.float32)
    normalized = seed_points / scale
    eps = 1e-5
    return np.clip(normalized, eps, 1.0 - eps)


def build_candidate_pool(
    points: np.ndarray,
    candidate_limit: int,
    voxel_bins: tuple[int, int, int],
    rng_seed: int,
    progress: bool = False,
) -> np.ndarray:
    if len(points) <= candidate_limit:
        if progress:
            print(f"candidate pool unchanged: {len(points)}")
        return points.astype(np.float32, copy=False)

    clipped = np.clip(points, 0.0, 1.0 - 1e-7)
    bins = np.array(voxel_bins, dtype=np.int32)
    grid_idx = np.floor(clipped * bins[None, :]).astype(np.int32)
    keys = np.ravel_multi_index(grid_idx.T, bins)
    _, unique_indices = np.unique(keys, return_index=True)
    pooled = points[np.sort(unique_indices)].astype(np.float32, copy=False)
    if progress:
        print(
            f"candidate pool after voxel bucketing: {len(pooled)} "
            f"(from {len(points)}, bins={voxel_bins[0]}x{voxel_bins[1]}x{voxel_bins[2]})"
        )

    if len(pooled) > candidate_limit:
        rng = np.random.default_rng(rng_seed)
        selection = rng.choice(len(pooled), size=candidate_limit, replace=False)
        pooled = pooled[selection]
        if progress:
            print(f"candidate pool downsampled to: {len(pooled)}")

    return pooled


def farthest_point_sampling(
    points: np.ndarray,
    count: int,
    progress: bool = False,
) -> np.ndarray:
    if count >= len(points):
        if progress:
            print(f"candidate pool smaller than requested representative seeds: {len(points)}")
        return points.copy()

    selected = np.empty((count, 3), dtype=np.float32)
    centroid = points.mean(axis=0, dtype=np.float32)
    distances_to_centroid = np.sum((points - centroid) ** 2, axis=1, dtype=np.float32)
    first_index = int(np.argmax(distances_to_centroid))
    selected[0] = points[first_index]

    min_dist_sq = np.sum((points - selected[0]) ** 2, axis=1, dtype=np.float32)
    for i in range(1, count):
        next_index = int(np.argmax(min_dist_sq))
        selected[i] = points[next_index]
        dist_sq = np.sum((points - selected[i]) ** 2, axis=1, dtype=np.float32)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)
        if progress and (i == 1 or i + 1 == count or (i + 1) % max(1, count // 20) == 0):
            print(f"selected representative seed {i + 1}/{count}")
    return selected
