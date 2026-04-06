from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay


def build_delaunay_neighbor_map(seed_points: np.ndarray) -> dict[int, tuple[int, ...]]:
    seed_points = np.asarray(seed_points, dtype=np.float64)
    if len(seed_points) == 0:
        return {}
    if len(seed_points) == 1:
        return {0: tuple()}
    if len(seed_points) < 5:
        return {
            idx: tuple(sorted(jdx for jdx in range(seed_points.shape[0]) if jdx != idx))
            for idx in range(seed_points.shape[0])
        }

    neighbors: dict[int, set[int]] = {idx: set() for idx in range(seed_points.shape[0])}
    try:
        delaunay = Delaunay(seed_points, qhull_options="QJ")
        for simplex in delaunay.simplices:
            simplex = [int(v) for v in simplex]
            for i, left in enumerate(simplex):
                for right in simplex[i + 1 :]:
                    neighbors[left].add(right)
                    neighbors[right].add(left)
    except Exception:
        return {
            idx: tuple(sorted(jdx for jdx in range(seed_points.shape[0]) if jdx != idx))
            for idx in range(seed_points.shape[0])
        }
    return {seed_id: tuple(sorted(values)) for seed_id, values in neighbors.items()}
