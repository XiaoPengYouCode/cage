from __future__ import annotations

from pathlib import Path

import numpy as np
from stl import mesh as stl_mesh


def load_stl(path: Path) -> stl_mesh.Mesh:
    return stl_mesh.Mesh.from_file(str(path))


def stl_bounds(m: stl_mesh.Mesh) -> tuple[np.ndarray, np.ndarray]:
    verts = m.vectors.reshape(-1, 3)
    return verts.min(axis=0), verts.max(axis=0)


def voxelize_stl(
    m: stl_mesh.Mesh,
    voxel_size_mm: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxelize a surface mesh by ray-casting along the z axis.

    For each (x, y) column, cast a ray and toggle solid/void at each
    triangle intersection.  This correctly fills the interior of a
    closed watertight mesh.

    Returns
    -------
    occupancy : np.ndarray bool, shape (nx, ny, nz)
    origin    : np.ndarray float, shape (3,)  — mm coordinate of voxel [0,0,0] centre
    spacing   : np.ndarray float, shape (3,)  — actual voxel size in mm
    """
    lo, hi = stl_bounds(m)
    # Grid dimensions
    size = hi - lo
    nx = max(4, int(np.ceil(size[0] / voxel_size_mm)))
    ny = max(4, int(np.ceil(size[1] / voxel_size_mm)))
    nz = max(4, int(np.ceil(size[2] / voxel_size_mm)))
    spacing = size / np.array([nx, ny, nz], dtype=np.float64)

    # Voxel centre coordinates
    cx = lo[0] + (np.arange(nx) + 0.5) * spacing[0]
    cy = lo[1] + (np.arange(ny) + 0.5) * spacing[1]
    cz = lo[2] + (np.arange(nz) + 0.5) * spacing[2]

    occupancy = _ray_cast_z(m, cx, cy, cz, lo, spacing)
    origin = lo + spacing * 0.5
    return occupancy, origin, spacing


def _ray_cast_z(
    m: stl_mesh.Mesh,
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
    lo: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """Fill interior voxels using z-axis ray casting (parity test)."""
    nx, ny, nz = len(cx), len(cy), len(cz)
    occupancy = np.zeros((nx, ny, nz), dtype=bool)

    v0 = m.vectors[:, 0, :]  # (T, 3)
    v1 = m.vectors[:, 1, :]
    v2 = m.vectors[:, 2, :]

    # Pre-filter: only triangles whose xy bounding box overlaps the column
    tri_xmin = np.minimum(np.minimum(v0[:, 0], v1[:, 0]), v2[:, 0])
    tri_xmax = np.maximum(np.maximum(v0[:, 0], v1[:, 0]), v2[:, 0])
    tri_ymin = np.minimum(np.minimum(v0[:, 1], v1[:, 1]), v2[:, 1])
    tri_ymax = np.maximum(np.maximum(v0[:, 1], v1[:, 1]), v2[:, 1])

    for ix, x in enumerate(cx):
        x_mask = (tri_xmin <= x) & (tri_xmax >= x)
        for iy, y in enumerate(cy):
            mask = x_mask & (tri_ymin <= y) & (tri_ymax >= y)
            if not mask.any():
                continue
            hits = _ray_z_intersections(
                x, y, v0[mask], v1[mask], v2[mask]
            )
            if len(hits) == 0:
                continue
            hits.sort()
            # Parity fill: toggle inside/outside at each hit
            inside = False
            hit_idx = 0
            for iz in range(nz):
                while hit_idx < len(hits) and hits[hit_idx] < cz[iz]:
                    inside = not inside
                    hit_idx += 1
                occupancy[ix, iy, iz] = inside

    return occupancy


def _ray_z_intersections(
    x: float,
    y: float,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> list[float]:
    """Return z-values where the vertical ray (x,y,*) intersects each triangle."""
    hits: list[float] = []
    eps = 1e-10
    for i in range(len(v0)):
        a, b, c = v0[i], v1[i], v2[i]
        # Barycentric coords of (x,y) in the triangle projected to xy plane
        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denom) < eps:
            continue
        u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denom
        v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denom
        w = 1.0 - u - v
        if u >= -eps and v >= -eps and w >= -eps:
            z = u * a[2] + v * b[2] + w * c[2]
            hits.append(float(z))
    return hits
