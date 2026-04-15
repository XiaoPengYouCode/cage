"""Step 6.5 — Centroidal Voronoi Tessellation (CVT) via Lloyd's algorithm.

This implements the *standard* Lloyd iteration, which moves each seed point
to the **geometric centroid** of its Voronoi cell (not density-weighted).
The result is a spatially uniform distribution of seeds — cells converge
toward equal-volume convex polytopes.

Why geometric (not density-weighted) centroid:
    Standard CVT theory uses the unweighted centroid so that cells become as
    equal in size and shape as possible.  Density-weighted CVT would bias
    seeds back toward high-density regions, partially undoing the equi-spacing
    goal.  If anisotropic seeding is desired, keep more seeds in `sample_seeds`
    via gamma — do not conflate that with the relaxation step.

Reference:
    Du, Faber, Gunzburger (1999) "Centroidal Voronoi Tessellations:
    Applications and Algorithms", SIAM Review 41(4):637-676.

Usage (via CLI):
    matlab2stl-pipeline run-pipeline --cvt-iters 500 ...
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.spatial


# ---------------------------------------------------------------------------
# Internal helpers (mirrors box_voronoi helpers to keep this module standalone)
# ---------------------------------------------------------------------------

def _mirror_seeds(
    seeds: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> np.ndarray:
    """Return seeds + 6-face mirror copies (7× total).

    Mirror seeds guarantee finite Voronoi vertices for boundary cells,
    so every real seed gets a bounded cell.
    """
    reflected = [seeds]
    for axis in range(3):
        lo, hi = box_min[axis], box_max[axis]
        r_lo = seeds.copy()
        r_lo[:, axis] = 2 * lo - seeds[:, axis]
        r_hi = seeds.copy()
        r_hi[:, axis] = 2 * hi - seeds[:, axis]
        reflected.extend([r_lo, r_hi])
    return np.vstack(reflected)


def _cell_geometric_centroid(
    vor: scipy.spatial.Voronoi,
    seed_index: int,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> np.ndarray | None:
    """Compute the geometric centroid of the box-clipped Voronoi cell.

    Strategy:
        1. Collect the Voronoi vertices for this seed's region.
        2. Clip the convex hull to the box using HalfspaceIntersection.
        3. Return the centroid of the clipped vertex cloud.

    This is the *geometric* centroid of the clipped polytope, which is the
    standard Lloyd update.  Each vertex is treated with equal weight (no
    volume weighting per tetrahedron) — this is a good approximation and
    avoids decomposing the polytope into tetrahedra.

    Returns None for degenerate cells (< 4 vertices, clipping failure, etc.).
    """
    region_idx = vor.point_region[seed_index]
    region = vor.regions[region_idx]

    if not region or -1 in region:
        return None

    verts = vor.vertices[np.array(region)]
    if len(verts) < 4:
        return None

    # --- Clip to box via HalfspaceIntersection ---
    try:
        cell_hull = scipy.spatial.ConvexHull(verts)
    except Exception:
        return None

    cell_hs = cell_hull.equations  # (F, 4): n·x + d ≤ 0

    box_hs = []
    for axis in range(3):
        h_pos = np.zeros(4)
        h_pos[axis] = 1.0
        h_pos[3] = -box_max[axis]
        box_hs.append(h_pos)

        h_neg = np.zeros(4)
        h_neg[axis] = -1.0
        h_neg[3] = box_min[axis]
        box_hs.append(h_neg)
    box_hs = np.array(box_hs)

    halfspaces = np.vstack([cell_hs, box_hs])

    # Feasible interior point: centroid of Voronoi vertices clamped to box
    interior = np.clip(verts.mean(axis=0), box_min + 1e-6, box_max - 1e-6)

    def _feasible(pt: np.ndarray) -> bool:
        return bool(np.all(halfspaces[:, :3] @ pt + halfspaces[:, 3] <= 1e-9))

    if not _feasible(interior):
        # Fallback: box centre
        interior = (box_min + box_max) / 2.0
        if not _feasible(interior):
            return None  # degenerate cell; keep old seed

    try:
        hs_int = scipy.spatial.HalfspaceIntersection(halfspaces, interior)
        pts = hs_int.intersections
        if len(pts) < 4:
            return None
        # Geometric centroid of clipped vertex cloud
        return pts.mean(axis=0).astype(np.float64)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lloyd_relax(
    seeds_npz_path: Path,
    aligned_npz_path: Path,
    output_path: Path,
    num_iters: int = 500,
) -> np.ndarray:
    """Run Lloyd's algorithm (standard CVT) on seed points.

    Each iteration:
        1. Mirror seeds across the 6 box faces to produce bounded cells.
        2. Compute scipy Voronoi on the mirrored set.
        3. For each real seed: clip its cell to the box, compute geometric
           centroid, move the seed there.
        4. Clip all updated seeds strictly inside the box.

    Parameters
    ----------
    seeds_npz_path  : NPZ from Step 6 (contains ``seed_points``, grid metadata).
    aligned_npz_path: NPZ from Step 5 (contains ``grid_shape_xyz``).
    output_path     : Destination ``.npz`` (same schema as Step 6 output).
    num_iters       : Number of Lloyd iterations (default 500).

    Returns
    -------
    seed_points : float32 (N, 3) relaxed seeds in voxel-index coordinates.
    """
    seeds_data = np.load(str(seeds_npz_path))
    aligned_data = np.load(str(aligned_npz_path))

    seeds: np.ndarray = seeds_data["seed_points"].astype(np.float64)  # (N, 3)
    grid_shape = aligned_data["grid_shape_xyz"].tolist()               # [nx, ny, nz]
    voxel_size_xyz_m: np.ndarray = seeds_data["voxel_size_xyz_m"]
    origin_m: np.ndarray = seeds_data["origin_m"]
    num_seeds = len(seeds)

    box_min = np.zeros(3, dtype=np.float64)
    box_max = np.array([s - 1 for s in grid_shape], dtype=np.float64)

    print(f"  CVT Lloyd: {num_iters} iterations, {num_seeds} seeds …")

    for it in range(num_iters):
        all_seeds = _mirror_seeds(seeds, box_min, box_max)
        vor = scipy.spatial.Voronoi(all_seeds)

        new_seeds = np.empty_like(seeds)
        n_kept = 0  # seeds whose centroid computation failed → kept in place

        for i in range(num_seeds):
            centroid = _cell_geometric_centroid(vor, i, box_min, box_max)
            if centroid is not None:
                new_seeds[i] = centroid
            else:
                new_seeds[i] = seeds[i]
                n_kept += 1

        seeds = np.clip(new_seeds, box_min + 1e-3, box_max - 1e-3)

        if (it + 1) % 50 == 0 or it == num_iters - 1:
            print(f"    iter {it + 1:4d}/{num_iters}  (degenerate cells kept: {n_kept})")

    seed_points = seeds.astype(np.float32)
    seed_points_m = origin_m + seed_points * voxel_size_xyz_m

    payload = {
        "seed_points": seed_points,
        "seed_points_m": seed_points_m,
        "num_seeds": np.int32(num_seeds),
        "gamma": seeds_data["gamma"],
        "cvt_iters": np.int32(num_iters),
        "grid_shape_xyz": np.array(grid_shape, dtype=np.int32),
        "voxel_size_xyz_m": voxel_size_xyz_m,
        "origin_m": origin_m,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **payload)
    print(f"  CVT seeds → {output_path}")
    return seed_points
