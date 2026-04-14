"""Steps 7 & 8 — Restricted Voronoi inside an axis-aligned box + edge extraction.

Step 7: Build Voronoi diagram clipped to the aligned bounding box.
  Method:
    1. scipy.spatial.Voronoi on seed points.
    2. Mirror (reflect) all seed points across each of the 6 box faces so that
       near-boundary cells gain finite Voronoi vertices.
    3. For every real seed's Voronoi region: collect its vertices, then clip the
       convex hull of those vertices to the box half-spaces to get the final
       bounded cell.

Step 8: Extract all unique edges from the clipped cells.
  We first recover each cell's *true polygonal faces* by merging coplanar hull
  triangles. Edges are then taken from those face loops instead of from the raw
  triangulation. This keeps real Voronoi / box-boundary edges while removing
  triangulation diagonals introduced by ConvexHull on flat box-clipped faces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.spatial

# Visually distinct HSL-derived colours for up to ~32 cells, then it cycles
_CELL_PALETTE = [
    (0.92, 0.35, 0.35), (0.35, 0.72, 0.45), (0.35, 0.55, 0.92),
    (0.92, 0.75, 0.30), (0.72, 0.35, 0.85), (0.30, 0.82, 0.82),
    (0.95, 0.55, 0.20), (0.50, 0.90, 0.35), (0.40, 0.40, 0.90),
    (0.88, 0.30, 0.60), (0.30, 0.70, 0.60), (0.85, 0.85, 0.25),
    (0.60, 0.25, 0.90), (0.25, 0.85, 0.65), (0.90, 0.50, 0.50),
    (0.50, 0.65, 0.90), (0.75, 0.90, 0.35), (0.90, 0.35, 0.75),
    (0.35, 0.90, 0.90), (0.65, 0.45, 0.25), (0.45, 0.25, 0.65),
    (0.25, 0.65, 0.45), (0.80, 0.60, 0.40), (0.40, 0.80, 0.60),
    (0.60, 0.40, 0.80), (0.90, 0.80, 0.60), (0.60, 0.90, 0.80),
    (0.80, 0.60, 0.90), (0.70, 0.30, 0.30), (0.30, 0.70, 0.30),
    (0.30, 0.30, 0.70), (0.70, 0.70, 0.30),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mirror_seeds(seeds: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """Return the 6-face mirror images of *seeds* appended to *seeds*.

    Each of the 6 faces produces a full copy of the seed array reflected
    across that face plane.  The mirrors are only used to produce finite
    Voronoi vertices near boundaries; they are discarded afterward.
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


def _clip_convex_hull_to_box(
    vertices: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> np.ndarray | None:
    """Clip a convex polytope (given by Voronoi *vertices*) to the box.

    Uses scipy.spatial.HalfspaceIntersection with:
      - The 6 box face half-spaces
      - The Voronoi cell half-spaces (one per face of the cell's convex hull)

    The feasible interior point is the seed point (centroid of vertices),
    guaranteed to be inside both the cell and (after clipping) the box.

    Returns vertex array of clipped cell, or None if degenerate.
    """
    # Feasible interior point = centroid of Voronoi vertices (always inside cell)
    interior = vertices.mean(axis=0)

    # Clamp interior point to strictly inside the box
    interior = np.clip(interior, box_min + 1e-6, box_max - 1e-6)

    # Build Voronoi cell half-spaces from its convex hull faces
    # scipy ConvexHull equations: [normal | offset] with normal·x + offset ≤ 0
    try:
        cell_hull = scipy.spatial.ConvexHull(vertices)
    except Exception:
        return None
    cell_hs = cell_hull.equations   # (F, 4)

    # Build box half-spaces: n·x + d ≤ 0
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
    box_hs = np.array(box_hs)  # (6, 4)

    halfspaces = np.vstack([cell_hs, box_hs])

    # Find a feasible interior point that satisfies ALL half-spaces.
    # The centroid of the Voronoi vertices may lie outside the box (e.g. when
    # the seed itself is slightly outside due to sub-voxel jitter in sampling).
    # Strategy: try the centroid first; if it fails, try the seed point clamped
    # to the box; finally fall back to the Chebyshev centre via linear program.
    def _is_feasible(pt: np.ndarray) -> bool:
        return bool(np.all(halfspaces[:, :3] @ pt + halfspaces[:, 3] <= 1e-9))

    feasible_pt = None
    if _is_feasible(interior):
        feasible_pt = interior
    else:
        # Try box centre
        box_centre = (box_min + box_max) / 2.0
        if _is_feasible(box_centre):
            feasible_pt = box_centre
        else:
            # Chebyshev centre: largest inscribed sphere — always exists for
            # non-empty bounded polytope
            try:
                from scipy.optimize import linprog
                n_hs = halfspaces.shape[0]
                norms = np.linalg.norm(halfspaces[:, :3], axis=1, keepdims=True)
                A = np.hstack([halfspaces[:, :3] / norms, np.ones((n_hs, 1))])
                b = -halfspaces[:, 3] / norms.ravel()
                # Minimise -r  (maximise inscribed sphere radius r)
                c = np.zeros(4); c[3] = -1.0
                res = linprog(c, A_ub=A, b_ub=b,
                              bounds=[(None, None)] * 3 + [(0, None)],
                              method="highs")
                if res.success and res.x[3] > 1e-9:
                    candidate = res.x[:3]
                    if _is_feasible(candidate):
                        feasible_pt = candidate
            except Exception:
                pass

    if feasible_pt is None:
        return None

    try:
        hs_intersection = scipy.spatial.HalfspaceIntersection(halfspaces, feasible_pt)
        pts = hs_intersection.intersections
        if len(pts) < 4:
            return None
        hull = scipy.spatial.ConvexHull(pts)
        return pts[hull.vertices].astype(np.float32)
    except Exception:
        return None


def _point_in_box(p: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> bool:
    return bool(np.all(p >= box_min - 1e-9) and np.all(p <= box_max + 1e-9))


def _canonical_plane_equation(equation: np.ndarray) -> np.ndarray:
    """Normalize a plane equation for stable coplanar facet grouping."""
    eq = np.asarray(equation, dtype=np.float64)
    normal = eq[:3]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return eq.copy()
    eq = eq / norm
    for value in eq:
        if abs(value) <= 1e-12:
            continue
        if value < 0:
            eq = -eq
        break
    return eq


def _order_face_vertices(points: np.ndarray, vertex_ids: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return ``vertex_ids`` ordered as a convex polygon on the face plane."""
    face_points = points[vertex_ids]
    centroid = face_points.mean(axis=0)

    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    axis_u = np.cross(normal, helper)
    axis_u_norm = float(np.linalg.norm(axis_u))
    if axis_u_norm <= 1e-12:
        helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis_u = np.cross(normal, helper)
        axis_u_norm = float(np.linalg.norm(axis_u))
    axis_u /= axis_u_norm
    axis_v = np.cross(normal, axis_u)

    local = face_points - centroid
    angles = np.arctan2(local @ axis_v, local @ axis_u)
    order = np.argsort(angles)
    return vertex_ids[order].astype(np.int32)


def _merge_coplanar_hull_faces(
    points: np.ndarray,
    simplices: np.ndarray,
    equations: np.ndarray,
    tol: float = 1e-6,
) -> np.ndarray:
    """Merge hull triangles that lie on the same plane into polygonal faces."""
    if simplices.shape[0] == 0:
        return np.empty((0,), dtype=object)

    grouped: dict[tuple[float, float, float, float], dict[str, object]] = {}
    for simplex, equation in zip(simplices, equations):
        canonical = _canonical_plane_equation(equation)
        key = tuple(np.round(canonical / tol) * tol)
        group = grouped.setdefault(
            key,
            {
                "vertex_ids": set(),
                "normals": [],
            },
        )
        group["vertex_ids"].update(int(idx) for idx in simplex)
        group["normals"].append(canonical[:3])

    faces = np.empty(len(grouped), dtype=object)
    for face_idx, group in enumerate(grouped.values()):
        vertex_ids = np.array(sorted(group["vertex_ids"]), dtype=np.int32)
        normal = np.mean(np.asarray(group["normals"], dtype=np.float64), axis=0)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        faces[face_idx] = _order_face_vertices(points.astype(np.float64), vertex_ids, normal)
    return faces


def _build_cell_faces(vertices: np.ndarray) -> np.ndarray:
    """Rebuild polygonal faces for one convex cell from its hull."""
    if vertices.shape[0] < 4:
        return np.empty((0,), dtype=object)
    hull = scipy.spatial.ConvexHull(vertices.astype(np.float64))
    return _merge_coplanar_hull_faces(vertices.astype(np.float64), hull.simplices, hull.equations)


# ---------------------------------------------------------------------------
# Step 7 — Build clipped Voronoi cells
# ---------------------------------------------------------------------------

def build_box_voronoi(
    seeds_npz_path: Path,
    aligned_npz_path: Path,
    output_path: Path,
) -> dict:
    """Compute Voronoi cells clipped to the aligned bounding box.

    Returns / saves a dict with:
        cell_vertices : object array of float32 (V_i, 3) per cell
        cell_faces    : object array of int arrays per cell, one loop per face
        cell_hull_simplices : object array of int arrays per cell
        seed_points   : float32 (N, 3) voxel-index coords
        box_min       : float32 (3,)
        box_max       : float32 (3,)
    """
    seeds_data = np.load(str(seeds_npz_path))
    aligned_data = np.load(str(aligned_npz_path))

    seed_points: np.ndarray = seeds_data["seed_points"]   # (N, 3) voxel coords
    grid_shape = aligned_data["grid_shape_xyz"].tolist()   # [nx, ny, nz]

    box_min = np.zeros(3, dtype=np.float64)
    box_max = np.array(grid_shape, dtype=np.float64) - 1  # inclusive voxel index range

    # Mirror seeds across all 6 faces
    all_seeds = _mirror_seeds(seed_points.astype(np.float64), box_min, box_max)
    n_real = len(seed_points)

    vor = scipy.spatial.Voronoi(all_seeds)

    cell_vertices_list = []
    cell_faces_list = []
    cell_simplices_list = []

    for i in range(n_real):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        # Skip if region is empty or contains infinite vertex (-1)
        if not region:
            cell_vertices_list.append(np.zeros((0, 3), dtype=np.float32))
            cell_faces_list.append(np.empty((0,), dtype=object))
            cell_simplices_list.append(np.zeros((0, 3), dtype=np.int32))
            continue

        # Collect Voronoi vertices for this cell
        if -1 in region:
            # Mirror trick should have eliminated these; fall back gracefully
            valid_indices = [v for v in region if v >= 0]
            verts = vor.vertices[valid_indices] if valid_indices else np.empty((0, 3))
        else:
            verts = vor.vertices[np.array(region)]

        if len(verts) < 4:
            cell_vertices_list.append(np.zeros((0, 3), dtype=np.float32))
            cell_faces_list.append(np.empty((0,), dtype=object))
            cell_simplices_list.append(np.zeros((0, 3), dtype=np.int32))
            continue

        # Clip to box
        clipped = _clip_convex_hull_to_box(verts, box_min, box_max)
        if clipped is None or len(clipped) < 4:
            cell_vertices_list.append(np.zeros((0, 3), dtype=np.float32))
            cell_faces_list.append(np.empty((0,), dtype=object))
            cell_simplices_list.append(np.zeros((0, 3), dtype=np.int32))
            continue

        try:
            hull = scipy.spatial.ConvexHull(clipped)
            cell_vertices_list.append(clipped.astype(np.float32))
            cell_faces_list.append(_merge_coplanar_hull_faces(clipped, hull.simplices, hull.equations))
            cell_simplices_list.append(hull.simplices.astype(np.int32))
        except Exception:
            cell_vertices_list.append(np.zeros((0, 3), dtype=np.float32))
            cell_faces_list.append(np.empty((0,), dtype=object))
            cell_simplices_list.append(np.zeros((0, 3), dtype=np.int32))

    cell_vertices_arr = np.empty(n_real, dtype=object)
    cell_faces_arr = np.empty(n_real, dtype=object)
    cell_simplices_arr = np.empty(n_real, dtype=object)
    for i, (v, f, s) in enumerate(zip(cell_vertices_list, cell_faces_list, cell_simplices_list)):
        cell_vertices_arr[i] = v
        cell_faces_arr[i] = f
        cell_simplices_arr[i] = s

    payload = {
        "cell_vertices": cell_vertices_arr,
        "cell_faces": cell_faces_arr,
        "cell_hull_simplices": cell_simplices_arr,
        "seed_points": seed_points.astype(np.float32),
        "box_min": box_min.astype(np.float32),
        "box_max": box_max.astype(np.float32),
        "n_cells": np.int32(n_real),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **payload)
    return payload


# ---------------------------------------------------------------------------
# Step 7b — Export Voronoi cell colour map as GLB
# ---------------------------------------------------------------------------

def export_voronoi_cells_glb(
    voronoi_npz_path: Path,
    output_glb_path: Path,
    aligned_npz_path: Path,
) -> Path:
    """Render each clipped Voronoi cell as a solid coloured polyhedron in a GLB.

    Each cell gets a distinct colour from the palette (cycles if N > palette).
    World coordinates are in mm (same as all other GLBs in the pipeline).

    Parameters
    ----------
    voronoi_npz_path  : output of Step 7 (cell_vertices, cell_hull_simplices, …)
    output_glb_path   : destination .glb file
    aligned_npz_path  : needed for voxel_size_xyz_m → mm conversion and origin

    Returns
    -------
    output_glb_path
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ct_reconstruction.glb_export import _build_glb_multi

    data = np.load(str(voronoi_npz_path), allow_pickle=True)
    aligned = np.load(str(aligned_npz_path))

    cell_vertices_arr: np.ndarray = data["cell_vertices"]
    cell_simplices_arr: np.ndarray = data["cell_hull_simplices"]

    voxel_size_m: np.ndarray = aligned["voxel_size_xyz_m"].astype(np.float64)  # (3,)
    origin_m: np.ndarray = aligned["origin_m"].astype(np.float64)              # (3,)
    voxel_size_mm = voxel_size_m * 1e3
    origin_mm = origin_m * 1e3

    parts = []  # (verts_mm, normals, faces, color, name)

    for i, (verts_vox, simplices) in enumerate(zip(cell_vertices_arr, cell_simplices_arr)):
        if verts_vox.shape[0] < 4 or simplices.shape[0] == 0:
            continue

        # Convert voxel-index coords → mm world coords
        verts_mm = (verts_vox.astype(np.float64) * voxel_size_mm + origin_mm).astype(np.float32)

        # Build face index array and per-vertex outward normals from simplices
        faces = simplices.astype(np.uint32)

        # Compute per-vertex normals as mean of adjacent face normals
        normals = np.zeros_like(verts_mm)
        v0 = verts_mm[faces[:, 0]]
        v1 = verts_mm[faces[:, 1]]
        v2 = verts_mm[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0).astype(np.float32)  # face normals (unnorm)

        # Ensure face normals point outward (away from centroid)
        centroid = verts_mm.mean(axis=0)
        face_centers = (v0 + v1 + v2) / 3.0
        outward = face_centers - centroid
        flip = (fn * outward).sum(axis=1) < 0
        fn[flip] *= -1
        faces[flip] = faces[flip][:, [0, 2, 1]]  # fix winding

        for fi, (i0, i1, i2) in enumerate(faces):
            normals[i0] += fn[fi]
            normals[i1] += fn[fi]
            normals[i2] += fn[fi]

        norms_len = np.linalg.norm(normals, axis=1, keepdims=True)
        norms_len = np.where(norms_len < 1e-12, 1.0, norms_len)
        normals = (normals / norms_len).astype(np.float32)

        color = _CELL_PALETTE[i % len(_CELL_PALETTE)]
        parts.append((verts_mm, normals, faces, color, f"cell_{i}"))

    if not parts:
        raise ValueError("No valid Voronoi cells to export.")

    from ct_reconstruction.glb_export import _build_glb_multi
    glb_bytes = _build_glb_multi(parts)
    output_glb_path.parent.mkdir(parents=True, exist_ok=True)
    output_glb_path.write_bytes(glb_bytes)
    return output_glb_path


# ---------------------------------------------------------------------------
# Step 8 — Extract all edges
# ---------------------------------------------------------------------------

def _clip_segment_to_box(
    p: np.ndarray,
    d: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip the ray p + t*d to the box using slab method.

    Returns (p0, p1) endpoints of the clipped segment inside the box, or None.
    """
    t_min, t_max = -np.inf, np.inf
    for ax in range(3):
        if abs(d[ax]) < 1e-12:
            if p[ax] < box_min[ax] - 1e-9 or p[ax] > box_max[ax] + 1e-9:
                return None
        else:
            t0 = (box_min[ax] - p[ax]) / d[ax]
            t1 = (box_max[ax] - p[ax]) / d[ax]
            if t0 > t1:
                t0, t1 = t1, t0
            t_min = max(t_min, t0)
            t_max = min(t_max, t1)
    if t_min > t_max + 1e-9:
        return None
    return p + t_min * d, p + t_max * d


def extract_voronoi_edges(
    voronoi_npz_path: Path,
    output_path: Path,
) -> np.ndarray:
    """Extract true Voronoi ridge edges directly from scipy.spatial.Voronoi.

    This is the mathematically correct approach: use vor.ridge_vertices, which
    gives the actual polygon vertex loop for each Voronoi ridge (the shared face
    between two adjacent cells). Adjacent vertex pairs in each loop are the real
    ridge edges. This completely bypasses ConvexHull triangulation and therefore
    produces zero triangulation-diagonal artefacts, on box faces or internally.

    Ridges with infinite vertices (-1) are handled by shooting a ray from the
    finite endpoint in the ridge direction and clipping it to the box.

    Returns / saves ``edges`` : float32 (E, 2, 3) — each row is (p0, p1).
    """
    data = np.load(str(voronoi_npz_path), allow_pickle=True)
    seed_points: np.ndarray = data["seed_points"].astype(np.float64)
    box_min: np.ndarray = data["box_min"].astype(np.float64)
    box_max: np.ndarray = data["box_max"].astype(np.float64)
    n_real = int(data["n_cells"])

    # Rebuild Voronoi (fast — same deterministic result as Step 7)
    all_seeds = _mirror_seeds(seed_points, box_min, box_max)
    vor = scipy.spatial.Voronoi(all_seeds)

    edge_set: dict[tuple, tuple] = {}
    n_inf_clipped = 0
    n_outside = 0

    for (i0, i1), ridge_verts in zip(vor.ridge_points, vor.ridge_vertices):
        # Keep ridges where at least one endpoint is a real seed.
        # Ridges between a real seed and a mirror seed produce the boundary
        # edges of cells that touch the box faces / edges / corners.
        if i0 >= n_real and i1 >= n_real:
            continue

        ridge_verts = list(ridge_verts)

        if -1 in ridge_verts:
            # Ridge has an infinite vertex — clip the ray to the box
            finite_idx = [v for v in ridge_verts if v >= 0]
            if not finite_idx:
                continue
            # Direction: perpendicular to the line joining the two seeds,
            # in the plane of the ridge (midpoint-normal direction)
            s0 = all_seeds[i0]
            s1 = all_seeds[i1]
            midpoint = (s0 + s1) / 2.0
            seed_dir = s1 - s0
            # Ridge direction: cross product of seed_dir with any ridge edge dir
            finite_pts = vor.vertices[finite_idx]
            if len(finite_pts) >= 2:
                edge_dir = finite_pts[1] - finite_pts[0]
                ray_dir = np.cross(seed_dir, edge_dir)
            else:
                # Only one finite vertex — use perpendicular in the seed plane
                ray_dir = np.array([seed_dir[1], -seed_dir[0], 0.0])
            ray_norm = np.linalg.norm(ray_dir)
            if ray_norm < 1e-12:
                continue
            ray_dir /= ray_norm
            # Shoot from each finite vertex pair, then from finite→infinity
            for fi in range(len(finite_idx) - 1):
                p0 = vor.vertices[finite_idx[fi]]
                p1 = vor.vertices[finite_idx[fi + 1]]
                _add_edge(p0, p1, box_min, box_max, edge_set)
            # Clip the infinite ray
            p_start = vor.vertices[finite_idx[-1]]
            clipped = _clip_segment_to_box(p_start, ray_dir, box_min, box_max)
            if clipped is not None:
                _add_edge(clipped[0], clipped[1], box_min, box_max, edge_set)
                n_inf_clipped += 1
            continue

        # All finite vertices — extract adjacent pairs in the polygon loop
        pts = vor.vertices[ridge_verts]

        # Order the polygon vertices (they may be unordered from scipy)
        if len(ridge_verts) >= 3:
            centroid = pts.mean(axis=0)
            normal = all_seeds[i1] - all_seeds[i0]
            normal /= max(np.linalg.norm(normal), 1e-12)
            helper = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(helper, normal)) > 0.9:
                helper = np.array([0.0, 1.0, 0.0])
            u = np.cross(normal, helper)
            u /= max(np.linalg.norm(u), 1e-12)
            v = np.cross(normal, u)
            local = pts - centroid
            angles = np.arctan2(local @ v, local @ u)
            order = np.argsort(angles)
            pts = pts[order]

        for k in range(len(pts)):
            p0 = pts[k]
            p1 = pts[(k + 1) % len(pts)]
            _add_edge(p0, p1, box_min, box_max, edge_set)

    edges = np.array(list(edge_set.values()), dtype=np.float32) if edge_set else np.zeros((0, 2, 3), dtype=np.float32)
    print(f"  {len(edges)} ridge edges  ({n_inf_clipped} infinite ridges clipped to box, {n_outside} outside box skipped)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), edges=edges)
    return edges


def _add_edge(
    p0: np.ndarray,
    p1: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
    edge_set: dict,
    tol: float = 1e-4,
) -> None:
    """Clip edge to box and add to edge_set if it has non-zero length inside."""
    # Both points must be inside (or on) the box
    p0c = np.clip(p0, box_min, box_max)
    p1c = np.clip(p1, box_min, box_max)
    if np.linalg.norm(p1c - p0c) < tol:
        return
    key = tuple(sorted([tuple(np.round(p0c, 4)), tuple(np.round(p1c, 4))]))
    edge_set.setdefault(key, (p0c.astype(np.float32), p1c.astype(np.float32)))
