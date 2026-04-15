"""Steps 7 & 8 — Restricted Voronoi inside an axis-aligned box + edge extraction.

Step 7 builds each cell from its physical support planes:
  1. Real Voronoi bisector planes against Delaunay neighbors.
  2. The 6 box planes.
  3. scipy.spatial.HalfspaceIntersection to solve the bounded convex polyhedron.
  4. Each face loop is recovered directly from the support plane it lies on.

This avoids the earlier failure mode where a clipped polygonal face was first
triangulated and Step 8 then had to guess which triangle edges were fake. We
still emit triangle simplices for GLB rendering, but those triangles are now a
pure visualization detail derived from the already-recovered polygon faces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.spatial

from topopt_sampling.neighbors import build_delaunay_neighbor_map

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


@dataclass(frozen=True)
class PlaneSupport:
    label: str
    equation: np.ndarray


def _normalize_halfspace_equation(equation: np.ndarray) -> np.ndarray:
    eq = np.asarray(equation, dtype=np.float64)
    norm = float(np.linalg.norm(eq[:3]))
    if norm <= 1e-12:
        return eq.copy()
    return eq / norm


def _canonical_plane_equation(equation: np.ndarray) -> np.ndarray:
    normalized = _normalize_halfspace_equation(equation)
    for value in normalized:
        if abs(float(value)) <= 1e-12:
            continue
        if value < 0:
            normalized = -normalized
        break
    return normalized


def _build_box_supports(box_min: np.ndarray, box_max: np.ndarray) -> tuple[PlaneSupport, ...]:
    return (
        PlaneSupport("box:x_min", _normalize_halfspace_equation(np.array([-1.0, 0.0, 0.0, box_min[0]], dtype=np.float64))),
        PlaneSupport("box:x_max", _normalize_halfspace_equation(np.array([1.0, 0.0, 0.0, -box_max[0]], dtype=np.float64))),
        PlaneSupport("box:y_min", _normalize_halfspace_equation(np.array([0.0, -1.0, 0.0, box_min[1]], dtype=np.float64))),
        PlaneSupport("box:y_max", _normalize_halfspace_equation(np.array([0.0, 1.0, 0.0, -box_max[1]], dtype=np.float64))),
        PlaneSupport("box:z_min", _normalize_halfspace_equation(np.array([0.0, 0.0, -1.0, box_min[2]], dtype=np.float64))),
        PlaneSupport("box:z_max", _normalize_halfspace_equation(np.array([0.0, 0.0, 1.0, -box_max[2]], dtype=np.float64))),
    )


def _build_bisector_support(seed_i: np.ndarray, seed_j: np.ndarray, neighbor_id: int) -> PlaneSupport:
    normal = 2.0 * (seed_j - seed_i)
    offset = float(np.dot(seed_i, seed_i) - np.dot(seed_j, seed_j))
    equation = _normalize_halfspace_equation(np.concatenate([normal, [offset]]))
    return PlaneSupport(label=f"bisector:{neighbor_id}", equation=equation)


def _is_feasible_point(halfspaces: np.ndarray, point: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(np.all(halfspaces[:, :3] @ point + halfspaces[:, 3] <= tol))


def _find_feasible_point(
    halfspaces: np.ndarray,
    seed_point: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> np.ndarray | None:
    candidate = np.clip(seed_point.astype(np.float64), box_min + 1e-6, box_max - 1e-6)
    if _is_feasible_point(halfspaces, candidate):
        return candidate

    box_centre = (box_min + box_max) / 2.0
    if _is_feasible_point(halfspaces, box_centre):
        return box_centre

    try:
        from scipy.optimize import linprog

        norms = np.linalg.norm(halfspaces[:, :3], axis=1, keepdims=True)
        safe_norms = np.where(norms <= 1e-12, 1.0, norms)
        a_ub = np.hstack([halfspaces[:, :3] / safe_norms, np.ones((halfspaces.shape[0], 1))])
        b_ub = -halfspaces[:, 3] / safe_norms.ravel()
        c_obj = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float64)
        result = linprog(
            c_obj,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=[(None, None), (None, None), (None, None), (0.0, None)],
            method="highs",
        )
        if result.success and result.x[3] > 1e-9:
            candidate = result.x[:3]
            if _is_feasible_point(halfspaces, candidate):
                return candidate.astype(np.float64)
    except Exception:
        return None

    return None


def _dedupe_points(points: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    unique: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()
    scale = 1.0 / tol
    for point in np.asarray(points, dtype=np.float64):
        key = tuple(int(round(float(value) * scale)) for value in point)
        if key in seen:
            continue
        seen.add(key)
        unique.append(point.astype(np.float64))
    if not unique:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(unique, dtype=np.float64)


def _order_face_vertices(points: np.ndarray, vertex_ids: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return ``vertex_ids`` ordered as a convex polygon on the face plane."""
    vertex_ids = np.asarray(sorted(set(int(idx) for idx in vertex_ids)), dtype=np.int32)
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
    ordered = vertex_ids[order].astype(np.int32)

    polygon = points[ordered]
    accum = np.zeros(3, dtype=np.float64)
    for idx in range(polygon.shape[0]):
        accum += np.cross(polygon[idx], polygon[(idx + 1) % polygon.shape[0]])
    if float(np.dot(accum, normal)) < 0:
        ordered = ordered[::-1]
    return ordered.astype(np.int32)


def _polygon_area_vector(points: np.ndarray) -> np.ndarray:
    accum = np.zeros(3, dtype=np.float64)
    for idx in range(points.shape[0]):
        accum += np.cross(points[idx], points[(idx + 1) % points.shape[0]])
    return accum


def _merge_coplanar_hull_faces(
    points: np.ndarray,
    simplices: np.ndarray,
    equations: np.ndarray,
    plane_tol: float = 1e-6,
) -> np.ndarray:
    if simplices.shape[0] == 0:
        return np.empty((0,), dtype=object)

    grouped: dict[tuple[float, float, float, float], dict[str, object]] = {}
    for simplex, equation in zip(simplices, equations):
        canonical = _canonical_plane_equation(equation)
        key = tuple(np.round(canonical / plane_tol) * plane_tol)
        group = grouped.setdefault(key, {"vertex_ids": set(), "normal": canonical[:3].copy()})
        cast_ids: set[int] = group["vertex_ids"]  # type: ignore[assignment]
        cast_ids.update(int(vertex_id) for vertex_id in simplex)

    faces = np.empty(len(grouped), dtype=object)
    for face_idx, group in enumerate(grouped.values()):
        vertex_ids = np.asarray(sorted(group["vertex_ids"]), dtype=np.int32)
        normal = np.asarray(group["normal"], dtype=np.float64)
        faces[face_idx] = _order_face_vertices(points.astype(np.float64), vertex_ids, normal)
    return faces


def _build_cell_faces(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape[0] < 4:
        return np.empty((0,), dtype=object)

    try:
        hull = scipy.spatial.ConvexHull(vertices)
    except Exception:
        return np.empty((0,), dtype=object)

    return _merge_coplanar_hull_faces(vertices, hull.simplices, hull.equations)


def _build_faces_from_supports(
    points: np.ndarray,
    supports: tuple[PlaneSupport, ...],
    plane_tol: float = 1e-6,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    faces: list[np.ndarray] = []
    face_equations: list[np.ndarray] = []

    for support in supports:
        residual = np.abs(points @ support.equation[:3] + support.equation[3])
        vertex_ids = np.flatnonzero(residual <= plane_tol)
        if vertex_ids.shape[0] < 3:
            continue

        ordered = _order_face_vertices(points, vertex_ids.astype(np.int32), support.equation[:3])
        if ordered.shape[0] < 3:
            continue

        area = np.linalg.norm(_polygon_area_vector(points[ordered]))
        if area <= 1e-8:
            continue

        faces.append(ordered.astype(np.int32))
        face_equations.append(support.equation.astype(np.float32))

    return faces, face_equations


def _triangulate_face_loops(face_loops: list[np.ndarray]) -> np.ndarray:
    triangles: list[list[int]] = []
    for face in face_loops:
        loop = np.asarray(face, dtype=np.int32)
        if loop.shape[0] < 3:
            continue
        for idx in range(1, loop.shape[0] - 1):
            triangles.append([int(loop[0]), int(loop[idx]), int(loop[idx + 1])])
    if not triangles:
        return np.zeros((0, 3), dtype=np.int32)
    return np.asarray(triangles, dtype=np.int32)


def _build_polyhedral_cell(
    seed_id: int,
    seed_points: np.ndarray,
    neighbor_ids: tuple[int, ...],
    box_supports: tuple[PlaneSupport, ...],
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    seed_point = seed_points[seed_id].astype(np.float64)
    bisector_supports = tuple(
        _build_bisector_support(seed_point, seed_points[neighbor_id].astype(np.float64), int(neighbor_id))
        for neighbor_id in neighbor_ids
        if int(neighbor_id) != seed_id
    )
    supports = bisector_supports + box_supports
    if not supports:
        return None

    halfspaces = np.asarray([support.equation for support in supports], dtype=np.float64)
    feasible_point = _find_feasible_point(halfspaces, seed_point, box_min, box_max)
    if feasible_point is None:
        return None

    try:
        intersections = scipy.spatial.HalfspaceIntersection(halfspaces, feasible_point).intersections
    except Exception:
        return None

    points = _dedupe_points(intersections)
    if points.shape[0] < 4:
        return None

    faces, _ = _build_faces_from_supports(points, supports)
    if not faces:
        return None

    simplices = _triangulate_face_loops(faces)

    faces_arr = np.empty(len(faces), dtype=object)
    for idx, face in enumerate(faces):
        faces_arr[idx] = face.astype(np.int32)

    return points.astype(np.float32), faces_arr, simplices.astype(np.int32)


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

    n_real = len(seed_points)
    neighbor_map = build_delaunay_neighbor_map(seed_points.astype(np.float64))
    box_supports = _build_box_supports(box_min, box_max)

    cell_vertices_list = []
    cell_faces_list = []
    cell_simplices_list = []

    for i in range(n_real):
        neighbor_ids = tuple(int(neighbor_id) for neighbor_id in neighbor_map.get(i, tuple()) if int(neighbor_id) != i)
        result = _build_polyhedral_cell(i, seed_points.astype(np.float64), neighbor_ids, box_supports, box_min, box_max)

        if result is None:
            fallback_neighbor_ids = tuple(j for j in range(n_real) if j != i)
            result = _build_polyhedral_cell(i, seed_points.astype(np.float64), fallback_neighbor_ids, box_supports, box_min, box_max)

        if result is None:
            cell_vertices_list.append(np.zeros((0, 3), dtype=np.float32))
            cell_faces_list.append(np.empty((0,), dtype=object))
            cell_simplices_list.append(np.zeros((0, 3), dtype=np.int32))
            continue

        vertices, faces, simplices = result
        cell_vertices_list.append(vertices.astype(np.float32))
        cell_faces_list.append(faces)
        cell_simplices_list.append(simplices.astype(np.int32))

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

def extract_voronoi_edges(
    voronoi_npz_path: Path,
    output_path: Path,
) -> np.ndarray:
    """Extract Voronoi skeleton edges from polygonal face boundaries.

    Step 7 stores each clipped convex cell as vertices plus ordered face loops.
    Step 8 walks those loops and emits the unique line segments on their
    boundaries, which are exactly the one-dimensional features of the
    restricted Voronoi complex.

    Returns / saves ``edges`` : float32 (E, 2, 3) — each row is (p0, p1).
    """
    data = np.load(str(voronoi_npz_path), allow_pickle=True)
    cell_vertices_arr: np.ndarray = data["cell_vertices"]
    cell_faces_arr: np.ndarray | None = data["cell_faces"] if "cell_faces" in data.files else None

    edge_set: dict[tuple, tuple] = {}

    for cell_idx, verts in enumerate(cell_vertices_arr):
        if verts.shape[0] < 2:
            continue

        if cell_faces_arr is not None:
            faces = cell_faces_arr[cell_idx]
        else:
            faces = _build_cell_faces(verts)

        for face in faces:
            loop = np.asarray(face, dtype=np.int32)
            if loop.shape[0] < 2:
                continue
            for edge_idx in range(loop.shape[0]):
                p0 = verts[int(loop[edge_idx])].astype(np.float64)
                p1 = verts[int(loop[(edge_idx + 1) % loop.shape[0]])].astype(np.float64)
                if np.linalg.norm(p1 - p0) < 1e-4:
                    continue
                key = tuple(sorted([
                    tuple(np.round(p0, 4)),
                    tuple(np.round(p1, 4)),
                ]))
                edge_set.setdefault(key, (p0.astype(np.float32), p1.astype(np.float32)))

    edges = np.array(list(edge_set.values()), dtype=np.float32) if edge_set else np.zeros((0, 2, 3), dtype=np.float32)
    print(f"  {len(edges)} face-boundary edges")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), edges=edges)
    return edges
