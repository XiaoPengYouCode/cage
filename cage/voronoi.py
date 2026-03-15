from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

from cage.models import EdgeSegment

CUBE_HALFSPACES = np.array(
    [
        [1.0, 0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -1.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.0],
        [0.0, 0.0, -1.0, 0.0],
    ],
    dtype=float,
)

FACE_SPECS = [
    ("x", 0.0),
    ("x", 1.0),
    ("y", 0.0),
    ("y", 1.0),
    ("z", 0.0),
    ("z", 1.0),
]


def generate_seeds(count: int, rng_seed: int) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return rng.random((count, 3))


def build_halfspaces(seed: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    halfspaces = [*CUBE_HALFSPACES]

    for other in seeds:
        if np.array_equal(seed, other):
            continue
        normal = 2.0 * (other - seed)
        offset = float(np.dot(seed, seed) - np.dot(other, other))
        halfspaces.append(np.append(normal, offset))

    return np.array(halfspaces, dtype=float)


def unique_vertices(vertices: np.ndarray, decimals: int = 10) -> np.ndarray:
    rounded = np.round(vertices, decimals=decimals)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    return vertices[np.sort(unique_indices)]


def build_voronoi_cells(seeds: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    cells: list[np.ndarray] = []
    halfspace_sets: list[np.ndarray] = []

    for seed in seeds:
        halfspaces = build_halfspaces(seed, seeds)
        intersections = HalfspaceIntersection(halfspaces, interior_point=seed).intersections
        cells.append(unique_vertices(intersections))
        halfspace_sets.append(halfspaces)

    return cells, halfspace_sets


def cube_edge_segments() -> list[EdgeSegment]:
    cube_vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    edge_indices = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return [(cube_vertices[start], cube_vertices[end]) for start, end in edge_indices]


def face_polygon(
    vertices: np.ndarray,
    axis_name: str,
    axis_value: float,
    tol: float = 1e-8,
) -> np.ndarray | None:
    axis_index = {"x": 0, "y": 1, "z": 2}[axis_name]
    on_face = vertices[np.isclose(vertices[:, axis_index], axis_value, atol=tol)]
    if len(on_face) < 3:
        return None

    rounded = np.round(on_face, decimals=10)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    face_vertices = on_face[np.sort(unique_indices)]
    if len(face_vertices) < 3:
        return None

    if axis_name == "x":
        planar = face_vertices[:, [1, 2]]
    elif axis_name == "y":
        planar = face_vertices[:, [0, 2]]
    else:
        planar = face_vertices[:, [0, 1]]

    if len(face_vertices) == 3:
        return face_vertices

    hull = ConvexHull(planar)
    return face_vertices[hull.vertices]


def canonical_segment_key(
    a: np.ndarray,
    b: np.ndarray,
    decimals: int = 8,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    point_a = tuple(np.round(a, decimals=decimals))
    point_b = tuple(np.round(b, decimals=decimals))
    return (point_a, point_b) if point_a <= point_b else (point_b, point_a)


def extract_cell_edges(
    vertices: np.ndarray,
    halfspaces: np.ndarray,
    tol: float = 1e-7,
) -> list[EdgeSegment]:
    if len(vertices) < 2:
        return []

    slacks = halfspaces[:, :3] @ vertices.T + halfspaces[:, 3:4]
    active_constraints = [
        set(np.flatnonzero(np.abs(slacks[:, vertex_index]) <= tol))
        for vertex_index in range(len(vertices))
    ]
    edges: dict[
        tuple[tuple[float, float, float], tuple[float, float, float]],
        EdgeSegment,
    ] = {}

    for constraint_index in range(len(halfspaces)):
        vertex_indices = [
            vertex_index
            for vertex_index, constraints in enumerate(active_constraints)
            if constraint_index in constraints
        ]
        if len(vertex_indices) < 3:
            continue

        plane_vertices = vertices[vertex_indices]
        rounded = np.round(plane_vertices, decimals=10)
        _, unique_indices = np.unique(rounded, axis=0, return_index=True)
        plane_vertices = plane_vertices[np.sort(unique_indices)]
        if len(plane_vertices) < 3:
            continue

        normal = halfspaces[constraint_index, :3]
        normal_norm = np.linalg.norm(normal)
        if normal_norm <= tol:
            continue
        normal = normal / normal_norm

        helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        basis_u = np.cross(normal, helper)
        basis_u /= np.linalg.norm(basis_u)
        basis_v = np.cross(normal, basis_u)

        centroid = plane_vertices.mean(axis=0)
        planar = np.column_stack(
            (
                (plane_vertices - centroid) @ basis_u,
                (plane_vertices - centroid) @ basis_v,
            )
        )

        if len(plane_vertices) == 3:
            ordered_vertices = plane_vertices
        else:
            hull = ConvexHull(planar)
            ordered_vertices = plane_vertices[hull.vertices]

        for start_index in range(len(ordered_vertices)):
            start_point = ordered_vertices[start_index]
            end_point = ordered_vertices[(start_index + 1) % len(ordered_vertices)]
            if np.linalg.norm(end_point - start_point) <= tol:
                continue
            edges.setdefault(
                canonical_segment_key(start_point, end_point),
                (start_point, end_point),
            )

    return list(edges.values())


def extract_unique_edges(
    cells: list[np.ndarray],
    halfspace_sets: list[np.ndarray],
) -> list[EdgeSegment]:
    unique_edges: dict[
        tuple[tuple[float, float, float], tuple[float, float, float]],
        EdgeSegment,
    ] = {}

    for vertices, halfspaces in zip(cells, halfspace_sets):
        for start, end in extract_cell_edges(vertices, halfspaces):
            unique_edges.setdefault(canonical_segment_key(start, end), (start, end))

    return list(unique_edges.values())
