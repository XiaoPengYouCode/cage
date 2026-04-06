from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from topopt_sampling.exact_restricted_voronoi_3d import (
    AnnularCylinderDomain,
    ExactRestrictedCell,
    ExactRestrictedVoronoiDiagram,
    build_exact_restricted_voronoi_diagram,
)

_TOL = 1e-7


@dataclass(frozen=True)
class PlaneSupport:
    key: str
    normal: np.ndarray
    rhs: float
    support_type: str
    neighbor_seed_id: int | None = None


@dataclass(frozen=True)
class CylinderSupport:
    key: str
    center_xy: np.ndarray
    radius: float
    support_type: str


Support = PlaneSupport | CylinderSupport


@dataclass(frozen=True)
class ExactVertex:
    vertex_id: int
    point: np.ndarray
    support_keys: tuple[str, ...]


@dataclass(frozen=True)
class ExactLineSegment:
    kind: str
    start: np.ndarray
    end: np.ndarray


@dataclass(frozen=True)
class ExactCircleArc:
    kind: str
    center: np.ndarray
    radius: float
    normal: np.ndarray
    start_angle: float
    end_angle: float
    z_value: float
    start: np.ndarray
    end: np.ndarray


@dataclass(frozen=True)
class ExactCylinderPlaneCurve:
    kind: str
    cylinder_radius: float
    cylinder_center_xy: np.ndarray
    plane_normal: np.ndarray
    plane_rhs: float
    theta_start: float
    theta_end: float
    start: np.ndarray
    end: np.ndarray
    vertical_theta: float | None = None


ExactCurve = ExactLineSegment | ExactCircleArc | ExactCylinderPlaneCurve


@dataclass(frozen=True)
class ExactEdge:
    edge_id: int
    support_keys: tuple[str, str]
    vertex_ids: tuple[int, int]
    curve: ExactCurve


@dataclass(frozen=True)
class ExactFace:
    face_id: int
    seed_id: int
    support_key: str
    support_type: str
    loop_edge_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class PolyhedralFace:
    face_id: int
    seed_id: int
    support_key: str
    support_type: str
    loop_edge_ids: tuple[tuple[int, ...], ...]
    neighbor_seed_id: int | None = None


@dataclass(frozen=True)
class PolyhedralVoronoiCell:
    seed_id: int
    supports: tuple[PlaneSupport, ...]
    faces: tuple[PolyhedralFace, ...]
    edges: tuple[ExactEdge, ...]
    vertices: tuple[ExactVertex, ...]


@dataclass(frozen=True)
class TrimmedCellPatchSet:
    kept_plane_face_ids: tuple[int, ...]
    generated_cylinder_face_ids: tuple[int, ...]
    plane_cylinder_edge_ids: tuple[int, ...]
    cap_cylinder_edge_ids: tuple[int, ...]
    new_vertex_ids: tuple[int, ...]


@dataclass(frozen=True)
class TrimmedAnnularCell:
    seed_id: int
    polyhedral_cell: PolyhedralVoronoiCell
    supports: tuple[Support, ...]
    faces: tuple[ExactFace, ...]
    edges: tuple[ExactEdge, ...]
    vertices: tuple[ExactVertex, ...]
    trim_summary: TrimmedCellPatchSet


@dataclass(frozen=True)
class HybridExactCellBRep:
    seed_id: int
    polyhedral_cell: PolyhedralVoronoiCell
    supports: tuple[Support, ...]
    faces: tuple[ExactFace, ...]
    edges: tuple[ExactEdge, ...]
    vertices: tuple[ExactVertex, ...]
    trim_summary: TrimmedCellPatchSet


@dataclass(frozen=True)
class HybridExactDiagramBRep:
    cells: tuple[HybridExactCellBRep, ...]


@dataclass(frozen=True)
class HybridExactSummary:
    num_cells: int
    num_faces: int
    num_edges: int
    num_vertices: int


def _plane_support_for_neighbor(cell: ExactRestrictedCell, diagram: ExactRestrictedVoronoiDiagram, neighbor_seed_id: int) -> PlaneSupport:
    seed_i = diagram.seed_points[cell.seed_id]
    seed_j = diagram.seed_points[int(neighbor_seed_id)]
    normal = 2.0 * (seed_j - seed_i)
    rhs = float(np.dot(seed_j, seed_j) - np.dot(seed_i, seed_i))
    return PlaneSupport(
        key=f"bisector:{cell.seed_id}:{int(neighbor_seed_id)}",
        normal=normal.astype(np.float64),
        rhs=rhs,
        support_type="plane",
        neighbor_seed_id=int(neighbor_seed_id),
    )


def _cap_support(key: str, z_value: float, inward: str) -> PlaneSupport:
    if inward == "down":
        return PlaneSupport(
            key=key,
            normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            rhs=float(z_value),
            support_type="cap",
        )
    return PlaneSupport(
        key=key,
        normal=np.array([0.0, 0.0, -1.0], dtype=np.float64),
        rhs=float(-z_value),
        support_type="cap",
    )


def _cylinder_support(key: str, domain: AnnularCylinderDomain, radius: float) -> CylinderSupport:
    return CylinderSupport(
        key=key,
        center_xy=domain.center_xy.astype(np.float64),
        radius=float(radius),
        support_type="cylinder",
    )


def _box_supports(domain: AnnularCylinderDomain) -> tuple[PlaneSupport, ...]:
    cx, cy = float(domain.center_xy[0]), float(domain.center_xy[1])
    radius = float(domain.outer_radius)
    x_min = cx - radius
    x_max = cx + radius
    y_min = cy - radius
    y_max = cy + radius
    return (
        PlaneSupport("box:x_min", np.array([-1.0, 0.0, 0.0], dtype=np.float64), -x_min, "box"),
        PlaneSupport("box:x_max", np.array([1.0, 0.0, 0.0], dtype=np.float64), x_max, "box"),
        PlaneSupport("box:y_min", np.array([0.0, -1.0, 0.0], dtype=np.float64), -y_min, "box"),
        PlaneSupport("box:y_max", np.array([0.0, 1.0, 0.0], dtype=np.float64), y_max, "box"),
        PlaneSupport("box:z_min", np.array([0.0, 0.0, -1.0], dtype=np.float64), -float(domain.z_min), "box"),
        PlaneSupport("box:z_max", np.array([0.0, 0.0, 1.0], dtype=np.float64), float(domain.z_max), "box"),
    )


def _point_in_cell(cell: ExactRestrictedCell, domain: AnnularCylinderDomain, point: np.ndarray, tol: float = _TOL) -> bool:
    return cell.contains_point(np.asarray(point, dtype=np.float64), domain, tol=tol)


def _point_in_trim_supports(
    point: np.ndarray,
    supports: Iterable[Support],
    domain: AnnularCylinderDomain,
    tol: float = _TOL,
) -> bool:
    point = np.asarray(point, dtype=np.float64)
    if not domain.contains_point(point, tol=tol):
        return False
    for support in supports:
        if isinstance(support, PlaneSupport):
            if float(np.dot(support.normal, point) - support.rhs) > tol:
                return False
            continue
        radial = math.hypot(float(point[0] - support.center_xy[0]), float(point[1] - support.center_xy[1]))
        if support.key == "outer_cylinder":
            if radial - support.radius > tol:
                return False
        elif support.key == "inner_cylinder":
            if support.radius - radial > tol:
                return False
        elif abs(radial - support.radius) > tol:
            return False
    return True


def _point_in_plane_supports(point: np.ndarray, supports: Iterable[PlaneSupport], tol: float = _TOL) -> bool:
    point = np.asarray(point, dtype=np.float64)
    for support in supports:
        if float(np.dot(support.normal, point) - support.rhs) > tol:
            return False
    return True


def _points_in_plane_supports(points: np.ndarray, supports: Iterable[PlaneSupport], tol: float = _TOL) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    supports = tuple(supports)
    if len(points) == 0:
        return np.zeros((0,), dtype=bool)
    if not supports:
        return np.ones(points.shape[0], dtype=bool)
    normals = np.asarray([support.normal for support in supports], dtype=np.float64)
    rhs = np.asarray([support.rhs for support in supports], dtype=np.float64)
    return np.all(points @ normals.T - rhs[None, :] <= tol, axis=1)


def _points_in_trim_supports(points: np.ndarray, supports: Iterable[Support], domain: AnnularCylinderDomain, tol: float = _TOL) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    supports = tuple(supports)
    mask = domain.contains_points(points, tol=tol)
    if not np.any(mask):
        return mask
    plane_supports = tuple(support for support in supports if isinstance(support, PlaneSupport))
    if plane_supports:
        local_points = points[mask]
        local_mask = _points_in_plane_supports(local_points, plane_supports, tol=tol)
        mask[mask] = local_mask
    if not np.any(mask):
        return mask
    for support in supports:
        if isinstance(support, PlaneSupport):
            continue
        active_points = points[mask]
        radial = np.hypot(active_points[:, 0] - support.center_xy[0], active_points[:, 1] - support.center_xy[1])
        if support.key == "outer_cylinder":
            local_mask = radial - support.radius <= tol
        elif support.key == "inner_cylinder":
            local_mask = support.radius - radial <= tol
        else:
            local_mask = np.abs(radial - support.radius) <= tol
        mask[mask] = local_mask
        if not np.any(mask):
            break
    return mask


def _on_support(point: np.ndarray, support: Support, tol: float = 1e-6) -> bool:
    if isinstance(support, PlaneSupport):
        return abs(float(np.dot(support.normal, point) - support.rhs)) <= tol
    radial = math.hypot(float(point[0] - support.center_xy[0]), float(point[1] - support.center_xy[1]))
    return abs(radial - support.radius) <= tol


def _line_from_two_planes(left: PlaneSupport, right: PlaneSupport) -> tuple[np.ndarray, np.ndarray] | None:
    direction = np.cross(left.normal, right.normal)
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return None
    direction = direction / norm
    matrix = np.vstack((left.normal, right.normal, direction))
    rhs = np.array([left.rhs, right.rhs, 0.0], dtype=np.float64)
    try:
        point = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        point, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    return point.astype(np.float64), direction.astype(np.float64)


def _intersect_line_with_cylinder(point: np.ndarray, direction: np.ndarray, cylinder: CylinderSupport) -> list[np.ndarray]:
    px = float(point[0] - cylinder.center_xy[0])
    py = float(point[1] - cylinder.center_xy[1])
    dx = float(direction[0])
    dy = float(direction[1])
    a_value = dx * dx + dy * dy
    b_value = 2.0 * (px * dx + py * dy)
    c_value = px * px + py * py - cylinder.radius * cylinder.radius
    if a_value <= 1e-12:
        return []
    disc = b_value * b_value - 4.0 * a_value * c_value
    if disc < -1e-10:
        return []
    disc = max(0.0, disc)
    root = math.sqrt(disc)
    params = [(-b_value - root) / (2.0 * a_value), (-b_value + root) / (2.0 * a_value)]
    return [(point + value * direction).astype(np.float64) for value in params]


def _solve_triple_intersection(supports: tuple[Support, Support, Support]) -> list[np.ndarray]:
    cylinders = [support for support in supports if isinstance(support, CylinderSupport)]
    planes = [support for support in supports if isinstance(support, PlaneSupport)]
    if len(cylinders) == 0 and len(planes) == 3:
        matrix = np.vstack([plane.normal for plane in planes])
        rhs = np.array([plane.rhs for plane in planes], dtype=np.float64)
        try:
            point = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            return []
        return [point.astype(np.float64)]
    if len(cylinders) == 1 and len(planes) == 2:
        line = _line_from_two_planes(planes[0], planes[1])
        if line is None:
            return []
        return _intersect_line_with_cylinder(line[0], line[1], cylinders[0])
    return []


def _quantize_point(point: np.ndarray, scale: float = 1e8) -> tuple[int, int, int]:
    return tuple(int(round(float(value) * scale)) for value in point)


def _points_on_support(points: np.ndarray, support: Support, tol: float = 1e-6) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if isinstance(support, PlaneSupport):
        return np.abs(points @ support.normal - support.rhs) <= tol
    radial = np.hypot(points[:, 0] - support.center_xy[0], points[:, 1] - support.center_xy[1])
    return np.abs(radial - support.radius) <= tol


@lru_cache(maxsize=None)
def _combination_index_array(size: int, choose: int) -> np.ndarray:
    if choose <= 0 or size < choose:
        return np.empty((0, choose), dtype=np.int32)
    return np.asarray(list(combinations(range(size), choose)), dtype=np.int32)


def _batched_plane_triple_candidates(plane_supports: tuple[PlaneSupport, ...]) -> list[tuple[np.ndarray, tuple[str, ...]]]:
    if len(plane_supports) < 3:
        return []
    triple_indices = _combination_index_array(len(plane_supports), 3)
    if triple_indices.size == 0:
        return []
    normals = np.asarray([support.normal for support in plane_supports], dtype=np.float64)
    rhs_all = np.asarray([support.rhs for support in plane_supports], dtype=np.float64)
    support_keys = tuple(support.key for support in plane_supports)
    matrices = normals[triple_indices]
    rhs = rhs_all[triple_indices]
    valid = np.abs(np.linalg.det(matrices)) > 1e-12
    if not np.any(valid):
        return []
    solved = np.linalg.solve(matrices[valid], rhs[valid][..., None]).squeeze(-1)
    valid_triples = triple_indices[valid]
    return [
        (
            solved[result_index].astype(np.float64),
            tuple(sorted((support_keys[int(triple[0])], support_keys[int(triple[1])], support_keys[int(triple[2])]))),
        )
        for result_index, triple in enumerate(valid_triples)
    ]


def _plane_pair_cylinder_candidates(
    plane_supports: tuple[PlaneSupport, ...],
    cylinder_supports: tuple[CylinderSupport, ...],
) -> list[tuple[np.ndarray, tuple[str, ...]]]:
    if len(plane_supports) < 2 or not cylinder_supports:
        return []

    pair_indices = _combination_index_array(len(plane_supports), 2)
    if pair_indices.size == 0:
        return []

    normals = np.asarray([support.normal for support in plane_supports], dtype=np.float64)
    rhs_all = np.asarray([support.rhs for support in plane_supports], dtype=np.float64)
    support_keys = tuple(support.key for support in plane_supports)

    left_normals = normals[pair_indices[:, 0]]
    right_normals = normals[pair_indices[:, 1]]
    directions = np.cross(left_normals, right_normals)
    norms = np.linalg.norm(directions, axis=1)
    valid_lines = norms > 1e-12
    if not np.any(valid_lines):
        return []

    pair_indices = pair_indices[valid_lines]
    directions = directions[valid_lines] / norms[valid_lines, None]
    matrices = np.stack((left_normals[valid_lines], right_normals[valid_lines], directions), axis=1)
    rhs = np.stack((rhs_all[pair_indices[:, 0]], rhs_all[pair_indices[:, 1]], np.zeros(pair_indices.shape[0], dtype=np.float64)), axis=1)
    try:
        line_points = np.linalg.solve(matrices, rhs[..., None]).squeeze(-1)
    except np.linalg.LinAlgError:
        line_points = (np.linalg.pinv(matrices) @ rhs[..., None]).squeeze(-1)

    candidates: list[tuple[np.ndarray, tuple[str, ...]]] = []
    for cylinder in cylinder_supports:
        px = line_points[:, 0] - float(cylinder.center_xy[0])
        py = line_points[:, 1] - float(cylinder.center_xy[1])
        dx = directions[:, 0]
        dy = directions[:, 1]
        a_value = dx * dx + dy * dy
        b_value = 2.0 * (px * dx + py * dy)
        c_value = px * px + py * py - float(cylinder.radius * cylinder.radius)
        disc = b_value * b_value - 4.0 * a_value * c_value
        valid_intersections = (a_value > 1e-12) & (disc >= -1e-10)
        if not np.any(valid_intersections):
            continue
        disc = np.maximum(disc[valid_intersections], 0.0)
        roots = np.sqrt(disc)
        a_valid = a_value[valid_intersections]
        b_valid = b_value[valid_intersections]
        params = np.stack(
            (
                (-b_valid - roots) / (2.0 * a_valid),
                (-b_valid + roots) / (2.0 * a_valid),
            ),
            axis=1,
        )
        base_points = line_points[valid_intersections]
        base_dirs = directions[valid_intersections]
        pair_valid = pair_indices[valid_intersections]
        intersection_points = base_points[:, None, :] + params[..., None] * base_dirs[:, None, :]
        for pair, points in zip(pair_valid, intersection_points, strict=False):
            support_key_triplet = tuple(sorted((support_keys[int(pair[0])], support_keys[int(pair[1])], cylinder.key)))
            for point in points:
                candidates.append((point.astype(np.float64), support_key_triplet))
    return candidates


def _build_vertices(
    supports: tuple[Support, ...],
    contains_point: Callable[[np.ndarray], bool],
    contains_points: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[ExactVertex, ...]:
    plane_supports = tuple(support for support in supports if isinstance(support, PlaneSupport))
    cylinder_supports = tuple(support for support in supports if isinstance(support, CylinderSupport))
    candidates = _batched_plane_triple_candidates(plane_supports)
    candidates.extend(_plane_pair_cylinder_candidates(plane_supports, cylinder_supports))
    if not candidates:
        return tuple()

    candidate_points = np.asarray([point for point, _ in candidates], dtype=np.float64)
    contains_mask = contains_points(candidate_points) if contains_points is not None else np.asarray([contains_point(point) for point in candidate_points], dtype=bool)

    support_lookup = _support_by_key(supports)
    keyed: dict[tuple[int, int, int], ExactVertex] = {}
    for candidate_index, (point, support_keys) in enumerate(candidates):
        if not contains_mask[candidate_index]:
            continue
        triple_supports = tuple(support_lookup[support_key] for support_key in support_keys)
        if not all(_on_support(point, support) for support in triple_supports):
            continue
        key = _quantize_point(point)
        keyed[key] = ExactVertex(vertex_id=-1, point=point, support_keys=support_keys)

    vertices: list[ExactVertex] = []
    for vertex_id, key in enumerate(sorted(keyed)):
        vertex = keyed[key]
        vertices.append(ExactVertex(vertex_id=vertex_id, point=vertex.point, support_keys=vertex.support_keys))
    return tuple(vertices)


def _support_by_key(supports: tuple[Support, ...]) -> dict[str, Support]:
    return {support.key: support for support in supports}


def _angular_distance(theta_a: float, theta_b: float) -> float:
    delta = (theta_a - theta_b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(delta)



def _pair_midpoint(left: Support, right: Support, first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    if isinstance(left, PlaneSupport) and isinstance(right, PlaneSupport):
        return 0.5 * (first + second)
    if isinstance(left, CylinderSupport) and isinstance(right, PlaneSupport):
        left, right = right, left
    if isinstance(left, PlaneSupport) and isinstance(right, CylinderSupport):
        if abs(left.normal[2]) <= 1e-12:
            return 0.5 * (first + second)
        theta_a = math.atan2(first[1] - right.center_xy[1], first[0] - right.center_xy[0])
        theta_b = math.atan2(second[1] - right.center_xy[1], second[0] - right.center_xy[0])
        options = [
            0.5 * (theta_a + theta_b),
            0.5 * (theta_a + theta_b + 2.0 * math.pi),
            0.5 * (theta_a + theta_b - 2.0 * math.pi),
        ]
        theta_mid = min(
            options,
            key=lambda value: max(_angular_distance(value, theta_a), _angular_distance(value, theta_b)),
        )
        x_coord = right.center_xy[0] + right.radius * math.cos(theta_mid)
        y_coord = right.center_xy[1] + right.radius * math.sin(theta_mid)
        z_coord = (left.rhs - left.normal[0] * x_coord - left.normal[1] * y_coord) / left.normal[2]
        return np.array([x_coord, y_coord, z_coord], dtype=np.float64)
    return 0.5 * (first + second)


def _build_curve(left: Support, right: Support, start: np.ndarray, end: np.ndarray) -> ExactCurve:
    if isinstance(left, PlaneSupport) and isinstance(right, PlaneSupport):
        return ExactLineSegment(kind="line_segment", start=start, end=end)
    if isinstance(left, CylinderSupport) and isinstance(right, PlaneSupport):
        left, right = right, left
    if isinstance(left, PlaneSupport) and isinstance(right, CylinderSupport):
        if right.key in {"outer_cylinder", "inner_cylinder"} and left.key in {"top_cap", "bottom_cap"}:
            center = np.array([right.center_xy[0], right.center_xy[1], 0.0], dtype=np.float64)
            z_value = float(start[2])
            center[2] = z_value
            theta_start = math.atan2(start[1] - right.center_xy[1], start[0] - right.center_xy[0])
            theta_end = math.atan2(end[1] - right.center_xy[1], end[0] - right.center_xy[0])
            return ExactCircleArc(
                kind="circle_arc",
                center=center,
                radius=right.radius,
                normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                start_angle=theta_start,
                end_angle=theta_end,
                z_value=z_value,
                start=start,
                end=end,
            )
        vertical_theta = None
        if abs(left.normal[2]) <= 1e-12:
            vertical_theta = math.atan2(start[1] - right.center_xy[1], start[0] - right.center_xy[0])
            theta_start = vertical_theta
            theta_end = vertical_theta
        else:
            theta_start = math.atan2(start[1] - right.center_xy[1], start[0] - right.center_xy[0])
            theta_end = math.atan2(end[1] - right.center_xy[1], end[0] - right.center_xy[0])
        return ExactCylinderPlaneCurve(
            kind="cylinder_plane_curve",
            cylinder_radius=right.radius,
            cylinder_center_xy=right.center_xy,
            plane_normal=left.normal,
            plane_rhs=left.rhs,
            theta_start=theta_start,
            theta_end=theta_end,
            start=start,
            end=end,
            vertical_theta=vertical_theta,
        )
    return ExactLineSegment(kind="line_segment", start=start, end=end)


def _curve_parameter(curve: ExactCurve, point: np.ndarray) -> float:
    if isinstance(curve, ExactLineSegment):
        direction = curve.end - curve.start
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            return 0.0
        direction = direction / norm
        return float(np.dot(point, direction))
    if isinstance(curve, ExactCircleArc):
        return math.atan2(point[1] - curve.center[1], point[0] - curve.center[0])
    if curve.vertical_theta is not None:
        return float(point[2])
    return math.atan2(point[1] - curve.cylinder_center_xy[1], point[0] - curve.cylinder_center_xy[0])


def _build_edges(
    supports: tuple[Support, ...],
    vertices: tuple[ExactVertex, ...],
    contains_point: Callable[[np.ndarray], bool],
) -> tuple[ExactEdge, ...]:
    support_lookup = _support_by_key(supports)
    pair_to_vertices: dict[tuple[str, str], list[int]] = {}
    for vertex in vertices:
        for left_key, right_key in combinations(vertex.support_keys, 2):
            pair_to_vertices.setdefault(tuple(sorted((left_key, right_key))), []).append(vertex.vertex_id)

    edges: list[ExactEdge] = []
    edge_id = 0
    for pair_key, vertex_ids in sorted(pair_to_vertices.items()):
        unique_vertex_ids = sorted(set(vertex_ids))
        if len(unique_vertex_ids) < 2:
            continue
        left = support_lookup[pair_key[0]]
        right = support_lookup[pair_key[1]]
        prototype = _build_curve(left, right, vertices[unique_vertex_ids[0]].point, vertices[unique_vertex_ids[1]].point)
        params = sorted(((_curve_parameter(prototype, vertices[vertex_id].point), vertex_id) for vertex_id in unique_vertex_ids), key=lambda item: item[0])
        cyclic = isinstance(prototype, ExactCircleArc) or (isinstance(prototype, ExactCylinderPlaneCurve) and prototype.vertical_theta is None)
        intervals: list[tuple[int, int]] = []
        for idx in range(len(params) - 1):
            intervals.append((params[idx][1], params[idx + 1][1]))
        if cyclic and len(params) > 2:
            intervals.append((params[-1][1], params[0][1]))

        for start_id, end_id in intervals:
            start = vertices[start_id].point
            end = vertices[end_id].point
            midpoint = _pair_midpoint(left, right, start, end)
            if midpoint is None or not contains_point(midpoint):
                continue
            edges.append(
                ExactEdge(
                    edge_id=edge_id,
                    support_keys=pair_key,
                    vertex_ids=(start_id, end_id),
                    curve=_build_curve(left, right, start, end),
                )
            )
            edge_id += 1
    return tuple(edges)


def _build_face_loops(support_key: str, edges: tuple[ExactEdge, ...]) -> tuple[tuple[int, ...], ...]:
    face_edges = [edge for edge in edges if support_key in edge.support_keys]
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for edge in face_edges:
        start_id, end_id = edge.vertex_ids
        adjacency.setdefault(start_id, []).append((edge.edge_id, end_id))
        adjacency.setdefault(end_id, []).append((edge.edge_id, start_id))
    unused = {edge.edge_id for edge in face_edges}
    loops: list[tuple[int, ...]] = []
    edge_lookup = {edge.edge_id: edge for edge in face_edges}
    while unused:
        current_edge_id = next(iter(unused))
        current_edge = edge_lookup[current_edge_id]
        start_vertex = current_edge.vertex_ids[0]
        current_vertex = current_edge.vertex_ids[1]
        loop = [current_edge_id]
        unused.remove(current_edge_id)
        while current_vertex != start_vertex:
            options = [(edge_id, other_vertex) for edge_id, other_vertex in adjacency.get(current_vertex, []) if edge_id in unused]
            if not options:
                break
            next_edge_id, next_vertex = options[0]
            loop.append(next_edge_id)
            unused.remove(next_edge_id)
            current_vertex = next_vertex
        loops.append(tuple(loop))
    return tuple(loops)


def _build_polyhedral_faces(seed_id: int, supports: tuple[PlaneSupport, ...], edges: tuple[ExactEdge, ...]) -> tuple[PolyhedralFace, ...]:
    faces: list[PolyhedralFace] = []
    face_id = 0
    for support in supports:
        loops = _build_face_loops(support.key, edges)
        if not loops:
            continue
        faces.append(
            PolyhedralFace(
                face_id=face_id,
                seed_id=seed_id,
                support_key=support.key,
                support_type=support.support_type,
                loop_edge_ids=loops,
                neighbor_seed_id=support.neighbor_seed_id,
            )
        )
        face_id += 1
    return tuple(faces)


def build_polyhedral_voronoi_cell(
    cell: ExactRestrictedCell,
    diagram: ExactRestrictedVoronoiDiagram,
    neighbor_seed_ids: Iterable[int],
) -> PolyhedralVoronoiCell:
    supports = tuple([_plane_support_for_neighbor(cell, diagram, n) for n in neighbor_seed_ids] + list(_box_supports(diagram.domain)))
    contains_point = lambda point: _point_in_plane_supports(point, supports, tol=1e-6)
    contains_points = lambda points: _points_in_plane_supports(points, supports, tol=1e-6)
    vertices = _build_vertices(supports, contains_point, contains_points)
    edges = _build_edges(supports, vertices, contains_point)
    active_support_keys = {support_key for edge in edges for support_key in edge.support_keys}
    active_supports = tuple(support for support in supports if support.key in active_support_keys)
    active_vertices = tuple(
        ExactVertex(vertex_id=vertex.vertex_id, point=vertex.point, support_keys=tuple(key for key in vertex.support_keys if key in active_support_keys))
        for vertex in vertices
        if any(key in active_support_keys for key in vertex.support_keys)
    )
    faces = _build_polyhedral_faces(cell.seed_id, active_supports, edges)
    return PolyhedralVoronoiCell(
        seed_id=cell.seed_id,
        supports=active_supports,
        faces=faces,
        edges=edges,
        vertices=active_vertices,
    )


def _build_trim_supports(polyhedral_cell: PolyhedralVoronoiCell, diagram: ExactRestrictedVoronoiDiagram) -> tuple[Support, ...]:
    supports: list[Support] = [support for support in polyhedral_cell.supports if support.support_type == "plane"]
    supports.append(_cap_support("top_cap", diagram.domain.z_max, inward="down"))
    supports.append(_cap_support("bottom_cap", diagram.domain.z_min, inward="up"))
    supports.append(_cylinder_support("outer_cylinder", diagram.domain, diagram.domain.outer_radius))
    if diagram.domain.inner_radius > 0.0:
        supports.append(_cylinder_support("inner_cylinder", diagram.domain, diagram.domain.inner_radius))
    return tuple(supports)


def _build_exact_faces(seed_id: int, supports: tuple[Support, ...], edges: tuple[ExactEdge, ...]) -> tuple[ExactFace, ...]:
    faces: list[ExactFace] = []
    face_id = 0
    for support in supports:
        loops = _build_face_loops(support.key, edges)
        if not loops:
            continue
        faces.append(
            ExactFace(
                face_id=face_id,
                seed_id=seed_id,
                support_key=support.key,
                support_type=support.support_type,
                loop_edge_ids=loops,
            )
        )
        face_id += 1
    return tuple(faces)


def _build_trim_summary(polyhedral_cell: PolyhedralVoronoiCell, faces: tuple[ExactFace, ...], edges: tuple[ExactEdge, ...], vertices: tuple[ExactVertex, ...]) -> TrimmedCellPatchSet:
    polyhedral_vertex_keys = {_quantize_point(vertex.point) for vertex in polyhedral_cell.vertices}
    kept_plane_face_ids = tuple(sorted(face.face_id for face in faces if face.support_type == "plane"))
    generated_cylinder_face_ids = tuple(sorted(face.face_id for face in faces if face.support_key in {"outer_cylinder", "inner_cylinder"}))
    plane_cylinder_edge_ids = tuple(
        sorted(
            edge.edge_id
            for edge in edges
            if any(key in {"outer_cylinder", "inner_cylinder"} for key in edge.support_keys)
            and any(key.startswith("bisector:") for key in edge.support_keys)
        )
    )
    cap_cylinder_edge_ids = tuple(
        sorted(
            edge.edge_id
            for edge in edges
            if any(key in {"outer_cylinder", "inner_cylinder"} for key in edge.support_keys)
            and any(key in {"top_cap", "bottom_cap"} for key in edge.support_keys)
        )
    )
    new_vertex_ids = tuple(sorted(vertex.vertex_id for vertex in vertices if _quantize_point(vertex.point) not in polyhedral_vertex_keys))
    return TrimmedCellPatchSet(
        kept_plane_face_ids=kept_plane_face_ids,
        generated_cylinder_face_ids=generated_cylinder_face_ids,
        plane_cylinder_edge_ids=plane_cylinder_edge_ids,
        cap_cylinder_edge_ids=cap_cylinder_edge_ids,
        new_vertex_ids=new_vertex_ids,
    )


def trim_polyhedral_cell_with_annular_cylinder(
    polyhedral_cell: PolyhedralVoronoiCell,
    cell: ExactRestrictedCell,
    diagram: ExactRestrictedVoronoiDiagram,
) -> TrimmedAnnularCell:
    supports = _build_trim_supports(polyhedral_cell, diagram)
    contains_point = lambda point: _point_in_trim_supports(point, supports, diagram.domain, tol=1e-6)
    contains_points = lambda points: _points_in_trim_supports(points, supports, diagram.domain, tol=1e-6)
    vertices = _build_vertices(supports, contains_point, contains_points)
    edges = _build_edges(supports, vertices, contains_point)
    faces = _build_exact_faces(cell.seed_id, supports, edges)
    active_support_keys = {face.support_key for face in faces}
    active_supports = tuple(support for support in supports if support.key in active_support_keys)
    trim_summary = _build_trim_summary(polyhedral_cell, vertices=vertices, edges=edges, faces=faces)
    return TrimmedAnnularCell(
        seed_id=cell.seed_id,
        polyhedral_cell=polyhedral_cell,
        supports=active_supports,
        faces=faces,
        edges=edges,
        vertices=vertices,
        trim_summary=trim_summary,
    )


def rebuild_hybrid_exact_brep_from_trimmed_cell(trimmed_cell: TrimmedAnnularCell) -> HybridExactCellBRep:
    return HybridExactCellBRep(
        seed_id=trimmed_cell.seed_id,
        polyhedral_cell=trimmed_cell.polyhedral_cell,
        supports=trimmed_cell.supports,
        faces=trimmed_cell.faces,
        edges=trimmed_cell.edges,
        vertices=trimmed_cell.vertices,
        trim_summary=trimmed_cell.trim_summary,
    )


def build_hybrid_exact_cell_brep(
    cell: ExactRestrictedCell,
    diagram: ExactRestrictedVoronoiDiagram,
    neighbor_seed_ids: Iterable[int],
) -> HybridExactCellBRep:
    polyhedral_cell = build_polyhedral_voronoi_cell(cell, diagram, neighbor_seed_ids)
    trimmed_cell = trim_polyhedral_cell_with_annular_cylinder(polyhedral_cell, cell, diagram)
    return rebuild_hybrid_exact_brep_from_trimmed_cell(trimmed_cell)


def build_hybrid_exact_diagram_brep(
    seed_points: np.ndarray,
    domain: AnnularCylinderDomain,
    seed_ids: Iterable[int] | None = None,
    max_workers: int | None = None,
) -> HybridExactDiagramBRep:
    diagram = build_exact_restricted_voronoi_diagram(seed_points=seed_points, domain=domain, include_support_traces=False)
    selected = [int(seed_id) for seed_id in seed_ids] if seed_ids is not None else list(range(len(diagram.cells)))

    def _build_one(seed_id: int) -> HybridExactCellBRep:
        cell = diagram.cells[seed_id]
        return build_hybrid_exact_cell_brep(cell, diagram, cell.neighboring_seed_ids)

    if max_workers is not None and max_workers > 1 and len(selected) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            cells = tuple(executor.map(_build_one, selected))
    else:
        cells = tuple(_build_one(seed_id) for seed_id in selected)
    return HybridExactDiagramBRep(cells=cells)


def summarize_hybrid_exact_brep(diagram_brep: HybridExactDiagramBRep) -> HybridExactSummary:
    return HybridExactSummary(
        num_cells=len(diagram_brep.cells),
        num_faces=sum(len(cell.faces) for cell in diagram_brep.cells),
        num_edges=sum(len(cell.edges) for cell in diagram_brep.cells),
        num_vertices=sum(len(cell.vertices) for cell in diagram_brep.cells),
    )


def write_hybrid_exact_brep_json(diagram_brep: HybridExactDiagramBRep, output_path: Path) -> None:
    def curve_payload(curve: ExactCurve) -> dict[str, object]:
        if isinstance(curve, ExactLineSegment):
            return {"kind": curve.kind, "start": curve.start.tolist(), "end": curve.end.tolist()}
        if isinstance(curve, ExactCircleArc):
            return {
                "kind": curve.kind,
                "center": curve.center.tolist(),
                "radius": curve.radius,
                "normal": curve.normal.tolist(),
                "start_angle": curve.start_angle,
                "end_angle": curve.end_angle,
                "z_value": curve.z_value,
                "start": curve.start.tolist(),
                "end": curve.end.tolist(),
            }
        return {
            "kind": curve.kind,
            "cylinder_radius": curve.cylinder_radius,
            "cylinder_center_xy": curve.cylinder_center_xy.tolist(),
            "plane_normal": curve.plane_normal.tolist(),
            "plane_rhs": curve.plane_rhs,
            "theta_start": curve.theta_start,
            "theta_end": curve.theta_end,
            "vertical_theta": curve.vertical_theta,
            "start": curve.start.tolist(),
            "end": curve.end.tolist(),
        }

    def support_payload(support: Support) -> dict[str, object]:
        if isinstance(support, PlaneSupport):
            return {
                "key": support.key,
                "type": support.support_type,
                "normal": support.normal.tolist(),
                "rhs": support.rhs,
                "neighbor_seed_id": support.neighbor_seed_id,
            }
        return {
            "key": support.key,
            "type": support.support_type,
            "center_xy": support.center_xy.tolist(),
            "radius": support.radius,
        }

    payload = {
        "cells": [
            {
                "seed_id": cell.seed_id,
                "step1_polyhedral_cell": {
                    "supports": [support_payload(support) for support in cell.polyhedral_cell.supports],
                    "faces": [
                        {
                            "face_id": face.face_id,
                            "support_key": face.support_key,
                            "support_type": face.support_type,
                            "neighbor_seed_id": face.neighbor_seed_id,
                            "loop_edge_ids": [list(loop) for loop in face.loop_edge_ids],
                        }
                        for face in cell.polyhedral_cell.faces
                    ],
                    "edges": [
                        {
                            "edge_id": edge.edge_id,
                            "support_keys": list(edge.support_keys),
                            "vertex_ids": list(edge.vertex_ids),
                            "curve": curve_payload(edge.curve),
                        }
                        for edge in cell.polyhedral_cell.edges
                    ],
                    "vertices": [
                        {
                            "vertex_id": vertex.vertex_id,
                            "point": vertex.point.tolist(),
                            "support_keys": list(vertex.support_keys),
                        }
                        for vertex in cell.polyhedral_cell.vertices
                    ],
                },
                "step2_trimmed_patch_summary": {
                    "kept_plane_face_ids": list(cell.trim_summary.kept_plane_face_ids),
                    "generated_cylinder_face_ids": list(cell.trim_summary.generated_cylinder_face_ids),
                    "plane_cylinder_edge_ids": list(cell.trim_summary.plane_cylinder_edge_ids),
                    "cap_cylinder_edge_ids": list(cell.trim_summary.cap_cylinder_edge_ids),
                    "new_vertex_ids": list(cell.trim_summary.new_vertex_ids),
                },
                "step3_hybrid_brep": {
                    "supports": [support_payload(support) for support in cell.supports],
                    "faces": [
                        {
                            "face_id": face.face_id,
                            "support_key": face.support_key,
                            "support_type": face.support_type,
                            "loop_edge_ids": [list(loop) for loop in face.loop_edge_ids],
                        }
                        for face in cell.faces
                    ],
                    "edges": [
                        {
                            "edge_id": edge.edge_id,
                            "support_keys": list(edge.support_keys),
                            "vertex_ids": list(edge.vertex_ids),
                            "curve": curve_payload(edge.curve),
                        }
                        for edge in cell.edges
                    ],
                    "vertices": [
                        {
                            "vertex_id": vertex.vertex_id,
                            "point": vertex.point.tolist(),
                            "support_keys": list(vertex.support_keys),
                        }
                        for vertex in cell.vertices
                    ],
                },
            }
            for cell in diagram_brep.cells
        ]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
