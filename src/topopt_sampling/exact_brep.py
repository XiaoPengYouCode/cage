from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from topopt_sampling.exact_restricted_voronoi_3d import (
    AnnularCylinderDomain,
    ExactRestrictedCell,
    ExactRestrictedVoronoiDiagram,
    build_exact_restricted_voronoi_diagram,
)


@dataclass(frozen=True)
class FaceLoop3D:
    points: np.ndarray


@dataclass(frozen=True)
class BRepFace:
    face_id: int
    seed_id: int
    support_type: str
    support_key: str
    loops: tuple[FaceLoop3D, ...]
    neighbor_seed_id: int | None = None


@dataclass(frozen=True)
class BRepEdge:
    edge_id: int
    points: np.ndarray
    face_ids: tuple[int, ...]


@dataclass(frozen=True)
class BRepVertex:
    vertex_id: int
    point: np.ndarray
    incident_edge_ids: tuple[int, ...]


@dataclass(frozen=True)
class CellBRep:
    seed_id: int
    faces: tuple[BRepFace, ...]
    edges: tuple[BRepEdge, ...]
    vertices: tuple[BRepVertex, ...]
    unmatched_loop_count: int


@dataclass(frozen=True)
class DiagramBRep:
    cells: tuple[CellBRep, ...]


@dataclass(frozen=True)
class DiagramBRepSummary:
    num_cells: int
    num_faces: int
    num_edges: int
    num_vertices: int
    unmatched_loops: int


def _extract_contours(x_grid: np.ndarray, y_grid: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    if not np.any(mask):
        return []
    figure, axis = plt.subplots()
    try:
        contour = axis.contour(x_grid, y_grid, mask.astype(np.float64), levels=[0.5])
        loops = [segment.astype(np.float64) for segment in contour.allsegs[0] if len(segment) >= 3]
    finally:
        plt.close(figure)
    return loops


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _close_loop(points: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    if len(points) == 0:
        return points
    if np.linalg.norm(points[0] - points[-1]) <= tol:
        return points
    return np.vstack((points, points[0]))


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal / np.linalg.norm(normal)
    if abs(normal[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u_vec = np.cross(normal, ref)
    u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(normal, u_vec)
    v_vec /= np.linalg.norm(v_vec)
    return u_vec, v_vec


def _plane_point(seed_i: np.ndarray, seed_j: np.ndarray) -> np.ndarray:
    return 0.5 * (seed_i + seed_j)


def _sample_plane_face_loops(
    cell: ExactRestrictedCell,
    diagram: ExactRestrictedVoronoiDiagram,
    neighbor_seed_id: int,
    samples: int = 220,
) -> tuple[FaceLoop3D, ...]:
    seed_i = diagram.seed_points[cell.seed_id]
    seed_j = diagram.seed_points[int(neighbor_seed_id)]
    normal = seed_j - seed_i
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-12:
        return tuple()

    u_vec, v_vec = _plane_basis(normal)
    origin = _plane_point(seed_i, seed_j)
    half_extent = max(diagram.domain.outer_radius * 1.25, (diagram.domain.z_max - diagram.domain.z_min) * 1.25)
    u_values = np.linspace(-half_extent, half_extent, samples, dtype=np.float64)
    v_values = np.linspace(-half_extent, half_extent, samples, dtype=np.float64)
    u_grid, v_grid = np.meshgrid(u_values, v_values, indexing="xy")
    points = origin + u_grid[..., None] * u_vec + v_grid[..., None] * v_vec
    flat_points = points.reshape(-1, 3)
    mask = cell.contains_points(flat_points, diagram.domain).reshape(samples, samples)
    loops_2d = _extract_contours(u_grid, v_grid, mask)
    loops_3d: list[FaceLoop3D] = []
    for loop in loops_2d:
        points_3d = origin + loop[:, [0]] * u_vec + loop[:, [1]] * v_vec
        points_3d = _close_loop(points_3d)
        if _polyline_length(points_3d) > 1e-3:
            loops_3d.append(FaceLoop3D(points=points_3d.astype(np.float64)))
    return tuple(loops_3d)


def _sample_cap_face_loops(
    cell: ExactRestrictedCell,
    diagram: ExactRestrictedVoronoiDiagram,
    z_value: float,
    samples_theta: int = 360,
    samples_r: int = 180,
) -> tuple[FaceLoop3D, ...]:
    theta_values = np.linspace(0.0, 2.0 * np.pi, samples_theta, endpoint=False, dtype=np.float64)
    radial_values = np.linspace(diagram.domain.inner_radius, diagram.domain.outer_radius, samples_r, dtype=np.float64)
    theta_grid, radial_grid = np.meshgrid(theta_values, radial_values, indexing="xy")
    x_grid = diagram.domain.center_xy[0] + radial_grid * np.cos(theta_grid)
    y_grid = diagram.domain.center_xy[1] + radial_grid * np.sin(theta_grid)
    z_grid = np.full_like(x_grid, z_value)
    flat_points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
    mask = cell.contains_points(flat_points, diagram.domain).reshape(radial_grid.shape)
    loops_2d = _extract_contours(x_grid, y_grid, mask)
    loops_3d: list[FaceLoop3D] = []
    for loop in loops_2d:
        points_3d = np.column_stack((loop[:, 0], loop[:, 1], np.full(loop.shape[0], z_value, dtype=np.float64)))
        points_3d = _close_loop(points_3d)
        if _polyline_length(points_3d) > 1e-3:
            loops_3d.append(FaceLoop3D(points=points_3d.astype(np.float64)))
    return tuple(loops_3d)


def _sample_cylinder_face_loops(
    cell: ExactRestrictedCell,
    diagram: ExactRestrictedVoronoiDiagram,
    radius: float,
    surface_name: str,
    samples_theta: int = 540,
    samples_z: int = 220,
) -> tuple[FaceLoop3D, ...]:
    theta_values = np.linspace(0.0, 4.0 * np.pi, samples_theta * 2, endpoint=False, dtype=np.float64)
    z_values = np.linspace(diagram.domain.z_min, diagram.domain.z_max, samples_z, dtype=np.float64)
    theta_grid, z_grid = np.meshgrid(theta_values, z_values, indexing="xy")
    x_grid = diagram.domain.center_xy[0] + radius * np.cos(theta_grid)
    y_grid = diagram.domain.center_xy[1] + radius * np.sin(theta_grid)
    flat_points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
    mask = cell.contains_points(flat_points, diagram.domain).reshape(theta_grid.shape)
    loops_2d = _extract_contours(theta_grid, z_grid, mask)
    loops_3d: list[FaceLoop3D] = []
    for loop in loops_2d:
        mean_theta = float(np.mean(loop[:, 0]))
        if not (np.pi <= mean_theta <= 3.0 * np.pi):
            continue
        theta_local = np.mod(loop[:, 0], 2.0 * np.pi)
        order = np.arange(loop.shape[0])
        xyz = np.column_stack(
            (
                diagram.domain.center_xy[0] + radius * np.cos(theta_local[order]),
                diagram.domain.center_xy[1] + radius * np.sin(theta_local[order]),
                loop[:, 1][order],
            )
        )
        xyz = _close_loop(xyz)
        if _polyline_length(xyz) > 1e-3:
            loops_3d.append(FaceLoop3D(points=xyz.astype(np.float64)))
    return tuple(loops_3d)


def build_cell_brep(
    cell: ExactRestrictedCell,
    diagram: ExactRestrictedVoronoiDiagram,
    neighbor_seed_ids: Iterable[int],
) -> CellBRep:
    faces: list[BRepFace] = []
    face_id = 0

    for neighbor_seed_id in neighbor_seed_ids:
        loops = _sample_plane_face_loops(cell=cell, diagram=diagram, neighbor_seed_id=int(neighbor_seed_id))
        if loops:
            faces.append(
                BRepFace(
                    face_id=face_id,
                    seed_id=cell.seed_id,
                    support_type="plane",
                    support_key=f"bisector:{cell.seed_id}:{int(neighbor_seed_id)}",
                    loops=loops,
                    neighbor_seed_id=int(neighbor_seed_id),
                )
            )
            face_id += 1

    top_loops = _sample_cap_face_loops(cell=cell, diagram=diagram, z_value=diagram.domain.z_max)
    if top_loops:
        faces.append(
            BRepFace(
                face_id=face_id,
                seed_id=cell.seed_id,
                support_type="cap",
                support_key="top_cap",
                loops=top_loops,
            )
        )
        face_id += 1

    bottom_loops = _sample_cap_face_loops(cell=cell, diagram=diagram, z_value=diagram.domain.z_min)
    if bottom_loops:
        faces.append(
            BRepFace(
                face_id=face_id,
                seed_id=cell.seed_id,
                support_type="cap",
                support_key="bottom_cap",
                loops=bottom_loops,
            )
        )
        face_id += 1

    outer_loops = _sample_cylinder_face_loops(
        cell=cell,
        diagram=diagram,
        radius=diagram.domain.outer_radius,
        surface_name="outer_cylinder",
    )
    if outer_loops:
        faces.append(
            BRepFace(
                face_id=face_id,
                seed_id=cell.seed_id,
                support_type="cylinder",
                support_key="outer_cylinder",
                loops=outer_loops,
            )
        )
        face_id += 1

    if diagram.domain.inner_radius > 0.0:
        inner_loops = _sample_cylinder_face_loops(
            cell=cell,
            diagram=diagram,
            radius=diagram.domain.inner_radius,
            surface_name="inner_cylinder",
        )
        if inner_loops:
            faces.append(
                BRepFace(
                    face_id=face_id,
                    seed_id=cell.seed_id,
                    support_type="cylinder",
                    support_key="inner_cylinder",
                    loops=inner_loops,
                )
            )
            face_id += 1

    edges, unmatched_loop_count = _pair_face_loops_into_edges(faces)
    vertices = _build_vertices_from_edges(edges)
    return CellBRep(
        seed_id=cell.seed_id,
        faces=tuple(faces),
        edges=tuple(edges),
        vertices=tuple(vertices),
        unmatched_loop_count=unmatched_loop_count,
    )


def _resample_polyline(points: np.ndarray, num_points: int = 64) -> np.ndarray:
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total = float(seg_lengths.sum())
    if total <= 1e-12:
        return np.repeat(points[:1], num_points, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    targets = np.linspace(0.0, total, num_points, dtype=np.float64)
    result = np.empty((num_points, 3), dtype=np.float64)
    seg_index = 0
    for idx, target in enumerate(targets):
        while seg_index + 1 < len(cumulative) and cumulative[seg_index + 1] < target:
            seg_index += 1
        next_index = min(seg_index + 1, len(points) - 1)
        length = cumulative[next_index] - cumulative[seg_index]
        if length <= 1e-12:
            result[idx] = points[seg_index]
            continue
        alpha = (target - cumulative[seg_index]) / length
        result[idx] = (1.0 - alpha) * points[seg_index] + alpha * points[next_index]
    return result


def _polyline_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_s = _resample_polyline(lhs)
    rhs_s = _resample_polyline(rhs)
    tree_rhs = cKDTree(rhs_s)
    dist_lr, _ = tree_rhs.query(lhs_s, k=1)
    tree_lhs = cKDTree(lhs_s)
    dist_rl, _ = tree_lhs.query(rhs_s, k=1)
    return float(max(dist_lr.mean(), dist_rl.mean()))


def _pair_face_loops_into_edges(faces: Iterable[BRepFace], tol: float = 0.75) -> tuple[list[BRepEdge], int]:
    loop_records: list[tuple[int, np.ndarray]] = []
    for face in faces:
        for loop in face.loops:
            loop_records.append((face.face_id, loop.points))

    used = [False] * len(loop_records)
    edges: list[BRepEdge] = []
    edge_id = 0
    unmatched = 0
    for idx, (face_id, points) in enumerate(loop_records):
        if used[idx]:
            continue
        used[idx] = True
        best_match = None
        best_distance = math.inf
        for jdx in range(idx + 1, len(loop_records)):
            other_face_id, other_points = loop_records[jdx]
            if used[jdx] or other_face_id == face_id:
                continue
            distance = _polyline_distance(points, other_points)
            if distance < tol and distance < best_distance:
                best_match = jdx
                best_distance = distance
        if best_match is None:
            edges.append(BRepEdge(edge_id=edge_id, points=points, face_ids=(face_id,)))
            unmatched += 1
        else:
            used[best_match] = True
            other_face_id, other_points = loop_records[best_match]
            chosen = points if _polyline_length(points) >= _polyline_length(other_points) else other_points
            edges.append(BRepEdge(edge_id=edge_id, points=chosen, face_ids=(face_id, other_face_id)))
        edge_id += 1
    return edges, unmatched


def _quantize_point(point: np.ndarray, scale: float = 1e5) -> tuple[int, int, int]:
    return tuple(int(round(float(value) * scale)) for value in point)


def _build_vertices_from_edges(edges: Iterable[BRepEdge], tol_scale: float = 1e5) -> list[BRepVertex]:
    vertex_to_edges: dict[tuple[int, int, int], set[int]] = {}
    vertex_points: dict[tuple[int, int, int], np.ndarray] = {}
    for edge in edges:
        if len(edge.points) == 0:
            continue
        for point in (edge.points[0], edge.points[-1]):
            key = _quantize_point(point, scale=tol_scale)
            vertex_to_edges.setdefault(key, set()).add(edge.edge_id)
            vertex_points.setdefault(key, point)
    vertices: list[BRepVertex] = []
    for vertex_id, key in enumerate(sorted(vertex_to_edges)):
        vertices.append(
            BRepVertex(
                vertex_id=vertex_id,
                point=vertex_points[key],
                incident_edge_ids=tuple(sorted(vertex_to_edges[key])),
            )
        )
    return vertices


def build_diagram_brep(
    seed_points: np.ndarray,
    domain: AnnularCylinderDomain,
    seed_ids: Iterable[int] | None = None,
) -> DiagramBRep:
    diagram = build_exact_restricted_voronoi_diagram(seed_points=seed_points, domain=domain)
    selected = list(seed_ids) if seed_ids is not None else list(range(len(diagram.cells)))
    cells = tuple(
        build_cell_brep(
            cell=diagram.cells[int(seed_id)],
            diagram=diagram,
            neighbor_seed_ids=diagram.cells[int(seed_id)].neighboring_seed_ids,
        )
        for seed_id in selected
    )
    return DiagramBRep(cells=cells)


def summarize_diagram_brep(diagram_brep: DiagramBRep) -> DiagramBRepSummary:
    return DiagramBRepSummary(
        num_cells=len(diagram_brep.cells),
        num_faces=sum(len(cell.faces) for cell in diagram_brep.cells),
        num_edges=sum(len(cell.edges) for cell in diagram_brep.cells),
        num_vertices=sum(len(cell.vertices) for cell in diagram_brep.cells),
        unmatched_loops=sum(cell.unmatched_loop_count for cell in diagram_brep.cells),
    )


def write_diagram_brep_json(diagram_brep: DiagramBRep, output_path: Path) -> None:
    payload = {
        "cells": [
            {
                "seed_id": cell.seed_id,
                "unmatched_loop_count": cell.unmatched_loop_count,
                "faces": [
                    {
                        "face_id": face.face_id,
                        "support_type": face.support_type,
                        "support_key": face.support_key,
                        "neighbor_seed_id": face.neighbor_seed_id,
                        "loops": [loop.points.tolist() for loop in face.loops],
                    }
                    for face in cell.faces
                ],
                "edges": [
                    {
                        "edge_id": edge.edge_id,
                        "face_ids": list(edge.face_ids),
                        "points": edge.points.tolist(),
                    }
                    for edge in cell.edges
                ],
                "vertices": [
                    {
                        "vertex_id": vertex.vertex_id,
                        "point": vertex.point.tolist(),
                        "incident_edge_ids": list(vertex.incident_edge_ids),
                    }
                    for vertex in cell.vertices
                ],
            }
            for cell in diagram_brep.cells
        ]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
