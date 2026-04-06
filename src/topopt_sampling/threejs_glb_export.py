from __future__ import annotations

from dataclasses import dataclass
import json
import math
import struct
from pathlib import Path

import matplotlib.path as mpath
import matplotlib.tri as mtri
import numpy as np
from matplotlib import colors as mcolors

from topopt_sampling.exact_restricted_voronoi_3d import AnnularCylinderDomain, ExactRestrictedVoronoiDiagram, build_exact_restricted_voronoi_diagram
from topopt_sampling.hybrid_exact_brep import (
    ExactCircleArc,
    ExactCylinderPlaneCurve,
    ExactLineSegment,
    HybridExactCellBRep,
    HybridExactDiagramBRep,
    build_hybrid_exact_cell_brep,
)

SHELL_SUPPORT_KEYS = {"outer_cylinder", "inner_cylinder", "top_cap", "bottom_cap"}
CYLINDER_THETA_GRID_SAMPLES = 144
CYLINDER_Z_GRID_SAMPLES = 40
CYLINDER_THETA_GRID_MIN = 16
CYLINDER_THETA_GRID_MAX = 144
CYLINDER_Z_GRID_MIN = 8
CYLINDER_Z_GRID_MAX = 40
POLYGON_GRID_MIN = 8
POLYGON_GRID_MAX = 20
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_MODE_LINES = 1
_MODE_TRIANGLES = 4
_COMPONENT_FLOAT = 5126
_COMPONENT_UINT = 5125


@dataclass(frozen=True)
class ThreeJSGLBExportSummary:
    num_cells: int
    num_shell_cells: int
    num_exported_cells: int
    num_faces: int
    num_triangles: int
    num_boundaries: int
    output_bytes: int = 0


class _GLBBuilder:
    def __init__(self) -> None:
        self._binary = bytearray()
        self.buffer_views: list[dict[str, object]] = []
        self.accessors: list[dict[str, object]] = []
        self.meshes: list[dict[str, object]] = []
        self.nodes: list[dict[str, object]] = []
        self.materials: list[dict[str, object]] = []

    @property
    def binary_blob(self) -> bytes:
        return bytes(self._binary)

    def _align(self, alignment: int = 4) -> None:
        while len(self._binary) % alignment != 0:
            self._binary.append(0)

    def add_blob(self, blob: bytes, target: int | None = None) -> int:
        self._align(4)
        offset = len(self._binary)
        self._binary.extend(blob)
        view: dict[str, object] = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(blob),
        }
        if target is not None:
            view["target"] = target
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def add_accessor(self, array: np.ndarray, type_name: str, component_type: int, target: int | None = None) -> int:
        contiguous = np.ascontiguousarray(array)
        view_index = self.add_blob(contiguous.tobytes(), target=target)
        accessor: dict[str, object] = {
            "bufferView": view_index,
            "componentType": component_type,
            "count": int(contiguous.shape[0]),
            "type": type_name,
        }
        if contiguous.ndim == 1:
            accessor["min"] = [float(np.min(contiguous))]
            accessor["max"] = [float(np.max(contiguous))]
        else:
            accessor["min"] = contiguous.min(axis=0).astype(float).tolist()
            accessor["max"] = contiguous.max(axis=0).astype(float).tolist()
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def add_material(self, color: tuple[float, float, float], double_sided: bool = True, unlit: bool = False) -> int:
        material: dict[str, object] = {
            "pbrMetallicRoughness": {
                "baseColorFactor": [float(color[0]), float(color[1]), float(color[2]), 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.95,
            },
            "doubleSided": bool(double_sided),
        }
        if unlit:
            material["extensions"] = {"KHR_materials_unlit": {}}
        self.materials.append(material)
        return len(self.materials) - 1

    def add_mesh(self, primitives: list[dict[str, object]], name: str) -> int:
        self.meshes.append({"name": name, "primitives": primitives})
        return len(self.meshes) - 1

    def add_node(self, node: dict[str, object]) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1

    def to_glb_bytes(self, scene_nodes: list[int]) -> bytes:
        gltf = {
            "asset": {"version": "2.0", "generator": "topopt_sampling.threejs_glb_export"},
            "extensionsUsed": ["KHR_materials_unlit"],
            "buffers": [{"byteLength": len(self._binary)}],
            "bufferViews": self.buffer_views,
            "accessors": self.accessors,
            "materials": self.materials,
            "meshes": self.meshes,
            "nodes": self.nodes,
            "scenes": [{"nodes": scene_nodes}],
            "scene": 0,
        }
        json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        while len(json_bytes) % 4 != 0:
            json_bytes += b" "
        bin_bytes = self.binary_blob
        while len(bin_bytes) % 4 != 0:
            bin_bytes += b"\x00"
        total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
        return b"".join(
            [
                struct.pack("<4sII", b"glTF", 2, total_length),
                struct.pack("<I4s", len(json_bytes), b"JSON"),
                json_bytes,
                struct.pack("<I4s", len(bin_bytes), b"BIN\x00"),
                bin_bytes,
            ]
        )


def scientific_block_color(seed_id: int) -> tuple[float, float, float]:
    hue = (seed_id * 0.6180339887498949) % 1.0
    sat = 0.42 + 0.12 * (((seed_id * 7) % 11) / 10.0)
    val = 0.72 + 0.12 * (((seed_id * 13) % 9) / 8.0)
    red, green, blue = mcolors.hsv_to_rgb((hue, sat, val))
    return float(red), float(green), float(blue)


def sample_curve(curve: object, num: int = 24) -> np.ndarray:
    if isinstance(curve, ExactLineSegment):
        return np.vstack((curve.start, curve.end))
    if isinstance(curve, ExactCircleArc):
        start = curve.start_angle
        end = curve.end_angle
        delta = end - start
        if delta > math.pi:
            end -= 2.0 * math.pi
        elif delta < -math.pi:
            end += 2.0 * math.pi
        theta = np.linspace(start, end, num)
        return np.column_stack(
            (
                curve.center[0] + curve.radius * np.cos(theta),
                curve.center[1] + curve.radius * np.sin(theta),
                np.full_like(theta, curve.z_value),
            )
        )
    if isinstance(curve, ExactCylinderPlaneCurve):
        if curve.vertical_theta is not None:
            z_values = np.linspace(curve.start[2], curve.end[2], num)
            theta = np.full_like(z_values, curve.vertical_theta)
        else:
            start = curve.theta_start
            end = curve.theta_end
            delta = end - start
            if delta > math.pi:
                end -= 2.0 * math.pi
            elif delta < -math.pi:
                end += 2.0 * math.pi
            theta = np.linspace(start, end, num)
            z_values = (
                curve.plane_rhs
                - curve.plane_normal[0] * (curve.cylinder_center_xy[0] + curve.cylinder_radius * np.cos(theta))
                - curve.plane_normal[1] * (curve.cylinder_center_xy[1] + curve.cylinder_radius * np.sin(theta))
            ) / curve.plane_normal[2]
        return np.column_stack(
            (
                curve.cylinder_center_xy[0] + curve.cylinder_radius * np.cos(theta),
                curve.cylinder_center_xy[1] + curve.cylinder_radius * np.sin(theta),
                z_values,
            )
        )
    raise TypeError(f"Unsupported curve type: {type(curve)!r}")


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal / np.linalg.norm(normal)
    reference = np.array([0.0, 0.0, 1.0], dtype=float) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=float)
    u_vec = np.cross(normal, reference)
    u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(normal, u_vec)
    v_vec /= np.linalg.norm(v_vec)
    return u_vec, v_vec


def ordered_loop_points(face, edge_lookup: dict[int, object], seam_points_lookup: dict[int, np.ndarray] | None = None) -> list[np.ndarray]:
    loops: list[np.ndarray] = []
    for loop_edge_ids in face.loop_edge_ids:
        point_parts: list[np.ndarray] = []
        last_point: np.ndarray | None = None
        for edge_id in loop_edge_ids:
            edge = edge_lookup[edge_id]
            points = seam_points_lookup[edge_id].copy() if seam_points_lookup is not None and edge_id in seam_points_lookup else sample_curve(edge.curve)
            if last_point is not None:
                if np.linalg.norm(points[-1] - last_point) < np.linalg.norm(points[0] - last_point):
                    points = points[::-1]
            if point_parts:
                points = points[1:]
            point_parts.append(points)
            last_point = points[-1]
        loop = np.vstack(point_parts)
        if np.linalg.norm(loop[0] - loop[-1]) > 1e-6:
            loop = np.vstack((loop, loop[0]))
        loops.append(loop)
    return loops


def _adaptive_polygon_grid_samples(points_2d: np.ndarray, holes_2d: list[np.ndarray] | None = None) -> int:
    holes_2d = holes_2d or []
    all_points = [points_2d[:, :2]] + [hole[:, :2] for hole in holes_2d]
    stacked = np.vstack(all_points)
    span = stacked.max(axis=0) - stacked.min(axis=0)
    span_score = max(float(np.max(span)), 0.0)
    boundary_points = int(sum(loop.shape[0] for loop in all_points))
    sample_count = int(max(POLYGON_GRID_MIN, min(POLYGON_GRID_MAX, math.ceil(span_score / 8.0) + math.ceil(math.sqrt(max(boundary_points, 1))) // 2)))
    return sample_count


def triangulate_polygon_2d(points_2d: np.ndarray, holes_2d: list[np.ndarray] | None = None, samples: int | None = None):
    holes_2d = holes_2d or []
    outer_path = mpath.Path(points_2d[:, :2])
    all_points = [points_2d[:, :2]] + [hole[:, :2] for hole in holes_2d]
    stacked = np.vstack(all_points)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    if np.any(maxs - mins < 1e-8):
        return None, None
    sample_count = _adaptive_polygon_grid_samples(points_2d, holes_2d) if samples is None else int(samples)
    x_values = np.linspace(mins[0], maxs[0], sample_count)
    y_values = np.linspace(mins[1], maxs[1], sample_count)
    grid = np.array(np.meshgrid(x_values, y_values, indexing="xy")).reshape(2, -1).T
    mask = outer_path.contains_points(grid)
    for hole in holes_2d:
        mask &= ~mpath.Path(hole[:, :2]).contains_points(grid)
    interior = grid[mask]
    points = np.vstack((stacked, interior))
    try:
        triangulation = mtri.Triangulation(points[:, 0], points[:, 1])
    except Exception:
        jitter = 1e-9 * np.arange(points.shape[0], dtype=np.float64)
        triangulation = mtri.Triangulation(points[:, 0] + jitter, points[:, 1] - jitter)
    centers = np.column_stack((points[triangulation.triangles].mean(axis=1)[:, 0], points[triangulation.triangles].mean(axis=1)[:, 1]))
    triangle_mask = ~outer_path.contains_points(centers)
    for hole in holes_2d:
        triangle_mask |= mpath.Path(hole[:, :2]).contains_points(centers)
    triangulation.set_mask(triangle_mask)
    return points, triangulation


def plane_face_triangles(
    face,
    edge_lookup: dict[int, object],
    support_lookup: dict[str, object],
    seam_points_lookup: dict[int, np.ndarray] | None = None,
) -> list[np.ndarray]:
    loops = ordered_loop_points(face, edge_lookup, seam_points_lookup)
    if not loops:
        return []
    support = support_lookup[face.support_key]
    origin = loops[0][0]
    if face.support_type == "cap":
        outer_2d = loops[0][:, :2]
        holes_2d = [loop[:, :2] for loop in loops[1:]]
        points_2d, triangulation = triangulate_polygon_2d(outer_2d, holes_2d)
        if triangulation is None:
            return []
        z_value = loops[0][0, 2]
        points_3d = np.column_stack((points_2d[:, 0], points_2d[:, 1], np.full(len(points_2d), z_value)))
    else:
        u_vec, v_vec = plane_basis(support.normal)
        loops_uv = []
        for loop in loops:
            relative = loop - origin
            loops_uv.append(np.column_stack((relative @ u_vec, relative @ v_vec)))
        points_2d, triangulation = triangulate_polygon_2d(loops_uv[0], loops_uv[1:])
        if triangulation is None:
            return []
        points_3d = origin + points_2d[:, [0]] * u_vec + points_2d[:, [1]] * v_vec
    return [points_3d[np.asarray(indexes, dtype=int)] for indexes in triangulation.get_masked_triangles()]


def _cylinder_loops_tz_with_seam(loops: list[np.ndarray], center_x: float, center_y: float, seam_theta: float) -> list[np.ndarray]:
    loops_tz: list[np.ndarray] = []
    theta_values: list[np.ndarray] = []
    for loop in loops:
        theta = np.mod(np.arctan2(loop[:, 1] - center_y, loop[:, 0] - center_x) - seam_theta, 2.0 * np.pi)
        theta_values.append(theta)
    all_theta = np.concatenate(theta_values) if theta_values else np.array([], dtype=np.float64)
    seam_crossing = bool(all_theta.size and (np.max(all_theta) - np.min(all_theta) > np.pi))
    for loop, theta in zip(loops, theta_values, strict=False):
        if seam_crossing:
            theta = np.where(theta < np.pi, theta + 2.0 * np.pi, theta)
        loops_tz.append(np.column_stack((theta, loop[:, 2])))
    return loops_tz


def _select_best_cylinder_atlas(loops: list[np.ndarray], center_x: float, center_y: float) -> tuple[list[np.ndarray], float]:
    candidates = []
    for seam_theta in (0.0, np.pi):
        loops_tz = _cylinder_loops_tz_with_seam(loops, center_x, center_y, seam_theta)
        stacked = np.vstack(loops_tz)
        theta_span = float(np.max(stacked[:, 0]) - np.min(stacked[:, 0]))
        candidates.append((theta_span, seam_theta, loops_tz))
    candidates.sort(key=lambda item: item[0])
    _, seam_theta, loops_tz = candidates[0]
    return loops_tz, seam_theta


def _adaptive_cylinder_grid_samples(theta_min: float, theta_max: float, z_min: float, z_max: float) -> tuple[int, int]:
    theta_span = max(0.0, float(theta_max - theta_min))
    z_span = max(0.0, float(z_max - z_min))
    theta_samples = int(np.clip(math.ceil(theta_span / (2.0 * math.pi) * CYLINDER_THETA_GRID_SAMPLES), CYLINDER_THETA_GRID_MIN, CYLINDER_THETA_GRID_MAX))
    z_samples = int(np.clip(math.ceil(z_span / 8.0), CYLINDER_Z_GRID_MIN, CYLINDER_Z_GRID_MAX))
    return theta_samples, z_samples


def cylinder_face_triangles(
    face,
    edge_lookup: dict[int, object],
    support_lookup: dict[str, object],
    domain: AnnularCylinderDomain,
    seam_points_lookup: dict[int, np.ndarray] | None = None,
) -> list[np.ndarray]:
    loops = ordered_loop_points(face, edge_lookup, seam_points_lookup)
    if not loops:
        return []
    support = support_lookup[face.support_key]
    center_x, center_y = float(support.center_xy[0]), float(support.center_xy[1])
    loops_tz, seam_theta = _select_best_cylinder_atlas(loops, center_x, center_y)
    stacked = np.vstack(loops_tz)
    theta_min = float(np.min(stacked[:, 0]))
    theta_max = float(np.max(stacked[:, 0]))
    z_min = float(domain.z_min)
    z_max = float(domain.z_max)

    theta_samples, z_samples = _adaptive_cylinder_grid_samples(theta_min, theta_max, z_min, z_max)
    base_theta = np.linspace(0.0, 2.0 * np.pi, theta_samples, endpoint=True, dtype=np.float64)
    theta_grid = np.concatenate((base_theta, base_theta[1:] + 2.0 * np.pi))
    theta_grid = theta_grid[(theta_grid >= theta_min - 1e-9) & (theta_grid <= theta_max + 1e-9)]
    z_grid = np.linspace(z_min, z_max, z_samples, endpoint=True, dtype=np.float64)
    theta_mesh, z_mesh = np.meshgrid(theta_grid, z_grid, indexing="xy")
    interior_grid = np.column_stack((theta_mesh.ravel(), z_mesh.ravel()))

    outer_path = mpath.Path(loops_tz[0][:, :2])
    interior_mask = outer_path.contains_points(interior_grid)
    for hole in loops_tz[1:]:
        interior_mask &= ~mpath.Path(hole[:, :2]).contains_points(interior_grid)
    clipped_interior = interior_grid[interior_mask]

    polygon_points = [loop[:, :2] for loop in loops_tz]
    points_2d = np.vstack(polygon_points + [clipped_interior]) if len(clipped_interior) else np.vstack(polygon_points)
    try:
        triangulation = mtri.Triangulation(points_2d[:, 0], points_2d[:, 1])
    except Exception:
        jitter = 1e-9 * np.arange(points_2d.shape[0], dtype=np.float64)
        triangulation = mtri.Triangulation(points_2d[:, 0] + jitter, points_2d[:, 1] - jitter)

    centers = np.column_stack((points_2d[triangulation.triangles].mean(axis=1)[:, 0], points_2d[triangulation.triangles].mean(axis=1)[:, 1]))
    tri_mask = ~outer_path.contains_points(centers)
    for hole in loops_tz[1:]:
        tri_mask |= mpath.Path(hole[:, :2]).contains_points(centers)
    triangulation.set_mask(tri_mask)

    theta_wrapped = np.mod(points_2d[:, 0] + seam_theta, 2.0 * np.pi)
    points_3d = np.column_stack(
        (
            center_x + support.radius * np.cos(theta_wrapped),
            center_y + support.radius * np.sin(theta_wrapped),
            points_2d[:, 1],
        )
    )
    return [points_3d[np.asarray(indexes, dtype=int)] for indexes in triangulation.get_masked_triangles()]


def face_triangles(
    face,
    edge_lookup: dict[int, object],
    support_lookup: dict[str, object],
    domain: AnnularCylinderDomain,
    seam_points_lookup: dict[int, np.ndarray] | None = None,
) -> list[np.ndarray]:
    if face.support_type in {"plane", "cap"}:
        return plane_face_triangles(face, edge_lookup, support_lookup, seam_points_lookup)
    if face.support_type == "cylinder":
        return cylinder_face_triangles(face, edge_lookup, support_lookup, domain, seam_points_lookup)
    return []


def _quantize_point(point: np.ndarray, scale: float = 1e7) -> tuple[int, int, int]:
    return tuple(int(round(float(value) * scale)) for value in point)


def _edge_signature(points: np.ndarray, support_keys: tuple[str, str]) -> tuple[tuple[str, str], tuple[int, int, int], tuple[int, int, int]]:
    start_key = _quantize_point(points[0])
    end_key = _quantize_point(points[-1])
    if start_key <= end_key:
        return tuple(sorted(support_keys)), start_key, end_key
    return tuple(sorted(support_keys)), end_key, start_key


def _indexed_vertices(
    positions: np.ndarray,
    normals: np.ndarray,
    scale: float = 1e7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    normals = np.ascontiguousarray(normals, dtype=np.float32)
    if positions.shape[0] == 0:
        return positions, normals, np.zeros((0,), dtype=np.uint32)
    pos_q = np.rint(positions.astype(np.float64) * scale).astype(np.int64)
    normal_q = np.rint(normals.astype(np.float64) * scale).astype(np.int64)
    packed = np.concatenate((pos_q, normal_q), axis=1)
    unique_packed, unique_indices, inverse = np.unique(packed, axis=0, return_index=True, return_inverse=True)
    del unique_packed
    return positions[unique_indices], normals[unique_indices], inverse.astype(np.uint32)


def _snap_polyline_points(points: np.ndarray, shared_snap: dict[tuple[int, int, int], np.ndarray]) -> np.ndarray:
    snapped = np.empty_like(points, dtype=np.float32)
    for idx, point in enumerate(np.asarray(points, dtype=np.float32)):
        key = _quantize_point(point)
        if key not in shared_snap:
            shared_snap[key] = point.astype(np.float32, copy=True)
        snapped[idx] = shared_snap[key]
    return snapped


def _face_loop_centroid(face, edge_lookup: dict[int, object], seam_points_lookup: dict[int, np.ndarray]) -> np.ndarray:
    loops = ordered_loop_points(face, edge_lookup, seam_points_lookup)
    if not loops:
        return np.zeros(3, dtype=np.float32)
    return np.mean(np.vstack(loops), axis=0).astype(np.float32)


def _face_offset_polyline(
    face,
    points: np.ndarray,
    face_centroid: np.ndarray,
    support_lookup: dict[str, object],
    offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    support = support_lookup[face.support_key]
    tangents = np.zeros_like(points)
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangent_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangent_norm = np.where(tangent_norm > 1e-12, tangent_norm, 1.0)
    tangents = tangents / tangent_norm

    if face.support_type == "cylinder":
        center_x, center_y = float(support.center_xy[0]), float(support.center_xy[1])
        normals = points.copy()
        normals[:, 0] -= center_x
        normals[:, 1] -= center_y
        normals[:, 2] = 0.0
        normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normal_norm = np.where(normal_norm > 1e-12, normal_norm, 1.0)
        normals = normals / normal_norm
        cand_a = np.cross(normals, tangents)
        cand_b = -cand_a
    else:
        normal = np.asarray(support.normal, dtype=np.float32)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        normals = np.repeat(normal.reshape(1, 3), len(points), axis=0)
        cand_a = np.cross(normal.reshape(1, 3), tangents)
        cand_b = -cand_a
    to_centroid = face_centroid.reshape(1, 3) - points
    use_a = np.sum(cand_a * to_centroid, axis=1) >= np.sum(cand_b * to_centroid, axis=1)
    inward = np.where(use_a[:, None], cand_a, cand_b)
    inward_norm = np.linalg.norm(inward, axis=1, keepdims=True)
    inward_norm = np.where(inward_norm > 1e-12, inward_norm, 1.0)
    inward = inward / inward_norm
    return points + offset * inward.astype(np.float32), normals.astype(np.float32)


def _build_seam_strip_triangles(
    edge_points: np.ndarray,
    left_face,
    right_face,
    left_centroid: np.ndarray,
    right_centroid: np.ndarray,
    support_lookup: dict[str, object],
    offset: float = 0.18,
) -> tuple[list[np.ndarray], np.ndarray]:
    left_points, left_normals = _face_offset_polyline(left_face, edge_points, left_centroid, support_lookup, offset)
    right_points, right_normals = _face_offset_polyline(right_face, edge_points, right_centroid, support_lookup, offset)
    avg_normals = left_normals + right_normals
    avg_norm = np.linalg.norm(avg_normals, axis=1, keepdims=True)
    avg_norm = np.where(avg_norm > 1e-12, avg_norm, 1.0)
    avg_normals = (avg_normals / avg_norm).astype(np.float32)
    triangles: list[np.ndarray] = []
    normal_rows: list[np.ndarray] = []
    for idx in range(len(edge_points) - 1):
        a0 = left_points[idx]
        a1 = left_points[idx + 1]
        b0 = right_points[idx]
        b1 = right_points[idx + 1]
        triangles.append(np.vstack((a0, b0, b1)).astype(np.float32))
        triangles.append(np.vstack((a0, b1, a1)).astype(np.float32))
        n0 = avg_normals[idx]
        n1 = avg_normals[idx + 1]
        normal_rows.append(np.vstack((n0, n0, n1)).astype(np.float32))
        normal_rows.append(np.vstack((n0, n1, n1)).astype(np.float32))
    return triangles, np.vstack(normal_rows).astype(np.float32) if normal_rows else np.zeros((0, 3), dtype=np.float32)


def _triangle_normals_for_face(face, triangles: list[np.ndarray], support_lookup: dict[str, object]) -> np.ndarray:
    if face.support_type == "cylinder":
        support = support_lookup[face.support_key]
        center_x, center_y = float(support.center_xy[0]), float(support.center_xy[1])
        verts = np.asarray(triangles, dtype=np.float32).reshape(-1, 3)
        normals = verts.copy()
        normals[:, 0] -= center_x
        normals[:, 1] -= center_y
        normals[:, 2] = 0.0
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.where(lengths > 1e-12, lengths, 1.0)
        return (normals / lengths).astype(np.float32)
    support = support_lookup[face.support_key]
    normal = np.asarray(support.normal, dtype=np.float32)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    count = len(triangles) * 3
    return np.repeat(normal.reshape(1, 3), count, axis=0).astype(np.float32)


def _is_shell_cell(cell: HybridExactCellBRep) -> bool:
    return any(face.support_key in SHELL_SUPPORT_KEYS for face in cell.faces)


def build_hybrid_exact_diagram_brep_from_diagram(diagram: ExactRestrictedVoronoiDiagram) -> HybridExactDiagramBRep:
    cells = tuple(
        build_hybrid_exact_cell_brep(cell, diagram, cell.neighboring_seed_ids)
        for cell in diagram.cells
    )
    return HybridExactDiagramBRep(cells=cells)


def serialize_threejs_shell_glb(
    diagram_brep: HybridExactDiagramBRep,
    domain: AnnularCylinderDomain,
) -> tuple[bytes, ThreeJSGLBExportSummary]:
    cells = list(diagram_brep.cells)
    num_shell_cells = sum(1 for cell in cells if _is_shell_cell(cell))

    builder = _GLBBuilder()
    line_material_index = builder.add_material((0.05, 0.05, 0.05), double_sided=True, unlit=True)
    scene_root_nodes: list[int] = []
    total_faces = 0
    total_triangles = 0
    total_boundaries = 0
    global_seam_registry: dict[tuple[tuple[str, str], tuple[int, int, int], tuple[int, int, int]], np.ndarray] = {}
    cell_seam_points: dict[int, dict[int, np.ndarray]] = {}

    for cell in cells:
        seam_lookup: dict[int, np.ndarray] = {}
        for edge in cell.edges:
            sampled = sample_curve(edge.curve, num=36).astype(np.float32)
            signature = _edge_signature(sampled, edge.support_keys)
            if signature not in global_seam_registry:
                canonical = sampled.copy()
                if _quantize_point(canonical[0]) > _quantize_point(canonical[-1]):
                    canonical = canonical[::-1].copy()
                global_seam_registry[signature] = canonical
            canonical = global_seam_registry[signature]
            if np.linalg.norm(sampled[0] - canonical[0]) <= np.linalg.norm(sampled[0] - canonical[-1]):
                seam_lookup[edge.edge_id] = canonical.copy()
            else:
                seam_lookup[edge.edge_id] = canonical[::-1].copy()
        cell_seam_points[cell.seed_id] = seam_lookup

    for cell in cells:
        is_shell_cell = _is_shell_cell(cell)
        edge_lookup = {edge.edge_id: edge for edge in cell.edges}
        support_lookup = {support.key: support for support in cell.supports}
        line_positions: list[np.ndarray] = []
        seen_edges: set[int] = set()
        face_groups: dict[str, dict[str, object]] = {
            "plane": {"positions": [], "normals": []},
            "cap": {"positions": [], "normals": []},
            "cylinder": {"positions": [], "normals": []},
        }

        seam_points_lookup = cell_seam_points[cell.seed_id]
        shell_faces = [face for face in cell.faces if face.support_key in SHELL_SUPPORT_KEYS]
        face_centroids = {face.face_id: _face_loop_centroid(face, edge_lookup, seam_points_lookup) for face in shell_faces}
        edge_to_shell_faces: dict[int, list[object]] = {}

        for face in cell.faces:
            tris = face_triangles(face, edge_lookup, support_lookup, domain, seam_points_lookup)
            if tris:
                total_faces += 1
                total_triangles += len(tris)
                triangle_positions = np.asarray(tris, dtype=np.float32).reshape(-1, 3)
                normals = _triangle_normals_for_face(face, tris, support_lookup)
                face_groups[face.support_type]["positions"].append(triangle_positions)
                face_groups[face.support_type]["normals"].append(normals)
            for loop in face.loop_edge_ids:
                for edge_id in loop:
                    if edge_id not in seen_edges:
                        points = seam_points_lookup[edge_id]
                        segments = np.empty((max(0, 2 * (len(points) - 1)), 3), dtype=np.float32)
                        segments[0::2] = points[:-1]
                        segments[1::2] = points[1:]
                        line_positions.append(segments)
                        seen_edges.add(edge_id)
                        total_boundaries += 1
                    if face.support_key in SHELL_SUPPORT_KEYS:
                        edge_to_shell_faces.setdefault(edge_id, []).append(face)

        seam_strip_positions: list[np.ndarray] = []
        seam_strip_normals: list[np.ndarray] = []
        for edge_id, incident_faces in edge_to_shell_faces.items():
            if len(incident_faces) != 2:
                continue
            left_face, right_face = incident_faces
            face_types = {left_face.support_type, right_face.support_type}
            if "cylinder" not in face_types or not ({"plane", "cap"} & face_types):
                continue
            strip_tris, strip_normals = _build_seam_strip_triangles(
                seam_points_lookup[edge_id],
                left_face,
                right_face,
                face_centroids[left_face.face_id],
                face_centroids[right_face.face_id],
                support_lookup,
            )
            if strip_tris:
                seam_strip_positions.append(np.asarray(strip_tris, dtype=np.float32).reshape(-1, 3))
                seam_strip_normals.append(strip_normals)

        primitives: list[dict[str, object]] = []
        child_nodes: list[int] = []
        base_color = scientific_block_color(cell.seed_id)
        material_by_group = {
            "plane": builder.add_material(base_color, double_sided=True, unlit=False),
            "cap": builder.add_material(tuple(min(1.0, channel * 0.96 + 0.04) for channel in base_color), double_sided=True, unlit=False),
            "cylinder": builder.add_material(tuple(max(0.0, channel * 0.92) for channel in base_color), double_sided=True, unlit=False),
            "seam": builder.add_material(tuple(min(1.0, channel * 0.94 + 0.03) for channel in base_color), double_sided=True, unlit=False),
        }
        for group_name in ("plane", "cap", "cylinder"):
            if not face_groups[group_name]["positions"]:
                continue
            position_array = np.vstack(face_groups[group_name]["positions"]).astype(np.float32)
            normal_array = np.vstack(face_groups[group_name]["normals"]).astype(np.float32)
            indexed_positions, indexed_normals, indices = _indexed_vertices(position_array, normal_array)
            position_accessor = builder.add_accessor(indexed_positions, type_name="VEC3", component_type=_COMPONENT_FLOAT, target=_ARRAY_BUFFER)
            normal_accessor = builder.add_accessor(indexed_normals, type_name="VEC3", component_type=_COMPONENT_FLOAT, target=_ARRAY_BUFFER)
            index_accessor = builder.add_accessor(indices, type_name="SCALAR", component_type=_COMPONENT_UINT, target=_ELEMENT_ARRAY_BUFFER)
            primitives.append(
                {
                    "attributes": {"POSITION": position_accessor, "NORMAL": normal_accessor},
                    "indices": index_accessor,
                    "material": material_by_group[group_name],
                    "mode": _MODE_TRIANGLES,
                }
            )

        if seam_strip_positions:
            seam_position_array = np.vstack(seam_strip_positions).astype(np.float32)
            seam_normal_array = np.vstack(seam_strip_normals).astype(np.float32)
            indexed_seam_positions, indexed_seam_normals, seam_indices = _indexed_vertices(seam_position_array, seam_normal_array)
            seam_position_accessor = builder.add_accessor(indexed_seam_positions, type_name="VEC3", component_type=_COMPONENT_FLOAT, target=_ARRAY_BUFFER)
            seam_normal_accessor = builder.add_accessor(indexed_seam_normals, type_name="VEC3", component_type=_COMPONENT_FLOAT, target=_ARRAY_BUFFER)
            seam_index_accessor = builder.add_accessor(seam_indices, type_name="SCALAR", component_type=_COMPONENT_UINT, target=_ELEMENT_ARRAY_BUFFER)
            primitives.append(
                {
                    "attributes": {"POSITION": seam_position_accessor, "NORMAL": seam_normal_accessor},
                    "indices": seam_index_accessor,
                    "material": material_by_group["seam"],
                    "mode": _MODE_TRIANGLES,
                }
            )

        if not primitives:
            continue

        face_mesh_index = builder.add_mesh(primitives, name=f"cell-{cell.seed_id}-faces")
        face_node_index = builder.add_node({"mesh": face_mesh_index, "name": f"cell-{cell.seed_id}-faces"})
        child_nodes.append(face_node_index)

        if line_positions:
            line_array = np.vstack(line_positions).astype(np.float32)
            line_accessor = builder.add_accessor(line_array, type_name="VEC3", component_type=_COMPONENT_FLOAT, target=_ARRAY_BUFFER)
            line_mesh_index = builder.add_mesh(
                [
                    {
                        "attributes": {"POSITION": line_accessor},
                        "material": line_material_index,
                        "mode": _MODE_LINES,
                    }
                ],
                name=f"cell-{cell.seed_id}-lines",
            )
            line_node_index = builder.add_node({"mesh": line_mesh_index, "name": f"cell-{cell.seed_id}-lines"})
            child_nodes.append(line_node_index)

        parent_node_index = builder.add_node(
            {
                "name": f"cell-{cell.seed_id}",
                "extras": {
                    "seedId": int(cell.seed_id),
                    "isShell": bool(is_shell_cell),
                    "cellLabel": "shell" if is_shell_cell else "non-shell",
                },
                "children": child_nodes,
            }
        )
        scene_root_nodes.append(parent_node_index)

    glb_bytes = builder.to_glb_bytes(scene_root_nodes)
    summary = ThreeJSGLBExportSummary(
        num_cells=len(diagram_brep.cells),
        num_shell_cells=num_shell_cells,
        num_exported_cells=len(scene_root_nodes),
        num_faces=total_faces,
        num_triangles=total_triangles,
        num_boundaries=total_boundaries,
        output_bytes=len(glb_bytes),
    )
    return glb_bytes, summary


def build_threejs_shell_glb_from_diagram(
    diagram: ExactRestrictedVoronoiDiagram,
) -> tuple[bytes, ThreeJSGLBExportSummary]:
    diagram_brep = build_hybrid_exact_diagram_brep_from_diagram(diagram)
    return serialize_threejs_shell_glb(diagram_brep=diagram_brep, domain=diagram.domain)


def build_threejs_shell_glb(seed_points: np.ndarray, domain: AnnularCylinderDomain) -> tuple[bytes, ThreeJSGLBExportSummary]:
    diagram = build_exact_restricted_voronoi_diagram(seed_points=seed_points, domain=domain, include_support_traces=False)
    return build_threejs_shell_glb_from_diagram(diagram)


def write_threejs_shell_glb(seed_points: np.ndarray, domain: AnnularCylinderDomain, output_path: Path) -> ThreeJSGLBExportSummary:
    glb_bytes, summary = build_threejs_shell_glb(seed_points=seed_points, domain=domain)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(glb_bytes)
    return summary
