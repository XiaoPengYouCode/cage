from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from topopt_sampling import build_annular_cylinder_domain
from topopt_sampling.exact_brep import build_delaunay_neighbor_map
from topopt_sampling.exact_restricted_voronoi_3d import build_exact_restricted_voronoi_diagram
from topopt_sampling.hybrid_exact_brep import (
    ExactCircleArc,
    ExactCylinderPlaneCurve,
    ExactLineSegment,
    HybridExactDiagramBRep,
    build_hybrid_exact_cell_brep,
)


SAMPLES_PER_EDGE = 10
GRID_FACE_SAMPLES = 16
MAX_CELLS = None  # set int for debug
SHELL_SUPPORT_KEYS = {"outer_cylinder", "inner_cylinder", "top_cap", "bottom_cap"}


def scientific_block_color(seed_id: int) -> tuple[float, float, float, float]:
    hue = ((seed_id * 0.6180339887498949) % 1.0)
    sat = 0.42 + 0.12 * (((seed_id * 7) % 11) / 10.0)
    val = 0.72 + 0.12 * (((seed_id * 13) % 9) / 8.0)
    r, g, b = mcolors.hsv_to_rgb((hue, sat, val))
    return (float(r), float(g), float(b), 1.0)


def sample_curve(curve: object, num: int = SAMPLES_PER_EDGE) -> np.ndarray:
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
            z = np.linspace(curve.start[2], curve.end[2], num)
            theta = np.full_like(z, curve.vertical_theta)
        else:
            start = curve.theta_start
            end = curve.theta_end
            delta = end - start
            if delta > math.pi:
                end -= 2.0 * math.pi
            elif delta < -math.pi:
                end += 2.0 * math.pi
            theta = np.linspace(start, end, num)
            z = (
                curve.plane_rhs
                - curve.plane_normal[0] * (curve.cylinder_center_xy[0] + curve.cylinder_radius * np.cos(theta))
                - curve.plane_normal[1] * (curve.cylinder_center_xy[1] + curve.cylinder_radius * np.sin(theta))
            ) / curve.plane_normal[2]
        return np.column_stack(
            (
                curve.cylinder_center_xy[0] + curve.cylinder_radius * np.cos(theta),
                curve.cylinder_center_xy[1] + curve.cylinder_radius * np.sin(theta),
                z,
            )
        )
    raise TypeError(type(curve))


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal / np.linalg.norm(normal)
    ref = np.array([0.0, 0.0, 1.0], dtype=float) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=float)
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return u, v


def ordered_loop_points(face, edge_lookup: dict[int, object]) -> list[np.ndarray]:
    loops = []
    for loop_edge_ids in face.loop_edge_ids:
        pts_parts: list[np.ndarray] = []
        last_point = None
        for edge_id in loop_edge_ids:
            edge = edge_lookup[edge_id]
            pts = sample_curve(edge.curve)
            if last_point is not None:
                d0 = np.linalg.norm(pts[0] - last_point)
                d1 = np.linalg.norm(pts[-1] - last_point)
                if d1 < d0:
                    pts = pts[::-1]
            if pts_parts:
                pts = pts[1:]
            pts_parts.append(pts)
            last_point = pts[-1]
        loop = np.vstack(pts_parts)
        if np.linalg.norm(loop[0] - loop[-1]) > 1e-6:
            loop = np.vstack((loop, loop[0]))
        loops.append(loop)
    return loops


def triangulate_polygon_2d(points_2d: np.ndarray, holes_2d: list[np.ndarray] | None = None, samples: int = GRID_FACE_SAMPLES):
    holes_2d = holes_2d or []
    outer_path = mpath.Path(points_2d[:, :2])
    all_pts = [points_2d[:, :2]] + [hole[:, :2] for hole in holes_2d]
    all_pts_stack = np.vstack(all_pts)
    mins = all_pts_stack.min(axis=0)
    maxs = all_pts_stack.max(axis=0)
    if np.any(maxs - mins < 1e-8):
        return None, None
    xs = np.linspace(mins[0], maxs[0], samples)
    ys = np.linspace(mins[1], maxs[1], samples)
    grid = np.array(np.meshgrid(xs, ys, indexing="xy")).reshape(2, -1).T
    mask = outer_path.contains_points(grid)
    for hole in holes_2d:
        hole_path = mpath.Path(hole[:, :2])
        mask &= ~hole_path.contains_points(grid)
    interior = grid[mask]
    pts2 = np.vstack([all_pts_stack, interior])
    tri = mtri.Triangulation(pts2[:, 0], pts2[:, 1])
    centers = np.column_stack([
        pts2[tri.triangles].mean(axis=1)[:, 0],
        pts2[tri.triangles].mean(axis=1)[:, 1],
    ])
    tri_mask = ~outer_path.contains_points(centers)
    for hole in holes_2d:
        hole_path = mpath.Path(hole[:, :2])
        tri_mask |= hole_path.contains_points(centers)
    tri.set_mask(tri_mask)
    return pts2, tri


def plane_face_triangles(face, edge_lookup: dict[int, object], support_lookup: dict[str, object]) -> list[np.ndarray]:
    loops = ordered_loop_points(face, edge_lookup)
    if not loops:
        return []
    support = support_lookup[face.support_key]
    origin = loops[0][0]
    if face.support_type == "cap":
        outer_2d = loops[0][:, :2]
        holes_2d = [loop[:, :2] for loop in loops[1:]]
        pts2, tri = triangulate_polygon_2d(outer_2d, holes_2d)
        if tri is None:
            return []
        z = loops[0][0, 2]
        pts3 = np.column_stack((pts2[:, 0], pts2[:, 1], np.full(len(pts2), z)))
    else:
        u, v = plane_basis(support.normal)
        loops_uv = []
        for loop in loops:
            rel = loop - origin
            loops_uv.append(np.column_stack((rel @ u, rel @ v)))
        pts2, tri = triangulate_polygon_2d(loops_uv[0], loops_uv[1:])
        if tri is None:
            return []
        pts3 = origin + pts2[:, [0]] * u + pts2[:, [1]] * v
    triangles = []
    for tri_ix in tri.get_masked_triangles():
        triangles.append(pts3[np.asarray(tri_ix, dtype=int)])
    return triangles


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



def _select_best_cylinder_atlas(loops: list[np.ndarray], center_x: float, center_y: float) -> list[np.ndarray]:
    candidates = []
    for seam_theta in (0.0, np.pi):
        loops_tz = _cylinder_loops_tz_with_seam(loops, center_x, center_y, seam_theta)
        stacked = np.vstack(loops_tz)
        theta_span = float(np.max(stacked[:, 0]) - np.min(stacked[:, 0]))
        candidates.append((theta_span, loops_tz))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]



def cylinder_face_triangles(face, edge_lookup: dict[int, object], support_lookup: dict[str, object]) -> list[np.ndarray]:
    loops = ordered_loop_points(face, edge_lookup)
    if not loops:
        return []
    support = support_lookup[face.support_key]
    cx, cy = support.center_xy
    loops_tz = _select_best_cylinder_atlas(loops, float(cx), float(cy))
    pts2, tri = triangulate_polygon_2d(loops_tz[0], loops_tz[1:])
    if tri is None:
        return []
    pts3 = np.column_stack(
        (
            cx + support.radius * np.cos(pts2[:, 0]),
            cy + support.radius * np.sin(pts2[:, 0]),
            pts2[:, 1],
        )
    )
    triangles = []
    for tri_ix in tri.get_masked_triangles():
        triangles.append(pts3[np.asarray(tri_ix, dtype=int)])
    return triangles


def face_triangles(face, edge_lookup: dict[int, object], support_lookup: dict[str, object]) -> list[np.ndarray]:
    if face.support_type in {"plane", "cap"}:
        return plane_face_triangles(face, edge_lookup, support_lookup)
    if face.support_type == "cylinder":
        return cylinder_face_triangles(face, edge_lookup, support_lookup)
    return []


def draw_domain_guides(ax, center_xy: np.ndarray, radius: float, z_min: float, z_max: float, color: str, alpha: float) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    x = center_xy[0] + radius * np.cos(theta)
    y = center_xy[1] + radius * np.sin(theta)
    ax.plot(x, y, np.full_like(theta, z_min), color=color, alpha=alpha, linewidth=0.6)
    ax.plot(x, y, np.full_like(theta, z_max), color=color, alpha=alpha, linewidth=0.6)


def main() -> None:
    seed_npz = Path("datasets/topopt/seed_probability_mapping_2000.npz")
    output_png = Path("docs/assets/hybrid_exact_new_pipeline_2000seeds_filled.png")
    with np.load(seed_npz) as data:
        seed_points = data["seed_points"].astype(float)

    domain = build_annular_cylinder_domain(xy_size=200, z_size=80, outer_radius=100.0, inner_radius=50.0)

    t0 = time.time()
    print("[1/4] building restricted Voronoi diagram shell...", flush=True)
    diagram = build_exact_restricted_voronoi_diagram(seed_points=seed_points, domain=domain, include_support_traces=False)
    print(f"      done: cells={len(diagram.cells)} elapsed={time.time() - t0:.1f}s", flush=True)

    print("[2/4] building Delaunay neighbor map...", flush=True)
    neighbor_map = build_delaunay_neighbor_map(diagram.seed_points)
    print(f"      done: neighbor entries={len(neighbor_map)} elapsed={time.time() - t0:.1f}s", flush=True)

    print("[3/4] building hybrid exact cells with progress...", flush=True)
    built_cells = []
    total = len(diagram.cells) if MAX_CELLS is None else min(MAX_CELLS, len(diagram.cells))
    for idx, cell in enumerate(diagram.cells[:total], start=1):
        built_cells.append(build_hybrid_exact_cell_brep(cell, diagram, neighbor_map.get(int(cell.seed_id), tuple())))
        if idx <= 10 or idx % 50 == 0 or idx == total:
            print(f"      cell {idx}/{total} elapsed={time.time() - t0:.1f}s", flush=True)
    diagram_brep = HybridExactDiagramBRep(cells=tuple(built_cells))
    print(f"      done: built {len(diagram_brep.cells)} cells elapsed={time.time() - t0:.1f}s", flush=True)

    print("[4/4] triangulating and rendering filled shell faces only...", flush=True)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    shell_cells = [
        cell for cell in diagram_brep.cells if any(face.support_key in SHELL_SUPPORT_KEYS for face in cell.faces)
    ]
    print(f"      shell cells selected: {len(shell_cells)}/{len(diagram_brep.cells)}", flush=True)

    total_faces = 0
    total_tris = 0

    for idx, cell in enumerate(shell_cells, start=1):
        edge_lookup = {edge.edge_id: edge for edge in cell.edges}
        support_lookup = {support.key: support for support in cell.supports}
        color = scientific_block_color(cell.seed_id)
        face_polys: list[np.ndarray] = []
        for face in cell.faces:
            if face.support_key not in SHELL_SUPPORT_KEYS:
                continue
            tris = face_triangles(face, edge_lookup, support_lookup)
            if tris:
                face_polys.extend(tris)
                total_faces += 1
                total_tris += len(tris)
        if face_polys:
            coll = Poly3DCollection(
                face_polys,
                facecolor=color,
                edgecolor="none",
                linewidths=0.0,
                alpha=1.0,
                antialiased=False,
            )
            ax.add_collection3d(coll)

            for edge in cell.edges:
                if not any(key in SHELL_SUPPORT_KEYS for key in edge.support_keys):
                    continue
                pts = sample_curve(edge.curve, num=max(24, SAMPLES_PER_EDGE * 3))
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    color=(0.03, 0.03, 0.03, 0.95),
                    linewidth=1.35,
                    solid_capstyle="round",
                )
        if idx <= 10 or idx % 50 == 0 or idx == len(shell_cells):
            print(f"      rendered shell cell {idx}/{len(shell_cells)} faces={total_faces} tris={total_tris} elapsed={time.time() - t0:.1f}s", flush=True)

    draw_domain_guides(ax, domain.center_xy, domain.outer_radius, domain.z_min, domain.z_max, "#111827", 0.10)
    draw_domain_guides(ax, domain.center_xy, domain.inner_radius, domain.z_min, domain.z_max, "#111827", 0.10)

    ax.set_xlim(domain.center_xy[0] - domain.outer_radius, domain.center_xy[0] + domain.outer_radius)
    ax.set_ylim(domain.center_xy[1] - domain.outer_radius, domain.center_xy[1] + domain.outer_radius)
    ax.set_zlim(domain.z_min, domain.z_max)
    ax.set_box_aspect((2, 2, 0.8))
    ax.view_init(elev=24, azim=36)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Hybrid exact B-rep filled shell blocks | 2000 seeds | shell_cells={len(shell_cells)} | faces={total_faces} | triangles={total_tris}")
    ax.grid(True, alpha=0.22)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved image: {output_png} total_elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
