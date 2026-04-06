from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def draw_domain_guides(ax, center_xy: np.ndarray, radius: float, z_min: float, z_max: float, color: str, alpha: float) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    x = center_xy[0] + radius * np.cos(theta)
    y = center_xy[1] + radius * np.sin(theta)
    ax.plot(x, y, np.full_like(theta, z_min), color=color, alpha=alpha, linewidth=0.8)
    ax.plot(x, y, np.full_like(theta, z_max), color=color, alpha=alpha, linewidth=0.8)
    for angle in np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False):
        xv = center_xy[0] + radius * math.cos(angle)
        yv = center_xy[1] + radius * math.sin(angle)
        ax.plot([xv, xv], [yv, yv], [z_min, z_max], color=color, alpha=alpha * 0.5, linewidth=0.5)


def main() -> None:
    seed_npz = Path("datasets/topopt/seed_probability_mapping_2000.npz")
    output_png = Path("docs/assets/hybrid_exact_new_pipeline_2000seeds.png")
    with np.load(seed_npz) as data:
        seed_points = data["seed_points"].astype(float)

    domain = build_annular_cylinder_domain(
        xy_size=200,
        z_size=80,
        outer_radius=100.0,
        inner_radius=50.0,
    )

    t0 = time.time()
    print("[1/4] building restricted Voronoi diagram shell...", flush=True)
    diagram = build_exact_restricted_voronoi_diagram(seed_points=seed_points, domain=domain, include_support_traces=False)
    print(f"      done: cells={len(diagram.cells)} elapsed={time.time() - t0:.1f}s", flush=True)

    print("[2/4] building Delaunay neighbor map...", flush=True)
    neighbor_map = build_delaunay_neighbor_map(diagram.seed_points)
    print(f"      done: neighbor entries={len(neighbor_map)} elapsed={time.time() - t0:.1f}s", flush=True)

    print("[3/4] building hybrid exact cells with progress...", flush=True)
    built_cells = []
    for idx, cell in enumerate(diagram.cells, start=1):
        built_cells.append(build_hybrid_exact_cell_brep(cell, diagram, neighbor_map.get(int(cell.seed_id), tuple())))
        if idx <= 10 or idx % 50 == 0 or idx == len(diagram.cells):
            print(f"      cell {idx}/{len(diagram.cells)} elapsed={time.time() - t0:.1f}s", flush=True)
    diagram_brep = HybridExactDiagramBRep(cells=tuple(built_cells))
    print(f"      done: built {len(diagram_brep.cells)} cells elapsed={time.time() - t0:.1f}s", flush=True)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.0, 1.0, len(diagram_brep.cells)))

    print("[4/4] rendering matplotlib figure with progress...", flush=True)
    total_edges = 0
    total_vertices = 0
    for idx, cell in enumerate(diagram_brep.cells, start=1):
        color = colors[idx - 1]
        for edge in cell.edges:
            pts = sample_curve(edge.curve, num=18)
            if any(key in {"outer_cylinder", "inner_cylinder"} for key in edge.support_keys):
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=0.35, alpha=0.32)
            else:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=0.25, alpha=0.22)
            total_edges += 1
        total_vertices += len(cell.vertices)
        if idx <= 10 or idx % 100 == 0 or idx == len(diagram_brep.cells):
            print(f"      rendered cell {idx}/{len(diagram_brep.cells)} edges={total_edges} elapsed={time.time() - t0:.1f}s", flush=True)

    seed_mask = domain.contains_points(seed_points)
    seeds = seed_points[seed_mask]
    ax.scatter(seeds[:, 0], seeds[:, 1], seeds[:, 2], c="#111827", s=1.2, alpha=0.35)

    draw_domain_guides(ax, domain.center_xy, domain.outer_radius, domain.z_min, domain.z_max, "#6b7280", 0.16)
    draw_domain_guides(ax, domain.center_xy, domain.inner_radius, domain.z_min, domain.z_max, "#9ca3af", 0.16)

    ax.set_xlim(domain.center_xy[0] - domain.outer_radius, domain.center_xy[0] + domain.outer_radius)
    ax.set_ylim(domain.center_xy[1] - domain.outer_radius, domain.center_xy[1] + domain.outer_radius)
    ax.set_zlim(domain.z_min, domain.z_max)
    ax.set_box_aspect((2, 2, 0.8))
    ax.view_init(elev=24, azim=36)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Hybrid exact B-rep with new pipeline | 2000 seeds | edges={total_edges} | vertices={total_vertices}")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"saved image: {output_png} total_elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
