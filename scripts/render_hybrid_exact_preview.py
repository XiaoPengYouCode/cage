from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from topopt_sampling import build_annular_cylinder_domain, build_hybrid_exact_diagram_brep
from topopt_sampling.hybrid_exact_brep import (
    ExactCircleArc,
    ExactCylinderPlaneCurve,
    ExactLineSegment,
)


def _sample_curve(curve: object, num: int = 80) -> np.ndarray:
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
    raise TypeError(f"Unsupported curve type: {type(curve)!r}")


def _set_axes_equal(ax: plt.Axes, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _draw_domain_guides(ax: plt.Axes, center_xy: np.ndarray, radius: float, z_min: float, z_max: float, color: str) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    x = center_xy[0] + radius * np.cos(theta)
    y = center_xy[1] + radius * np.sin(theta)
    ax.plot(x, y, np.full_like(theta, z_min), color=color, alpha=0.22, linewidth=1.0)
    ax.plot(x, y, np.full_like(theta, z_max), color=color, alpha=0.22, linewidth=1.0)
    for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
        xv = center_xy[0] + radius * math.cos(angle)
        yv = center_xy[1] + radius * math.sin(angle)
        ax.plot([xv, xv], [yv, yv], [z_min, z_max], color=color, alpha=0.10, linewidth=0.8)


def render_preview(seed_npz: Path, output_png: Path, seed_ids: list[int], xy_size: int, z_size: int, outer_radius: float, inner_radius: float) -> None:
    with np.load(seed_npz) as data:
        seed_points = data["seed_points"].astype(float)
    domain = build_annular_cylinder_domain(
        xy_size=xy_size,
        z_size=z_size,
        outer_radius=outer_radius,
        inner_radius=inner_radius,
    )
    diagram_brep = build_hybrid_exact_diagram_brep(seed_points=seed_points, domain=domain, seed_ids=seed_ids)

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(diagram_brep.cells), 3)))
    fig = plt.figure(figsize=(14, 7))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")

    all_points: list[np.ndarray] = []
    for cell, color in zip(diagram_brep.cells, colors):
        seed = seed_points[cell.seed_id]
        ax0.scatter(*seed, color=color, s=22)
        ax1.scatter(*seed, color=color, s=22)
        ax0.text(seed[0], seed[1], seed[2], str(cell.seed_id), fontsize=7)
        ax1.text(seed[0], seed[1], seed[2], str(cell.seed_id), fontsize=7)
        all_points.append(seed.reshape(1, 3))

        for edge in cell.polyhedral_cell.edges:
            pts = _sample_curve(edge.curve, num=2)
            ax0.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=1.2)
            all_points.append(pts)

        for edge in cell.edges:
            pts = _sample_curve(edge.curve, num=120)
            lw = 1.6
            if any(k in {"outer_cylinder", "inner_cylinder"} for k in edge.support_keys):
                lw = 2.0
            ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=lw)
            all_points.append(pts)

        poly_points = np.array([vertex.point for vertex in cell.polyhedral_cell.vertices])
        if len(poly_points):
            ax0.scatter(poly_points[:, 0], poly_points[:, 1], poly_points[:, 2], color=color, s=8, alpha=0.8)
            all_points.append(poly_points)

        new_vertex_ids = set(cell.trim_summary.new_vertex_ids)
        if new_vertex_ids:
            new_points = np.array([vertex.point for vertex in cell.vertices if vertex.vertex_id in new_vertex_ids])
            if len(new_points):
                ax1.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], color=color, s=18, marker="^")
                all_points.append(new_points)

    _draw_domain_guides(ax1, domain.center_xy, domain.outer_radius, domain.z_min, domain.z_max, "#6b7280")
    if domain.inner_radius > 0.0:
        _draw_domain_guides(ax1, domain.center_xy, domain.inner_radius, domain.z_min, domain.z_max, "#9ca3af")

    cloud = np.vstack(all_points)
    for ax in (ax0, ax1):
        _set_axes_equal(ax, cloud)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=22, azim=34)

    ax0.set_title("Step 1: box polyhedral Voronoi cells")
    ax1.set_title("Step 3: trimmed hybrid exact B-rep")
    fig.suptitle(f"Hybrid exact preview with new pipeline | seeds={seed_ids}")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output_png)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render hybrid exact B-rep preview image.")
    parser.add_argument("--seed-npz", type=Path, default=Path("datasets/topopt/seed_probability_mapping_2000.npz"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/hybrid_exact_new_pipeline_preview.png"))
    parser.add_argument("--xy-size", type=int, default=200)
    parser.add_argument("--z-size", type=int, default=80)
    parser.add_argument("--outer-radius", type=float, default=100.0)
    parser.add_argument("--inner-radius", type=float, default=50.0)
    parser.add_argument("--seed-ids", type=int, nargs="*", default=[528, 526, 424, 423, 393, 480])
    args = parser.parse_args()
    render_preview(
        seed_npz=args.seed_npz,
        output_png=args.output,
        seed_ids=args.seed_ids,
        xy_size=args.xy_size,
        z_size=args.z_size,
        outer_radius=args.outer_radius,
        inner_radius=args.inner_radius,
    )


if __name__ == "__main__":
    main()
