from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from helix_voronoi.models import EdgeSegment, RenderConfig, RowGeometry
from helix_voronoi.rods import RodStyle
from helix_voronoi.voronoi import (
    FACE_SPECS,
    canonical_segment_key,
    cube_edge_segments,
    face_polygon,
)


def generate_colors(count: int) -> list[tuple[float, float, float]]:
    hues = np.linspace(0.0, 1.0, count, endpoint=False)
    colors = [mcolors.hsv_to_rgb((hue, 0.65, 0.95)) for hue in hues]
    return [tuple(color) for color in colors]


def configure_axes(
    ax: plt.Axes,
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (0.0, 1.0),
    zlim: tuple[float, float] = (0.0, 1.0),
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=35)


def draw_cube_edges(ax: plt.Axes) -> None:
    for start, end in cube_edge_segments():
        segment = np.array([start, end])
        ax.plot(
            segment[:, 0], segment[:, 1], segment[:, 2], color="#111827", linewidth=1.2
        )


def render_surface_view(
    ax: plt.Axes,
    row: RowGeometry,
    colors: list[tuple[float, float, float]],
) -> None:
    for index, (seed, vertices) in enumerate(zip(row.seeds, row.cells)):
        surface_polygons = []
        for axis_name, axis_value in FACE_SPECS:
            polygon = face_polygon(vertices, axis_name, axis_value)
            if polygon is not None:
                surface_polygons.append(polygon)

        if surface_polygons:
            poly3d = Poly3DCollection(
                surface_polygons,
                facecolors=colors[index],
                edgecolors="#1f2933",
                linewidths=1.1,
                alpha=1.0,
            )
            ax.add_collection3d(poly3d)

        ax.scatter(*seed, color="#111827", s=45, depthshade=False)
        ax.text(
            seed[0] + 0.02, seed[1] + 0.02, seed[2] + 0.02, f"S{index + 1}", fontsize=10
        )

    draw_cube_edges(ax)
    configure_axes(ax)
    ax.set_title(f"Surface Coloring (rng seed={row.rng_seed})")


def tile_edges(
    edges: list[EdgeSegment],
    repeats: tuple[int, int, int],
) -> list[EdgeSegment]:
    tiled_edges: dict[
        tuple[tuple[float, float, float], tuple[float, float, float]],
        EdgeSegment,
    ] = {}

    for offset_x in range(repeats[0]):
        for offset_y in range(repeats[1]):
            for offset_z in range(repeats[2]):
                offset = np.array([offset_x, offset_y, offset_z], dtype=float)
                for start, end in edges:
                    shifted_start = start + offset
                    shifted_end = end + offset
                    tiled_edges.setdefault(
                        canonical_segment_key(shifted_start, shifted_end),
                        (shifted_start, shifted_end),
                    )

    return list(tiled_edges.values())


def render_rod_view(
    ax: plt.Axes,
    edges: list[EdgeSegment],
    rod_style: RodStyle,
    radius: float,
    title: str,
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (0.0, 1.0),
    zlim: tuple[float, float] = (0.0, 1.0),
) -> None:
    for start, end in edges:
        rod_style.draw_segment(ax, start, end, radius=radius, color="#1f2937")

    configure_axes(ax, xlim=xlim, ylim=ylim, zlim=zlim)
    ax.set_title(title)


def plot_grid(
    rows: list[RowGeometry],
    render_config: RenderConfig,
    rod_styles: list[RodStyle],
) -> None:
    figure_rows = len(rows)
    figure_cols = 1 + 2 * len(rod_styles)
    fig = plt.figure(figsize=(8 * figure_cols, 8 * figure_rows))

    for row_index, row in enumerate(rows, start=1):
        colors = generate_colors(len(row.seeds))
        tiled_edges = tile_edges(row.edges, render_config.tile_repeats)
        row_base = (row_index - 1) * figure_cols
        ax_surface = fig.add_subplot(
            figure_rows,
            figure_cols,
            row_base + 1,
            projection="3d",
        )
        render_surface_view(ax_surface, row, colors)
        ax_surface.set_title(f"Surface Coloring (rng seed={row.rng_seed})")

        for style_index, rod_style in enumerate(rod_styles):
            rod_label = rod_style.name.replace("_", " ").title()
            col_offset = 1 + style_index * 2
            ax_rods = fig.add_subplot(
                figure_rows,
                figure_cols,
                row_base + col_offset + 1,
                projection="3d",
            )
            ax_tiled = fig.add_subplot(
                figure_rows,
                figure_cols,
                row_base + col_offset + 2,
                projection="3d",
            )

            render_rod_view(
                ax_rods,
                row.edges,
                rod_style=rod_style,
                radius=render_config.rod_radius,
                title=f"{rod_label} Rods (rng seed={row.rng_seed})",
            )

            render_rod_view(
                ax_tiled,
                tiled_edges,
                rod_style=rod_style,
                radius=render_config.tiled_rod_radius,
                title=f"3x3x3 Tiled {rod_label} Rods (rng seed={row.rng_seed})",
                xlim=(0.0, float(render_config.tile_repeats[0])),
                ylim=(0.0, float(render_config.tile_repeats[1])),
                zlim=(0.0, float(render_config.tile_repeats[2])),
            )

    fig.tight_layout()
    render_config.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(render_config.output_path, dpi=220)

    if render_config.show:
        plt.show()

    plt.close(fig)
