from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from helix_voronoi.rods import HelixRodStyle


def load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def make_blue_red_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "blue_to_red_density",
        [
            (0.00, "#2563eb"),
            (0.35, "#38bdf8"),
            (0.70, "#fb7185"),
            (1.00, "#b91c1c"),
        ],
    )


def make_probability_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "probability_map",
        [
            (0.00, "#dbeafe"),
            (0.20, "#60a5fa"),
            (0.45, "#22c55e"),
            (0.70, "#f59e0b"),
            (1.00, "#dc2626"),
        ],
    )


def downsample_density(
    density_milli: np.ndarray,
    max_display_size: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    steps = tuple(max(1, int(np.ceil(size / max_display_size))) for size in density_milli.shape)
    sampled_density = density_milli[:: steps[0], :: steps[1], :: steps[2]].astype(np.float32) / 1000.0
    occupancy = sampled_density > 0.0
    return occupancy, sampled_density, steps


def density_to_probability_intensity(density: np.ndarray, gamma: float) -> np.ndarray:
    weights = np.where(density > 0.0, density**gamma, 0.0)
    max_weight = float(weights.max()) if weights.size else 0.0
    if max_weight <= 0.0:
        return np.zeros_like(density, dtype=np.float32)
    return (weights / max_weight).astype(np.float32)


def build_density_facecolors(voxels: np.ndarray, density: np.ndarray) -> np.ndarray:
    cmap = make_blue_red_cmap()
    normalized = np.clip(density, 0.0, 1.0)
    facecolors = cmap(normalized)
    alpha = 0.22 + 0.68 * normalized
    facecolors[..., 3] = np.where(voxels, alpha, 0.0)
    return facecolors


def build_probability_facecolors(voxels: np.ndarray, probability_intensity: np.ndarray) -> np.ndarray:
    cmap = make_probability_cmap()
    normalized = np.clip(probability_intensity, 0.0, 1.0)
    facecolors = cmap(normalized)
    alpha = 0.14 + 0.80 * normalized
    facecolors[..., 3] = np.where(voxels, alpha, 0.0)
    return facecolors


def configure_voxel_axes(
    ax: plt.Axes,
    displayed_shape: tuple[int, int, int],
    original_shape: tuple[int, int, int],
    steps: tuple[int, int, int],
    title: str,
) -> None:
    x_ticks = np.linspace(0, displayed_shape[0], 6)
    y_ticks = np.linspace(0, displayed_shape[1], 6)
    z_ticks = np.linspace(0, displayed_shape[2], 6)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)
    ax.set_xticklabels([str(int(round(v * steps[0]))) for v in x_ticks])
    ax.set_yticklabels([str(int(round(v * steps[1]))) for v in y_ticks])
    ax.set_zticklabels([str(int(round(v * steps[2]))) for v in z_ticks])
    ax.set_xlim(0, displayed_shape[0])
    ax.set_ylim(0, displayed_shape[1])
    ax.set_zlim(0, displayed_shape[2])
    ax.set_box_aspect(original_shape)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=38)
    ax.set_title(title)


def configure_rod_axes(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    box_aspect: tuple[float, float, float],
    title: str,
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect(box_aspect)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=38)
    ax.set_title(title)


def sample_edges_for_preview(
    tiled_edges: list[tuple[np.ndarray, np.ndarray]],
    max_render_edges: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(tiled_edges) <= max_render_edges:
        print(f"render edge count unchanged: {len(tiled_edges)}")
        return tiled_edges
    indices = np.linspace(0, len(tiled_edges) - 1, max_render_edges, dtype=int)
    sampled = [tiled_edges[index] for index in indices]
    print(f"render edge count reduced: {len(tiled_edges)} -> {len(sampled)}")
    return sampled


def render_quad(
    output_png: Path,
    occupancy: np.ndarray,
    display_density: np.ndarray,
    display_probability: np.ndarray,
    display_steps: tuple[int, int, int],
    original_shape: tuple[int, int, int],
    seed_points: np.ndarray,
    tiled_edges: list[tuple[np.ndarray, np.ndarray]],
    representative_seed_count: int,
    num_input_seeds: int,
    gamma: float,
    max_render_edges: int,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(32, 8))

    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax1.voxels(
        occupancy,
        facecolors=build_density_facecolors(occupancy, display_density),
        edgecolor="#111827",
        linewidth=0.05,
    )
    configure_voxel_axes(ax1, occupancy.shape, original_shape, display_steps, "1) Fake density result")

    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    ax2.voxels(
        occupancy,
        facecolors=build_probability_facecolors(occupancy, display_probability),
        edgecolor="#111827",
        linewidth=0.05,
    )
    configure_voxel_axes(ax2, occupancy.shape, original_shape, display_steps, f"2) Spatial probability map (gamma={gamma:.2f})")

    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    seed_colors = np.clip(seed_points[:, 2] / max(original_shape[2], 1), 0.0, 1.0)
    ax3.scatter(
        seed_points[:, 0],
        seed_points[:, 1],
        seed_points[:, 2],
        c=seed_colors,
        cmap="coolwarm",
        s=0.7,
        alpha=0.28,
        linewidths=0,
    )
    ax3.set_xlim(0, original_shape[0])
    ax3.set_ylim(0, original_shape[1])
    ax3.set_zlim(0, original_shape[2])
    ax3.set_box_aspect(original_shape)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.view_init(elev=24, azim=38)
    ax3.set_title(f"3) Voronoi seed points ({num_input_seeds:,})")

    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    helix_style = HelixRodStyle(
        cycles_per_segment=2.1,
        amplitude_ratio=0.045,
        wire_radius_ratio=0.80,
        tube_sides=8,
        min_steps=18,
        steps_per_cycle=10,
    )
    z_ratio = original_shape[2] / original_shape[0]
    render_edges = sample_edges_for_preview(tiled_edges, max_render_edges=max_render_edges)
    total_edges = len(render_edges)
    for edge_index, (start, end) in enumerate(render_edges, start=1):
        start_scaled = np.array([start[0], start[1], start[2] * z_ratio], dtype=float)
        end_scaled = np.array([end[0], end[1], end[2] * z_ratio], dtype=float)
        helix_style.draw_segment(ax4, start_scaled, end_scaled, radius=0.010, color="#7c3aed")
        if edge_index == 1 or edge_index == total_edges or edge_index % 250 == 0:
            print(f"rerendered helix edge {edge_index}/{total_edges}")
    configure_rod_axes(
        ax4,
        xlim=(0.0, 3.0),
        ylim=(0.0, 3.0),
        zlim=(0.0, 3.0 * z_ratio),
        box_aspect=(3.0, 3.0, 3.0 * z_ratio),
        title=f"4) 3x3x3 helix Voronoi cell (repr. seeds={representative_seed_count})",
    )

    fig.suptitle("Fake density → probability → 100k seeds → 3x3x3 helix Voronoi", fontsize=20, y=0.99)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05, wspace=0.08)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)
    print(f"saved quad figure: {output_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerender the Voronoi helix quad figure from saved NPZ results.")
    parser.add_argument(
        "--fake-density-npz",
        type=Path,
        default=Path("datasets/topopt/fake_density_annular_cylinder_full.npz"),
    )
    parser.add_argument(
        "--seed-mapping-npz",
        type=Path,
        default=Path("datasets/topopt/seed_probability_mapping_100k.npz"),
    )
    parser.add_argument(
        "--voronoi-helix-npz",
        type=Path,
        default=Path("datasets/topopt/voronoi_helix_quad_100k_to_255_fast.npz"),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.8,
    )
    parser.add_argument(
        "--max-display-size",
        type=int,
        default=36,
    )
    parser.add_argument(
        "--max-render-edges",
        type=int,
        default=6000,
        help="Maximum helix edges rendered in panel 4 for a fast preview.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=130,
        help="PNG dpi for fast preview rendering.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("docs/assets/voronoi_helix_quad_100k_to_255_fast.png"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fake_density = load_npz(args.fake_density_npz)
    seed_mapping = load_npz(args.seed_mapping_npz)
    voronoi_helix = load_npz(args.voronoi_helix_npz)

    density_milli = fake_density["density_milli"]
    full_seed_points = seed_mapping["seed_points"].astype(np.float32)
    occupancy, display_density, display_steps = downsample_density(density_milli, args.max_display_size)
    display_probability = density_to_probability_intensity(display_density, args.gamma)

    tiled_starts = voronoi_helix["tiled_edge_starts"].astype(np.float32)
    tiled_ends = voronoi_helix["tiled_edge_ends"].astype(np.float32)
    tiled_edges = list(zip(tiled_starts, tiled_ends))
    representative_seed_count = int(voronoi_helix["num_representative_seeds"].item())

    render_quad(
        output_png=args.output_png,
        occupancy=occupancy,
        display_density=display_density,
        display_probability=display_probability,
        display_steps=display_steps,
        original_shape=density_milli.shape,
        seed_points=full_seed_points,
        tiled_edges=tiled_edges,
        representative_seed_count=representative_seed_count,
        num_input_seeds=len(full_seed_points),
        gamma=args.gamma,
        max_render_edges=args.max_render_edges,
        dpi=args.dpi,
    )
