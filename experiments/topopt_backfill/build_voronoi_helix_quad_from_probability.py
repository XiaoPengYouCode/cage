from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from helix_voronoi.rendering import tile_edges
from helix_voronoi.rods import HelixRodStyle
from helix_voronoi.voronoi import build_voronoi_cells, extract_unique_edges


def load_result(npz_path: Path) -> dict[str, np.ndarray]:
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


def normalize_seed_points(seed_points: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    scale = np.array(target_shape, dtype=np.float32)
    normalized = seed_points / scale
    eps = 1e-5
    return np.clip(normalized, eps, 1.0 - eps)


def build_candidate_pool(
    points: np.ndarray,
    candidate_limit: int,
    voxel_bins: tuple[int, int, int],
    rng_seed: int,
) -> np.ndarray:
    if len(points) <= candidate_limit:
        print(f"candidate pool unchanged: {len(points)}")
        return points.astype(np.float32, copy=False)

    clipped = np.clip(points, 0.0, 1.0 - 1e-7)
    bins = np.array(voxel_bins, dtype=np.int32)
    grid_idx = np.floor(clipped * bins[None, :]).astype(np.int32)
    keys = np.ravel_multi_index(grid_idx.T, bins)
    _, unique_indices = np.unique(keys, return_index=True)
    pooled = points[np.sort(unique_indices)].astype(np.float32, copy=False)
    print(
        f"candidate pool after voxel bucketing: {len(pooled)} "
        f"(from {len(points)}, bins={voxel_bins[0]}x{voxel_bins[1]}x{voxel_bins[2]})"
    )

    if len(pooled) > candidate_limit:
        rng = np.random.default_rng(rng_seed)
        selection = rng.choice(len(pooled), size=candidate_limit, replace=False)
        pooled = pooled[selection]
        print(f"candidate pool downsampled to: {len(pooled)}")

    return pooled


def farthest_point_sampling(points: np.ndarray, count: int) -> np.ndarray:
    if count >= len(points):
        print(f"candidate pool smaller than requested representative seeds: {len(points)}")
        return points.copy()

    selected = np.empty((count, 3), dtype=np.float32)
    centroid = points.mean(axis=0, dtype=np.float32)
    distances_to_centroid = np.sum((points - centroid) ** 2, axis=1, dtype=np.float32)
    first_index = int(np.argmax(distances_to_centroid))
    selected[0] = points[first_index]

    min_dist_sq = np.sum((points - selected[0]) ** 2, axis=1, dtype=np.float32)
    for i in range(1, count):
        next_index = int(np.argmax(min_dist_sq))
        selected[i] = points[next_index]
        dist_sq = np.sum((points - selected[i]) ** 2, axis=1, dtype=np.float32)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)
        if i == 1 or i + 1 == count or (i + 1) % max(1, count // 20) == 0:
            print(f"selected representative seed {i + 1}/{count}")
    return selected


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
) -> None:
    fig = plt.figure(figsize=(32, 8))

    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax1.voxels(
        occupancy,
        facecolors=build_density_facecolors(occupancy, display_density),
        edgecolor="#111827",
        linewidth=0.05,
    )
    configure_voxel_axes(
        ax1,
        displayed_shape=occupancy.shape,
        original_shape=original_shape,
        steps=display_steps,
        title="1) Fake density result",
    )

    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    ax2.voxels(
        occupancy,
        facecolors=build_probability_facecolors(occupancy, display_probability),
        edgecolor="#111827",
        linewidth=0.05,
    )
    configure_voxel_axes(
        ax2,
        displayed_shape=occupancy.shape,
        original_shape=original_shape,
        steps=display_steps,
        title=f"2) Spatial probability map (gamma={gamma:.2f})",
    )

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
        cycles_per_segment=2.4,
        amplitude_ratio=0.05,
        wire_radius_ratio=0.82,
        tube_sides=14,
        min_steps=42,
        steps_per_cycle=20,
    )
    z_ratio = original_shape[2] / original_shape[0]
    total_edges = len(tiled_edges)
    for edge_index, (start, end) in enumerate(tiled_edges, start=1):
        start_scaled = np.array([start[0], start[1], start[2] * z_ratio], dtype=float)
        end_scaled = np.array([end[0], end[1], end[2] * z_ratio], dtype=float)
        helix_style.draw_segment(
            ax4,
            start_scaled,
            end_scaled,
            radius=0.010,
            color="#7c3aed",
        )
        if edge_index == 1 or edge_index == total_edges or edge_index % 500 == 0:
            print(f"rendered helix edge {edge_index}/{total_edges}")
    configure_rod_axes(
        ax4,
        xlim=(0.0, 3.0),
        ylim=(0.0, 3.0),
        zlim=(0.0, 3.0 * z_ratio),
        box_aspect=(3.0, 3.0, 3.0 * z_ratio),
        title=f"4) 3x3x3 helix Voronoi cell (repr. seeds={representative_seed_count})",
    )

    fig.suptitle(
        "Fake density → probability → 100k seeds → 3x3x3 helix Voronoi",
        fontsize=20,
        y=0.99,
    )
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)
    print(f"saved quad figure: {output_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 4-panel figure from fake density, seed probability, and a reduced helix Voronoi unit."
    )
    parser.add_argument(
        "fake_density_npz",
        type=Path,
        nargs="?",
        default=Path("datasets/topopt/fake_density_annular_cylinder_full.npz"),
        help="Fake density result NPZ.",
    )
    parser.add_argument(
        "seed_mapping_npz",
        type=Path,
        nargs="?",
        default=Path("datasets/topopt/seed_probability_mapping_100k.npz"),
        help="Seed probability mapping NPZ containing 100k sampled seeds.",
    )
    parser.add_argument(
        "--representative-seeds",
        type=int,
        default=81,
        help="Reduced seed count used for the exact Voronoi/helix stage.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.8,
        help="Gamma used for the shown probability map title.",
    )
    parser.add_argument(
        "--max-display-size",
        type=int,
        default=72,
        help="Maximum displayed voxel resolution on each axis for panels 1 and 2.",
    )
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=(999, 999, 399),
        metavar=("X", "Y", "Z"),
        help="3-friendly target shape used for normalized seed interpretation.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=15000,
        help="Maximum candidate points retained before representative-seed FPS.",
    )
    parser.add_argument(
        "--candidate-voxel-bins",
        type=int,
        nargs=3,
        default=(48, 48, 24),
        metavar=("BX", "BY", "BZ"),
        help="Voxel bins used to spatially compress the 100k seed cloud before FPS.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Random seed used by candidate-pool downsampling.",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("datasets/topopt/voronoi_helix_quad_100k_to_81.npz"),
        help="Output NPZ with representative seeds and edges.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("docs/assets/voronoi_helix_quad_100k_to_81.png"),
        help="Output quad figure.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fake_density = load_result(args.fake_density_npz)
    seed_mapping = load_result(args.seed_mapping_npz)

    density_milli = fake_density["density_milli"]
    full_seed_points = seed_mapping["seed_points"].astype(np.float32)
    occupancy, display_density, display_steps = downsample_density(
        density_milli=density_milli,
        max_display_size=args.max_display_size,
    )
    display_probability = density_to_probability_intensity(
        density=display_density,
        gamma=args.gamma,
    )

    target_shape = tuple(int(v) for v in args.target_shape)
    normalized_seed_points = normalize_seed_points(full_seed_points, target_shape)
    candidate_points = build_candidate_pool(
        normalized_seed_points,
        candidate_limit=args.candidate_limit,
        voxel_bins=tuple(args.candidate_voxel_bins),
        rng_seed=args.rng_seed,
    )
    representative_seeds = farthest_point_sampling(
        candidate_points,
        count=args.representative_seeds,
    )
    print("building Voronoi cells...")
    cells, halfspace_sets = build_voronoi_cells(
        representative_seeds,
        progress_every=max(1, args.representative_seeds // 20),
        progress=print,
    )
    print("extracting unique edges...")
    edges = extract_unique_edges(
        cells,
        halfspace_sets,
        progress_every=max(1, args.representative_seeds // 20),
        progress=print,
    )
    print(f"representative seeds: {len(representative_seeds)}")
    print(f"cells: {len(cells)}")
    print(f"unique edges: {len(edges)}")

    tiled_edges = tile_edges(edges, repeats=(3, 3, 3))
    print(f"tiled edges: {len(tiled_edges)}")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        representative_seeds=representative_seeds.astype(np.float32),
        num_representative_seeds=np.array(len(representative_seeds), dtype=np.int32),
        full_seed_points=full_seed_points.astype(np.float32),
        num_input_seeds=np.array(len(full_seed_points), dtype=np.int32),
        candidate_points=candidate_points.astype(np.float32),
        num_candidate_points=np.array(len(candidate_points), dtype=np.int32),
        target_shape=np.array(target_shape, dtype=np.int32),
        edge_starts=np.asarray([start for start, _ in edges], dtype=np.float32),
        edge_ends=np.asarray([end for _, end in edges], dtype=np.float32),
        tiled_edge_starts=np.asarray([start for start, _ in tiled_edges], dtype=np.float32),
        tiled_edge_ends=np.asarray([end for _, end in tiled_edges], dtype=np.float32),
    )
    print(f"saved Voronoi helix data: {args.output_npz}")

    render_quad(
        output_png=args.output_png,
        occupancy=occupancy,
        display_density=display_density,
        display_probability=display_probability,
        display_steps=display_steps,
        original_shape=density_milli.shape,
        seed_points=full_seed_points,
        tiled_edges=tiled_edges,
        representative_seed_count=len(representative_seeds),
        num_input_seeds=len(full_seed_points),
        gamma=args.gamma,
    )
