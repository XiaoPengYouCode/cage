from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from helix_voronoi.rods import HelixRodStyle
from helix_voronoi.voronoi import build_voronoi_cells, extract_unique_edges


def load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def make_blue_red_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "blue_to_red_density",
        [(0.00, "#2563eb"), (0.35, "#38bdf8"), (0.70, "#fb7185"), (1.00, "#b91c1c")],
    )


def make_probability_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "probability_map",
        [(0.00, "#dbeafe"), (0.20, "#60a5fa"), (0.45, "#22c55e"), (0.70, "#f59e0b"), (1.00, "#dc2626")],
    )


def downsample_density(density_milli: np.ndarray, max_xy: int, max_z: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    sx = max(1, int(np.ceil(density_milli.shape[0] / max_xy)))
    sy = max(1, int(np.ceil(density_milli.shape[1] / max_xy)))
    sz = max(1, int(np.ceil(density_milli.shape[2] / max_z)))
    steps = (sx, sy, sz)
    density = density_milli[::sx, ::sy, ::sz].astype(np.float32) / 1000.0
    occupancy = density > 0.0
    return occupancy, density, steps


def density_to_probability(density: np.ndarray, gamma: float) -> np.ndarray:
    weights = np.where(density > 0.0, density**gamma, 0.0)
    max_w = float(weights.max()) if weights.size else 0.0
    if max_w <= 0.0:
        return np.zeros_like(density, dtype=np.float32)
    return (weights / max_w).astype(np.float32)


def build_facecolors(voxels: np.ndarray, scalar: np.ndarray, cmap: LinearSegmentedColormap, alpha_min: float, alpha_max: float) -> np.ndarray:
    normalized = np.clip(scalar, 0.0, 1.0)
    facecolors = cmap(normalized)
    alpha = alpha_min + (alpha_max - alpha_min) * normalized
    facecolors[..., 3] = np.where(voxels, alpha, 0.0)
    return facecolors


def configure_voxel_axes(ax: plt.Axes, displayed_shape: tuple[int, int, int], full_shape: tuple[int, int, int], steps: tuple[int, int, int], title: str) -> None:
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
    ax.set_box_aspect(full_shape)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=38)
    ax.set_title(title)


def filter_valid_original_seeds(seed_points: np.ndarray, density_milli: np.ndarray) -> np.ndarray:
    coords = np.floor(seed_points).astype(np.int32)
    coords[:, 0] = np.clip(coords[:, 0], 0, density_milli.shape[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, density_milli.shape[1] - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, density_milli.shape[2] - 1)
    valid = density_milli[coords[:, 0], coords[:, 1], coords[:, 2]] > 0
    filtered = seed_points[valid].astype(np.float32)
    print(f"valid original seeds inside material: {len(filtered)} / {len(seed_points)}")
    return filtered


def tile_seed_points(seed_points: np.ndarray, base_shape: tuple[int, int, int], repeats: tuple[int, int, int]) -> np.ndarray:
    tiled = []
    base = np.array(base_shape, dtype=np.float32)
    for ox in range(repeats[0]):
        for oy in range(repeats[1]):
            for oz in range(repeats[2]):
                offset = np.array([ox, oy, oz], dtype=np.float32) * base
                tiled.append(seed_points + offset)
    result = np.vstack(tiled).astype(np.float32)
    print(f"tiled valid seeds: {len(result)}")
    return result


def build_candidate_pool(points: np.ndarray, full_shape: tuple[int, int, int], voxel_bins: tuple[int, int, int], candidate_limit: int, rng_seed: int) -> np.ndarray:
    bins = np.array(voxel_bins, dtype=np.int32)
    normalized = np.clip(points / np.array(full_shape, dtype=np.float32), 0.0, 1.0 - 1e-7)
    idx = np.floor(normalized * bins[None, :]).astype(np.int32)
    keys = np.ravel_multi_index(idx.T, bins)
    _, unique_indices = np.unique(keys, return_index=True)
    pooled = points[np.sort(unique_indices)].astype(np.float32)
    print(f"candidate pool after bucketing: {len(pooled)}")
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
    dist_centroid = np.sum((points - centroid) ** 2, axis=1, dtype=np.float32)
    first_idx = int(np.argmax(dist_centroid))
    selected[0] = points[first_idx]
    min_dist_sq = np.sum((points - selected[0]) ** 2, axis=1, dtype=np.float32)
    for i in range(1, count):
        next_idx = int(np.argmax(min_dist_sq))
        selected[i] = points[next_idx]
        dist_sq = np.sum((points - selected[i]) ** 2, axis=1, dtype=np.float32)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)
        if i == 1 or i + 1 == count or (i + 1) % max(1, count // 20) == 0:
            print(f"selected representative seed {i + 1}/{count}")
    return selected


def normalize_to_unit(points: np.ndarray, full_shape: tuple[int, int, int]) -> np.ndarray:
    eps = 1e-5
    normalized = points / np.array(full_shape, dtype=np.float32)
    return np.clip(normalized, eps, 1.0 - eps)


def sample_edges_for_preview(edges: list[tuple[np.ndarray, np.ndarray]], max_render_edges: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(edges) <= max_render_edges:
        print(f"render edge count unchanged: {len(edges)}")
        return edges
    indices = np.linspace(0, len(edges) - 1, max_render_edges, dtype=int)
    sampled = [edges[idx] for idx in indices]
    print(f"render edge count reduced: {len(edges)} -> {len(sampled)}")
    return sampled


def render_four_panel(output_png: Path, tiled_occ: np.ndarray, tiled_density: np.ndarray, tiled_probability: np.ndarray, tiled_steps: tuple[int, int, int], tiled_full_shape: tuple[int, int, int], tiled_seed_points_preview: np.ndarray, render_edges: list[tuple[np.ndarray, np.ndarray]], representative_seed_count: int) -> None:
    fig = plt.figure(figsize=(24, 8))
    density_cmap = make_blue_red_cmap()
    probability_cmap = make_probability_cmap()

    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax1.voxels(tiled_occ, facecolors=build_facecolors(tiled_occ, tiled_density, density_cmap, 0.22, 0.90), edgecolor="#111827", linewidth=0.04)
    configure_voxel_axes(ax1, tiled_occ.shape, tiled_full_shape, tiled_steps, "1) 3x3x3 tiled material / density")

    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    ax2.voxels(tiled_occ, facecolors=build_facecolors(tiled_occ, tiled_probability, probability_cmap, 0.14, 0.88), edgecolor="#111827", linewidth=0.04)
    configure_voxel_axes(ax2, tiled_occ.shape, tiled_full_shape, tiled_steps, "2) 3x3x3 tiled probability")

    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    preview_colors = np.clip(tiled_seed_points_preview[:, 2] / max(tiled_full_shape[2], 1), 0.0, 1.0)
    ax3.scatter(tiled_seed_points_preview[:, 0], tiled_seed_points_preview[:, 1], tiled_seed_points_preview[:, 2], c=preview_colors, cmap="coolwarm", s=0.45, alpha=0.22, linewidths=0)
    ax3.set_xlim(0, tiled_full_shape[0])
    ax3.set_ylim(0, tiled_full_shape[1])
    ax3.set_zlim(0, tiled_full_shape[2])
    ax3.set_box_aspect(tiled_full_shape)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.view_init(elev=24, azim=38)
    ax3.set_title(f"3) Valid tiled seeds preview ({len(tiled_seed_points_preview):,})")

    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    helix_style = HelixRodStyle(cycles_per_segment=2.1, amplitude_ratio=0.045, wire_radius_ratio=0.80, tube_sides=8, min_steps=18, steps_per_cycle=10)
    z_ratio = tiled_full_shape[2] / tiled_full_shape[0]
    total_edges = len(render_edges)
    for edge_index, (start, end) in enumerate(render_edges, start=1):
        start_scaled = np.array([start[0] * 3.0, start[1] * 3.0, start[2] * 3.0 * z_ratio], dtype=float)
        end_scaled = np.array([end[0] * 3.0, end[1] * 3.0, end[2] * 3.0 * z_ratio], dtype=float)
        helix_style.draw_segment(ax4, start_scaled, end_scaled, radius=0.010, color="#7c3aed")
        if edge_index == 1 or edge_index == total_edges or edge_index % 250 == 0:
            print(f"rendered helix edge {edge_index}/{total_edges}")
    ax4.set_xlim(0.0, 3.0)
    ax4.set_ylim(0.0, 3.0)
    ax4.set_zlim(0.0, 3.0 * z_ratio)
    ax4.set_box_aspect((3.0, 3.0, 3.0 * z_ratio))
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_zlabel("z")
    ax4.view_init(elev=24, azim=38)
    ax4.set_title(f"4) Helix Voronoi from tiled valid seeds (repr.={representative_seed_count})")

    fig.suptitle("3x3x3 tiled valid-seed domain → helix Voronoi preview", fontsize=18, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05, wspace=0.08)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=110)
    plt.close(fig)
    print(f"saved preview: {output_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview helix Voronoi built from valid seeds inside a 3x3x3 tiled material domain.")
    parser.add_argument("--fake-density-npz", type=Path, default=Path("datasets/topopt/fake_density_annular_cylinder_full.npz"))
    parser.add_argument("--seed-mapping-npz", type=Path, default=Path("datasets/topopt/seed_probability_mapping_100k.npz"))
    parser.add_argument("--gamma", type=float, default=1.8)
    parser.add_argument("--max-display-xy", type=int, default=24)
    parser.add_argument("--max-display-z", type=int, default=12)
    parser.add_argument("--representative-seeds", type=int, default=81)
    parser.add_argument("--candidate-limit", type=int, default=6000)
    parser.add_argument("--candidate-voxel-bins", type=int, nargs=3, default=(72, 72, 24), metavar=("BX", "BY", "BZ"))
    parser.add_argument("--seed-preview-count", type=int, default=25000)
    parser.add_argument("--max-render-edges", type=int, default=2500)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--output-png", type=Path, default=Path("docs/assets/tiled_valid_seed_helix_voronoi_preview.png"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fake_density = load_npz(args.fake_density_npz)
    seed_mapping = load_npz(args.seed_mapping_npz)

    density_milli = fake_density["density_milli"]
    base_shape = density_milli.shape
    tiled_full_shape = (base_shape[0] * 3, base_shape[1] * 3, base_shape[2] * 3)
    seed_points = seed_mapping["seed_points"].astype(np.float32)

    valid_original_seeds = filter_valid_original_seeds(seed_points, density_milli)
    tiled_valid_seeds = tile_seed_points(valid_original_seeds, base_shape, repeats=(3, 3, 3))

    occ, density, steps = downsample_density(density_milli, args.max_display_xy, args.max_display_z)
    probability = density_to_probability(density, args.gamma)
    tiled_occ = np.tile(occ, (3, 3, 3))
    tiled_density = np.tile(density, (3, 3, 3))
    tiled_probability = np.tile(probability, (3, 3, 3))
    tiled_steps = steps

    candidate_points = build_candidate_pool(
        tiled_valid_seeds,
        full_shape=tiled_full_shape,
        voxel_bins=tuple(args.candidate_voxel_bins),
        candidate_limit=args.candidate_limit,
        rng_seed=args.rng_seed,
    )
    representative_points = farthest_point_sampling(candidate_points, args.representative_seeds)
    normalized_representatives = normalize_to_unit(representative_points, tiled_full_shape)

    print("building Voronoi cells...")
    cells, halfspace_sets = build_voronoi_cells(
        normalized_representatives,
        progress_every=max(1, args.representative_seeds // 20),
        progress=print,
    )
    print("extracting edges...")
    edges = extract_unique_edges(
        cells,
        halfspace_sets,
        progress_every=max(1, args.representative_seeds // 20),
        progress=print,
    )
    print(f"representative seeds: {len(representative_points)}")
    print(f"unique edges: {len(edges)}")
    render_edges = sample_edges_for_preview(edges, args.max_render_edges)

    preview_seed_points = tiled_valid_seeds
    if len(preview_seed_points) > args.seed_preview_count:
        rng = np.random.default_rng(args.rng_seed)
        selection = rng.choice(len(preview_seed_points), size=args.seed_preview_count, replace=False)
        preview_seed_points = preview_seed_points[selection]
        print(f"seed preview downsampled to: {len(preview_seed_points)}")

    render_four_panel(
        output_png=args.output_png,
        tiled_occ=tiled_occ,
        tiled_density=tiled_density,
        tiled_probability=tiled_probability,
        tiled_steps=tiled_steps,
        tiled_full_shape=tiled_full_shape,
        tiled_seed_points_preview=preview_seed_points,
        render_edges=render_edges,
        representative_seed_count=len(representative_points),
    )
