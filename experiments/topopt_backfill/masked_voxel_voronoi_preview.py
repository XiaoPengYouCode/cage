from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from scipy.spatial import cKDTree


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
    max_display_xy: int,
    max_display_z: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    sx = max(1, int(np.ceil(density_milli.shape[0] / max_display_xy)))
    sy = max(1, int(np.ceil(density_milli.shape[1] / max_display_xy)))
    sz = max(1, int(np.ceil(density_milli.shape[2] / max_display_z)))
    steps = (sx, sy, sz)
    sampled_density = density_milli[::sx, ::sy, ::sz].astype(np.float32) / 1000.0
    occupancy = sampled_density > 0.0
    return occupancy, sampled_density, steps


def density_to_probability(density: np.ndarray, gamma: float) -> np.ndarray:
    prob = np.where(density > 0.0, density**gamma, 0.0)
    max_value = float(prob.max()) if prob.size else 0.0
    if max_value <= 0.0:
        return np.zeros_like(density, dtype=np.float32)
    return (prob / max_value).astype(np.float32)


def build_facecolors_from_scalar(
    voxels: np.ndarray,
    scalar: np.ndarray,
    cmap: LinearSegmentedColormap,
    alpha_min: float,
    alpha_max: float,
) -> np.ndarray:
    normalized = np.clip(scalar, 0.0, 1.0)
    facecolors = cmap(normalized)
    alpha = alpha_min + (alpha_max - alpha_min) * normalized
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


def coarse_unique_seed_voxels(
    seed_points: np.ndarray,
    occupancy_shape: tuple[int, int, int],
    steps: tuple[int, int, int],
    original_shape: tuple[int, int, int],
    mask: np.ndarray,
    max_seed_voxels: int,
    rng_seed: int,
) -> np.ndarray:
    coords = np.floor(seed_points / np.array(steps, dtype=np.float32)).astype(np.int32)
    coords[:, 0] = np.clip(coords[:, 0], 0, occupancy_shape[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, occupancy_shape[1] - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, occupancy_shape[2] - 1)
    inside = mask[coords[:, 0], coords[:, 1], coords[:, 2]]
    coords = coords[inside]
    if len(coords) == 0:
        raise ValueError("No coarse seeds remain inside the constrained mask.")
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    coarse = coords[np.sort(unique_indices)]
    print(f"coarse unique seeds inside mask: {len(coarse)}")
    if len(coarse) > max_seed_voxels:
        rng = np.random.default_rng(rng_seed)
        selection = rng.choice(len(coarse), size=max_seed_voxels, replace=False)
        coarse = coarse[selection]
        print(f"coarse seeds downsampled to: {len(coarse)}")
    return coarse.astype(np.int32)


def assign_masked_voronoi(mask: np.ndarray, seed_voxels: np.ndarray) -> np.ndarray:
    active = np.argwhere(mask)
    tree = cKDTree(seed_voxels.astype(np.float32))
    _, indices = tree.query(active.astype(np.float32), workers=-1)
    labels = np.full(mask.shape, fill_value=-1, dtype=np.int32)
    labels[active[:, 0], active[:, 1], active[:, 2]] = indices.astype(np.int32)
    return labels


def labels_to_scalar(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    scalar = np.zeros(labels.shape, dtype=np.float32)
    active = labels[mask]
    if active.size == 0:
        return scalar
    max_label = max(int(active.max()), 1)
    scalar[mask] = active.astype(np.float32) / max_label
    return scalar


def render_preview(
    output_png: Path,
    occupancy: np.ndarray,
    probability: np.ndarray,
    steps: tuple[int, int, int],
    original_shape: tuple[int, int, int],
    seed_voxels: np.ndarray,
    padded_mask: np.ndarray,
    labels: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(24, 8))

    density_cmap = make_blue_red_cmap()
    prob_cmap = make_probability_cmap()

    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax1.voxels(
        occupancy,
        facecolors=build_facecolors_from_scalar(occupancy, probability ** (1 / 1.8), density_cmap, 0.22, 0.90),
        edgecolor="#111827",
        linewidth=0.05,
    )
    configure_voxel_axes(ax1, occupancy.shape, original_shape, steps, "1) Material mask / density domain")

    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    ax2.voxels(
        padded_mask,
        facecolors=build_facecolors_from_scalar(padded_mask, probability * padded_mask, prob_cmap, 0.12, 0.88),
        edgecolor="#111827",
        linewidth=0.05,
    )
    configure_voxel_axes(ax2, padded_mask.shape, original_shape, steps, "2) Mask-constrained probability domain")

    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    ax3.scatter(
        seed_voxels[:, 0] + 0.5,
        seed_voxels[:, 1] + 0.5,
        seed_voxels[:, 2] + 0.5,
        c=seed_voxels[:, 2],
        cmap="coolwarm",
        s=8,
        alpha=0.65,
        linewidths=0,
    )
    ax3.set_xlim(0, occupancy.shape[0])
    ax3.set_ylim(0, occupancy.shape[1])
    ax3.set_zlim(0, occupancy.shape[2])
    ax3.set_box_aspect(original_shape)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.view_init(elev=24, azim=38)
    ax3.set_title(f"3) Seeds inside mask ({len(seed_voxels)})")

    label_scalar = labels_to_scalar(labels, padded_mask)
    partition_cmap = plt.cm.tab20
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    facecolors = partition_cmap(np.mod(label_scalar * 20.0, 1.0))
    facecolors[..., 3] = np.where(padded_mask, 0.92, 0.0)
    ax4.voxels(
        padded_mask,
        facecolors=facecolors,
        edgecolor="#111827",
        linewidth=0.03,
    )
    configure_voxel_axes(ax4, padded_mask.shape, original_shape, steps, "4) Masked voxel Voronoi partition")

    fig.suptitle("Masked voxel Voronoi preview (domain constrained to nonzero material)", fontsize=18, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05, wspace=0.08)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=120)
    plt.close(fig)
    print(f"saved preview: {output_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview mask-constrained voxel Voronoi on nonzero material domain.")
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
        "--gamma",
        type=float,
        default=1.8,
    )
    parser.add_argument(
        "--max-display-xy",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max-display-z",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--fill-iterations",
        type=int,
        default=1,
        help="Conservative 3x3x3-style mask completion by binary dilation iterations.",
    )
    parser.add_argument(
        "--max-seed-voxels",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("docs/assets/masked_voxel_voronoi_preview.png"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fake_density = load_npz(args.fake_density_npz)
    seed_mapping = load_npz(args.seed_mapping_npz)

    density_milli = fake_density["density_milli"]
    seed_points = seed_mapping["seed_points"].astype(np.float32)
    occupancy, density, steps = downsample_density(
        density_milli,
        max_display_xy=args.max_display_xy,
        max_display_z=args.max_display_z,
    )
    probability = density_to_probability(density, args.gamma)

    structure = np.ones((3, 3, 3), dtype=bool)
    padded_mask = ndimage.binary_dilation(
        occupancy,
        structure=structure,
        iterations=args.fill_iterations,
    )
    print(f"occupancy voxels: {int(occupancy.sum())}")
    print(f"padded mask voxels: {int(padded_mask.sum())}")

    seed_voxels = coarse_unique_seed_voxels(
        seed_points=seed_points,
        occupancy_shape=occupancy.shape,
        steps=steps,
        original_shape=density_milli.shape,
        mask=padded_mask,
        max_seed_voxels=args.max_seed_voxels,
        rng_seed=args.rng_seed,
    )
    print("assigning masked Voronoi labels...")
    labels = assign_masked_voronoi(padded_mask, seed_voxels)

    render_preview(
        output_png=args.output_png,
        occupancy=occupancy,
        probability=probability,
        steps=steps,
        original_shape=density_milli.shape,
        seed_voxels=seed_voxels,
        padded_mask=padded_mask,
        labels=labels,
    )
