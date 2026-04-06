from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from topopt_sampling.probability import density_to_probability_intensity


def load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def build_density_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "density_pipeline",
        [
            (0.00, "#020617"),
            (0.20, "#1d4ed8"),
            (0.48, "#38bdf8"),
            (0.75, "#fb7185"),
            (1.00, "#7f1d1d"),
        ],
    )


def build_probability_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "sampling_pipeline",
        [
            (0.00, "#f8fafc"),
            (0.18, "#bfdbfe"),
            (0.45, "#60a5fa"),
            (0.72, "#2563eb"),
            (1.00, "#0f172a"),
        ],
    )


def build_voxel_facecolors(values: np.ndarray, cmap: LinearSegmentedColormap) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(values, 0.0, 1.0)
    occupancy = clipped > 0.0
    facecolors = cmap(clipped)
    alpha = 0.50 + 0.40 * clipped
    facecolors[..., 3] = np.where(occupancy, alpha, 0.0)
    return occupancy, facecolors


def aggregate_scalar_field_for_display(
    values: np.ndarray,
    occupancy_mask: np.ndarray,
    block_shape: tuple[int, int, int] = (4, 4, 2),
) -> np.ndarray:
    bx, by, bz = block_shape
    sx, sy, sz = values.shape
    pad_x = (-sx) % bx
    pad_y = (-sy) % by
    pad_z = (-sz) % bz

    padded_values = np.pad(values, ((0, pad_x), (0, pad_y), (0, pad_z)), mode="constant")
    padded_mask = np.pad(occupancy_mask, ((0, pad_x), (0, pad_y), (0, pad_z)), mode="constant")

    reshaped_values = padded_values.reshape(
        padded_values.shape[0] // bx,
        bx,
        padded_values.shape[1] // by,
        by,
        padded_values.shape[2] // bz,
        bz,
    )
    reshaped_mask = padded_mask.reshape(
        padded_mask.shape[0] // bx,
        bx,
        padded_mask.shape[1] // by,
        by,
        padded_mask.shape[2] // bz,
        bz,
    )

    active_sum = (reshaped_values * reshaped_mask.astype(np.float32)).sum(axis=(1, 3, 5))
    active_count = reshaped_mask.sum(axis=(1, 3, 5))
    aggregated = np.divide(
        active_sum,
        np.maximum(active_count, 1),
        out=np.zeros_like(active_sum, dtype=np.float32),
        where=active_count > 0,
    )
    return aggregated.astype(np.float32)


def configure_3d_axes(
    ax: plt.Axes,
    rendered_shape: tuple[int, int, int],
    title: str,
    original_shape: tuple[int, int, int] | None = None,
) -> None:
    source_shape = rendered_shape if original_shape is None else original_shape
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(0, rendered_shape[0])
    ax.set_ylim(0, rendered_shape[1])
    ax.set_zlim(0, rendered_shape[2])
    ax.set_box_aspect(source_shape)
    ax.view_init(elev=21, azim=38)

    x_ticks = np.linspace(0, rendered_shape[0], 5)
    y_ticks = np.linspace(0, rendered_shape[1], 5)
    z_ticks = np.linspace(0, rendered_shape[2], 5)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)
    ax.set_xticklabels([str(int(round(v))) for v in np.linspace(0, source_shape[0], 5)])
    ax.set_yticklabels([str(int(round(v))) for v in np.linspace(0, source_shape[1], 5)])
    ax.set_zticklabels([str(int(round(v))) for v in np.linspace(0, source_shape[2], 5)])


def render_overview(
    density_npz: Path,
    seed_npz: Path,
    output_png: Path,
) -> None:
    density_result = load_npz(density_npz)
    seed_result = load_npz(seed_npz)

    density = density_result["density_milli"].astype(np.float32) / 1000.0
    seed_points = seed_result["seed_points"].astype(np.float32)
    gamma = float(seed_result["gamma"].item())
    num_seeds = int(seed_result["num_seeds"].item())
    probability = density_to_probability_intensity(density, gamma)
    occupancy_mask = density > 0.0
    display_density = aggregate_scalar_field_for_display(density, occupancy_mask)
    display_probability = aggregate_scalar_field_for_display(probability, occupancy_mask)

    density_cmap = build_density_cmap()
    probability_cmap = build_probability_cmap()
    density_occ, density_facecolors = build_voxel_facecolors(display_density, density_cmap)
    probability_occ, probability_facecolors = build_voxel_facecolors(display_probability, probability_cmap)

    fig = plt.figure(figsize=(20.5, 6.6))
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    fig.subplots_adjust(left=0.03, right=0.985, top=0.88, bottom=0.10, wspace=0.18)

    ax0.voxels(
        density_occ,
        facecolors=density_facecolors,
        edgecolor=(1.0, 1.0, 1.0, 0.28),
        linewidth=0.09,
        shade=False,
    )
    configure_3d_axes(ax0, display_density.shape, "1) Fake Topopt Density", density.shape)
    density_mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=1.0), cmap=density_cmap)
    density_mappable.set_array([])
    fig.colorbar(density_mappable, ax=ax0, fraction=0.040, pad=0.06, label="density")

    ax1.voxels(
        probability_occ,
        facecolors=probability_facecolors,
        edgecolor=(1.0, 1.0, 1.0, 0.28),
        linewidth=0.09,
        shade=False,
    )
    configure_3d_axes(ax1, display_probability.shape, f"2) Probability Field (gamma={gamma:.1f})", density.shape)
    probability_mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=1.0), cmap=probability_cmap)
    probability_mappable.set_array([])
    fig.colorbar(probability_mappable, ax=ax1, fraction=0.040, pad=0.06, label="probability")

    scatter = ax2.scatter(
        seed_points[:, 0],
        seed_points[:, 1],
        seed_points[:, 2],
        c=seed_points[:, 2],
        s=7,
        cmap="viridis",
        alpha=0.65,
        linewidths=0.0,
    )
    configure_3d_axes(ax2, density.shape, f"3) Random Seeds (n={num_seeds})", density.shape)
    fig.colorbar(scatter, ax=ax2, fraction=0.040, pad=0.06, label="seed z")

    fig.suptitle(
        "Topology density -> probability -> random seed sampling",
        fontsize=16,
    )
    fig.text(
        0.5,
        0.01,
        f"density grid={density.shape[0]}x{density.shape[1]}x{density.shape[2]} | no downsampling",
        ha="center",
        fontsize=10,
        color="#334155",
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved overview: {output_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the topology-density to seed-sampling overview figure.")
    parser.add_argument(
        "--density-npz",
        type=Path,
        default=Path("datasets/topopt/fake_density_annular_cylinder_200x200x80.npz"),
    )
    parser.add_argument(
        "--seed-npz",
        type=Path,
        default=Path("datasets/topopt/seed_probability_mapping_2000.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/topopt_sampling_pipeline_overview.png"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render_overview(args.density_npz, args.seed_npz, args.output)
