from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def max_project(volume: np.ndarray) -> np.ndarray:
    return np.max(volume, axis=2)


def seed_xy_histogram(seed_points: np.ndarray, shape_xy: tuple[int, int]) -> np.ndarray:
    x_bins = np.linspace(0.0, shape_xy[0], shape_xy[0] + 1)
    y_bins = np.linspace(0.0, shape_xy[1], shape_xy[1] + 1)
    histogram, _, _ = np.histogram2d(
        seed_points[:, 1],
        seed_points[:, 0],
        bins=(y_bins, x_bins),
    )
    return histogram.astype(np.float32)


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


def render_overview(
    density_npz: Path,
    seed_npz: Path,
    output_png: Path,
) -> None:
    density_result = load_npz(density_npz)
    seed_result = load_npz(seed_npz)

    density = density_result["density_milli"].astype(np.float32) / 1000.0
    display_density = seed_result["display_density"].astype(np.float32)
    display_probability = seed_result["display_probability"].astype(np.float32)
    seed_points = seed_result["seed_points"].astype(np.float32)
    gamma = float(seed_result["gamma"].item())
    num_seeds = int(seed_result["num_seeds"].item())

    density_projection = max_project(density)
    probability_projection = max_project(display_probability)
    seed_density = seed_xy_histogram(seed_points, density.shape[:2])

    cmap = build_probability_cmap()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    im0 = axes[0].imshow(density_projection.T, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    axes[0].set_title("1) Topopt Density")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="max density over z")

    im1 = axes[1].imshow(probability_projection.T, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)
    axes[1].set_title(f"2) Probability Field (gamma={gamma:.1f})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="max probability over z")

    im2 = axes[2].imshow(seed_density.T, origin="lower", cmap="viridis")
    axes[2].scatter(
        seed_points[:, 0],
        seed_points[:, 1],
        c=seed_points[:, 2],
        s=6,
        cmap="viridis",
        alpha=0.55,
        linewidths=0.0,
    )
    axes[2].set_title(f"3) Random Seeds (n={num_seeds})")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="seed count per xy voxel")

    fig.suptitle(
        "Topology density -> probability -> random seed sampling",
        fontsize=16,
    )
    fig.text(
        0.5,
        0.01,
        f"density grid={density.shape[0]}x{density.shape[1]}x{density.shape[2]} | display grid={display_density.shape[0]}x{display_density.shape[1]}x{display_density.shape[2]}",
        ha="center",
        fontsize=10,
        color="#334155",
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
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
