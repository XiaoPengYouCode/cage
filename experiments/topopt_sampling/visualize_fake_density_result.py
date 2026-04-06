from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def load_result(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def downsample_result(
    voxels: np.ndarray,
    density_milli: np.ndarray,
    max_display_size: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    steps = tuple(max(1, int(np.ceil(size / max_display_size))) for size in voxels.shape)
    voxels_ds = voxels[:: steps[0], :: steps[1], :: steps[2]] > 0
    density_ds = density_milli[:: steps[0], :: steps[1], :: steps[2]].astype(np.float32) / 1000.0
    density_ds = np.where(voxels_ds, density_ds, 0.0)
    return voxels_ds, density_ds, steps


def build_facecolors(voxels: np.ndarray, density: np.ndarray) -> np.ndarray:
    normalized = np.clip(density, 0.0, 1.0)
    cmap = LinearSegmentedColormap.from_list(
        "blue_to_red_density",
        [
            (0.00, "#2563eb"),
            (0.35, "#38bdf8"),
            (0.70, "#fb7185"),
            (1.00, "#b91c1c"),
        ],
    )
    facecolors = cmap(normalized)
    alpha = 0.22 + 0.68 * normalized
    facecolors[..., 3] = np.where(voxels, alpha, 0.0)
    return facecolors


def plot_density_voxels(
    voxels: np.ndarray,
    density: np.ndarray,
    steps: tuple[int, int, int],
    original_shape: tuple[int, int, int],
    output_path: Path | None,
    show: bool,
    title: str,
) -> None:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    facecolors = build_facecolors(voxels, density)
    ax.voxels(voxels, facecolors=facecolors, edgecolor="#111827", linewidth=0.06)

    displayed_shape = voxels.shape
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

    density_cmap = LinearSegmentedColormap.from_list(
        "blue_to_red_density",
        [
            (0.00, "#2563eb"),
            (0.35, "#38bdf8"),
            (0.70, "#fb7185"),
            (1.00, "#b91c1c"),
        ],
    )
    norm = plt.Normalize(vmin=0.001, vmax=1.0)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=density_cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.06, label="fake density")

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220)
        print(f"saved figure: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize fake FEA-like density results.")
    parser.add_argument(
        "result_npz",
        type=Path,
        nargs="?",
        default=Path("datasets/topopt/fake_density_annular_cylinder_200x200x80.npz"),
        help="Fake density result NPZ.",
    )
    parser.add_argument(
        "--max-display-size",
        type=int,
        default=96,
        help="Maximum voxel resolution used for block rendering on each axis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/fake_density_annular_cylinder_200x200x80.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the matplotlib window.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = load_result(args.result_npz)
    voxels = result["voxels"]
    density_milli = result["density_milli"]
    voxels_ds, density_ds, steps = downsample_result(
        voxels=voxels,
        density_milli=density_milli,
        max_display_size=args.max_display_size,
    )
    positive = density_milli[density_milli > 0]
    title = (
        f"Fake density result | grid={voxels.shape[0]}x{voxels.shape[1]}x{voxels.shape[2]} | "
        f"display={voxels_ds.shape[0]}x{voxels_ds.shape[1]}x{voxels_ds.shape[2]} | "
        f"density={positive.min() / 1000:.3f}..{positive.max() / 1000:.3f}"
    )
    print(f"loaded: {args.result_npz}")
    print(f"shape: {voxels.shape}")
    print(f"active voxels: {int(voxels.sum())}")
    print(f"density range: {positive.min() / 1000:.3f} .. {positive.max() / 1000:.3f}")
    print(f"display grid: {voxels_ds.shape}")
    print(f"display step: {steps}")
    plot_density_voxels(
        voxels=voxels_ds,
        density=density_ds,
        steps=steps,
        original_shape=voxels.shape,
        output_path=args.output,
        show=args.show,
        title=title,
    )
