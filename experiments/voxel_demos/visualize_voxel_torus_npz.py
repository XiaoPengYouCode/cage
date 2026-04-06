from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_voxels(npz_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(npz_path) as data:
        voxels = data["voxels"]
        metadata = {key: data[key] for key in data.files if key != "voxels"}
    return voxels, metadata


def downsample_voxels_any(
    voxels: np.ndarray, max_display_size: int
) -> tuple[np.ndarray, tuple[int, int, int]]:
    steps = tuple(max(1, int(np.ceil(size / max_display_size))) for size in voxels.shape)
    reduced = voxels[:: steps[0], :: steps[1], :: steps[2]] > 0
    return reduced, steps


def build_facecolors(occupancy: np.ndarray) -> np.ndarray:
    z_indices = np.indices(occupancy.shape)[2]
    z_norm = z_indices / max(1, occupancy.shape[2] - 1)
    facecolors = plt.cm.viridis(z_norm)
    facecolors[..., 3] = 0.85
    return facecolors


def plot_voxel_blocks(
    occupancy: np.ndarray,
    scale_steps: tuple[int, int, int],
    original_shape: tuple[int, int, int],
    output_path: Path | None,
    show: bool,
    title: str,
) -> None:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    facecolors = build_facecolors(occupancy)
    ax.voxels(occupancy, facecolors=facecolors, edgecolor="#111827", linewidth=0.08)

    displayed_shape = occupancy.shape
    x_ticks = np.linspace(0, displayed_shape[0], 6)
    y_ticks = np.linspace(0, displayed_shape[1], 6)
    z_ticks = np.linspace(0, displayed_shape[2], 6)
    x_labels = [str(int(round(value * scale_steps[0]))) for value in x_ticks]
    y_labels = [str(int(round(value * scale_steps[1]))) for value in y_ticks]
    z_labels = [str(int(round(value * scale_steps[2]))) for value in z_ticks]

    ax.set_xlim(0, displayed_shape[0])
    ax.set_ylim(0, displayed_shape[1])
    ax.set_zlim(0, displayed_shape[2])
    ax.set_box_aspect(original_shape)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.set_zticklabels(z_labels)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=38)
    ax.set_title(title)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220)
        print(f"saved figure: {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize voxel data from an NPZ file as voxel blocks.")
    parser.add_argument(
        "npz_path",
        type=Path,
        nargs="?",
        default=Path("datasets/voxel/voxel_annular_cylinder_200x200x200.npz"),
        help="Path to the NPZ file.",
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
        default=Path("docs/assets/voxel_annular_cylinder_200x200x200.png"),
        help="Optional image output path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the matplotlib window.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    voxels, metadata = load_voxels(args.npz_path)
    xy_size = int(metadata.get("xy_size", np.array(voxels.shape[0])).item())
    z_size = int(metadata.get("z_size", np.array(voxels.shape[2])).item())
    reduced_voxels, scale_steps = downsample_voxels_any(
        voxels=voxels,
        max_display_size=args.max_display_size,
    )
    outer_radius = float(metadata.get("outer_radius", np.array(np.nan)).item())
    inner_radius = float(metadata.get("inner_radius", np.array(np.nan)).item())
    shape_name = str(metadata.get("shape_name", np.array("unknown")).item())
    title = (
        f"Voxel {shape_name} | grid={xy_size}x{xy_size}x{z_size} | "
        f"display={reduced_voxels.shape[0]}x{reduced_voxels.shape[1]}x{reduced_voxels.shape[2]} | "
        f"outer={outer_radius:.1f}, inner={inner_radius:.1f}"
    )
    print(f"loaded: {args.npz_path}")
    print(f"shape: {voxels.shape}")
    print(f"active voxels: {int(voxels.sum())}")
    print(f"display grid: {reduced_voxels.shape}")
    print(f"display step: {scale_steps}")
    plot_voxel_blocks(
        occupancy=reduced_voxels,
        scale_steps=scale_steps,
        original_shape=voxels.shape,
        output_path=args.output,
        show=args.show,
        title=title,
    )
