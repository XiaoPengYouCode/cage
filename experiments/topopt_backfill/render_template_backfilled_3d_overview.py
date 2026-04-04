from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors


def load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def render_3d_overview(
    template_ids: np.ndarray,
    output_png: Path,
    stride_xy: int,
    stride_z: int,
    dpi: int,
) -> None:
    sampled = template_ids[::stride_xy, ::stride_xy, ::stride_z]
    mask = sampled >= 0
    coords = np.argwhere(mask)
    values = sampled[mask]

    if len(coords) == 0:
        raise ValueError("No active blocks found to render.")

    colors = plt.cm.turbo(np.linspace(0.05, 0.95, 10))
    point_colors = colors[values]
    point_colors[:, 3] = 0.9

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=point_colors,
        s=4,
        marker="s",
        linewidths=0,
        depthshade=False,
    )

    full_shape = template_ids.shape
    displayed_shape = sampled.shape
    ax.set_xlim(0, displayed_shape[0])
    ax.set_ylim(0, displayed_shape[1])
    ax.set_zlim(0, displayed_shape[2])
    ax.set_box_aspect(displayed_shape)

    xt = np.linspace(0, displayed_shape[0], 6)
    yt = np.linspace(0, displayed_shape[1], 6)
    zt = np.linspace(0, displayed_shape[2], 6)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_zticks(zt)
    ax.set_xticklabels([str(int(round(v * stride_xy))) for v in xt])
    ax.set_yticklabels([str(int(round(v * stride_xy))) for v in yt])
    ax.set_zticklabels([str(int(round(v * stride_z))) for v in zt])
    ax.set_xlabel("block x")
    ax.set_ylabel("block y")
    ax.set_zlabel("block z")
    ax.view_init(elev=24, azim=38)
    ax.set_title(
        f"3D overview of backfilled helix templates | grid={full_shape[0]}x{full_shape[1]}x{full_shape[2]} | displayed={displayed_shape[0]}x{displayed_shape[1]}x{displayed_shape[2]}"
    )

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, 10.5, 1), cmap.N)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.04, ticks=np.arange(10), label="template id")

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)
    print(f"saved 3d overview: {output_png}")
    print(f"active displayed blocks: {len(coords)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 3D overview of the full backfilled template cylinder.")
    parser.add_argument(
        "--template-npz",
        type=Path,
        default=Path("datasets/topopt/template_backfilled_helix_voronoi.npz"),
    )
    parser.add_argument(
        "--stride-xy",
        type=int,
        default=3,
        help="Sampling stride in x and y for fast 3D overview.",
    )
    parser.add_argument(
        "--stride-z",
        type=int,
        default=2,
        help="Sampling stride in z for fast 3D overview.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("docs/assets/template_backfilled_helix_voronoi_3d.png"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_npz(args.template_npz)
    render_3d_overview(
        template_ids=data["template_ids"],
        output_png=args.output_png,
        stride_xy=args.stride_xy,
        stride_z=args.stride_z,
        dpi=args.dpi,
    )
