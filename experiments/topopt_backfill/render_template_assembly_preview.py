from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from helix_voronoi.rods import HelixRodStyle


def load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def unpack_template_edges(data: dict[str, np.ndarray]) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    offsets = data["template_edge_offsets"]
    starts = data["template_edge_starts"]
    ends = data["template_edge_ends"]
    templates: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for i in range(len(offsets) - 1):
        a = int(offsets[i])
        b = int(offsets[i + 1])
        templates.append([(starts[j], ends[j]) for j in range(a, b)])
    return templates


def sample_active_blocks(
    template_ids: np.ndarray,
    sample_count: int,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.argwhere(template_ids >= 0)
    if len(active) == 0:
        raise ValueError("No active blocks found.")
    if len(active) > sample_count:
        rng = np.random.default_rng(rng_seed)
        selection = rng.choice(len(active), size=sample_count, replace=False)
        active = active[selection]
    tids = template_ids[active[:, 0], active[:, 1], active[:, 2]]
    return active.astype(np.int32), tids.astype(np.int32)


def render_assembly(
    output_png: Path,
    template_edges: list[list[tuple[np.ndarray, np.ndarray]]],
    sampled_blocks: np.ndarray,
    sampled_template_ids: np.ndarray,
    grid_shape: tuple[int, int, int],
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    helix_style = HelixRodStyle(
        cycles_per_segment=1.9,
        amplitude_ratio=0.04,
        wire_radius_ratio=0.78,
        tube_sides=7,
        min_steps=12,
        steps_per_cycle=8,
    )

    colors = plt.cm.turbo(np.linspace(0.05, 0.95, 10))
    total_blocks = len(sampled_blocks)
    z_ratio = grid_shape[2] / grid_shape[0]

    for block_index, (block_xyz, template_id) in enumerate(zip(sampled_blocks, sampled_template_ids), start=1):
        ox, oy, oz = block_xyz.astype(float)
        block_edges = template_edges[int(template_id)]
        for start, end in block_edges:
            start_scaled = np.array(
                [ox + start[0], oy + start[1], (oz + start[2]) * z_ratio],
                dtype=float,
            )
            end_scaled = np.array(
                [ox + end[0], oy + end[1], (oz + end[2]) * z_ratio],
                dtype=float,
            )
            helix_style.draw_segment(
                ax,
                start_scaled,
                end_scaled,
                radius=0.020,
                color=colors[int(template_id)],
            )
        if block_index == 1 or block_index == total_blocks or block_index % 100 == 0:
            print(f"assembled preview block {block_index}/{total_blocks}")

    ax.set_xlim(0, grid_shape[0])
    ax.set_ylim(0, grid_shape[1])
    ax.set_zlim(0, grid_shape[2] * z_ratio)
    ax.set_box_aspect((grid_shape[0], grid_shape[1], grid_shape[2] * z_ratio))
    ax.set_xlabel("block x")
    ax.set_ylabel("block y")
    ax.set_zlabel("block z")
    ax.view_init(elev=24, azim=38)
    ax.set_title(f"Template-assembled helix Voronoi preview | sampled blocks={len(sampled_blocks):,}")

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)
    print(f"saved assembly preview: {output_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a sampled assembly preview from template-backfilled helix Voronoi blocks.")
    parser.add_argument(
        "--template-npz",
        type=Path,
        default=Path("datasets/topopt/template_backfilled_helix_voronoi.npz"),
    )
    parser.add_argument(
        "--sample-blocks",
        type=int,
        default=1200,
        help="Number of active blocks sampled for the approximate assembly preview.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("docs/assets/template_assembly_preview.png"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_npz(args.template_npz)
    template_ids = data["template_ids"]
    template_edges = unpack_template_edges(data)
    sampled_blocks, sampled_template_ids = sample_active_blocks(
        template_ids=template_ids,
        sample_count=args.sample_blocks,
        rng_seed=args.rng_seed,
    )
    render_assembly(
        output_png=args.output_png,
        template_edges=template_edges,
        sampled_blocks=sampled_blocks,
        sampled_template_ids=sampled_template_ids,
        grid_shape=template_ids.shape,
        dpi=args.dpi,
    )
