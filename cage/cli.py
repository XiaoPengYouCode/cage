from __future__ import annotations

import argparse
from pathlib import Path

from cage.defaults import DEFAULT_OUTPUT_PATH, DEFAULT_ROW_SEEDS
from cage.models import PipelineConfig, RenderConfig
from cage.pipeline import VoronoiPipeline
from cage.rendering import plot_grid
from cage.rods import CylinderRodStyle, HelixRodStyle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a multi-row cube-bounded 3D Voronoi diagram with uniformly sampled seed points."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the generated image file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the matplotlib window after saving the image.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of seed points to sample uniformly inside the cube.",
    )
    parser.add_argument(
        "--row-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_ROW_SEEDS,
        help="Random seeds used for the row grid. Default renders three rows.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    if args.num_seeds < 2:
        raise ValueError("--num-seeds must be at least 2.")

    render = RenderConfig(output_path=Path(args.output), show=args.show)
    return PipelineConfig(
        num_seeds=args.num_seeds,
        row_seeds=tuple(args.row_seeds),
        render=render,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    pipeline = VoronoiPipeline()
    rows = pipeline.build_rows(config)
    plot_grid(
        rows,
        render_config=config.render,
        rod_styles=[CylinderRodStyle(), HelixRodStyle()],
    )
    print(
        f"Saved 3D Voronoi diagram to {config.render.output_path.resolve()} "
        f"with {config.num_seeds} uniformly sampled seeds per row "
        f"(row rng seeds={list(config.row_seeds)})"
    )
