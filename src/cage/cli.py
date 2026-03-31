from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from cage.analysis.config import ModulusAnalysisConfig
from cage.defaults import (
    DEFAULT_MODULUS_JSON_PATH,
    DEFAULT_MODULUS_MARKDOWN_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_ROW_SEEDS,
)
from cage.models import PipelineConfig, RenderConfig
from cage.pipeline import VoronoiPipeline
from cage.rendering import plot_grid
from cage.rods import CylinderRodStyle, HelixRodStyle


def build_render_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cage",
        description="Generate a multi-row cube-bounded 3D Voronoi diagram with uniformly sampled seed points.",
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
    return parser


def build_modulus_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cage modulus",
        description="Solve the approximate effective modulus for a Voronoi unit cell with SfePy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=55,
        help="Random seed used to generate the unit cell.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of Voronoi seeds in the unit cell.",
    )
    parser.add_argument(
        "--style",
        choices=("cylinder", "helix", "both"),
        default="both",
        help="Which rod styles to analyze.",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=(96, 128, 160),
        help="Voxel resolutions used for the convergence sweep.",
    )
    parser.add_argument(
        "--output-markdown",
        default=str(DEFAULT_MODULUS_MARKDOWN_PATH),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_MODULUS_JSON_PATH),
        help="JSON report output path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print the intended run without solving.",
    )
    return parser


def build_export_mixed_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cage export-mixed",
        description=(
            "Build a mixed-style Voronoi cage (straight rods on cube/face edges, "
            "helix tubes on interior edges) with manifold3d boolean union and export to STL."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=55,
        help="Random seed used to generate the unit cell.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=64,
        help="Number of Voronoi seeds in the unit cell.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.012,
        help="Rod radius.",
    )
    parser.add_argument(
        "--helix-cycles",
        type=float,
        default=1.0,
        help="Number of helix cycles per interior edge segment.",
    )
    parser.add_argument(
        "--helix-amplitude",
        type=float,
        default=0.06,
        help="Helix amplitude as a fraction of segment length.",
    )
    parser.add_argument(
        "--tube-sides",
        type=int,
        default=24,
        help="Number of sides for tube/cylinder cross-sections.",
    )
    parser.add_argument(
        "--stl-output",
        default="docs/assets/voronoi_mixed.stl",
        help="Path to write the generated STL file.",
    )
    return parser


def build_export_helix_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cage export-helix",
        description="Build the helix Voronoi cell and export it to STL.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=55,
        help="Random seed used to generate the unit cell.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of Voronoi seeds in the unit cell.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.012,
        help="Base rod radius before helix wire scaling.",
    )
    parser.add_argument(
        "--stl-output",
        default="docs/assets/voronoi_helix_seed55.stl",
        help="Path to write the generated STL file.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = list(argv) if argv is not None else sys.argv[1:]
    if args and args[0] == "modulus":
        parsed = build_modulus_parser().parse_args(args[1:])
        parsed.command = "modulus"
        return parsed
    if args and args[0] == "export-mixed":
        parsed = build_export_mixed_parser().parse_args(args[1:])
        parsed.command = "export-mixed"
        return parsed
    if args and args[0] == "export-helix":
        parsed = build_export_helix_parser().parse_args(args[1:])
        parsed.command = "export-helix"
        return parsed
    parsed = build_render_parser().parse_args(args)
    parsed.command = "render"
    return parsed


def build_config(args: argparse.Namespace) -> PipelineConfig:
    if args.num_seeds < 2:
        raise ValueError("--num-seeds must be at least 2.")

    render = RenderConfig(output_path=Path(args.output), show=args.show)
    return PipelineConfig(
        num_seeds=args.num_seeds,
        row_seeds=tuple(args.row_seeds),
        render=render,
    )


def build_modulus_config(args: argparse.Namespace) -> ModulusAnalysisConfig:
    if args.num_seeds < 2:
        raise ValueError("--num-seeds must be at least 2.")
    return ModulusAnalysisConfig(
        seed=args.seed,
        num_seeds=args.num_seeds,
        style=args.style,
        resolutions=tuple(args.resolutions),
        output_markdown=Path(args.output_markdown),
        output_json=Path(args.output_json),
        dry_run=args.dry_run,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.command == "modulus":
        config = build_modulus_config(args)
        if config.dry_run:
            print(
                f"Dry run: {config.backend} modulus analysis for seed={config.seed}, styles={list(config.selected_styles())}, "
                f"resolutions={list(config.resolutions)}, markdown={config.output_markdown}, json={config.output_json}"
            )
            return

        from cage.analysis import run_modulus_analysis

        summary = run_modulus_analysis(config)
        print(
            f"Saved modulus reports to {summary.markdown_path.resolve()} and {summary.json_path.resolve()} "
            f"for seed={config.seed} and styles={list(config.selected_styles())}"
        )
        return

    if args.command == "export-mixed":
        if args.num_seeds < 2:
            raise ValueError("--num-seeds must be at least 2.")
        from cage.helix_stl import export_mixed_edges_to_stl

        row = VoronoiPipeline().build_row(num_seeds=args.num_seeds, rng_seed=args.seed)
        summary = export_mixed_edges_to_stl(
            row.edges,
            radius=args.radius,
            output_path=Path(args.stl_output),
            helix_cycles=args.helix_cycles,
            helix_amplitude=args.helix_amplitude,
            tube_sides=args.tube_sides,
        )
        print(
            f"Exported mixed STL to {summary.output_path.resolve()} "
            f"for seed={args.seed}, num_seeds={args.num_seeds} "
            f"— {summary.straight_edge_count} straight edges, "
            f"{summary.interior_edge_count} helix edges, "
            f"{summary.node_sphere_count} junction spheres, "
            f"{summary.triangle_count} triangles total"
        )
        return

    if args.command == "export-helix":
        if args.num_seeds < 2:
            raise ValueError("--num-seeds must be at least 2.")
        from cage.helix_stl import export_helix_edges_to_stl

        row = VoronoiPipeline().build_row(num_seeds=args.num_seeds, rng_seed=args.seed)
        summary = export_helix_edges_to_stl(
            row.edges,
            radius=args.radius,
            output_path=Path(args.stl_output),
        )
        print(
            f"Exported helix STL to {summary.output_path.resolve()} "
            f"for seed={args.seed} "
            f"with {summary.edge_count} segments and {summary.triangle_count} triangles"
        )
        return

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
