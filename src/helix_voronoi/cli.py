from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

from helix_voronoi.defaults import DEFAULT_OUTPUT_PATH, DEFAULT_ROW_SEEDS
from helix_voronoi.models import PipelineConfig, RenderConfig, RowGeometry
from helix_voronoi.pipeline import VoronoiPipeline
from helix_voronoi.rendering import plot_grid
from helix_voronoi.rods import CylinderRodStyle, HelixRodStyle

CommandHandler = Callable[[argparse.Namespace], None]


def build_parser(prog: str, description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=prog, description=description)


def add_num_seeds_argument(parser: argparse.ArgumentParser, default: int) -> None:
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=default,
        help="Number of Voronoi seeds in the unit cell.",
    )


def validate_num_seeds(num_seeds: int) -> None:
    if num_seeds < 2:
        raise ValueError("--num-seeds must be at least 2.")


def build_render_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        prog="helix-voronoi",
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
    add_num_seeds_argument(parser, default=10)
    parser.add_argument(
        "--row-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_ROW_SEEDS,
        help="Random seeds used for the row grid. Default renders three rows.",
    )
    return parser


def build_export_mixed_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        prog="helix-voronoi export-mixed",
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
    add_num_seeds_argument(parser, default=64)
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
    parser = build_parser(
        prog="helix-voronoi export-helix",
        description="Build the helix Voronoi cell and export it to STL.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=55,
        help="Random seed used to generate the unit cell.",
    )
    add_num_seeds_argument(parser, default=10)
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


COMMAND_PARSERS: dict[str, Callable[[], argparse.ArgumentParser]] = {
    "export-mixed": build_export_mixed_parser,
    "export-helix": build_export_helix_parser,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = list(argv) if argv is not None else sys.argv[1:]
    command = args[0] if args and args[0] in COMMAND_PARSERS else "render"
    parser = build_render_parser() if command == "render" else COMMAND_PARSERS[command]()
    parsed = parser.parse_args(args if command == "render" else args[1:])
    parsed.command = command
    return parsed


def build_config(args: argparse.Namespace) -> PipelineConfig:
    validate_num_seeds(args.num_seeds)
    render = RenderConfig(output_path=Path(args.output), show=args.show)
    return PipelineConfig(
        num_seeds=args.num_seeds,
        row_seeds=tuple(args.row_seeds),
        render=render,
    )


def build_row(args: argparse.Namespace) -> RowGeometry:
    validate_num_seeds(args.num_seeds)
    return VoronoiPipeline().build_row(num_seeds=args.num_seeds, rng_seed=args.seed)


def handle_export_mixed(args: argparse.Namespace) -> None:
    from helix_voronoi.helix_stl import export_mixed_edges_to_stl

    row = build_row(args)
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


def handle_export_helix(args: argparse.Namespace) -> None:
    from helix_voronoi.helix_stl import export_helix_edges_to_stl

    row = build_row(args)
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


def handle_render(args: argparse.Namespace) -> None:
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


COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "export-mixed": handle_export_mixed,
    "export-helix": handle_export_helix,
    "render": handle_render,
}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    COMMAND_HANDLERS[args.command](args)
