from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from topopt_sampling.demo import (
    generate_annular_cylinder_npz,
    generate_fake_density_result,
    render_sampling_overview,
)
from topopt_sampling.workflows import map_density_to_seed_mapping


def build_parser(prog: str, description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=prog, description=description)


def build_sample_seeds_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        "topopt-sampling sample-seeds",
        "Map topology-optimization density (.npz or .mat) to probabilistic random seed points.",
    )
    parser.add_argument(
        "input_npz",
        type=Path,
        nargs="?",
        default=Path("datasets/topopt/fake_density_annular_cylinder_200x200x80.npz"),
        help="Topology density input (.npz or .mat).",
    )
    parser.add_argument("--num-seeds", type=int, default=2_000)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("datasets/topopt/seed_probability_mapping_2000.npz"),
    )
    return parser


def build_generate_voxels_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        "topopt-sampling generate-voxels",
        "Generate a voxel annular cylinder and save it as an NPZ file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/voxel/voxel_annular_cylinder_200x200x80.npz"),
        help="Output NPZ path.",
    )
    parser.add_argument(
        "--xy-size",
        type=int,
        default=200,
        help="Grid size in x and y.",
    )
    parser.add_argument(
        "--z-size",
        type=int,
        default=80,
        help="Grid size in z.",
    )
    parser.add_argument(
        "--outer-radius",
        type=float,
        default=100.0,
        help="Outer radius of the annular cylinder in voxel units.",
    )
    parser.add_argument(
        "--inner-radius",
        type=float,
        default=50.0,
        help="Inner radius of the annular cylinder in voxel units.",
    )
    parser.add_argument(
        "--chunk-depth",
        type=int,
        default=8,
        help="How many z slices to generate per chunk.",
    )
    return parser


def build_generate_fake_density_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        "topopt-sampling generate-fake-density",
        "Generate a fake FEA-like density result NPZ from voxel geometry.",
    )
    parser.add_argument(
        "source_npz",
        type=Path,
        nargs="?",
        default=Path("datasets/voxel/voxel_annular_cylinder_200x200x80.npz"),
        help="Source voxel geometry NPZ.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/topopt/fake_density_annular_cylinder_200x200x80.npz"),
        help="Output fake density result NPZ.",
    )
    parser.add_argument(
        "--chunk-depth",
        type=int,
        default=8,
        help="How many z slices to process per chunk.",
    )
    return parser


def build_render_overview_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        "topopt-sampling render-overview",
        "Render the topology-density to seed-sampling overview figure.",
    )
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
    return parser


COMMAND_PARSERS = {
    "generate-voxels": build_generate_voxels_parser,
    "generate-fake-density": build_generate_fake_density_parser,
    "render-overview": build_render_overview_parser,
    "sample-seeds": build_sample_seeds_parser,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args or args[0] not in COMMAND_PARSERS:
        parser = build_parser(
            "topopt-sampling",
            "Topology-density sampling workflow.",
        )
        parser.add_argument("command", choices=tuple(COMMAND_PARSERS))
        parsed = parser.parse_args(args[:1])
        command = parsed.command
        parser = COMMAND_PARSERS[command]()
        parsed = parser.parse_args(args[1:])
        parsed.command = command
        return parsed

    command = args[0]
    parser = COMMAND_PARSERS[command]()
    parsed = parser.parse_args(args[1:])
    parsed.command = command
    return parsed


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "generate-voxels":
        generate_annular_cylinder_npz(
            output_path=args.output,
            xy_size=args.xy_size,
            z_size=args.z_size,
            outer_radius=args.outer_radius,
            inner_radius=args.inner_radius,
            chunk_depth=args.chunk_depth,
        )
        return

    if args.command == "generate-fake-density":
        generate_fake_density_result(
            source_npz=args.source_npz,
            output_npz=args.output,
            chunk_depth=args.chunk_depth,
        )
        return

    if args.command == "render-overview":
        render_sampling_overview(
            density_npz=args.density_npz,
            seed_npz=args.seed_npz,
            output_png=args.output,
        )
        return

    result = map_density_to_seed_mapping(
        args.input_npz,
        args.output_npz,
        num_seeds=args.num_seeds,
        gamma=args.gamma,
        rng_seed=args.rng_seed,
        progress=True,
    )
    print(
        f"Saved seed mapping to {result.output_npz.resolve()} "
        f"with {len(result.seed_points)} sampled seeds"
    )
