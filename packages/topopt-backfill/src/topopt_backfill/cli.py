from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from topopt_backfill.workflows import build_template_backfill, map_density_to_seed_mapping


def build_parser(prog: str, description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=prog, description=description)


def build_sample_seeds_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        "topopt-backfill sample-seeds",
        "Map topology-optimization density NPZ to a probabilistic Voronoi seed cloud.",
    )
    parser.add_argument(
        "input_npz",
        type=Path,
        nargs="?",
        default=Path("datasets/topopt/fake_density_annular_cylinder_full.npz"),
    )
    parser.add_argument("--num-seeds", type=int, default=100_000)
    parser.add_argument("--gamma", type=float, default=1.8)
    parser.add_argument("--chunk-depth", type=int, default=8)
    parser.add_argument("--max-display-size", type=int, default=84)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("datasets/topopt/seed_probability_mapping_100k.npz"),
    )
    return parser


def build_backfill_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        "topopt-backfill backfill-templates",
        "Build Voronoi template backfill data from topology density NPZ plus seed mapping NPZ.",
    )
    parser.add_argument(
        "density_npz",
        type=Path,
        nargs="?",
        default=Path("datasets/topopt/fake_density_annular_cylinder_full.npz"),
    )
    parser.add_argument(
        "seed_mapping_npz",
        type=Path,
        nargs="?",
        default=Path("datasets/topopt/seed_probability_mapping_100k.npz"),
    )
    parser.add_argument("--representative-seeds", type=int, default=81)
    parser.add_argument("--candidate-limit", type=int, default=15_000)
    parser.add_argument(
        "--candidate-voxel-bins",
        type=int,
        nargs=3,
        default=(48, 48, 24),
        metavar=("BX", "BY", "BZ"),
    )
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=(999, 999, 399),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--template-seed-base", type=int, default=1000)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("datasets/topopt/template_backfilled_helix_voronoi.npz"),
    )
    return parser


COMMAND_PARSERS = {
    "sample-seeds": build_sample_seeds_parser,
    "backfill-templates": build_backfill_parser,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args or args[0] not in COMMAND_PARSERS:
        parser = build_parser(
            "topopt-backfill",
            "Topology-density to Voronoi-template workflows.",
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
    if args.command == "sample-seeds":
        result = map_density_to_seed_mapping(
            args.input_npz,
            args.output_npz,
            num_seeds=args.num_seeds,
            gamma=args.gamma,
            chunk_depth=args.chunk_depth,
            max_display_size=args.max_display_size,
            rng_seed=args.rng_seed,
            progress=True,
        )
        print(
            f"Saved seed mapping to {result.output_npz.resolve()} "
            f"with {len(result.seed_points)} sampled seeds"
        )
        return

    result = build_template_backfill(
        args.density_npz,
        args.seed_mapping_npz,
        args.output_npz,
        representative_seeds=args.representative_seeds,
        candidate_limit=args.candidate_limit,
        candidate_voxel_bins=tuple(args.candidate_voxel_bins),
        target_shape=tuple(args.target_shape),
        template_seed_base=args.template_seed_base,
        rng_seed=args.rng_seed,
        progress=True,
    )
    print(
        f"Saved template backfill to {result.output_npz.resolve()} "
        f"with {len(result.template_seed_counts)} templates and "
        f"{len(result.representative_seeds)} representative seeds"
    )
