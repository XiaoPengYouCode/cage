from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

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
    parser.add_argument("--gamma", type=float, default=1.8)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("datasets/topopt/seed_probability_mapping_2000.npz"),
    )
    return parser


COMMAND_PARSERS = {
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
