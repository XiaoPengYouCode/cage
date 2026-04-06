from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from topopt_sampling.benchmark import benchmark_fake_topopt_to_glb, write_benchmark_report
from topopt_sampling.demo import (
    generate_annular_cylinder_npz,
    generate_fake_density_result,
    render_sampling_overview,
)
from topopt_sampling.exact_brep import (
    build_diagram_brep,
    summarize_diagram_brep,
    write_diagram_brep_json,
)
from topopt_sampling.hybrid_exact_brep import (
    build_hybrid_exact_diagram_brep,
    summarize_hybrid_exact_brep,
    write_hybrid_exact_brep_json,
)
from topopt_sampling.exact_restricted_voronoi_3d import (
    build_annular_cylinder_domain,
    build_exact_restricted_voronoi_diagram,
    summarize_exact_diagram,
)
from topopt_sampling.threejs_glb_export import write_threejs_shell_glb
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


def build_exact_summary_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        "topopt-sampling exact-summary",
        "Build the exact restricted Voronoi diagram summary for an annular-cylinder seed set.",
    )
    parser.add_argument(
        "seed_npz",
        type=Path,
        nargs="?",
        default=Path("datasets/topopt/seed_probability_mapping_2000.npz"),
        help="Seed mapping NPZ containing seed_points.",
    )
    parser.add_argument("--xy-size", type=int, default=200)
    parser.add_argument("--z-size", type=int, default=80)
    parser.add_argument("--outer-radius", type=float, default=100.0)
    parser.add_argument("--inner-radius", type=float, default=50.0)
    return parser


def build_threejs_shell_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        "topopt-sampling export-threejs-shell-glb",
        "Export shell-cell GLB data for the Three.js viewer.",
    )
    parser.add_argument(
        "seed_npz",
        type=Path,
        nargs="?",
        default=Path("datasets/topopt/seed_probability_mapping_2000.npz"),
        help="Seed mapping NPZ containing seed_points.",
    )
    parser.add_argument("--xy-size", type=int, default=200)
    parser.add_argument("--z-size", type=int, default=80)
    parser.add_argument("--outer-radius", type=float, default=100.0)
    parser.add_argument("--inner-radius", type=float, default=50.0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("viewer/public/data/hybrid_exact_shell_2000.glb"),
    )
    return parser


def build_benchmark_end_to_end_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        "topopt-sampling benchmark-end-to-end",
        "Benchmark the full fake-topopt -> seeds -> exact Voronoi -> shell GLB pipeline.",
    )
    parser.add_argument(
        "--voxel-output",
        type=Path,
        default=Path("datasets/voxel/voxel_annular_cylinder_200x200x80.npz"),
    )
    parser.add_argument(
        "--density-output",
        type=Path,
        default=Path("datasets/topopt/fake_density_annular_cylinder_200x200x80.npz"),
    )
    parser.add_argument(
        "--seed-output",
        type=Path,
        default=Path("datasets/topopt/seed_probability_mapping_2000.npz"),
    )
    parser.add_argument(
        "--glb-output",
        type=Path,
        default=Path("viewer/public/data/hybrid_exact_shell_2000.glb"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("docs/analysis/topopt_end_to_end_benchmark_2000.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("docs/analysis/topopt_end_to_end_benchmark_2000.md"),
    )
    parser.add_argument("--xy-size", type=int, default=200)
    parser.add_argument("--z-size", type=int, default=80)
    parser.add_argument("--outer-radius", type=float, default=100.0)
    parser.add_argument("--inner-radius", type=float, default=50.0)
    parser.add_argument("--num-seeds", type=int, default=2_000)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--chunk-depth", type=int, default=8)
    return parser


def build_exact_brep_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        "topopt-sampling build-exact-brep",
        "Build a hybrid exact restricted Voronoi B-rep kernel using box polyhedral cells -> annular trim -> hybrid B-rep.",
    )
    parser.add_argument(
        "seed_npz",
        type=Path,
        nargs="?",
        default=Path("datasets/topopt/seed_probability_mapping_2000.npz"),
        help="Seed mapping NPZ containing seed_points.",
    )
    parser.add_argument("--xy-size", type=int, default=200)
    parser.add_argument("--z-size", type=int, default=80)
    parser.add_argument("--outer-radius", type=float, default=100.0)
    parser.add_argument("--inner-radius", type=float, default=50.0)
    parser.add_argument(
        "--seed-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of seed ids to build.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/analysis/restricted_voronoi_brep.json"),
    )
    return parser


COMMAND_PARSERS = {
    "benchmark-end-to-end": build_benchmark_end_to_end_parser,
    "build-exact-brep": build_exact_brep_parser,
    "exact-summary": build_exact_summary_parser,
    "export-threejs-shell-glb": build_threejs_shell_parser,
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

    if args.command == "exact-summary":
        with np.load(args.seed_npz) as data:
            seed_points = data["seed_points"].astype(float)
        domain = build_annular_cylinder_domain(
            xy_size=args.xy_size,
            z_size=args.z_size,
            outer_radius=args.outer_radius,
            inner_radius=args.inner_radius,
        )
        diagram = build_exact_restricted_voronoi_diagram(seed_points=seed_points, domain=domain, include_support_traces=True)
        summary = summarize_exact_diagram(diagram)
        print(
            f"Exact restricted diagram: seeds={summary.num_seeds}, "
            f"domain_volume={summary.domain_volume:.6f}, support_curves={summary.support_curve_count}"
        )
        return

    if args.command == "benchmark-end-to-end":
        result = benchmark_fake_topopt_to_glb(
            voxel_output=args.voxel_output,
            density_output=args.density_output,
            seed_output=args.seed_output,
            glb_output=args.glb_output,
            xy_size=args.xy_size,
            z_size=args.z_size,
            outer_radius=args.outer_radius,
            inner_radius=args.inner_radius,
            num_seeds=args.num_seeds,
            gamma=args.gamma,
            rng_seed=args.rng_seed,
            chunk_depth=args.chunk_depth,
        )
        write_benchmark_report(result, json_path=args.report_json, markdown_path=args.report_markdown)
        print(
            f"Benchmarked fake-topopt -> GLB: total={result.total_wall_seconds:.2f}s, "
            f"glb_bytes={result.output_bytes}, output={args.glb_output.resolve()}"
        )
        for stage in result.stages:
            print(f"  - {stage.name}: {stage.wall_seconds:.2f}s")
        print(f"  - user: {result.total_user_seconds:.2f}s")
        print(f"  - sys: {result.total_sys_seconds:.2f}s")
        if args.report_json is not None:
            print(f"  - report json: {args.report_json.resolve()}")
        if args.report_markdown is not None:
            print(f"  - report markdown: {args.report_markdown.resolve()}")
        return

    if args.command == "build-exact-brep":
        with np.load(args.seed_npz) as data:
            seed_points = data["seed_points"].astype(float)
        domain = build_annular_cylinder_domain(
            xy_size=args.xy_size,
            z_size=args.z_size,
            outer_radius=args.outer_radius,
            inner_radius=args.inner_radius,
        )
        diagram_brep = build_hybrid_exact_diagram_brep(seed_points=seed_points, domain=domain, seed_ids=args.seed_ids)
        write_hybrid_exact_brep_json(diagram_brep, args.output_json)
        summary = summarize_hybrid_exact_brep(diagram_brep)
        print(
            f"Built hybrid exact B-rep JSON: cells={summary.num_cells}, faces={summary.num_faces}, "
            f"edges={summary.num_edges}, vertices={summary.num_vertices}, output={args.output_json.resolve()}"
        )
        return

    if args.command == "export-threejs-shell-glb":
        with np.load(args.seed_npz) as data:
            seed_points = data["seed_points"].astype(float)
        domain = build_annular_cylinder_domain(
            xy_size=args.xy_size,
            z_size=args.z_size,
            outer_radius=args.outer_radius,
            inner_radius=args.inner_radius,
        )
        summary = write_threejs_shell_glb(seed_points=seed_points, domain=domain, output_path=args.output_json)
        print(
            f"Exported Three.js labeled GLB: cells={summary.num_cells}, exported_cells={summary.num_exported_cells}, "
            f"shell_cells={summary.num_shell_cells}, faces={summary.num_faces}, triangles={summary.num_triangles}, "
            f"boundaries={summary.num_boundaries}, bytes={summary.output_bytes}, output={args.output_json.resolve()}"
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
