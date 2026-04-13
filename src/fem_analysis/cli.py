from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

from fem_analysis.annular_cylinder import (
    AnnularCylinderConfig,
    MaterialConfig,
    TrussInfillConfig,
    run_annular_cylinder_demo,
)


def build_parser(prog: str, description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=prog, description=description)


def build_annular_cylinder_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        prog="fem-analysis annular-cylinder",
        description="Run a linear elastic SfePy demo for a loaded annular cylinder and save a visualization.",
    )
    parser.add_argument(
        "--outer-diameter-cm",
        type=float,
        default=3.0,
        help="Outer diameter of the annular cylinder in centimeters.",
    )
    parser.add_argument(
        "--inner-diameter-cm",
        type=float,
        default=2.0,
        help="Inner diameter of the annular cylinder in centimeters.",
    )
    parser.add_argument(
        "--height-cm",
        type=float,
        default=2.0,
        help="Cylinder height in centimeters. Defaults to 2 cm for this demo.",
    )
    parser.add_argument(
        "--load-n",
        type=float,
        default=1000.0,
        help="Total compressive force applied on the top face in Newtons.",
    )
    parser.add_argument(
        "--voxel-size-mm",
        type=float,
        default=0.4,
        help="Target voxel edge length in millimeters for the hexahedral mesh.",
    )
    parser.add_argument(
        "--youngs-modulus-gpa",
        type=float,
        default=110.0,
        help="Young's modulus of the shell material in GPa.",
    )
    parser.add_argument(
        "--poisson-ratio",
        type=float,
        default=0.34,
        help="Poisson ratio of the shell material.",
    )
    parser.add_argument(
        "--inner-fill",
        choices=("empty", "bone", "truss"),
        default="bone",
        help="How to model the inner cylindrical region.",
    )
    parser.add_argument(
        "--fill-youngs-modulus-gpa",
        type=float,
        default=1.0,
        help="Equivalent Young's modulus of the inner fill region in GPa.",
    )
    parser.add_argument(
        "--fill-poisson-ratio",
        type=float,
        default=0.30,
        help="Equivalent Poisson ratio of the inner fill region.",
    )
    parser.add_argument(
        "--truss-cell-mm",
        type=float,
        default=0.4,
        help="Approximate truss cell size in millimeters when --inner-fill truss is used.",
    )
    parser.add_argument(
        "--truss-rod-mm",
        type=float,
        default=0.1,
        help="Truss rod radius in millimeters when --inner-fill truss is used.",
    )
    parser.add_argument(
        "--output-image",
        default="docs/assets/annular_cylinder_fea.png",
        help="Path to write the visualization image.",
    )
    parser.add_argument(
        "--output-json",
        default="docs/analysis/annular_cylinder_fea.json",
        help="Path to write the analysis summary JSON.",
    )
    parser.add_argument(
        "--output-npz",
        default="datasets/topopt/annular_cylinder_fea_density.npz",
        help="Path to write the standardized 3D density NPZ for downstream packages.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = list(argv) if argv is not None else sys.argv[1:]
    command = args[0] if args and args[0] == "annular-cylinder" else "annular-cylinder"
    parser = build_annular_cylinder_parser()
    parsed = parser.parse_args(args[1:] if args and args[0] == command else args)
    parsed.command = command
    return parsed


def build_annular_cylinder_config(args: argparse.Namespace) -> AnnularCylinderConfig:
    material = MaterialConfig(
        name="TC4",
        youngs_modulus_gpa=args.youngs_modulus_gpa,
        poisson_ratio=args.poisson_ratio,
    )
    fill_material = MaterialConfig(
        name="Bone graft equivalent",
        youngs_modulus_gpa=args.fill_youngs_modulus_gpa,
        poisson_ratio=args.fill_poisson_ratio,
    )
    truss_infill = TrussInfillConfig(
        enabled=args.inner_fill == "truss",
        cell_size_m=args.truss_cell_mm / 1e3,
        rod_radius_m=args.truss_rod_mm / 1e3,
    )
    return AnnularCylinderConfig(
        outer_diameter_m=args.outer_diameter_cm / 100.0,
        inner_diameter_m=args.inner_diameter_cm / 100.0,
        height_m=args.height_cm / 100.0,
        total_force_n=args.load_n,
        voxel_size_m=args.voxel_size_mm / 1e3,
        material=material,
        inner_fill_mode=args.inner_fill,
        fill_material=fill_material,
        truss_infill=truss_infill,
        output_image=Path(args.output_image),
        output_json=Path(args.output_json),
        output_npz=Path(args.output_npz),
    )


def handle_annular_cylinder(args: argparse.Namespace) -> None:
    import traceback

    started_at = time.perf_counter()

    def progress(message: str) -> None:
        elapsed_s = time.perf_counter() - started_at
        print(f"[fem-analysis {elapsed_s:7.1f}s] {message}", flush=True)

    try:
        summary = run_annular_cylinder_demo(
            build_annular_cylinder_config(args),
            progress=progress,
        )
    except Exception:
        elapsed_s = time.perf_counter() - started_at
        print(f"[fem-analysis {elapsed_s:7.1f}s] ERROR: analysis failed", flush=True)
        traceback.print_exc()
        sys.exit(1)
    print(
        f"Saved annular-cylinder FEA to {summary.image_path.resolve()}, {summary.json_path.resolve()}, "
        f"and {summary.npz_path.resolve()} "
        f"(infill={summary.result.inner_fill_mode}, "
        f"max displacement={summary.result.max_displacement_mm:.4f} mm, "
        f"max von Mises={summary.result.max_von_mises_mpa:.2f} MPa)"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    handle_annular_cylinder(args)
