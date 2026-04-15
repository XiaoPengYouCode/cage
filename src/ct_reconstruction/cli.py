from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Sequence

from ct_reconstruction.voxelizer import load_stl, stl_bounds, voxelize_stl
from ct_reconstruction.npz_writer import build_voxel_npz_payload, write_npz
from ct_reconstruction.visualize import render_comparison
from ct_reconstruction.glb_export import voxels_to_glb


def _progress(started_at: float, message: str) -> None:
    print(f"[ct-reconstruction {time.perf_counter() - started_at:7.1f}s] {message}",
          flush=True)


def _run_voxelize(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    stl_path = Path(args.stl)
    output_npz = Path(args.output_npz)
    output_png = Path(args.output_png)

    try:
        _progress(t0, f"Loading STL from {stl_path} ...")
        m = load_stl(stl_path)
        lo, hi = stl_bounds(m)
        size = hi - lo
        _progress(t0, f"STL loaded: {len(m.vectors):,} triangles, "
                      f"size {size[0]:.1f}×{size[1]:.1f}×{size[2]:.1f} mm")

        _progress(t0, f"Voxelizing at {args.voxel_size:.2f} mm ...")
        occupancy, origin, spacing = voxelize_stl(m, voxel_size_mm=args.voxel_size)
        nx, ny, nz = occupancy.shape
        bone_pct = 100.0 * occupancy.sum() / occupancy.size
        _progress(t0, f"Grid {nx}×{ny}×{nz}, spacing {spacing[0]:.3f}×{spacing[1]:.3f}×{spacing[2]:.3f} mm, "
                      f"fill {bone_pct:.1f}%")

        _progress(t0, f"Writing NPZ to {output_npz} ...")
        payload = build_voxel_npz_payload(occupancy, origin, spacing)
        write_npz(payload, output_npz)

        _progress(t0, f"Rendering STL vs voxel comparison to {output_png} ...")
        render_comparison(m, occupancy, origin, spacing, output_png)

        output_glb = Path(args.output_glb)
        _progress(t0, f"Exporting voxel surface GLB to {output_glb} ...")
        voxels_to_glb(occupancy, origin, spacing, output_glb)

        _progress(t0, f"Done — NPZ: {output_npz.resolve()}")
        _progress(t0, f"Done — PNG: {output_png.resolve()}")
        _progress(t0, f"Done — GLB: {output_glb.resolve()}")

    except Exception:
        _progress(t0, "ERROR: voxelization failed")
        traceback.print_exc()
        sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ct-reconstruction",
        description="Voxelize a surface mesh STL and compare with the original.",
    )
    sub = parser.add_subparsers(dest="command")

    vox = sub.add_parser("voxelize", help="Voxelize an STL file.")
    vox.add_argument(
        "--stl",
        default="datasets/stl/LumbarVertebrae.stl",
        help="Input STL file path.",
    )
    vox.add_argument(
        "--voxel-size",
        type=float,
        default=0.4,
        help="Voxel edge length in mm (default 0.4).",
    )
    vox.add_argument(
        "--output-npz",
        default="datasets/topopt/lumbar_vertebra_voxels.npz",
        help="Output NPZ path.",
    )
    vox.add_argument(
        "--output-png",
        default="docs/assets/lumbar_vertebra_comparison.png",
        help="Output comparison PNG path.",
    )
    vox.add_argument(
        "--output-glb",
        default="viewer/public/data/lumbar_vertebra.glb",
        help="Output GLB path (loaded directly by the viewer).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "voxelize":
        _run_voxelize(args)
    else:
        parser.print_help()
        sys.exit(1)
