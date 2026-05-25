from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VARIABLE_RADIUS_EDGES_NPZ = (
    ROOT / "outputs" / "fjw_optimize_real_iter017" / "fjw_iter017_voronoi_edges_variable_radius.npz"
)
DEFAULT_ALIGNED_DENSITY_NPZ = ROOT / "outputs" / "fjw_optimize_real_iter017" / "fjw_iter017_aligned_density_gamma1.npz"
DEFAULT_OUTPUT_NPZ = ROOT / "outputs" / "fjw_optimize_real_iter017" / "fjw_iter017_skeleton_voxels_variable_radius.npz"
DEFAULT_OUTPUT_GLB = ROOT / "outputs" / "fjw_optimize_real_iter017" / "fjw_iter017_skeleton_variable_radius.glb"
DEFAULT_OUTPUT_STL = ROOT / "outputs" / "fjw_optimize_real_iter017" / "fjw_iter017_skeleton_variable_radius.stl"


def build_variable_radius_skeleton(
    *,
    variable_radius_edges_npz: Path,
    aligned_density_npz: Path,
    output_npz: Path,
    subdivision: int,
    radius_field_key: str,
    output_glb: Path | None,
    output_stl: Path | None,
    mc_smooth_sigma: float,
) -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "src"))
    from matlab2stl_pipeline.skeleton_voxelizer import mesh_from_voxels, voxelize_variable_radius_skeleton

    voxels = voxelize_variable_radius_skeleton(
        edges_npz_path=variable_radius_edges_npz,
        aligned_npz_path=aligned_density_npz,
        output_npz_path=output_npz,
        subdivision=subdivision,
        radius_field_key=radius_field_key,
    )
    if output_glb is not None and output_stl is not None:
        mesh_from_voxels(
            skeleton_npz_path=output_npz,
            output_glb_path=output_glb,
            output_stl_path=output_stl,
            smooth_sigma=mc_smooth_sigma,
            aligned_npz_path=aligned_density_npz,
        )
    payload = np.load(output_npz, allow_pickle=True)
    summary = {
        "output_npz": str(output_npz.resolve()),
        "voxel_count": int(np.asarray(voxels, dtype=np.uint8).sum()),
        "grid_shape_xyz": payload["grid_shape_xyz"].tolist(),
        "subdivision": int(payload["subdivision"]),
        "edge_count": int(payload["edge_count"]),
        "radius_field_key": str(payload["radius_field_key"].item()),
    }
    if output_glb is not None and output_stl is not None:
        summary["output_glb"] = str(output_glb.resolve())
        summary["output_stl"] = str(output_stl.resolve())
        summary["mc_smooth_sigma"] = float(mc_smooth_sigma)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Voxelize the iter_017 Voronoi skeleton with a per-edge calibrated radius field."
    )
    parser.add_argument("--variable-radius-edges-npz", type=Path, default=DEFAULT_VARIABLE_RADIUS_EDGES_NPZ)
    parser.add_argument("--aligned-density-npz", type=Path, default=DEFAULT_ALIGNED_DENSITY_NPZ)
    parser.add_argument("--output-npz", type=Path, default=DEFAULT_OUTPUT_NPZ)
    parser.add_argument("--output-glb", type=Path, default=DEFAULT_OUTPUT_GLB)
    parser.add_argument("--output-stl", type=Path, default=DEFAULT_OUTPUT_STL)
    parser.add_argument("--subdivision", type=int, default=10)
    parser.add_argument("--radius-field-key", type=str, default="assigned_radius_mm")
    parser.add_argument("--skip-mesh-export", action="store_true")
    parser.add_argument("--mc-smooth-sigma", type=float, default=1.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    summary = build_variable_radius_skeleton(
        variable_radius_edges_npz=args.variable_radius_edges_npz,
        aligned_density_npz=args.aligned_density_npz,
        output_npz=args.output_npz,
        subdivision=int(args.subdivision),
        radius_field_key=str(args.radius_field_key),
        output_glb=None if args.skip_mesh_export else args.output_glb,
        output_stl=None if args.skip_mesh_export else args.output_stl,
        mc_smooth_sigma=float(args.mc_smooth_sigma),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
