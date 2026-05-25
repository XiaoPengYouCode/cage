from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "voronoi_radius_calibration"
DEFAULT_BANDS_JSON = ROOT / "Post process" / "analysis" / "output" / "iter017_target_modulus_bands.json"


def _parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in str(raw).split(",")]
    parsed = [float(item) for item in values if item]
    if not parsed:
        raise ValueError("Expected at least one float value.")
    return parsed


def _load_band_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_row(num_seeds: int, rng_seed: int):
    sys.path.insert(0, str(ROOT / "src"))
    from helix_voronoi.pipeline import VoronoiPipeline

    return VoronoiPipeline().build_row(num_seeds=num_seeds, rng_seed=rng_seed)


def _scale_edges(edges, scale: float):
    scaled = []
    for start, end in edges:
        scaled.append((np.asarray(start, dtype=np.float64) * scale, np.asarray(end, dtype=np.float64) * scale))
    return scaled


def _export_stl(edges, radius: float, output_path: Path) -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "src"))
    from helix_voronoi.helix_stl import export_mixed_edges_to_stl

    summary = export_mixed_edges_to_stl(
        edges,
        radius=radius,
        output_path=output_path,
    )
    return {
        "output_path": str(output_path.resolve()),
        "triangle_count": int(summary.triangle_count),
        "straight_edge_count": int(summary.straight_edge_count),
        "interior_edge_count": int(summary.interior_edge_count),
        "node_sphere_count": int(summary.node_sphere_count),
    }


def generate_specs(
    *,
    bands_json: Path,
    output_dir: Path,
    num_seeds: int,
    rng_seed: int,
    radii_mm: list[float],
    cell_size_mm: float,
) -> dict[str, object]:
    band_summary = _load_band_summary(bands_json)
    row = _build_row(num_seeds=num_seeds, rng_seed=rng_seed)
    # STL coordinates are unitless; downstream voxelization code interprets them as mm.
    # Export the calibration geometry directly in mm so the FE evaluator sees the
    # intended physical specimen size.
    scaled_edges = _scale_edges(row.edges, scale=cell_size_mm)

    output_dir.mkdir(parents=True, exist_ok=True)
    specimens: list[dict[str, object]] = []

    for band in band_summary["bands"]:
        band_index = int(band["band_index"])
        representative_design = float(band["representative_design"])
        representative_target_modulus = float(band["representative_target_modulus"])

        for radius_mm in radii_mm:
            tag = f"band{band_index:02d}_cell{cell_size_mm:.3f}mm_r{radius_mm:.3f}mm".replace(".", "p")
            stl_path = output_dir / f"{tag}.stl"
            stl_summary = _export_stl(scaled_edges, radius=radius_mm, output_path=stl_path)
            specimens.append(
                {
                    "tag": tag,
                    "band_index": band_index,
                    "representative_design": representative_design,
                    "representative_target_modulus": representative_target_modulus,
                    "num_seeds": int(num_seeds),
                    "rng_seed": int(rng_seed),
                    "cell_size_mm": float(cell_size_mm),
                    "radius_mm": float(radius_mm),
                    "stl": stl_summary,
                }
            )

    manifest = {
        "source_band_json": str(bands_json.resolve()),
        "output_dir": str(output_dir.resolve()),
        "primary_variable": "effective_rod_radius",
        "fixed_variables": {
            "num_seeds": int(num_seeds),
            "rng_seed": int(rng_seed),
            "topology_family": "helix_voronoi export_mixed unit-cell",
            "seed_rule_note": "Fixed local unit-cell seed realization for first-pass radius sweep.",
        },
        "cell_size_mm": float(cell_size_mm),
        "radii_mm": [float(v) for v in radii_mm],
        "specimen_count": int(len(specimens)),
        "specimens": specimens,
    }
    manifest_path = output_dir / "calibration_spec_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate first-pass Voronoi radius calibration STL specimens from iter_017 target-modulus bands."
    )
    parser.add_argument(
        "--bands-json",
        type=Path,
        default=DEFAULT_BANDS_JSON,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=55,
    )
    parser.add_argument(
        "--radii-mm",
        type=str,
        default="0.10,0.15,0.20,0.25,0.30",
        help="Comma-separated rod radii in millimeters.",
    )
    parser.add_argument(
        "--cell-size-mm",
        type=float,
        default=4.0,
        help="Physical edge length of the unit-cell bounding box in millimeters.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    manifest = generate_specs(
        bands_json=args.bands_json,
        output_dir=args.output_dir,
        num_seeds=int(args.num_seeds),
        rng_seed=int(args.rng_seed),
        radii_mm=_parse_float_list(args.radii_mm),
        cell_size_mm=float(args.cell_size_mm),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
