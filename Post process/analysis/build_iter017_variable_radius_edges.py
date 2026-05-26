from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EDGES_NPZ = ROOT / "outputs" / "fjw_optimize_real_iter017" / "fjw_iter017_voronoi_edges_density.npz"
DEFAULT_ALIGNED_DENSITY_NPZ = ROOT / "outputs" / "fjw_optimize_real_iter017" / "fjw_iter017_aligned_density_gamma1.npz"
DEFAULT_LOOKUP_JSON = (
    ROOT / "Post process" / "analysis" / "output" / "iter017_band_radius_lookup_combined_seed55_plus_lowmid.json"
)
DEFAULT_OUTPUT_NPZ = (
    ROOT / "outputs" / "fjw_optimize_real_iter017" / "fjw_iter017_voronoi_edges_variable_radius_seed55_plus_lowmid.npz"
)

E0_CAGE_MPA = 110000.0
EMIN_CAGE_MPA = 11.0


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_modulus_from_density(density: np.ndarray) -> np.ndarray:
    density = np.clip(np.asarray(density, dtype=np.float64), 0.001, 1.0)
    modulus = EMIN_CAGE_MPA + E0_CAGE_MPA * np.power(density, 3)
    return np.minimum(modulus, E0_CAGE_MPA)


def _build_band_rows(lookup_payload: dict[str, object]) -> list[dict[str, float | int | str]]:
    rows = []
    for row in lookup_payload["lookup_rows"]:
        rows.append(
            {
                "band_index": int(row["band_index"]),
                "target_modulus_mpa": float(row["representative_target_modulus"]),
                "target_modulus_gpa": float(row["representative_target_modulus_gpa"]),
                "assigned_radius_mm": float(row["assigned_radius_mm"]),
                "assignment_status": str(row["assignment_status"]),
            }
        )
    rows.sort(key=lambda item: float(item["target_modulus_mpa"]))
    return rows


def _assign_rows_by_target_modulus(
    target_modulus_mpa: np.ndarray,
    band_rows: list[dict[str, float | int | str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = np.asarray([float(row["target_modulus_mpa"]) for row in band_rows], dtype=np.float64)
    if targets.size == 0:
        raise ValueError("lookup rows are empty.")

    midpoints = (targets[:-1] + targets[1:]) * 0.5
    flat = np.asarray(target_modulus_mpa, dtype=np.float64).reshape(-1)
    indices = np.searchsorted(midpoints, flat, side="right")

    band_indices = np.asarray([int(band_rows[idx]["band_index"]) for idx in indices], dtype=np.int32)
    radii_mm = np.asarray([float(band_rows[idx]["assigned_radius_mm"]) for idx in indices], dtype=np.float32)
    statuses = np.asarray([str(band_rows[idx]["assignment_status"]) for idx in indices], dtype=object)
    return (
        band_indices.reshape(target_modulus_mpa.shape),
        radii_mm.reshape(target_modulus_mpa.shape),
        statuses.reshape(target_modulus_mpa.shape),
    )


def build_variable_radius_edges(
    *,
    edges_npz: Path,
    aligned_density_npz: Path,
    lookup_json: Path,
    output_npz: Path,
) -> dict[str, object]:
    edge_payload = np.load(edges_npz)
    aligned_payload = np.load(aligned_density_npz)
    lookup_payload = _load_json(lookup_json)

    edges = np.asarray(edge_payload["edges"], dtype=np.float32)
    density = aligned_payload["density_milli"].astype(np.float32) / 1000.0
    if edges.ndim != 3 or edges.shape[1:] != (2, 3):
        raise ValueError("Expected edges to have shape (E, 2, 3).")

    band_rows = _build_band_rows(lookup_payload)
    edge_midpoints = edges.mean(axis=1)
    sample_indices = np.rint(edge_midpoints).astype(np.int32)
    grid_shape = np.asarray(density.shape, dtype=np.int32)
    sample_indices = np.clip(sample_indices, 0, grid_shape[None, :] - 1)

    sampled_density = density[
        sample_indices[:, 0],
        sample_indices[:, 1],
        sample_indices[:, 2],
    ]
    target_modulus_mpa = _target_modulus_from_density(sampled_density)
    band_indices, assigned_radius_mm, assignment_statuses = _assign_rows_by_target_modulus(
        target_modulus_mpa,
        band_rows,
    )

    payload = {
        "edges": edges,
        "edge_midpoints": edge_midpoints.astype(np.float32),
        "edge_midpoint_indices_xyz": sample_indices.astype(np.int32),
        "sampled_density": sampled_density.astype(np.float32),
        "target_modulus_mpa": target_modulus_mpa.astype(np.float32),
        "assigned_band_index": band_indices.astype(np.int32),
        "assigned_radius_mm": assigned_radius_mm.astype(np.float32),
        "assignment_status": assignment_statuses,
        "source_edges_npz": np.array(str(edges_npz.resolve())),
        "source_aligned_density_npz": np.array(str(aligned_density_npz.resolve())),
        "source_lookup_json": np.array(str(lookup_json.resolve())),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **payload)

    unique_bands, counts = np.unique(band_indices, return_counts=True)
    band_counts = {int(band): int(count) for band, count in zip(unique_bands, counts, strict=True)}
    return {
        "output_npz": str(output_npz.resolve()),
        "edge_count": int(edges.shape[0]),
        "assigned_radius_min_mm": float(assigned_radius_mm.min()),
        "assigned_radius_max_mm": float(assigned_radius_mm.max()),
        "band_counts": band_counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assign bandwise calibrated radii to iter_017 Voronoi edges by sampling the aligned density field."
    )
    parser.add_argument("--edges-npz", type=Path, default=DEFAULT_EDGES_NPZ)
    parser.add_argument("--aligned-density-npz", type=Path, default=DEFAULT_ALIGNED_DENSITY_NPZ)
    parser.add_argument("--lookup-json", type=Path, default=DEFAULT_LOOKUP_JSON)
    parser.add_argument("--output-npz", type=Path, default=DEFAULT_OUTPUT_NPZ)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    summary = build_variable_radius_edges(
        edges_npz=args.edges_npz,
        aligned_density_npz=args.aligned_density_npz,
        lookup_json=args.lookup_json,
        output_npz=args.output_npz,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
