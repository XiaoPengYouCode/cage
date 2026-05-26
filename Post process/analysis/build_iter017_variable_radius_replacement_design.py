from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "fjw_optimize_real_iter017"
DEFAULT_SKELETON_NPZ = OUTPUT_DIR / "fjw_iter017_skeleton_voxels_variable_radius.npz"
DEFAULT_ALIGNED_NPZ = OUTPUT_DIR / "fjw_iter017_aligned_density_gamma1.npz"
DEFAULT_LOOKUP_JSON = ROOT / "Post process" / "analysis" / "output" / "iter017_band_radius_lookup.json"
DEFAULT_OUTPUT_NPZ = OUTPUT_DIR / "fjw_iter017_replacement_design_variable_radius.npz"

X_MIN = 0.001
E0_CAGE_MPA = 110000.0
EMIN_CAGE_MPA = 11.0


def _load_workflow_state():
    sys.path.insert(0, str(ROOT / "src"))
    from fem_analysis.fjw_workflow_loaders import load_fjw_workflow_state

    return load_fjw_workflow_state(initial_design_mode="three_load")


def _restore_aligned_points_to_original_voxels(
    aligned_points_m: np.ndarray,
    *,
    restore_R: np.ndarray,
    restore_t_m: np.ndarray,
    voxel_size_m: float,
) -> np.ndarray:
    original_points_m = (restore_R @ aligned_points_m.T).T + restore_t_m
    return original_points_m / voxel_size_m


def _load_radius_modulus_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "stable_radius_support_mm" in payload and "stable_modulus_support_gpa" in payload:
        radii = np.asarray(payload["stable_radius_support_mm"], dtype=np.float64)
        moduli_gpa = np.asarray(payload["stable_modulus_support_gpa"], dtype=np.float64)
    else:
        rows = payload.get("inverse_lookup_rows") or payload["radius_calibration_rows"]
        radii = np.asarray([float(row["radius_mm"]) for row in rows], dtype=np.float64)
        moduli_gpa = np.asarray([float(row["apparent_modulus_gpa"]["mean"]) for row in rows], dtype=np.float64)
    order = np.argsort(radii)
    return radii[order], moduli_gpa[order]


def _radius_to_modulus_gpa(radius_mm: np.ndarray, support_r_mm: np.ndarray, support_e_gpa: np.ndarray) -> np.ndarray:
    flat = np.asarray(radius_mm, dtype=np.float64).reshape(-1)
    interp = np.interp(flat, support_r_mm, support_e_gpa, left=support_e_gpa[0], right=support_e_gpa[-1])
    return interp.reshape(radius_mm.shape)


def _modulus_mpa_to_design_field(modulus_mpa: np.ndarray) -> np.ndarray:
    modulus_mpa = np.asarray(modulus_mpa, dtype=np.float64)
    normalized = np.clip((modulus_mpa - EMIN_CAGE_MPA) / E0_CAGE_MPA, 0.0, 1.0)
    design = np.cbrt(normalized)
    return np.clip(design, X_MIN, 1.0)


def build_replacement_design(
    *,
    skeleton_npz: Path,
    aligned_npz: Path,
    lookup_json: Path,
    output_npz: Path,
) -> dict[str, object]:
    workflow_state = _load_workflow_state()
    skeleton = np.load(skeleton_npz, allow_pickle=True)
    aligned = np.load(aligned_npz, allow_pickle=True)

    occupancy = np.asarray(skeleton["voxels"], dtype=bool)
    voxel_radius_mm = np.asarray(skeleton["voxel_radius_mm"], dtype=np.float32)
    if occupancy.shape != voxel_radius_mm.shape:
        raise ValueError("voxels and voxel_radius_mm must have the same shape.")

    origin_m = np.asarray(skeleton["origin_m"], dtype=np.float64)
    fine_voxel_size_xyz_m = np.asarray(skeleton["voxel_size_xyz_m"], dtype=np.float64)
    restore_R = np.asarray(aligned["restore_R"], dtype=np.float64)
    restore_t_m = np.asarray(aligned["restore_t"], dtype=np.float64)

    fine_voxel_size_m = float(fine_voxel_size_xyz_m[0])
    coarse_voxel_size_mm = float(workflow_state.material_constants.voxel_volume ** (1.0 / 3.0))
    coarse_voxel_size_m = coarse_voxel_size_mm / 1e3
    raw_shape = np.asarray(workflow_state.mesh.grid_shape_xyz, dtype=np.int32)

    support_r_mm, support_e_gpa = _load_radius_modulus_curve(lookup_json)

    occupied_indices = np.argwhere(occupancy)
    aligned_points_m = origin_m + occupied_indices.astype(np.float64) * fine_voxel_size_m
    original_points_vox = _restore_aligned_points_to_original_voxels(
        aligned_points_m,
        restore_R=restore_R,
        restore_t_m=restore_t_m,
        voxel_size_m=coarse_voxel_size_m,
    )

    coarse_indices = np.floor(original_points_vox + 1e-8).astype(np.int32)
    valid_mask = np.all((coarse_indices >= 0) & (coarse_indices < raw_shape[None, :]), axis=1)
    coarse_indices = coarse_indices[valid_mask]
    occupied_indices = occupied_indices[valid_mask]

    occupied_radius_mm = voxel_radius_mm[
        occupied_indices[:, 0],
        occupied_indices[:, 1],
        occupied_indices[:, 2],
    ].astype(np.float64)
    occupied_modulus_gpa = _radius_to_modulus_gpa(occupied_radius_mm, support_r_mm, support_e_gpa)

    coarse_counts = np.zeros(tuple(int(v) for v in raw_shape.tolist()), dtype=np.int32)
    coarse_radius_sum = np.zeros_like(coarse_counts, dtype=np.float64)
    coarse_modulus_sum_gpa = np.zeros_like(coarse_counts, dtype=np.float64)
    coarse_radius_max_mm = np.zeros_like(coarse_counts, dtype=np.float32)

    np.add.at(coarse_counts, (coarse_indices[:, 0], coarse_indices[:, 1], coarse_indices[:, 2]), 1)
    np.add.at(coarse_radius_sum, (coarse_indices[:, 0], coarse_indices[:, 1], coarse_indices[:, 2]), occupied_radius_mm)
    np.add.at(coarse_modulus_sum_gpa, (coarse_indices[:, 0], coarse_indices[:, 1], coarse_indices[:, 2]), occupied_modulus_gpa)
    np.maximum.at(
        coarse_radius_max_mm,
        (coarse_indices[:, 0], coarse_indices[:, 1], coarse_indices[:, 2]),
        occupied_radius_mm.astype(np.float32),
    )

    subdivision = int(round(coarse_voxel_size_m / fine_voxel_size_m))
    fine_per_coarse = subdivision**3
    coarse_fill = coarse_counts.astype(np.float64) / float(fine_per_coarse)
    coarse_fill = np.clip(coarse_fill, 0.0, 1.0)

    coarse_radius_mean_mm = np.divide(
        coarse_radius_sum,
        np.maximum(coarse_counts, 1),
        out=np.zeros_like(coarse_radius_sum),
        where=coarse_counts > 0,
    )
    coarse_modulus_mean_gpa = np.divide(
        coarse_modulus_sum_gpa,
        np.maximum(coarse_counts, 1),
        out=np.zeros_like(coarse_modulus_sum_gpa),
        where=coarse_counts > 0,
    )

    # The calibrated E_eff(r) already comes from a homogenized Voronoi unit-cell response.
    # Multiplying it by another coarse fill fraction would penalize porosity twice:
    # once inside the local calibration, and once again during coarse aggregation.
    coarse_proxy_modulus_fill_scaled_gpa = coarse_fill * coarse_modulus_mean_gpa
    coarse_proxy_modulus_mean_only_gpa = np.where(
        coarse_counts > 0,
        coarse_modulus_mean_gpa,
        0.0,
    )
    coarse_proxy_modulus_gpa = coarse_proxy_modulus_mean_only_gpa
    coarse_proxy_modulus_mpa = coarse_proxy_modulus_gpa * 1e3
    coarse_design_modulus_weighted = np.where(
        coarse_counts > 0,
        _modulus_mpa_to_design_field(coarse_proxy_modulus_mpa),
        X_MIN,
    )

    coarse_design_binary = np.where(coarse_counts > 0, 1.0, X_MIN).astype(np.float64)
    coarse_design_fill_fraction = np.where(coarse_counts > 0, coarse_fill, X_MIN).astype(np.float64)

    design_anchor_indices = np.asarray(workflow_state.mesh.design_anchor_indices, dtype=np.int32)
    replacement_binary = coarse_design_binary[
        design_anchor_indices[:, 0],
        design_anchor_indices[:, 1],
        design_anchor_indices[:, 2],
    ]
    replacement_fill_fraction = coarse_design_fill_fraction[
        design_anchor_indices[:, 0],
        design_anchor_indices[:, 1],
        design_anchor_indices[:, 2],
    ]
    replacement_modulus_weighted = coarse_design_modulus_weighted[
        design_anchor_indices[:, 0],
        design_anchor_indices[:, 1],
        design_anchor_indices[:, 2],
    ]
    replacement_radius_mean_mm = coarse_radius_mean_mm[
        design_anchor_indices[:, 0],
        design_anchor_indices[:, 1],
        design_anchor_indices[:, 2],
    ]
    replacement_radius_max_mm = coarse_radius_max_mm[
        design_anchor_indices[:, 0],
        design_anchor_indices[:, 1],
        design_anchor_indices[:, 2],
    ]
    replacement_proxy_modulus_gpa = coarse_proxy_modulus_gpa[
        design_anchor_indices[:, 0],
        design_anchor_indices[:, 1],
        design_anchor_indices[:, 2],
    ]

    payload = {
        "design_cage_binary": replacement_binary.astype(np.float64),
        "design_cage_fill_fraction": replacement_fill_fraction.astype(np.float64),
        "design_cage_modulus_weighted": replacement_modulus_weighted.astype(np.float64),
        "design_radius_mean_mm": replacement_radius_mean_mm.astype(np.float32),
        "design_radius_max_mm": replacement_radius_max_mm.astype(np.float32),
        "design_proxy_modulus_gpa": replacement_proxy_modulus_gpa.astype(np.float32),
        "design_proxy_modulus_fill_scaled_gpa": coarse_proxy_modulus_fill_scaled_gpa[
            design_anchor_indices[:, 0],
            design_anchor_indices[:, 1],
            design_anchor_indices[:, 2],
        ].astype(np.float32),
        "design_anchor_indices": design_anchor_indices.astype(np.int32),
        "coarse_counts_grid": coarse_counts.astype(np.int32),
        "coarse_fill_grid": coarse_fill.astype(np.float32),
        "coarse_radius_mean_mm_grid": coarse_radius_mean_mm.astype(np.float32),
        "coarse_radius_max_mm_grid": coarse_radius_max_mm.astype(np.float32),
        "coarse_proxy_modulus_gpa_grid": coarse_proxy_modulus_gpa.astype(np.float32),
        "coarse_proxy_modulus_fill_scaled_gpa_grid": coarse_proxy_modulus_fill_scaled_gpa.astype(np.float32),
        "raw_grid_shape_xyz": raw_shape.astype(np.int32),
        "coarse_voxel_size_m": np.array(coarse_voxel_size_m, dtype=np.float32),
        "coarse_voxel_size_mm": np.array(coarse_voxel_size_mm, dtype=np.float32),
        "fine_voxel_size_m": np.array(fine_voxel_size_m, dtype=np.float32),
        "subdivision": np.array(subdivision, dtype=np.int32),
        "source_skeleton_npz": np.array(str(skeleton_npz.resolve())),
        "source_aligned_npz": np.array(str(aligned_npz.resolve())),
        "source_lookup_json": np.array(str(lookup_json.resolve())),
        "design_rule": np.array("modulus_mean_radius_proxy"),
        "design_rule_note": np.array(
            "Per-coarse-cell proxy modulus = mean(calibrated apparent modulus of occupied fine voxels). "
            "The old fill_fraction * E_eff variant is still stored as a diagnostic field, but is not used "
            "as the primary FE replacement because it double-penalizes porosity."
        ),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **payload)

    return {
        "output_npz": str(output_npz.resolve()),
        "replacement_nonzero_count": int(np.count_nonzero(replacement_fill_fraction > X_MIN)),
        "replacement_binary_sum": float(np.sum(replacement_binary, dtype=np.float64)),
        "replacement_fill_fraction_sum": float(np.sum(replacement_fill_fraction, dtype=np.float64)),
        "replacement_modulus_weighted_sum": float(np.sum(replacement_modulus_weighted, dtype=np.float64)),
        "replacement_modulus_weighted_mean": float(np.mean(replacement_modulus_weighted, dtype=np.float64)),
        "replacement_modulus_weighted_max": float(np.max(replacement_modulus_weighted)),
        "replacement_proxy_modulus_gpa_max": float(np.max(replacement_proxy_modulus_gpa)),
        "replacement_proxy_modulus_fill_scaled_gpa_max": float(
            np.max(
                coarse_proxy_modulus_fill_scaled_gpa[
                    design_anchor_indices[:, 0],
                    design_anchor_indices[:, 1],
                    design_anchor_indices[:, 2],
                ]
            )
        ),
        "subdivision": subdivision,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a first FE-ready replacement design from the variable-radius iter_017 skeleton."
    )
    parser.add_argument("--skeleton-npz", type=Path, default=DEFAULT_SKELETON_NPZ)
    parser.add_argument("--aligned-npz", type=Path, default=DEFAULT_ALIGNED_NPZ)
    parser.add_argument("--lookup-json", type=Path, default=DEFAULT_LOOKUP_JSON)
    parser.add_argument("--calibration-summary-json", type=Path, dest="lookup_json_legacy")
    parser.add_argument("--output-npz", type=Path, default=DEFAULT_OUTPUT_NPZ)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    summary = build_replacement_design(
        skeleton_npz=args.skeleton_npz,
        aligned_npz=args.aligned_npz,
        lookup_json=args.lookup_json_legacy or args.lookup_json,
        output_npz=args.output_npz,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
