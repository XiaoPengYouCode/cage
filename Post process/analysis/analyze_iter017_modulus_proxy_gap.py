from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs" / "fjw_optimize_real" / "iter_017"
OUTPUT_DIR = ROOT / "outputs" / "fjw_optimize_real_iter017"
DEFAULT_REPLACEMENT_NPZ = OUTPUT_DIR / "fjw_iter017_replacement_design_variable_radius.npz"
DEFAULT_VARIABLE_RADIUS_EDGES_NPZ = OUTPUT_DIR / "fjw_iter017_voronoi_edges_variable_radius.npz"
DEFAULT_LOOKUP_JSON = ROOT / "Post process" / "analysis" / "output" / "iter017_band_radius_lookup.json"
DEFAULT_OUTPUT_JSON = ROOT / "Post process" / "analysis" / "output" / "iter017_modulus_proxy_gap_summary.json"


def _load_workflow_state():
    sys.path.insert(0, str(ROOT / "src"))
    from fem_analysis.fjw_workflow_loaders import load_fjw_workflow_state

    return load_fjw_workflow_state(initial_design_mode="three_load")


def _load_design() -> np.ndarray:
    payload = np.load(RUN_DIR / "design_cage.npz")
    return np.asarray(payload["design_cage"], dtype=np.float64).reshape(-1)


def _target_modulus_gpa_from_design(design: np.ndarray) -> np.ndarray:
    sys.path.insert(0, str(ROOT / "src"))
    from fem_analysis.fjw_workflow_forward import cage_objective_modulus

    workflow_state = _load_workflow_state()
    modulus_mpa = cage_objective_modulus(design, workflow_state.material_constants)
    return np.asarray(modulus_mpa, dtype=np.float64) / 1e3


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "mean": float(np.mean(values, dtype=np.float64)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def build_summary(
    *,
    replacement_npz: Path,
    variable_radius_edges_npz: Path,
    lookup_json: Path,
) -> dict[str, object]:
    replacement = np.load(replacement_npz, allow_pickle=True)
    edges = np.load(variable_radius_edges_npz, allow_pickle=True)
    lookup = json.loads(lookup_json.read_text(encoding="utf-8"))

    design = _load_design()
    target_modulus_gpa = _target_modulus_gpa_from_design(design)
    proxy_modulus_gpa = np.asarray(replacement["design_proxy_modulus_gpa"], dtype=np.float64).reshape(-1)
    replacement_design = np.asarray(replacement["design_cage_modulus_weighted"], dtype=np.float64).reshape(-1)
    fill_fraction = np.asarray(replacement["design_cage_fill_fraction"], dtype=np.float64).reshape(-1)
    radius_mean_mm = np.asarray(replacement["design_radius_mean_mm"], dtype=np.float64).reshape(-1)
    radius_max_mm = np.asarray(replacement["design_radius_max_mm"], dtype=np.float64).reshape(-1)

    ratio = np.divide(
        proxy_modulus_gpa,
        np.maximum(target_modulus_gpa, 1e-12),
        out=np.zeros_like(proxy_modulus_gpa),
        where=target_modulus_gpa > 0.0,
    )
    abs_gap_gpa = proxy_modulus_gpa - target_modulus_gpa

    assigned_band_index = np.asarray(edges["assigned_band_index"], dtype=np.int32).reshape(-1)
    assignment_status = np.asarray(edges["assignment_status"]).reshape(-1)
    assigned_radius_mm_edges = np.asarray(edges["assigned_radius_mm"], dtype=np.float64).reshape(-1)
    edge_target_modulus_gpa = np.asarray(edges["target_modulus_mpa"], dtype=np.float64).reshape(-1) / 1e3

    unique_bands, band_counts = np.unique(assigned_band_index, return_counts=True)
    unique_status, status_counts = np.unique(assignment_status, return_counts=True)

    return {
        "source_design_npz": str((RUN_DIR / "design_cage.npz").resolve()),
        "source_replacement_npz": str(replacement_npz.resolve()),
        "source_variable_radius_edges_npz": str(variable_radius_edges_npz.resolve()),
        "source_lookup_json": str(lookup_json.resolve()),
        "lookup_support": {
            "stable_radius_support_mm": lookup["stable_radius_support_mm"],
            "stable_modulus_support_gpa": lookup["stable_modulus_support_gpa"],
        },
        "design_vs_proxy": {
            "target_modulus_gpa": _summary_stats(target_modulus_gpa),
            "proxy_modulus_gpa": _summary_stats(proxy_modulus_gpa),
            "proxy_over_target_ratio": _summary_stats(ratio),
            "proxy_minus_target_gpa": _summary_stats(abs_gap_gpa),
            "target_proxy_correlation": float(np.corrcoef(target_modulus_gpa, proxy_modulus_gpa)[0, 1]),
            "replacement_design_sum": float(np.sum(replacement_design, dtype=np.float64)),
            "reference_design_sum": float(np.sum(design, dtype=np.float64)),
            "replacement_fill_fraction_sum": float(np.sum(fill_fraction, dtype=np.float64)),
        },
        "replacement_geometry": {
            "fill_fraction": _summary_stats(fill_fraction),
            "radius_mean_mm": _summary_stats(radius_mean_mm),
            "radius_max_mm": _summary_stats(radius_max_mm),
            "nonzero_radius_cell_count": int(np.count_nonzero(radius_max_mm > 0.0)),
        },
        "edge_assignment": {
            "edge_target_modulus_gpa": _summary_stats(edge_target_modulus_gpa),
            "assigned_radius_mm": _summary_stats(assigned_radius_mm_edges),
            "band_counts": {int(b): int(c) for b, c in zip(unique_bands, band_counts, strict=True)},
            "assignment_status_counts": {str(s): int(c) for s, c in zip(unique_status.tolist(), status_counts, strict=True)},
            "clamped_low_fraction": float(np.mean(assignment_status == "clamped_low")),
            "clamped_high_fraction": float(np.mean(assignment_status == "clamped_high")),
        },
        "diagnosis": [
            "Current proxy modulus is far below the target modulus field on average.",
            "Most Voronoi edges collapse to the current minimum stable sampled radius.",
            "The current coarse proxy multiplies fill fraction by local apparent modulus, which strongly reduces the effective field relative to E_target.",
            "This replacement can serve as a first physically interpretable proxy, but not yet as the final structure-level equivalence map.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize the field-level gap between iter_017 target modulus and the current variable-radius proxy replacement.")
    parser.add_argument("--replacement-npz", type=Path, default=DEFAULT_REPLACEMENT_NPZ)
    parser.add_argument("--variable-radius-edges-npz", type=Path, default=DEFAULT_VARIABLE_RADIUS_EDGES_NPZ)
    parser.add_argument("--lookup-json", type=Path, default=DEFAULT_LOOKUP_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    payload = build_summary(
        replacement_npz=args.replacement_npz,
        variable_radius_edges_npz=args.variable_radius_edges_npz,
        lookup_json=args.lookup_json,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
