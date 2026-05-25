from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs" / "fjw_optimize_real" / "iter_017"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "iter017_target_modulus_bands.json"


def _load_workflow_state():
    sys.path.insert(0, str(ROOT / "src"))
    from fem_analysis.fjw_workflow_loaders import load_fjw_workflow_state

    return load_fjw_workflow_state(initial_design_mode="three_load")


def _load_design() -> np.ndarray:
    payload = np.load(RUN_DIR / "design_cage.npz")
    return np.asarray(payload["design_cage"], dtype=np.float64).reshape(-1)


def build_target_modulus_bands(
    *,
    quantiles: tuple[float, ...] = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0),
) -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "src"))
    from fem_analysis.fjw_workflow_forward import cage_objective_modulus

    workflow_state = _load_workflow_state()
    design = _load_design()
    modulus = cage_objective_modulus(design, workflow_state.material_constants)

    quantile_values_x = np.quantile(design, quantiles)
    quantile_values_E = np.quantile(modulus, quantiles)

    bands: list[dict[str, object]] = []
    for index in range(len(quantiles) - 1):
        q_lo = float(quantiles[index])
        q_hi = float(quantiles[index + 1])
        x_lo = float(quantile_values_x[index])
        x_hi = float(quantile_values_x[index + 1])
        e_lo = float(quantile_values_E[index])
        e_hi = float(quantile_values_E[index + 1])

        if index == len(quantiles) - 2:
            mask = (design >= x_lo) & (design <= x_hi)
        else:
            mask = (design >= x_lo) & (design < x_hi)

        band_design = design[mask]
        band_modulus = modulus[mask]
        if band_design.size == 0:
            continue

        representative_x = float(np.median(band_design))
        representative_E = float(np.median(band_modulus))
        bands.append(
            {
                "band_index": index,
                "quantile_range": [q_lo, q_hi],
                "design_range": [x_lo, x_hi],
                "target_modulus_range": [e_lo, e_hi],
                "representative_design": representative_x,
                "representative_target_modulus": representative_E,
                "element_count": int(band_design.size),
                "design_mean": float(np.mean(band_design, dtype=np.float64)),
                "target_modulus_mean": float(np.mean(band_modulus, dtype=np.float64)),
            }
        )

    constants = workflow_state.material_constants
    return {
        "source_design_npz": str((RUN_DIR / "design_cage.npz").resolve()),
        "design_count": int(design.size),
        "design_quantiles": list(quantiles),
        "design_quantile_values": [float(v) for v in quantile_values_x.tolist()],
        "target_modulus_quantile_values": [float(v) for v in quantile_values_E.tolist()],
        "global": {
            "design_min": float(np.min(design)),
            "design_mean": float(np.mean(design, dtype=np.float64)),
            "design_max": float(np.max(design)),
            "target_modulus_min": float(np.min(modulus)),
            "target_modulus_mean": float(np.mean(modulus, dtype=np.float64)),
            "target_modulus_max": float(np.max(modulus)),
            "cage_modulus_min": float(constants.cage_modulus_min),
            "cage_modulus_0": float(constants.cage_modulus_0),
            "interpolation_note": "E_target = E_min + E0 * x^3, clipped to E0 in current implementation.",
        },
        "recommended_first_pass_primary_variable": "effective_rod_radius",
        "recommended_first_pass_fixed_variables": {
            "seed_rule": "existing density-guided sampling with fixed gamma",
            "cvt_rule": "existing Lloyd CVT",
            "topology_family": "current Voronoi skeleton workflow",
            "subdivision_rule": "existing fine-grid subdivision",
        },
        "bands": bands,
    }


def main() -> int:
    payload = build_target_modulus_bands()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
