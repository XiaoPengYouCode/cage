from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fem_analysis.fjw_workflow_optimizer import FJWMMAOptimizer, FJWOptimizationTerms, build_initial_mma_state


def build_minimal_mma_golden(output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    design = np.array([0.3, 0.45, 0.6], dtype=np.float64)
    terms = FJWOptimizationTerms(
        objective=-1.25,
        objective_gradient=np.array([-0.9, 0.2, -0.35], dtype=np.float64),
        constraints=np.array([0.05], dtype=np.float64),
        constraint_gradients=np.array([[1.0, 1.0, 1.0]], dtype=np.float64),
    )
    result = FJWMMAOptimizer().step(design, terms, build_initial_mma_state(design))
    np.savez_compressed(
        output_directory / "mma_one_step.npz",
        design=design,
        xmma=result.design,
        low=result.state.low,
        up=result.state.up,
        objective=np.array(terms.objective, dtype=np.float64),
        objective_gradient=terms.objective_gradient,
        constraints=terms.constraints,
        constraint_gradients=terms.constraint_gradients,
        delta=np.array(result.diagnostics["delta"], dtype=np.float64),
    )
    manifest = {
        "case_name": "minimal_mma_one_step",
        "source": "Pure Python port of references/fjw_work/mmasub.m and subsolv.m",
        "historical_runtime_outputs_missing": [
            "Force_*.mat",
            "U1_ele_nod_dir*.mat",
            "obj_bo*.mat",
            "ob*.mat",
        ],
        "files": ["mma_one_step.npz"],
    }
    (output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Build small FJW golden fixtures.")
    parser.add_argument(
        "--output-directory",
        default="datasets/fjw_golden/minimal_mma",
        help="Directory to receive the generated golden fixture.",
    )
    args = parser.parse_args()
    output_directory = build_minimal_mma_golden(Path(args.output_directory))
    print(output_directory)


if __name__ == "__main__":
    main()
