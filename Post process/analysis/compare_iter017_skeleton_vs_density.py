from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "fjw_optimize_real_iter017"
RUN_DIR = ROOT / "runs" / "fjw_optimize_real" / "iter_017"
DEFAULT_REPLACEMENT_DESIGN_NPZ = OUTPUT_DIR / "fjw_iter017_replacement_design_variable_radius.npz"


def _load_workflow_state():
    sys.path.insert(0, str(ROOT / "src"))
    from fem_analysis.fjw_workflow_loaders import load_fjw_workflow_state

    return load_fjw_workflow_state(initial_design_mode="three_load")


def _load_iter017_design() -> np.ndarray:
    payload = np.load(RUN_DIR / "design_cage.npz")
    return np.asarray(payload["design_cage"], dtype=np.float64).reshape(-1)


def _load_iter017_reference_case(load_case_name: str) -> dict[str, object]:
    summary = json.loads((ROOT / "Post process" / "analysis" / "output" / "iter017_summary.json").read_text(encoding="utf-8"))
    forward_step = np.load(RUN_DIR / load_case_name / "forward_t0" / "forward_step.npz")
    case_summary = summary["cases"][load_case_name]
    ref_bone_s = np.asarray(forward_step["bone_s"], dtype=np.float64)
    ref_delta = np.asarray(forward_step["bone_density_delta"], dtype=np.float64)
    return {
        "max_displacement_mm": float(case_summary["element_displacement_max"]),
        "top_rp_displacement": None,
        "bo_sum_next": float(case_summary["bo_sum_final"]),
        "bone_s_mean": float(np.mean(ref_bone_s, dtype=np.float64)),
        "bone_s_p95": float(np.percentile(ref_bone_s, 95)),
        "bone_s_max": float(np.max(ref_bone_s)),
        "bone_density_delta_mean": float(np.mean(ref_delta, dtype=np.float64)),
        "bone_density_delta_sum": float(np.sum(ref_delta, dtype=np.float64)),
        "bone_s_array": ref_bone_s,
        "bone_density_delta_array": ref_delta,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _restore_aligned_points_to_original_voxels(
    aligned_points_m: np.ndarray,
    *,
    restore_R: np.ndarray,
    restore_t_m: np.ndarray,
    voxel_size_m: float,
) -> np.ndarray:
    original_points_m = (restore_R @ aligned_points_m.T).T + restore_t_m
    return original_points_m / voxel_size_m


def build_replacement_design_from_skeleton() -> dict[str, object]:
    workflow_state = _load_workflow_state()
    skeleton = np.load(OUTPUT_DIR / "fjw_iter017_skeleton_voxels_density.npz")
    aligned = np.load(OUTPUT_DIR / "fjw_iter017_aligned_density_gamma1.npz")

    occupancy = np.asarray(skeleton["voxels"], dtype=bool)
    origin_m = np.asarray(skeleton["origin_m"], dtype=np.float64)
    fine_voxel_size_xyz_m = np.asarray(skeleton["voxel_size_xyz_m"], dtype=np.float64)
    restore_R = np.asarray(aligned["restore_R"], dtype=np.float64)
    restore_t_m = np.asarray(aligned["restore_t"], dtype=np.float64)

    fine_voxel_size_m = float(fine_voxel_size_xyz_m[0])
    coarse_voxel_size_mm = float(workflow_state.material_constants.voxel_volume ** (1.0 / 3.0))
    coarse_voxel_size_m = coarse_voxel_size_mm / 1e3

    occupied_indices = np.argwhere(occupancy)
    aligned_points_m = origin_m + occupied_indices.astype(np.float64) * fine_voxel_size_m
    original_points_vox = _restore_aligned_points_to_original_voxels(
        aligned_points_m,
        restore_R=restore_R,
        restore_t_m=restore_t_m,
        voxel_size_m=coarse_voxel_size_m,
    )

    coarse_indices = np.floor(original_points_vox + 1e-8).astype(np.int32)
    raw_shape = np.asarray(workflow_state.mesh.grid_shape_xyz, dtype=np.int32)
    valid_mask = np.all((coarse_indices >= 0) & (coarse_indices < raw_shape), axis=1)
    coarse_indices = coarse_indices[valid_mask]

    coarse_counts = np.zeros(tuple(int(v) for v in raw_shape.tolist()), dtype=np.int32)
    np.add.at(
        coarse_counts,
        (coarse_indices[:, 0], coarse_indices[:, 1], coarse_indices[:, 2]),
        1,
    )

    subdivision = int(round(coarse_voxel_size_m / fine_voxel_size_m))
    fine_per_coarse = subdivision**3
    coarse_fill = coarse_counts.astype(np.float64) / float(fine_per_coarse)
    coarse_fill = np.clip(coarse_fill, 0.0, 1.0)

    design_anchor_indices = np.asarray(workflow_state.mesh.design_anchor_indices, dtype=np.int32)
    replacement_design = coarse_fill[
        design_anchor_indices[:, 0],
        design_anchor_indices[:, 1],
        design_anchor_indices[:, 2],
    ]

    x_min = 0.001
    replacement_design_binary = np.where(replacement_design > 0.0, 1.0, x_min).astype(np.float64)
    replacement_design_fraction = np.where(replacement_design > 0.0, replacement_design, x_min).astype(np.float64)

    payload = {
        "design_cage_binary": replacement_design_binary,
        "design_cage_fill_fraction": replacement_design_fraction,
        "design_anchor_indices": design_anchor_indices,
        "coarse_counts_grid": coarse_counts,
        "coarse_fill_grid": coarse_fill,
        "raw_grid_shape_xyz": raw_shape,
        "coarse_voxel_size_m": coarse_voxel_size_m,
        "coarse_voxel_size_mm": coarse_voxel_size_mm,
        "fine_voxel_size_m": fine_voxel_size_m,
        "subdivision": subdivision,
        "source_skeleton_npz": str((OUTPUT_DIR / "fjw_iter017_skeleton_voxels_density.npz").resolve()),
        "source_aligned_npz": str((OUTPUT_DIR / "fjw_iter017_aligned_density_gamma1.npz").resolve()),
    }

    out_path = OUTPUT_DIR / "fjw_iter017_replacement_design_from_skeleton.npz"
    np.savez_compressed(out_path, **payload)

    return {
        "output_npz": str(out_path),
        "replacement_nonzero_count": int(np.count_nonzero(replacement_design > 0.0)),
        "replacement_binary_sum": float(np.sum(replacement_design_binary, dtype=np.float64)),
        "replacement_fraction_sum": float(np.sum(replacement_design_fraction, dtype=np.float64)),
        "replacement_fraction_mean": float(np.mean(replacement_design_fraction, dtype=np.float64)),
        "replacement_fraction_max": float(np.max(replacement_design_fraction)),
        "subdivision": subdivision,
        "fine_voxel_size_m": fine_voxel_size_m,
        "coarse_voxel_size_m": coarse_voxel_size_m,
        "coarse_voxel_size_mm": coarse_voxel_size_mm,
    }


def _load_replacement_design(
    *,
    replacement_npz_path: Path,
    design_mode: str,
) -> np.ndarray:
    replacement_payload = np.load(replacement_npz_path, allow_pickle=True)
    if design_mode not in {"binary", "fill_fraction", "modulus_weighted"}:
        raise ValueError(f"Unsupported replacement design mode: {design_mode!r}.")
    key_map = {
        "binary": "design_cage_binary",
        "fill_fraction": "design_cage_fill_fraction",
        "modulus_weighted": "design_cage_modulus_weighted",
    }
    key = key_map[design_mode]
    if key not in replacement_payload:
        raise KeyError(f"{replacement_npz_path} does not contain replacement field {key!r}.")
    return np.asarray(replacement_payload[key], dtype=np.float64).reshape(-1)


def _build_forward_step_from_direct_solve(
    *,
    workflow_state,
    load_case_name: str,
    design_cage: np.ndarray,
    obj_bo: np.ndarray,
):
    from fem_analysis.fjw_direct_solver import FJWDirectSolverConfig, solve_fjw_direct_case
    from fem_analysis.fjw_workflow_forward import build_single_load_time_step_result
    from fem_analysis.fjw_workflow_vectors import build_element_displacement_cache

    started = time.time()
    print(f"[solve] start {load_case_name}", flush=True)
    direct_result = solve_fjw_direct_case(
        workflow_state,
        load_case_name=load_case_name,
        design_cage=design_cage,
        obj_bo=obj_bo,
        config=FJWDirectSolverConfig(store_nodal_displacements=True),
    )
    print(f"[solve] finished {load_case_name} in {time.time() - started:.2f}s", flush=True)
    if direct_result.nodal_displacements is None:
        raise RuntimeError(f"Expected nodal displacements for load case {load_case_name}.")

    started = time.time()
    print(f"[post] build element cache {load_case_name}", flush=True)
    element_cache = build_element_displacement_cache(direct_result.nodal_displacements, workflow_state)
    print(f"[post] element cache ready {load_case_name} in {time.time() - started:.2f}s", flush=True)
    started = time.time()
    print(f"[post] build forward step {load_case_name}", flush=True)
    forward_step = build_single_load_time_step_result(
        load_case_name=load_case_name,
        time_index=0,
        element_displacements=element_cache.vectors_2d,
        mesh=workflow_state.mesh,
        material_constants=workflow_state.material_constants,
        design_cage=design_cage,
        obj_bo=obj_bo,
    )
    print(f"[post] forward step ready {load_case_name} in {time.time() - started:.2f}s", flush=True)
    return direct_result, forward_step


def run_forward_comparison(
    *,
    design_mode: str,
    load_case_names: tuple[str, ...],
    replacement_npz_path: Path,
) -> dict[str, object]:
    workflow_state = _load_workflow_state()
    replacement_design = _load_replacement_design(
        replacement_npz_path=replacement_npz_path,
        design_mode=design_mode,
    )
    original_design = _load_iter017_design()
    obj_bo = workflow_state.initial_state.obj_bo.copy()

    results: dict[str, object] = {
        "design_mode": design_mode,
        "replacement_npz_path": str(replacement_npz_path.resolve()),
        "reference_design_sum": float(np.sum(original_design, dtype=np.float64)),
        "replacement_design_sum": float(np.sum(replacement_design, dtype=np.float64)),
        "cases": {},
    }
    case_tag = "-".join(load_case_names)
    progress_path = OUTPUT_DIR / f"fjw_iter017_skeleton_vs_density_{design_mode}_{case_tag}_progress.json"
    _write_json(
        progress_path,
        {
            "status": "started",
            "design_mode": design_mode,
            "load_case_names": list(load_case_names),
            "cases_completed": [],
        },
    )

    for load_case_name in load_case_names:
        print(f"[case] begin {load_case_name}", flush=True)
        reference_case = _load_iter017_reference_case(load_case_name)
        rep_direct, rep_step = _build_forward_step_from_direct_solve(
            workflow_state=workflow_state,
            load_case_name=load_case_name,
            design_cage=replacement_design,
            obj_bo=obj_bo,
        )

        ref_bone_s = np.asarray(reference_case["bone_s_array"], dtype=np.float64)
        rep_bone_s = np.asarray(rep_step.bone_s, dtype=np.float64)
        ref_delta = np.asarray(reference_case["bone_density_delta_array"], dtype=np.float64)
        rep_delta = np.asarray(rep_step.bone_density_delta, dtype=np.float64)

        cases_payload = {
            "reference": {
                "max_displacement_mm": float(reference_case["max_displacement_mm"]),
                "top_rp_displacement": reference_case["top_rp_displacement"],
                "bo_sum_next": float(reference_case["bo_sum_next"]),
                "bone_s_mean": float(reference_case["bone_s_mean"]),
                "bone_s_p95": float(reference_case["bone_s_p95"]),
                "bone_s_max": float(reference_case["bone_s_max"]),
                "bone_density_delta_mean": float(reference_case["bone_density_delta_mean"]),
                "bone_density_delta_sum": float(reference_case["bone_density_delta_sum"]),
            },
            "replacement": {
                "max_displacement_mm": rep_direct.max_displacement_mm,
                "top_rp_displacement": rep_direct.top_rp_displacement.tolist(),
                "bo_sum_next": rep_step.bo_sum_next,
                "bone_s_mean": float(np.mean(rep_bone_s, dtype=np.float64)),
                "bone_s_p95": float(np.percentile(rep_bone_s, 95)),
                "bone_s_max": float(np.max(rep_bone_s)),
                "bone_density_delta_mean": float(np.mean(rep_delta, dtype=np.float64)),
                "bone_density_delta_sum": float(np.sum(rep_delta, dtype=np.float64)),
            },
        }
        cases_payload["comparison"] = {
            "max_displacement_ratio": float(rep_direct.max_displacement_mm / float(reference_case["max_displacement_mm"])),
            "bo_sum_next_ratio": float(rep_step.bo_sum_next / float(reference_case["bo_sum_next"])),
            "bone_s_mean_ratio": float(np.mean(rep_bone_s, dtype=np.float64) / float(reference_case["bone_s_mean"])),
            "bone_density_delta_sum_ratio": float(np.sum(rep_delta, dtype=np.float64) / float(reference_case["bone_density_delta_sum"])),
            "bone_s_correlation": float(np.corrcoef(ref_bone_s, rep_bone_s)[0, 1]),
            "bone_density_delta_correlation": float(np.corrcoef(ref_delta, rep_delta)[0, 1]),
        }
        results["cases"][load_case_name] = cases_payload
        _write_json(
            progress_path,
            {
                "status": "running",
                "design_mode": design_mode,
                "load_case_names": list(load_case_names),
                "cases_completed": list(results["cases"].keys()),
                "latest_case": load_case_name,
            },
        )
        print(f"[case] finished {load_case_name}", flush=True)

    out_path = OUTPUT_DIR / f"fjw_iter017_skeleton_vs_density_{design_mode}_{case_tag}_comparison.json"
    _write_json(out_path, results)
    _write_json(
        progress_path,
        {
            "status": "finished",
            "design_mode": design_mode,
            "load_case_names": list(load_case_names),
            "cases_completed": list(results["cases"].keys()),
            "output_json": str(out_path),
        },
    )
    results["output_json"] = str(out_path)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("build_replacement_design", "run_comparison"),
        required=True,
    )
    parser.add_argument(
        "--design-mode",
        choices=("binary", "fill_fraction", "modulus_weighted"),
        default="binary",
        help="Which replacement design field to use during forward comparison.",
    )
    parser.add_argument(
        "--replacement-npz",
        type=Path,
        default=DEFAULT_REPLACEMENT_DESIGN_NPZ,
        help="Replacement design NPZ containing design_cage_* fields.",
    )
    parser.add_argument(
        "--load-case",
        action="append",
        dest="load_cases",
        choices=("force_1", "force_2", "force_3"),
        help="Restrict comparison to one or more load cases. Defaults to all three.",
    )
    args = parser.parse_args()

    if args.stage == "build_replacement_design":
        payload = build_replacement_design_from_skeleton()
    else:
        load_case_names = tuple(args.load_cases) if args.load_cases else ("force_1", "force_2", "force_3")
        payload = run_forward_comparison(
            design_mode=args.design_mode,
            load_case_names=load_case_names,
            replacement_npz_path=args.replacement_npz,
        )

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
