from __future__ import annotations

import argparse
import json
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
from fem_analysis.fjw_direct_solver import (
    FJWDirectSolverConfig,
    build_fjw_direct_problem,
    build_fjw_direct_problem_setup,
    solve_fjw_direct_case,
)
from fem_analysis.fjw_environment import (
    check_fjw_runtime_environment,
    write_fjw_preflight_report,
)
from fem_analysis.fjw_runtime_config import fjw_runtime_profile_names, get_fjw_runtime_config
from fem_analysis.fjw_workflow_driver import FJWWorkflowDriverRequest
from fem_analysis.fjw_workflow_loaders import load_fjw_workflow_state
from fem_analysis.fjw_workflow_pipeline import (
    FJWAbaqusWorkflowConfig,
    execute_workflow_jobs,
    prepare_workflow,
)
from fem_analysis.fjw_validation import (
    capture_fjw_golden_run,
    validate_run_directory,
    write_validation_report,
)
from fem_analysis.fjw_workflow_optimize import FJWOptimizationConfig, run_fjw_optimization
from fem_analysis.fjw_workflow_runner import run_fjw_sfepy_workflow_iteration


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
        default=2.4,
        help="Outer diameter of the annular cylinder in centimeters.",
    )
    parser.add_argument(
        "--inner-diameter-cm",
        type=float,
        default=1.2,
        help="Inner diameter of the annular cylinder in centimeters.",
    )
    parser.add_argument(
        "--height-cm",
        type=float,
        default=2.0,
        help="Cylinder height in centimeters.",
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


def build_fjw_workflow_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        prog="fem-analysis fjw-workflow",
        description="Generate a Python-managed dry-run of the archived FJW Abaqus workflow.",
    )
    parser.add_argument(
        "--reference-dir",
        default="references/fjw_work",
        help="Directory containing the archived FJW reference files.",
    )
    parser.add_argument(
        "--run-directory",
        default="runs/fjw_workflow",
        help="Directory where generated Abaqus inputs and workflow metadata should be written.",
    )
    parser.add_argument(
        "--mode",
        choices=("single-force", "three-force"),
        default="three-force",
        help="Whether to prepare one forward load case or all three archived load cases.",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=8,
        help="CPU count to encode into generated Abaqus job commands.",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=3,
        help="Number of archived time steps to stage in the workflow.",
    )
    parser.add_argument(
        "--abaqus-executable",
        default="abaqus",
        help="Abaqus executable name used when job commands are later executed.",
    )
    parser.add_argument(
        "--execute-jobs",
        action="store_true",
        help="Also execute the prepared job list and write workflow_execution_manifest.json.",
    )
    parser.add_argument(
        "--real-run",
        action="store_true",
        help="Actually launch Abaqus jobs instead of only producing dry-run execution artifacts.",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Only execute forward jobs when --execute-jobs is enabled.",
    )
    parser.add_argument(
        "--adjoint-only",
        action="store_true",
        help="Only execute adjoint jobs when --execute-jobs is enabled.",
    )
    return parser


def build_fjw_direct_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        prog="fem-analysis fjw-direct",
        description="Build or solve the FJW direct SfePy model from the archived reference inputs.",
    )
    parser.add_argument(
        "--reference-dir",
        default="references/fjw_work",
        help="Directory containing the archived FJW reference files.",
    )
    parser.add_argument(
        "--abaqus-inputs",
        default="datasets/fjw_abaqus_inputs.json",
        help="Structured material/load input JSON extracted from the archived .inp templates.",
    )
    parser.add_argument(
        "--input-inventory",
        default="datasets/fjw_input_inventory.json",
        help="Structured inventory JSON for static external inputs.",
    )
    parser.add_argument(
        "--end1-template",
        default="references/fjw_work/end1.inp",
        help="Template used to recover the original top/bottom node sets.",
    )
    parser.add_argument(
        "--initial-design-mode",
        choices=("single_load", "three_load"),
        default="three_load",
        help="Initial cage density field used to construct the material buckets.",
    )
    parser.add_argument(
        "--load-case",
        default="force_1",
        help="Archived load case name to build or solve.",
    )
    parser.add_argument(
        "--build-problem",
        action="store_true",
        help="Also instantiate the full SfePy Problem after building the setup summary.",
    )
    parser.add_argument(
        "--solve",
        action="store_true",
        help="Run the direct solve after setup/build validation. This can be very heavy on the full model.",
    )
    parser.add_argument(
        "--sfepy-linear-solver",
        choices=("scipy_direct", "scipy_iterative", "petsc_mumps"),
        default="scipy_iterative",
        help="SfePy linear solver profile. scipy_iterative is the Python-only default; scipy_direct is useful for small checks; petsc_mumps is optional.",
    )
    return parser


def build_fjw_sfepy_iteration_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        prog="fem-analysis fjw-sfepy-iterate",
        description="Run one full FJW optimization iteration with the SfePy direct solver backend.",
    )
    parser.add_argument(
        "--reference-dir",
        default="references/fjw_work",
        help="Directory containing the archived FJW reference files.",
    )
    parser.add_argument(
        "--abaqus-inputs",
        default="datasets/fjw_abaqus_inputs.json",
        help="Structured material/load input JSON extracted from the archived .inp templates.",
    )
    parser.add_argument(
        "--input-inventory",
        default="datasets/fjw_input_inventory.json",
        help="Structured inventory JSON for static external inputs.",
    )
    parser.add_argument(
        "--end1-template",
        default="references/fjw_work/end1.inp",
        help="Template used to recover the original top/bottom node sets.",
    )
    parser.add_argument(
        "--initial-design-mode",
        choices=("single_load", "three_load"),
        default="three_load",
        help="Initial cage density field used to construct the material buckets.",
    )
    parser.add_argument(
        "--num-time-steps",
        type=int,
        default=1,
        help="Number of biology time steps to run in this iteration.",
    )
    parser.add_argument(
        "--runtime-profile",
        choices=fjw_runtime_profile_names(),
        default="local",
        help="Runtime defaults for solver, case parallelism, and numeric thread budget.",
    )
    parser.add_argument(
        "--case-parallelism",
        type=int,
        default=None,
        help="Number of force cases to run at once. Defaults to the runtime profile.",
    )
    parser.add_argument(
        "--sfepy-linear-solver",
        choices=("scipy_direct", "scipy_iterative", "petsc_mumps"),
        default=None,
        help="SfePy linear solver profile. Defaults to the runtime profile.",
    )
    parser.add_argument(
        "--disable-sfepy-setup-cache",
        action="store_true",
        help="Disable direct-solver setup caching for debugging.",
    )
    return parser


def build_fjw_optimize_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        prog="fem-analysis fjw-optimize",
        description="Run the Python-managed FJW outer optimization loop with checkpoint/resume support.",
    )
    parser.add_argument("--reference-dir", default="references/fjw_work")
    parser.add_argument("--abaqus-inputs", default="datasets/fjw_abaqus_inputs.json")
    parser.add_argument("--input-inventory", default="datasets/fjw_input_inventory.json")
    parser.add_argument("--end1-template", default="references/fjw_work/end1.inp")
    parser.add_argument("--backend", choices=("abaqus", "sfepy"), default="sfepy")
    parser.add_argument("--mode", choices=("three-force",), default="three-force")
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--delta-tol", type=float, default=1.0e-4)
    parser.add_argument("--num-time-steps", type=int, default=3)
    parser.add_argument("--run-directory", default="runs/fjw_optimize")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--abaqus-executable", default="abaqus")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument(
        "--runtime-profile",
        choices=fjw_runtime_profile_names(),
        default="wuyinyun",
        help="Runtime defaults for solver, case parallelism, and numeric thread budget.",
    )
    parser.add_argument(
        "--case-parallelism",
        type=int,
        default=None,
        help="Number of force cases to run at once. Defaults to the runtime profile.",
    )
    parser.add_argument(
        "--solver-threads",
        type=int,
        default=None,
        help="Numeric thread budget per force-case worker. Defaults to the runtime profile.",
    )
    parser.add_argument(
        "--real-run",
        action="store_true",
        help="Required for the Abaqus backend; dry-run jobs do not produce displacement vectors.",
    )
    parser.add_argument(
        "--sfepy-linear-solver",
        choices=("scipy_direct", "scipy_iterative", "petsc_mumps"),
        default=None,
        help="SfePy linear solver profile. Defaults to the runtime profile.",
    )
    parser.add_argument(
        "--disable-sfepy-setup-cache",
        action="store_true",
        help="Disable direct-solver setup caching for debugging.",
    )
    return parser


def build_fjw_validate_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        prog="fem-analysis fjw-validate",
        description="Validate FJW workflow artifacts and optionally compare them with golden outputs.",
    )
    parser.add_argument("--run-directory", default="runs/fjw_optimize")
    parser.add_argument("--golden-directory", default=None)
    parser.add_argument("--output", default=None)
    return parser


def build_fjw_capture_golden_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        prog="fem-analysis fjw-capture-golden",
        description="Capture a compact golden manifest/checksum set from a completed FJW run.",
    )
    parser.add_argument("--run-directory", default="runs/fjw_optimize")
    parser.add_argument("--golden-directory", default="datasets/fjw_golden/captured_run")
    parser.add_argument(
        "--copy-max-bytes",
        type=int,
        default=5_000_000,
        help="Copy files up to this size into the golden directory; larger files are checksum-only.",
    )
    parser.add_argument(
        "--no-copy-files",
        action="store_true",
        help="Write only golden_manifest.json without copying selected files.",
    )
    parser.add_argument(
        "--allow-invalid-run",
        action="store_true",
        help="Allow capture even when the run directory fails structural validation.",
    )
    return parser


def build_fjw_preflight_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        prog="fem-analysis fjw-preflight",
        description="Check whether this machine can run the FJW runtime validation gates.",
    )
    parser.add_argument(
        "--reference-dir",
        default="references/fjw_work",
        help="Directory containing the archived FJW reference files.",
    )
    parser.add_argument(
        "--abaqus-executable",
        default="abaqus",
        help="Abaqus executable name or path to check.",
    )
    parser.add_argument(
        "--golden-directory",
        default=None,
        help="Optional captured golden directory containing golden_manifest.json.",
    )
    parser.add_argument(
        "--require-abaqus",
        action="store_true",
        help="Fail if Abaqus is unavailable.",
    )
    parser.add_argument(
        "--require-petsc-mumps",
        action="store_true",
        help="Fail if petsc4py is unavailable.",
    )
    parser.add_argument(
        "--require-golden",
        action="store_true",
        help="Fail if no historical or captured runtime golden outputs are found.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report path.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = list(argv) if argv is not None else sys.argv[1:]
    known_commands = {
        "annular-cylinder": build_annular_cylinder_parser,
        "fjw-workflow": build_fjw_workflow_parser,
        "fjw-direct": build_fjw_direct_parser,
        "fjw-sfepy-iterate": build_fjw_sfepy_iteration_parser,
        "fjw-optimize": build_fjw_optimize_parser,
        "fjw-validate": build_fjw_validate_parser,
        "fjw-capture-golden": build_fjw_capture_golden_parser,
        "fjw-preflight": build_fjw_preflight_parser,
    }
    command = args[0] if args and args[0] in known_commands else "annular-cylinder"
    parser = known_commands[command]()
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


def handle_fjw_workflow(args: argparse.Namespace) -> None:
    if args.forward_only and args.adjoint_only:
        raise ValueError("--forward-only and --adjoint-only cannot be used together.")

    config = FJWAbaqusWorkflowConfig(
        reference_dir=Path(args.reference_dir),
        run_directory=Path(args.run_directory),
        abaqus_executable=args.abaqus_executable,
        cpus=args.cpus,
        mode=args.mode,
        time_steps=args.time_steps,
        dry_run=not args.real_run,
    )
    prepared = prepare_workflow(config)
    manifest = prepared.manifest
    payload: dict[str, object] = {
        "run_directory": str(Path(args.run_directory).resolve()),
        "job_count": len(manifest.generated_jobs),
        "removed_stale_locks": manifest.removed_stale_locks,
    }

    if args.execute_jobs:
        execution_manifest = execute_workflow_jobs(
            prepared,
            include_forward=not args.adjoint_only,
            include_adjoint=not args.forward_only,
            dry_run=not args.real_run,
        )
        payload["execution_job_count"] = len(execution_manifest.jobs)
        payload["execution_manifest"] = str(
            (config.run_directory / "workflow_execution_manifest.json").resolve()
        )

    print(json.dumps(payload, indent=2))


def handle_fjw_direct(args: argparse.Namespace) -> None:
    workflow_state = _load_workflow_state_from_args(args)
    setup = build_fjw_direct_problem_setup(
        workflow_state,
        load_case_name=args.load_case,
    )
    payload: dict[str, object] = {
        "reference_dir": str(workflow_state.reference_dir.resolve()),
        "load_case": setup.load_case.name,
        "initial_design_mode": workflow_state.initial_state.mode,
        "node_count": int(setup.mesh_coordinates_mm.shape[0]),
        "element_count": int(setup.mesh_connectivity.shape[0]),
        "material_group_count": len(setup.material_groups),
        "top_node_count": int(setup.top_vertex_ids.size),
        "bottom_node_count": int(setup.bottom_vertex_ids.size),
        "top_rp_vertex_id": int(setup.top_rp_vertex_id),
        "bottom_rp_vertex_id": int(setup.bottom_rp_vertex_id),
        "load_vector": setup.load_vector.tolist(),
    }

    if args.build_problem:
        problem = build_fjw_direct_problem(
            setup,
            config=FJWDirectSolverConfig(linear_solver_kind=args.sfepy_linear_solver),
        )
        payload["problem_name"] = problem.name
        payload["equation_names"] = [equation.name for equation in problem.equations]

    if args.solve:
        result = solve_fjw_direct_case(
            workflow_state,
            load_case_name=args.load_case,
            config=FJWDirectSolverConfig(linear_solver_kind=args.sfepy_linear_solver),
        )
        payload["max_displacement_mm"] = result.max_displacement_mm
        payload["top_rp_displacement"] = result.top_rp_displacement.tolist()
        payload["top_rp_rotation"] = result.top_rp_rotation.tolist()

    print(json.dumps(payload, indent=2))


def handle_fjw_sfepy_iteration(args: argparse.Namespace) -> None:
    workflow_state = _load_workflow_state_from_args(args)
    runtime_config = get_fjw_runtime_config(args.runtime_profile)
    sfepy_linear_solver = args.sfepy_linear_solver or runtime_config.sfepy_linear_solver
    case_parallelism = args.case_parallelism or runtime_config.case_parallelism
    result = run_fjw_sfepy_workflow_iteration(
        driver_request=FJWWorkflowDriverRequest(
            workflow_state=workflow_state,
            initial_design_mode=args.initial_design_mode,
            num_time_steps=args.num_time_steps,
            case_parallelism=case_parallelism,
        ),
        solver_config=_build_sfepy_workflow_solver_config(
            sfepy_linear_solver,
            enable_setup_cache=not args.disable_sfepy_setup_cache,
        ),
    )
    aggregate = result.iteration_state.aggregate_terms
    payload: dict[str, object] = {
        "backend": "sfepy_direct",
        "runtime_profile": args.runtime_profile,
        "sfepy_linear_solver": sfepy_linear_solver,
        "case_parallelism": case_parallelism,
        "initial_design_mode": result.workflow_state.initial_state.mode,
        "num_time_steps": int(args.num_time_steps),
        "iteration_index": result.iteration_state.iteration_index,
        "has_placeholder_adjoint": result.iteration_state.has_placeholder_adjoint,
        "design_size": int(result.design.size),
        "initial_design_sum": float(result.design.sum()),
        "next_design_sum": None if result.iteration_state.next_design is None else float(result.iteration_state.next_design.sum()),
        "load_cases": list(result.load_case_names),
        "terminal_bo_sum_by_case": {
            case_result.load_case_name: case_result.terminal_bo_sum
            for case_result in result.single_case_results
        },
    }
    if aggregate is not None:
        payload["objective"] = float(aggregate.objective)
        payload["g2"] = float(aggregate.g2)
    if result.iteration_state.optimization_terms is not None:
        payload["constraint_names"] = list(result.iteration_state.optimization_terms.constraint_names)

    print(json.dumps(payload, indent=2))


def handle_fjw_optimize(args: argparse.Namespace) -> None:
    result = run_fjw_optimization(
        FJWOptimizationConfig(
            reference_dir=Path(args.reference_dir),
            abaqus_inputs_path=Path(args.abaqus_inputs),
            input_inventory_path=Path(args.input_inventory),
            end1_template_path=Path(args.end1_template),
            backend=args.backend,
            mode=args.mode,
            max_iterations=args.max_iterations,
            delta_tol=args.delta_tol,
            num_time_steps=args.num_time_steps,
            run_directory=Path(args.run_directory),
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            abaqus_executable=args.abaqus_executable,
            cpus=args.cpus,
            real_run=args.real_run,
            sfepy_linear_solver=args.sfepy_linear_solver,
            runtime_profile=args.runtime_profile,
            case_parallelism=args.case_parallelism,
            solver_threads=args.solver_threads,
            enable_sfepy_setup_cache=not args.disable_sfepy_setup_cache,
        )
    )
    payload = {
        "backend": result.config.backend,
        "mode": result.config.mode,
        "runtime_profile": result.config.runtime_profile,
        "sfepy_linear_solver": result.config.sfepy_linear_solver,
        "case_parallelism": result.config.case_parallelism,
        "solver_threads": result.config.solver_threads,
        "enable_sfepy_setup_cache": result.config.enable_sfepy_setup_cache,
        "iteration_count": len(result.iterations),
        "final_delta": result.final_delta,
        "stopped_reason": result.stopped_reason,
        "run_directory": str(result.config.run_directory.resolve()),
        "manifest_path": str(result.manifest_path.resolve()),
        "final_design_sum": float(result.final_design.sum()),
    }
    print(json.dumps(payload, indent=2))


def handle_fjw_validate(args: argparse.Namespace) -> None:
    report = validate_run_directory(
        Path(args.run_directory),
        golden_directory=None if args.golden_directory is None else Path(args.golden_directory),
    )
    output_path = write_validation_report(
        report,
        output_path=None if args.output is None else Path(args.output),
    )
    payload = report.as_jsonable()
    payload["report_path"] = str(output_path.resolve())
    print(json.dumps(payload, indent=2))


def handle_fjw_capture_golden(args: argparse.Namespace) -> None:
    report = capture_fjw_golden_run(
        Path(args.run_directory),
        Path(args.golden_directory),
        copy_max_bytes=args.copy_max_bytes,
        copy_files=not args.no_copy_files,
        require_valid_run=not args.allow_invalid_run,
    )
    print(json.dumps(report.as_jsonable(), indent=2))


def handle_fjw_preflight(args: argparse.Namespace) -> None:
    report = check_fjw_runtime_environment(
        reference_dir=Path(args.reference_dir),
        golden_directory=None if args.golden_directory is None else Path(args.golden_directory),
        abaqus_executable=args.abaqus_executable,
        require_abaqus=args.require_abaqus,
        require_petsc_mumps=args.require_petsc_mumps,
        require_golden=args.require_golden,
    )
    payload = report.as_jsonable()
    if args.output is not None:
        output_path = write_fjw_preflight_report(report, output_path=Path(args.output))
        payload["report_path"] = str(output_path.resolve())
    print(json.dumps(payload, indent=2))
    if not report.is_success:
        sys.exit(1)


def _load_workflow_state_from_args(args: argparse.Namespace):
    return load_fjw_workflow_state(
        reference_dir=Path(args.reference_dir),
        abaqus_inputs_path=Path(args.abaqus_inputs),
        input_inventory_path=Path(args.input_inventory),
        end1_template_path=Path(args.end1_template),
        initial_design_mode=args.initial_design_mode,
    )


def _build_sfepy_workflow_solver_config(
    linear_solver_kind: str,
    *,
    enable_setup_cache: bool = True,
):
    from fem_analysis.fjw_direct_solver import FJWDirectSolverConfig
    from fem_analysis.fjw_workflow_sfepy_solver_adapters import FJWSfePyWorkflowSolverConfig

    return FJWSfePyWorkflowSolverConfig(
        direct_solver_config=FJWDirectSolverConfig(linear_solver_kind=linear_solver_kind),
        enable_setup_cache=enable_setup_cache,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "annular-cylinder":
        handle_annular_cylinder(args)
        return

    if args.command == "fjw-workflow":
        handle_fjw_workflow(args)
        return

    if args.command == "fjw-direct":
        handle_fjw_direct(args)
        return

    if args.command == "fjw-sfepy-iterate":
        handle_fjw_sfepy_iteration(args)
        return

    if args.command == "fjw-optimize":
        handle_fjw_optimize(args)
        return

    if args.command == "fjw-validate":
        handle_fjw_validate(args)
        return

    if args.command == "fjw-capture-golden":
        handle_fjw_capture_golden(args)
        return

    if args.command == "fjw-preflight":
        handle_fjw_preflight(args)
        return

    raise ValueError(f"Unsupported command: {args.command}")
