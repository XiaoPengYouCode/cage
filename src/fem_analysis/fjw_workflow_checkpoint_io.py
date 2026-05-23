from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .fjw_workflow_driver import FJWWorkflowDriverResult
from .fjw_workflow_iteration_state import FJWWorkflowIterationState
from .fjw_workflow_optimizer import FJWMMAState
from .fjw_workflow_templates import render_cload_entries


@dataclass(frozen=True, slots=True)
class FJWResumeState:
    iteration_index: int
    design: np.ndarray
    mma_state: FJWMMAState
    checkpoint_directory: Path


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _embed_design_in_cage_grid(result: FJWWorkflowDriverResult, design: np.ndarray) -> np.ndarray:
    cage_3d = result.workflow_state.initial_state.cage_3d.copy()
    anchors = result.workflow_state.mesh.design_anchor_indices
    cage_3d[anchors[:, 0], anchors[:, 1], anchors[:, 2]] = design
    return cage_3d


def write_initial_checkpoint(
    *,
    run_directory: Path,
    workflow_state,
    design: np.ndarray,
    mma_state: FJWMMAState,
) -> Path:
    checkpoint_dir = Path(run_directory) / "iter_000"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    design = np.asarray(design, dtype=np.float64).reshape(-1)
    np.savez_compressed(checkpoint_dir / "design_cage.npz", design_cage=design)
    np.savez_compressed(
        checkpoint_dir / "mma_state.npz",
        iteration=np.array(mma_state.iteration, dtype=np.int64),
        xold1=mma_state.xold1,
        xold2=mma_state.xold2,
        xmin=mma_state.xmin,
        xmax=mma_state.xmax,
        low=mma_state.low,
        up=mma_state.up,
        a0=np.array(mma_state.a0, dtype=np.float64),
        a=mma_state.a,
        c=mma_state.c,
        d=mma_state.d,
    )
    cage_3d = workflow_state.initial_state.cage_3d.copy()
    anchors = workflow_state.mesh.design_anchor_indices
    cage_3d[anchors[:, 0], anchors[:, 1], anchors[:, 2]] = design
    np.savez_compressed(checkpoint_dir / "cage_3d.npz", cage_3d=cage_3d)
    _write_json(
        checkpoint_dir / "iteration_state.json",
        {
            "iteration_index": 0,
            "checkpoint_kind": "initial",
            "design_size": int(design.size),
            "design_sum": float(np.sum(design, dtype=np.float64)),
            "mma_iteration": int(mma_state.iteration),
        },
    )
    return checkpoint_dir


def write_iteration_checkpoint(
    *,
    run_directory: Path,
    result: FJWWorkflowDriverResult,
    delta: float,
) -> Path:
    iteration_state = result.iteration_state
    checkpoint_dir = Path(run_directory) / f"iter_{iteration_state.iteration_index:03d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    current_design = iteration_state.design
    next_design = iteration_state.next_design
    np.savez_compressed(
        checkpoint_dir / "design_cage.npz",
        design_cage=current_design,
        next_design=next_design,
    )
    if next_design is not None:
        np.savez_compressed(checkpoint_dir / "cage_3d.npz", cage_3d=_embed_design_in_cage_grid(result, next_design))
    _write_mma_state(checkpoint_dir / "mma_state.npz", iteration_state.mma_state)
    _write_aggregate_terms(checkpoint_dir / "aggregate_terms.npz", iteration_state)

    case_payloads = []
    for case_result in result.single_case_results:
        case_dir = checkpoint_dir / case_result.load_case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            case_dir / "case_history.npz",
            obj_bo_history=case_result.obj_bo_history,
            bo_sum_history=case_result.bo_sum_history,
            fai_history=case_result.fai_history,
            initial_design_sensitivity=case_result.initial_design_sensitivity,
        )
        for forward_step in case_result.forward_steps:
            step_dir = case_dir / f"forward_t{forward_step.time_index}"
            step_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                step_dir / "forward_step.npz",
                full_element_displacements=forward_step.full_element_displacements,
                obj_bo_previous=forward_step.obj_bo_previous,
                obj_bo_next=forward_step.obj_bo_next,
                bone_s=forward_step.bone_s,
                bone_density_delta=forward_step.bone_density_delta,
                design_sensitivity=forward_step.design_sensitivity,
            )
        for adjoint_step in case_result.adjoint_steps:
            step_dir = case_dir / f"adjoint_t{adjoint_step.time_index}"
            step_dir.mkdir(parents=True, exist_ok=True)
            load_vector = adjoint_step.solve_request.load_vector
            np.savez_compressed(
                step_dir / "fv.npz",
                nodal_forces_flat=load_vector.nodal_forces_flat,
                active_node_ids=load_vector.active_node_ids,
                active_forces_xyz=load_vector.active_forces_xyz,
            )
            np.savez_compressed(
                step_dir / "fai.npz",
                fai_next=adjoint_step.state.fai_next,
                fai_current=adjoint_step.state.fai_current,
                adjoint_element_displacements=adjoint_step.solve_result.element_displacements,
            )
            cload_text = render_cload_entries(load_vector.active_node_ids, load_vector.nodal_forces_flat)
            (step_dir / "fv_cload.inp").write_text(cload_text, encoding="utf-8")
            _write_json(
                step_dir / "fv_manifest.json",
                {
                    "load_case": case_result.load_case_name,
                    "time_index": int(adjoint_step.time_index),
                    "active_node_count": int(load_vector.active_node_ids.size),
                    "dense_vector_checksum": float(np.sum(load_vector.nodal_forces_flat, dtype=np.float64)),
                    "dense_l2_norm": float(np.linalg.norm(load_vector.nodal_forces_flat)),
                    "cload_path": str(step_dir / "fv_cload.inp"),
                },
            )
        case_payloads.append(
            {
                "load_case_name": case_result.load_case_name,
                "terminal_bo_sum": case_result.terminal_bo_sum,
                "forward_step_count": len(case_result.forward_steps),
                "adjoint_step_count": len(case_result.adjoint_steps),
            }
        )

    optimizer_diagnostics = (
        {} if iteration_state.optimizer_step is None else iteration_state.optimizer_step.diagnostics
    )
    _write_json(
        checkpoint_dir / "iteration_state.json",
        {
            "iteration_index": int(iteration_state.iteration_index),
            "delta": float(delta),
            "design_size": int(current_design.size),
            "design_sum": float(np.sum(current_design, dtype=np.float64)),
            "next_design_sum": None if next_design is None else float(np.sum(next_design, dtype=np.float64)),
            "has_placeholder_adjoint": iteration_state.has_placeholder_adjoint,
            "optimizer": iteration_state.metadata.get("optimizer"),
            "optimizer_diagnostics": optimizer_diagnostics,
            "objective": None if iteration_state.aggregate_terms is None else float(iteration_state.aggregate_terms.objective),
            "g2": None if iteration_state.aggregate_terms is None else float(iteration_state.aggregate_terms.g2),
            "cases": case_payloads,
        },
    )
    return checkpoint_dir


def _write_mma_state(path: Path, mma_state: FJWMMAState) -> None:
    np.savez_compressed(
        path,
        iteration=np.array(mma_state.iteration, dtype=np.int64),
        xold1=mma_state.xold1,
        xold2=mma_state.xold2,
        xmin=mma_state.xmin,
        xmax=mma_state.xmax,
        low=mma_state.low,
        up=mma_state.up,
        a0=np.array(mma_state.a0, dtype=np.float64),
        a=mma_state.a,
        c=mma_state.c,
        d=mma_state.d,
    )


def _write_aggregate_terms(path: Path, iteration_state: FJWWorkflowIterationState) -> None:
    if iteration_state.aggregate_terms is None:
        return
    aggregate = iteration_state.aggregate_terms
    np.savez_compressed(
        path,
        objective=np.array(aggregate.objective, dtype=np.float64),
        d_ob=aggregate.d_ob,
        g2=np.array(aggregate.g2, dtype=np.float64),
        d_g2=aggregate.d_g2,
    )


def write_workflow_manifest(
    *,
    run_directory: Path,
    payload: dict[str, object],
) -> Path:
    path = Path(run_directory) / "workflow_manifest.json"
    _write_json(path, payload)
    return path


def find_iteration_checkpoints(run_directory: Path) -> tuple[Path, ...]:
    root = Path(run_directory)
    if not root.exists():
        return ()
    candidates: list[Path] = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith("iter_") and (path / "mma_state.npz").exists():
            candidates.append(path)
    return tuple(sorted(candidates, key=lambda item: item.name))


def load_resume_state(run_directory: Path) -> FJWResumeState:
    checkpoints = find_iteration_checkpoints(run_directory)
    if not checkpoints:
        raise FileNotFoundError(f"No FJW optimization checkpoints found under {run_directory}.")
    checkpoint_dir = checkpoints[-1]
    design_payload = np.load(checkpoint_dir / "design_cage.npz")
    if "next_design" in design_payload:
        design = np.asarray(design_payload["next_design"], dtype=np.float64).reshape(-1)
    else:
        design = np.asarray(design_payload["design_cage"], dtype=np.float64).reshape(-1)
    mma_state = load_mma_state(checkpoint_dir / "mma_state.npz")
    metadata = json.loads((checkpoint_dir / "iteration_state.json").read_text(encoding="utf-8"))
    return FJWResumeState(
        iteration_index=int(metadata["iteration_index"]),
        design=design,
        mma_state=mma_state,
        checkpoint_directory=checkpoint_dir,
    )


def load_mma_state(path: Path) -> FJWMMAState:
    payload = np.load(path)
    return FJWMMAState(
        iteration=int(np.asarray(payload["iteration"]).item()),
        xold1=np.asarray(payload["xold1"], dtype=np.float64),
        xold2=np.asarray(payload["xold2"], dtype=np.float64),
        xmin=np.asarray(payload["xmin"], dtype=np.float64),
        xmax=np.asarray(payload["xmax"], dtype=np.float64),
        low=np.asarray(payload["low"], dtype=np.float64),
        up=np.asarray(payload["up"], dtype=np.float64),
        a0=float(np.asarray(payload["a0"]).item()),
        a=np.asarray(payload["a"], dtype=np.float64),
        c=np.asarray(payload["c"], dtype=np.float64),
        d=np.asarray(payload["d"], dtype=np.float64),
    )


def completed_iteration_numbers(run_directory: Path) -> tuple[int, ...]:
    numbers: list[int] = []
    for checkpoint in find_iteration_checkpoints(run_directory):
        try:
            numbers.append(int(checkpoint.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return tuple(sorted(numbers))


def checkpoint_paths(run_directory: Path) -> Iterable[Path]:
    return find_iteration_checkpoints(run_directory)


__all__ = [
    "FJWResumeState",
    "checkpoint_paths",
    "completed_iteration_numbers",
    "find_iteration_checkpoints",
    "load_mma_state",
    "load_resume_state",
    "write_initial_checkpoint",
    "write_iteration_checkpoint",
    "write_workflow_manifest",
]
