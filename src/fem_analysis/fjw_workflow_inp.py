from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fjw_workflow_artifacts import build_job_artifacts
from .fjw_workflow_models import FJWLoadCase, FJWWorkflowState
from .fjw_workflow_templates import (
    read_text,
    render_cload_entries,
    render_element_block,
    render_elset_block,
    render_node_block,
    render_nset_generate,
)


@dataclass(frozen=True, slots=True)
class GeneratedInputFiles:
    ini_nodesi: Path
    ini_desicage: Path
    ini_noend: Path
    forward_inputs: dict[str, Path]
    adjoint_inputs: dict[str, Path]


def _node_coordinates_mm(workflow_state: FJWWorkflowState) -> np.ndarray:
    return (workflow_state.mesh.node_coordinates.astype(np.float64) - 1.0) * 0.6


def _bucket_elements(
    element_ids: np.ndarray,
    bucket_indices: np.ndarray,
    *,
    prefix: str,
    bucket_max_index: int,
) -> list[tuple[str, np.ndarray]]:
    buckets = {idx: [] for idx in range(bucket_max_index + 1)}
    for element_id, bucket_index in zip(element_ids.tolist(), bucket_indices.tolist(), strict=True):
        buckets[int(bucket_index)].append(int(element_id))
    return [
        (f"{prefix}{index}", np.asarray(buckets[index], dtype=np.int32))
        for index in range(bucket_max_index + 1)
    ]


def _design_density_bucket_indices(workflow_state: FJWWorkflowState) -> np.ndarray:
    return np.rint(
        workflow_state.initial_state.design_cage.clip(0.0, 1.0) * 10.0
    ).astype(np.int32)


def _obj_bo_bucket_indices(workflow_state: FJWWorkflowState) -> np.ndarray:
    constants = workflow_state.material_constants
    return np.rint(
        workflow_state.initial_state.obj_bo.clip(0.0, constants.bone_density_upper_bound)
        / constants.bone_density_upper_bound
        * constants.bone_bucket_max_index
    ).astype(np.int32)


def _render_nodesi_body(workflow_state: FJWWorkflowState) -> str:
    mesh = workflow_state.mesh
    objective_elements = np.asarray(mesh.objective_elements, dtype=np.int32).reshape(-1)
    cor_elements = np.setdiff1d(mesh.cor_elements, objective_elements, assume_unique=False)
    tra_elements = np.setdiff1d(mesh.tra_elements, objective_elements, assume_unique=False)
    return (
        render_node_block(_node_coordinates_mm(workflow_state))
        + render_element_block(mesh.element_nodes)
        + render_nset_generate("alln", 1, int(mesh.node_coordinates.shape[0]))
        + f"*Elset, elset=alle, generate\n1,{int(mesh.element_nodes.shape[0])},1\n"
        + render_elset_block("NODESI_ELE_COR", cor_elements)
        + render_elset_block("NODESI_ELE_TRA", tra_elements)
    )


def render_ini_nodesi(workflow_state: FJWWorkflowState) -> str:
    base_text = read_text(workflow_state.reference_dir / "ini.inp").rstrip()
    return base_text + "\n\n" + _render_nodesi_body(workflow_state)


def render_ini_desicage(workflow_state: FJWWorkflowState) -> str:
    parts = [render_ini_nodesi(workflow_state).rstrip(), "", render_elset_block("desi_ele", workflow_state.mesh.design_elements)]
    for name, values in _bucket_elements(
        workflow_state.mesh.design_elements,
        _design_density_bucket_indices(workflow_state),
        prefix="desi_ele",
        bucket_max_index=10,
    ):
        parts.append(render_elset_block(name, values))
    return "\n".join(parts).rstrip() + "\n"


def render_ini_noend(workflow_state: FJWWorkflowState) -> str:
    parts = [render_ini_desicage(workflow_state).rstrip(), ""]

    for name, values in _bucket_elements(
        workflow_state.mesh.objective_elements,
        _obj_bo_bucket_indices(workflow_state),
        prefix="obj_bo_ele",
        bucket_max_index=workflow_state.material_constants.bone_bucket_max_index,
    ):
        parts.append(render_elset_block(name, values))

    for name, values in _bucket_elements(
        workflow_state.mesh.objective_elements,
        workflow_state.initial_state.modulus_buckets.obj_bucket_indices,
        prefix="obj_e_ele",
        bucket_max_index=workflow_state.material_constants.bone_bucket_max_index,
    ):
        parts.append(render_elset_block(name, values))

    for name, values in _bucket_elements(
        workflow_state.mesh.design_elements,
        workflow_state.initial_state.modulus_buckets.cage_bucket_indices,
        prefix="desi_e_ele",
        bucket_max_index=workflow_state.material_constants.cage_bucket_max_index,
    ):
        parts.append(render_elset_block(name, values))

    return "\n".join(parts).rstrip() + "\n"


def _find_load_case(workflow_state: FJWWorkflowState, load_case_name: str) -> FJWLoadCase:
    for load_case in workflow_state.load_cases:
        if load_case.name == load_case_name:
            return load_case
    raise KeyError(f"Unknown FJW load case: {load_case_name}")


def render_forward_input(
    workflow_state: FJWWorkflowState,
    *,
    load_case_name: str,
) -> str:
    load_case = _find_load_case(workflow_state, load_case_name)
    return render_ini_noend(workflow_state).rstrip() + "\n\n" + "\n".join(load_case.template_lines).strip() + "\n"


def render_adjoint_input(
    workflow_state: FJWWorkflowState,
    *,
    fv_vector: np.ndarray,
) -> str:
    prefix = read_text(workflow_state.reference_dir / "end_Fv_p1.inp").strip()
    suffix = read_text(workflow_state.reference_dir / "end_Fv_p2.inp").strip()
    node_ids = np.nonzero(np.any(fv_vector.reshape(-1, 3) != 0.0, axis=1))[0].astype(np.int32) + 1
    return (
        render_ini_desicage(workflow_state).rstrip()
        + "\n\n"
        + prefix
        + "\n"
        + render_cload_entries(node_ids, fv_vector)
        + "\n"
        + suffix
        + "\n"
    )


def generate_workflow_input_files(
    workflow_state: FJWWorkflowState,
    *,
    run_directory: Path,
    mode: str,
    time_steps: int | None = None,
) -> GeneratedInputFiles:
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    time_steps = (
        workflow_state.material_constants.num_time_steps
        if time_steps is None
        else int(time_steps)
    )

    ini_nodesi_path = run_directory / "ini_nodesi.inp"
    ini_desicage_path = run_directory / "ini_desicage.inp"
    ini_noend_path = run_directory / "ini_noend.inp"
    ini_nodesi_path.write_text(render_ini_nodesi(workflow_state), encoding="utf-8")
    ini_desicage_path.write_text(render_ini_desicage(workflow_state), encoding="utf-8")
    ini_noend_path.write_text(render_ini_noend(workflow_state), encoding="utf-8")

    forward_inputs: dict[str, Path] = {}
    load_case_names = ["force_1"] if mode == "single-force" else ["force_1", "force_2", "force_3"]
    for load_case_name in load_case_names:
        job_name = f"vert_{load_case_name}"
        output_path = build_job_artifacts(run_directory, job_name).inp_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            render_forward_input(workflow_state, load_case_name=load_case_name),
            encoding="utf-8",
        )
        forward_inputs[load_case_name] = output_path

    adjoint_inputs: dict[str, Path] = {}
    zero_fv = np.zeros(workflow_state.mesh.node_coordinates.shape[0] * 3, dtype=np.float64)
    for load_case_name in load_case_names:
        for time_index in range(time_steps - 1, -1, -1):
            job_name = f"adjoint_{load_case_name}_t{time_index}"
            output_path = build_job_artifacts(run_directory, job_name).inp_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                render_adjoint_input(workflow_state, fv_vector=zero_fv),
                encoding="utf-8",
            )
            adjoint_inputs[f"{load_case_name}:t{time_index}"] = output_path

    (run_directory / "generated_inputs_manifest.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "time_steps": time_steps,
                "adjoint_inputs_are_static_templates": True,
                "adjoint_template_note": (
                    "Static adjoint inputs contain zero Fv and are only suitable for "
                    "template/diff checks. Runtime optimization must render adjoint "
                    "jobs from the dynamic Fv generated after each forward solve."
                ),
                "forward_inputs": {key: str(value) for key, value in forward_inputs.items()},
                "adjoint_inputs": {key: str(value) for key, value in adjoint_inputs.items()},
                "initial_design_total": workflow_state.initial_state.initial_design_total,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return GeneratedInputFiles(
        ini_nodesi=ini_nodesi_path,
        ini_desicage=ini_desicage_path,
        ini_noend=ini_noend_path,
        forward_inputs=forward_inputs,
        adjoint_inputs=adjoint_inputs,
    )


__all__ = [
    "GeneratedInputFiles",
    "generate_workflow_input_files",
    "render_adjoint_input",
    "render_forward_input",
    "render_ini_desicage",
    "render_ini_nodesi",
    "render_ini_noend",
]
