from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.io import whosmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "references" / "fjw_work"
STRUCTURED_INPUTS_PATH = PROJECT_ROOT / "datasets" / "fjw_abaqus_inputs.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "fjw_input_inventory.json"


def _mat_metadata(path: Path) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for name, shape, matlab_type in whosmat(path):
        payload.append(
            {
                "variable": name,
                "shape": list(shape),
                "matlab_type": matlab_type,
            }
        )
    return payload


def _parse_nset_members(path: Path, nset_name: str) -> list[int]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    target = f"nset={nset_name.lower()}"
    members: list[int] = []
    capture = False
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("*nset") and target in lower:
            capture = True
            continue
        if capture and stripped.startswith("*"):
            break
        if not capture:
            continue
        members.extend(int(float(token.strip())) for token in stripped.split(",") if token.strip())
    return members


def _build_payload() -> dict[str, object]:
    structured_inputs = json.loads(STRUCTURED_INPUTS_PATH.read_text(encoding="utf-8"))

    mat_files = {
        "nod_coo.mat": "Global node coordinates.",
        "ele_nod.mat": "Hexahedral element-to-node connectivity.",
        "B_3d.mat": "Element strain-displacement matrix kernel.",
        "D_3d.mat": "Constitutive matrix kernel.",
        "cage_ele.mat": "All cage-domain element ids.",
        "desi_ele.mat": "Optimizable cage element ids.",
        "obj_ele.mat": "Bone objective-region element ids.",
        "cor_ele.mat": "Background cortical-bone element ids.",
        "tra_ele.mat": "Background trabecular-bone element ids.",
    }

    mesh_and_sets: list[dict[str, object]] = []
    for file_name, description in mat_files.items():
        path = REFERENCE_DIR / file_name
        mesh_and_sets.append(
            {
                "file": path.relative_to(PROJECT_ROOT).as_posix(),
                "description": description,
                "variables": _mat_metadata(path),
            }
        )

    top_nod = _parse_nset_members(REFERENCE_DIR / "end1.inp", "top_nod")
    bot_nod = _parse_nset_members(REFERENCE_DIR / "end1.inp", "bot_nod")

    return {
        "metadata": {
            "source": "Inventory of all currently identified FJW inputs needed by the MATLAB + Abaqus workflow.",
            "reference_dir": REFERENCE_DIR.relative_to(PROJECT_ROOT).as_posix(),
            "built_from": [
                "datasets/fjw_abaqus_inputs.json",
                "references/fjw_work/time_dep.m",
                "references/fjw_work/Three_Force_Time_Dep.m",
                "references/fjw_work/end1.inp",
                "references/fjw_work/end2.inp",
                "references/fjw_work/end3.inp",
                "references/fjw_work/end_Fv_p1.inp",
                "references/fjw_work/end_Fv_p2.inp",
            ],
        },
        "status_summary": {
            "static_inputs_ready": True,
            "static_inputs_comment": "The workflow's fixed external inputs are present in the repository.",
            "runtime_state_still_generated_during_solve": True,
            "runtime_state_comment": "State vectors such as obj_bo(t), E_obj(t), E_cage(t), U(t), V(t), and Fv(t) are not external files to prepare in advance.",
        },
        "static_external_inputs": {
            "mesh_and_region_files": mesh_and_sets,
            "node_sets_from_templates": [
                {
                    "name": "TOP_NOD",
                    "source_file": "references/fjw_work/end1.inp",
                    "count": len(top_nod),
                    "first_node_ids": top_nod[:5],
                    "last_node_ids": top_nod[-5:],
                    "used_for": "Top-end coupled loading surface.",
                },
                {
                    "name": "BOT_NOD",
                    "source_file": "references/fjw_work/end1.inp",
                    "count": len(bot_nod),
                    "first_node_ids": bot_nod[:5],
                    "last_node_ids": bot_nod[-5:],
                    "used_for": "Bottom-end fixed support surface.",
                },
            ],
            "boundary_and_load_inputs": {
                "assembly_controls": structured_inputs["assembly_controls"],
                "boundary_conditions": structured_inputs["boundary_conditions"],
                "forward_load_cases": structured_inputs["forward_load_cases"],
                "adjoint_load_template": structured_inputs["adjoint_load_template"],
            },
            "material_and_discretization_inputs": {
                "materials": structured_inputs["materials"]["families"],
                "section_assignments_count": len(structured_inputs["section_assignments"]),
                "section_assignments_source": "datasets/fjw_abaqus_inputs.json#section_assignments",
                "discretization_rules": structured_inputs["discretization_rules"],
            },
            "global_constants": {
                "grid_shape": [152, 131, 134],
                "simulation_constants": structured_inputs["simulation_constants"],
            },
            "input_templates": [
                {
                    "file": "references/fjw_work/ini.inp",
                    "role": "Base Abaqus part template before node/element and material grouping edits.",
                },
                {
                    "file": "references/fjw_work/end1.inp",
                    "role": "Forward load case 1 template tail.",
                },
                {
                    "file": "references/fjw_work/end2.inp",
                    "role": "Forward load case 2 template tail.",
                },
                {
                    "file": "references/fjw_work/end3.inp",
                    "role": "Forward load case 3 template tail.",
                },
                {
                    "file": "references/fjw_work/end_Fv_p1.inp",
                    "role": "Adjoint load template prefix.",
                },
                {
                    "file": "references/fjw_work/end_Fv_p2.inp",
                    "role": "Adjoint load template suffix.",
                },
            ],
        },
        "derived_initial_state_inputs": [
            {
                "name": "design_cage(0)",
                "how_built": "Sample ini_cage on the first node of each desi_ele element.",
                "source_files": ["references/fjw_work/time_dep.m", "references/fjw_work/Three_Force_Time_Dep.m"],
                "value": {
                    "single_load_default": 0.2,
                    "three_load_default": 0.3,
                },
            },
            {
                "name": "obj_bo(0)",
                "how_built": "Sample ini_str on the first node of each obj_ele element.",
                "source_files": ["references/fjw_work/time_dep.m", "references/fjw_work/Calculate_Force_1.m"],
                "value": 0.36,
            },
        ],
        "runtime_state_variables": [
            {
                "name": "E_obj(t)",
                "role": "Bone-region modulus per time step.",
                "built_from": "obj_bo(t), b_max, E0_bo, Emin_bo",
            },
            {
                "name": "E_cage(t)",
                "role": "Cage design-domain modulus per time step.",
                "built_from": "design_cage, E0_cage, Emin_cage",
            },
            {
                "name": "U(t)",
                "role": "Forward displacement field from Abaqus or a replacement solver.",
                "built_from": "Mesh, BCs, loads, material assignment at time t",
            },
            {
                "name": "bone_s(t)",
                "role": "Mechanical stimulus on objective bone elements.",
                "built_from": "U(t), B, D, E_obj(t), obj_bo(t)",
            },
            {
                "name": "obj_bo(t+1)",
                "role": "Updated bone density after one remodeling step.",
                "built_from": "obj_bo(t), bone_delta(bone_s(t)), dt",
            },
            {
                "name": "Fv(t)",
                "role": "Adjoint right-hand-side nodal load vector.",
                "built_from": "d_bone_delta, Fai(t+1), U(t), obj_bo(t), B, D",
            },
            {
                "name": "V(t)",
                "role": "Adjoint displacement field.",
                "built_from": "Mesh, BCs, adjoint load Fv(t), material assignment at time t",
            },
        ],
        "what_is_missing_before_python_reimplementation": [
            {
                "item": "Direct Python-side loader for MAT arrays into a unified domain object.",
                "severity": "implementation_gap",
            },
            {
                "item": "Direct Python-side reconstruction of the full Abaqus model from mesh arrays and set arrays.",
                "severity": "implementation_gap",
            },
            {
                "item": "Reference result comparison between Abaqus outputs and the future Python solver outputs.",
                "severity": "validation_gap",
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a complete inventory of FJW workflow inputs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = _build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
