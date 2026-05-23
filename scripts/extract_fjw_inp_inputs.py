from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "references" / "fjw_work"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "fjw_abaqus_inputs.json"


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _parse_keyword_header(line: str) -> tuple[str, dict[str, str], list[str]]:
    assert line.startswith("*"), line
    parts = [part.strip() for part in line[1:].split(",")]
    keyword = parts[0].lower()
    params: dict[str, str] = {}
    flags: list[str] = []
    for part in parts[1:]:
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            params[key.strip().lower()] = value.strip()
        else:
            flags.append(part.lower())
    return keyword, params, flags


def _parse_data_tokens(line: str) -> list[str]:
    return [token.strip() for token in line.split(",") if token.strip()]


def _parse_number(token: str) -> float:
    return float(token.strip())


def _collect_data_lines(lines: list[str], start: int) -> tuple[list[str], int]:
    data: list[str] = []
    index = start
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("**"):
            index += 1
            continue
        if stripped.startswith("*"):
            break
        data.append(stripped)
        index += 1
    return data, index


def _parse_materials(lines: list[str]) -> list[dict[str, object]]:
    materials: list[dict[str, object]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("**"):
            index += 1
            continue
        if not stripped.lower().startswith("*material"):
            index += 1
            continue
        _, params, _ = _parse_keyword_header(stripped)
        name = params["name"]
        material: dict[str, object] = {"name": name}
        index += 1
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped or stripped.startswith("**"):
                index += 1
                continue
            if stripped.lower().startswith("*material"):
                break
            if not stripped.startswith("*"):
                index += 1
                continue
            keyword, _, _ = _parse_keyword_header(stripped)
            data_lines, next_index = _collect_data_lines(lines, index + 1)
            if keyword == "density" and data_lines:
                material["density"] = _parse_number(_parse_data_tokens(data_lines[0])[0])
            elif keyword == "elastic" and data_lines:
                tokens = _parse_data_tokens(data_lines[0])
                material["youngs_modulus"] = _parse_number(tokens[0])
                if len(tokens) > 1:
                    material["poisson_ratio"] = _parse_number(tokens[1])
            index = next_index
        materials.append(material)
    return materials


def _parse_sections(lines: list[str]) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.lower().startswith("*solid section"):
            continue
        _, params, _ = _parse_keyword_header(stripped)
        sections.append(
            {
                "elset": params["elset"].upper(),
                "material": params["material"].upper(),
            }
        )
    return sections


def _parse_assembly_controls(lines: list[str]) -> dict[str, object]:
    in_assembly = False
    helper_nsets: list[dict[str, object]] = []
    surfaces: list[dict[str, str]] = []
    couplings: list[dict[str, object]] = []
    reference_nodes: list[dict[str, object]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        lower = stripped.lower()
        if lower.startswith("*assembly"):
            in_assembly = True
            index += 1
            continue
        if in_assembly and lower.startswith("*end assembly"):
            break
        if not in_assembly or not stripped or stripped.startswith("**"):
            index += 1
            continue
        if lower == "*node":
            data_lines, next_index = _collect_data_lines(lines, index + 1)
            for raw in data_lines:
                tokens = _parse_data_tokens(raw)
                if len(tokens) < 4:
                    continue
                reference_nodes.append(
                    {
                        "node_id": int(float(tokens[0])),
                        "coordinates": [float(tokens[1]), float(tokens[2]), float(tokens[3])],
                    }
                )
            index = next_index
            continue
        if lower.startswith("*nset"):
            _, params, _ = _parse_keyword_header(stripped)
            data_lines, next_index = _collect_data_lines(lines, index + 1)
            members: list[int] = []
            for raw in data_lines:
                members.extend(int(float(token)) for token in _parse_data_tokens(raw))
            helper_nsets.append({"name": params["nset"].upper(), "members": members})
            index = next_index
            continue
        if lower.startswith("*surface"):
            _, params, _ = _parse_keyword_header(stripped)
            data_lines, next_index = _collect_data_lines(lines, index + 1)
            source = data_lines[0] if data_lines else ""
            surfaces.append(
                {
                    "name": params["name"].upper(),
                    "type": params.get("type", "").upper(),
                    "source": source.upper(),
                }
            )
            index = next_index
            continue
        if lower.startswith("*coupling"):
            _, params, _ = _parse_keyword_header(stripped)
            _, next_index = _collect_data_lines(lines, index + 1)
            couplings.append(
                {
                    "name": params.get("constraint name", "").upper(),
                    "reference_node_set": params.get("ref node", "").upper(),
                    "surface": params.get("surface", "").upper(),
                    "type": "KINEMATIC",
                }
            )
            index = next_index
            continue
        index += 1
    return {
        "reference_nodes": reference_nodes,
        "helper_node_sets": helper_nsets,
        "surfaces": surfaces,
        "couplings": couplings,
    }


def _parse_boundary(lines: list[str]) -> list[dict[str, object]]:
    boundary_conditions: list[dict[str, object]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped.lower().startswith("*boundary"):
            index += 1
            continue
        data_lines, next_index = _collect_data_lines(lines, index + 1)
        for raw in data_lines:
            tokens = _parse_data_tokens(raw)
            if len(tokens) == 2:
                boundary_conditions.append(
                    {
                        "target": tokens[0].upper(),
                        "constraint": tokens[1].upper(),
                    }
                )
            elif len(tokens) >= 3:
                entry: dict[str, object] = {
                    "target": tokens[0].upper(),
                    "dof_start": int(float(tokens[1])),
                    "dof_end": int(float(tokens[2])),
                }
                if len(tokens) > 3:
                    entry["value"] = float(tokens[3])
                boundary_conditions.append(entry)
        index = next_index
    return boundary_conditions


def _parse_static_step(lines: list[str]) -> dict[str, object]:
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped.lower().startswith("*step"):
            index += 1
            continue
        _, step_params, step_flags = _parse_keyword_header(stripped)
        static_params: list[float] = []
        inner = index + 1
        while inner < len(lines):
            current = lines[inner].strip()
            if not current or current.startswith("**"):
                inner += 1
                continue
            if current.lower().startswith("*static"):
                data_lines, _ = _collect_data_lines(lines, inner + 1)
                if data_lines:
                    static_params = [_parse_number(token) for token in _parse_data_tokens(data_lines[0])]
                break
            inner += 1
        return {
            "name": step_params.get("name", ""),
            "nlgeom": step_params.get("nlgeom", "").upper(),
            "solver": step_params.get("solver", "").upper(),
            "flags": [flag.upper() for flag in step_flags],
            "static_parameters": static_params,
        }
    raise ValueError("No *Step block found.")


def _parse_cloads(lines: list[str]) -> list[dict[str, object]]:
    loads: list[dict[str, object]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped.lower().startswith("*cload"):
            index += 1
            continue
        _, params, _ = _parse_keyword_header(stripped)
        data_lines, next_index = _collect_data_lines(lines, index + 1)
        for raw in data_lines:
            tokens = _parse_data_tokens(raw)
            if len(tokens) < 3:
                continue
            loads.append(
                {
                    "target": tokens[0].upper(),
                    "dof": int(float(tokens[1])),
                    "magnitude": float(tokens[2]),
                    "op": params.get("op", "").upper(),
                }
            )
        index = next_index
    return loads


def _build_material_families(materials: Iterable[dict[str, object]]) -> dict[str, object]:
    by_name = {str(material["name"]).upper(): material for material in materials}
    cage_materials = [by_name[f"CAGE_{index}"] for index in range(101)]
    bone_materials = [by_name[f"BONE_{index}"] for index in range(11)]
    return {
        "cage_design_domain": {
            "material_prefix": "CAGE_",
            "section_elset_prefix": "DESI_E_ELE",
            "bucket_index_range": [0, 100],
            "bucket_index_expression": "round(E_cage / E0_cage * 100)",
            "constitutive_expression": "E_cage = min(E0_cage, Emin_cage + E0_cage * design_cage^3)",
            "discretization_note": "Bucket 0 maps to a nonzero floor modulus in the Abaqus template.",
            "materials": cage_materials,
        },
        "bone_objective_domain": {
            "material_prefix": "BONE_",
            "section_elset_prefix": "OBJ_E_ELE",
            "bucket_index_range": [0, 10],
            "bucket_index_expression": "round(E_obj / E0_bo * 10)",
            "constitutive_expression": "E_obj = min(E0_bo, Emin_bo + E0_bo * (obj_bo / b_max)^3)",
            "discretization_note": "Bucket 0 maps to a nonzero floor modulus in the Abaqus template.",
            "materials": bone_materials,
        },
        "bone_background_domains": [
            {
                "elset": "NODESI_ELE_COR",
                "material": "BONE_COR",
                "youngs_modulus": by_name["BONE_COR"]["youngs_modulus"],
                "poisson_ratio": by_name["BONE_COR"]["poisson_ratio"],
            },
            {
                "elset": "NODESI_ELE_TRA",
                "material": "BONE_1",
                "youngs_modulus": by_name["BONE_1"]["youngs_modulus"],
                "poisson_ratio": by_name["BONE_1"]["poisson_ratio"],
            },
        ],
    }


def _build_payload() -> dict[str, object]:
    template_1 = REFERENCE_DIR / "end1.inp"
    template_2 = REFERENCE_DIR / "end2.inp"
    template_3 = REFERENCE_DIR / "end3.inp"
    adjoint_prefix = REFERENCE_DIR / "end_Fv_p1.inp"
    adjoint_suffix = REFERENCE_DIR / "end_Fv_p2.inp"

    lines_1 = _read_lines(template_1)
    lines_2 = _read_lines(template_2)
    lines_3 = _read_lines(template_3)
    adjoint_prefix_lines = _read_lines(adjoint_prefix)

    materials = _parse_materials(lines_1)
    sections = _parse_sections(lines_1)
    assembly_controls = _parse_assembly_controls(lines_1)
    boundary_conditions = _parse_boundary(lines_1)
    step = _parse_static_step(lines_1)

    forward_cases = {
        "force_1": {
            "template": template_1.relative_to(PROJECT_ROOT).as_posix(),
            "loads": _parse_cloads(lines_1),
        },
        "force_2": {
            "template": template_2.relative_to(PROJECT_ROOT).as_posix(),
            "loads": _parse_cloads(lines_2),
        },
        "force_3": {
            "template": template_3.relative_to(PROJECT_ROOT).as_posix(),
            "loads": _parse_cloads(lines_3),
        },
    }

    return {
        "metadata": {
            "source": "Structured extraction from FJW Abaqus templates and MATLAB driver assumptions.",
            "reference_dir": REFERENCE_DIR.relative_to(PROJECT_ROOT).as_posix(),
            "generated_from": [
                template_1.relative_to(PROJECT_ROOT).as_posix(),
                template_2.relative_to(PROJECT_ROOT).as_posix(),
                template_3.relative_to(PROJECT_ROOT).as_posix(),
                adjoint_prefix.relative_to(PROJECT_ROOT).as_posix(),
                adjoint_suffix.relative_to(PROJECT_ROOT).as_posix(),
                "references/fjw_work/time_dep.m",
                "references/fjw_work/Calculate_Force_1.m",
                "references/fjw_work/Calculate_Force_2.m",
                "references/fjw_work/Calculate_Force_3.m",
                "references/fjw_work/edit_objbo_inp.m",
                "references/fjw_work/editF_desiele_inp.m",
            ],
        },
        "simulation_constants": {
            "voxel_volume": 0.216,
            "time_step_dt": 1.0,
            "num_time_steps": 3,
            "bone_density_upper_bound_b_max": 1.86,
            "bone_modulus": {"E0_bo": 12000.0, "Emin_bo": 1.2},
            "cage_modulus": {"E0_cage": 110000.0, "Emin_cage": 11.0},
            "initial_conditions": {
                "single_load_design_cage": 0.2,
                "three_load_design_cage": 0.3,
                "initial_bone_density": 0.36,
            },
        },
        "assembly_controls": assembly_controls,
        "boundary_conditions": boundary_conditions,
        "forward_load_cases": forward_cases,
        "adjoint_load_template": {
            "prefix_template": adjoint_prefix.relative_to(PROJECT_ROOT).as_posix(),
            "suffix_template": adjoint_suffix.relative_to(PROJECT_ROOT).as_posix(),
            "boundary_conditions": _parse_boundary(adjoint_prefix_lines),
            "step": _parse_static_step(adjoint_prefix_lines),
            "injection_rule": {
                "description": "Insert one *Cload block per node in Fv_set between prefix and suffix templates.",
                "target_format": "VERT-1.<node_id>",
                "components": [1, 2, 3],
                "magnitude_source": "Fv[(node_id-1)*3 + component_index]",
            },
        },
        "materials": {
            "all": materials,
            "families": _build_material_families(materials),
        },
        "section_assignments": sections,
        "discretization_rules": {
            "cage_elset_generation": {
                "expression": "index = round(E_cage / E0_cage * 100)",
                "elset_name": "DESI_E_ELE{index}",
                "material_name": "CAGE_{index}",
            },
            "objective_bone_elset_generation": {
                "expression": "index = round(E_obj / E0_bo * 10)",
                "elset_name": "OBJ_E_ELE{index}",
                "material_name": "BONE_{index}",
            },
            "objective_bone_state_groups": {
                "expression": "index = round(obj_bo / b_max * 10)",
                "elset_name": "OBJ_BO_ELE{index}",
                "usage": "Bookkeeping groups emitted by edit_objbo_inp.m before the template tail is appended.",
            },
        },
        "known_gaps_for_python_solver": [
            "Top and bottom node memberships live in the mesh/template build stage, not in this extracted tail spec.",
            "The dynamic Fv vector is computed during the adjoint loop and cannot be listed statically here.",
            "Element memberships for DESI_E_ELE*, OBJ_E_ELE*, NODESI_ELE_* are generated from MAT files and current state variables, not stored as fixed arrays in this JSON.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured inputs from FJW Abaqus .inp templates.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = _build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
