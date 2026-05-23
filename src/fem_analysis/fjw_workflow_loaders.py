from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from .fjw_reference import (
    DEFAULT_OBJECTIVE_LAYER_COUNT,
    DEFAULT_OBJECTIVE_REGION,
    DEFAULT_REFERENCE_DIR,
    load_fjw_reference_model,
)
from .fjw_workflow_models import (
    FJWBoundaryCondition,
    FJWInitialState,
    FJWLoad,
    FJWLoadCase,
    FJWMaterialBucket,
    FJWMaterialConstants,
    FJWModulusBuckets,
    FJWReferenceMeshContext,
    FJWWorkflowState,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ABAQUS_INPUTS_PATH = PROJECT_ROOT / "datasets" / "fjw_abaqus_inputs.json"
DEFAULT_INPUT_INVENTORY_PATH = PROJECT_ROOT / "datasets" / "fjw_input_inventory.json"
DEFAULT_END1_TEMPLATE_PATH = DEFAULT_REFERENCE_DIR / "end1.inp"


def _resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_required_mat_array(path: Path, variable_name: str) -> np.ndarray:
    payload = loadmat(path)
    if variable_name not in payload:
        raise KeyError(f"Variable '{variable_name}' not found in {path}.")
    return np.asarray(payload[variable_name])


def _parse_nset_members(path: Path, nset_name: str) -> np.ndarray:
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
    return np.asarray(members, dtype=np.int32)


def _inventory_node_set_counts(input_inventory: Mapping[str, Any]) -> dict[str, int]:
    entries = input_inventory["static_external_inputs"]["node_sets_from_templates"]
    return {
        str(entry["name"]).upper(): int(entry["count"])
        for entry in entries
    }


def _infer_grid_shape(node_coordinates: np.ndarray) -> tuple[int, int, int]:
    node_coordinates = np.asarray(node_coordinates)
    node_min = node_coordinates.min(axis=0)
    if np.any(node_min < 1):
        raise ValueError(f"Unexpected FJW node minimum coordinate: {node_min.tolist()}.")
    node_max = node_coordinates.max(axis=0)
    return tuple(int(value - 1) for value in node_max.tolist())


def _build_element_anchor_indices(
    node_coordinates: np.ndarray,
    element_nodes: np.ndarray,
    grid_shape_xyz: tuple[int, int, int],
) -> np.ndarray:
    first_node_indices = np.asarray(element_nodes[:, 0], dtype=np.int64) - 1
    if np.any(first_node_indices < 0) or np.any(first_node_indices >= node_coordinates.shape[0]):
        raise ValueError("FJW element connectivity references an out-of-range first-node id.")

    anchors = np.asarray(node_coordinates[first_node_indices], dtype=np.int32) - 1
    if np.any(anchors < 0):
        raise ValueError("FJW element anchor coordinates must be non-negative after 1-based to 0-based conversion.")

    max_indices = np.asarray(grid_shape_xyz, dtype=np.int32)
    if np.any(anchors >= max_indices):
        raise ValueError("FJW element anchor coordinates exceed the inferred grid shape.")
    return anchors


def _build_boundary_condition(entry: Mapping[str, Any]) -> FJWBoundaryCondition:
    return FJWBoundaryCondition(
        target=str(entry["target"]),
        constraint=str(entry["constraint"]) if "constraint" in entry else None,
        dof_start=int(entry["dof_start"]) if "dof_start" in entry else None,
        dof_end=int(entry["dof_end"]) if "dof_end" in entry else None,
        value=float(entry["value"]) if "value" in entry else None,
    )


def _bucket_index_from_material_name(material_name: str) -> int:
    return int(material_name.rsplit("_", 1)[-1])


def _build_material_bucket_catalogs(
    structured_inputs: Mapping[str, Any],
) -> tuple[tuple[FJWMaterialBucket, ...], tuple[FJWMaterialBucket, ...], tuple[FJWMaterialBucket, ...]]:
    materials_payload = structured_inputs["materials"]
    all_materials = {
        str(material["name"]).upper(): material
        for material in materials_payload["all"]
    }
    families = materials_payload["families"]

    def build_family_catalog(family_key: str) -> tuple[FJWMaterialBucket, ...]:
        family = families[family_key]
        section_elset_prefix = str(family["section_elset_prefix"])
        buckets: list[FJWMaterialBucket] = []
        for material in family["materials"]:
            material_name = str(material["name"]).upper()
            index = _bucket_index_from_material_name(material_name)
            buckets.append(
                FJWMaterialBucket(
                    index=index,
                    material_name=material_name,
                    section_elset=f"{section_elset_prefix}{index}",
                    youngs_modulus=float(material["youngs_modulus"]),
                    poisson_ratio=float(material["poisson_ratio"]),
                    density=float(material.get("density", 0.0)),
                )
            )
        return tuple(sorted(buckets, key=lambda bucket: bucket.index))

    background_buckets: list[FJWMaterialBucket] = []
    for entry in families["bone_background_domains"]:
        material_name = str(entry["material"]).upper()
        material = all_materials.get(material_name, entry)
        background_buckets.append(
            FJWMaterialBucket(
                index=-1,
                material_name=material_name,
                section_elset=str(entry["elset"]).upper(),
                youngs_modulus=float(material["youngs_modulus"]),
                poisson_ratio=float(material["poisson_ratio"]),
                density=float(material.get("density", entry.get("density", 0.0))),
            )
        )

    return (
        build_family_catalog("cage_design_domain"),
        build_family_catalog("bone_objective_domain"),
        tuple(background_buckets),
    )


def _bucket_map(catalog: Iterable[FJWMaterialBucket] | None) -> dict[int, FJWMaterialBucket]:
    if catalog is None:
        return {}
    return {int(bucket.index): bucket for bucket in catalog}


def _material_arrays_from_indices(
    indices: np.ndarray,
    bucket_map: Mapping[int, FJWMaterialBucket],
    *,
    material_prefix: str,
    fallback_moduli: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    names = np.empty(indices.shape, dtype=object)
    moduli = np.empty(indices.shape, dtype=np.float64)

    for flat_index, bucket_index in enumerate(indices.tolist()):
        bucket = bucket_map.get(int(bucket_index))
        if bucket is None:
            names[flat_index] = f"{material_prefix}{int(bucket_index)}"
            moduli[flat_index] = float(fallback_moduli[flat_index])
            continue
        names[flat_index] = bucket.material_name
        moduli[flat_index] = float(bucket.youngs_modulus)

    return names, moduli


def load_fjw_material_constants(
    structured_inputs: Mapping[str, Any] | None = None,
    *,
    abaqus_inputs_path: Path = DEFAULT_ABAQUS_INPUTS_PATH,
) -> FJWMaterialConstants:
    if structured_inputs is None:
        structured_inputs = _load_json(abaqus_inputs_path)

    simulation_constants = structured_inputs["simulation_constants"]
    initial_conditions = simulation_constants["initial_conditions"]
    families = structured_inputs["materials"]["families"]
    cage_bucket_range = families["cage_design_domain"]["bucket_index_range"]
    bone_bucket_range = families["bone_objective_domain"]["bucket_index_range"]

    return FJWMaterialConstants(
        voxel_volume=float(simulation_constants["voxel_volume"]),
        time_step_dt=float(simulation_constants["time_step_dt"]),
        num_time_steps=int(simulation_constants["num_time_steps"]),
        bone_density_upper_bound=float(simulation_constants["bone_density_upper_bound_b_max"]),
        bone_modulus_0=float(simulation_constants["bone_modulus"]["E0_bo"]),
        bone_modulus_min=float(simulation_constants["bone_modulus"]["Emin_bo"]),
        cage_modulus_0=float(simulation_constants["cage_modulus"]["E0_cage"]),
        cage_modulus_min=float(simulation_constants["cage_modulus"]["Emin_cage"]),
        initial_bone_density=float(initial_conditions["initial_bone_density"]),
        single_load_initial_design_cage=float(initial_conditions["single_load_design_cage"]),
        three_load_initial_design_cage=float(initial_conditions["three_load_design_cage"]),
        cage_bucket_max_index=int(cage_bucket_range[1]),
        bone_bucket_max_index=int(bone_bucket_range[1]),
    )


def load_fjw_reference_mesh_context(
    *,
    reference_dir: Path = DEFAULT_REFERENCE_DIR,
    input_inventory_path: Path = DEFAULT_INPUT_INVENTORY_PATH,
    end1_template_path: Path = DEFAULT_END1_TEMPLATE_PATH,
    objective_region: str = DEFAULT_OBJECTIVE_REGION,
    objective_layer_count: int = DEFAULT_OBJECTIVE_LAYER_COUNT,
) -> FJWReferenceMeshContext:
    reference_dir = _resolve_repo_path(reference_dir)
    input_inventory_path = _resolve_repo_path(input_inventory_path)
    end1_template_path = _resolve_repo_path(end1_template_path)

    input_inventory = _load_json(input_inventory_path)
    reference_model = load_fjw_reference_model(
        reference_dir,
        objective_region=objective_region,
        objective_layer_count=objective_layer_count,
    )
    strain_displacement_matrix = np.asarray(
        _load_required_mat_array(reference_dir / "B_3d.mat", "B"),
        dtype=np.float64,
    )
    constitutive_matrix = np.asarray(
        _load_required_mat_array(reference_dir / "D_3d.mat", "D"),
        dtype=np.float64,
    )

    inferred_grid_shape = _infer_grid_shape(reference_model.node_coordinates)
    inventory_grid_shape = tuple(
        int(value)
        for value in input_inventory["static_external_inputs"]["global_constants"]["grid_shape"]
    )
    if inferred_grid_shape != inventory_grid_shape:
        raise ValueError(
            "FJW grid shape mismatch between reference mesh and inventory: "
            f"{inferred_grid_shape} != {inventory_grid_shape}."
        )

    top_node_ids = _parse_nset_members(end1_template_path, "top_nod")
    bottom_node_ids = _parse_nset_members(end1_template_path, "bot_nod")
    inventory_node_counts = _inventory_node_set_counts(input_inventory)
    if top_node_ids.size != inventory_node_counts["TOP_NOD"]:
        raise ValueError(
            "TOP_NOD count mismatch between end1.inp and fjw_input_inventory.json: "
            f"{top_node_ids.size} != {inventory_node_counts['TOP_NOD']}."
        )
    if bottom_node_ids.size != inventory_node_counts["BOT_NOD"]:
        raise ValueError(
            "BOT_NOD count mismatch between end1.inp and fjw_input_inventory.json: "
            f"{bottom_node_ids.size} != {inventory_node_counts['BOT_NOD']}."
        )

    element_anchor_indices = _build_element_anchor_indices(
        reference_model.node_coordinates,
        reference_model.element_nodes,
        inferred_grid_shape,
    )

    return FJWReferenceMeshContext(
        reference_model=reference_model,
        strain_displacement_matrix=strain_displacement_matrix,
        constitutive_matrix=constitutive_matrix,
        grid_shape_xyz=inferred_grid_shape,
        top_node_ids=top_node_ids,
        bottom_node_ids=bottom_node_ids,
        element_anchor_indices=element_anchor_indices,
        design_anchor_indices=element_anchor_indices[reference_model.design_elements - 1],
        objective_anchor_indices=element_anchor_indices[reference_model.objective_elements - 1],
    )


def load_fjw_load_cases(
    structured_inputs: Mapping[str, Any] | None = None,
    *,
    abaqus_inputs_path: Path = DEFAULT_ABAQUS_INPUTS_PATH,
) -> tuple[FJWLoadCase, ...]:
    if structured_inputs is None:
        structured_inputs = _load_json(abaqus_inputs_path)

    boundary_conditions = tuple(
        _build_boundary_condition(entry)
        for entry in structured_inputs["boundary_conditions"]
    )
    forward_load_cases = structured_inputs["forward_load_cases"]
    ordered_case_names = sorted(
        forward_load_cases,
        key=lambda name: int(str(name).rsplit("_", 1)[-1]),
    )

    load_cases: list[FJWLoadCase] = []
    for case_name in ordered_case_names:
        payload = forward_load_cases[case_name]
        template_path = _resolve_repo_path(payload["template"])
        load_cases.append(
            FJWLoadCase(
                name=str(case_name),
                template_path=template_path,
                template_lines=tuple(
                    template_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                ),
                boundary_conditions=boundary_conditions,
                loads=tuple(
                    FJWLoad(
                        target=str(load_entry["target"]),
                        dof=int(load_entry["dof"]),
                        magnitude=float(load_entry["magnitude"]),
                        op=str(load_entry.get("op", "NEW")),
                    )
                    for load_entry in payload["loads"]
                ),
            )
        )
    return tuple(load_cases)


def compute_modulus_buckets(
    design_cage: np.ndarray,
    obj_bo: np.ndarray,
    material_constants: FJWMaterialConstants,
    *,
    cage_material_buckets: Iterable[FJWMaterialBucket] | None = None,
    bone_material_buckets: Iterable[FJWMaterialBucket] | None = None,
) -> FJWModulusBuckets:
    design_cage = np.asarray(design_cage, dtype=np.float64).reshape(-1)
    obj_bo = np.asarray(obj_bo, dtype=np.float64).reshape(-1)

    design_cage_clipped = np.clip(design_cage, 0.001, 1.0)
    obj_bo_clipped = np.clip(obj_bo, 0.001, material_constants.bone_density_upper_bound)

    E_cage = material_constants.cage_modulus_min + (
        material_constants.cage_modulus_0 * np.power(design_cage_clipped, 3)
    )
    E_cage = np.minimum(E_cage, material_constants.cage_modulus_0)

    E_obj = material_constants.bone_modulus_min + (
        material_constants.bone_modulus_0
        * np.power(obj_bo_clipped / material_constants.bone_density_upper_bound, 3)
    )
    E_obj = np.minimum(E_obj, material_constants.bone_modulus_0)

    cage_bucket_indices = np.rint(
        E_cage / material_constants.cage_modulus_0 * material_constants.cage_bucket_max_index
    ).astype(np.int32)
    cage_bucket_indices = np.clip(
        cage_bucket_indices,
        0,
        material_constants.cage_bucket_max_index,
    )

    obj_bucket_indices = np.rint(
        E_obj / material_constants.bone_modulus_0 * material_constants.bone_bucket_max_index
    ).astype(np.int32)
    obj_bucket_indices = np.clip(
        obj_bucket_indices,
        0,
        material_constants.bone_bucket_max_index,
    )

    cage_names, cage_bucket_moduli = _material_arrays_from_indices(
        cage_bucket_indices,
        _bucket_map(cage_material_buckets),
        material_prefix="CAGE_",
        fallback_moduli=E_cage,
    )
    bone_names, obj_bucket_moduli = _material_arrays_from_indices(
        obj_bucket_indices,
        _bucket_map(bone_material_buckets),
        material_prefix="BONE_",
        fallback_moduli=E_obj,
    )

    return FJWModulusBuckets(
        design_cage_clipped=design_cage_clipped,
        obj_bo_clipped=obj_bo_clipped,
        E_cage=E_cage,
        E_obj=E_obj,
        cage_bucket_indices=cage_bucket_indices,
        obj_bucket_indices=obj_bucket_indices,
        cage_bucket_moduli=cage_bucket_moduli,
        obj_bucket_moduli=obj_bucket_moduli,
        cage_material_names=cage_names,
        obj_material_names=bone_names,
    )


def build_initial_design_state(
    mesh: FJWReferenceMeshContext,
    material_constants: FJWMaterialConstants,
    *,
    mode: str = "single_load",
    cage_material_buckets: Iterable[FJWMaterialBucket] | None = None,
    bone_material_buckets: Iterable[FJWMaterialBucket] | None = None,
) -> FJWInitialState:
    initial_design_value = material_constants.initial_design_value(mode)
    cage_3d = np.full(mesh.grid_shape_xyz, initial_design_value, dtype=np.float64)
    bone_3d = np.full(mesh.grid_shape_xyz, material_constants.initial_bone_density, dtype=np.float64)

    design_anchor_indices = mesh.design_anchor_indices
    objective_anchor_indices = mesh.objective_anchor_indices
    design_cage = np.asarray(
        cage_3d[
            design_anchor_indices[:, 0],
            design_anchor_indices[:, 1],
            design_anchor_indices[:, 2],
        ],
        dtype=np.float64,
    )
    obj_bo = np.asarray(
        bone_3d[
            objective_anchor_indices[:, 0],
            objective_anchor_indices[:, 1],
            objective_anchor_indices[:, 2],
        ],
        dtype=np.float64,
    )

    modulus_buckets = compute_modulus_buckets(
        design_cage,
        obj_bo,
        material_constants,
        cage_material_buckets=cage_material_buckets,
        bone_material_buckets=bone_material_buckets,
    )

    return FJWInitialState(
        mode=mode,
        cage_3d=cage_3d,
        bone_3d=bone_3d,
        design_cage=design_cage,
        obj_bo=obj_bo,
        initial_design_total=float(np.sum(design_cage, dtype=np.float64)),
        xold1=design_cage.copy(),
        xold2=design_cage.copy(),
        modulus_buckets=modulus_buckets,
    )


def load_fjw_workflow_state(
    *,
    reference_dir: Path = DEFAULT_REFERENCE_DIR,
    abaqus_inputs_path: Path = DEFAULT_ABAQUS_INPUTS_PATH,
    input_inventory_path: Path = DEFAULT_INPUT_INVENTORY_PATH,
    end1_template_path: Path = DEFAULT_END1_TEMPLATE_PATH,
    initial_design_mode: str = "single_load",
    objective_region: str = DEFAULT_OBJECTIVE_REGION,
    objective_layer_count: int = DEFAULT_OBJECTIVE_LAYER_COUNT,
) -> FJWWorkflowState:
    reference_dir = _resolve_repo_path(reference_dir)
    abaqus_inputs_path = _resolve_repo_path(abaqus_inputs_path)
    input_inventory_path = _resolve_repo_path(input_inventory_path)
    end1_template_path = _resolve_repo_path(end1_template_path)

    structured_inputs = _load_json(abaqus_inputs_path)
    input_inventory = _load_json(input_inventory_path)
    material_constants = load_fjw_material_constants(structured_inputs)
    mesh = load_fjw_reference_mesh_context(
        reference_dir=reference_dir,
        input_inventory_path=input_inventory_path,
        end1_template_path=end1_template_path,
        objective_region=objective_region,
        objective_layer_count=objective_layer_count,
    )
    load_cases = load_fjw_load_cases(structured_inputs)
    (
        cage_material_buckets,
        bone_material_buckets,
        background_material_buckets,
    ) = _build_material_bucket_catalogs(structured_inputs)
    initial_state = build_initial_design_state(
        mesh,
        material_constants,
        mode=initial_design_mode,
        cage_material_buckets=cage_material_buckets,
        bone_material_buckets=bone_material_buckets,
    )

    return FJWWorkflowState(
        reference_dir=reference_dir,
        abaqus_inputs_path=abaqus_inputs_path,
        input_inventory_path=input_inventory_path,
        end1_template_path=end1_template_path,
        mesh=mesh,
        material_constants=material_constants,
        load_cases=load_cases,
        cage_material_buckets=cage_material_buckets,
        bone_material_buckets=bone_material_buckets,
        background_material_buckets=background_material_buckets,
        initial_state=initial_state,
        assembly_controls=structured_inputs["assembly_controls"],
        adjoint_load_template=structured_inputs["adjoint_load_template"],
        structured_inputs=structured_inputs,
        input_inventory=input_inventory,
    )


__all__ = [
    "DEFAULT_ABAQUS_INPUTS_PATH",
    "DEFAULT_END1_TEMPLATE_PATH",
    "DEFAULT_INPUT_INVENTORY_PATH",
    "build_initial_design_state",
    "compute_modulus_buckets",
    "load_fjw_load_cases",
    "load_fjw_material_constants",
    "load_fjw_reference_mesh_context",
    "load_fjw_workflow_state",
]
