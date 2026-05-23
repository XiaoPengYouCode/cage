from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "references" / "fjw_work"
DEFAULT_OUTPUT_NPZ = PROJECT_ROOT / "datasets" / "topopt" / "fjw_reference_fem_voxels.npz"
DEFAULT_VOXEL_SIZE_MM = 0.6
OBJECTIVE_REGION_ARCHIVED_INNER_CAGE = "archived_inner_cage"
OBJECTIVE_REGION_C3_CONTACT_BONE = "c3_contact_bone"
DEFAULT_OBJECTIVE_REGION = OBJECTIVE_REGION_C3_CONTACT_BONE
DEFAULT_OBJECTIVE_LAYER_COUNT = 3


@dataclass(frozen=True)
class FJWReferenceModel:
    node_coordinates: np.ndarray
    element_nodes: np.ndarray
    cor_elements: np.ndarray
    tra_elements: np.ndarray
    cage_elements: np.ndarray
    design_elements: np.ndarray
    objective_elements: np.ndarray


@dataclass(frozen=True)
class FJWReferenceExportSummary:
    output_npz: Path
    grid_shape_xyz: tuple[int, int, int]
    active_voxel_count: int
    voxel_size_mm: float


def _load_mat_array(path: Path, variable_name: str) -> np.ndarray:
    raw = loadmat(path)
    if variable_name not in raw:
        raise KeyError(f"Variable '{variable_name}' not found in {path}.")
    return np.asarray(raw[variable_name])


def _load_element_set(path: Path, variable_name: str) -> np.ndarray:
    values = _load_mat_array(path, variable_name).reshape(-1)
    return np.asarray(values, dtype=np.int32)


def build_c3_contact_bone_objective_elements(
    model: FJWReferenceModel,
    *,
    layer_count: int = DEFAULT_OBJECTIVE_LAYER_COUNT,
) -> np.ndarray:
    """Build the C3-side bone ring under the outer cage contact footprint."""
    layer_count = int(layer_count)
    if layer_count <= 0:
        raise ValueError("layer_count must be positive.")

    voxel_indices, grid_shape = _element_lower_corners(
        model.node_coordinates,
        model.element_nodes,
    )
    design_coords = voxel_indices[np.asarray(model.design_elements, dtype=np.int64) - 1]
    if design_coords.size == 0:
        raise ValueError("design_elements is empty.")

    c3_contact_z = int(design_coords[:, 2].max())
    contact_layer_coords = design_coords[design_coords[:, 2] == c3_contact_z]
    if contact_layer_coords.size == 0:
        raise ValueError("Could not identify the C3 cage contact layer.")

    footprint_mask = np.zeros(grid_shape[:2], dtype=bool)
    footprint_mask[contact_layer_coords[:, 0], contact_layer_coords[:, 1]] = True

    bone_elements = np.unique(
        np.concatenate(
            (
                np.asarray(model.cor_elements, dtype=np.int32).reshape(-1),
                np.asarray(model.tra_elements, dtype=np.int32).reshape(-1),
            )
        )
    )
    bone_coords = voxel_indices[bone_elements.astype(np.int64) - 1]
    xy_in_footprint = footprint_mask[bone_coords[:, 0], bone_coords[:, 1]]
    in_c3_layers = (
        (bone_coords[:, 2] > c3_contact_z)
        & (bone_coords[:, 2] <= c3_contact_z + layer_count)
    )
    objective_elements = np.sort(bone_elements[xy_in_footprint & in_c3_layers])
    if objective_elements.size == 0:
        raise ValueError("C3 contact objective region is empty.")
    return np.asarray(objective_elements, dtype=np.int32)


def load_fjw_reference_model(
    reference_dir: Path = DEFAULT_REFERENCE_DIR,
    *,
    objective_region: str = DEFAULT_OBJECTIVE_REGION,
    objective_layer_count: int = DEFAULT_OBJECTIVE_LAYER_COUNT,
) -> FJWReferenceModel:
    reference_dir = Path(reference_dir)
    model = FJWReferenceModel(
        node_coordinates=np.asarray(
            _load_mat_array(reference_dir / "nod_coo.mat", "nod_coo"),
            dtype=np.int32,
        ),
        element_nodes=np.asarray(
            _load_mat_array(reference_dir / "ele_nod.mat", "ele_nod"),
            dtype=np.int32,
        ),
        cor_elements=_load_element_set(reference_dir / "cor_ele.mat", "cor_ele"),
        tra_elements=_load_element_set(reference_dir / "tra_ele.mat", "tra_ele"),
        cage_elements=_load_element_set(reference_dir / "cage_ele.mat", "cage_ele"),
        design_elements=_load_element_set(reference_dir / "desi_ele.mat", "desi_ele"),
        objective_elements=_load_element_set(reference_dir / "obj_ele.mat", "obj_ele"),
    )
    if objective_region == OBJECTIVE_REGION_ARCHIVED_INNER_CAGE:
        return model
    if objective_region == OBJECTIVE_REGION_C3_CONTACT_BONE:
        return replace(
            model,
            objective_elements=build_c3_contact_bone_objective_elements(
                model,
                layer_count=objective_layer_count,
            ),
        )
    raise ValueError(
        "Unsupported objective_region "
        f"{objective_region!r}; expected {OBJECTIVE_REGION_C3_CONTACT_BONE!r} "
        f"or {OBJECTIVE_REGION_ARCHIVED_INNER_CAGE!r}."
    )


def _validate_indices(element_ids: np.ndarray, num_elements: int, label: str) -> None:
    if element_ids.size == 0:
        raise ValueError(f"{label} is empty.")
    if np.any(element_ids < 1) or np.any(element_ids > num_elements):
        raise ValueError(f"{label} contains out-of-range element ids.")
    if np.unique(element_ids).size != element_ids.size:
        raise ValueError(f"{label} contains duplicate element ids.")


def _element_lower_corners(
    node_coordinates: np.ndarray,
    element_nodes: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    if node_coordinates.ndim != 2 or node_coordinates.shape[1] != 3:
        raise ValueError("node_coordinates must have shape (N, 3).")
    if element_nodes.ndim != 2 or element_nodes.shape[1] != 8:
        raise ValueError("element_nodes must have shape (M, 8).")
    if np.any(element_nodes < 1) or np.any(element_nodes > node_coordinates.shape[0]):
        raise ValueError("element_nodes references out-of-range node ids.")

    node_min = node_coordinates.min(axis=0)
    if not np.array_equal(node_min, np.ones(3, dtype=np.int32)):
        raise ValueError(f"Expected node coordinates to start at [1, 1, 1], got {node_min.tolist()}.")

    node_max = node_coordinates.max(axis=0)
    grid_shape = tuple(int(value - 1) for value in node_max.tolist())

    element_points = node_coordinates[element_nodes - 1]
    lower_corners = element_points.min(axis=1)
    spans = element_points.max(axis=1) - lower_corners
    if not np.all(spans == 1):
        unique_spans = np.unique(spans, axis=0)
        raise ValueError(
            "Reference mesh is not a pure unit hexahedral lattice. "
            f"Observed spans: {unique_spans.tolist()[:5]}"
        )

    voxel_indices = lower_corners - 1
    if np.any(voxel_indices < 0):
        raise ValueError("Computed negative voxel indices from element lower corners.")
    grid_shape_array = np.array(grid_shape, dtype=np.int32)
    if np.any(voxel_indices >= grid_shape_array):
        raise ValueError("Computed voxel index outside inferred grid shape.")
    if np.unique(voxel_indices, axis=0).shape[0] != voxel_indices.shape[0]:
        raise ValueError("Multiple elements map to the same voxel index.")
    return voxel_indices, grid_shape


def build_fjw_reference_npz_payload(
    model: FJWReferenceModel,
    *,
    voxel_size_mm: float = DEFAULT_VOXEL_SIZE_MM,
    source_reference_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    if voxel_size_mm <= 0.0:
        raise ValueError("voxel_size_mm must be positive.")

    num_elements = int(model.element_nodes.shape[0])
    for label, element_ids in (
        ("cor_elements", model.cor_elements),
        ("tra_elements", model.tra_elements),
        ("cage_elements", model.cage_elements),
        ("design_elements", model.design_elements),
        ("objective_elements", model.objective_elements),
    ):
        _validate_indices(element_ids, num_elements, label)

    primary_count = (
        model.cor_elements.size
        + model.tra_elements.size
        + model.cage_elements.size
    )
    primary_unique = np.unique(
        np.concatenate(
            (
                model.cor_elements,
                model.tra_elements,
                model.cage_elements,
            )
        )
    )
    if primary_count != num_elements or primary_unique.size != num_elements:
        raise ValueError("cor/tra/cage element sets do not form a full partition of the reference mesh.")
    if np.intersect1d(model.cor_elements, model.tra_elements).size > 0:
        raise ValueError("cor_elements and tra_elements overlap.")
    if np.intersect1d(model.cor_elements, model.cage_elements).size > 0:
        raise ValueError("cor_elements and cage_elements overlap.")
    if np.intersect1d(model.tra_elements, model.cage_elements).size > 0:
        raise ValueError("tra_elements and cage_elements overlap.")
    if np.setdiff1d(model.design_elements, model.cage_elements).size > 0:
        raise ValueError("design_elements must be a subset of cage_elements.")
    if np.intersect1d(model.objective_elements, model.design_elements).size > 0:
        raise ValueError("objective_elements must not overlap design_elements.")

    voxel_indices, grid_shape = _element_lower_corners(
        model.node_coordinates,
        model.element_nodes,
    )
    grid_shape_array = np.array(grid_shape, dtype=np.int32)
    occupancy = np.zeros(grid_shape, dtype=np.uint8)
    material_id = np.full(grid_shape, -1, dtype=np.int8)
    element_id = np.full(grid_shape, -1, dtype=np.int32)
    design_mask = np.zeros(grid_shape, dtype=np.uint8)
    objective_mask = np.zeros(grid_shape, dtype=np.uint8)

    x_idx = voxel_indices[:, 0]
    y_idx = voxel_indices[:, 1]
    z_idx = voxel_indices[:, 2]
    occupancy[x_idx, y_idx, z_idx] = 1
    element_id[x_idx, y_idx, z_idx] = np.arange(1, num_elements + 1, dtype=np.int32)

    for set_ids, value in (
        (model.cor_elements, 0),
        (model.tra_elements, 1),
        (model.cage_elements, 2),
    ):
        coords = voxel_indices[set_ids - 1]
        material_id[coords[:, 0], coords[:, 1], coords[:, 2]] = value

    design_coords = voxel_indices[model.design_elements - 1]
    objective_coords = voxel_indices[model.objective_elements - 1]
    design_mask[design_coords[:, 0], design_coords[:, 1], design_coords[:, 2]] = 1
    objective_mask[objective_coords[:, 0], objective_coords[:, 1], objective_coords[:, 2]] = 1

    voxel_size_m = np.float32(voxel_size_mm / 1e3)
    voxel_size_xyz_m = np.array([voxel_size_m, voxel_size_m, voxel_size_m], dtype=np.float32)
    density_milli = occupancy.astype(np.uint16) * np.uint16(1000)

    return {
        "voxels": occupancy,
        "density_milli": density_milli,
        "material_id": material_id,
        "design_mask": design_mask,
        "objective_mask": objective_mask,
        "element_id": element_id,
        "grid_shape_xyz": grid_shape_array,
        "x_size": np.array(grid_shape[0], dtype=np.int32),
        "y_size": np.array(grid_shape[1], dtype=np.int32),
        "z_size": np.array(grid_shape[2], dtype=np.int32),
        "origin_m": np.zeros(3, dtype=np.float32),
        "voxel_size_m": voxel_size_m,
        "voxel_size_xyz_m": voxel_size_xyz_m,
        "shape_name": np.array("fjw_reference"),
        "result_type": np.array("fjw_reference_fem_voxels"),
        "density_kind": np.array("binary_occupancy"),
        "density_precision": np.array(3, dtype=np.int32),
        "density_min_nonzero": np.array(1.0, dtype=np.float32),
        "density_max": np.array(1.0, dtype=np.float32),
        "reference_voxel_size_mm": np.array(voxel_size_mm, dtype=np.float32),
        "material_labels": np.array(["cor", "tra", "cage"]),
        "source_reference_dir": np.array(str(source_reference_dir or DEFAULT_REFERENCE_DIR)),
    }


def write_fjw_reference_npz(
    output_path: Path,
    *,
    reference_dir: Path = DEFAULT_REFERENCE_DIR,
    voxel_size_mm: float = DEFAULT_VOXEL_SIZE_MM,
    objective_region: str = DEFAULT_OBJECTIVE_REGION,
    objective_layer_count: int = DEFAULT_OBJECTIVE_LAYER_COUNT,
) -> FJWReferenceExportSummary:
    model = load_fjw_reference_model(
        reference_dir,
        objective_region=objective_region,
        objective_layer_count=objective_layer_count,
    )
    payload = build_fjw_reference_npz_payload(
        model,
        voxel_size_mm=voxel_size_mm,
        source_reference_dir=reference_dir,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return FJWReferenceExportSummary(
        output_npz=output_path,
        grid_shape_xyz=tuple(int(v) for v in payload["grid_shape_xyz"].tolist()),
        active_voxel_count=int(payload["voxels"].sum()),
        voxel_size_mm=float(voxel_size_mm),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/export_fjw_reference_voxels.py",
        description="Convert the archived FJW Abaqus hexahedral reference mesh into the repository-standard voxel NPZ format.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="Directory containing the FJW reference .mat files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_NPZ,
        help="Destination NPZ path.",
    )
    parser.add_argument(
        "--objective-region",
        choices=(OBJECTIVE_REGION_C3_CONTACT_BONE, OBJECTIVE_REGION_ARCHIVED_INNER_CAGE),
        default=DEFAULT_OBJECTIVE_REGION,
        help="Objective region source.",
    )
    parser.add_argument(
        "--objective-layer-count",
        type=int,
        default=DEFAULT_OBJECTIVE_LAYER_COUNT,
        help="Number of C3-side bone layers used by c3_contact_bone.",
    )
    parser.add_argument(
        "--voxel-size-mm",
        type=float,
        default=DEFAULT_VOXEL_SIZE_MM,
        help="Physical voxel size in millimetres. Defaults to the 0.6 mm spacing used by the reference Abaqus export.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = write_fjw_reference_npz(
        output_path=args.output,
        reference_dir=args.reference_dir,
        voxel_size_mm=args.voxel_size_mm,
        objective_region=args.objective_region,
        objective_layer_count=args.objective_layer_count,
    )
    print(
        "saved FJW reference voxel NPZ to "
        f"{summary.output_npz} "
        f"(shape={summary.grid_shape_xyz}, active_voxels={summary.active_voxel_count}, voxel_size_mm={summary.voxel_size_mm:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
