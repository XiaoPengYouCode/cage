from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

with redirect_stdout(StringIO()):
    from sfepy.base.base import output as sfepy_output
    from sfepy.discrete import Equation, Equations, FieldVariable, Integral, Material, Problem
    from sfepy.discrete.conditions import Conditions, EssentialBC
    from sfepy.discrete.common.region import Region
    from sfepy.discrete.fem import FEDomain, Field, Mesh
    from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
    from sfepy.solvers.ls import ScipyDirect
    from sfepy.solvers.nls import Newton
    from sfepy.terms import Term


def _load_stl(path: Path):
    sys.path.insert(0, str(ROOT / "src"))
    from ct_reconstruction.voxelizer import load_stl

    return load_stl(path)


def _voxelize_stl(path: Path, voxel_size_mm: float):
    sys.path.insert(0, str(ROOT / "src"))
    from ct_reconstruction.voxelizer import voxelize_stl

    stl_mesh = _load_stl(path)
    return voxelize_stl(stl_mesh, voxel_size_mm=voxel_size_mm)


@dataclass(frozen=True)
class MaterialConfig:
    name: str
    youngs_modulus_gpa: float
    poisson_ratio: float

    @property
    def youngs_modulus_pa(self) -> float:
        return self.youngs_modulus_gpa * 1e9


@dataclass(frozen=True)
class HexMesh:
    coordinates: np.ndarray
    connectivity: np.ndarray
    descriptor: str


def _quiet_sfepy_output():
    class _Ctx:
        def __enter__(self):
            sfepy_output.set_output(quiet=True)
            return self

        def __exit__(self, exc_type, exc, tb):
            sfepy_output.set_output(quiet=False)
            return False

    return _Ctx()


def _region_from_vertices(domain: FEDomain, name: str, vertex_ids: np.ndarray) -> Region:
    region = Region.from_vertices(vertex_ids, domain, name=name, kind="vertex")
    region.finalize()
    region.update_shape()
    domain.regions.append(region)
    return region


def _build_bottom_anchor_regions(
    domain: FEDomain,
    coordinates: np.ndarray,
    bottom_vertex_ids: np.ndarray,
) -> tuple[Region, Region, str]:
    bottom_coords = coordinates[bottom_vertex_ids]
    origin_local_index = int(np.argmin(bottom_coords[:, 0] + bottom_coords[:, 1]))
    origin_vertex_id = int(bottom_vertex_ids[origin_local_index])
    origin_xy = bottom_coords[origin_local_index, :2]

    deltas_xy = bottom_coords[:, :2] - origin_xy
    y_spread_local_index = int(np.argmax(np.abs(deltas_xy[:, 1])))
    x_spread_local_index = int(np.argmax(np.abs(deltas_xy[:, 0])))
    y_spread = float(np.abs(deltas_xy[y_spread_local_index, 1]))
    x_spread = float(np.abs(deltas_xy[x_spread_local_index, 0]))

    if y_spread >= x_spread and y_spread > 1e-12:
        twist_vertex_id = int(bottom_vertex_ids[y_spread_local_index])
        twist_component = "u.0"
    elif x_spread > 1e-12:
        twist_vertex_id = int(bottom_vertex_ids[x_spread_local_index])
        twist_component = "u.1"
    else:
        raise ValueError("The bottom support does not provide enough distinct nodes for stabilization.")

    in_plane_anchor = _region_from_vertices(
        domain,
        "BottomAnchorInPlane",
        np.array([origin_vertex_id], dtype=np.uint32),
    )
    twist_anchor = _region_from_vertices(
        domain,
        "BottomAnchorTwist",
        np.array([twist_vertex_id], dtype=np.uint32),
    )
    return in_plane_anchor, twist_anchor, twist_component


def build_hex_mesh_from_occupancy(
    occupancy: np.ndarray,
    origin_mm: np.ndarray,
    spacing_mm: np.ndarray,
) -> tuple[HexMesh, np.ndarray, np.ndarray, float, np.ndarray]:
    active_elements = np.argwhere(occupancy)
    if len(active_elements) == 0:
        raise ValueError("Occupancy grid is empty.")

    nx, ny, nz = occupancy.shape
    local_offsets = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.int32,
    )
    node_shape = (nx + 1, ny + 1, nz + 1)
    node_triplets = active_elements[:, None, :] + local_offsets[None, :, :]
    global_node_ids = np.ravel_multi_index(node_triplets.reshape(-1, 3).T, node_shape)
    unique_nodes, inverse = np.unique(global_node_ids, return_inverse=True)
    connectivity = inverse.reshape(-1, 8).astype(np.int32)

    indices = np.array(np.unravel_index(unique_nodes, node_shape), dtype=np.float64).T
    coordinates_mm = origin_mm[None, :] - 0.5 * spacing_mm[None, :] + indices * spacing_mm[None, :]

    top_mask = occupancy[:, :, -1]
    loaded_top_area_mm2 = float(top_mask.sum() * spacing_mm[0] * spacing_mm[1])
    extents_mm = np.array(occupancy.shape, dtype=np.float64) * spacing_mm
    return (
        HexMesh(
            coordinates=coordinates_mm,
            connectivity=connectivity,
            descriptor="3_8",
        ),
        active_elements,
        extents_mm,
        loaded_top_area_mm2,
        spacing_mm,
    )


def solve_voxel_compression(
    occupancy: np.ndarray,
    origin_mm: np.ndarray,
    spacing_mm: np.ndarray,
    *,
    total_force_n: float,
    material: MaterialConfig,
) -> dict[str, object]:
    hex_mesh_mm, active_elements, extents_mm, loaded_top_area_mm2, spacing_mm = build_hex_mesh_from_occupancy(
        occupancy, origin_mm, spacing_mm
    )
    coordinates_m = hex_mesh_mm.coordinates * 1e-3
    connectivity = hex_mesh_mm.connectivity
    specimen_height_mm = float(extents_mm[2])
    specimen_height_m = specimen_height_mm * 1e-3
    spacing_m = spacing_mm * 1e-3
    origin_m = origin_mm * 1e-3

    mesh = Mesh.from_data(
        "voxel_specimen",
        coordinates_m,
        np.zeros(len(coordinates_m), dtype=np.int32),
        [connectivity],
        [np.zeros(len(connectivity), dtype=np.int32)],
        [hex_mesh_mm.descriptor],
    )
    domain = FEDomain("domain", mesh)
    omega = domain.create_region("Omega", "all")
    body_region = domain.create_region("Body", "cells of group 0")

    tol = float(spacing_m[2] * 0.5 + 1e-12)
    z_bottom_threshold = float(origin_m[2]) + tol
    z_top_threshold = float(origin_m[2] - 0.5 * spacing_m[2] + specimen_height_m) - tol
    bottom_vertices = domain.create_region("BottomVertices", f"vertices in z < {z_bottom_threshold}", "vertex")
    top_vertices = domain.create_region("TopVertices", f"vertices in z > {z_top_threshold}", "vertex")
    top_surface = domain.create_region("TopSurface", f"vertices in z > {z_top_threshold}", "facet")
    if len(bottom_vertices.vertices) == 0 or len(top_vertices.vertices) == 0:
        raise ValueError("The specimen does not touch both loading faces.")

    field = Field.from_args("displacement", np.float64, "vector", omega, approx_order=1)
    u = FieldVariable("u", "unknown", field)
    v = FieldVariable("v", "test", field, primary_var_name="u")

    body = Material(
        "body",
        D=stiffness_from_youngpoisson(
            dim=3,
            young=material.youngs_modulus_pa,
            poisson=material.poisson_ratio,
        ),
    )
    traction = Material(
        "traction",
        val=np.array([[0.0], [0.0], [-total_force_n / (loaded_top_area_mm2 * 1e-6)]], dtype=np.float64),
    )

    in_plane_anchor, twist_anchor, twist_component = _build_bottom_anchor_regions(
        domain,
        coordinates_m,
        bottom_vertices.vertices,
    )
    integral = Integral("i", order=2)
    balance = Equation(
        "balance",
        Term.new("dw_lin_elastic(body.D, v, u)", integral, body_region, body=body, v=v, u=u)
        - Term.new("dw_surface_ltr(traction.val, v)", integral, top_surface, traction=traction, v=v),
    )
    problem = Problem("voxel_specimen_compression", equations=Equations([balance]))
    problem.set_bcs(
        ebcs=Conditions(
            [
                EssentialBC("support_bottom", bottom_vertices, {"u.2": 0.0}),
                EssentialBC("anchor_in_plane_origin", in_plane_anchor, {"u.0": 0.0, "u.1": 0.0}),
                EssentialBC("anchor_bottom_twist", twist_anchor, {twist_component: 0.0}),
            ]
        )
    )
    problem.set_solver(Newton({"i_max": 1, "eps_a": 1e-10, "eps_r": 1.0}, lin_solver=ScipyDirect({})))

    with _quiet_sfepy_output():
        variables = problem.solve(save_results=False, verbose=False)

    nodal_displacements = variables.get_state_parts()["u"].reshape((-1, 3), order="C")
    top_displacement_mm = np.abs(nodal_displacements[top_vertices.vertices, 2])
    mean_top_displacement_mm = float(np.mean(top_displacement_mm) * 1e3)
    apparent_stiffness_kn_per_mm = float(total_force_n / max(mean_top_displacement_mm, 1e-12) / 1e3)
    loaded_top_area_m2 = loaded_top_area_mm2 * 1e-6
    apparent_modulus_gpa = float(
        (total_force_n / max(loaded_top_area_m2, 1e-12))
        / max(mean_top_displacement_mm * 1e-3 / max(specimen_height_m, 1e-12), 1e-12)
        / 1e9
    )
    return {
        "grid_shape_xyz": [int(v) for v in occupancy.shape],
        "active_element_count": int(len(active_elements)),
        "active_node_count": int(len(coordinates_m)),
        "loaded_top_area_mm2": loaded_top_area_mm2,
        "specimen_height_mm": specimen_height_mm,
        "solid_volume_fraction": float(np.mean(occupancy.astype(np.float64))),
        "max_displacement_mm": float(np.max(np.linalg.norm(nodal_displacements, axis=1)) * 1e3),
        "mean_top_displacement_mm": mean_top_displacement_mm,
        "apparent_stiffness_kn_per_mm": apparent_stiffness_kn_per_mm,
        "apparent_modulus_gpa": apparent_modulus_gpa,
    }


def evaluate_manifest(
    *,
    manifest_path: Path,
    voxel_size_mm: float,
    total_force_n: float,
    material_name: str,
    youngs_modulus_gpa: float,
    poisson_ratio: float,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    material = MaterialConfig(
        name=material_name,
        youngs_modulus_gpa=youngs_modulus_gpa,
        poisson_ratio=poisson_ratio,
    )

    results = []
    for specimen in manifest["specimens"]:
        stl_path = Path(specimen["stl"]["output_path"])
        occupancy, origin_mm, spacing_mm = _voxelize_stl(stl_path, voxel_size_mm=voxel_size_mm)
        metrics = solve_voxel_compression(
            occupancy,
            origin_mm,
            spacing_mm,
            total_force_n=total_force_n,
            material=material,
        )
        base_record = {
            "tag": specimen["tag"],
            "radius_mm": float(specimen["radius_mm"]),
            "voxel_size_mm": float(voxel_size_mm),
            "material_name": material.name,
            "youngs_modulus_gpa": float(material.youngs_modulus_gpa),
            "poisson_ratio": float(material.poisson_ratio),
            "metrics": metrics,
        }
        if "band_index" in specimen:
            results.append(
                {
                    **base_record,
                    "band_index": int(specimen["band_index"]),
                    "representative_design": float(specimen["representative_design"]),
                    "representative_target_modulus": float(specimen["representative_target_modulus"]),
                }
            )
        else:
            results.append(
                {
                    **base_record,
                    "support_bands": specimen.get("support_bands", []),
                }
            )

    payload = {
        "source_manifest": str(manifest_path.resolve()),
        "evaluation_mode": "local_voxel_compression",
        "load_n": float(total_force_n),
        "material": {
            "name": material.name,
            "youngs_modulus_gpa": float(material.youngs_modulus_gpa),
            "poisson_ratio": float(material.poisson_ratio),
        },
        "voxel_size_mm": float(voxel_size_mm),
        "specimen_count": int(len(results)),
        "results": results,
    }
    output_path = manifest_path.with_name("calibration_fe_results.json")
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["output_path"] = str(output_path.resolve())
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Voronoi radius calibration specimens with local voxel compression FE.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--voxel-size-mm",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--load-n",
        type=float,
        default=1000.0,
    )
    parser.add_argument(
        "--material-name",
        type=str,
        default="TC4",
    )
    parser.add_argument(
        "--youngs-modulus-gpa",
        type=float,
        default=110.0,
    )
    parser.add_argument(
        "--poisson-ratio",
        type=float,
        default=0.34,
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    payload = evaluate_manifest(
        manifest_path=args.manifest,
        voxel_size_mm=float(args.voxel_size_mm),
        total_force_n=float(args.load_n),
        material_name=str(args.material_name),
        youngs_modulus_gpa=float(args.youngs_modulus_gpa),
        poisson_ratio=float(args.poisson_ratio),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
