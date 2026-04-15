from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import Normalize
from sfepy.base.base import output as sfepy_output
from sfepy.discrete.common.region import Region
from sfepy.discrete import Equation, Equations, FieldVariable, Integral, Material, Problem
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.discrete.fem import FEDomain, Field, Mesh
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.terms import Term

try:
    from pyparsing.exceptions import PyparsingDeprecationWarning
except ImportError:  # pragma: no cover
    PyparsingDeprecationWarning = None

if PyparsingDeprecationWarning is not None:
    warnings.filterwarnings("ignore", category=PyparsingDeprecationWarning)


ProgressCallback = Callable[[str], None]


def _emit_progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


@dataclass(frozen=True)
class MaterialConfig:
    name: str
    youngs_modulus_gpa: float
    poisson_ratio: float

    @property
    def youngs_modulus_pa(self) -> float:
        return self.youngs_modulus_gpa * 1e9


@dataclass(frozen=True)
class SegmentCloud:
    starts: np.ndarray
    ends: np.ndarray
    radius: float

    def contains_points(
        self, points: np.ndarray, chunk_size: int = 100_000
    ) -> np.ndarray:
        mask = np.zeros(len(points), dtype=bool)
        if len(points) == 0 or len(self.starts) == 0:
            return mask

        for offset in range(0, len(points), chunk_size):
            point_chunk = points[offset : offset + chunk_size]
            local_mask = np.zeros(len(point_chunk), dtype=bool)

            for start, end in zip(self.starts, self.ends):
                local_mask |= (
                    point_segment_distance_squared(point_chunk, start, end)
                    <= self.radius**2
                )
                if np.all(local_mask):
                    break

            mask[offset : offset + chunk_size] = local_mask

        return mask


def point_segment_distance_squared(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    segment = end - start
    segment_norm_sq = float(np.dot(segment, segment))
    if segment_norm_sq <= 1e-18:
        diff = points - start
        return np.einsum("ij,ij->i", diff, diff)

    projection = (points - start) @ segment / segment_norm_sq
    projection = np.clip(projection, 0.0, 1.0)
    closest = start + projection[:, None] * segment
    diff = points - closest
    return np.einsum("ij,ij->i", diff, diff)


@dataclass(frozen=True)
class HexMesh:
    coordinates: np.ndarray
    connectivity: np.ndarray
    material_ids: np.ndarray
    descriptor: str
    resolution: int
    voxel_size: float
    active_voxel_count: int

    @property
    def active_element_count(self) -> int:
        return int(len(self.connectivity))

    @property
    def active_node_count(self) -> int:
        return int(len(self.coordinates))


@contextmanager
def quiet_sfepy_output():
    sfepy_output.set_output(quiet=True)
    try:
        yield
    finally:
        sfepy_output.set_output(quiet=False)


@dataclass(frozen=True)
class TrussInfillConfig:
    enabled: bool = False
    cell_size_m: float = 0.0004
    rod_radius_m: float = 0.0001
    include_vertical_struts: bool = True
    include_horizontal_struts: bool = True
    include_planar_diagonals: bool = True
    include_diagonal_struts: bool = True


@dataclass(frozen=True)
class AnnularCylinderConfig:
    outer_diameter_m: float = 0.03
    inner_diameter_m: float = 0.02
    height_m: float = 0.02
    total_force_n: float = 1000.0
    voxel_size_m: float = 0.0004
    material: MaterialConfig = MaterialConfig(
        name="TC4",
        youngs_modulus_gpa=110.0,
        poisson_ratio=0.34,
    )
    inner_fill_mode: str = "bone"
    fill_material: MaterialConfig = MaterialConfig(
        name="Bone graft equivalent",
        youngs_modulus_gpa=1.0,
        poisson_ratio=0.30,
    )
    truss_infill: TrussInfillConfig = TrussInfillConfig()
    output_image: Path = Path("docs/assets/annular_cylinder_fea.png")
    output_json: Path = Path("docs/analysis/annular_cylinder_fea.json")
    output_npz: Path = Path("datasets/topopt/annular_cylinder_fea_density.npz")

    @property
    def outer_radius_m(self) -> float:
        return self.outer_diameter_m / 2.0

    @property
    def inner_radius_m(self) -> float:
        return self.inner_diameter_m / 2.0

    @property
    def top_area_m2(self) -> float:
        return float(
            np.pi * (self.outer_radius_m**2 - self.inner_radius_m**2)
        )

    @property
    def applied_pressure_pa(self) -> float:
        return self.total_force_n / self.top_area_m2


@dataclass(frozen=True)
class AnnularCylinderMesh:
    hex_mesh: HexMesh
    occupancy: np.ndarray
    shell_mask: np.ndarray
    fill_mask: np.ndarray
    active_elements: np.ndarray
    spacing: np.ndarray
    extents: np.ndarray
    grid_shape: tuple[int, int, int]
    loaded_top_area_m2: float
    inner_void_voxel_count: int
    shell_element_count: int
    fill_element_count: int


@dataclass(frozen=True)
class AnnularCylinderResult:
    outer_diameter_mm: float
    inner_diameter_mm: float
    height_mm: float
    target_voxel_size_mm: float
    actual_voxel_size_x_mm: float
    actual_voxel_size_y_mm: float
    actual_voxel_size_z_mm: float
    grid_shape_xyz: tuple[int, int, int]
    applied_force_n: float
    applied_pressure_mpa: float
    loaded_top_area_mm2: float
    projected_ring_area_mm2: float
    material_name: str
    youngs_modulus_gpa: float
    poisson_ratio: float
    inner_fill_mode: str
    fill_material_name: str
    fill_youngs_modulus_gpa: float
    fill_poisson_ratio: float
    fill_volume_fraction: float
    total_volume_fraction: float
    active_element_count: int
    shell_element_count: int
    fill_element_count: int
    active_node_count: int
    top_contact_nodes: int
    bottom_contact_nodes: int
    max_displacement_mm: float
    mean_top_displacement_mm: float
    max_von_mises_mpa: float
    max_axial_stress_mpa: float
    mean_axial_stress_mpa: float
    apparent_stiffness_kn_per_mm: float


@dataclass(frozen=True)
class AnnularCylinderRunSummary:
    result: AnnularCylinderResult
    image_path: Path
    json_path: Path
    npz_path: Path


@dataclass(frozen=True)
class _FieldResult:
    result: AnnularCylinderResult
    nodal_displacements: np.ndarray
    element_centers: np.ndarray
    element_displacements: np.ndarray
    element_von_mises_mpa: np.ndarray
    element_axial_stress_mpa: np.ndarray
    mesh: AnnularCylinderMesh


def _validate_config(config: AnnularCylinderConfig) -> None:
    if config.outer_diameter_m <= 0.0:
        raise ValueError("outer_diameter_m must be positive.")
    if config.inner_diameter_m <= 0.0:
        raise ValueError("inner_diameter_m must be positive.")
    if config.inner_diameter_m >= config.outer_diameter_m:
        raise ValueError("inner_diameter_m must be smaller than outer_diameter_m.")
    if config.height_m <= 0.0:
        raise ValueError("height_m must be positive.")
    if config.total_force_n <= 0.0:
        raise ValueError("total_force_n must be positive.")
    if config.voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m must be positive.")
    if config.inner_fill_mode not in {"empty", "bone", "truss"}:
        raise ValueError("inner_fill_mode must be one of: empty, bone, truss.")
    if config.truss_infill.cell_size_m <= 0.0:
        raise ValueError("truss cell_size_m must be positive.")
    if config.truss_infill.rod_radius_m <= 0.0:
        raise ValueError("truss rod_radius_m must be positive.")


def _line_segments_from_pairs(points: dict[tuple[int, int, int], np.ndarray], pairs: set[tuple[tuple[int, int, int], tuple[int, int, int]]]) -> SegmentCloud:
    starts = []
    ends = []
    for a, b in sorted(pairs):
        starts.append(points[a])
        ends.append(points[b])
    return SegmentCloud(
        starts=np.asarray(starts, dtype=np.float64),
        ends=np.asarray(ends, dtype=np.float64),
        radius=0.0,
    )


def build_inner_truss_segment_cloud(
    config: AnnularCylinderConfig,
    extents: np.ndarray,
) -> SegmentCloud | None:
    truss = config.truss_infill
    if not truss.enabled:
        return None

    center = extents[:2] / 2.0
    inner_radius = config.inner_radius_m

    def axis_positions(length: float, step: float) -> np.ndarray:
        count = max(2, int(np.floor(length / step)) + 1)
        positions = np.linspace(0.0, length, count)
        if positions[-1] < length:
            positions = np.append(positions, length)
        return np.unique(np.round(positions, 10))

    x_positions = axis_positions(2.0 * inner_radius, truss.cell_size_m) + (center[0] - inner_radius)
    y_positions = axis_positions(2.0 * inner_radius, truss.cell_size_m) + (center[1] - inner_radius)
    z_positions = axis_positions(config.height_m, truss.cell_size_m)

    points: dict[tuple[int, int, int], np.ndarray] = {}
    radial_limit = inner_radius - truss.rod_radius_m * 0.2
    for ix, x in enumerate(x_positions):
        for iy, y in enumerate(y_positions):
            if np.hypot(x - center[0], y - center[1]) > radial_limit:
                continue
            for iz, z in enumerate(z_positions):
                points[(ix, iy, iz)] = np.array([x, y, z], dtype=np.float64)

    if not points:
        return None

    pairs: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    keys = list(points)
    key_set = set(points)

    def add_pair(a: tuple[int, int, int], b: tuple[int, int, int]) -> None:
        if a in key_set and b in key_set:
            pairs.add(tuple(sorted((a, b))))

    for ix, iy, iz in keys:
        if truss.include_horizontal_struts:
            add_pair((ix, iy, iz), (ix + 1, iy, iz))
            add_pair((ix, iy, iz), (ix, iy + 1, iz))
        if truss.include_vertical_struts:
            add_pair((ix, iy, iz), (ix, iy, iz + 1))
        if truss.include_planar_diagonals:
            add_pair((ix, iy, iz), (ix + 1, iy + 1, iz))
            add_pair((ix, iy, iz), (ix + 1, iy - 1, iz))
        if truss.include_diagonal_struts:
            add_pair((ix, iy, iz), (ix + 1, iy + 1, iz + 1))
            add_pair((ix, iy, iz), (ix + 1, iy - 1, iz + 1))
            add_pair((ix, iy, iz), (ix - 1, iy + 1, iz + 1))
            add_pair((ix, iy, iz), (ix - 1, iy - 1, iz + 1))

    if not pairs:
        return None

    cloud = _line_segments_from_pairs(points, pairs)
    return SegmentCloud(
        starts=cloud.starts,
        ends=cloud.ends,
        radius=truss.rod_radius_m,
    )


def build_annular_cylinder_mesh(config: AnnularCylinderConfig) -> AnnularCylinderMesh:
    _validate_config(config)

    extents = np.array(
        [config.outer_diameter_m, config.outer_diameter_m, config.height_m],
        dtype=np.float64,
    )
    nx = max(8, int(round(extents[0] / config.voxel_size_m)))
    ny = max(8, int(round(extents[1] / config.voxel_size_m)))
    nz = max(8, int(round(extents[2] / config.voxel_size_m)))
    spacing = extents / np.array([nx, ny, nz], dtype=np.float64)

    x = (np.arange(nx, dtype=np.float64) + 0.5) * spacing[0]
    y = (np.arange(ny, dtype=np.float64) + 0.5) * spacing[1]
    z = (np.arange(nz, dtype=np.float64) + 0.5) * spacing[2]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    center_xy = extents[:2] / 2.0
    radial_distance = np.sqrt((xx - center_xy[0]) ** 2 + (yy - center_xy[1]) ** 2)
    shell_mask = (radial_distance >= config.inner_radius_m) & (
        radial_distance <= config.outer_radius_m
    )
    fill_mask = np.zeros_like(shell_mask, dtype=bool)
    inner_region_mask = radial_distance <= config.inner_radius_m
    if config.inner_fill_mode == "bone":
        fill_mask = inner_region_mask
    elif config.inner_fill_mode == "truss":
        truss_cloud = build_inner_truss_segment_cloud(config, extents)
        if truss_cloud is not None:
            centers = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
            inner_region = inner_region_mask.ravel()
            if np.any(inner_region):
                fill_hits = np.zeros(len(centers), dtype=bool)
                fill_hits[inner_region] = truss_cloud.contains_points(centers[inner_region])
                fill_mask = fill_hits.reshape(shell_mask.shape)

    occupancy = shell_mask | fill_mask
    active_elements = np.argwhere(occupancy)
    if len(active_elements) == 0:
        raise ValueError("The annular cylinder mesh is empty at the selected voxel size.")

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
    coordinates = indices * spacing

    active_shell = shell_mask[tuple(active_elements.T)]
    material_ids = np.where(active_shell, 0, 1).astype(np.int32)

    hex_mesh = HexMesh(
        coordinates=coordinates,
        connectivity=connectivity,
        material_ids=material_ids,
        descriptor="3_8",
        resolution=nx,
        voxel_size=float(np.min(spacing)),
        active_voxel_count=int(len(active_elements)),
    )
    loaded_top_area_m2 = float(occupancy[:, :, -1].sum() * spacing[0] * spacing[1])
    inner_void_voxel_count = int((radial_distance <= config.inner_radius_m).sum())
    return AnnularCylinderMesh(
        hex_mesh=hex_mesh,
        occupancy=occupancy,
        shell_mask=shell_mask,
        fill_mask=fill_mask,
        active_elements=active_elements,
        spacing=spacing,
        extents=extents,
        grid_shape=(nx, ny, nz),
        loaded_top_area_m2=loaded_top_area_m2,
        inner_void_voxel_count=inner_void_voxel_count,
        shell_element_count=int(shell_mask.sum()),
        fill_element_count=int(fill_mask.sum()),
    )


def _build_problem(
    mesh_data: AnnularCylinderMesh,
    config: AnnularCylinderConfig,
) -> tuple[Problem, Integral, np.ndarray, np.ndarray]:
    mesh = Mesh.from_data(
        "annular_cylinder",
        mesh_data.hex_mesh.coordinates,
        np.zeros(len(mesh_data.hex_mesh.coordinates), dtype=np.int32),
        [mesh_data.hex_mesh.connectivity],
        [mesh_data.hex_mesh.material_ids],
        [mesh_data.hex_mesh.descriptor],
    )
    domain = FEDomain("domain", mesh)
    omega = domain.create_region("Omega", "all")
    shell_region = domain.create_region("Shell", "cells of group 0")
    fill_region = None
    if mesh_data.fill_element_count > 0:
        fill_region = domain.create_region("Fill", "cells of group 1")

    tol = float(mesh_data.spacing[2] * 0.5 + 1e-12)
    bottom_vertices = domain.create_region("BottomVertices", f"vertices in z < {tol}", "vertex")
    top_vertices = domain.create_region(
        "TopVertices",
        f"vertices in z > {config.height_m - tol}",
        "vertex",
    )
    top_surface = domain.create_region(
        "TopSurface",
        f"vertices in z > {config.height_m - tol}",
        "facet",
    )
    if len(bottom_vertices.vertices) == 0 or len(top_vertices.vertices) == 0:
        raise ValueError("The annular cylinder does not touch both loading plates.")

    field = Field.from_args("displacement", np.float64, "vector", omega, approx_order=1)
    u = FieldVariable("u", "unknown", field)
    v = FieldVariable("v", "test", field, primary_var_name="u")

    shell = Material(
        "shell",
        D=stiffness_from_youngpoisson(
            dim=3,
            young=config.material.youngs_modulus_pa,
            poisson=config.material.poisson_ratio,
        ),
    )
    fill = Material(
        "fill",
        D=stiffness_from_youngpoisson(
            dim=3,
            young=config.fill_material.youngs_modulus_pa,
            poisson=config.fill_material.poisson_ratio,
        ),
    )
    traction = Material(
        "traction",
        val=np.array(
            [[0.0], [0.0], [-config.total_force_n / mesh_data.loaded_top_area_m2]],
            dtype=np.float64,
        ),
    )
    in_plane_anchor, twist_anchor, twist_component = _build_bottom_anchor_regions(
        domain,
        mesh_data.hex_mesh.coordinates,
        bottom_vertices.vertices,
    )
    integral = Integral("i", order=2)
    balance_term = Term.new(
        "dw_lin_elastic(shell.D, v, u)",
        integral,
        shell_region,
        shell=shell,
        v=v,
        u=u,
    )
    if fill_region is not None:
        balance_term = balance_term + Term.new(
            "dw_lin_elastic(fill.D, v, u)",
            integral,
            fill_region,
            fill=fill,
            v=v,
            u=u,
        )
    balance = Equation(
        "balance",
        balance_term
        - Term.new(
            "dw_surface_ltr(traction.val, v)",
            integral,
            top_surface,
            traction=traction,
            v=v,
        ),
    )
    problem = Problem("annular_cylinder_compression", equations=Equations([balance]))
    problem.set_bcs(
        ebcs=Conditions(
            [
                EssentialBC("support_bottom", bottom_vertices, {"u.2": 0.0}),
                EssentialBC("anchor_in_plane_origin", in_plane_anchor, {"u.0": 0.0, "u.1": 0.0}),
                EssentialBC(
                    "anchor_bottom_twist",
                    twist_anchor,
                    {twist_component: 0.0},
                ),
            ]
        )
    )
    problem.set_solver(
        Newton(
            {"i_max": 1, "eps_a": 1e-10, "eps_r": 1.0},
            lin_solver=ScipyDirect({}),
        )
    )
    return problem, integral, bottom_vertices.vertices, top_vertices.vertices


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


def _region_from_vertices(
    domain: FEDomain,
    name: str,
    vertex_ids: np.ndarray,
) -> Region:
    region = Region.from_vertices(vertex_ids, domain, name=name, kind="vertex")
    region.finalize()
    region.update_shape()
    domain.regions.append(region)
    return region


def _von_mises_from_cauchy(stress_pa: np.ndarray) -> np.ndarray:
    sxx = stress_pa[:, 0, 0]
    syy = stress_pa[:, 1, 0]
    szz = stress_pa[:, 2, 0]
    sxy = stress_pa[:, 3, 0]
    sxz = stress_pa[:, 4, 0]
    syz = stress_pa[:, 5, 0]
    von_mises_pa = np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + sxz**2 + syz**2)
    )
    return von_mises_pa / 1e6


def solve_annular_cylinder(
    config: AnnularCylinderConfig,
    progress: ProgressCallback | None = None,
) -> _FieldResult:
    _emit_progress(
        progress,
        (
            "Building voxel mesh "
            f"(voxel={config.voxel_size_m * 1e3:.3f} mm, "
            f"OD={config.outer_diameter_m * 1e3:.1f} mm, "
            f"ID={config.inner_diameter_m * 1e3:.1f} mm, "
            f"H={config.height_m * 1e3:.1f} mm)..."
        ),
    )
    mesh = build_annular_cylinder_mesh(config)
    _emit_progress(
        progress,
        (
            "Mesh ready: "
            f"grid={mesh.grid_shape[0]}x{mesh.grid_shape[1]}x{mesh.grid_shape[2]}, "
            f"elements={mesh.hex_mesh.active_element_count}, "
            f"nodes={mesh.hex_mesh.active_node_count}"
        ),
    )
    with quiet_sfepy_output():
        _emit_progress(progress, "Assembling finite element problem...")
        problem, integral, bottom_nodes, top_nodes = _build_problem(mesh, config)
        _emit_progress(progress, "Solving linear system...")
        variables = problem.solve(save_results=False, verbose=False)

        _emit_progress(progress, "Evaluating stress field...")
        stress = np.zeros((mesh.hex_mesh.active_element_count, 6, 1), dtype=np.float64)
        shell_mask = mesh.hex_mesh.material_ids == 0
        stress[shell_mask] = problem.evaluate(
            "ev_cauchy_stress.i.Shell(shell.D, u)",
            mode="el_avg",
            copy_materials=False,
            verbose=False,
            integrals={"i": integral},
        ).reshape(-1, 6, 1)
        if np.any(~shell_mask):
            stress[~shell_mask] = problem.evaluate(
                "ev_cauchy_stress.i.Fill(fill.D, u)",
                mode="el_avg",
                copy_materials=False,
                verbose=False,
                integrals={"i": integral},
            ).reshape(-1, 6, 1)
    nodal_displacements = variables.get_state_parts()["u"].reshape(
        (-1, 3), order="C"
    )
    element_centers = mesh.hex_mesh.coordinates[mesh.hex_mesh.connectivity].mean(axis=1)
    element_displacements = nodal_displacements[mesh.hex_mesh.connectivity].mean(axis=1)

    displacement_magnitude = np.linalg.norm(nodal_displacements, axis=1)
    top_displacement = np.abs(nodal_displacements[top_nodes, 2])
    axial_stress_mpa = stress[:, 2, 0] / 1e6
    von_mises_mpa = _von_mises_from_cauchy(stress)

    mean_top_displacement_mm = float(np.mean(top_displacement) * 1e3)
    result = AnnularCylinderResult(
        outer_diameter_mm=config.outer_diameter_m * 1e3,
        inner_diameter_mm=config.inner_diameter_m * 1e3,
        height_mm=config.height_m * 1e3,
        target_voxel_size_mm=config.voxel_size_m * 1e3,
        actual_voxel_size_x_mm=mesh.spacing[0] * 1e3,
        actual_voxel_size_y_mm=mesh.spacing[1] * 1e3,
        actual_voxel_size_z_mm=mesh.spacing[2] * 1e3,
        grid_shape_xyz=mesh.grid_shape,
        applied_force_n=config.total_force_n,
        applied_pressure_mpa=(config.total_force_n / mesh.loaded_top_area_m2) / 1e6,
        loaded_top_area_mm2=mesh.loaded_top_area_m2 * 1e6,
        projected_ring_area_mm2=config.top_area_m2 * 1e6,
        material_name=config.material.name,
        youngs_modulus_gpa=config.material.youngs_modulus_gpa,
        poisson_ratio=config.material.poisson_ratio,
        inner_fill_mode=config.inner_fill_mode,
        fill_material_name=config.fill_material.name,
        fill_youngs_modulus_gpa=config.fill_material.youngs_modulus_gpa,
        fill_poisson_ratio=config.fill_material.poisson_ratio,
        fill_volume_fraction=float(
            mesh.fill_element_count / max(mesh.inner_void_voxel_count, 1)
        ),
        total_volume_fraction=float(mesh.occupancy.mean()),
        active_element_count=mesh.hex_mesh.active_element_count,
        shell_element_count=mesh.shell_element_count,
        fill_element_count=mesh.fill_element_count,
        active_node_count=mesh.hex_mesh.active_node_count,
        top_contact_nodes=int(len(top_nodes)),
        bottom_contact_nodes=int(len(bottom_nodes)),
        max_displacement_mm=float(np.max(displacement_magnitude) * 1e3),
        mean_top_displacement_mm=mean_top_displacement_mm,
        max_von_mises_mpa=float(np.max(von_mises_mpa)),
        max_axial_stress_mpa=float(np.max(np.abs(axial_stress_mpa))),
        mean_axial_stress_mpa=float(np.mean(axial_stress_mpa)),
        apparent_stiffness_kn_per_mm=float(
            config.total_force_n / max(mean_top_displacement_mm, 1e-12) / 1e3
        ),
    )
    _emit_progress(
        progress,
        (
            "Post-processing complete: "
            f"max_disp={result.max_displacement_mm:.4f} mm, "
            f"max_vm={result.max_von_mises_mpa:.2f} MPa"
        ),
    )
    return _FieldResult(
        result=result,
        nodal_displacements=nodal_displacements,
        element_centers=element_centers,
        element_displacements=element_displacements,
        element_von_mises_mpa=von_mises_mpa,
        element_axial_stress_mpa=axial_stress_mpa,
        mesh=mesh,
    )


def _surface_element_mask(occupancy: np.ndarray) -> np.ndarray:
    surface = np.zeros_like(occupancy, dtype=bool)
    for axis in range(3):
        for shift in (-1, 1):
            shifted = np.roll(occupancy, shift=shift, axis=axis)
            if shift == -1:
                index = [slice(None)] * 3
                index[axis] = -1
                shifted[tuple(index)] = False
            else:
                index = [slice(None)] * 3
                index[axis] = 0
                shifted[tuple(index)] = False
            surface |= occupancy & ~shifted
    return surface


def render_annular_cylinder_result(
    field_result: _FieldResult,
    output_path: Path,
) -> Path:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    result = field_result.result
    mesh = field_result.mesh

    surface_mask = _surface_element_mask(mesh.occupancy)
    surface_lookup = np.zeros_like(mesh.occupancy, dtype=bool)
    surface_lookup[tuple(mesh.active_elements.T)] = surface_mask[tuple(mesh.active_elements.T)]
    visible = surface_lookup[tuple(mesh.active_elements.T)]

    centers = field_result.element_centers[visible]
    displacements = field_result.element_displacements[visible]
    von_mises = field_result.element_von_mises_mpa[visible]

    max_disp = float(np.max(np.linalg.norm(field_result.nodal_displacements, axis=1)))
    deform_scale = 1.0 if max_disp == 0.0 else min(120.0, 0.2 * mesh.extents[2] / max_disp)
    deformed_centers = centers + displacements * deform_scale

    mid_z = mesh.extents[2] * 0.5
    slice_tol = mesh.spacing[2] * 0.75
    slice_mask = np.abs(field_result.element_centers[:, 2] - mid_z) <= slice_tol
    slice_centers = field_result.element_centers[slice_mask]
    slice_von_mises = field_result.element_von_mises_mpa[slice_mask]

    norm = Normalize(vmin=float(np.min(von_mises)), vmax=float(np.max(von_mises)))
    figure = plt.figure(figsize=(15, 5.4), constrained_layout=True)
    ax3d = figure.add_subplot(1, 3, 1, projection="3d")
    ax3d.scatter(
        deformed_centers[:, 0] * 1e3,
        deformed_centers[:, 1] * 1e3,
        deformed_centers[:, 2] * 1e3,
        c=von_mises,
        cmap="viridis",
        norm=norm,
        s=11,
        linewidths=0.0,
    )
    ax3d.set_title("Deformed surface")
    ax3d.set_xlabel("x (mm)")
    ax3d.set_ylabel("y (mm)")
    ax3d.set_zlabel("z (mm)")
    ax3d.set_box_aspect(
        (
            mesh.extents[0],
            mesh.extents[1],
            mesh.extents[2],
        )
    )

    ax_top = figure.add_subplot(1, 3, 2)
    scatter = ax_top.scatter(
        slice_centers[:, 0] * 1e3,
        slice_centers[:, 1] * 1e3,
        c=slice_von_mises,
        cmap="viridis",
        norm=norm,
        s=28,
    )
    ax_top.set_title("Mid-height stress slice")
    ax_top.set_xlabel("x (mm)")
    ax_top.set_ylabel("y (mm)")
    ax_top.set_aspect("equal", adjustable="box")

    ax_side = figure.add_subplot(1, 3, 3)
    axial_disp_mm = -field_result.element_displacements[:, 2] * 1e3
    side = ax_side.scatter(
        field_result.element_centers[:, 0] * 1e3,
        field_result.element_centers[:, 2] * 1e3,
        c=axial_disp_mm,
        cmap="magma",
        s=16,
    )
    ax_side.set_title("Axial displacement")
    ax_side.set_xlabel("x (mm)")
    ax_side.set_ylabel("z (mm)")
    ax_side.set_aspect("equal", adjustable="box")

    cbar = figure.colorbar(scatter, ax=[ax3d, ax_top], shrink=0.82, pad=0.03)
    cbar.set_label("von Mises stress (MPa)")
    disp_bar = figure.colorbar(side, ax=ax_side, shrink=0.82, pad=0.03)
    disp_bar.set_label("Axial displacement (mm)")

    figure.suptitle(
        "Annular cylinder compression\n"
        f"{result.outer_diameter_mm:.0f} mm OD, {result.inner_diameter_mm:.0f} mm ID, "
        f"{result.height_mm:.0f} mm height, {result.applied_force_n:.0f} N load"
        + (f" with {result.inner_fill_mode} fill" if result.inner_fill_mode != "empty" else ""),
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_annular_cylinder_npz_payload(
    config: AnnularCylinderConfig,
    mesh: AnnularCylinderMesh,
) -> dict[str, np.ndarray]:
    nx, ny, nz = mesh.grid_shape
    voxels = mesh.occupancy.astype(np.uint8)
    density_milli = voxels.astype(np.uint16) * np.uint16(1000)

    material_id = np.full(mesh.grid_shape, -1, dtype=np.int8)
    material_id[mesh.shell_mask] = 0
    material_id[mesh.fill_mask] = 1

    center = np.array(
        [(nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0],
        dtype=np.float32,
    )
    outer_radius_voxels = np.float32(config.outer_radius_m / mesh.spacing[0])
    inner_radius_voxels = np.float32(config.inner_radius_m / mesh.spacing[0])

    return {
        "voxels": voxels,
        "density_milli": density_milli,
        "material_id": material_id,
        "xy_size": np.array(nx, dtype=np.int32),
        "z_size": np.array(nz, dtype=np.int32),
        "grid_shape_xyz": np.array(mesh.grid_shape, dtype=np.int32),
        "center": center,
        "outer_radius": np.array(outer_radius_voxels, dtype=np.float32),
        "inner_radius": np.array(inner_radius_voxels, dtype=np.float32),
        "target_voxel_size_m": np.array(config.voxel_size_m, dtype=np.float32),
        "voxel_size_m": np.array(float(np.min(mesh.spacing)), dtype=np.float32),
        "voxel_size_xyz_m": mesh.spacing.astype(np.float32),
        "outer_diameter_m": np.array(config.outer_diameter_m, dtype=np.float32),
        "inner_diameter_m": np.array(config.inner_diameter_m, dtype=np.float32),
        "height_m": np.array(config.height_m, dtype=np.float32),
        "load_n": np.array(config.total_force_n, dtype=np.float32),
        "shape_name": np.array("annular_cylinder"),
        "result_type": np.array("fem_annular_cylinder_density"),
        "density_kind": np.array("binary_occupancy"),
        "density_precision": np.array(3, dtype=np.int32),
        "density_min_nonzero": np.array(1.0, dtype=np.float32),
        "density_max": np.array(1.0, dtype=np.float32),
        "material_name": np.array(config.material.name),
        "fill_material_name": np.array(config.fill_material.name),
        "inner_fill_mode": np.array(config.inner_fill_mode),
    }


def write_annular_cylinder_npz(
    config: AnnularCylinderConfig,
    mesh: AnnularCylinderMesh,
    output_path: Path,
) -> Path:
    payload = build_annular_cylinder_npz_payload(config, mesh)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


def write_annular_cylinder_report(
    config: AnnularCylinderConfig,
    result: AnnularCylinderResult,
    output_path: Path,
) -> Path:
    payload = {
        "geometry": {
            "outer_diameter_mm": result.outer_diameter_mm,
            "inner_diameter_mm": result.inner_diameter_mm,
            "height_mm": result.height_mm,
        },
        "load": {
            "applied_force_n": result.applied_force_n,
            "loaded_top_area_mm2": result.loaded_top_area_mm2,
            "projected_ring_area_mm2": result.projected_ring_area_mm2,
            "applied_pressure_mpa": result.applied_pressure_mpa,
        },
        "material": {
            "name": result.material_name,
            "youngs_modulus_gpa": result.youngs_modulus_gpa,
            "poisson_ratio": result.poisson_ratio,
        },
        "inner_fill": {
            "mode": config.inner_fill_mode,
            "material_name": config.fill_material.name,
            "youngs_modulus_gpa": config.fill_material.youngs_modulus_gpa,
            "poisson_ratio": config.fill_material.poisson_ratio,
        },
        "truss_infill": {
            "enabled": config.inner_fill_mode == "truss",
            "cell_size_mm": config.truss_infill.cell_size_m * 1e3,
            "rod_radius_mm": config.truss_infill.rod_radius_m * 1e3,
            "include_vertical_struts": config.truss_infill.include_vertical_struts,
            "include_horizontal_struts": config.truss_infill.include_horizontal_struts,
            "include_planar_diagonals": config.truss_infill.include_planar_diagonals,
            "include_diagonal_struts": config.truss_infill.include_diagonal_struts,
        },
        "solver": {
            "backend": "sfepy",
            "target_voxel_size_mm": result.target_voxel_size_mm,
            "actual_voxel_size_xyz_mm": [
                result.actual_voxel_size_x_mm,
                result.actual_voxel_size_y_mm,
                result.actual_voxel_size_z_mm,
            ],
            "grid_shape_xyz": list(result.grid_shape_xyz),
            "active_element_count": result.active_element_count,
            "active_node_count": result.active_node_count,
            "top_contact_nodes": result.top_contact_nodes,
            "bottom_contact_nodes": result.bottom_contact_nodes,
            "output_npz": str(config.output_npz),
        },
        "results": asdict(result),
        "assumptions": [
            "The default cylinder height is set to 20 mm to keep the specimen a bit shorter than the original 30 mm setup.",
            "A linear elastic isotropic material model is used.",
            "The bottom face is supported only in the axial direction, with two bottom reference nodes adding minimal in-plane constraints to remove rigid-body drift while the top face receives a uniform compressive traction.",
            "The default inner region is modeled as a continuous low-modulus fill so the center behaves like equivalent bone graft rather than a finely resolved lattice.",
            "The exported NPZ stores a binary occupancy density field so downstream topology-sampling tools can consume the same 3D grid directly.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def run_annular_cylinder_demo(
    config: AnnularCylinderConfig,
    progress: ProgressCallback | None = None,
) -> AnnularCylinderRunSummary:
    field_result = solve_annular_cylinder(config, progress=progress)
    _emit_progress(progress, f"Rendering result image to {config.output_image}...")
    image_path = render_annular_cylinder_result(field_result, config.output_image)
    _emit_progress(progress, f"Writing analysis JSON to {config.output_json}...")
    json_path = write_annular_cylinder_report(config, field_result.result, config.output_json)
    _emit_progress(progress, f"Writing density NPZ to {config.output_npz}...")
    npz_path = write_annular_cylinder_npz(config, field_result.mesh, config.output_npz)
    _emit_progress(progress, "Analysis outputs are ready.")
    return AnnularCylinderRunSummary(
        result=field_result.result,
        image_path=image_path,
        json_path=json_path,
        npz_path=npz_path,
    )
