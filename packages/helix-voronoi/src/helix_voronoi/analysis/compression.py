from __future__ import annotations

from dataclasses import dataclass

from helix_voronoi.analysis.config import CompressionConfig, MaterialConfig, StructureStyle
from helix_voronoi.analysis.elasticity import solve_linear_elasticity
from helix_voronoi.analysis.voxel import VoxelGrid, build_hex_mesh


@dataclass(frozen=True)
class CompressionResult:
    style: StructureStyle
    resolution: int
    effective_modulus_gpa: float
    reaction_force_n: float
    active_voxel_count: int
    active_element_count: int
    active_node_count: int
    top_contact_nodes: int
    bottom_contact_nodes: int
    solid_volume_fraction: float


def solve_compression(
    style: StructureStyle,
    grid: VoxelGrid,
    material: MaterialConfig,
    compression: CompressionConfig,
) -> CompressionResult:
    mesh_data = build_hex_mesh(grid)
    system = solve_linear_elasticity(mesh_data, material, compression)
    effective_modulus_pa = system.reaction_force_n / compression.applied_strain

    return CompressionResult(
        style=style,
        resolution=grid.resolution,
        effective_modulus_gpa=effective_modulus_pa / 1e9,
        reaction_force_n=system.reaction_force_n,
        active_voxel_count=grid.active_voxel_count,
        active_element_count=system.active_element_count,
        active_node_count=system.active_node_count,
        top_contact_nodes=system.top_contact_nodes,
        bottom_contact_nodes=system.bottom_contact_nodes,
        solid_volume_fraction=system.solid_volume,
    )
