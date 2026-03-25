from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cage.analysis.geometry import SegmentCloud


@dataclass(frozen=True)
class VoxelGrid:
    resolution: int
    voxel_size: float
    occupancy: np.ndarray

    @property
    def active_voxel_count(self) -> int:
        return int(self.occupancy.sum())


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

    @property
    def cell_volume(self) -> float:
        return self.voxel_size**3


def voxel_centers(resolution: int) -> np.ndarray:
    coords = (np.arange(resolution, dtype=float) + 0.5) / resolution
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def voxelize_segment_cloud(
    geometry: SegmentCloud,
    resolution: int,
    chunk_size: int = 100_000,
) -> VoxelGrid:
    centers = voxel_centers(resolution)
    occupancy = geometry.contains_points(centers, chunk_size=chunk_size).reshape(
        (resolution, resolution, resolution)
    )
    return VoxelGrid(
        resolution=resolution,
        voxel_size=1.0 / resolution,
        occupancy=occupancy,
    )


def build_hex_mesh(grid: VoxelGrid) -> HexMesh:
    active_elements = np.argwhere(grid.occupancy)
    if len(active_elements) == 0:
        raise ValueError("No active voxels found for the selected geometry.")

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
    node_shape = (grid.resolution + 1, grid.resolution + 1, grid.resolution + 1)
    node_triplets = active_elements[:, None, :] + local_offsets[None, :, :]
    global_node_ids = np.ravel_multi_index(node_triplets.reshape(-1, 3).T, node_shape)
    unique_nodes, inverse = np.unique(global_node_ids, return_inverse=True)
    connectivity = inverse.reshape(-1, 8).astype(np.int32)
    coordinates = np.array(np.unravel_index(unique_nodes, node_shape), dtype=float).T / grid.resolution

    return HexMesh(
        coordinates=coordinates,
        connectivity=connectivity,
        material_ids=np.zeros(len(connectivity), dtype=np.int32),
        descriptor="3_8",
        resolution=grid.resolution,
        voxel_size=grid.voxel_size,
        active_voxel_count=grid.active_voxel_count,
    )
