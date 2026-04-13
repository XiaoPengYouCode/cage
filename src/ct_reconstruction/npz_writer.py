from __future__ import annotations

from pathlib import Path

import numpy as np


def build_voxel_npz_payload(
    occupancy: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    shape_name: str = "lumbar_vertebra",
) -> dict[str, np.ndarray]:
    """Build NPZ payload compatible with the fem_analysis density schema."""
    nx, ny, nz = occupancy.shape
    voxels = occupancy.astype(np.uint8)
    return {
        "voxels": voxels,
        "density_milli": (voxels.astype(np.uint16) * np.uint16(1000)),
        "material_id": np.where(occupancy, np.int8(0), np.int8(-1)),
        "grid_shape_xyz": np.array([nx, ny, nz], dtype=np.int32),
        "origin_m": (origin / 1e3).astype(np.float32),
        "voxel_size_m": np.float32(spacing.min() / 1e3),
        "voxel_size_xyz_m": (spacing / 1e3).astype(np.float32),
        "shape_name": np.array(shape_name),
        "result_type": np.array("stl_voxelized"),
        "density_kind": np.array("binary_occupancy"),
        "density_precision": np.int32(3),
        "density_min_nonzero": np.float32(1.0),
        "density_max": np.float32(1.0),
    }


def write_npz(payload: dict[str, np.ndarray], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path
