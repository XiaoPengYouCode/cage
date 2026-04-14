"""Step 1 — Read 681.mat and convert to standard NPZ voxel format.

The .mat file contains a single 3-D float64 array named ``cage_3D1``
with shape (nx, ny, nz) where each element is a pseudo-density in [0, 1].
Each voxel corresponds to a physical cube of side 0.4 mm (400 µm).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io


VOXEL_SIZE_MM: float = 0.4  # 400 µm per voxel (physical)
MAT_VARIABLE: str = "cage_3D1"


def load_mat_to_npz(
    mat_path: Path,
    output_path: Path,
    voxel_size_mm: float = VOXEL_SIZE_MM,
) -> dict:
    """Load 681.mat and write a standard NPZ voxel payload.

    Parameters
    ----------
    mat_path:
        Path to the source ``.mat`` file.
    output_path:
        Destination ``.npz`` file.
    voxel_size_mm:
        Physical edge length of each voxel in millimetres.

    Returns
    -------
    dict with the same keys written to the NPZ file.
    """
    mat = scipy.io.loadmat(str(mat_path))

    # Find the density variable — prefer cage_3D1, fall back to first non-meta key
    variable_name = MAT_VARIABLE
    if variable_name not in mat:
        candidates = [k for k in mat if not k.startswith("_")]
        if not candidates:
            raise KeyError(f"No data variable found in {mat_path}")
        variable_name = candidates[0]

    density_raw: np.ndarray = mat[variable_name].astype(np.float32)

    nx, ny, nz = density_raw.shape

    # Clip and convert to uint16 (0–1000)
    density_clipped = np.clip(density_raw, 0.0, 1.0)
    density_milli = (density_clipped * 1000).astype(np.uint16)

    # Binary occupancy: any non-zero density is part of the design domain.
    # 0 = void (outside), >0 = material (including low intermediate densities).
    voxels = (density_clipped > 0.0).astype(np.uint8)

    voxel_size_m = voxel_size_mm / 1e3
    origin_m = np.zeros(3, dtype=np.float32)
    voxel_size_xyz_m = np.array([voxel_size_m, voxel_size_m, voxel_size_m], dtype=np.float32)
    grid_shape_xyz = np.array([nx, ny, nz], dtype=np.int32)

    payload = {
        "density_milli": density_milli,
        "voxels": voxels,
        "grid_shape_xyz": grid_shape_xyz,
        "origin_m": origin_m,
        "voxel_size_xyz_m": voxel_size_xyz_m,
        "shape_name": np.array("681", dtype=object),
        "result_type": np.array("mat_imported", dtype=object),
        "density_kind": np.array("pseudo_density", dtype=object),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **payload)
    return payload
