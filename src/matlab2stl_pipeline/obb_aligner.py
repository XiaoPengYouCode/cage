"""Steps 3, 4, 5 — OBB fitting and density field alignment.

Step 3: Fit an Oriented Bounding Box (OBB) to the occupied voxels using PCA.
Step 4: Compute the probability density field from pseudo-density.
Step 5: Resample the density field into the OBB-aligned (axis-aligned) frame
        using scipy.ndimage.affine_transform.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.ndimage


# ---------------------------------------------------------------------------
# Step 3 — OBB fitting
# ---------------------------------------------------------------------------

def fit_obb(npz_path: Path, output_path: Path) -> dict:
    """Fit an Oriented Bounding Box to the occupied voxels via PCA.

    Returns a dict (also saved to *output_path*) with:
        center_voxel  : float32 (3,)  – OBB centre in voxel index space
        axes          : float32 (3,3) – rows are the three PCA axes (unit vectors)
                                        in voxel index space
        half_extents_voxel : float32 (3,) – half-lengths along each axis (voxels)
        half_extents_m     : float32 (3,) – half-lengths in metres
        rotation_matrix    : float32 (3,3) – same as axes (kept for explicitness)
        voxel_size_xyz_m   : float32 (3,) – copied from source NPZ
    """
    data = np.load(str(npz_path))
    voxel_size_xyz_m: np.ndarray = data["voxel_size_xyz_m"]  # (3,) float32

    # Use density_milli > 0 as the point cloud — this captures the full
    # extent of the pseudo-density field, not just the high-density binary mask,
    # which is important for topology-optimization results where most values are
    # low but the geometry spans the full domain.
    density_milli: np.ndarray = data["density_milli"]
    idx = np.argwhere(density_milli > 0).astype(np.float64)  # (M, 3)

    if idx.shape[0] < 4:
        raise ValueError("Too few occupied voxels to fit an OBB.")

    center = idx.mean(axis=0)
    idx_centered = idx - center

    # PCA — covariance matrix
    cov = (idx_centered.T @ idx_centered) / len(idx_centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending eigenvalues; sort descending (largest variance first)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order].T.astype(np.float32)   # (3, 3), rows = axes

    # Ensure axes forms a proper rotation matrix (det = +1), not a reflection.
    # If det == -1, flipping one axis restores the right-hand orientation.
    if np.linalg.det(axes) < 0:
        axes[2] = -axes[2]

    # Project all points onto each axis to find half-extents
    projections = idx_centered @ axes.T    # (M, 3)
    half_extents_voxel = (projections.max(axis=0) - projections.min(axis=0)) / 2.0
    half_extents_voxel = half_extents_voxel.astype(np.float32)
    half_extents_m = half_extents_voxel * voxel_size_xyz_m

    payload = {
        "center_voxel": center.astype(np.float32),
        "axes": axes,
        "half_extents_voxel": half_extents_voxel,
        "half_extents_m": half_extents_m,
        "rotation_matrix": axes,   # axes IS the rotation matrix (OBB→world)
        "voxel_size_xyz_m": voxel_size_xyz_m,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **payload)
    return payload


# ---------------------------------------------------------------------------
# Step 4 — Probability density field
# ---------------------------------------------------------------------------

def compute_probability_field(
    density_milli: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Convert pseudo-density (uint16 0..1000) to a normalised probability field.

    prob[i] = density[i]^gamma / sum(density^gamma)
    Returns float32 array of same shape, sum == 1 (over non-zero voxels).
    """
    density = density_milli.astype(np.float32) / 1000.0
    powered = np.where(density > 0, density ** gamma, 0.0)
    total = powered.sum()
    if total == 0:
        raise ValueError("All densities are zero — cannot compute probability field.")
    return (powered / total).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5 — Align density field to OBB frame
# ---------------------------------------------------------------------------

def align_density(
    npz_path: Path,
    obb_path: Path,
    output_path: Path,
    gamma: float = 1.0,
) -> dict:
    """Resample density into an axis-aligned box matching the OBB dimensions.

    The new grid has the same physical voxel size as the original but the
    coordinate axes are aligned with the world X/Y/Z axes.  The physical
    size of the new grid equals the OBB extent (2 * half_extents_m).

    Parameters
    ----------
    npz_path : source NPZ (from Step 1)
    obb_path : OBB NPZ (from Step 3)
    output_path : destination NPZ
    gamma : power for probability field (Step 4)
    """
    src = np.load(str(npz_path))
    obb = np.load(str(obb_path))

    density_milli_src: np.ndarray = src["density_milli"]    # (nx, ny, nz) uint16
    voxel_size_xyz_m: np.ndarray = src["voxel_size_xyz_m"]  # (3,) float32

    center_voxel: np.ndarray = obb["center_voxel"]          # (3,) float32
    axes: np.ndarray = obb["axes"]                           # (3,3) float32 row=axis
    half_extents_voxel: np.ndarray = obb["half_extents_voxel"]  # (3,) float32

    # New grid shape: ceil of OBB extent in each axis direction (voxels)
    new_shape = tuple(int(np.ceil(2 * h)) + 1 for h in half_extents_voxel)
    new_center = np.array([(s - 1) / 2.0 for s in new_shape], dtype=np.float64)

    # affine_transform maps new-grid index → old-grid index via:
    #   old_idx = R^T · (new_idx - new_center) + center_voxel
    # scipy convention: output[o] = input[matrix @ o + offset]
    # matrix = axes^T  (each column is an OBB axis in original voxel space)
    matrix = axes.T.astype(np.float64)          # (3,3)
    offset = center_voxel - matrix @ new_center  # (3,)

    density_src_f = density_milli_src.astype(np.float32)
    density_aligned_f = scipy.ndimage.affine_transform(
        density_src_f,
        matrix=matrix,
        offset=offset,
        output_shape=new_shape,
        order=1,           # trilinear interpolation
        mode="constant",
        cval=0.0,
    )
    density_milli_aligned = np.clip(density_aligned_f, 0, 1000).astype(np.uint16)
    voxels_aligned = (density_milli_aligned > 500).astype(np.uint8)

    # Probability field (Step 4 result, in aligned frame)
    prob_field = compute_probability_field(density_milli_aligned, gamma)

    nx, ny, nz = new_shape
    origin_m = np.zeros(3, dtype=np.float32)  # aligned box starts at origin

    # ------------------------------------------------------------------
    # Inverse transform: aligned physical coords → original physical coords
    #
    # The alignment maps:  old_idx = axes.T @ (new_idx - new_center) + center_voxel
    # In physical (m) space with voxel_size v:
    #   p_aligned_m = new_idx * v   (origin at 0)
    #   p_orig_m    = axes.T @ p_aligned_m + t_restore_m
    # where:
    #   t_restore_m = center_voxel * v - axes.T @ (new_center * v)
    #
    # Stored as:
    #   restore_R : float64 (3,3) — rotation to apply to aligned vertices
    #   restore_t : float64 (3,)  — translation to add after rotation
    # ------------------------------------------------------------------
    restore_R = axes.T.astype(np.float64)  # aligned→original rotation
    new_center_m = new_center * voxel_size_xyz_m.astype(np.float64)
    center_orig_m = center_voxel.astype(np.float64) * voxel_size_xyz_m.astype(np.float64)
    restore_t = center_orig_m - restore_R @ new_center_m

    payload = {
        "density_milli": density_milli_aligned,
        "voxels": voxels_aligned,
        "probability_field": prob_field,
        "grid_shape_xyz": np.array([nx, ny, nz], dtype=np.int32),
        "origin_m": origin_m,
        "voxel_size_xyz_m": voxel_size_xyz_m,
        "shape_name": np.array("681_aligned", dtype=object),
        "result_type": np.array("obb_aligned", dtype=object),
        "density_kind": np.array("pseudo_density", dtype=object),
        "gamma": np.float32(gamma),
        # Inverse transform back to original pose
        "restore_R": restore_R,
        "restore_t": restore_t,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **payload)
    return payload
