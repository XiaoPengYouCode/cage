"""Steps 9 & 10 — Voronoi skeleton voxelization and Marching Cubes mesh export.

Step 9 — Voxelize + dilate
--------------------------
Each Voronoi edge is rasterized as a line of voxels in a fine-resolution grid,
then binary-dilated with a spherical structuring element to produce solid rods.
Output: voxel occupancy grid (NPZ only — no mesh at this stage).

Step 10 — Marching Cubes mesh export
-------------------------------------
The dilated voxel grid from Step 9 is passed through Marching Cubes to extract
a smooth isosurface.  Output: GLB (viewer) + binary STL (deliverable).

Resolution strategy
-------------------
The original voxels are 400 µm.  For scaffold detail we want ~40 µm voxels
(configurable).  The fine grid is derived by subdividing each original voxel
into ``subdivision`` sub-voxels per axis.

    fine_voxel_size_mm = 0.4 / subdivision
    fine_grid_shape    = original_grid_shape * subdivision

Default subdivision = 10 → 40 µm.
The dilation radius is also expressed in fine voxels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.ndimage


def _rasterize_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    grid_shape: tuple[int, int, int],
    out: np.ndarray,
) -> None:
    """Mark voxels along the segment, clipping to grid bounds (legacy, unused)."""
    _rasterize_segment_unbounded(p0, p1, grid_shape, out, clip=True)


def _rasterize_segment_unbounded(
    p0: np.ndarray,
    p1: np.ndarray,
    grid_shape: tuple[int, int, int],
    out: np.ndarray,
    clip: bool = False,
) -> None:
    """Mark voxels along the segment from p0 to p1 in *out* (in-place).

    Uses uniform sampling at 2 samples/voxel to avoid gaps.
    p0, p1 are in fine-voxel index coordinates (may be outside grid when
    edges touch the box boundary and the grid is padded).

    When clip=False, samples that fall outside the grid are silently skipped
    only if they truly exceed the allocated array — with padding this should
    never happen for Voronoi edges.
    """
    delta = p1 - p0
    length = np.linalg.norm(delta)
    if length < 1e-9:
        ix, iy, iz = int(np.round(p0[0])), int(np.round(p0[1])), int(np.round(p0[2]))
        if 0 <= ix < grid_shape[0] and 0 <= iy < grid_shape[1] and 0 <= iz < grid_shape[2]:
            out[ix, iy, iz] = True
        return

    n_steps = max(2, int(np.ceil(length * 2)))
    for t in np.linspace(0.0, 1.0, n_steps):
        pt = p0 + t * delta
        ix = int(np.round(pt[0]))
        iy = int(np.round(pt[1]))
        iz = int(np.round(pt[2]))
        if 0 <= ix < grid_shape[0] and 0 <= iy < grid_shape[1] and 0 <= iz < grid_shape[2]:
            out[ix, iy, iz] = True


def _make_sphere_struct(radius_voxels: float) -> np.ndarray:
    """Return a boolean spherical structuring element."""
    r = int(np.ceil(radius_voxels))
    size = 2 * r + 1
    struct = np.zeros((size, size, size), dtype=bool)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if (i - r) ** 2 + (j - r) ** 2 + (k - r) ** 2 <= radius_voxels ** 2:
                    struct[i, j, k] = True
    return struct


# ---------------------------------------------------------------------------
# Step 9 — Voxelize + dilate (pure voxel output, no mesh)
# ---------------------------------------------------------------------------

def voxelize_skeleton(
    edges_npz_path: Path,
    aligned_npz_path: Path,
    output_npz_path: Path,
    subdivision: int = 10,
    dilation_radius_fine_voxels: float = 3.0,
) -> np.ndarray:
    """Rasterize Voronoi edges and dilate to form solid rods.

    The bounding box is respected during Voronoi construction and edge
    extraction (Steps 7–8).  From this step onward the box constraint is
    lifted: edges that lie on box faces/edges are rasterized and then dilated
    freely beyond the box boundary so they form full solid rods rather than
    flat-cut stubs.

    Implementation: we allocate a grid that is padded by *dilation_radius*
    fine voxels on every side, rasterize all edges into the padded grid
    (offsetting coordinates by the padding amount), run binary dilation on the
    full padded grid, then save the result as-is (no clipping back to the box).

    Parameters
    ----------
    edges_npz_path              : output of Step 8 (``edges`` array, coarse voxel coords)
    aligned_npz_path            : aligned density NPZ for grid dimensions & voxel size
    output_npz_path             : destination NPZ (voxel occupancy grid only)
    subdivision                 : fine-grid subdivision factor (default 10 → 40 µm voxels)
    dilation_radius_fine_voxels : sphere radius for morphological dilation

    Returns
    -------
    dilated : uint8 (padded_fnx, padded_fny, padded_fnz) occupancy grid
    """
    edges_data = np.load(str(edges_npz_path))
    aligned_data = np.load(str(aligned_npz_path))

    edges: np.ndarray = edges_data["edges"]                       # (E, 2, 3) coarse voxel coords
    grid_shape_coarse = aligned_data["grid_shape_xyz"].tolist()   # [nx, ny, nz]
    voxel_size_xyz_m: np.ndarray = aligned_data["voxel_size_xyz_m"]

    fine_voxel_size_m = voxel_size_xyz_m / subdivision

    # Padding: enough room for dilation to expand freely beyond the box
    pad = int(np.ceil(dilation_radius_fine_voxels)) + 1

    fnx = grid_shape_coarse[0] * subdivision + 2 * pad
    fny = grid_shape_coarse[1] * subdivision + 2 * pad
    fnz = grid_shape_coarse[2] * subdivision + 2 * pad
    fine_shape = (fnx, fny, fnz)

    print(f"  Fine grid (with pad={pad}): {fnx}×{fny}×{fnz}  "
          f"({fine_voxel_size_m[0]*1e3:.3f} mm voxels)")

    skeleton = np.zeros(fine_shape, dtype=bool)
    scale = float(subdivision)
    for edge in edges:
        p0_fine = edge[0].astype(np.float64) * scale + pad
        p1_fine = edge[1].astype(np.float64) * scale + pad
        _rasterize_segment_unbounded(p0_fine, p1_fine, fine_shape, skeleton)

    print(f"  Rasterized {len(edges)} edges → {skeleton.sum()} lit voxels")

    struct = _make_sphere_struct(dilation_radius_fine_voxels)
    dilated = scipy.ndimage.binary_dilation(
        skeleton, structure=struct, border_value=0,
    ).astype(np.uint8)
    print(f"  After dilation (r={dilation_radius_fine_voxels} fine voxels): "
          f"{dilated.sum()} voxels")

    # Origin shifts back by pad voxels (the padded grid starts before the box)
    origin_m = aligned_data["origin_m"] - pad * fine_voxel_size_m

    payload = {
        "voxels": dilated,
        "grid_shape_xyz": np.array([fnx, fny, fnz], dtype=np.int32),
        "origin_m": origin_m.astype(np.float32),
        "voxel_size_xyz_m": fine_voxel_size_m.astype(np.float32),
        "subdivision": np.int32(subdivision),
        "dilation_radius_fine_voxels": np.float32(dilation_radius_fine_voxels),
        "pad_fine_voxels": np.int32(pad),
    }
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_npz_path), **payload)
    return dilated


# ---------------------------------------------------------------------------
# Step 10 — Marching Cubes mesh export (GLB + STL)
# ---------------------------------------------------------------------------

def mesh_from_voxels(
    skeleton_npz_path: Path,
    output_glb_path: Path,
    output_stl_path: Path,
    color: tuple[float, float, float] = (0.55, 0.75, 0.45),
    smooth_sigma: float = 1.0,
    aligned_npz_path: Path | None = None,
) -> None:
    """Run Marching Cubes on the Step 9 voxel grid and write GLB + binary STL.

    Parameters
    ----------
    skeleton_npz_path : NPZ produced by Step 9 (contains ``voxels``,
                        ``origin_m``, ``voxel_size_xyz_m``).
    output_glb_path   : destination GLB file (for the viewer).
    output_stl_path   : destination binary STL file (deliverable mesh).
    color             : RGB base color for the GLB material.
    smooth_sigma      : Gaussian pre-smoothing sigma in voxels before MC
                        (default 1.0; 0 = disabled).
    aligned_npz_path  : if provided, read ``restore_R`` / ``restore_t`` from
                        this NPZ and apply the inverse OBB transform to all
                        mesh vertices/normals, restoring the original pose.
    """
    import sys
    import struct as _struct
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ct_reconstruction.glb_export import voxels_to_glb, _marching_cubes_surface, _build_glb

    data = np.load(str(skeleton_npz_path))
    occupancy: np.ndarray = data["voxels"].astype(bool)
    origin_m: np.ndarray = data["origin_m"]
    voxel_size_xyz_m: np.ndarray = data["voxel_size_xyz_m"]

    origin_mm = origin_m * 1e3
    spacing_mm = voxel_size_xyz_m * 1e3

    # Run Marching Cubes once; results are in the aligned (OBB) frame
    verts, normals, faces = _marching_cubes_surface(occupancy, origin_mm, spacing_mm, smooth_sigma)

    # --- Apply inverse OBB transform if requested ---
    if aligned_npz_path is not None:
        aligned_data = np.load(str(aligned_npz_path))
        if "restore_R" in aligned_data and "restore_t" in aligned_data:
            restore_R = aligned_data["restore_R"].astype(np.float64)   # (3,3)
            restore_t = aligned_data["restore_t"].astype(np.float64)   # (3,)
            # restore_t is in metres; convert to mm for consistency with verts
            restore_t_mm = restore_t * 1e3

            verts_orig = (restore_R @ verts.T).T + restore_t_mm
            # Normals are direction vectors — same rotation, no translation
            normals_orig = (restore_R @ normals.T).T
            # Re-normalise (rotation is orthogonal so this is a no-op numerically,
            # but guards against float accumulation)
            norms = np.linalg.norm(normals_orig, axis=1, keepdims=True)
            norms = np.where(norms > 1e-12, norms, 1.0)
            normals_orig = normals_orig / norms

            verts   = verts_orig.astype(np.float32)
            normals = normals_orig.astype(np.float32)
            print(f"  Applied inverse OBB transform (restore to original pose)")

    # --- GLB ---
    output_glb_path.parent.mkdir(parents=True, exist_ok=True)
    glb_bytes = _build_glb(verts, normals, faces, color)
    output_glb_path.write_bytes(glb_bytes)
    print(f"  GLB written → {output_glb_path}")

    # --- STL ---
    n_triangles = len(faces)
    output_stl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_stl_path, "wb") as f:
        f.write(b"Binary STL exported by matlab2stl_pipeline" + b"\x00" * 38)
        f.write(_struct.pack("<I", n_triangles))
        for i0, i1, i2 in faces:
            f.write(_struct.pack("<fff", *normals[i0]))
            f.write(_struct.pack("<fff", *verts[i0]))
            f.write(_struct.pack("<fff", *verts[i1]))
            f.write(_struct.pack("<fff", *verts[i2]))
            f.write(_struct.pack("<H", 0))
    print(f"  STL written → {output_stl_path}  ({n_triangles:,} triangles)")
