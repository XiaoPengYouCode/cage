"""Regression tests for the OBB inverse transform in the matlab2stl pipeline.

Tests verify that:
1. align_density() saves restore_R and restore_t into the aligned NPZ.
2. The inverse transform round-trips correctly: a point in the original voxel
   frame, aligned forward then restored back, returns to the original position.
3. mesh_from_voxels() applies the transform: output STL vertex centroid is
   shifted from the centroid produced without the transform.
"""
from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_raw_npz(path: Path, shape: tuple[int, int, int] = (20, 20, 20)) -> Path:
    """Write a minimal raw-density NPZ that obb_aligner can consume."""
    nx, ny, nz = shape
    # Occupied region: an elongated box along the first axis so PCA yields a
    # non-trivial rotation (not the identity).
    density = np.zeros(shape, dtype=np.uint16)
    density[2:18, 8:12, 8:12] = 800  # rod along axis-0

    voxel_size = np.array([0.4e-3, 0.4e-3, 0.4e-3], dtype=np.float32)  # 0.4 mm
    voxels = (density > 500).astype(np.uint8)
    origin = np.zeros(3, dtype=np.float32)

    np.savez_compressed(
        str(path),
        density_milli=density,
        voxels=voxels,
        voxel_size_xyz_m=voxel_size,
        origin_m=origin,
        grid_shape_xyz=np.array([nx, ny, nz], dtype=np.int32),
    )
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TransformMetadataSavedTest(unittest.TestCase):
    """align_density() must store restore_R and restore_t in the output NPZ."""

    def test_restore_keys_present(self) -> None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from matlab2stl_pipeline.obb_aligner import fit_obb, align_density

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw_npz = _make_synthetic_raw_npz(td / "raw.npz")
            obb_npz = td / "obb.npz"
            aligned_npz = td / "aligned.npz"

            fit_obb(raw_npz, obb_npz)
            align_density(raw_npz, obb_npz, aligned_npz, gamma=1.0)

            data = np.load(str(aligned_npz))
            self.assertIn("restore_R", data, "restore_R must be saved to aligned NPZ")
            self.assertIn("restore_t", data, "restore_t must be saved to aligned NPZ")

            R = data["restore_R"]
            t = data["restore_t"]
            self.assertEqual(R.shape, (3, 3))
            self.assertEqual(t.shape, (3,))

    def test_restore_R_is_rotation(self) -> None:
        """restore_R must be an orthogonal matrix (R @ R.T ≈ I, det ≈ +1)."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from matlab2stl_pipeline.obb_aligner import fit_obb, align_density

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw_npz = _make_synthetic_raw_npz(td / "raw.npz")
            obb_npz = td / "obb.npz"
            aligned_npz = td / "aligned.npz"

            fit_obb(raw_npz, obb_npz)
            align_density(raw_npz, obb_npz, aligned_npz, gamma=1.0)

            R = np.load(str(aligned_npz))["restore_R"]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6,
                                       err_msg="restore_R must be orthogonal")
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=6)


class InverseTransformRoundTripTest(unittest.TestCase):
    """The forward align + inverse restore must be the identity transform."""

    def test_round_trip(self) -> None:
        """A point in original physical space, aligned forward then restored, round-trips."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from matlab2stl_pipeline.obb_aligner import fit_obb, align_density

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw_npz = _make_synthetic_raw_npz(td / "raw.npz")
            obb_npz = td / "obb.npz"
            aligned_npz = td / "aligned.npz"

            fit_obb(raw_npz, obb_npz)
            align_density(raw_npz, obb_npz, aligned_npz, gamma=1.0)

            raw = np.load(str(raw_npz))
            obb = np.load(str(obb_npz))
            aligned_meta = np.load(str(aligned_npz))

            voxel_size = raw["voxel_size_xyz_m"].astype(np.float64)  # (3,)
            center_voxel = obb["center_voxel"].astype(np.float64)    # (3,)
            axes = obb["axes"].astype(np.float64)                     # (3,3)
            restore_R = aligned_meta["restore_R"]                     # (3,3)
            restore_t = aligned_meta["restore_t"]                     # (3,)

            # The aligned grid center in index space
            aligned_shape = aligned_meta["grid_shape_xyz"].astype(np.float64)
            new_center = (aligned_shape - 1) / 2.0  # (3,)

            # Pick an arbitrary point in the original frame (physical metres)
            # Use the OBB centre — its image in the aligned frame is exactly new_center.
            p_orig = center_voxel * voxel_size  # physical centre, metres

            # Forward: original physical → aligned index → aligned physical
            orig_idx = p_orig / voxel_size          # = center_voxel
            aligned_idx = axes @ (orig_idx - center_voxel) + new_center  # = new_center
            p_aligned_m = aligned_idx * voxel_size                        # = new_center * vs

            # Inverse transform: aligned physical → original physical
            p_restored = restore_R @ p_aligned_m + restore_t

            np.testing.assert_allclose(
                p_restored, p_orig, atol=1e-9,
                err_msg="Inverse transform must recover original position",
            )


class MeshTransformAppliedTest(unittest.TestCase):
    """mesh_from_voxels with aligned_npz_path must shift vertices vs. without."""

    def _read_stl_centroids(self, stl_path: Path) -> np.ndarray:
        """Return (N,3) array of per-triangle vertex-0 positions from a binary STL."""
        with open(stl_path, "rb") as f:
            f.read(80)  # header
            n = struct.unpack("<I", f.read(4))[0]
            verts = []
            for _ in range(n):
                f.read(12)  # normal
                v0 = struct.unpack("<fff", f.read(12))
                f.read(24)  # v1, v2
                f.read(2)   # attr
                verts.append(v0)
        return np.array(verts, dtype=np.float32)

    def test_transform_shifts_stl_centroid(self) -> None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from matlab2stl_pipeline.obb_aligner import fit_obb, align_density
        from matlab2stl_pipeline.seed_sampler import sample_seeds
        from matlab2stl_pipeline.cvt_relaxation import lloyd_relax
        from matlab2stl_pipeline.box_voronoi import build_box_voronoi, extract_voronoi_edges
        from matlab2stl_pipeline.skeleton_voxelizer import voxelize_skeleton, mesh_from_voxels

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw_npz = _make_synthetic_raw_npz(td / "raw.npz", shape=(30, 30, 30))
            obb_npz = td / "obb.npz"
            aligned_npz = td / "aligned.npz"

            fit_obb(raw_npz, obb_npz)
            align_density(raw_npz, obb_npz, aligned_npz, gamma=1.0)

            seeds_npz = td / "seeds.npz"
            sample_seeds(aligned_npz, seeds_npz, num_seeds=20, gamma=1.0)

            cvt_npz = td / "seeds_cvt.npz"
            lloyd_relax(seeds_npz, aligned_npz, cvt_npz, num_iters=1)

            vor_npz = td / "voronoi.npz"
            build_box_voronoi(cvt_npz, aligned_npz, vor_npz)

            edges_npz = td / "edges.npz"
            extract_voronoi_edges(vor_npz, edges_npz)

            skel_npz = td / "skeleton.npz"
            voxelize_skeleton(edges_npz, aligned_npz, skel_npz,
                              subdivision=3, dilation_radius_fine_voxels=1.5)

            stl_aligned = td / "scaffold_aligned.stl"
            stl_restored = td / "scaffold_restored.stl"

            mesh_from_voxels(skel_npz, td / "dummy_a.glb", stl_aligned,
                             smooth_sigma=0.5, aligned_npz_path=None)
            mesh_from_voxels(skel_npz, td / "dummy_r.glb", stl_restored,
                             smooth_sigma=0.5, aligned_npz_path=aligned_npz)

            verts_aligned  = self._read_stl_centroids(stl_aligned)
            verts_restored = self._read_stl_centroids(stl_restored)

            centroid_aligned  = verts_aligned.mean(axis=0)
            centroid_restored = verts_restored.mean(axis=0)

            shift = np.linalg.norm(centroid_restored - centroid_aligned)
            # If restore_R is the identity (unlikely for a non-axis-aligned OBB)
            # the shift could be zero; but the translation t must still differ.
            # We use a very loose bound: centroids must not be identical.
            # (They could coincide only if the OBB rotation is exactly the identity,
            # which the synthetic data is designed to avoid.)
            self.assertGreater(
                shift, 1e-3,
                "Restored mesh centroid must differ from aligned centroid (transform applied)",
            )


if __name__ == "__main__":
    unittest.main()
