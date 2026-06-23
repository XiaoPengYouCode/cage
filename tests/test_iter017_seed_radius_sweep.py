from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class Iter017SeedRadiusSweepTest(unittest.TestCase):
    def test_small_sweep_writes_porosity_summary(self) -> None:
        import sys

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))

        from post_process.analysis.run_iter017_seed_radius_sweep import run_sweep

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            aligned_npz = tmp / "aligned.npz"
            density = np.ones((6, 6, 6), dtype=np.uint16) * 1000
            np.savez_compressed(
                aligned_npz,
                density_milli=density,
                probability_field=np.full(density.shape, 1.0 / density.size, dtype=np.float32),
                grid_shape_xyz=np.array(density.shape, dtype=np.int32),
                voxel_size_xyz_m=np.array([0.0004, 0.0004, 0.0004], dtype=np.float32),
                origin_m=np.zeros(3, dtype=np.float32),
            )

            summary = run_sweep(
                aligned_npz=aligned_npz,
                output_dir=tmp / "sweep",
                seed_counts=[8],
                radii_mm=[0.08],
                gamma=1.0,
                rng_seed=3,
                cvt_iters=1,
                subdivision=2,
                stages={"seeds", "voronoi", "skeleton"},
                resume=True,
                skip_mesh_export=True,
            )

            self.assertEqual(summary["row_count"], 1)
            row = summary["rows"][0]
            self.assertEqual(row["seed_count"], 8)
            self.assertAlmostEqual(row["radius_mm"], 0.08)
            self.assertGreater(row["edge_count"], 0)
            self.assertGreater(row["solid_fraction_density_domain"], 0.0)
            self.assertLessEqual(row["solid_fraction_density_domain"], 1.0)
            self.assertAlmostEqual(
                row["porosity_density_domain"],
                1.0 - row["solid_fraction_density_domain"],
            )
            self.assertTrue(Path(row["skeleton_npz"]).exists())


if __name__ == "__main__":
    unittest.main()
