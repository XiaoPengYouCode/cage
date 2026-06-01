from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class Iter017MaskSeedRadiusSweepTest(unittest.TestCase):
    def test_small_sweep_clips_skeleton_to_design_mask(self) -> None:
        import sys

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))

        from post_process.analysis.run_iter017_mask_seed_radius_sweep import run_sweep

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            reference_npz = tmp / "reference.npz"
            cage_npz = tmp / "cage.npz"

            shape = (8, 8, 8)
            design_mask = np.zeros(shape, dtype=np.uint8)
            design_mask[2:6, 2:6, 2:6] = 1
            density = np.zeros(shape, dtype=np.float64)
            density[design_mask.astype(bool)] = 0.8

            np.savez_compressed(
                reference_npz,
                design_mask=design_mask,
                voxel_size_xyz_m=np.array([0.0006, 0.0006, 0.0006], dtype=np.float32),
            )
            np.savez_compressed(cage_npz, cage_3d=density)

            summary = run_sweep(
                reference_npz=reference_npz,
                cage_npz=cage_npz,
                output_dir=tmp / "sweep",
                seed_counts=[8],
                radii_mm=[0.12],
                gamma=1.0,
                rng_seed=5,
                cvt_iters=1,
                subdivision=2,
                stages={"domain", "seeds", "voronoi", "skeleton"},
                resume=True,
                mesh_cases=set(),
            )

            self.assertEqual(summary["row_count"], 1)
            row = summary["rows"][0]
            self.assertEqual(row["domain_mode"], "design_mask_clipped")
            self.assertEqual(row["design_mask_voxels"], int(design_mask.sum()))
            self.assertGreater(row["solid_fraction_design_mask"], 0.0)
            self.assertLessEqual(row["solid_fraction_design_mask"], 1.0)

            skeleton = np.load(row["skeleton_npz"])
            voxels = skeleton["voxels"].astype(bool)
            pad = int(skeleton["pad_fine_voxels"])
            subdivision = int(skeleton["subdivision"])
            outside_count = 0
            for x in range(shape[0]):
                fx0 = pad + x * subdivision
                fx1 = fx0 + subdivision
                slab = voxels[
                    fx0:fx1,
                    pad : pad + shape[1] * subdivision,
                    pad : pad + shape[2] * subdivision,
                ]
                slab_mask = np.repeat(np.repeat(design_mask[x].astype(bool), subdivision, axis=0), subdivision, axis=1)
                outside_count += int(np.logical_and(slab, ~slab_mask[None, :, :]).sum())

            self.assertEqual(outside_count, 0)


if __name__ == "__main__":
    unittest.main()
