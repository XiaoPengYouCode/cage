from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scipy.io import savemat

from topopt_sampling.demo import (
    generate_annular_cylinder_npz,
    generate_fake_density_result,
    render_sampling_overview,
)
from topopt_sampling.probability import sample_seed_points
from topopt_sampling.workflows import map_density_to_seed_mapping


class SeedSamplingTest(unittest.TestCase):
    def test_sample_seed_points_respects_count(self) -> None:
        density = np.zeros((6, 6, 6), dtype=np.float32)
        density[1:5, 1:5, 1:5] = 1000.0
        seeds = sample_seed_points(
            density_milli=density,
            num_seeds=25,
            gamma=1.0,
            rng_seed=0,
        )
        self.assertEqual(seeds.shape, (25, 3))
        self.assertTrue(np.all(seeds >= 0.0))
        self.assertTrue(np.all(seeds[:, 0] < 6.0))
        self.assertTrue(np.all(seeds[:, 1] < 6.0))
        self.assertTrue(np.all(seeds[:, 2] < 6.0))

class SeedMappingWorkflowTest(unittest.TestCase):
    def test_workflows_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            density_npz = temp_path / "density.npz"
            mapping_npz = temp_path / "mapping.npz"
            density = np.full((9, 9, 6), 1000, dtype=np.uint16)
            np.savez_compressed(density_npz, density_milli=density)

            mapping = map_density_to_seed_mapping(
                density_npz,
                mapping_npz,
                num_seeds=64,
            )
            self.assertTrue(mapping_npz.exists())
            self.assertEqual(mapping.seed_points.shape, (64, 3))
            self.assertEqual(mapping.probability.shape, density.shape)

    def test_sample_seeds_accepts_mat_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            density_mat = temp_path / "density.mat"
            mapping_npz = temp_path / "mapping_mat.npz"

            density = np.full((9, 9, 6), 1000, dtype=np.uint16)
            savemat(density_mat, {"density_milli": density})

            mapping = map_density_to_seed_mapping(
                density_mat,
                mapping_npz,
                num_seeds=32,
            )
            self.assertTrue(mapping_npz.exists())
            self.assertEqual(mapping.seed_points.shape, (32, 3))


class DemoWorkflowTest(unittest.TestCase):
    def test_generate_voxels_writes_expected_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_npz = Path(temp_dir) / "voxels.npz"
            generate_annular_cylinder_npz(
                output_path=output_npz,
                xy_size=12,
                z_size=6,
                outer_radius=6.0,
                inner_radius=2.0,
                chunk_depth=2,
            )

            with np.load(output_npz) as data:
                voxels = data["voxels"]
                self.assertEqual(voxels.shape, (12, 12, 6))
                self.assertGreater(int(voxels.sum()), 0)

    def test_generate_fake_density_and_render_overview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voxel_npz = temp_path / "voxels.npz"
            density_npz = temp_path / "density.npz"
            mapping_npz = temp_path / "mapping.npz"
            overview_png = temp_path / "overview.png"

            generate_annular_cylinder_npz(
                output_path=voxel_npz,
                xy_size=12,
                z_size=6,
                outer_radius=6.0,
                inner_radius=2.0,
                chunk_depth=2,
            )
            generate_fake_density_result(
                source_npz=voxel_npz,
                output_npz=density_npz,
                chunk_depth=2,
            )
            mapping = map_density_to_seed_mapping(
                density_npz,
                mapping_npz,
                num_seeds=24,
            )
            render_sampling_overview(
                density_npz=density_npz,
                seed_npz=mapping.output_npz,
                output_png=overview_png,
            )

            self.assertTrue(density_npz.exists())
            self.assertTrue(overview_png.exists())


if __name__ == "__main__":
    unittest.main()
