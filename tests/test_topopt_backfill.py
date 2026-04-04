from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from topopt_backfill.probability import sample_seed_points
from topopt_backfill.templates import build_backfilled_templates
from topopt_backfill.workflows import build_template_backfill, map_density_to_seed_mapping


class SeedSamplingTest(unittest.TestCase):
    def test_sample_seed_points_respects_count(self) -> None:
        density = np.zeros((6, 6, 6), dtype=np.float32)
        density[1:5, 1:5, 1:5] = 1000.0
        seeds = sample_seed_points(
            density_milli=density,
            num_seeds=25,
            gamma=1.0,
            chunk_depth=2,
            rng_seed=0,
        )
        self.assertEqual(seeds.shape, (25, 3))
        self.assertTrue(np.all(seeds >= 0.0))
        self.assertTrue(np.all(seeds[:, 0] < 6.0))
        self.assertTrue(np.all(seeds[:, 1] < 6.0))
        self.assertTrue(np.all(seeds[:, 2] < 6.0))


class BackfillTemplateTest(unittest.TestCase):
    def test_build_backfilled_templates_returns_ten_templates(self) -> None:
        density = np.full((9, 9, 6), 1000, dtype=np.uint16)
        _, block_active, template_ids, seed_counts, templates = build_backfilled_templates(
            density,
            progress=False,
        )
        self.assertEqual(block_active.shape, (3, 3, 2))
        self.assertEqual(template_ids.shape, (3, 3, 2))
        self.assertEqual(len(seed_counts), 10)
        self.assertEqual(len(templates), 10)


class CageFillWorkflowTest(unittest.TestCase):
    def test_workflows_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            density_npz = temp_path / "density.npz"
            mapping_npz = temp_path / "mapping.npz"
            backfill_npz = temp_path / "backfill.npz"

            density = np.full((9, 9, 6), 1000, dtype=np.uint16)
            np.savez_compressed(density_npz, density_milli=density)

            mapping = map_density_to_seed_mapping(
                density_npz,
                mapping_npz,
                num_seeds=64,
                chunk_depth=32,
                max_display_size=12,
            )
            self.assertTrue(mapping_npz.exists())
            self.assertEqual(mapping.seed_points.shape, (64, 3))

            backfill = build_template_backfill(
                density_npz,
                mapping_npz,
                backfill_npz,
                representative_seeds=12,
                candidate_limit=32,
                candidate_voxel_bins=(4, 4, 4),
                target_shape=(9, 9, 6),
            )
            self.assertTrue(backfill_npz.exists())
            self.assertEqual(len(backfill.representative_seeds), 12)
            self.assertEqual(len(backfill.template_seed_counts), 10)


if __name__ == "__main__":
    unittest.main()
