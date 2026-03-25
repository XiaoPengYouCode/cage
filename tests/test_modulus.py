from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from cage.analysis.compression import solve_compression
from cage.analysis.config import CompressionConfig, MaterialConfig
from cage.analysis.geometry import SegmentCloud
from cage.analysis.voxel import VoxelGrid, build_hex_mesh
from cage.cli import main


class GeometryMembershipTest(unittest.TestCase):
    def test_cylinder_segment_cloud_contains_axis_points(self) -> None:
        geometry = SegmentCloud(
            starts=np.array([[0.1, 0.5, 0.5]], dtype=float),
            ends=np.array([[0.9, 0.5, 0.5]], dtype=float),
            radius=0.1,
            style="cylinder",
        )
        points = np.array(
            [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.65],
                [0.5, 0.5, 0.71],
            ],
            dtype=float,
        )
        mask = geometry.contains_points(points, chunk_size=2)
        self.assertEqual(mask.tolist(), [True, False, False])


class HexMeshTest(unittest.TestCase):
    def test_build_hex_mesh_counts_shared_nodes(self) -> None:
        grid = VoxelGrid(
            resolution=2,
            voxel_size=0.5,
            occupancy=np.array(
                [
                    [[True, True], [False, False]],
                    [[False, False], [False, False]],
                ],
                dtype=bool,
            ),
        )
        mesh = build_hex_mesh(grid)
        self.assertEqual(mesh.active_element_count, 2)
        self.assertEqual(mesh.active_node_count, 12)


class SfePyCompressionSmokeTest(unittest.TestCase):
    def test_solid_cube_modulus_is_reasonable(self) -> None:
        grid = VoxelGrid(
            resolution=3,
            voxel_size=1.0 / 3.0,
            occupancy=np.ones((3, 3, 3), dtype=bool),
        )
        result = solve_compression(
            style="cylinder",
            grid=grid,
            material=MaterialConfig(name="test", youngs_modulus_gpa=10.0, poisson_ratio=0.0),
            compression=CompressionConfig(applied_strain=1e-3),
        )
        self.assertGreater(result.effective_modulus_gpa, 9.0)
        self.assertLess(result.effective_modulus_gpa, 11.0)
        self.assertGreater(result.reaction_force_n, 0.0)
        self.assertEqual(result.active_element_count, 27)
        self.assertAlmostEqual(result.solid_volume_fraction, 1.0, places=6)


class ModulusCliTest(unittest.TestCase):
    def test_modulus_dry_run_does_not_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            markdown = temp_path / "report.md"
            json_path = temp_path / "report.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main(
                    [
                        "modulus",
                        "--seed",
                        "55",
                        "--style",
                        "both",
                        "--dry-run",
                        "--output-markdown",
                        str(markdown),
                        "--output-json",
                        str(json_path),
                    ]
                )

            output = stdout.getvalue()
            self.assertIn("Dry run: sfepy modulus analysis", output)
            self.assertFalse(markdown.exists())
            self.assertFalse(json_path.exists())


if __name__ == "__main__":
    unittest.main()
