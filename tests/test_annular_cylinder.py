from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fem_analysis.annular_cylinder import (
    AnnularCylinderConfig,
    TrussInfillConfig,
    build_annular_cylinder_npz_payload,
    build_annular_cylinder_mesh,
    write_annular_cylinder_npz,
)
from fem_analysis.cli import build_annular_cylinder_config, parse_args
from topopt_sampling.workflows import map_density_to_seed_mapping


class AnnularCylinderCliTest(unittest.TestCase):
    def test_default_voxel_size_is_directly_configurable(self) -> None:
        config = build_annular_cylinder_config(parse_args(["annular-cylinder"]))

        self.assertAlmostEqual(config.height_m, 0.02)
        self.assertAlmostEqual(config.voxel_size_m, 0.0004)
        self.assertEqual(config.material.name, "TC4")
        self.assertAlmostEqual(config.material.youngs_modulus_gpa, 110.0)
        self.assertEqual(config.inner_fill_mode, "bone")
        self.assertAlmostEqual(config.fill_material.youngs_modulus_gpa, 1.0)
        self.assertAlmostEqual(config.truss_infill.cell_size_m, 0.0004)
        self.assertAlmostEqual(config.truss_infill.rod_radius_m, 0.0001)
        self.assertEqual(
            config.output_npz,
            Path("datasets/topopt/annular_cylinder_fea_density.npz"),
        )

    def test_parse_annular_cylinder_command(self) -> None:
        args = parse_args(
            [
                "annular-cylinder",
                "--outer-diameter-cm",
                "3",
                "--inner-diameter-cm",
                "2",
                "--height-cm",
                "3",
                "--load-n",
                "1000",
                "--voxel-size-mm",
                "0.5",
            ]
        )

        self.assertEqual(args.command, "annular-cylinder")
        config = build_annular_cylinder_config(args)
        self.assertAlmostEqual(config.outer_diameter_m, 0.03)
        self.assertAlmostEqual(config.inner_diameter_m, 0.02)
        self.assertAlmostEqual(config.height_m, 0.03)
        self.assertAlmostEqual(config.total_force_n, 1000.0)
        self.assertAlmostEqual(config.voxel_size_m, 0.0005)
        self.assertEqual(config.inner_fill_mode, "bone")
        self.assertFalse(config.truss_infill.enabled)

    def test_parse_annular_cylinder_fill_modes(self) -> None:
        args = parse_args(
            [
                "annular-cylinder",
                "--fill-youngs-modulus-gpa",
                "2.5",
                "--fill-poisson-ratio",
                "0.28",
            ]
        )

        config = build_annular_cylinder_config(args)
        self.assertEqual(config.inner_fill_mode, "bone")
        self.assertAlmostEqual(config.fill_material.youngs_modulus_gpa, 2.5)
        self.assertAlmostEqual(config.fill_material.poisson_ratio, 0.28)

        truss_args = parse_args(
            [
                "annular-cylinder",
                "--inner-fill",
                "truss",
                "--truss-cell-mm",
                "4.5",
                "--truss-rod-mm",
                "0.7",
            ]
        )

        truss_config = build_annular_cylinder_config(truss_args)
        self.assertEqual(truss_config.inner_fill_mode, "truss")
        self.assertTrue(truss_config.truss_infill.enabled)
        self.assertAlmostEqual(truss_config.truss_infill.cell_size_m, 0.0045)
        self.assertAlmostEqual(truss_config.truss_infill.rod_radius_m, 0.0007)


class AnnularCylinderMeshTest(unittest.TestCase):
    def test_mesh_extents_and_counts(self) -> None:
        config = AnnularCylinderConfig(voxel_size_m=0.03 / 24.0, inner_fill_mode="empty")

        mesh = build_annular_cylinder_mesh(config)

        self.assertGreater(mesh.hex_mesh.active_element_count, 0)
        self.assertGreater(mesh.hex_mesh.active_node_count, 0)
        self.assertEqual(mesh.occupancy.shape, (24, 24, 16))
        self.assertAlmostEqual(mesh.hex_mesh.coordinates[:, 0].max(), 0.03)
        self.assertAlmostEqual(mesh.hex_mesh.coordinates[:, 1].max(), 0.03)
        self.assertAlmostEqual(mesh.hex_mesh.coordinates[:, 2].max(), 0.02)
        self.assertEqual(mesh.fill_element_count, 0)

    def test_bone_fill_fills_inner_region(self) -> None:
        config = AnnularCylinderConfig(voxel_size_m=0.03 / 24.0, inner_fill_mode="bone")

        mesh = build_annular_cylinder_mesh(config)

        self.assertGreater(mesh.fill_element_count, 0)
        self.assertEqual(mesh.fill_element_count, mesh.inner_void_voxel_count)
        self.assertTrue(mesh.fill_mask[:, :, mesh.fill_mask.shape[2] // 2].any())

    def test_truss_infill_adds_sparse_inner_structure(self) -> None:
        config = AnnularCylinderConfig(
            voxel_size_m=0.03 / 24.0,
            inner_fill_mode="truss",
            truss_infill=TrussInfillConfig(
                enabled=True,
                cell_size_m=0.006,
                rod_radius_m=0.0009,
            ),
        )

        mesh = build_annular_cylinder_mesh(config)

        self.assertGreater(mesh.fill_element_count, 0)
        self.assertLess(mesh.fill_element_count, mesh.inner_void_voxel_count)
        self.assertGreater(mesh.loaded_top_area_m2, 0.0)
        self.assertTrue(mesh.fill_mask[:, :, mesh.fill_mask.shape[2] // 2].any())

    def test_npz_payload_matches_repo_density_convention(self) -> None:
        config = AnnularCylinderConfig(voxel_size_m=0.03 / 24.0, inner_fill_mode="bone")
        mesh = build_annular_cylinder_mesh(config)

        payload = build_annular_cylinder_npz_payload(config, mesh)

        self.assertEqual(payload["voxels"].shape, mesh.grid_shape)
        self.assertEqual(payload["density_milli"].shape, mesh.grid_shape)
        self.assertEqual(payload["material_id"].shape, mesh.grid_shape)
        self.assertEqual(payload["density_milli"].dtype, np.uint16)
        self.assertEqual(int(payload["density_milli"].max()), 1000)
        self.assertEqual(int(payload["density_milli"].min()), 0)
        self.assertEqual(int(payload["xy_size"].item()), mesh.grid_shape[0])
        self.assertEqual(int(payload["z_size"].item()), mesh.grid_shape[2])
        self.assertEqual(str(payload["result_type"].item()), "fem_annular_cylinder_density")
        self.assertEqual(str(payload["density_kind"].item()), "binary_occupancy")
        self.assertTrue(np.all(payload["material_id"][mesh.shell_mask] == 0))
        self.assertTrue(np.all(payload["material_id"][mesh.fill_mask] == 1))
        self.assertTrue(np.all(payload["material_id"][~mesh.occupancy] == -1))

    def test_exported_npz_can_feed_seed_sampling_workflow(self) -> None:
        config = AnnularCylinderConfig(voxel_size_m=0.03 / 24.0, inner_fill_mode="bone")
        mesh = build_annular_cylinder_mesh(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            density_npz = Path(tmp_dir) / "annular_cylinder_density.npz"
            seed_npz = Path(tmp_dir) / "seed_mapping.npz"

            write_annular_cylinder_npz(config, mesh, density_npz)
            mapping = map_density_to_seed_mapping(
                density_npz,
                seed_npz,
                num_seeds=32,
                gamma=1.0,
                rng_seed=7,
            )

            self.assertTrue(density_npz.exists())
            self.assertTrue(seed_npz.exists())
            self.assertEqual(mapping.seed_points.shape, (32, 3))
            self.assertEqual(mapping.probability.shape, mesh.grid_shape)
            self.assertGreater(float(mapping.probability.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
