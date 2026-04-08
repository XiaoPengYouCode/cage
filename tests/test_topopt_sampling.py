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
from topopt_sampling.exact_brep import build_diagram_brep, summarize_diagram_brep
from topopt_sampling.hybrid_exact_brep import (
    CylinderSupport,
    PlaneSupport,
    _pair_midpoint,
    build_hybrid_exact_diagram_brep,
    summarize_hybrid_exact_brep,
)
from topopt_sampling.exact_restricted_voronoi_3d import (
    build_annular_cylinder_domain,
    build_exact_restricted_cell,
    build_exact_restricted_voronoi_diagram,
    summarize_exact_diagram,
)
from topopt_sampling.exact_voronoi import (
    build_cap_surface_boundary_curves,
    build_cap_surface_patch_mesh,
    build_cylinder_surface_boundary_curves,
    build_cylinder_surface_patch_mesh,
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


class ExactRestrictedVoronoiTest(unittest.TestCase):
    def test_exact_restricted_cell_classifies_points_by_nearest_seed(self) -> None:
        seed_points = np.array(
            [
                [6.0, 5.0, 2.0],
                [6.0, 5.0, 8.0],
            ],
            dtype=np.float64,
        )
        domain = build_annular_cylinder_domain(
            xy_size=11,
            z_size=11,
            outer_radius=5.0,
            inner_radius=0.0,
        )
        cell0 = build_exact_restricted_cell(seed_points=seed_points, domain=domain, seed_id=0)
        points = np.array(
            [
                [10.0, 5.0, 2.0],
                [10.0, 5.0, 8.0],
                [20.0, 20.0, 2.0],
            ],
            dtype=np.float64,
        )

        mask = cell0.contains_points(points, domain)
        self.assertEqual(mask.tolist(), [True, False, False])
        self.assertTrue(cell0.contains_point(points[0], domain))
        self.assertFalse(cell0.contains_point(points[1], domain))
        self.assertFalse(cell0.contains_point(points[2], domain))

    def test_exact_restricted_diagram_builds_surface_traces(self) -> None:
        seed_points = np.array(
            [
                [6.0, 5.0, 2.0],
                [6.0, 5.0, 8.0],
            ],
            dtype=np.float64,
        )
        domain = build_annular_cylinder_domain(
            xy_size=11,
            z_size=11,
            outer_radius=5.0,
            inner_radius=0.0,
        )
        diagram = build_exact_restricted_voronoi_diagram(seed_points=seed_points, domain=domain, include_support_traces=True)

        self.assertEqual(len(diagram.cells), 2)
        self.assertEqual(diagram.classify_points(np.array([[10.0, 5.0, 2.0], [10.0, 5.0, 8.0]])).tolist(), [0, 1])

        outer_counts = [
            sum(len(trace.curves) for trace in cell.support_traces if trace.surface_name == "outer_cylinder")
            for cell in diagram.cells
        ]
        self.assertEqual(outer_counts, [1, 1])

        summary = summarize_exact_diagram(diagram)
        self.assertEqual(summary.num_seeds, 2)
        self.assertGreater(summary.domain_volume, 0.0)
        self.assertGreater(summary.support_curve_count, 0)

    def test_brep_pipeline_builds_faces_edges_and_vertices(self) -> None:
        seed_points = np.array(
            [
                [6.0, 5.0, 2.0],
                [6.0, 5.0, 8.0],
                [9.0, 6.5, 5.0],
                [3.0, 4.0, 5.0],
            ],
            dtype=np.float64,
        )
        domain = build_annular_cylinder_domain(
            xy_size=11,
            z_size=11,
            outer_radius=5.0,
            inner_radius=0.0,
        )
        diagram_brep = build_diagram_brep(seed_points=seed_points, domain=domain, seed_ids=[0])
        summary = summarize_diagram_brep(diagram_brep)

        self.assertEqual(summary.num_cells, 1)
        self.assertGreater(summary.num_faces, 0)
        self.assertGreater(summary.num_edges, 0)
        self.assertGreater(summary.num_vertices, 0)

    def test_hybrid_exact_kernel_builds_analytic_curve_objects(self) -> None:
        seed_points = np.array(
            [
                [6.0, 5.0, 2.0],
                [6.0, 5.0, 8.0],
                [9.0, 6.5, 5.0],
                [3.0, 4.0, 5.0],
            ],
            dtype=np.float64,
        )
        domain = build_annular_cylinder_domain(
            xy_size=11,
            z_size=11,
            outer_radius=5.0,
            inner_radius=0.0,
        )
        diagram_brep = build_hybrid_exact_diagram_brep(seed_points=seed_points, domain=domain, seed_ids=[0])
        summary = summarize_hybrid_exact_brep(diagram_brep)

        self.assertEqual(summary.num_cells, 1)
        self.assertGreater(summary.num_faces, 0)
        self.assertGreater(summary.num_edges, 0)
        self.assertGreater(summary.num_vertices, 0)

        cell = diagram_brep.cells[0]
        self.assertGreater(len(cell.polyhedral_cell.faces), 0)
        self.assertGreater(len(cell.polyhedral_cell.edges), 0)
        self.assertGreater(len(cell.polyhedral_cell.vertices), 0)

        curve_kinds = {edge.curve.kind for edge in cell.edges}
        self.assertIn("line_segment", curve_kinds)
        self.assertTrue(any(kind in curve_kinds for kind in {"circle_arc", "cylinder_plane_curve"}))
        self.assertTrue(any(face.loop_edge_ids for face in cell.faces))
        self.assertTrue(any(face.support_type == "plane" for face in cell.faces))
        self.assertTrue(any(face.support_key == "outer_cylinder" for face in cell.faces))
        self.assertGreaterEqual(len(cell.trim_summary.kept_plane_face_ids), 1)
        self.assertGreaterEqual(len(cell.trim_summary.generated_cylinder_face_ids), 1)


class DemoWorkflowTest(unittest.TestCase):
    def test_cap_patch_mesh_contains_multiple_regions(self) -> None:
        seed_points = np.array(
            [
                [2.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        mesh = build_cap_surface_patch_mesh(
            seed_points=seed_points,
            center_xy=np.array([0.0, 0.0], dtype=np.float64),
            inner_radius=0.0,
            outer_radius=5.0,
            z_value=0.0,
            surface_name="top_cap",
        )

        self.assertGreater(len(mesh.patches), 0)
        self.assertEqual(set(mesh.seed_ids.tolist()), {0, 1})

    def test_cylinder_patch_mesh_stays_on_requested_radius(self) -> None:
        seed_points = np.array(
            [
                [2.0, 0.0, 2.0],
                [-2.0, 0.0, 8.0],
            ],
            dtype=np.float32,
        )
        mesh = build_cylinder_surface_patch_mesh(
            seed_points=seed_points,
            center_xy=np.array([0.0, 0.0], dtype=np.float64),
            radius=5.0,
            z_min=0.0,
            z_max=10.0,
            surface_name="outer_cylinder",
        )

        self.assertGreater(len(mesh.patches), 0)
        radii = np.sqrt(mesh.patches[0][:, 0] ** 2 + mesh.patches[0][:, 1] ** 2)
        self.assertTrue(np.allclose(radii, 5.0, atol=1e-5))

    def test_exact_cap_boundary_is_straight_segment(self) -> None:
        seed_points = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        curves = build_cap_surface_boundary_curves(
            seed_points=seed_points,
            center_xy=np.array([0.0, 0.0], dtype=np.float64),
            inner_radius=0.0,
            outer_radius=5.0,
            z_value=0.0,
            surface_name="top_cap",
            active_seed_ids=np.array([0, 1], dtype=np.int32),
            candidate_pairs=((0, 1),),
        )

        self.assertEqual(len(curves), 1)
        points = curves[0].points
        self.assertTrue(np.allclose(points[:, 0], 0.0, atol=1e-5))
        self.assertAlmostEqual(float(points[:, 1].min()), -5.0, places=5)
        self.assertAlmostEqual(float(points[:, 1].max()), 5.0, places=5)

    def test_exact_cylinder_boundary_can_form_horizontal_ring(self) -> None:
        seed_points = np.array(
            [
                [1.0, 0.0, 2.0],
                [1.0, 0.0, 8.0],
            ],
            dtype=np.float32,
        )
        curves = build_cylinder_surface_boundary_curves(
            seed_points=seed_points,
            center_xy=np.array([0.0, 0.0], dtype=np.float64),
            radius=5.0,
            z_min=0.0,
            z_max=10.0,
            surface_name="outer_cylinder",
            active_seed_ids=np.array([0, 1], dtype=np.int32),
            candidate_pairs=((0, 1),),
        )

        self.assertEqual(len(curves), 1)
        points = curves[0].points
        self.assertTrue(np.allclose(points[:, 2], 5.0, atol=1e-5))
        radii = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        self.assertTrue(np.allclose(radii, 5.0, atol=1e-5))

    def test_cylinder_plane_midpoint_prefers_periodic_seam_branch(self) -> None:
        plane = PlaneSupport(
            key="test_plane",
            normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            rhs=5.0,
            support_type="plane",
        )
        cylinder = CylinderSupport(
            key="outer_cylinder",
            center_xy=np.array([0.0, 0.0], dtype=np.float64),
            radius=5.0,
            support_type="cylinder",
        )
        first = np.array([-4.99, 0.20, 5.0], dtype=np.float64)
        second = np.array([-4.99, -0.20, 5.0], dtype=np.float64)

        midpoint = _pair_midpoint(plane, cylinder, first, second)

        self.assertIsNotNone(midpoint)
        self.assertLess(float(midpoint[0]), -4.9)
        self.assertAlmostEqual(float(midpoint[1]), 0.0, places=3)
        self.assertAlmostEqual(float(midpoint[2]), 5.0, places=6)

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
