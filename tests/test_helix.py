from __future__ import annotations

import unittest

import numpy as np

from cage.helix import HelixSpec, build_helix_centerline, build_tangents, build_tube_mesh
from cage.rods import segment_frame


class HelixGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.start = np.array([0.0, 0.0, 0.0], dtype=float)
        self.end = np.array([1.0, 0.0, 0.0], dtype=float)
        _, _, basis_u, basis_v = segment_frame(self.start, self.end)
        self.basis_u = basis_u
        self.basis_v = basis_v

    def test_centerline_keeps_segment_endpoints(self) -> None:
        centerline, progress = build_helix_centerline(
            self.start,
            self.end,
            self.basis_u,
            self.basis_v,
            HelixSpec(),
        )

        np.testing.assert_allclose(centerline[0], self.start, atol=1e-12)
        np.testing.assert_allclose(centerline[-1], self.end, atol=1e-12)
        self.assertEqual(progress[0], 0.0)
        self.assertEqual(progress[-1], 1.0)

    def test_tube_mesh_has_expected_shape_and_no_gaps(self) -> None:
        spec = HelixSpec()
        centerline, _ = build_helix_centerline(
            self.start,
            self.end,
            self.basis_u,
            self.basis_v,
            spec,
        )
        tangents = build_tangents(centerline)
        radii = np.full(len(centerline), 0.012 * spec.wire_radius_ratio)
        mesh = build_tube_mesh(centerline, tangents, radii, tube_sides=spec.tube_sides)

        self.assertEqual(mesh.shape, (len(centerline), spec.tube_sides, 3))
        self.assertTrue(np.isfinite(mesh).all())
        self.assertGreater(np.linalg.norm(mesh[1, 0] - mesh[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
