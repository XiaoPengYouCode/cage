from __future__ import annotations

import unittest

import numpy as np

from fem_analysis.fjw_coupling import rigid_kinematic_coupling_matrix, rigid_kinematic_displacements


class FJWCouplingTest(unittest.TestCase):
    def test_rigid_kinematic_displacements_follow_cross_product_formula(self) -> None:
        coordinates = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
        displacement = rigid_kinematic_displacements(
            coordinates,
            reference_point=np.zeros(3),
            reference_displacement=np.array([0.1, 0.2, 0.3]),
            reference_rotation=np.array([0.0, 0.0, 2.0]),
        )

        np.testing.assert_allclose(
            displacement,
            np.array([[0.1, 2.2, 0.3], [-3.9, 0.2, 0.3]], dtype=np.float64),
        )

    def test_coupling_matrix_maps_reference_dofs_to_node_displacements(self) -> None:
        coordinates = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        matrix = rigid_kinematic_coupling_matrix(
            coordinates,
            reference_point=np.array([0.5, 1.0, 1.5]),
        )
        reference_dofs = np.array([0.1, 0.2, 0.3, 0.4, -0.2, 0.6], dtype=np.float64)

        np.testing.assert_allclose(
            matrix @ reference_dofs,
            rigid_kinematic_displacements(
                coordinates,
                reference_point=np.array([0.5, 1.0, 1.5]),
                reference_displacement=reference_dofs[:3],
                reference_rotation=reference_dofs[3:],
            ).reshape(-1),
        )


if __name__ == "__main__":
    unittest.main()
