from __future__ import annotations

import numpy as np


def rigid_kinematic_displacements(
    node_coordinates: np.ndarray,
    *,
    reference_point: np.ndarray,
    reference_displacement: np.ndarray,
    reference_rotation: np.ndarray,
) -> np.ndarray:
    """Abaqus-style small-rotation kinematic coupling displacement field.

    For each coupled node, the displacement is
    `u_i = u_ref + theta_ref x (x_i - x_ref)`.
    Coordinates and displacements must use the same length unit.
    """

    coordinates = np.asarray(node_coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("node_coordinates must have shape (n, 3).")
    ref = np.asarray(reference_point, dtype=np.float64).reshape(3)
    disp = np.asarray(reference_displacement, dtype=np.float64).reshape(3)
    rot = np.asarray(reference_rotation, dtype=np.float64).reshape(3)
    offsets = coordinates - ref[None, :]
    return disp[None, :] + np.cross(rot[None, :], offsets)


def rigid_kinematic_coupling_matrix(
    node_coordinates: np.ndarray,
    *,
    reference_point: np.ndarray,
) -> np.ndarray:
    """Return the dense matrix mapping 6 reference DOFs to 3N node DOFs."""

    coordinates = np.asarray(node_coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("node_coordinates must have shape (n, 3).")
    ref = np.asarray(reference_point, dtype=np.float64).reshape(3)
    matrix = np.zeros((coordinates.shape[0] * 3, 6), dtype=np.float64)
    for row, offset in enumerate(coordinates - ref[None, :]):
        base = row * 3
        rx, ry, rz = offset
        matrix[base : base + 3, :3] = np.eye(3)
        matrix[base : base + 3, 3:] = np.array(
            [
                [0.0, rz, -ry],
                [-rz, 0.0, rx],
                [ry, -rx, 0.0],
            ],
            dtype=np.float64,
        )
    return matrix


__all__ = [
    "rigid_kinematic_coupling_matrix",
    "rigid_kinematic_displacements",
]
