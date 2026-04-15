from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.spatial

from matlab2stl_pipeline.box_voronoi import _build_cell_faces, extract_voronoi_edges


def _cube_vertices() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )


class BoxVoronoiFaceRecoveryTest(unittest.TestCase):
    def test_build_cell_faces_merges_cube_triangles_into_six_quads(self) -> None:
        faces = _build_cell_faces(_cube_vertices())

        self.assertEqual(len(faces), 6)
        self.assertEqual(sorted(len(np.asarray(face)) for face in faces), [4, 4, 4, 4, 4, 4])

    def test_extract_voronoi_edges_recovers_true_cube_edges_from_legacy_payload(self) -> None:
        vertices = _cube_vertices()
        hull = scipy.spatial.ConvexHull(vertices.astype(np.float64))

        cell_vertices = np.empty(1, dtype=object)
        cell_vertices[0] = vertices

        cell_simplices = np.empty(1, dtype=object)
        cell_simplices[0] = hull.simplices.astype(np.int32)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voronoi_npz = temp_path / "voronoi.npz"
            edges_npz = temp_path / "edges.npz"
            np.savez_compressed(
                voronoi_npz,
                cell_vertices=cell_vertices,
                cell_hull_simplices=cell_simplices,
                seed_points=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
                box_min=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                box_max=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                n_cells=np.int32(1),
            )

            edges = extract_voronoi_edges(voronoi_npz, edges_npz)

        self.assertEqual(edges.shape, (12, 2, 3))


if __name__ == "__main__":
    unittest.main()
