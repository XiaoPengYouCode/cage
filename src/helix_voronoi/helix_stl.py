from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import manifold3d as m3d
import numpy as np
from stl import Mode, mesh as stl_mesh

from helix_voronoi.models import EdgeSegment
from helix_voronoi.rods import CylinderRodStyle, HelixRodStyle
from helix_voronoi.voronoi import CUBE_HALFSPACES

# ── Edge classification ───────────────────────────────────────────────────────

EdgeRole = Literal["cube_edge", "face_edge", "half_face", "interior"]

_FACE_NORMALS = CUBE_HALFSPACES[:, :3]
_FACE_OFFSETS = CUBE_HALFSPACES[:, 3]


def _face_set(point: np.ndarray, tol: float = 1e-6) -> frozenset[int]:
    """Return the set of cube-face indices that *point* lies on."""
    slacks = _FACE_NORMALS @ point + _FACE_OFFSETS
    return frozenset(int(i) for i in np.flatnonzero(np.abs(slacks) < tol))


def classify_edge(start: np.ndarray, end: np.ndarray, tol: float = 1e-6) -> EdgeRole:
    """Classify a Voronoi edge relative to the unit-cube boundary.

    cube_edge  – both endpoints sit on ≥2 cube faces and share a common face
                 (i.e. the segment lies along one of the 12 cube edges)
    face_edge  – both endpoints share at least one cube face
                 (segment is entirely on a cube face)
    half_face  – exactly one endpoint is on a cube face, the other is interior
    interior   – both endpoints are strictly inside the cube
    """
    fs = _face_set(start, tol)
    fe = _face_set(end, tol)
    common = fs & fe
    if len(fs) >= 2 and len(fe) >= 2 and common:
        return "cube_edge"
    if common:
        return "face_edge"
    if fs or fe:
        return "half_face"
    return "interior"


def partition_edges(
    edges: list[EdgeSegment],
    tol: float = 1e-6,
) -> dict[EdgeRole, list[EdgeSegment]]:
    """Split *edges* into the four role buckets."""
    buckets: dict[EdgeRole, list[EdgeSegment]] = {
        "cube_edge": [],
        "face_edge": [],
        "half_face": [],
        "interior": [],
    }
    for start, end in edges:
        buckets[classify_edge(start, end, tol)].append((start, end))
    return buckets


# ── Summary dataclasses ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class HelixStlExportSummary:
    output_path: Path
    triangle_count: int
    edge_count: int


@dataclass(frozen=True)
class MixedStlExportSummary:
    output_path: Path
    triangle_count: int
    straight_edge_count: int
    interior_edge_count: int
    node_sphere_count: int


# ── manifold3d helpers ────────────────────────────────────────────────────────


def _make_cylinder_manifold(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    segments: int,
) -> m3d.Manifold | None:
    axis = np.asarray(end, float) - np.asarray(start, float)
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return None
    axis_n = axis / length
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, axis_n)
    c = float(np.dot(z, axis_n))
    if np.linalg.norm(v) < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        s = np.linalg.norm(v)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s**2)
    T = np.hstack([R, np.asarray(start, float).reshape(3, 1)])
    cyl = m3d.Manifold.cylinder(
        height=length, radius_low=radius, radius_high=radius, circular_segments=segments
    )
    return cyl.transform(T)


def _manifold_to_triangles(manifold: m3d.Manifold) -> np.ndarray:
    """Convert a manifold3d Manifold to an (F, 3, 3) triangle array."""
    mesh = manifold.to_mesh()
    verts = np.array(mesh.vert_properties, dtype=np.float64)
    faces = np.array(mesh.tri_verts, dtype=np.int32)
    return verts[faces]


def _union_manifolds(manifolds: list[m3d.Manifold]) -> m3d.Manifold:
    result = manifolds[0]
    for manifold in manifolds[1:]:
        result = result + manifold
    return result


def _pt_key(p: np.ndarray, decimals: int = 8) -> tuple:
    return tuple(np.round(p, decimals))


def tube_mesh_triangles(
    rings: np.ndarray,
    cap_ends: bool = True,
) -> np.ndarray:
    if rings.ndim != 3 or rings.shape[0] < 2 or rings.shape[1] < 3:
        raise ValueError(
            "Tube mesh must contain at least two rings with three points each."
        )

    ring_count, side_count, _ = rings.shape
    triangles: list[np.ndarray] = []

    for ring_index in range(ring_count - 1):
        next_ring_index = ring_index + 1
        for side_index in range(side_count):
            next_side_index = (side_index + 1) % side_count
            p00 = rings[ring_index, side_index]
            p01 = rings[ring_index, next_side_index]
            p10 = rings[next_ring_index, side_index]
            p11 = rings[next_ring_index, next_side_index]
            triangles.append(np.array([p00, p11, p10]))
            triangles.append(np.array([p00, p01, p11]))

    if cap_ends:
        start_center = rings[0].mean(axis=0)
        end_center = rings[-1].mean(axis=0)
        for side_index in range(side_count):
            next_side_index = (side_index + 1) % side_count
            triangles.append(
                np.array(
                    [start_center, rings[0, side_index], rings[0, next_side_index]]
                )
            )
            triangles.append(
                np.array(
                    [end_center, rings[-1, next_side_index], rings[-1, side_index]]
                )
            )

    return np.array(triangles)


def write_ascii_stl(
    triangles: np.ndarray,
    output_path: Path,
    solid_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    solid = stl_mesh.Mesh(
        np.zeros(len(triangles), dtype=stl_mesh.Mesh.dtype), remove_empty_areas=False
    )
    solid.vectors[:] = triangles
    solid.name = solid_name
    solid.save(str(output_path), mode=Mode.ASCII)


def export_helix_edges_to_stl(
    edges: list[EdgeSegment],
    radius: float,
    output_path: Path,
    rod_style: HelixRodStyle | None = None,
    solid_name: str = "cage_helix",
) -> HelixStlExportSummary:
    if not edges:
        raise ValueError("Cannot export STL for an empty edge list.")

    style = rod_style or HelixRodStyle()
    segment_meshes = [
        tube_mesh_triangles(style.build_segment_mesh(start, end, radius))
        for start, end in edges
    ]
    triangles = np.concatenate(segment_meshes, axis=0)
    write_ascii_stl(triangles, output_path=output_path, solid_name=solid_name)
    return HelixStlExportSummary(
        output_path=output_path,
        triangle_count=len(triangles),
        edge_count=len(edges),
    )


def export_mixed_edges_to_stl(
    edges: list[EdgeSegment],
    radius: float,
    output_path: Path,
    helix_cycles: float = 1.0,
    helix_amplitude: float = 0.06,
    tube_sides: int = 24,
    solid_name: str = "cage_mixed",
    tol: float = 1e-6,
) -> MixedStlExportSummary:
    """Export a mixed-style STL from Voronoi edges.

    Rules
    -----
    cube_edge / face_edge / half_face
        → straight cylinder rods, boolean-unioned via manifold3d so all
          rod intersections are cleanly merged into a single watertight solid.
    interior
        → helix tube (no end-caps) + one sphere per junction node, also
          boolean-unioned so the spheres fuse the tube ends together.
    """
    buckets = partition_edges(edges, tol=tol)
    helix_style = HelixRodStyle(
        cycles_per_segment=helix_cycles,
        amplitude_ratio=helix_amplitude,
        tube_sides=tube_sides,
    )
    sphere_radius = radius * 1.25

    all_triangles: list[np.ndarray] = []

    # ── straight rods: manifold3d boolean union ──────────────────────────────
    straight_edges = buckets["cube_edge"] + buckets["face_edge"] + buckets["half_face"]
    if straight_edges:
        cyls = [
            c
            for s, e in straight_edges
            if (c := _make_cylinder_manifold(s, e, radius, tube_sides)) is not None
        ]
        if cyls:
            all_triangles.append(_manifold_to_triangles(_union_manifolds(cyls)))

    # ── interior helix rods: tube sides (no caps) + junction spheres ──────────
    interior_edges = buckets["interior"]
    if interior_edges:
        # helix tube side-faces without end-caps
        for start, end in interior_edges:
            rings = helix_style.build_segment_mesh(start, end, radius)
            all_triangles.append(tube_mesh_triangles(rings, cap_ends=False))

        # one sphere per unique junction node, boolean-unioned
        node_keys: dict[tuple, np.ndarray] = {}
        for start, end in interior_edges:
            for pt in (start, end):
                key = _pt_key(pt)
                if key not in node_keys:
                    node_keys[key] = pt
        spheres = [
            m3d.Manifold.sphere(sphere_radius, circular_segments=tube_sides).translate(
                [float(pt[0]), float(pt[1]), float(pt[2])]
            )
            for pt in node_keys.values()
        ]
        all_triangles.append(_manifold_to_triangles(_union_manifolds(spheres)))

    if not all_triangles:
        raise ValueError("No geometry to export after filtering.")

    triangles = np.concatenate(all_triangles, axis=0)
    write_ascii_stl(triangles, output_path=output_path, solid_name=solid_name)

    return MixedStlExportSummary(
        output_path=output_path,
        triangle_count=len(triangles),
        straight_edge_count=len(straight_edges),
        interior_edge_count=len(interior_edges),
        node_sphere_count=len(node_keys) if interior_edges else 0,
    )
