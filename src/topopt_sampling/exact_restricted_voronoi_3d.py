from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from topopt_sampling.exact_voronoi import (
    BoundaryCurve3D,
    build_cap_surface_boundary_curves,
    build_cylinder_surface_boundary_curves,
    discover_cap_candidates,
    discover_cylinder_candidates,
)
from topopt_sampling.neighbors import build_delaunay_neighbor_map


@dataclass(frozen=True)
class AnnularCylinderDomain:
    center_xy: np.ndarray
    z_min: float
    z_max: float
    outer_radius: float
    inner_radius: float = 0.0

    def contains_points(self, points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        radial_sq = (points[:, 0] - self.center_xy[0]) ** 2 + (points[:, 1] - self.center_xy[1]) ** 2
        return (
            (points[:, 2] >= self.z_min - tol)
            & (points[:, 2] <= self.z_max + tol)
            & (radial_sq <= self.outer_radius * self.outer_radius + tol)
            & (radial_sq >= self.inner_radius * self.inner_radius - tol)
        )

    def contains_point(self, point: np.ndarray, tol: float = 1e-9) -> bool:
        return bool(self.contains_points(np.asarray(point, dtype=np.float64).reshape(1, 3), tol=tol)[0])


@dataclass(frozen=True)
class VoronoiHalfspace:
    owner_seed_id: int
    other_seed_id: int
    normal: np.ndarray
    rhs: float

    def evaluate_point(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=np.float64)
        return float(np.dot(point, self.normal) - self.rhs)

    def evaluate_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return points @ self.normal - self.rhs

    def contains_point(self, point: np.ndarray, tol: float = 1e-9) -> bool:
        return self.evaluate_point(point) <= tol

    def contains_points(self, points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        return self.evaluate_points(points) <= tol


@dataclass(frozen=True)
class SupportTraceSet:
    surface_name: str
    curves: tuple[BoundaryCurve3D, ...]

    @property
    def neighbor_seed_ids(self) -> tuple[int, ...]:
        neighbors: set[int] = set()
        for curve in self.curves:
            neighbors.update(int(seed_id) for seed_id in curve.seed_pair)
        return tuple(sorted(neighbors))


@dataclass(frozen=True)
class ExactRestrictedCell:
    seed_id: int
    seed_point: np.ndarray
    halfspaces: tuple[VoronoiHalfspace, ...]
    support_traces: tuple[SupportTraceSet, ...] = field(default_factory=tuple)
    halfspace_normals: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64), repr=False)
    halfspace_rhs: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64), repr=False)

    @property
    def support_trace_count(self) -> int:
        return sum(len(trace.curves) for trace in self.support_traces)

    @property
    def neighboring_seed_ids(self) -> tuple[int, ...]:
        neighbors: set[int] = set()
        for halfspace in self.halfspaces:
            neighbors.add(int(halfspace.other_seed_id))
        return tuple(sorted(neighbors))

    def contains_point(self, point: np.ndarray, domain: AnnularCylinderDomain, tol: float = 1e-9) -> bool:
        point = np.asarray(point, dtype=np.float64)
        if not domain.contains_point(point, tol=tol):
            return False
        if self.halfspace_normals.size == 0:
            return True
        values = self.halfspace_normals @ point - self.halfspace_rhs
        return bool(np.all(values <= tol))

    def contains_points(self, points: np.ndarray, domain: AnnularCylinderDomain, tol: float = 1e-9) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        mask = domain.contains_points(points, tol=tol)
        if not np.any(mask) or self.halfspace_normals.size == 0:
            return mask
        inside_values = points[mask] @ self.halfspace_normals.T - self.halfspace_rhs[None, :]
        mask[mask] = np.all(inside_values <= tol, axis=1)
        return mask


@dataclass(frozen=True)
class ExactRestrictedVoronoiDiagram:
    seed_points: np.ndarray
    domain: AnnularCylinderDomain
    cells: tuple[ExactRestrictedCell, ...]

    def classify_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        labels = np.full(points.shape[0], -1, dtype=np.int32)
        in_domain = self.domain.contains_points(points)
        if not np.any(in_domain):
            return labels
        domain_points = points[in_domain]
        distances_sq = np.sum((domain_points[:, None, :] - self.seed_points[None, :, :]) ** 2, axis=2)
        labels[in_domain] = np.argmin(distances_sq, axis=1).astype(np.int32)
        return labels

    def cell_for_seed(self, seed_id: int) -> ExactRestrictedCell:
        return self.cells[int(seed_id)]


@dataclass(frozen=True)
class ExactDiagramSummary:
    num_seeds: int
    domain_volume: float
    support_curve_count: int


def build_annular_cylinder_domain(
    xy_size: int,
    z_size: int,
    outer_radius: float,
    inner_radius: float,
) -> AnnularCylinderDomain:
    return AnnularCylinderDomain(
        center_xy=np.array([(xy_size - 1) / 2.0, (xy_size - 1) / 2.0], dtype=np.float64),
        z_min=0.5,
        z_max=z_size - 0.5,
        outer_radius=float(outer_radius),
        inner_radius=float(inner_radius),
    )


def build_voronoi_halfspaces(
    seed_points: np.ndarray,
    seed_id: int,
    neighbor_seed_ids: Iterable[int] | None = None,
) -> tuple[VoronoiHalfspace, ...]:
    seed_points = np.asarray(seed_points, dtype=np.float64)
    seed_i = seed_points[int(seed_id)]
    other_seed_ids = (
        [int(other_seed_id) for other_seed_id in neighbor_seed_ids if int(other_seed_id) != int(seed_id)]
        if neighbor_seed_ids is not None
        else [other_seed_id for other_seed_id in range(seed_points.shape[0]) if other_seed_id != int(seed_id)]
    )
    halfspaces: list[VoronoiHalfspace] = []
    for other_seed_id in other_seed_ids:
        seed_j = seed_points[int(other_seed_id)]
        normal = 2.0 * (seed_j - seed_i)
        rhs = float(np.dot(seed_j, seed_j) - np.dot(seed_i, seed_i))
        halfspaces.append(
            VoronoiHalfspace(
                owner_seed_id=int(seed_id),
                other_seed_id=int(other_seed_id),
                normal=normal.astype(np.float64, copy=False),
                rhs=rhs,
            )
        )
    return tuple(halfspaces)


def _filter_curves_for_seed(curves: Iterable[BoundaryCurve3D], seed_id: int) -> tuple[BoundaryCurve3D, ...]:
    return tuple(curve for curve in curves if int(seed_id) in curve.seed_pair)


def build_support_traces_for_cell(
    seed_points: np.ndarray,
    domain: AnnularCylinderDomain,
    seed_id: int,
) -> tuple[SupportTraceSet, ...]:
    traces: list[SupportTraceSet] = []

    outer_candidates = discover_cylinder_candidates(
        seed_points=seed_points,
        center_xy=domain.center_xy,
        radius=domain.outer_radius,
        z_min=domain.z_min,
        z_max=domain.z_max,
    )
    outer_curves = _filter_curves_for_seed(
        build_cylinder_surface_boundary_curves(
            seed_points=seed_points,
            center_xy=domain.center_xy,
            radius=domain.outer_radius,
            z_min=domain.z_min,
            z_max=domain.z_max,
            surface_name="outer_cylinder",
            active_seed_ids=outer_candidates.active_seed_ids,
            candidate_pairs=outer_candidates.candidate_pairs,
        ),
        seed_id=seed_id,
    )
    traces.append(SupportTraceSet(surface_name="outer_cylinder", curves=outer_curves))

    if domain.inner_radius > 0.0:
        inner_candidates = discover_cylinder_candidates(
            seed_points=seed_points,
            center_xy=domain.center_xy,
            radius=domain.inner_radius,
            z_min=domain.z_min,
            z_max=domain.z_max,
        )
        inner_curves = _filter_curves_for_seed(
            build_cylinder_surface_boundary_curves(
                seed_points=seed_points,
                center_xy=domain.center_xy,
                radius=domain.inner_radius,
                z_min=domain.z_min,
                z_max=domain.z_max,
                surface_name="inner_cylinder",
                active_seed_ids=inner_candidates.active_seed_ids,
                candidate_pairs=inner_candidates.candidate_pairs,
            ),
            seed_id=seed_id,
        )
        traces.append(SupportTraceSet(surface_name="inner_cylinder", curves=inner_curves))

    top_candidates = discover_cap_candidates(
        seed_points=seed_points,
        center_xy=domain.center_xy,
        inner_radius=domain.inner_radius,
        outer_radius=domain.outer_radius,
        z_value=domain.z_max,
    )
    top_curves = _filter_curves_for_seed(
        build_cap_surface_boundary_curves(
            seed_points=seed_points,
            center_xy=domain.center_xy,
            inner_radius=domain.inner_radius,
            outer_radius=domain.outer_radius,
            z_value=domain.z_max,
            surface_name="top_cap",
            active_seed_ids=top_candidates.active_seed_ids,
            candidate_pairs=top_candidates.candidate_pairs,
        ),
        seed_id=seed_id,
    )
    traces.append(SupportTraceSet(surface_name="top_cap", curves=top_curves))

    bottom_candidates = discover_cap_candidates(
        seed_points=seed_points,
        center_xy=domain.center_xy,
        inner_radius=domain.inner_radius,
        outer_radius=domain.outer_radius,
        z_value=domain.z_min,
    )
    bottom_curves = _filter_curves_for_seed(
        build_cap_surface_boundary_curves(
            seed_points=seed_points,
            center_xy=domain.center_xy,
            inner_radius=domain.inner_radius,
            outer_radius=domain.outer_radius,
            z_value=domain.z_min,
            surface_name="bottom_cap",
            active_seed_ids=bottom_candidates.active_seed_ids,
            candidate_pairs=bottom_candidates.candidate_pairs,
        ),
        seed_id=seed_id,
    )
    traces.append(SupportTraceSet(surface_name="bottom_cap", curves=bottom_curves))

    return tuple(traces)


def build_exact_restricted_cell(
    seed_points: np.ndarray,
    domain: AnnularCylinderDomain,
    seed_id: int,
    include_support_traces: bool = False,
    neighbor_seed_ids: Iterable[int] | None = None,
) -> ExactRestrictedCell:
    seed_points = np.asarray(seed_points, dtype=np.float64)
    support_traces = build_support_traces_for_cell(seed_points, domain, seed_id) if include_support_traces else tuple()
    halfspaces = build_voronoi_halfspaces(seed_points, seed_id, neighbor_seed_ids=neighbor_seed_ids)
    halfspace_normals = np.asarray([halfspace.normal for halfspace in halfspaces], dtype=np.float64)
    if halfspace_normals.size == 0:
        halfspace_normals = np.empty((0, 3), dtype=np.float64)
    halfspace_rhs = np.asarray([halfspace.rhs for halfspace in halfspaces], dtype=np.float64)
    return ExactRestrictedCell(
        seed_id=int(seed_id),
        seed_point=seed_points[int(seed_id)].astype(np.float64, copy=True),
        halfspaces=halfspaces,
        support_traces=support_traces,
        halfspace_normals=halfspace_normals,
        halfspace_rhs=halfspace_rhs,
    )


def build_exact_restricted_voronoi_diagram_from_neighbor_map(
    seed_points: np.ndarray,
    domain: AnnularCylinderDomain,
    neighbor_map: dict[int, tuple[int, ...]],
    include_support_traces: bool = False,
) -> ExactRestrictedVoronoiDiagram:
    seed_points = np.asarray(seed_points, dtype=np.float64)
    cells = tuple(
        build_exact_restricted_cell(
            seed_points=seed_points,
            domain=domain,
            seed_id=seed_id,
            include_support_traces=include_support_traces,
            neighbor_seed_ids=neighbor_map.get(seed_id),
        )
        for seed_id in range(seed_points.shape[0])
    )
    return ExactRestrictedVoronoiDiagram(seed_points=seed_points, domain=domain, cells=cells)


def build_exact_restricted_voronoi_diagram(
    seed_points: np.ndarray,
    domain: AnnularCylinderDomain,
    include_support_traces: bool = False,
) -> ExactRestrictedVoronoiDiagram:
    seed_points = np.asarray(seed_points, dtype=np.float64)
    neighbor_map = build_delaunay_neighbor_map(seed_points)
    return build_exact_restricted_voronoi_diagram_from_neighbor_map(
        seed_points=seed_points,
        domain=domain,
        neighbor_map=neighbor_map,
        include_support_traces=include_support_traces,
    )


def summarize_exact_diagram(diagram: ExactRestrictedVoronoiDiagram) -> ExactDiagramSummary:
    height = diagram.domain.z_max - diagram.domain.z_min
    domain_volume = float(
        np.pi * (diagram.domain.outer_radius * diagram.domain.outer_radius - diagram.domain.inner_radius * diagram.domain.inner_radius) * height
    )
    support_curve_count = sum(cell.support_trace_count for cell in diagram.cells)
    return ExactDiagramSummary(
        num_seeds=int(diagram.seed_points.shape[0]),
        domain_volume=domain_volume,
        support_curve_count=int(support_curve_count),
    )
