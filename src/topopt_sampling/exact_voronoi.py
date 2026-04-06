from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class BoundaryCurve3D:
    surface_name: str
    seed_pair: tuple[int, int]
    points: np.ndarray


@dataclass(frozen=True)
class SurfaceCandidates:
    active_seed_ids: np.ndarray
    candidate_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SurfacePatchMesh:
    surface_name: str
    patches: list[np.ndarray]
    seed_ids: np.ndarray


def build_surface_voronoi_patches(
    seed_points: np.ndarray,
    xy_size: int,
    z_size: int,
    outer_radius: float,
    inner_radius: float,
) -> list[SurfacePatchMesh]:
    center_xy = np.array([(xy_size - 1) / 2.0, (xy_size - 1) / 2.0], dtype=np.float64)
    z_min = 0.5
    z_max = z_size - 0.5

    meshes = [
        build_cylinder_surface_patch_mesh(
            seed_points=seed_points,
            center_xy=center_xy,
            radius=outer_radius,
            z_min=z_min,
            z_max=z_max,
            surface_name="outer_cylinder",
        )
    ]
    if inner_radius > 0.0:
        meshes.append(
            build_cylinder_surface_patch_mesh(
                seed_points=seed_points,
                center_xy=center_xy,
                radius=inner_radius,
                z_min=z_min,
                z_max=z_max,
                surface_name="inner_cylinder",
            )
        )

    meshes.append(
        build_cap_surface_patch_mesh(
            seed_points=seed_points,
            center_xy=center_xy,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            z_value=z_max,
            surface_name="top_cap",
        )
    )
    meshes.append(
        build_cap_surface_patch_mesh(
            seed_points=seed_points,
            center_xy=center_xy,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            z_value=z_min,
            surface_name="bottom_cap",
        )
    )
    return meshes


def build_exact_boundary_curves(
    seed_points: np.ndarray,
    xy_size: int,
    z_size: int,
    outer_radius: float,
    inner_radius: float,
) -> list[BoundaryCurve3D]:
    center_xy = np.array([(xy_size - 1) / 2.0, (xy_size - 1) / 2.0], dtype=np.float64)
    z_min = 0.5
    z_max = z_size - 0.5

    curves: list[BoundaryCurve3D] = []

    outer_candidates = discover_cylinder_candidates(
        seed_points=seed_points,
        center_xy=center_xy,
        radius=outer_radius,
        z_min=z_min,
        z_max=z_max,
    )
    curves.extend(
        build_cylinder_surface_boundary_curves(
            seed_points=seed_points,
            center_xy=center_xy,
            radius=outer_radius,
            z_min=z_min,
            z_max=z_max,
            surface_name="outer_cylinder",
            active_seed_ids=outer_candidates.active_seed_ids,
            candidate_pairs=outer_candidates.candidate_pairs,
        )
    )

    if inner_radius > 0.0:
        inner_candidates = discover_cylinder_candidates(
            seed_points=seed_points,
            center_xy=center_xy,
            radius=inner_radius,
            z_min=z_min,
            z_max=z_max,
        )
        curves.extend(
            build_cylinder_surface_boundary_curves(
                seed_points=seed_points,
                center_xy=center_xy,
                radius=inner_radius,
                z_min=z_min,
                z_max=z_max,
                surface_name="inner_cylinder",
                active_seed_ids=inner_candidates.active_seed_ids,
                candidate_pairs=inner_candidates.candidate_pairs,
            )
        )

    top_candidates = discover_cap_candidates(
        seed_points=seed_points,
        center_xy=center_xy,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        z_value=z_max,
    )
    curves.extend(
        build_cap_surface_boundary_curves(
            seed_points=seed_points,
            center_xy=center_xy,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            z_value=z_max,
            surface_name="top_cap",
            active_seed_ids=top_candidates.active_seed_ids,
            candidate_pairs=top_candidates.candidate_pairs,
        )
    )

    bottom_candidates = discover_cap_candidates(
        seed_points=seed_points,
        center_xy=center_xy,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        z_value=z_min,
    )
    curves.extend(
        build_cap_surface_boundary_curves(
            seed_points=seed_points,
            center_xy=center_xy,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            z_value=z_min,
            surface_name="bottom_cap",
            active_seed_ids=bottom_candidates.active_seed_ids,
            candidate_pairs=bottom_candidates.candidate_pairs,
        )
    )
    return curves


def build_cylinder_surface_patch_mesh(
    seed_points: np.ndarray,
    center_xy: np.ndarray,
    radius: float,
    z_min: float,
    z_max: float,
    surface_name: str,
) -> SurfacePatchMesh:
    theta_steps = max(240, int(math.ceil(radius * 6.0)))
    z_steps = max(96, int(math.ceil((z_max - z_min) * 2.4)))
    theta_edges = np.linspace(0.0, 2.0 * np.pi, theta_steps + 1, dtype=np.float64)
    z_edges = np.linspace(z_min, z_max, z_steps + 1, dtype=np.float64)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    theta_grid, z_grid = np.meshgrid(theta_centers, z_centers, indexing="ij")
    sample_points = np.column_stack(
        (
            center_xy[0] + radius * np.cos(theta_grid).ravel(),
            center_xy[1] + radius * np.sin(theta_grid).ravel(),
            z_grid.ravel(),
        )
    )
    seed_ids = _assign_points_to_nearest_seed(sample_points, seed_points).reshape(theta_steps, z_steps)

    cos_edges = np.cos(theta_edges)
    sin_edges = np.sin(theta_edges)
    patches: list[np.ndarray] = []
    patch_seed_ids: list[int] = []
    for theta_index in range(theta_steps):
        next_theta_index = theta_index + 1
        x0 = center_xy[0] + radius * cos_edges[theta_index]
        y0 = center_xy[1] + radius * sin_edges[theta_index]
        x1 = center_xy[0] + radius * cos_edges[next_theta_index]
        y1 = center_xy[1] + radius * sin_edges[next_theta_index]
        for z_index in range(z_steps):
            z0 = z_edges[z_index]
            z1 = z_edges[z_index + 1]
            patches.append(
                np.array(
                    [
                        [x0, y0, z0],
                        [x1, y1, z0],
                        [x1, y1, z1],
                        [x0, y0, z1],
                    ],
                    dtype=np.float32,
                )
            )
            patch_seed_ids.append(int(seed_ids[theta_index, z_index]))

    return SurfacePatchMesh(
        surface_name=surface_name,
        patches=patches,
        seed_ids=np.asarray(patch_seed_ids, dtype=np.int32),
    )


def build_cap_surface_patch_mesh(
    seed_points: np.ndarray,
    center_xy: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    z_value: float,
    surface_name: str,
) -> SurfacePatchMesh:
    theta_steps = max(240, int(math.ceil(outer_radius * 6.0)))
    radial_steps = max(48, int(math.ceil(max(outer_radius - inner_radius, 1.0) * 2.0)))
    theta_edges = np.linspace(0.0, 2.0 * np.pi, theta_steps + 1, dtype=np.float64)
    radial_edges = np.linspace(inner_radius, outer_radius, radial_steps + 1, dtype=np.float64)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])

    radial_grid, theta_grid = np.meshgrid(radial_centers, theta_centers, indexing="ij")
    sample_points = np.column_stack(
        (
            center_xy[0] + radial_grid.ravel() * np.cos(theta_grid).ravel(),
            center_xy[1] + radial_grid.ravel() * np.sin(theta_grid).ravel(),
            np.full(radial_grid.size, z_value, dtype=np.float64),
        )
    )
    seed_ids = _assign_points_to_nearest_seed(sample_points, seed_points).reshape(radial_steps, theta_steps)

    cos_edges = np.cos(theta_edges)
    sin_edges = np.sin(theta_edges)
    patches: list[np.ndarray] = []
    patch_seed_ids: list[int] = []
    for radial_index in range(radial_steps):
        r0 = radial_edges[radial_index]
        r1 = radial_edges[radial_index + 1]
        for theta_index in range(theta_steps):
            next_theta_index = theta_index + 1
            patches.append(
                np.array(
                    [
                        [center_xy[0] + r0 * cos_edges[theta_index], center_xy[1] + r0 * sin_edges[theta_index], z_value],
                        [center_xy[0] + r1 * cos_edges[theta_index], center_xy[1] + r1 * sin_edges[theta_index], z_value],
                        [center_xy[0] + r1 * cos_edges[next_theta_index], center_xy[1] + r1 * sin_edges[next_theta_index], z_value],
                        [center_xy[0] + r0 * cos_edges[next_theta_index], center_xy[1] + r0 * sin_edges[next_theta_index], z_value],
                    ],
                    dtype=np.float32,
                )
            )
            patch_seed_ids.append(int(seed_ids[radial_index, theta_index]))

    return SurfacePatchMesh(
        surface_name=surface_name,
        patches=patches,
        seed_ids=np.asarray(patch_seed_ids, dtype=np.int32),
    )


def discover_cylinder_candidates(
    seed_points: np.ndarray,
    center_xy: np.ndarray,
    radius: float,
    z_min: float,
    z_max: float,
) -> SurfaceCandidates:
    theta_samples = max(240, int(math.ceil(radius * 6.0)))
    z_samples = max(96, int(math.ceil((z_max - z_min) * 3.0)))
    theta = np.linspace(0.0, 2.0 * np.pi, theta_samples, endpoint=False, dtype=np.float64)
    z_coords = np.linspace(z_min, z_max, z_samples, dtype=np.float64)
    theta_grid, z_grid = np.meshgrid(theta, z_coords, indexing="ij")
    sample_points = np.column_stack(
        (
            center_xy[0] + radius * np.cos(theta_grid).ravel(),
            center_xy[1] + radius * np.sin(theta_grid).ravel(),
            z_grid.ravel(),
        )
    )
    labels = _assign_points_to_nearest_seed(sample_points, seed_points).reshape(theta_samples, z_samples)
    return SurfaceCandidates(
        active_seed_ids=np.unique(labels).astype(np.int32, copy=False),
        candidate_pairs=_discover_adjacent_pairs(labels, periodic_axes=(0,)),
    )


def discover_cap_candidates(
    seed_points: np.ndarray,
    center_xy: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    z_value: float,
) -> SurfaceCandidates:
    theta_samples = max(240, int(math.ceil(outer_radius * 6.0)))
    radial_samples = max(64, int(math.ceil(max(outer_radius - inner_radius, 1.0) * 4.0)))
    theta = np.linspace(0.0, 2.0 * np.pi, theta_samples, endpoint=False, dtype=np.float64)
    radial = np.linspace(inner_radius, outer_radius, radial_samples, dtype=np.float64)
    radial_grid, theta_grid = np.meshgrid(radial, theta, indexing="ij")
    sample_points = np.column_stack(
        (
            center_xy[0] + radial_grid.ravel() * np.cos(theta_grid).ravel(),
            center_xy[1] + radial_grid.ravel() * np.sin(theta_grid).ravel(),
            np.full(radial_grid.size, z_value, dtype=np.float64),
        )
    )
    labels = _assign_points_to_nearest_seed(sample_points, seed_points).reshape(radial_samples, theta_samples)
    return SurfaceCandidates(
        active_seed_ids=np.unique(labels).astype(np.int32, copy=False),
        candidate_pairs=_discover_adjacent_pairs(labels, periodic_axes=(1,)),
    )


def build_cylinder_surface_boundary_curves(
    seed_points: np.ndarray,
    center_xy: np.ndarray,
    radius: float,
    z_min: float,
    z_max: float,
    surface_name: str,
    active_seed_ids: np.ndarray,
    candidate_pairs: tuple[tuple[int, int], ...],
) -> list[BoundaryCurve3D]:
    if len(active_seed_ids) < 2 or not candidate_pairs:
        return []

    active_seed_set = {int(seed_id) for seed_id in active_seed_ids}
    curves: list[BoundaryCurve3D] = []
    for seed_i, seed_j in candidate_pairs:
        if seed_i not in active_seed_set or seed_j not in active_seed_set:
            continue

        pair_coeff = cylinder_pair_coefficients(
            seed_i=seed_points[seed_i],
            seed_j=seed_points[seed_j],
            center_xy=center_xy,
            radius=radius,
        )
        a_ij, b_ij, c_ij, d_ij = pair_coeff

        if abs(c_ij) <= 1e-9:
            curves.extend(
                _build_vertical_cylinder_boundaries(
                    seed_points=seed_points,
                    center_xy=center_xy,
                    radius=radius,
                    z_min=z_min,
                    z_max=z_max,
                    surface_name=surface_name,
                    active_seed_ids=active_seed_ids,
                    seed_i=seed_i,
                    seed_j=seed_j,
                    pair_coeff=pair_coeff,
                )
            )
            continue

        inequalities: list[tuple[float, float, float]] = [
            (a_ij / c_ij, b_ij / c_ij, d_ij / c_ij + z_min),
            (-a_ij / c_ij, -b_ij / c_ij, -d_ij / c_ij - z_max),
        ]
        for seed_k in active_seed_ids:
            if int(seed_k) in (seed_i, seed_j):
                continue
            a_ik, b_ik, c_ik, d_ik = cylinder_pair_coefficients(
                seed_i=seed_points[seed_i],
                seed_j=seed_points[int(seed_k)],
                center_xy=center_xy,
                radius=radius,
            )
            inequalities.append(
                (
                    a_ik - (c_ik * a_ij / c_ij),
                    b_ik - (c_ik * b_ij / c_ij),
                    d_ik - (c_ik * d_ij / c_ij),
                )
            )

        for theta_start, theta_stop in solve_periodic_trig_inequalities(inequalities):
            num_points = max(32, int(math.ceil((theta_stop - theta_start) / (2.0 * np.pi) * 240.0)))
            theta_values = np.linspace(theta_start, theta_stop, num_points, dtype=np.float64)
            z_values = -(
                a_ij * np.cos(theta_values) + b_ij * np.sin(theta_values) + d_ij
            ) / c_ij
            points = np.column_stack(
                (
                    center_xy[0] + radius * np.cos(theta_values),
                    center_xy[1] + radius * np.sin(theta_values),
                    z_values,
                )
            ).astype(np.float32)
            curves.append(BoundaryCurve3D(surface_name=surface_name, seed_pair=(seed_i, seed_j), points=points))
    return curves


def build_cap_surface_boundary_curves(
    seed_points: np.ndarray,
    center_xy: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    z_value: float,
    surface_name: str,
    active_seed_ids: np.ndarray,
    candidate_pairs: tuple[tuple[int, int], ...],
) -> list[BoundaryCurve3D]:
    if len(active_seed_ids) < 2 or not candidate_pairs:
        return []

    active_seed_set = {int(seed_id) for seed_id in active_seed_ids}
    curves: list[BoundaryCurve3D] = []
    for seed_i, seed_j in candidate_pairs:
        if seed_i not in active_seed_set or seed_j not in active_seed_set:
            continue

        a_ij, b_ij, c_ij = plane_pair_coefficients(
            seed_i=seed_points[seed_i],
            seed_j=seed_points[seed_j],
            center_xy=center_xy,
            z_value=z_value,
        )
        normal_sq = a_ij * a_ij + b_ij * b_ij
        if normal_sq <= 1e-12:
            continue

        normal = np.array([a_ij, b_ij], dtype=np.float64)
        direction = np.array([-b_ij, a_ij], dtype=np.float64) / math.sqrt(normal_sq)
        offset = -(c_ij / normal_sq) * normal
        radial_sq = float(np.dot(offset, offset))
        outer_span_sq = outer_radius * outer_radius - radial_sq
        if outer_span_sq <= 1e-12:
            continue
        outer_span = math.sqrt(max(outer_span_sq, 0.0))
        segments: list[tuple[float, float]] = [(-outer_span, outer_span)]

        if inner_radius > 0.0:
            inner_span_sq = inner_radius * inner_radius - radial_sq
            if inner_span_sq > 1e-12:
                inner_span = math.sqrt(inner_span_sq)
                next_segments: list[tuple[float, float]] = []
                if -outer_span < -inner_span:
                    next_segments.append((-outer_span, -inner_span))
                if inner_span < outer_span:
                    next_segments.append((inner_span, outer_span))
                segments = next_segments
            elif radial_sq < inner_radius * inner_radius:
                continue

        for seed_k in active_seed_ids:
            if int(seed_k) in (seed_i, seed_j):
                continue
            a_ik, b_ik, c_ik = plane_pair_coefficients(
                seed_i=seed_points[seed_i],
                seed_j=seed_points[int(seed_k)],
                center_xy=center_xy,
                z_value=z_value,
            )
            next_segments = []
            for seg_start, seg_stop in segments:
                clipped = _clip_linear_interval(
                    seg_start,
                    seg_stop,
                    alpha=a_ik * direction[0] + b_ik * direction[1],
                    beta=a_ik * offset[0] + b_ik * offset[1] + c_ik,
                )
                if clipped is not None:
                    next_segments.append(clipped)
            segments = next_segments
            if not segments:
                break

        for seg_start, seg_stop in segments:
            if seg_stop - seg_start <= 1e-6:
                continue
            local_points = np.vstack((offset + seg_start * direction, offset + seg_stop * direction))
            world_points = np.column_stack(
                (
                    center_xy[0] + local_points[:, 0],
                    center_xy[1] + local_points[:, 1],
                    np.full(2, z_value, dtype=np.float64),
                )
            ).astype(np.float32)
            curves.append(BoundaryCurve3D(surface_name=surface_name, seed_pair=(seed_i, seed_j), points=world_points))
    return curves


def cylinder_pair_coefficients(
    seed_i: np.ndarray,
    seed_j: np.ndarray,
    center_xy: np.ndarray,
    radius: float,
) -> tuple[float, float, float, float]:
    dx_i = float(seed_i[0] - center_xy[0])
    dy_i = float(seed_i[1] - center_xy[1])
    dz_i = float(seed_i[2])
    dx_j = float(seed_j[0] - center_xy[0])
    dy_j = float(seed_j[1] - center_xy[1])
    dz_j = float(seed_j[2])
    return (
        -2.0 * radius * (dx_i - dx_j),
        -2.0 * radius * (dy_i - dy_j),
        -2.0 * (dz_i - dz_j),
        dx_i * dx_i + dy_i * dy_i + dz_i * dz_i - dx_j * dx_j - dy_j * dy_j - dz_j * dz_j,
    )


def plane_pair_coefficients(
    seed_i: np.ndarray,
    seed_j: np.ndarray,
    center_xy: np.ndarray,
    z_value: float,
) -> tuple[float, float, float]:
    x_i = float(seed_i[0] - center_xy[0])
    y_i = float(seed_i[1] - center_xy[1])
    z_i = float(seed_i[2])
    x_j = float(seed_j[0] - center_xy[0])
    y_j = float(seed_j[1] - center_xy[1])
    z_j = float(seed_j[2])
    return (
        -2.0 * (x_i - x_j),
        -2.0 * (y_i - y_j),
        x_i * x_i + y_i * y_i + z_i * z_i - x_j * x_j - y_j * y_j - z_j * z_j - 2.0 * z_value * (z_i - z_j),
    )


def solve_periodic_trig_inequalities(
    inequalities: list[tuple[float, float, float]],
) -> list[tuple[float, float]]:
    roots = [0.0, 2.0 * np.pi]
    for a_value, b_value, c_value in inequalities:
        roots.extend(trig_roots(a_value, b_value, c_value))

    roots = _dedupe_sorted_values(roots, period=2.0 * np.pi)
    intervals: list[tuple[float, float]] = []
    for start, stop in zip(roots[:-1], roots[1:], strict=False):
        if stop - start <= 1e-9:
            continue
        mid = 0.5 * (start + stop)
        if all(_evaluate_trig_inequality(a_value, b_value, c_value, mid) <= 1e-9 for a_value, b_value, c_value in inequalities):
            intervals.append((start, stop))
    return intervals


def trig_roots(a_value: float, b_value: float, c_value: float) -> list[float]:
    amplitude = math.hypot(a_value, b_value)
    if amplitude <= 1e-12:
        return []

    ratio = -c_value / amplitude
    if ratio < -1.0 - 1e-9 or ratio > 1.0 + 1e-9:
        return []

    ratio = min(1.0, max(-1.0, ratio))
    phase = math.atan2(b_value, a_value)
    angle = math.acos(ratio)
    roots = [
        (phase + angle) % (2.0 * np.pi),
        (phase - angle) % (2.0 * np.pi),
    ]
    return _dedupe_sorted_values(roots, period=2.0 * np.pi)


def build_hollow_cylinder_outline(
    xy_size: int,
    z_size: int,
    outer_radius: float,
    inner_radius: float,
) -> list[np.ndarray]:
    center_xy = np.array([(xy_size - 1) / 2.0, (xy_size - 1) / 2.0], dtype=np.float64)
    z_min = 0.5
    z_max = z_size - 0.5
    theta = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=True, dtype=np.float64)
    segments: list[np.ndarray] = []

    for radius in (outer_radius, inner_radius):
        if radius <= 0.0:
            continue
        circle_xy = np.column_stack(
            (
                center_xy[0] + radius * np.cos(theta),
                center_xy[1] + radius * np.sin(theta),
            )
        )
        segments.append(
            np.column_stack((circle_xy, np.full(theta.size, z_min, dtype=np.float64))).astype(np.float32)
        )
        segments.append(
            np.column_stack((circle_xy, np.full(theta.size, z_max, dtype=np.float64))).astype(np.float32)
        )

    for theta_value in np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False, dtype=np.float64):
        for radius in (outer_radius, inner_radius):
            if radius <= 0.0:
                continue
            x_coord = center_xy[0] + radius * math.cos(float(theta_value))
            y_coord = center_xy[1] + radius * math.sin(float(theta_value))
            segments.append(
                np.array(
                    [
                        [x_coord, y_coord, z_min],
                        [x_coord, y_coord, z_max],
                    ],
                    dtype=np.float32,
                )
            )
    return segments


def _build_vertical_cylinder_boundaries(
    seed_points: np.ndarray,
    center_xy: np.ndarray,
    radius: float,
    z_min: float,
    z_max: float,
    surface_name: str,
    active_seed_ids: np.ndarray,
    seed_i: int,
    seed_j: int,
    pair_coeff: tuple[float, float, float, float],
) -> list[BoundaryCurve3D]:
    a_ij, b_ij, _, d_ij = pair_coeff
    curves: list[BoundaryCurve3D] = []
    for theta_value in trig_roots(a_ij, b_ij, d_ij):
        cos_theta = math.cos(theta_value)
        sin_theta = math.sin(theta_value)
        z_interval: tuple[float, float] | None = (z_min, z_max)
        for seed_k in active_seed_ids:
            if int(seed_k) in (seed_i, seed_j):
                continue
            a_ik, b_ik, c_ik, d_ik = cylinder_pair_coefficients(
                seed_i=seed_points[seed_i],
                seed_j=seed_points[int(seed_k)],
                center_xy=center_xy,
                radius=radius,
            )
            z_interval = _clip_linear_interval(
                z_interval[0],
                z_interval[1],
                alpha=c_ik,
                beta=a_ik * cos_theta + b_ik * sin_theta + d_ik,
            ) if z_interval is not None else None
            if z_interval is None:
                break
        if z_interval is None or z_interval[1] - z_interval[0] <= 1e-6:
            continue
        x_coord = center_xy[0] + radius * cos_theta
        y_coord = center_xy[1] + radius * sin_theta
        points = np.array(
            [
                [x_coord, y_coord, z_interval[0]],
                [x_coord, y_coord, z_interval[1]],
            ],
            dtype=np.float32,
        )
        curves.append(BoundaryCurve3D(surface_name=surface_name, seed_pair=(seed_i, seed_j), points=points))
    return curves


def _assign_points_to_nearest_seed(points: np.ndarray, seed_points: np.ndarray) -> np.ndarray:
    tree = cKDTree(seed_points)
    _, nearest_seed_indices = tree.query(points, k=1)
    return nearest_seed_indices.astype(np.int32)


def _discover_adjacent_pairs(labels: np.ndarray, periodic_axes: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    pairs: set[tuple[int, int]] = set()
    ndim = labels.ndim
    for axis in range(ndim):
        shifted = np.roll(labels, -1, axis=axis)
        if axis not in periodic_axes:
            slicer = [slice(None)] * ndim
            slicer[axis] = slice(0, -1)
            current = labels[tuple(slicer)]
            neighbor = shifted[tuple(slicer)]
        else:
            current = labels
            neighbor = shifted

        mask = current != neighbor
        if not np.any(mask):
            continue

        left = current[mask].astype(np.int32, copy=False)
        right = neighbor[mask].astype(np.int32, copy=False)
        mins = np.minimum(left, right)
        maxs = np.maximum(left, right)
        pairs.update(zip(mins.tolist(), maxs.tolist(), strict=False))

    return tuple(sorted(pairs))


def _clip_linear_interval(lower: float, upper: float, alpha: float, beta: float) -> tuple[float, float] | None:
    if upper - lower <= 1e-12:
        return None
    if abs(alpha) <= 1e-12:
        return (lower, upper) if beta <= 1e-12 else None

    boundary = -beta / alpha
    if alpha > 0.0:
        upper = min(upper, boundary)
    else:
        lower = max(lower, boundary)
    if upper - lower <= 1e-9:
        return None
    return (lower, upper)


def _dedupe_sorted_values(values: list[float], period: float) -> list[float]:
    normalized = sorted(float(value % period) for value in values if 0.0 <= float(value % period) <= period)
    result: list[float] = []
    for value in normalized:
        if result and abs(value - result[-1]) <= 1e-8:
            continue
        if value >= period - 1e-8:
            continue
        result.append(value)
    if not result or result[0] > 1e-8:
        result.insert(0, 0.0)
    if abs(result[-1] - period) > 1e-8:
        result.append(period)
    return result


def _evaluate_trig_inequality(a_value: float, b_value: float, c_value: float, theta: float) -> float:
    return a_value * math.cos(theta) + b_value * math.sin(theta) + c_value
