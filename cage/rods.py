from __future__ import annotations

from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np


class RodStyle(Protocol):
    name: str

    def draw_segment(
        self,
        ax: plt.Axes,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        color: str,
    ) -> None: ...


class CylinderRodStyle:
    name = "cylinder"

    def draw_segment(
        self,
        ax: plt.Axes,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        color: str,
    ) -> None:
        direction = end - start
        length = np.linalg.norm(direction)
        if length <= 1e-9:
            return

        axis = direction / length
        helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        basis_u = np.cross(axis, helper)
        basis_u /= np.linalg.norm(basis_u)
        basis_v = np.cross(axis, basis_u)

        theta = np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False)
        axis_distance = np.array([0.0, length])
        theta_grid, distance_grid = np.meshgrid(theta, axis_distance)

        x = (
            start[0]
            + axis[0] * distance_grid
            + radius * (basis_u[0] * np.cos(theta_grid) + basis_v[0] * np.sin(theta_grid))
        )
        y = (
            start[1]
            + axis[1] * distance_grid
            + radius * (basis_u[1] * np.cos(theta_grid) + basis_v[1] * np.sin(theta_grid))
        )
        z = (
            start[2]
            + axis[2] * distance_grid
            + radius * (basis_u[2] * np.cos(theta_grid) + basis_v[2] * np.sin(theta_grid))
        )

        ax.plot_surface(x, y, z, color=color, linewidth=0.0, antialiased=True, shade=True)


def segment_frame(
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    direction = end - start
    length = np.linalg.norm(direction)
    if length <= 1e-9:
        raise ValueError("Cannot build a rod for a zero-length segment.")

    axis = direction / length
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(axis, helper)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)
    return axis, length, basis_u, basis_v


def draw_tube_along_curve(
    ax: plt.Axes,
    centerline: np.ndarray,
    tangents: np.ndarray,
    radii: np.ndarray,
    color: str,
    tube_sides: int,
) -> None:
    thetas = np.linspace(0.0, 2.0 * np.pi, tube_sides, endpoint=False)
    ring_points: list[np.ndarray] = []
    helper = np.array([0.0, 0.0, 1.0])

    for index, tangent in enumerate(tangents):
        local_helper = helper if abs(np.dot(tangent, helper)) < 0.9 else np.array([0.0, 1.0, 0.0])
        normal = np.cross(tangent, local_helper)
        normal /= np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)
        ring = np.array(
            [
                radii[index] * (np.cos(theta) * normal + np.sin(theta) * binormal)
                for theta in thetas
            ]
        )
        ring_points.append(ring)

    rings = centerline[:, None, :] + np.array(ring_points)
    ax.plot_surface(
        rings[:, :, 0],
        rings[:, :, 1],
        rings[:, :, 2],
        color=color,
        linewidth=0.0,
        antialiased=True,
        shade=True,
    )


class HelixRodStyle:
    name = "helix"

    def __init__(
        self,
        cycles_per_segment: float = 3.0,
        amplitude_ratio: float = 0.06,
        wire_radius_ratio: float = 0.5,
        tube_sides: int = 10,
        min_steps: int = 48,
        steps_per_cycle: int = 24,
    ) -> None:
        self.cycles_per_segment = cycles_per_segment
        self.amplitude_ratio = amplitude_ratio
        self.wire_radius_ratio = wire_radius_ratio
        self.tube_sides = tube_sides
        self.min_steps = min_steps
        self.steps_per_cycle = steps_per_cycle

    def draw_segment(
        self,
        ax: plt.Axes,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        color: str,
    ) -> None:
        axis, length, basis_u, basis_v = segment_frame(start, end)
        steps = max(self.min_steps, int(np.ceil(self.cycles_per_segment * self.steps_per_cycle)))
        progress = np.linspace(0.0, 1.0, steps)
        phase = np.linspace(0.0, 2.0 * np.pi * self.cycles_per_segment, steps)
        travel = progress * length

        # Force only the helix offset to collapse to zero at the ends so the
        # centerline stays locked to the shared Voronoi node positions.
        endpoint_envelope = np.sin(np.pi * progress) ** 2
        coil_radius = self.amplitude_ratio * length
        wire_radius = radius * self.wire_radius_ratio
        offsets = (
            np.outer(np.cos(phase), basis_u) + np.outer(np.sin(phase), basis_v)
        ) * (coil_radius * endpoint_envelope)[:, None]
        centerline = start + np.outer(travel, axis) + offsets

        tangents = np.gradient(centerline, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
        radius_profile = np.full(steps, wire_radius)
        draw_tube_along_curve(
            ax,
            centerline=centerline,
            tangents=tangents,
            radii=radius_profile,
            color=color,
            tube_sides=self.tube_sides,
        )
