from __future__ import annotations

from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np

from cage.helix import HelixSpec, build_helix_centerline, build_tangents, build_tube_mesh


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


class HelixRodStyle:
    name = "helix"

    def __init__(
        self,
        cycles_per_segment: float = 3.0,
        amplitude_ratio: float = 0.06,
        wire_radius_ratio: float = 1.0,
        tube_sides: int = 24,
        min_steps: int = 72,
        steps_per_cycle: int = 36,
    ) -> None:
        self.spec = HelixSpec(
            cycles_per_segment=cycles_per_segment,
            amplitude_ratio=amplitude_ratio,
            wire_radius_ratio=wire_radius_ratio,
            tube_sides=tube_sides,
            min_steps=min_steps,
            steps_per_cycle=steps_per_cycle,
        )

    def draw_segment(
        self,
        ax: plt.Axes,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        color: str,
    ) -> None:
        _, _, basis_u, basis_v = segment_frame(start, end)
        centerline, _ = build_helix_centerline(start, end, basis_u, basis_v, self.spec)
        tangents = build_tangents(centerline)
        radius_profile = np.full(len(centerline), radius * self.spec.wire_radius_ratio)
        rings = build_tube_mesh(
            centerline=centerline,
            tangents=tangents,
            radii=radius_profile,
            tube_sides=self.spec.tube_sides,
        )
        ax.plot_surface(
            rings[:, :, 0],
            rings[:, :, 1],
            rings[:, :, 2],
            color=color,
            linewidth=0.0,
            antialiased=True,
            shade=True,
        )
