from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize_vector(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm > 1e-9:
        return vector / norm
    if fallback is not None:
        return normalize_vector(fallback)
    raise ValueError("Cannot normalize a zero-length vector without a fallback.")


def initial_frame_normal(tangent: np.ndarray) -> np.ndarray:
    helper = np.array([0.0, 0.0, 1.0]) if abs(tangent[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return normalize_vector(np.cross(tangent, helper))


def build_transport_frames(tangents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = np.zeros_like(tangents)
    binormals = np.zeros_like(tangents)

    normals[0] = initial_frame_normal(tangents[0])
    binormals[0] = normalize_vector(np.cross(tangents[0], normals[0]))

    for index in range(1, len(tangents)):
        projected_normal = normals[index - 1] - np.dot(normals[index - 1], tangents[index]) * tangents[index]
        if np.linalg.norm(projected_normal) <= 1e-9:
            projected_normal = np.cross(binormals[index - 1], tangents[index])
        normals[index] = normalize_vector(projected_normal, fallback=initial_frame_normal(tangents[index]))
        binormals[index] = normalize_vector(np.cross(tangents[index], normals[index]))

    return normals, binormals


@dataclass(frozen=True)
class HelixSpec:
    cycles_per_segment: float = 3.0
    amplitude_ratio: float = 0.06
    wire_radius_ratio: float = 1.0
    tube_sides: int = 24
    min_steps: int = 72
    steps_per_cycle: int = 36


def build_helix_centerline(
    start: np.ndarray,
    end: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    spec: HelixSpec,
) -> tuple[np.ndarray, np.ndarray]:
    direction = end - start
    length = np.linalg.norm(direction)
    if length <= 1e-9:
        raise ValueError("Cannot build a helix for a zero-length segment.")

    axis = direction / length
    steps = max(spec.min_steps, int(np.ceil(spec.cycles_per_segment * spec.steps_per_cycle)))
    progress = np.linspace(0.0, 1.0, steps)
    phase = np.linspace(0.0, 2.0 * np.pi * spec.cycles_per_segment, steps)
    travel = progress * length

    # Only the helix offset collapses at the ends; the rod thickness stays constant.
    endpoint_envelope = np.sin(np.pi * progress) ** 2
    coil_radius = spec.amplitude_ratio * length
    offsets = (
        np.outer(np.cos(phase), basis_u) + np.outer(np.sin(phase), basis_v)
    ) * (coil_radius * endpoint_envelope)[:, None]
    centerline = start + np.outer(travel, axis) + offsets
    return centerline, progress


def build_tangents(centerline: np.ndarray) -> np.ndarray:
    tangents = np.gradient(centerline, axis=0)
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return tangents / np.maximum(tangent_norms, 1e-9)


def build_tube_mesh(
    centerline: np.ndarray,
    tangents: np.ndarray,
    radii: np.ndarray,
    tube_sides: int,
) -> np.ndarray:
    thetas = np.linspace(0.0, 2.0 * np.pi, tube_sides, endpoint=False)
    normals, binormals = build_transport_frames(tangents)
    rings = []

    for index in range(len(tangents)):
        normal = normals[index]
        binormal = binormals[index]
        ring = np.array(
            [
                radii[index] * (np.cos(theta) * normal + np.sin(theta) * binormal)
                for theta in thetas
            ]
        )
        rings.append(ring)

    return centerline[:, None, :] + np.array(rings)
