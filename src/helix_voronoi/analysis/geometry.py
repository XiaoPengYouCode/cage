from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from helix_voronoi.analysis.config import ModulusAnalysisConfig, StructureStyle
from helix_voronoi.helix import HelixSpec, build_helix_centerline
from helix_voronoi.models import EdgeSegment
from helix_voronoi.rods import segment_frame


@dataclass(frozen=True)
class SegmentCloud:
    starts: np.ndarray
    ends: np.ndarray
    radius: float
    style: StructureStyle

    def contains_points(
        self, points: np.ndarray, chunk_size: int = 100_000
    ) -> np.ndarray:
        mask = np.zeros(len(points), dtype=bool)
        if len(points) == 0 or len(self.starts) == 0:
            return mask

        for offset in range(0, len(points), chunk_size):
            point_chunk = points[offset : offset + chunk_size]
            local_mask = np.zeros(len(point_chunk), dtype=bool)

            for start, end in zip(self.starts, self.ends):
                local_mask |= (
                    point_segment_distance_squared(point_chunk, start, end)
                    <= self.radius**2
                )
                if np.all(local_mask):
                    break

            mask[offset : offset + chunk_size] = local_mask

        return mask


def point_segment_distance_squared(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    segment = end - start
    segment_norm_sq = float(np.dot(segment, segment))
    if segment_norm_sq <= 1e-18:
        diff = points - start
        return np.einsum("ij,ij->i", diff, diff)

    projection = (points - start) @ segment / segment_norm_sq
    projection = np.clip(projection, 0.0, 1.0)
    closest = start + projection[:, None] * segment
    diff = points - closest
    return np.einsum("ij,ij->i", diff, diff)


def build_helix_spec(config: ModulusAnalysisConfig) -> HelixSpec:
    return HelixSpec(
        cycles_per_segment=config.helix_cycles_per_segment,
        amplitude_ratio=config.helix_amplitude_ratio,
        wire_radius_ratio=1.0,
        min_steps=config.helix_min_steps,
        steps_per_cycle=config.helix_steps_per_cycle,
    )


def build_segment_cloud_from_edges(
    edges: list[EdgeSegment],
    config: ModulusAnalysisConfig,
    style: StructureStyle,
) -> SegmentCloud:

    if style == "cylinder":
        starts = np.array([start for start, _ in edges], dtype=float)
        ends = np.array([end for _, end in edges], dtype=float)
        return SegmentCloud(
            starts=starts, ends=ends, radius=config.rod_radius, style=style
        )

    helix_spec = build_helix_spec(config)
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for start, end in edges:
        _, _, basis_u, basis_v = segment_frame(start, end)
        centerline, _ = build_helix_centerline(
            np.asarray(start, dtype=float),
            np.asarray(end, dtype=float),
            basis_u,
            basis_v,
            helix_spec,
        )
        starts.extend(centerline[:-1])
        ends.extend(centerline[1:])

    return SegmentCloud(
        starts=np.asarray(starts, dtype=float),
        ends=np.asarray(ends, dtype=float),
        radius=config.rod_radius,
        style=style,
    )


def build_segment_cloud(
    config: ModulusAnalysisConfig, style: StructureStyle
) -> SegmentCloud:
    from helix_voronoi.pipeline import VoronoiPipeline

    pipeline = VoronoiPipeline()
    row = pipeline.build_row(num_seeds=config.num_seeds, rng_seed=config.seed)
    return build_segment_cloud_from_edges(row.edges, config, style)
