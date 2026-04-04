from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

EdgeSegment = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class RenderConfig:
    output_path: Path
    show: bool
    tile_repeats: tuple[int, int, int] = (3, 3, 3)
    rod_radius: float = 0.012
    tiled_rod_radius: float = 0.028


@dataclass(frozen=True)
class PipelineConfig:
    num_seeds: int
    row_seeds: tuple[int, ...]
    render: RenderConfig


@dataclass
class RowGeometry:
    rng_seed: int
    seeds: np.ndarray
    cells: list[np.ndarray]
    halfspace_sets: list[np.ndarray]
    edges: list[EdgeSegment]
