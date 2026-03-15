from __future__ import annotations

import numpy as np

from cage.models import PipelineConfig, RowGeometry
from cage.voronoi import build_voronoi_cells, extract_unique_edges, generate_seeds


class SeedSamplingNode:
    name = "seed_sampling"

    def run(self, num_seeds: int, rng_seed: int) -> np.ndarray:
        return generate_seeds(num_seeds, rng_seed)


class VoronoiBuildNode:
    name = "voronoi_build"

    def run(self, seeds: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return build_voronoi_cells(seeds)


class EdgeExtractionNode:
    name = "edge_extraction"

    def run(
        self,
        cells: list[np.ndarray],
        halfspace_sets: list[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return extract_unique_edges(cells, halfspace_sets)


class VoronoiPipeline:
    def __init__(
        self,
        seed_sampler: SeedSamplingNode | None = None,
        voronoi_builder: VoronoiBuildNode | None = None,
        edge_extractor: EdgeExtractionNode | None = None,
    ) -> None:
        self.seed_sampler = seed_sampler or SeedSamplingNode()
        self.voronoi_builder = voronoi_builder or VoronoiBuildNode()
        self.edge_extractor = edge_extractor or EdgeExtractionNode()

    def build_rows(self, config: PipelineConfig) -> list[RowGeometry]:
        rows: list[RowGeometry] = []

        for rng_seed in config.row_seeds:
            seeds = self.seed_sampler.run(config.num_seeds, rng_seed)
            cells, halfspace_sets = self.voronoi_builder.run(seeds)
            edges = self.edge_extractor.run(cells, halfspace_sets)
            rows.append(
                RowGeometry(
                    rng_seed=rng_seed,
                    seeds=seeds,
                    cells=cells,
                    halfspace_sets=halfspace_sets,
                    edges=edges,
                )
            )

        return rows
