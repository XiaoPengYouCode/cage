from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cage.analysis.compression import CompressionResult, solve_compression
from cage.analysis.config import ModulusAnalysisConfig
from cage.analysis.geometry import build_segment_cloud_from_edges
from cage.analysis.report import write_report
from cage.analysis.voxel import voxelize_segment_cloud
from cage.pipeline import VoronoiPipeline


@dataclass(frozen=True)
class ModulusRunSummary:
    markdown_path: Path
    json_path: Path
    results: dict[str, list[CompressionResult]]


def run_modulus_analysis(config: ModulusAnalysisConfig) -> ModulusRunSummary:
    if config.dry_run:
        raise ValueError("Dry-run mode should not execute the modulus solver.")

    results: dict[str, list[CompressionResult]] = {}
    row = VoronoiPipeline().build_row(num_seeds=config.num_seeds, rng_seed=config.seed)

    for style in config.selected_styles():
        geometry = build_segment_cloud_from_edges(row.edges, config, style)
        style_results: list[CompressionResult] = []
        for resolution in config.resolutions:
            grid = voxelize_segment_cloud(geometry, resolution, chunk_size=config.chunk_size)
            style_results.append(
                solve_compression(
                    style=style,
                    grid=grid,
                    material=config.material,
                    compression=config.compression,
                )
            )
        results[style] = style_results

    markdown_path, json_path = write_report(config, results)
    return ModulusRunSummary(markdown_path=markdown_path, json_path=json_path, results=results)
