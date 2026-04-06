from __future__ import annotations

import json
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np

from topopt_sampling.demo import generate_annular_cylinder_npz, generate_fake_density_result
from topopt_sampling.exact_restricted_voronoi_3d import (
    ExactRestrictedVoronoiDiagram,
    build_annular_cylinder_domain,
    build_exact_restricted_cell,
)
from topopt_sampling.neighbors import build_delaunay_neighbor_map
from topopt_sampling.probability import (
    density_to_probability_intensity,
    load_density_input,
    sample_seed_points,
    save_seed_mapping_result,
)
from topopt_sampling.threejs_glb_export import (
    build_hybrid_exact_diagram_brep_from_diagram,
    serialize_threejs_shell_glb,
)

T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkStage:
    name: str
    wall_seconds: float


@dataclass(frozen=True)
class EndToEndBenchmarkResult:
    voxel_npz: Path
    density_npz: Path
    seed_npz: Path
    glb_path: Path
    xy_size: int
    z_size: int
    outer_radius: float
    inner_radius: float
    num_seeds: int
    gamma: float
    rng_seed: int
    stages: tuple[BenchmarkStage, ...]
    total_wall_seconds: float
    total_user_seconds: float
    total_sys_seconds: float
    output_bytes: int
    num_cells: int
    num_exported_cells: int
    num_shell_cells: int
    num_faces: int
    num_triangles: int
    num_boundaries: int

    def to_dict(self) -> dict[str, object]:
        return {
            "voxel_npz": str(self.voxel_npz),
            "density_npz": str(self.density_npz),
            "seed_npz": str(self.seed_npz),
            "glb_path": str(self.glb_path),
            "xy_size": self.xy_size,
            "z_size": self.z_size,
            "outer_radius": self.outer_radius,
            "inner_radius": self.inner_radius,
            "num_seeds": self.num_seeds,
            "gamma": self.gamma,
            "rng_seed": self.rng_seed,
            "stages": [
                {"name": stage.name, "wall_seconds": round(stage.wall_seconds, 6)}
                for stage in self.stages
            ],
            "total_wall_seconds": round(self.total_wall_seconds, 6),
            "total_user_seconds": round(self.total_user_seconds, 6),
            "total_sys_seconds": round(self.total_sys_seconds, 6),
            "output_bytes": self.output_bytes,
            "num_cells": self.num_cells,
            "num_exported_cells": self.num_exported_cells,
            "num_shell_cells": self.num_shell_cells,
            "num_faces": self.num_faces,
            "num_triangles": self.num_triangles,
            "num_boundaries": self.num_boundaries,
        }

    def to_markdown(self) -> str:
        lines = [
            "# End-to-end benchmark",
            "",
            "## Outputs",
            "",
            f"- voxel npz: `{self.voxel_npz}`",
            f"- fake density npz: `{self.density_npz}`",
            f"- seed npz: `{self.seed_npz}`",
            f"- GLB: `{self.glb_path}`",
            f"- GLB bytes: {self.output_bytes:,}",
            f"- cells: {self.num_cells}",
            f"- exported_cells: {self.num_exported_cells}",
            f"- shell_cells: {self.num_shell_cells}",
            f"- faces: {self.num_faces}",
            f"- triangles: {self.num_triangles}",
            f"- boundaries: {self.num_boundaries}",
            "",
            "## Timing",
            "",
            "| Stage | Wall time |",
            "| --- | ---: |",
        ]
        for stage in self.stages:
            lines.append(f"| {stage.name} | {stage.wall_seconds:.2f}s |")
        lines.extend(
            [
                f"| total | {self.total_wall_seconds:.2f}s |",
                "",
                "## CPU",
                "",
                f"- user: {self.total_user_seconds:.2f}s",
                f"- sys: {self.total_sys_seconds:.2f}s",
            ]
        )
        return "\n".join(lines)


class _StageTimer:
    def __init__(self) -> None:
        self.stages: list[BenchmarkStage] = []

    def measure(self, name: str, func: Callable[[], T]) -> T:
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        self.stages.append(BenchmarkStage(name=name, wall_seconds=elapsed))
        return result


def write_benchmark_report(result: EndToEndBenchmarkResult, json_path: Path | None, markdown_path: Path | None) -> None:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(result.to_markdown() + "\n", encoding="utf-8")


def benchmark_fake_topopt_to_glb(
    *,
    voxel_output: Path,
    density_output: Path,
    seed_output: Path,
    glb_output: Path,
    xy_size: int = 200,
    z_size: int = 80,
    outer_radius: float = 100.0,
    inner_radius: float = 50.0,
    num_seeds: int = 2_000,
    gamma: float = 1.0,
    rng_seed: int = 42,
    chunk_depth: int = 8,
) -> EndToEndBenchmarkResult:
    timer = _StageTimer()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    total_start = time.perf_counter()

    timer.measure(
        "generate voxels",
        lambda: generate_annular_cylinder_npz(
            output_path=voxel_output,
            xy_size=xy_size,
            z_size=z_size,
            outer_radius=outer_radius,
            inner_radius=inner_radius,
            chunk_depth=chunk_depth,
        ),
    )
    timer.measure(
        "generate fake density",
        lambda: generate_fake_density_result(
            source_npz=voxel_output,
            output_npz=density_output,
            chunk_depth=chunk_depth,
        ),
    )

    density_result = timer.measure("load fake density input", lambda: load_density_input(density_output))
    density_milli = np.asarray(density_result["density_milli"], dtype=np.uint16)
    density = density_milli.astype(np.float32) / 1000.0
    _ = timer.measure("build probability field", lambda: density_to_probability_intensity(density=density, gamma=gamma))
    seed_points = timer.measure(
        "sample seed points",
        lambda: sample_seed_points(
            density_milli=density_milli,
            num_seeds=num_seeds,
            gamma=gamma,
            rng_seed=rng_seed,
            progress=True,
        ),
    )
    timer.measure(
        "save seed mapping npz",
        lambda: save_seed_mapping_result(
            output_npz=seed_output,
            seed_points=seed_points,
            original_shape=density_milli.shape,
            gamma=gamma,
            num_seeds=num_seeds,
            input_npz=density_output,
        ),
    )

    domain = timer.measure(
        "build annular cylinder domain",
        lambda: build_annular_cylinder_domain(
            xy_size=xy_size,
            z_size=z_size,
            outer_radius=outer_radius,
            inner_radius=inner_radius,
        ),
    )
    seed_points64 = np.asarray(seed_points, dtype=np.float64)
    neighbor_map = timer.measure("build Delaunay neighbor map", lambda: build_delaunay_neighbor_map(seed_points64))
    cells = timer.measure(
        "build exact restricted Voronoi cells",
        lambda: tuple(
            build_exact_restricted_cell(
                seed_points=seed_points64,
                domain=domain,
                seed_id=seed_id,
                include_support_traces=False,
                neighbor_seed_ids=neighbor_map.get(seed_id),
            )
            for seed_id in range(seed_points64.shape[0])
        ),
    )
    diagram = ExactRestrictedVoronoiDiagram(seed_points=seed_points64, domain=domain, cells=cells)
    diagram_brep = timer.measure(
        "build hybrid exact shell cells",
        lambda: build_hybrid_exact_diagram_brep_from_diagram(diagram),
    )
    glb_bytes, glb_summary = timer.measure(
        "serialize Three.js shell GLB",
        lambda: serialize_threejs_shell_glb(diagram_brep=diagram_brep, domain=domain),
    )
    timer.measure(
        "write GLB file",
        lambda: _write_bytes(glb_output, glb_bytes),
    )

    total_wall_seconds = time.perf_counter() - total_start
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    return EndToEndBenchmarkResult(
        voxel_npz=voxel_output,
        density_npz=density_output,
        seed_npz=seed_output,
        glb_path=glb_output,
        xy_size=xy_size,
        z_size=z_size,
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        num_seeds=num_seeds,
        gamma=gamma,
        rng_seed=rng_seed,
        stages=tuple(timer.stages),
        total_wall_seconds=total_wall_seconds,
        total_user_seconds=usage_end.ru_utime - usage_start.ru_utime,
        total_sys_seconds=usage_end.ru_stime - usage_start.ru_stime,
        output_bytes=glb_summary.output_bytes,
        num_cells=glb_summary.num_cells,
        num_exported_cells=glb_summary.num_exported_cells,
        num_shell_cells=glb_summary.num_shell_cells,
        num_faces=glb_summary.num_faces,
        num_triangles=glb_summary.num_triangles,
        num_boundaries=glb_summary.num_boundaries,
    )


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
