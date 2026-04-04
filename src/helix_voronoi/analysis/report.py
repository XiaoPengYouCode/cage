from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from helix_voronoi.analysis.compression import CompressionResult
from helix_voronoi.analysis.config import ModulusAnalysisConfig


def convergence_delta(results: list[CompressionResult]) -> float | None:
    if len(results) < 2:
        return None
    fine = results[-1].effective_modulus_gpa
    coarse = results[-2].effective_modulus_gpa
    if fine == 0.0:
        return None
    return abs(fine - coarse) / fine


def report_payload(
    config: ModulusAnalysisConfig,
    results: dict[str, list[CompressionResult]],
) -> dict:
    styles = {}
    for style, style_results in results.items():
        delta = convergence_delta(style_results)
        styles[style] = {
            "results": [asdict(result) for result in style_results],
            "convergence_delta": delta,
            "converged": None if delta is None else delta < 0.05,
        }

    return {
        "backend": config.backend,
        "seed": config.seed,
        "num_seeds": config.num_seeds,
        "resolutions": list(config.resolutions),
        "rod_radius": config.rod_radius,
        "helix": {
            "cycles_per_segment": config.helix_cycles_per_segment,
            "amplitude_ratio": config.helix_amplitude_ratio,
        },
        "material": {
            "name": config.material.name,
            "youngs_modulus_gpa": config.material.youngs_modulus_gpa,
            "poisson_ratio": config.material.poisson_ratio,
        },
        "compression": {
            "axis": config.compression.loaded_axis,
            "applied_strain": config.compression.applied_strain,
            "boundary_condition": config.compression.boundary_condition,
        },
        "styles": styles,
    }


def markdown_report(config: ModulusAnalysisConfig, payload: dict) -> str:
    lines = [
        "# Modulus Analysis",
        "",
        f"- Backend: `{config.backend}`",
        f"- Seed: `{config.seed}`",
        f"- Num seeds: `{config.num_seeds}`",
        f"- Resolutions: `{list(config.resolutions)}`",
        f"- Rod radius: `{config.rod_radius}`",
        f"- Helix cycles per segment: `{config.helix_cycles_per_segment}`",
        f"- Helix amplitude ratio: `{config.helix_amplitude_ratio}`",
        f"- Material: `{config.material.name}`",
        f"- Young's modulus: `{config.material.youngs_modulus_gpa} GPa`",
        f"- Poisson ratio: `{config.material.poisson_ratio}`",
        f"- Compression axis: `{config.compression.loaded_axis}`",
        f"- Applied strain: `{config.compression.applied_strain}`",
        f"- Boundary condition: `{config.compression.boundary_condition}`",
        "",
    ]

    for style, style_payload in payload["styles"].items():
        lines.extend(
            [
                f"## {style.title()}",
                "",
                "| Resolution | E_z (GPa) | Reaction (N) | Solid Volume | Active Voxels | Active Elements | Active Nodes |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for result in style_payload["results"]:
            lines.append(
                f"| {result['resolution']} | {result['effective_modulus_gpa']:.6f} | "
                f"{result['reaction_force_n']:.6f} | {result['solid_volume_fraction']:.6f} | "
                f"{result['active_voxel_count']} | {result['active_element_count']} | {result['active_node_count']} |"
            )
        delta = style_payload["convergence_delta"]
        lines.append("")
        if delta is None:
            lines.append("- Convergence: n/a")
        else:
            lines.append(f"- Convergence delta: `{delta:.4%}`")
            lines.append(f"- Converged (< 5%): `{style_payload['converged']}`")
        lines.append("")

    return "\n".join(lines)


def write_report(
    config: ModulusAnalysisConfig,
    results: dict[str, list[CompressionResult]],
) -> tuple[Path, Path]:
    payload = report_payload(config, results)
    markdown = markdown_report(config, payload)

    config.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_markdown.write_text(markdown, encoding="utf-8")
    config.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config.output_markdown, config.output_json
