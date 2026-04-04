from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

StructureStyle = Literal["cylinder", "helix"]
StyleSelection = Literal["cylinder", "helix", "both"]


@dataclass(frozen=True)
class MaterialConfig:
    name: str = "Ti-6Al-4V ELI"
    youngs_modulus_gpa: float = 115.0
    poisson_ratio: float = 0.34

    @property
    def youngs_modulus_pa(self) -> float:
        return self.youngs_modulus_gpa * 1e9


@dataclass(frozen=True)
class CompressionConfig:
    applied_strain: float = 1e-3
    loaded_axis: str = "z"
    boundary_condition: str = "bonded_plates"


@dataclass(frozen=True)
class ModulusAnalysisConfig:
    seed: int = 55
    num_seeds: int = 10
    style: StyleSelection = "both"
    resolutions: tuple[int, ...] = (96, 128, 160)
    rod_radius: float = 0.012
    helix_cycles_per_segment: float = 3.0
    helix_amplitude_ratio: float = 0.06
    helix_steps_per_cycle: int = 36
    helix_min_steps: int = 72
    backend: str = "sfepy"
    output_markdown: Path = Path("docs/analysis/modulus_seed55.md")
    output_json: Path = Path("docs/analysis/modulus_seed55.json")
    material: MaterialConfig = MaterialConfig()
    compression: CompressionConfig = CompressionConfig()
    chunk_size: int = 100_000
    dry_run: bool = False

    def selected_styles(self) -> tuple[StructureStyle, ...]:
        if self.style == "both":
            return ("cylinder", "helix")
        return (self.style,)
