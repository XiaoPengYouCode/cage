from __future__ import annotations

from cage.analysis.config import ModulusAnalysisConfig


def run_modulus_analysis(config: ModulusAnalysisConfig):
    from cage.analysis.runner import run_modulus_analysis as _run_modulus_analysis

    return _run_modulus_analysis(config)


__all__ = ["run_modulus_analysis"]
