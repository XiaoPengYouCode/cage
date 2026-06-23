from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP_DIR = ROOT / "outputs" / "fjw_optimize_real_iter017" / "seed_radius_sweep_cvt500_subdiv4"
DEFAULT_OUTPUT_JSON = ROOT / "post_process" / "analysis" / "output" / "iter017_seed_radius_replacement_candidates.json"
REPLACEMENT_SCRIPT = ROOT / "post_process" / "analysis" / "build_iter017_variable_radius_replacement_design.py"
LEGACY_REPLACEMENT_SCRIPT = ROOT / "Post process" / "analysis" / "build_iter017_variable_radius_replacement_design.py"


def _radius_tag(radius_mm: float) -> str:
    return f"r{radius_mm:.3f}mm".replace(".", "p")


def _load_rows(summary_json: Path) -> list[dict[str, object]]:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    return list(payload["rows"])


def _select_candidates(rows: list[dict[str, object]], bands: tuple[tuple[float, float], ...]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[tuple[int, float]] = set()
    for lo, hi in bands:
        candidates = [
            row
            for row in rows
            if lo <= float(row["solid_fraction_density_domain"]) <= hi
        ]
        candidates.sort(key=lambda row: float(row["porosity_density_domain"]), reverse=True)
        if not candidates:
            continue
        row = candidates[0]
        key = (int(row["seed_count"]), float(row["radius_mm"]))
        if key in seen:
            continue
        selected.append(row)
        seen.add(key)
    return selected


def build_candidates(
    *,
    sweep_dir: Path,
    output_json: Path,
    bands: tuple[tuple[float, float], ...],
    aggregation_mode: str,
    dry_run: bool,
) -> dict[str, object]:
    summary_json = sweep_dir / "seed_radius_sweep_summary.json"
    rows = _load_rows(summary_json)
    selected = _select_candidates(rows, bands)
    script = REPLACEMENT_SCRIPT if REPLACEMENT_SCRIPT.exists() else LEGACY_REPLACEMENT_SCRIPT
    if not script.exists():
        raise FileNotFoundError(f"Replacement script not found: {REPLACEMENT_SCRIPT} or {LEGACY_REPLACEMENT_SCRIPT}")

    built: list[dict[str, object]] = []
    for row in selected:
        seed_count = int(row["seed_count"])
        radius_mm = float(row["radius_mm"])
        tag = f"seeds{seed_count}_{_radius_tag(radius_mm)}_{aggregation_mode}"
        output_npz = sweep_dir / f"fjw_iter017_replacement_design_{tag}.npz"
        command = [
            sys.executable,
            str(script),
            "--skeleton-npz",
            str(row["skeleton_npz"]),
            "--output-npz",
            str(output_npz),
            "--aggregation-mode",
            aggregation_mode,
        ]
        result: dict[str, object] | None = None
        if not dry_run:
            completed = subprocess.run(command, check=True, text=True, capture_output=True)
            result = json.loads(completed.stdout)
        built.append(
            {
                "seed_count": seed_count,
                "radius_mm": radius_mm,
                "solid_fraction_density_domain": float(row["solid_fraction_density_domain"]),
                "porosity_density_domain": float(row["porosity_density_domain"]),
                "skeleton_npz": str(row["skeleton_npz"]),
                "replacement_npz": str(output_npz.resolve()),
                "aggregation_mode": aggregation_mode,
                "command": command,
                "result": result,
            }
        )

    payload = {
        "source_summary_json": str(summary_json.resolve()),
        "bands": [{"solid_fraction_min": lo, "solid_fraction_max": hi} for lo, hi in bands],
        "aggregation_mode": aggregation_mode,
        "dry_run": dry_run,
        "candidates": built,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build FE replacement designs for representative seed-radius candidates.")
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--solid-fraction-bands",
        type=str,
        default="0.005:0.010,0.010:0.020,0.020:0.040,0.040:0.070",
    )
    parser.add_argument(
        "--aggregation-mode",
        choices=("mean_only", "fill_scaled", "local_support"),
        default="mean_only",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _parse_bands(raw: str) -> tuple[tuple[float, float], ...]:
    bands: list[tuple[float, float]] = []
    for item in str(raw).split(","):
        if not item.strip():
            continue
        lo, hi = item.split(":", 1)
        bands.append((float(lo), float(hi)))
    if not bands:
        raise ValueError("Expected at least one solid-fraction band.")
    return tuple(bands)


def main() -> int:
    args = build_parser().parse_args()
    payload = build_candidates(
        sweep_dir=args.sweep_dir,
        output_json=args.output_json,
        bands=_parse_bands(args.solid_fraction_bands),
        aggregation_mode=args.aggregation_mode,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
