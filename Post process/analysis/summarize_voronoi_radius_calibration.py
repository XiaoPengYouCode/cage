from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = ROOT / "outputs" / "voronoi_radius_calibration_remote_smoke" / "calibration_fe_results.json"
DEFAULT_OUTPUT_PATH = ROOT / "Post process" / "analysis" / "output" / "voronoi_radius_calibration_summary.json"


def _load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(input_path: Path) -> dict[str, object]:
    payload = _load_results(input_path)
    grouped: dict[float, list[dict]] = defaultdict(list)
    for item in payload["results"]:
        grouped[float(item["radius_mm"])].append(item)

    radius_rows: list[dict[str, object]] = []
    for radius_mm in sorted(grouped):
        items = grouped[radius_mm]
        moduli = [float(item["metrics"]["apparent_modulus_gpa"]) for item in items]
        stiffnesses = [float(item["metrics"]["apparent_stiffness_kn_per_mm"]) for item in items]
        volume_fractions = [float(item["metrics"]["solid_volume_fraction"]) for item in items]
        radius_rows.append(
            {
                "radius_mm": radius_mm,
                "sample_count": len(items),
                "apparent_modulus_gpa": {
                    "min": min(moduli),
                    "mean": sum(moduli) / len(moduli),
                    "max": max(moduli),
                },
                "apparent_stiffness_kn_per_mm": {
                    "min": min(stiffnesses),
                    "mean": sum(stiffnesses) / len(stiffnesses),
                    "max": max(stiffnesses),
                },
                "solid_volume_fraction": {
                    "min": min(volume_fractions),
                    "mean": sum(volume_fractions) / len(volume_fractions),
                    "max": max(volume_fractions),
                },
            }
        )

    target_bands = []
    seen_band_indices: set[int] = set()
    for item in payload["results"]:
        if "band_index" in item:
            band_rows = [
                {
                    "band_index": int(item["band_index"]),
                    "representative_design": float(item["representative_design"]),
                    "representative_target_modulus": float(item["representative_target_modulus"]),
                }
            ]
        else:
            band_rows = [
                {
                    "band_index": int(band["band_index"]),
                    "representative_design": float(band["representative_design"]),
                    "representative_target_modulus": float(band["representative_target_modulus"]),
                }
                for band in item.get("support_bands", [])
            ]
        for band in band_rows:
            band_index = band["band_index"]
            if band_index in seen_band_indices:
                continue
            seen_band_indices.add(band_index)
            target_bands.append(band)
    target_bands.sort(key=lambda row: row["band_index"])

    return {
        "source_results_json": str(input_path.resolve()),
        "specimen_count": int(payload["specimen_count"]),
        "evaluation_mode": str(payload["evaluation_mode"]),
        "voxel_size_mm": float(payload["voxel_size_mm"]),
        "load_n": float(payload["load_n"]),
        "material": payload["material"],
        "target_bands": target_bands,
        "radius_calibration_rows": radius_rows,
        "current_interpretation": {
            "status": "partial_baseline_ready",
            "established_map": "radius_mm -> apparent_modulus_gpa",
            "missing_map": "representative_target_modulus -> calibrated_radius_mm",
            "limitation": (
                "All current band samples reuse one fixed local Voronoi topology and seed realization, "
                "so this summary establishes a real radius baseline but not yet the final inverse calibration table."
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Voronoi radius calibration FE results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    summary = build_summary(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
