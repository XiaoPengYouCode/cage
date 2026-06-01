from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "fjw_optimize_real_iter017"
DEFAULT_ALIGNED_NPZ = OUTPUT_DIR / "fjw_iter017_aligned_density_gamma1.npz"
DEFAULT_SWEEP_DIR = OUTPUT_DIR / "seed_radius_sweep_cvt500"
DEFAULT_SEED_COUNTS = (500, 750, 1000, 1250, 1500)
DEFAULT_RADII_MM = (0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.30)


def _parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def _parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float.")
    return values


def _stage_set(raw: str) -> set[str]:
    allowed = {"seeds", "voronoi", "skeleton"}
    stages = {item.strip() for item in str(raw).split(",") if item.strip()}
    unknown = stages - allowed
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    return stages or set(allowed)


def _radius_tag(radius_mm: float) -> str:
    return f"r{radius_mm:.3f}mm".replace(".", "p")


def _seed_tag(seed_count: int, gamma: float, cvt_iters: int) -> str:
    return f"seeds{seed_count}_gamma{gamma:g}_cvt{cvt_iters}"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_existing_jsonl(path: Path) -> dict[tuple[int, float], dict[str, object]]:
    if not path.exists():
        return {}
    rows: dict[tuple[int, float], dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[(int(row["seed_count"]), float(row["radius_mm"]))] = row
    return rows


def _build_constant_radius_edges(edges_npz: Path, radius_mm: float, output_npz: Path) -> int:
    data = np.load(edges_npz)
    edges = np.asarray(data["edges"], dtype=np.float32)
    payload = {
        "edges": edges,
        "assigned_radius_mm": np.full(edges.shape[0], float(radius_mm), dtype=np.float32),
        "radius_mode": np.array("constant"),
        "radius_mm": np.float32(radius_mm),
        "source_edges_npz": np.array(str(edges_npz.resolve())),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **payload)
    return int(edges.shape[0])


def _count_occupied_inside_density_domain(
    *,
    voxels: np.ndarray,
    aligned_npz: Path,
    subdivision: int,
    pad: int,
) -> tuple[int, int]:
    aligned = np.load(aligned_npz)
    density_mask = np.asarray(aligned["density_milli"], dtype=np.uint16) > 0
    domain_total = int(density_mask.sum()) * int(subdivision**3)
    if bool(density_mask.all()):
        box_shape = tuple(int(v) * subdivision for v in aligned["grid_shape_xyz"].tolist())
        box = voxels[
            pad : pad + box_shape[0],
            pad : pad + box_shape[1],
            pad : pad + box_shape[2],
        ]
        return int(box.sum()), domain_total

    occupied_total = 0
    nx = int(density_mask.shape[0])
    for coarse_x in range(nx):
        if not bool(density_mask[coarse_x].any()):
            continue
        fx0 = pad + coarse_x * subdivision
        fx1 = fx0 + subdivision
        slab = voxels[
            fx0:fx1,
            pad : pad + density_mask.shape[1] * subdivision,
            pad : pad + density_mask.shape[2] * subdivision,
        ]
        slab_mask = np.repeat(np.repeat(density_mask[coarse_x], subdivision, axis=0), subdivision, axis=1)
        occupied_total += int(np.logical_and(slab, slab_mask[None, :, :]).sum())
    return occupied_total, domain_total


def _summarize_skeleton(
    *,
    skeleton_npz: Path,
    aligned_npz: Path,
    seed_count: int,
    radius_mm: float,
    edge_count: int,
) -> dict[str, object]:
    payload = np.load(skeleton_npz)
    voxels = np.asarray(payload["voxels"], dtype=bool)
    grid_shape = tuple(int(v) for v in payload["grid_shape_xyz"].tolist())
    occupied = int(voxels.sum())
    total = int(voxels.size)
    subdivision = int(payload["subdivision"])
    pad = int(payload["pad_fine_voxels"])
    source_shape = tuple(int(v) for v in np.load(aligned_npz)["grid_shape_xyz"].tolist())
    box_shape = tuple(v * subdivision for v in source_shape)
    box = voxels[
        pad : pad + box_shape[0],
        pad : pad + box_shape[1],
        pad : pad + box_shape[2],
    ]
    box_occupied = int(box.sum())
    box_total = int(box.size)
    mask_occupied, mask_total = _count_occupied_inside_density_domain(
        voxels=voxels,
        aligned_npz=aligned_npz,
        subdivision=subdivision,
        pad=pad,
    )

    def fraction(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    return {
        "seed_count": int(seed_count),
        "radius_mm": float(radius_mm),
        "edge_count": int(edge_count),
        "skeleton_npz": str(skeleton_npz.resolve()),
        "grid_shape_xyz": list(grid_shape),
        "subdivision": subdivision,
        "pad_fine_voxels": pad,
        "voxel_size_xyz_m": payload["voxel_size_xyz_m"].astype(float).tolist(),
        "occupied_voxels_total_grid": occupied,
        "solid_fraction_total_grid": fraction(occupied, total),
        "porosity_total_grid": 1.0 - fraction(occupied, total),
        "occupied_voxels_source_box": box_occupied,
        "solid_fraction_source_box": fraction(box_occupied, box_total),
        "porosity_source_box": 1.0 - fraction(box_occupied, box_total),
        "occupied_voxels_density_domain": mask_occupied,
        "density_domain_voxels": mask_total,
        "solid_fraction_density_domain": fraction(mask_occupied, mask_total),
        "porosity_density_domain": 1.0 - fraction(mask_occupied, mask_total),
    }


def _combine_rows(summary_jsonl: Path, output_json: Path) -> dict[str, object]:
    rows = list(_load_existing_jsonl(summary_jsonl).values())
    rows.sort(key=lambda row: (int(row["seed_count"]), float(row["radius_mm"])))
    payload = {
        "summary_jsonl": str(summary_jsonl.resolve()),
        "row_count": len(rows),
        "seed_counts": sorted({int(row["seed_count"]) for row in rows}),
        "radii_mm": sorted({float(row["radius_mm"]) for row in rows}),
        "rows": rows,
    }
    _write_json(output_json, payload)
    return payload


def run_sweep(
    *,
    aligned_npz: Path,
    output_dir: Path,
    seed_counts: Iterable[int],
    radii_mm: Iterable[float],
    gamma: float,
    rng_seed: int,
    cvt_iters: int,
    subdivision: int,
    stages: set[str],
    resume: bool,
    skip_mesh_export: bool,
) -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "src"))
    from matlab2stl_pipeline.box_voronoi import build_box_voronoi, extract_voronoi_edges
    from matlab2stl_pipeline.cvt_relaxation import lloyd_relax
    from matlab2stl_pipeline.seed_sampler import sample_seeds
    from matlab2stl_pipeline.skeleton_voxelizer import mesh_from_voxels, voxelize_variable_radius_skeleton

    seed_counts = [int(v) for v in seed_counts]
    radii_mm = [float(v) for v in radii_mm]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "seed_radius_sweep_manifest.json"
    summary_jsonl = output_dir / "seed_radius_sweep_rows.jsonl"
    summary_json = output_dir / "seed_radius_sweep_summary.json"
    completed = _load_existing_jsonl(summary_jsonl) if resume else {}

    manifest = {
        "aligned_npz": str(aligned_npz.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed_counts": seed_counts,
        "radii_mm": radii_mm,
        "gamma": float(gamma),
        "rng_seed": int(rng_seed),
        "cvt_iters": int(cvt_iters),
        "subdivision": int(subdivision),
        "stages": sorted(stages),
        "skip_mesh_export": bool(skip_mesh_export),
        "method": "density-weighted seed sampling -> CVT -> box-restricted Voronoi -> constant-radius skeleton sweep",
    }
    _write_json(manifest_path, manifest)

    for seed_count in seed_counts:
        tag = _seed_tag(seed_count, gamma, cvt_iters)
        initial_seeds = output_dir / f"fjw_iter017_seeds_{seed_count}_gamma{gamma:g}.npz"
        cvt_seeds = output_dir / f"fjw_iter017_seeds_{seed_count}_gamma{gamma:g}_cvt{cvt_iters}.npz"
        voronoi_npz = output_dir / f"fjw_iter017_voronoi_{tag}.npz"
        edges_npz = output_dir / f"fjw_iter017_voronoi_edges_{tag}.npz"

        if "seeds" in stages:
            if not (resume and initial_seeds.exists()):
                print(f"[seeds] sample {seed_count}", flush=True)
                sample_seeds(aligned_npz, initial_seeds, num_seeds=seed_count, gamma=gamma, rng_seed=rng_seed)
            if not (resume and cvt_seeds.exists()):
                print(f"[seeds] CVT {seed_count}, iters={cvt_iters}", flush=True)
                lloyd_relax(initial_seeds, aligned_npz, cvt_seeds, num_iters=cvt_iters)

        if "voronoi" in stages:
            if not cvt_seeds.exists():
                raise FileNotFoundError(f"Missing CVT seeds: {cvt_seeds}")
            if not (resume and voronoi_npz.exists()):
                print(f"[voronoi] build cells {tag}", flush=True)
                build_box_voronoi(cvt_seeds, aligned_npz, voronoi_npz)
            if not (resume and edges_npz.exists()):
                print(f"[voronoi] extract edges {tag}", flush=True)
                extract_voronoi_edges(voronoi_npz, edges_npz)

        if "skeleton" not in stages:
            continue
        if not edges_npz.exists():
            raise FileNotFoundError(f"Missing Voronoi edges: {edges_npz}")

        for radius_mm in radii_mm:
            key = (seed_count, float(radius_mm))
            if resume and key in completed:
                print(f"[skeleton] skip completed {tag} {_radius_tag(radius_mm)}", flush=True)
                continue
            started = time.time()
            rtag = _radius_tag(radius_mm)
            radius_edges_npz = output_dir / f"fjw_iter017_voronoi_edges_{tag}_{rtag}.npz"
            skeleton_npz = output_dir / f"fjw_iter017_skeleton_voxels_{tag}_{rtag}.npz"
            glb_path = output_dir / f"fjw_iter017_skeleton_{tag}_{rtag}.glb"
            stl_path = output_dir / f"fjw_iter017_skeleton_{tag}_{rtag}.stl"
            print(f"[skeleton] {tag} {rtag}", flush=True)
            edge_count = _build_constant_radius_edges(edges_npz, radius_mm, radius_edges_npz)
            voxelize_variable_radius_skeleton(
                edges_npz_path=radius_edges_npz,
                aligned_npz_path=aligned_npz,
                output_npz_path=skeleton_npz,
                subdivision=subdivision,
                radius_field_key="assigned_radius_mm",
            )
            if not skip_mesh_export:
                mesh_from_voxels(
                    skeleton_npz,
                    glb_path,
                    stl_path,
                    aligned_npz_path=aligned_npz,
                )
            row = _summarize_skeleton(
                skeleton_npz=skeleton_npz,
                aligned_npz=aligned_npz,
                seed_count=seed_count,
                radius_mm=radius_mm,
                edge_count=edge_count,
            )
            row["constant_radius_edges_npz"] = str(radius_edges_npz.resolve())
            row["elapsed_seconds"] = float(time.time() - started)
            if not skip_mesh_export:
                row["glb_path"] = str(glb_path.resolve())
                row["stl_path"] = str(stl_path.resolve())
            _append_jsonl(summary_jsonl, row)

    summary = _combine_rows(summary_jsonl, summary_json)
    summary["manifest_path"] = str(manifest_path.resolve())
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full-domain iter_017 seed-count × rod-radius Voronoi porosity sweep."
    )
    parser.add_argument("--aligned-npz", type=Path, default=DEFAULT_ALIGNED_NPZ)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--seed-counts", type=str, default=",".join(str(v) for v in DEFAULT_SEED_COUNTS))
    parser.add_argument("--radii-mm", type=str, default=",".join(f"{v:g}" for v in DEFAULT_RADII_MM))
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--cvt-iters", type=int, default=500)
    parser.add_argument("--subdivision", type=int, default=10)
    parser.add_argument("--stages", type=str, default="seeds,voronoi,skeleton")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-mesh-export", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_sweep(
        aligned_npz=args.aligned_npz,
        output_dir=args.output_dir,
        seed_counts=_parse_int_list(args.seed_counts),
        radii_mm=_parse_float_list(args.radii_mm),
        gamma=float(args.gamma),
        rng_seed=int(args.rng_seed),
        cvt_iters=int(args.cvt_iters),
        subdivision=int(args.subdivision),
        stages=_stage_set(args.stages),
        resume=not bool(args.no_resume),
        skip_mesh_export=bool(args.skip_mesh_export),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
