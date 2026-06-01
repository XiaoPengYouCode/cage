from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "fjw_optimize_real_iter017"
DEFAULT_REFERENCE_NPZ = ROOT / "datasets" / "topopt" / "fjw_reference_fem_voxels.npz"
DEFAULT_CAGE_NPZ = ROOT / "runs" / "fjw_optimize_real" / "iter_017" / "cage_3d.npz"
DEFAULT_SWEEP_DIR = OUTPUT_DIR / "seed_radius_sweep_cvt500_design_mask_subdiv4"
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
    allowed = {"domain", "seeds", "voronoi", "skeleton", "mesh"}
    stages = {item.strip() for item in str(raw).split(",") if item.strip()}
    unknown = stages - allowed
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    return stages or {"domain", "seeds", "voronoi", "skeleton"}


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


def _load_reference_and_density(reference_npz: Path, cage_npz: Path) -> tuple[np.ndarray, np.ndarray, float]:
    reference = np.load(reference_npz, allow_pickle=True)
    design_mask = np.asarray(reference["design_mask"], dtype=bool)
    voxel_size_m = float(np.asarray(reference["voxel_size_xyz_m"], dtype=np.float64)[0])
    density = np.asarray(np.load(cage_npz)["cage_3d"], dtype=np.float64)
    if density.shape != design_mask.shape:
        raise ValueError(f"density shape {density.shape} does not match design_mask shape {design_mask.shape}.")
    if not np.any(design_mask):
        raise ValueError("design_mask is empty.")
    return density, design_mask, voxel_size_m


def _write_design_mask_domain_npz(
    *,
    reference_npz: Path,
    cage_npz: Path,
    output_npz: Path,
    gamma: float,
) -> dict[str, object]:
    density, design_mask, voxel_size_m = _load_reference_and_density(reference_npz, cage_npz)
    masked_density = np.where(design_mask, np.clip(density, 0.0, 1.0), 0.0)
    powered = np.where(masked_density > 0.0, masked_density**float(gamma), 0.0)
    total = float(powered.sum(dtype=np.float64))
    if total <= 0.0:
        raise ValueError("No positive density inside design_mask.")
    probability = (powered / total).astype(np.float32)
    density_milli = np.rint(masked_density * 1000.0).clip(0, 1000).astype(np.uint16)
    grid_shape = np.array(density.shape, dtype=np.int32)
    payload = {
        "density_milli": density_milli,
        "voxels": design_mask.astype(np.uint8),
        "design_mask": design_mask.astype(np.uint8),
        "probability_field": probability,
        "grid_shape_xyz": grid_shape,
        "origin_m": np.zeros(3, dtype=np.float32),
        "voxel_size_xyz_m": np.array([voxel_size_m, voxel_size_m, voxel_size_m], dtype=np.float32),
        "restore_R": np.eye(3, dtype=np.float64),
        "restore_t": np.zeros(3, dtype=np.float64),
        "shape_name": np.array("iter017_design_mask"),
        "result_type": np.array("design_mask_domain"),
        "density_kind": np.array("pseudo_density_inside_design_mask"),
        "gamma": np.float32(gamma),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **payload)
    return {
        "domain_npz": str(output_npz.resolve()),
        "grid_shape_xyz": grid_shape.astype(int).tolist(),
        "design_mask_voxels": int(design_mask.sum()),
        "voxel_size_m": voxel_size_m,
        "gamma": float(gamma),
    }


def _active_mask_points(design_mask: np.ndarray) -> np.ndarray:
    return np.argwhere(design_mask).astype(np.float64)


def _project_points_to_mask(points: np.ndarray, mask_points: np.ndarray, tree: object | None = None) -> np.ndarray:
    from scipy.spatial import cKDTree

    if tree is None:
        tree = cKDTree(mask_points)
    _dist, ids = tree.query(np.asarray(points, dtype=np.float64), k=1)
    return mask_points[np.asarray(ids, dtype=np.int64)].astype(np.float64)


def _sample_design_mask_seeds(
    *,
    domain_npz: Path,
    output_npz: Path,
    num_seeds: int,
    gamma: float,
    rng_seed: int,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    data = np.load(domain_npz)
    probability = np.asarray(data["probability_field"], dtype=np.float64)
    design_mask = np.asarray(data["design_mask"], dtype=bool)
    mask_points = _active_mask_points(design_mask)
    tree = cKDTree(mask_points)

    flat_prob = probability.ravel()
    cdf = np.cumsum(flat_prob)
    cdf /= cdf[-1]

    rng = np.random.default_rng(rng_seed)
    flat_ids = np.searchsorted(cdf, rng.random(num_seeds)).clip(0, len(cdf) - 1)
    indices = np.stack(np.unravel_index(flat_ids, probability.shape), axis=1).astype(np.float64)
    seeds = indices + rng.uniform(-0.45, 0.45, size=indices.shape)
    nearest = np.rint(seeds).astype(np.int64)
    valid = np.all((nearest >= 0) & (nearest < np.asarray(design_mask.shape)[None, :]), axis=1)
    in_mask = np.zeros(num_seeds, dtype=bool)
    in_mask[valid] = design_mask[nearest[valid, 0], nearest[valid, 1], nearest[valid, 2]]
    if not np.all(in_mask):
        seeds[~in_mask] = _project_points_to_mask(seeds[~in_mask], mask_points, tree=tree)

    voxel_size_xyz_m = np.asarray(data["voxel_size_xyz_m"], dtype=np.float32)
    origin_m = np.asarray(data["origin_m"], dtype=np.float32)
    seed_points_m = origin_m + seeds.astype(np.float32) * voxel_size_xyz_m
    payload = {
        "seed_points": seeds.astype(np.float32),
        "seed_points_m": seed_points_m.astype(np.float32),
        "num_seeds": np.int32(num_seeds),
        "gamma": np.float32(gamma),
        "grid_shape_xyz": np.asarray(data["grid_shape_xyz"], dtype=np.int32),
        "voxel_size_xyz_m": voxel_size_xyz_m,
        "origin_m": origin_m,
        "domain_mode": np.array("design_mask"),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **payload)
    return seeds.astype(np.float32)


def _mask_lloyd_relax(
    *,
    seeds_npz: Path,
    domain_npz: Path,
    output_npz: Path,
    num_iters: int,
    progress_interval: int = 50,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    seed_data = np.load(seeds_npz)
    domain = np.load(domain_npz)
    seeds = np.asarray(seed_data["seed_points"], dtype=np.float64).copy()
    mask_points = _active_mask_points(np.asarray(domain["design_mask"], dtype=bool))
    mask_tree = cKDTree(mask_points)
    history = []

    for iteration in range(int(num_iters)):
        old = seeds.copy()
        seed_tree = cKDTree(seeds)
        _dist, labels = seed_tree.query(mask_points, k=1)
        new = seeds.copy()
        empty_count = 0
        for seed_id in range(seeds.shape[0]):
            cluster = mask_points[labels == seed_id]
            if cluster.shape[0] == 0:
                empty_count += 1
                continue
            new[seed_id] = cluster.mean(axis=0)
        seeds = _project_points_to_mask(new, mask_points, tree=mask_tree)
        displacement = np.linalg.norm(seeds - old, axis=1)
        history.append(
            (
                iteration + 1,
                float(displacement.max()) if displacement.size else 0.0,
                float(displacement.mean()) if displacement.size else 0.0,
                int(empty_count),
            )
        )
        if progress_interval and ((iteration + 1) % progress_interval == 0 or iteration == num_iters - 1):
            print(
                f"    mask CVT iter {iteration + 1:4d}/{num_iters} "
                f"(empty seeds: {empty_count}, max disp: {history[-1][1]:.3f})",
                flush=True,
            )

    voxel_size_xyz_m = np.asarray(domain["voxel_size_xyz_m"], dtype=np.float32)
    origin_m = np.asarray(domain["origin_m"], dtype=np.float32)
    seed_points_m = origin_m + seeds.astype(np.float32) * voxel_size_xyz_m
    history_arr = np.asarray(
        history,
        dtype=[
            ("iteration", np.int32),
            ("max_displacement", np.float64),
            ("mean_displacement", np.float64),
            ("empty_seed_count", np.int32),
        ],
    )
    payload = {
        "seed_points": seeds.astype(np.float32),
        "seed_points_m": seed_points_m.astype(np.float32),
        "num_seeds": np.int32(seeds.shape[0]),
        "gamma": seed_data["gamma"],
        "cvt_iters": np.int32(num_iters),
        "grid_shape_xyz": np.asarray(domain["grid_shape_xyz"], dtype=np.int32),
        "voxel_size_xyz_m": voxel_size_xyz_m,
        "origin_m": origin_m,
        "domain_mode": np.array("design_mask"),
        "cvt_history": history_arr,
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **payload)
    return seeds.astype(np.float32)


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


def _clip_skeleton_to_design_mask(*, skeleton_npz: Path, domain_npz: Path) -> tuple[int, int]:
    skeleton = np.load(skeleton_npz, allow_pickle=True)
    domain = np.load(domain_npz)
    voxels = np.asarray(skeleton["voxels"], dtype=bool)
    radius = np.asarray(skeleton["voxel_radius_mm"], dtype=np.float32)
    design_mask = np.asarray(domain["design_mask"], dtype=bool)
    subdivision = int(skeleton["subdivision"])
    pad = int(skeleton["pad_fine_voxels"])
    nx, ny, nz = design_mask.shape

    clipped = np.zeros_like(voxels, dtype=np.uint8)
    clipped_radius = np.zeros_like(radius, dtype=np.float32)
    y0, y1 = pad, pad + ny * subdivision
    z0, z1 = pad, pad + nz * subdivision
    for x in range(nx):
        if not bool(design_mask[x].any()):
            continue
        fx0 = pad + x * subdivision
        fx1 = fx0 + subdivision
        mask_yz = np.repeat(np.repeat(design_mask[x], subdivision, axis=0), subdivision, axis=1)
        keep = voxels[fx0:fx1, y0:y1, z0:z1] & mask_yz[None, :, :]
        clipped[fx0:fx1, y0:y1, z0:z1] = keep.astype(np.uint8)
        clipped_radius[fx0:fx1, y0:y1, z0:z1] = np.where(
            keep,
            radius[fx0:fx1, y0:y1, z0:z1],
            0.0,
        ).astype(np.float32)

    payload = {key: skeleton[key] for key in skeleton.files if key not in {"voxels", "voxel_radius_mm"}}
    payload["voxels"] = clipped
    payload["voxel_radius_mm"] = clipped_radius
    payload["domain_mode"] = np.array("design_mask_clipped")
    payload["source_unclipped_skeleton_npz"] = np.array(str(skeleton_npz.resolve()))
    np.savez_compressed(skeleton_npz, **payload)
    return int(clipped.sum()), int(design_mask.sum()) * int(subdivision**3)


def _summarize_skeleton(
    *,
    skeleton_npz: Path,
    domain_npz: Path,
    seed_count: int,
    radius_mm: float,
    edge_count: int,
) -> dict[str, object]:
    skeleton = np.load(skeleton_npz, allow_pickle=True)
    domain = np.load(domain_npz)
    voxels = np.asarray(skeleton["voxels"], dtype=bool)
    design_mask = np.asarray(domain["design_mask"], dtype=bool)
    occupied = int(voxels.sum())
    domain_total = int(design_mask.sum()) * int(int(skeleton["subdivision"]) ** 3)
    solid_fraction = float(occupied / domain_total) if domain_total else 0.0
    return {
        "seed_count": int(seed_count),
        "radius_mm": float(radius_mm),
        "edge_count": int(edge_count),
        "skeleton_npz": str(skeleton_npz.resolve()),
        "grid_shape_xyz": [int(v) for v in skeleton["grid_shape_xyz"].tolist()],
        "subdivision": int(skeleton["subdivision"]),
        "pad_fine_voxels": int(skeleton["pad_fine_voxels"]),
        "voxel_size_xyz_m": np.asarray(skeleton["voxel_size_xyz_m"], dtype=float).tolist(),
        "domain_mode": "design_mask_clipped",
        "design_mask_voxels": int(design_mask.sum()),
        "occupied_voxels_design_mask": occupied,
        "design_mask_fine_voxels": domain_total,
        "solid_fraction_design_mask": solid_fraction,
        "porosity_design_mask": 1.0 - solid_fraction,
    }


def _combine_rows(summary_jsonl: Path, output_json: Path, output_csv: Path) -> dict[str, object]:
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
    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return payload


def run_sweep(
    *,
    reference_npz: Path,
    cage_npz: Path,
    output_dir: Path,
    seed_counts: Iterable[int],
    radii_mm: Iterable[float],
    gamma: float,
    rng_seed: int,
    cvt_iters: int,
    subdivision: int,
    stages: set[str],
    resume: bool,
    mesh_cases: set[tuple[int, float]],
) -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "src"))
    from matlab2stl_pipeline.box_voronoi import build_box_voronoi, extract_voronoi_edges
    from matlab2stl_pipeline.skeleton_voxelizer import mesh_from_voxels, voxelize_variable_radius_skeleton

    seed_counts = [int(v) for v in seed_counts]
    radii_mm = [float(v) for v in radii_mm]
    output_dir.mkdir(parents=True, exist_ok=True)
    domain_npz = output_dir / "fjw_iter017_design_mask_domain_gamma1.npz"
    manifest_path = output_dir / "seed_radius_sweep_manifest.json"
    summary_jsonl = output_dir / "seed_radius_sweep_rows.jsonl"
    summary_json = output_dir / "seed_radius_sweep_summary.json"
    summary_csv = output_dir / "seed_radius_sweep_summary.csv"
    completed = _load_existing_jsonl(summary_jsonl) if resume else {}

    domain_summary: dict[str, object] | None = None
    if "domain" in stages or not domain_npz.exists():
        domain_summary = _write_design_mask_domain_npz(
            reference_npz=reference_npz,
            cage_npz=cage_npz,
            output_npz=domain_npz,
            gamma=gamma,
        )

    manifest = {
        "reference_npz": str(reference_npz.resolve()),
        "cage_npz": str(cage_npz.resolve()),
        "domain_npz": str(domain_npz.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed_counts": seed_counts,
        "radii_mm": radii_mm,
        "gamma": float(gamma),
        "rng_seed": int(rng_seed),
        "cvt_iters": int(cvt_iters),
        "subdivision": int(subdivision),
        "stages": sorted(stages),
        "mesh_cases": sorted([list(item) for item in mesh_cases]),
        "method": (
            "density-weighted seed sampling inside design_mask -> discrete design_mask CVT -> "
            "box Voronoi topology -> variable-radius voxelization -> strict design_mask clipping"
        ),
        "domain_summary": domain_summary,
    }
    _write_json(manifest_path, manifest)

    for seed_count in seed_counts:
        tag = _seed_tag(seed_count, gamma, cvt_iters)
        initial_seeds = output_dir / f"fjw_iter017_designmask_seeds_{seed_count}_gamma{gamma:g}.npz"
        cvt_seeds = output_dir / f"fjw_iter017_designmask_seeds_{seed_count}_gamma{gamma:g}_cvt{cvt_iters}.npz"
        voronoi_npz = output_dir / f"fjw_iter017_designmask_voronoi_{tag}.npz"
        edges_npz = output_dir / f"fjw_iter017_designmask_voronoi_edges_{tag}.npz"

        if "seeds" in stages:
            if not (resume and initial_seeds.exists()):
                print(f"[seeds] sample design-mask {seed_count}", flush=True)
                _sample_design_mask_seeds(
                    domain_npz=domain_npz,
                    output_npz=initial_seeds,
                    num_seeds=seed_count,
                    gamma=gamma,
                    rng_seed=rng_seed,
                )
            if not (resume and cvt_seeds.exists()):
                print(f"[seeds] design-mask CVT {seed_count}, iters={cvt_iters}", flush=True)
                _mask_lloyd_relax(
                    seeds_npz=initial_seeds,
                    domain_npz=domain_npz,
                    output_npz=cvt_seeds,
                    num_iters=cvt_iters,
                )

        if "voronoi" in stages:
            if not cvt_seeds.exists():
                raise FileNotFoundError(f"Missing CVT seeds: {cvt_seeds}")
            if not (resume and voronoi_npz.exists()):
                print(f"[voronoi] build box topology {tag}", flush=True)
                build_box_voronoi(cvt_seeds, domain_npz, voronoi_npz)
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
                if "mesh" in stages and key in mesh_cases:
                    row = completed[key]
                    glb_path = output_dir / f"fjw_iter017_designmask_skeleton_{tag}_{_radius_tag(radius_mm)}.glb"
                    stl_path = output_dir / f"fjw_iter017_designmask_skeleton_{tag}_{_radius_tag(radius_mm)}.stl"
                    if not glb_path.exists():
                        mesh_from_voxels(Path(row["skeleton_npz"]), glb_path, stl_path)
                continue
            started = time.time()
            rtag = _radius_tag(radius_mm)
            radius_edges_npz = output_dir / f"fjw_iter017_designmask_voronoi_edges_{tag}_{rtag}.npz"
            skeleton_npz = output_dir / f"fjw_iter017_designmask_skeleton_voxels_{tag}_{rtag}.npz"
            glb_path = output_dir / f"fjw_iter017_designmask_skeleton_{tag}_{rtag}.glb"
            stl_path = output_dir / f"fjw_iter017_designmask_skeleton_{tag}_{rtag}.stl"
            print(f"[skeleton] {tag} {rtag}", flush=True)
            edge_count = _build_constant_radius_edges(edges_npz, radius_mm, radius_edges_npz)
            voxelize_variable_radius_skeleton(
                edges_npz_path=radius_edges_npz,
                aligned_npz_path=domain_npz,
                output_npz_path=skeleton_npz,
                subdivision=subdivision,
                radius_field_key="assigned_radius_mm",
            )
            _clip_skeleton_to_design_mask(skeleton_npz=skeleton_npz, domain_npz=domain_npz)
            if "mesh" in stages and key in mesh_cases:
                mesh_from_voxels(skeleton_npz, glb_path, stl_path)
            row = _summarize_skeleton(
                skeleton_npz=skeleton_npz,
                domain_npz=domain_npz,
                seed_count=seed_count,
                radius_mm=radius_mm,
                edge_count=edge_count,
            )
            row["constant_radius_edges_npz"] = str(radius_edges_npz.resolve())
            row["elapsed_seconds"] = float(time.time() - started)
            if glb_path.exists():
                row["glb_path"] = str(glb_path.resolve())
            if stl_path.exists():
                row["stl_path"] = str(stl_path.resolve())
            _append_jsonl(summary_jsonl, row)

    summary = _combine_rows(summary_jsonl, summary_json, summary_csv)
    summary["manifest_path"] = str(manifest_path.resolve())
    summary["summary_csv"] = str(summary_csv.resolve())
    return summary


def _parse_mesh_cases(raw: str) -> set[tuple[int, float]]:
    cases: set[tuple[int, float]] = set()
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        seed_raw, radius_raw = item.split(":", 1)
        cases.add((int(seed_raw), float(radius_raw)))
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run iter017 seed-count × radius sweep clipped to the true FJW design_mask cage domain."
    )
    parser.add_argument("--reference-npz", type=Path, default=DEFAULT_REFERENCE_NPZ)
    parser.add_argument("--cage-npz", type=Path, default=DEFAULT_CAGE_NPZ)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--seed-counts", type=str, default=",".join(str(v) for v in DEFAULT_SEED_COUNTS))
    parser.add_argument("--radii-mm", type=str, default=",".join(f"{v:g}" for v in DEFAULT_RADII_MM))
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--cvt-iters", type=int, default=500)
    parser.add_argument("--subdivision", type=int, default=4)
    parser.add_argument("--stages", type=str, default="domain,seeds,voronoi,skeleton")
    parser.add_argument("--mesh-cases", type=str, default="")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_sweep(
        reference_npz=args.reference_npz,
        cage_npz=args.cage_npz,
        output_dir=args.output_dir,
        seed_counts=_parse_int_list(args.seed_counts),
        radii_mm=_parse_float_list(args.radii_mm),
        gamma=float(args.gamma),
        rng_seed=int(args.rng_seed),
        cvt_iters=int(args.cvt_iters),
        subdivision=int(args.subdivision),
        stages=_stage_set(args.stages),
        resume=not bool(args.no_resume),
        mesh_cases=_parse_mesh_cases(args.mesh_cases),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
