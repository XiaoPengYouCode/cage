"""Batch runner: 3 seed counts × 4 CVT iterations = 12 experiments.

Reuses existing Steps 1-5 outputs (raw_density, obb, aligned_density).
Each experiment writes to outputs/matlab2stl_pipeline/batch/seeds{N}_cvt{K}/.
"""
import sys
from pathlib import Path

sys.path.insert(0, "src")

from matlab2stl_pipeline.seed_sampler import sample_seeds
from matlab2stl_pipeline.cvt_relaxation import lloyd_relax
from matlab2stl_pipeline.box_voronoi import build_box_voronoi, extract_voronoi_edges
from matlab2stl_pipeline.skeleton_voxelizer import voxelize_skeleton, mesh_from_voxels

# --- Shared inputs (Steps 1-5 already done) ---
SHARED = Path("outputs/matlab2stl_pipeline")
ALIGNED_NPZ = SHARED / "681_aligned_density_gamma1.npz"

BATCH_DIR = SHARED / "batch"

SEED_COUNTS = [300, 400, 500]
CVT_ITERS   = [1, 50, 100, 500]

GAMMA            = 1.0
SUBDIVISION      = 10
DILATION_RADIUS  = 3.0
MC_SMOOTH_SIGMA  = 1.0


def run_experiment(num_seeds: int, cvt_iters: int) -> None:
    tag = f"seeds{num_seeds}_cvt{cvt_iters}"
    out = BATCH_DIR / tag
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Experiment: {tag}")
    print(f"{'='*60}")

    # Step 6 — seed sampling
    seeds_npz = out / f"seeds_{num_seeds}.npz"
    print(f"\n[Step 6] Sampling {num_seeds} seeds …")
    sample_seeds(ALIGNED_NPZ, seeds_npz, num_seeds=num_seeds, gamma=GAMMA)
    print(f"  ✓ → {seeds_npz}")

    # Step 6.5 — CVT relaxation
    print(f"\n[Step 6.5] CVT Lloyd relaxation ({cvt_iters} iters) …")
    cvt_npz = out / f"seeds_{num_seeds}_cvt{cvt_iters}.npz"
    lloyd_relax(seeds_npz, ALIGNED_NPZ, cvt_npz, num_iters=cvt_iters)
    print(f"  ✓ → {cvt_npz}")

    # Step 7 — Voronoi
    print(f"\n[Step 7] Building Voronoi …")
    vor_npz = out / "voronoi.npz"
    vor_payload = build_box_voronoi(cvt_npz, ALIGNED_NPZ, vor_npz)
    n_valid = sum(1 for v in vor_payload["cell_vertices"] if v.shape[0] >= 4)
    print(f"  {n_valid}/{len(vor_payload['cell_vertices'])} valid cells  ✓ → {vor_npz}")

    # Step 8 — edges
    print(f"\n[Step 8] Extracting edges …")
    edges_npz = out / "voronoi_edges.npz"
    edges = extract_voronoi_edges(vor_npz, edges_npz)
    print(f"  {len(edges)} unique edges  ✓ → {edges_npz}")

    # Step 9 — voxelize
    print(f"\n[Step 9] Voxelizing skeleton …")
    skel_npz = out / "skeleton_voxels.npz"
    voxelize_skeleton(
        edges_npz_path=edges_npz,
        aligned_npz_path=ALIGNED_NPZ,
        output_npz_path=skel_npz,
        subdivision=SUBDIVISION,
        dilation_radius_fine_voxels=DILATION_RADIUS,
    )
    print(f"  ✓ → {skel_npz}")

    # Step 10 — MC mesh
    print(f"\n[Step 10] Marching Cubes mesh export …")
    glb = out / "scaffold.glb"
    stl = out / "scaffold.stl"
    mesh_from_voxels(
        skeleton_npz_path=skel_npz,
        output_glb_path=glb,
        output_stl_path=stl,
        smooth_sigma=MC_SMOOTH_SIGMA,
    )
    print(f"  ✓ → {glb}")
    print(f"  ✓ → {stl}")


if __name__ == "__main__":
    total = len(SEED_COUNTS) * len(CVT_ITERS)
    done = 0
    for n in SEED_COUNTS:
        for k in CVT_ITERS:
            done += 1
            print(f"\n\033[1m[{done}/{total}] seeds={n}  cvt={k}\033[0m")
            run_experiment(n, k)

    print(f"\n\033[1m\033[32m✓ All {total} experiments complete.\033[0m")
    print(f"  Results: {BATCH_DIR}")
