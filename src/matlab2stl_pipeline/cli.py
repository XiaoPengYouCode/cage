"""CLI entrypoint for the matlab2stl pipeline.

Usage
-----
    matlab2stl-pipeline run-pipeline \\
        --mat datasets/681.mat \\
        --output-dir outputs/matlab2stl_pipeline \\
        [--viewer-dir viewer/public/data] \\
        [--num-seeds 200] \\
        [--gamma 1.0] \\
        [--subdivision 10] \\
        [--dilation-radius 3.0]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# ANSI colour helpers — degrade gracefully on non-TTY (e.g. redirected output)
def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if sys.stdout.isatty() else s

def _cyan(s: str) -> str:
    return f"\033[36m{s}\033[0m" if sys.stdout.isatty() else s

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if sys.stdout.isatty() else s


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    mat_path: Path,
    output_dir: Path,
    viewer_dir: Path | None,
    num_seeds: int,
    gamma: float,
    subdivision: int,
    dilation_radius: float,
    cvt_iters: int = 500,
    mc_smooth_sigma: float = 1.0,
) -> None:
    from .mat_importer import load_mat_to_npz
    from .obb_aligner import fit_obb, align_density
    from .seed_sampler import sample_seeds
    from .cvt_relaxation import lloyd_relax
    from .box_voronoi import build_box_voronoi, extract_voronoi_edges, export_voronoi_cells_glb
    from .skeleton_voxelizer import voxelize_skeleton, mesh_from_voxels

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Import .mat → NPZ
    # ------------------------------------------------------------------
    print(_cyan("\n[Step 1] Importing .mat → NPZ …"))
    raw_npz = output_dir / "681_raw_density.npz"
    load_mat_to_npz(mat_path, raw_npz)
    print(_green(f"  ✓ → {raw_npz}"))

    # ------------------------------------------------------------------
    # Step 2 — Raw voxels → GLB (viewer preview)
    # ------------------------------------------------------------------
    print(_cyan("\n[Step 2] Exporting raw voxels → GLB …"))
    raw_glb = output_dir / "681_raw.glb"
    _export_raw_glb(raw_npz, raw_glb)
    print(_green(f"  ✓ → {raw_glb}"))

    # ------------------------------------------------------------------
    # Steps 3+4 — OBB fitting
    # ------------------------------------------------------------------
    print(_cyan("\n[Step 3/4] Fitting OBB …"))
    obb_npz = output_dir / "681_obb.npz"
    fit_obb(raw_npz, obb_npz)
    print(_green(f"  ✓ → {obb_npz}"))

    # ------------------------------------------------------------------
    # Step 5 — Align density field + compute probability
    # ------------------------------------------------------------------
    print(_cyan(f"\n[Step 5] Aligning density field (gamma={gamma}) …"))
    aligned_npz = output_dir / f"681_aligned_density_gamma{gamma:.0f}.npz"
    payload = align_density(raw_npz, obb_npz, aligned_npz, gamma=gamma)
    gs = payload["grid_shape_xyz"]
    print(f"  Aligned grid: {gs[0]}×{gs[1]}×{gs[2]}")
    print(_green(f"  ✓ → {aligned_npz}"))

    # ------------------------------------------------------------------
    # Step 6 — Seed sampling
    # ------------------------------------------------------------------
    print(_cyan(f"\n[Step 6] Sampling {num_seeds} seeds (gamma={gamma}) …"))
    seeds_npz = output_dir / f"681_seeds_{num_seeds}_gamma{gamma:.0f}.npz"
    sample_seeds(aligned_npz, seeds_npz, num_seeds=num_seeds, gamma=gamma)
    print(_green(f"  ✓ → {seeds_npz}"))

    # ------------------------------------------------------------------
    # Step 6.5 — CVT Lloyd relaxation (optional, default 500 iters)
    # ------------------------------------------------------------------
    if cvt_iters > 0:
        print(_cyan(f"\n[Step 6.5] CVT Lloyd relaxation ({cvt_iters} iters) …"))
        cvt_seeds_npz = output_dir / f"681_seeds_{num_seeds}_gamma{gamma:.0f}_cvt{cvt_iters}.npz"
        lloyd_relax(seeds_npz, aligned_npz, cvt_seeds_npz, num_iters=cvt_iters)
        print(_green(f"  ✓ → {cvt_seeds_npz}"))
    else:
        cvt_seeds_npz = None

    # ------------------------------------------------------------------
    # Helper: run Steps 7-9 for a given seed set, return the GLB path
    # ------------------------------------------------------------------
    def _run_voronoi_to_mesh(s_npz: Path, tag: str) -> tuple[Path, Path]:
        print(_cyan(f"\n[Step 7] Building box-restricted Voronoi ({tag}) …"))
        vor_npz = output_dir / f"681_voronoi_{tag}.npz"
        vor_payload = build_box_voronoi(s_npz, aligned_npz, vor_npz)
        n_valid = sum(1 for v in vor_payload["cell_vertices"] if v.shape[0] >= 4)
        print(f"  {n_valid}/{len(vor_payload['cell_vertices'])} valid cells")
        print(_green(f"  ✓ → {vor_npz}"))

        cells_glb = output_dir / f"681_voronoi_cells_{tag}.glb"
        export_voronoi_cells_glb(vor_npz, cells_glb, aligned_npz)
        print(_green(f"  ✓ → {cells_glb}"))

        print(_cyan(f"\n[Step 8] Extracting Voronoi edges ({tag}) …"))
        edges_npz = output_dir / f"681_voronoi_edges_{tag}.npz"
        edges = extract_voronoi_edges(vor_npz, edges_npz)
        print(f"  {len(edges)} unique edges")
        print(_green(f"  ✓ → {edges_npz}"))

        print(_cyan(f"\n[Step 9] Voxelizing skeleton ({tag}, subdivision={subdivision}, "
              f"dilation_radius={dilation_radius}) …"))
        skel_npz = output_dir / f"681_skeleton_voxels_{tag}.npz"
        voxelize_skeleton(
            edges_npz_path=edges_npz,
            aligned_npz_path=aligned_npz,
            output_npz_path=skel_npz,
            subdivision=subdivision,
            dilation_radius_fine_voxels=dilation_radius,
        )
        print(_green(f"  ✓ → {skel_npz}"))

        print(_cyan(f"\n[Step 10] Marching Cubes mesh export ({tag}, smooth_sigma={mc_smooth_sigma}) …"))
        skel_glb = output_dir / f"681_skeleton_{tag}.glb"
        skel_stl = output_dir / f"681_skeleton_{tag}.stl"
        mesh_from_voxels(
            skeleton_npz_path=skel_npz,
            output_glb_path=skel_glb,
            output_stl_path=skel_stl,
            smooth_sigma=mc_smooth_sigma,
        )
        print(_green(f"  ✓ → {skel_glb}"))
        print(_green(f"  ✓ → {skel_stl}"))
        return skel_glb, skel_stl

    # Run Steps 7-10 for density-sampled seeds (inverse-transform sampling)
    skeleton_glb, skeleton_stl = _run_voronoi_to_mesh(seeds_npz, "density")

    # Run Steps 7-10 for CVT seeds (if enabled)
    cvt_glb, cvt_stl = None, None
    if cvt_seeds_npz is not None:
        cvt_glb, cvt_stl = _run_voronoi_to_mesh(cvt_seeds_npz, f"cvt{cvt_iters}")

    # ------------------------------------------------------------------
    # Copy GLBs (viewer) to viewer_dir if requested
    # STL files stay in output_dir — they are the final deliverable meshes
    # ------------------------------------------------------------------
    if viewer_dir is not None:
        viewer_dir.mkdir(parents=True, exist_ok=True)
        glbs_to_copy = [raw_glb, skeleton_glb]
        if cvt_glb is not None:
            glbs_to_copy.append(cvt_glb)
        for glb in glbs_to_copy:
            dest = viewer_dir / glb.name
            shutil.copy2(glb, dest)
            print(_green(f"  ✓ Copied → {dest}"))

    print(_bold(_green("\n✓ Pipeline complete.")))
    print(f"  Outputs: {output_dir}")
    print(f"  STL files: {skeleton_stl.name}" + (f", {cvt_stl.name}" if cvt_stl else ""))
    if viewer_dir:
        models = "?model=681_raw  |  ?model=681_skeleton_density"
        if cvt_glb is not None:
            models += f"  |  ?model=681_skeleton_cvt{cvt_iters}"
        print(f"  Viewer:  {models}")


def _export_raw_glb(raw_npz: Path, output_glb: Path) -> None:
    import sys
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ct_reconstruction.glb_export import voxels_to_glb

    data = np.load(str(raw_npz))
    occupancy = data["voxels"].astype(bool)
    origin_mm = data["origin_m"] * 1e3
    spacing_mm = data["voxel_size_xyz_m"] * 1e3

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    voxels_to_glb(
        occupancy=occupancy,
        origin=origin_mm,
        spacing=spacing_mm,
        output_path=output_glb,
        color=(0.76, 0.60, 0.42),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="matlab2stl-pipeline",
        description="End-to-end pipeline: 681.mat → scaffold mesh (STL + GLB)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run-pipeline", help="Run the 9-step pipeline")
    p.add_argument("--mat", type=Path, default=Path("datasets/681.mat"),
                   help="Input .mat file (default: datasets/681.mat)")
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/matlab2stl_pipeline"),
                   help="Directory for all intermediate and final outputs")
    p.add_argument("--viewer-dir", type=Path, default=None,
                   help="If set, copy GLBs here (e.g. viewer/public/data)")
    p.add_argument("--num-seeds", type=int, default=200)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--subdivision", type=int, default=10,
                   help="Fine-grid subdivision factor (default 10 → 40 µm voxels)")
    p.add_argument("--dilation-radius", type=float, default=3.0,
                   help="Morphological dilation radius in fine voxels (default 3)")
    p.add_argument("--cvt-iters", type=int, default=500,
                   help="Lloyd CVT relaxation iterations after seed sampling (0 = disabled, default 500)")
    p.add_argument("--mc-smooth-sigma", type=float, default=1.0,
                   help="Gaussian pre-smoothing sigma (voxels) before Marching Cubes (0 = disabled, default 1.0)")

    args = parser.parse_args()

    if args.command == "run-pipeline":
        run_pipeline(
            mat_path=args.mat,
            output_dir=args.output_dir,
            viewer_dir=args.viewer_dir,
            num_seeds=args.num_seeds,
            gamma=args.gamma,
            subdivision=args.subdivision,
            dilation_radius=args.dilation_radius,
            cvt_iters=args.cvt_iters,
            mc_smooth_sigma=args.mc_smooth_sigma,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
