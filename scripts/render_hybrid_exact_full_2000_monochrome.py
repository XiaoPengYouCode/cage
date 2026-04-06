from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topopt_sampling import build_annular_cylinder_domain
from topopt_sampling.exact_brep import build_delaunay_neighbor_map
from topopt_sampling.exact_restricted_voronoi_3d import build_exact_restricted_voronoi_diagram
from topopt_sampling.hybrid_exact_brep import HybridExactDiagramBRep, build_hybrid_exact_cell_brep

sys.path.append(str(Path(__file__).resolve().parent))
from render_hybrid_exact_full_2000_filled import (
    SHELL_SUPPORT_KEYS,
    face_triangles,
    sample_curve,
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def main() -> None:
    seed_npz = Path("datasets/topopt/seed_probability_mapping_2000.npz")
    output_png = Path("docs/assets/hybrid_exact_new_pipeline_2000seeds_monochrome.png")

    import numpy as np

    with np.load(seed_npz) as data:
        seed_points = data["seed_points"].astype(float)

    domain = build_annular_cylinder_domain(xy_size=200, z_size=80, outer_radius=100.0, inner_radius=50.0)

    t0 = time.time()
    print("[1/4] building restricted Voronoi diagram shell...", flush=True)
    diagram = build_exact_restricted_voronoi_diagram(seed_points=seed_points, domain=domain, include_support_traces=False)
    print(f"      done: cells={len(diagram.cells)} elapsed={time.time() - t0:.1f}s", flush=True)

    print("[2/4] building Delaunay neighbor map...", flush=True)
    neighbor_map = build_delaunay_neighbor_map(diagram.seed_points)
    print(f"      done: neighbor entries={len(neighbor_map)} elapsed={time.time() - t0:.1f}s", flush=True)

    print("[3/4] building hybrid exact cells with progress...", flush=True)
    built_cells = []
    for idx, cell in enumerate(diagram.cells, start=1):
        built_cells.append(build_hybrid_exact_cell_brep(cell, diagram, neighbor_map.get(int(cell.seed_id), tuple())))
        if idx <= 10 or idx % 100 == 0 or idx == len(diagram.cells):
            print(f"      cell {idx}/{len(diagram.cells)} elapsed={time.time() - t0:.1f}s", flush=True)
    diagram_brep = HybridExactDiagramBRep(cells=tuple(built_cells))
    print(f"      done: built {len(diagram_brep.cells)} cells elapsed={time.time() - t0:.1f}s", flush=True)

    shell_cells = [cell for cell in diagram_brep.cells if any(face.support_key in SHELL_SUPPORT_KEYS for face in cell.faces)]
    print(f"[4/4] rendering monochrome shell cells: {len(shell_cells)}", flush=True)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    face_color = (0.88, 0.90, 0.93, 1.0)
    total_faces = 0
    total_tris = 0
    total_edges = 0

    for idx, cell in enumerate(shell_cells, start=1):
        edge_lookup = {edge.edge_id: edge for edge in cell.edges}
        support_lookup = {support.key: support for support in cell.supports}
        face_polys = []
        boundary_drawn = set()
        for face in cell.faces:
            if face.support_key not in SHELL_SUPPORT_KEYS:
                continue
            tris = face_triangles(face, edge_lookup, support_lookup)
            if tris:
                face_polys.extend(tris)
                total_faces += 1
                total_tris += len(tris)
            for loop in face.loop_edge_ids:
                for edge_id in loop:
                    if edge_id in boundary_drawn:
                        continue
                    edge = edge_lookup[edge_id]
                    if not any(key in SHELL_SUPPORT_KEYS for key in edge.support_keys):
                        continue
                    pts = sample_curve(edge.curve, num=36)
                    ax.plot(
                        pts[:, 0], pts[:, 1], pts[:, 2],
                        color=(0.05, 0.05, 0.05, 1.0), linewidth=1.15, solid_capstyle="round"
                    )
                    boundary_drawn.add(edge_id)
                    total_edges += 1
        if face_polys:
            coll = Poly3DCollection(face_polys, facecolor=face_color, edgecolor="none", alpha=1.0, antialiased=False)
            ax.add_collection3d(coll)
        if idx <= 10 or idx % 100 == 0 or idx == len(shell_cells):
            print(f"      rendered shell cell {idx}/{len(shell_cells)} faces={total_faces} tris={total_tris} edges={total_edges} elapsed={time.time() - t0:.1f}s", flush=True)

    ax.set_xlim(domain.center_xy[0] - domain.outer_radius, domain.center_xy[0] + domain.outer_radius)
    ax.set_ylim(domain.center_xy[1] - domain.outer_radius, domain.center_xy[1] + domain.outer_radius)
    ax.set_zlim(domain.z_min, domain.z_max)
    ax.set_box_aspect((2, 2, 0.8))
    ax.view_init(elev=24, azim=36)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True, alpha=0.18)
    ax.set_title(f"Monochrome shell blocks + black boundaries | shell_cells={len(shell_cells)} | faces={total_faces} | edges={total_edges}")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved image: {output_png} total_elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
