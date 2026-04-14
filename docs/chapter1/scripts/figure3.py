"""
Figure 1-3: Lloyd 迭代收敛曲线（最大位移 vs 迭代次数）
输出: docs/chapter1/figures/1-3-lloyd-convergence.png

通过对 seeds=100, cvt=500 重新执行 lloyd_relax 并记录每步位移来生成数据。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import _font_setup  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from matlab2stl_pipeline.box_voronoi import _mirror_seeds, _clip_convex_hull_to_box
import scipy.spatial

ROOT = Path(__file__).parents[3]
aligned_npz = ROOT / "outputs/matlab2stl_pipeline/681_aligned_density_gamma1.npz"
seeds_npz   = ROOT / "outputs/matlab2stl_pipeline/seeds_100.npz"

d = np.load(aligned_npz)
s = np.load(seeds_npz)
seed_points = s["seed_points"].astype(np.float64)
grid_shape = d["grid_shape_xyz"].tolist()

box_min = np.zeros(3)
box_max = np.array(grid_shape, dtype=np.float64) - 1

# Run Lloyd manually, recording max displacement per iteration
seeds = seed_points.copy()
N = len(seeds)
K = 200
displacements = []

for it in range(K):
    all_seeds = _mirror_seeds(seeds, box_min, box_max)
    vor = scipy.spatial.Voronoi(all_seeds)
    new_seeds = seeds.copy()
    for i in range(N):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
        verts = vor.vertices[np.array(region)]
        if len(verts) < 4:
            continue
        clipped = _clip_convex_hull_to_box(verts, box_min, box_max)
        if clipped is not None and len(clipped) >= 4:
            new_seeds[i] = clipped.mean(axis=0)
    disp = np.max(np.linalg.norm(new_seeds - seeds, axis=1))
    displacements.append(disp)
    seeds = new_seeds
    if (it + 1) % 20 == 0:
        print(f"  iter {it+1}/{K}  max_disp={disp:.4f}")

displacements = np.array(displacements)

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(np.arange(1, K+1), displacements, color="steelblue", lw=1.5,
            label="最大位移 $\\max_i \\|\\mathbf{s}_i^{(t+1)}-\\mathbf{s}_i^{(t)}\\|$")
ax.set_xlabel("Lloyd 迭代次数 $t$", fontsize=11)
ax.set_ylabel(r"最大位移（体素单位）", fontsize=11)
ax.set_title("图 1-3  Lloyd 迭代收敛曲线（$N=100$ 种子点）", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()

out = Path(__file__).parents[1] / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "1-3-lloyd-convergence.png", dpi=150, bbox_inches="tight")
print("Saved 1-3-lloyd-convergence.png")
