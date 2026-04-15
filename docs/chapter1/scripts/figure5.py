"""
Figure 1-5: 受限 Voronoi 单元三维可视化（随机着色多面体）
输出: docs/chapter1/figures/1-5-voronoi-cells-3d.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import _font_setup  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.spatial

ROOT = Path(__file__).parents[3]
vor_npz = ROOT / "outputs/matlab2stl_pipeline/batch/seeds100_cvt500/voronoi.npz"

d = np.load(vor_npz, allow_pickle=True)
cell_vertices_arr = d["cell_vertices"]
cell_simplices_arr = d["cell_hull_simplices"]
seed_points = d["seed_points"]
box_min = d["box_min"].astype(float)
box_max = d["box_max"].astype(float)

# Display up to 30 cells
n_display = min(30, len(cell_vertices_arr))
cmap = plt.cm.tab20
colors = [cmap(i / n_display) for i in range(n_display)]

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")

for i in range(n_display):
    verts = cell_vertices_arr[i]
    simplices = cell_simplices_arr[i]
    if verts.shape[0] < 4 or simplices.shape[0] == 0:
        continue
    tris = [[verts[j] for j in tri] for tri in simplices]
    poly = Poly3DCollection(tris, alpha=0.18,
                            facecolor=colors[i], edgecolor="k", linewidth=0.2)
    ax.add_collection3d(poly)

# Seed points
sp = seed_points[:n_display]
ax.scatter(sp[:, 0], sp[:, 1], sp[:, 2],
           c="crimson", s=20, zorder=5, label="种子点")

ax.set_xlim(box_min[0], box_max[0])
ax.set_ylim(box_min[1], box_max[1])
ax.set_zlim(box_min[2], box_max[2])
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
ax.legend(fontsize=10)
ax.set_title(f"图 1-5  受限 Voronoi 单元（展示 {n_display} 个，CVT 500 步）",
             fontsize=11, pad=12)
fig.tight_layout()

out = Path(__file__).parents[1] / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "1-5-voronoi-cells-3d.png", dpi=150, bbox_inches="tight")
print("Saved 1-5-voronoi-cells-3d.png")
