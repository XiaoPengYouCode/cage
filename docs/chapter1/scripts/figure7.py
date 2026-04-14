"""
Figure 1-7: Voronoi 脊线网络三维可视化
输出: docs/chapter1/figures/1-7-ridge-network-3d.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import _font_setup  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

ROOT = Path(__file__).parents[3]
vor_npz   = ROOT / "outputs/matlab2stl_pipeline/batch/seeds100_cvt500/voronoi.npz"
edges_npz = ROOT / "outputs/matlab2stl_pipeline/batch/seeds100_cvt500/voronoi_edges.npz"

d = np.load(vor_npz, allow_pickle=True)
seed_points = d["seed_points"]
box_min = d["box_min"].astype(float)
box_max = d["box_max"].astype(float)

e = np.load(edges_npz)
edges = e["edges"]   # (E, 2, 3)

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")

lc = Line3DCollection(edges, colors="steelblue", linewidths=0.6, alpha=0.6)
ax.add_collection3d(lc)

ax.scatter(seed_points[:, 0], seed_points[:, 1], seed_points[:, 2],
           c="crimson", s=15, zorder=5, label=f"种子点 ($N={len(seed_points)}$)")

ax.set_xlim(box_min[0], box_max[0])
ax.set_ylim(box_min[1], box_max[1])
ax.set_zlim(box_min[2], box_max[2])
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
ax.legend(fontsize=10)
ax.set_title(f"图 1-7  Voronoi 脊线网络（{len(edges)} 条脊线，CVT 500 步）",
             fontsize=11, pad=12)
fig.tight_layout()

out = Path(__file__).parents[1] / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "1-7-ridge-network-3d.png", dpi=150, bbox_inches="tight")
print("Saved 1-7-ridge-network-3d.png")
