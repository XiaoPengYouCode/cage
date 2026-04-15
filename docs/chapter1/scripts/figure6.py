"""
Figure 1-6: Voronoi 单元体积分布直方图
输出: docs/chapter1/figures/1-6-cell-volume-distribution.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import _font_setup  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial

ROOT = Path(__file__).parents[3]

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

for ax, tag, label in zip(
    axes,
    ["seeds100_cvt1", "seeds100_cvt500"],
    ["密度采样（CVT 0 步）", "CVT 收敛（500 步）"]
):
    vor_npz = ROOT / f"outputs/matlab2stl_pipeline/batch/{tag}/voronoi.npz"
    d = np.load(vor_npz, allow_pickle=True)
    verts_arr = d["cell_vertices"]
    simplices_arr = d["cell_hull_simplices"]
    box_min = d["box_min"].astype(float)
    box_max = d["box_max"].astype(float)
    N = int(d["n_cells"])

    volumes = []
    for verts, simplices in zip(verts_arr, simplices_arr):
        if verts.shape[0] < 4 or simplices.shape[0] == 0:
            volumes.append(0.0)
            continue
        try:
            hull = scipy.spatial.ConvexHull(verts.astype(float))
            volumes.append(hull.volume)
        except Exception:
            volumes.append(0.0)

    volumes = np.array(volumes)
    box_vol = np.prod(box_max - box_min)
    ideal_vol = box_vol / N

    ax.hist(volumes, bins=30, density=True, color="steelblue", alpha=0.7,
            edgecolor="white", linewidth=0.5, label="实际分布")
    ax.axvline(ideal_vol, color="crimson", linestyle="--", lw=1.5,
               label=f"理想均匀体积 = {ideal_vol:.1f}")
    ax.set_xlabel("单元体积（体素单位³）", fontsize=10)
    ax.set_ylabel("频率密度", fontsize=10)
    ax.set_title(f"$N={N}$，{label}", fontsize=11)
    ax.legend(fontsize=9)

    cv = volumes[volumes > 0].std() / volumes[volumes > 0].mean()
    ax.text(0.97, 0.95, f"CV = {cv:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.7))

fig.suptitle("图 1-6  Voronoi 单元体积分布（CVT 均匀性评估）",
             fontsize=12, y=1.02)
fig.tight_layout()

out = Path(__file__).parents[1] / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "1-6-cell-volume-distribution.png", dpi=150, bbox_inches="tight")
print("Saved 1-6-cell-volume-distribution.png")
