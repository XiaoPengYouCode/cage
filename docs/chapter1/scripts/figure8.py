"""
Figure 1-8: 脊线度数分布（每个种子点连接的脊线段数）
输出: docs/chapter1/figures/1-8-ridge-degree-distribution.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import _font_setup  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from matlab2stl_pipeline.box_voronoi import _mirror_seeds
import scipy.spatial

ROOT = Path(__file__).parents[3]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, tag, label in zip(
    axes,
    ["seeds100_cvt1", "seeds100_cvt500"],
    ["CVT 1 步（近似初始采样）", "CVT 500 步（收敛）"]
):
    vor_npz = ROOT / f"outputs/matlab2stl_pipeline/batch/{tag}/voronoi.npz"
    d = np.load(vor_npz, allow_pickle=True)
    seed_points = d["seed_points"].astype(np.float64)
    box_min = d["box_min"].astype(np.float64)
    box_max = d["box_max"].astype(np.float64)
    N = int(d["n_cells"])

    all_seeds = _mirror_seeds(seed_points, box_min, box_max)
    vor = scipy.spatial.Voronoi(all_seeds)

    # Count ridges per real seed
    degree = np.zeros(N, dtype=int)
    for (i, j) in vor.ridge_points:
        if i < N:
            degree[i] += 1
        if j < N:
            degree[j] += 1

    vals, counts = np.unique(degree, return_counts=True)
    ax.bar(vals, counts, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(degree.mean(), color="crimson", linestyle="--", lw=1.5,
               label=f"均值 = {degree.mean():.1f}")
    ax.set_xlabel("脊线度数（每个种子点）", fontsize=10)
    ax.set_ylabel("种子点数量", fontsize=10)
    ax.set_title(f"$N={N}$，{label}", fontsize=11)
    ax.legend(fontsize=9)

fig.suptitle("图 1-8  Voronoi 脊线度数分布（晶格节点连通性）",
             fontsize=12, y=1.02)
fig.tight_layout()

out = Path(__file__).parents[1] / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "1-8-ridge-degree-distribution.png", dpi=150, bbox_inches="tight")
print("Saved 1-8-ridge-degree-distribution.png")
