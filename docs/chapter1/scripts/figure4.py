"""
Figure 1-4: Lloyd 迭代前后种子点分布对比
输出: docs/chapter1/figures/1-4-lloyd-before-after.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import _font_setup  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[3]
aligned_npz = ROOT / "outputs/matlab2stl_pipeline/681_aligned_density_gamma1.npz"
seeds_before_npz = ROOT / "outputs/matlab2stl_pipeline/seeds_100.npz"
seeds_after_npz  = ROOT / "outputs/matlab2stl_pipeline/batch/seeds100_cvt500/seeds_cvt500.npz"

d = np.load(aligned_npz)
density = d["density_milli"].astype(np.float32)
nx, ny, nz = density.shape
z_mid = nz // 2
slice_xz = density[:, :, z_mid].T

before = np.load(seeds_before_npz)["seed_points"]
after  = np.load(seeds_after_npz)["seed_points"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, seeds, title in zip(
    axes,
    [before, after],
    [f"Lloyd 迭代前（密度采样，$N={len(before)}$）",
     f"Lloyd 迭代后（CVT 收敛，500 步）"]
):
    ax.imshow(slice_xz, origin="lower", cmap="Blues", vmin=0, vmax=1,
              aspect="equal", alpha=0.5)
    ax.scatter(seeds[:, 0], seeds[:, 1],
               c="crimson", s=8, alpha=0.85, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("$x$ (voxel)")

axes[0].set_ylabel("$y$ (voxel)")

fig.suptitle("图 1-4  Lloyd 迭代前后种子点分布对比（$z = z_{\\mathrm{mid}}$ 投影）",
             fontsize=12, y=1.01)
fig.tight_layout()

out = Path(__file__).parents[1] / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "1-4-lloyd-before-after.png", dpi=150, bbox_inches="tight")
print("Saved 1-4-lloyd-before-after.png")
