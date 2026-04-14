"""
Figure 1-2: 集中参数 γ 对采样分布的影响（γ = 1, 2, 3, 5）
输出: docs/chapter1/figures/1-2-gamma-sensitivity.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import _font_setup  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[3]
aligned_npz = ROOT / "outputs/matlab2stl_pipeline/681_aligned_density_gamma1.npz"

d = np.load(aligned_npz)
density = d["density_milli"].astype(np.float32)
nx, ny, nz = density.shape
z_mid = nz // 2

rng = np.random.default_rng(42)
N = 100
gammas = [1, 2, 3, 5]

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
slice_xz = density[:, :, z_mid].T

for ax, gamma in zip(axes, gammas):
    # Build probability field
    prob = density.astype(np.float64) ** gamma
    flat_idx = np.where(prob.ravel() > 0)[0]
    p_flat = prob.ravel()[flat_idx]
    p_flat /= p_flat.sum()

    chosen = rng.choice(len(flat_idx), size=N, replace=False, p=p_flat)
    chosen_idx = flat_idx[chosen]
    xi = chosen_idx % nx
    yi = (chosen_idx // nx) % ny

    rho_at = density.ravel()[flat_idx[chosen]]

    ax.imshow(slice_xz, origin="lower", cmap="Blues", vmin=0, vmax=1,
              aspect="equal", alpha=0.5)
    sc = ax.scatter(xi, yi, c=rho_at, cmap="Reds", s=8,
                    alpha=0.9, linewidths=0, vmin=0, vmax=1)
    ax.set_title(rf"$\gamma = {gamma}$", fontsize=12)
    ax.set_xlabel("$x$ (voxel)")
    if ax is axes[0]:
        ax.set_ylabel("$y$ (voxel)")

# shared colorbar
fig.subplots_adjust(right=0.88, wspace=0.08)
cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(0, 1))
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label=r"$\rho$ at seed")

fig.suptitle(r"图 1-2  集中参数 $\gamma$ 对采样分布的影响（$N=100$）",
             fontsize=12, y=1.02)

out = Path(__file__).parents[1] / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "1-2-gamma-sensitivity.png", dpi=150, bbox_inches="tight")
print("Saved 1-2-gamma-sensitivity.png")
