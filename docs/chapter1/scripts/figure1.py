"""
Figure 1-1: 密度场切片与概率采样种子点可视化
输出: docs/chapter1/figures/1-density-field-and-seeds.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import _font_setup  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).parents[3]
aligned_npz = ROOT / "outputs/matlab2stl_pipeline/681_aligned_density_gamma1.npz"
# Use seeds from any batch experiment (cvt=1 = density-sampled only)
seeds_npz = ROOT / "outputs/matlab2stl_pipeline/seeds_100.npz"

d = np.load(aligned_npz)
density = d["density_milli"].astype(np.float32)   # (nx, ny, nz)
nx, ny, nz = density.shape

s = np.load(seeds_npz)
seeds = s["seed_points"]   # (N, 3) voxel coords

gamma = 1.0
prob = density ** gamma
prob_sum = prob.sum()
prob_norm = prob / prob_sum if prob_sum > 0 else prob

# Density at each seed (for colour)
seed_idx = np.clip(seeds.astype(int), 0,
                   np.array([nx-1, ny-1, nz-1]))
rho_at_seeds = density[seed_idx[:, 0], seed_idx[:, 1], seed_idx[:, 2]]

fig = plt.figure(figsize=(13, 5))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# --- Left: density field mid-slice (z-mid) ---
ax0 = fig.add_subplot(gs[0])
z_mid = nz // 2
slice_xz = density[:, :, z_mid].T   # shape (ny, nx), imshow expects (rows, cols)
im0 = ax0.imshow(slice_xz, origin="lower", cmap="Blues",
                 vmin=0, vmax=1, aspect="equal")
plt.colorbar(im0, ax=ax0, label=r"$\rho(\mathbf{x})$", fraction=0.046, pad=0.04)
ax0.set_title(r"SIMP 伪密度场（$z = z_{\mathrm{mid}}$ 截面）", fontsize=11)
ax0.set_xlabel("$x$ (voxel)")
ax0.set_ylabel("$y$ (voxel)")

# --- Right: seed points projected onto xy plane, coloured by rho ---
ax1 = fig.add_subplot(gs[1])
# Background: density mid-slice
ax1.imshow(slice_xz, origin="lower", cmap="Blues", vmin=0, vmax=1,
           aspect="equal", alpha=0.4)
sc = ax1.scatter(seeds[:, 0], seeds[:, 1],
                 c=rho_at_seeds, cmap="Reds",
                 s=6, alpha=0.85, linewidths=0,
                 vmin=0, vmax=1)
plt.colorbar(sc, ax=ax1, label=r"$\rho$ at seed", fraction=0.046, pad=0.04)
ax1.set_xlim(0, nx)
ax1.set_ylim(0, ny)
ax1.set_title(f"概率加权采样种子点（$N={len(seeds)}$，$\\gamma={gamma}$）", fontsize=11)
ax1.set_xlabel("$x$ (voxel)")
ax1.set_ylabel("$y$ (voxel)")

fig.suptitle("图 1-1  密度场与概率采样可视化", fontsize=12, y=1.01)
fig.tight_layout()

out = Path(__file__).parents[1] / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "1-1-density-field-and-seeds.png", dpi=150, bbox_inches="tight")
print("Saved 1-1-density-field-and-seeds.png")
