"""
Figure 1-10: 膨胀半径 r_dil 对晶格相对密度的影响
输出: docs/chapter1/figures/1-10-dilation-radius-vs-density.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import _font_setup  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from matlab2stl_pipeline.skeleton_voxelizer import voxelize_skeleton

ROOT = Path(__file__).parents[3]
aligned_npz = ROOT / "outputs/matlab2stl_pipeline/681_aligned_density_gamma1.npz"
edges_npz   = ROOT / "outputs/matlab2stl_pipeline/batch/seeds100_cvt500/voronoi_edges.npz"

import tempfile, os

radii = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
densities = []

print("Sweeping dilation radii...")
for r in radii:
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp = Path(f.name)
    try:
        voxelize_skeleton(edges_npz, aligned_npz, tmp,
                          subdivision=10, dilation_radius_fine_voxels=r)
        vd = np.load(tmp)
        dilated = vd["voxels"].astype(bool)
        phi = dilated.sum() / dilated.size
        densities.append(phi)
        print(f"  r={r:.1f}  phi={phi:.4f}")
    finally:
        os.unlink(tmp)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(radii, [d * 100 for d in densities],
        "o-", color="steelblue", lw=2, ms=7, label="相对密度 $\\phi$")
ax.axvline(3.0, color="crimson", linestyle="--", lw=1.5,
           label="默认值 $r_{\\mathrm{dil}} = 3.0$")
ax.set_xlabel("膨胀半径 $r_{\\mathrm{dil}}$（精细体素单位）", fontsize=11)
ax.set_ylabel("相对密度 $\\phi$ (%)", fontsize=11)
ax.set_title("图 1-10  膨胀半径对晶格相对密度的影响（$N=100$，CVT 500 步）",
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(densities) * 110)
fig.tight_layout()

out = Path(__file__).parents[1] / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "1-10-dilation-radius-vs-density.png", dpi=150, bbox_inches="tight")
print("Saved 1-10-dilation-radius-vs-density.png")
