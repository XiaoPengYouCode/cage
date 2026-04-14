"""
Figure 1-9: 骨架体素化三步对比图
(a) 脊线线框  (b) 二值骨架切片  (c) 形态学膨胀后切片
输出: docs/chapter1/figures/1-9-voxelization-steps.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import _font_setup  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[3]
batch = ROOT / "outputs/matlab2stl_pipeline/batch/seeds100_cvt500"
edges_npz = batch / "voronoi_edges.npz"
vox_npz   = batch / "skeleton_voxels.npz"

edges = np.load(edges_npz)["edges"]   # (E, 2, 3) voxel coords
vox_d = np.load(vox_npz)
dilated = vox_d["voxels"].astype(bool)   # after dilation

# Reconstruct sparse skeleton inline (rasterize without dilation)
grid_shape = vox_d["grid_shape_xyz"].tolist()   # [nx, ny, nz] of fine grid
pad = int(vox_d["pad_fine_voxels"])
skeleton = np.zeros(grid_shape, dtype=bool)
n_sub = 10
for edge in edges:
    p0, p1 = edge[0].astype(np.float64), edge[1].astype(np.float64)
    for k in range(n_sub):
        t = k / max(n_sub - 1, 1)
        pt = p0 + t * (p1 - p0)
        # Convert voxel coord → fine grid index
        scale = (np.array(grid_shape) - 2 * pad - 1) / (np.array(dilated.shape) - 2 * pad - 1 + 1e-9)
        ix = int(np.clip(round(pt[0] * scale[0] + pad), 0, grid_shape[0]-1))
        iy = int(np.clip(round(pt[1] * scale[1] + pad), 0, grid_shape[1]-1))
        iz = int(np.clip(round(pt[2] * scale[2] + pad), 0, grid_shape[2]-1))
        skeleton[ix, iy, iz] = True

nx, ny, nz = dilated.shape
z_mid = nz // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- (a) Ridge line segments projected onto xy plane ---
ax = axes[0]
for edge in edges:
    p0, p1 = edge[0], edge[1]
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
            color="steelblue", lw=0.6, alpha=0.7)
ax.set_xlim(0, dilated.shape[0])
ax.set_ylim(0, dilated.shape[1])
ax.set_aspect("equal")
ax.set_title(f"(a) 脊线段（$xy$ 投影，共 {len(edges)} 条）", fontsize=10)
ax.set_xlabel("$x$ (voxel)")
ax.set_ylabel("$y$ (voxel)")
ax.set_facecolor("#f5f5f5")

# --- (b) Binary skeleton mid-slice ---
ax = axes[1]
ax.imshow(skeleton[:, :, z_mid].T, origin="lower", cmap="Greys",
          vmin=0, vmax=1, aspect="equal")
ax.set_title(f"(b) 二值骨架（$z={z_mid}$ 截面）\n体素数 = {skeleton.sum():,}", fontsize=10)
ax.set_xlabel("$x$ (voxel)")
ax.set_ylabel("$y$ (voxel)")

# --- (c) Dilated voxels mid-slice ---
ax = axes[2]
ax.imshow(dilated[:, :, z_mid].T, origin="lower", cmap="Blues",
          vmin=0, vmax=1, aspect="equal")
ax.set_title(f"(c) 形态学膨胀后（$z={z_mid}$ 截面）\n体素数 = {dilated.sum():,}", fontsize=10)
ax.set_xlabel("$x$ (voxel)")
ax.set_ylabel("$y$ (voxel)")

fig.suptitle("图 1-9  骨架体素化三步对比", fontsize=12, y=1.01)
fig.tight_layout()

out = Path(__file__).parents[1] / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "1-9-voxelization-steps.png", dpi=150, bbox_inches="tight")
print("Saved 1-9-voxelization-steps.png")
