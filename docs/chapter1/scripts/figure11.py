"""
Figure 1-11: 最终晶格网格三视角效果图（正视 / 侧视 / 轴测）
输出: docs/chapter1/figures/1-11-final-mesh-views.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import _font_setup  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).parents[3]
vox_npz = ROOT / "outputs/matlab2stl_pipeline/batch/seeds100_cvt500/skeleton_voxels.npz"

# Rebuild mesh via marching cubes inline for flexible rendering
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter

vd = np.load(vox_npz)
dilated = vd["voxels"].astype(np.float32)
origin_m = vd["origin_m"].astype(np.float64)
voxel_size_m = vd["voxel_size_xyz_m"].astype(np.float64)

field = np.pad(dilated, 1, mode="constant", constant_values=0.0)
field = gaussian_filter(field, sigma=1.0)
spacing = voxel_size_m * 1e3   # mm

mc_verts, mc_faces, mc_normals, _ = marching_cubes(
    field, level=0.5,
    spacing=(spacing[0], spacing[1], spacing[2]),
    allow_degenerate=False
)
mc_verts = mc_verts - spacing                        # undo pad offset
mc_verts = mc_verts + origin_m * 1e3                # world coords mm
mc_faces_fixed = mc_faces[:, [0, 2, 1]]             # outward winding

# Subsample faces for rendering speed
rng = np.random.default_rng(0)
n_faces = len(mc_faces_fixed)
max_faces = 40000
if n_faces > max_faces:
    idx = rng.choice(n_faces, max_faces, replace=False)
    faces_plot = mc_faces_fixed[idx]
else:
    faces_plot = mc_faces_fixed

tris = mc_verts[faces_plot]   # (F, 3, 3)

views = [
    ("正视（XY）",  (90, -90)),
    ("侧视（YZ）",  (0,  0)),
    ("轴测",        (25, -50)),
]

fig = plt.figure(figsize=(15, 5))
for col, (title, (elev, azim)) in enumerate(views):
    ax = fig.add_subplot(1, 3, col + 1, projection="3d")
    poly = Poly3DCollection(tris, alpha=0.85,
                            facecolor="lightsteelblue",
                            edgecolor="none")
    ax.add_collection3d(poly)
    # Set limits
    mins = mc_verts.min(axis=0)
    maxs = mc_verts.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("$x$ (mm)", fontsize=8)
    ax.set_ylabel("$y$ (mm)", fontsize=8)
    ax.set_zlabel("$z$ (mm)", fontsize=8)
    ax.tick_params(labelsize=7)

fig.suptitle(
    f"图 1-11  最终晶格网格三视角（$N=100$，CVT 500 步，{n_faces:,} 三角面片）",
    fontsize=12, y=1.01
)
fig.tight_layout()

out = Path(__file__).parents[1] / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "1-11-final-mesh-views.png", dpi=150, bbox_inches="tight")
print("Saved 1-11-final-mesh-views.png")
