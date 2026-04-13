from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np


def _add_stl_surface(ax, m, color: str = "#c8a882", alpha: float = 0.7) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    verts = m.vectors  # (T, 3, 3)  x/y/z
    # Reorder to (x, y, z) — STL is already in that order
    mesh = Poly3DCollection(verts, alpha=alpha, linewidth=0)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)


def _add_voxel_surface(ax, occupancy: np.ndarray, origin: np.ndarray,
                       spacing: np.ndarray, color: str = "#4a90d9",
                       alpha: float = 0.85) -> None:
    from skimage.measure import marching_cubes
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt

    verts, faces, _, _ = marching_cubes(
        occupancy.astype(np.float32), level=0.5,
        spacing=(spacing[0], spacing[1], spacing[2]),
    )
    # verts in (x,y,z) mm relative to grid, shift by origin
    verts_world = verts + origin

    z_vals = verts_world[faces, 2].mean(axis=1)
    z_norm = (z_vals - z_vals.min()) / max(z_vals.max() - z_vals.min(), 1.0)
    colors = plt.cm.Blues(0.35 + 0.5 * z_norm)

    mesh = Poly3DCollection(verts_world[faces], alpha=alpha, linewidth=0)
    mesh.set_facecolor(colors)
    ax.add_collection3d(mesh)


def _set_equal_axes(ax, lo: np.ndarray, hi: np.ndarray) -> None:
    max_range = (hi - lo).max()
    mid = (lo + hi) / 2
    for setter, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        setter(m - max_range / 2, m + max_range / 2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (mm)", fontsize=8, labelpad=2)
    ax.set_ylabel("y (mm)", fontsize=8, labelpad=2)
    ax.set_zlabel("z (mm)", fontsize=8, labelpad=2)
    ax.tick_params(labelsize=7)


# Four canonical views: (elev, azim, label)
_VIEWS = [
    (0,   0,   "Front"),
    (0,   90,  "Side"),
    (90,  0,   "Top"),
    (30, -60,  "Isometric"),
]


def render_comparison(
    m,                      # stl_mesh.Mesh
    occupancy: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    output_path: Path,
) -> Path:
    """2×4 grid: top row = STL surface, bottom row = voxelized mesh.
    Columns are the 4 canonical views (Front / Side / Top / Isometric).
    """
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from stl import mesh as stl_mesh
    verts = m.vectors.reshape(-1, 3)
    stl_lo, stl_hi = verts.min(axis=0), verts.max(axis=0)

    nx, ny, nz = occupancy.shape
    vox_lo = origin
    vox_hi = origin + np.array([nx, ny, nz]) * spacing

    # Common bounds that cover both
    lo = np.minimum(stl_lo, vox_lo)
    hi = np.maximum(stl_hi, vox_hi)

    fig, axes = plt.subplots(
        2, 4, figsize=(20, 10),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )

    for col, (elev, azim, label) in enumerate(_VIEWS):
        # --- STL row ---
        ax = axes[0, col]
        _add_stl_surface(ax, m)
        _set_equal_axes(ax, lo, hi)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"STL — {label}", fontsize=9)

        # --- Voxel row ---
        ax = axes[1, col]
        _add_voxel_surface(ax, occupancy, origin, spacing)
        _set_equal_axes(ax, lo, hi)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Voxel — {label}", fontsize=9)

    bone_pct = 100.0 * occupancy.sum() / occupancy.size
    fig.suptitle(
        f"LumbarVertebrae  STL ({len(m.vectors):,} triangles)  vs  "
        f"Voxel {nx}×{ny}×{nz} @ {spacing[0]:.2f} mm  "
        f"(fill {bone_pct:.1f}%)",
        fontsize=12,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
