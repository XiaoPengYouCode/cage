from __future__ import annotations

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

from topopt_sampling.probability import density_to_probability_intensity


def build_annular_cylinder_chunk(
    xy_size: int,
    z_start: int,
    z_stop: int,
    center_xy: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
) -> np.ndarray:
    cx, cy = center_xy

    xy_coords = np.arange(xy_size, dtype=np.float32) + 0.5
    x, y = np.meshgrid(xy_coords, xy_coords, indexing="ij")
    radial_sq = (x - cx) ** 2 + (y - cy) ** 2
    annulus_mask = (radial_sq >= inner_radius**2) & (radial_sq <= outer_radius**2)

    chunk_depth = z_stop - z_start
    return np.repeat(annulus_mask[:, :, None].astype(np.uint8), chunk_depth, axis=2)


def generate_annular_cylinder_npz(
    output_path: Path,
    xy_size: int = 200,
    z_size: int = 80,
    outer_radius: float = 100.0,
    inner_radius: float = 50.0,
    chunk_depth: int = 8,
) -> None:
    if xy_size <= 0 or z_size <= 0:
        raise ValueError("xy_size and z_size must be positive.")
    if outer_radius <= 0 or inner_radius < 0:
        raise ValueError("outer_radius must be positive and inner_radius must be non-negative.")
    if inner_radius >= outer_radius:
        raise ValueError("inner_radius must be smaller than outer_radius.")
    if outer_radius > xy_size / 2:
        raise ValueError("outer_radius must be smaller than or equal to xy_size / 2 to fit inside the XY plane.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    center_xy = ((xy_size - 1) / 2.0, (xy_size - 1) / 2.0)
    center = np.array([center_xy[0], center_xy[1], (z_size - 1) / 2.0], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        mmap_path = Path(tmp_dir) / "annular_cylinder_voxels.uint8"
        voxels = np.memmap(
            mmap_path,
            dtype=np.uint8,
            mode="w+",
            shape=(xy_size, xy_size, z_size),
        )

        for z_start in range(0, z_size, chunk_depth):
            z_stop = min(z_start + chunk_depth, z_size)
            voxels[:, :, z_start:z_stop] = build_annular_cylinder_chunk(
                xy_size=xy_size,
                z_start=z_start,
                z_stop=z_stop,
                center_xy=center_xy,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
            )
            print(f"generated z-slices [{z_start}, {z_stop})")

        voxel_array = np.asarray(voxels)
        np.savez_compressed(
            output_path,
            voxels=voxel_array,
            xy_size=np.array(xy_size, dtype=np.int32),
            z_size=np.array(z_size, dtype=np.int32),
            outer_radius=np.array(outer_radius, dtype=np.float32),
            inner_radius=np.array(inner_radius, dtype=np.float32),
            center=center,
            shape_name=np.array("annular_cylinder"),
        )

        active_voxels = int(voxel_array.sum())
        fill_ratio = active_voxels / float(xy_size * xy_size * z_size)
        print(f"saved npz: {output_path}")
        print(f"shape: {voxel_array.shape}")
        print(f"active voxels: {active_voxels}")
        print(f"fill ratio: {fill_ratio:.6%}")


def load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def generate_fake_density_result(
    source_npz: Path,
    output_npz: Path,
    chunk_depth: int = 8,
) -> None:
    with np.load(source_npz) as data:
        voxels = data["voxels"]
        metadata = {key: data[key] for key in data.files if key != "voxels"}

    xy_size, _, z_size = voxels.shape
    outer_radius = float(metadata["outer_radius"].item())
    inner_radius = float(metadata["inner_radius"].item())
    center_xy = ((xy_size - 1) / 2.0, (xy_size - 1) / 2.0)

    output_npz.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        mmap_path = Path(tmp_dir) / "fake_density.float32"
        density = np.memmap(
            mmap_path,
            dtype=np.float32,
            mode="w+",
            shape=voxels.shape,
        )

        x_coords = np.arange(xy_size, dtype=np.float32) + 0.5
        y_coords = np.arange(xy_size, dtype=np.float32) + 0.5

        for z_start in range(0, z_size, chunk_depth):
            z_stop = min(z_start + chunk_depth, z_size)
            z_coords = np.arange(z_start, z_stop, dtype=np.float32) + 0.5
            density[:, :, z_start:z_stop] = build_density_chunk(
                voxels_chunk=voxels[:, :, z_start:z_stop],
                x_coords=x_coords,
                y_coords=y_coords,
                z_coords=z_coords,
                z_size=z_size,
                center_xy=center_xy,
                outer_radius=outer_radius,
                inner_radius=inner_radius,
            )
            print(f"generated density z-slices [{z_start}, {z_stop})")

        density_array = np.asarray(density)
        density_milli = np.rint(density_array * 1000.0).astype(np.uint16)
        density_milli = np.where(voxels > 0, density_milli, 0).astype(np.uint16)

        np.savez_compressed(
            output_npz,
            voxels=voxels.astype(np.uint8),
            density_milli=density_milli,
            xy_size=np.array(xy_size, dtype=np.int32),
            z_size=np.array(z_size, dtype=np.int32),
            outer_radius=np.array(outer_radius, dtype=np.float32),
            inner_radius=np.array(inner_radius, dtype=np.float32),
            shape_name=np.array(str(metadata.get("shape_name", np.array("annular_cylinder")).item())),
            result_type=np.array("fake_fea_density"),
            density_precision=np.array(3, dtype=np.int32),
            density_min_nonzero=np.array(0.001, dtype=np.float32),
            density_max=np.array(1.000, dtype=np.float32),
        )

        nonzero = density_milli[density_milli > 0]
        print(f"saved fake density result: {output_npz}")
        print(f"shape: {voxels.shape}")
        print(f"active voxels: {int(voxels.sum())}")
        print(f"nonzero density count: {int(len(nonzero))}")
        print(f"density range: {nonzero.min() / 1000:.3f} .. {nonzero.max() / 1000:.3f}")


def build_density_chunk(
    voxels_chunk: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    z_size: int,
    center_xy: tuple[float, float],
    outer_radius: float,
    inner_radius: float,
) -> np.ndarray:
    cx, cy = center_xy
    x, y, z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

    radial = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    annulus_thickness = max(outer_radius - inner_radius, 1e-6)
    radial_norm = np.clip((radial - inner_radius) / annulus_thickness, 0.0, 1.0)
    z_norm = np.clip((z - 0.5) / max(float(z_size - 1), 1.0), 0.0, 1.0)
    theta = np.arctan2(y - cy, x - cx)

    raw_field = (
        0.52
        + 0.23 * np.sin(3.0 * theta + 5.5 * z_norm)
        + 0.18 * np.cos(2.0 * np.pi * radial_norm)
        + 0.11 * np.sin(7.0 * radial_norm - 4.0 * z_norm)
        + 0.08 * np.cos(9.0 * theta * radial_norm + 2.5 * z_norm)
    )

    amplitude_sum = 0.23 + 0.18 + 0.11 + 0.08
    field_min = 0.52 - amplitude_sum
    field_max = 0.52 + amplitude_sum
    field = np.clip((raw_field - field_min) / max(field_max - field_min, 1e-6), 0.0, 1.0)
    density = 0.001 + 0.999 * field
    density = np.round(density, 3)
    density = np.where(voxels_chunk > 0, density, 0.0)
    return density.astype(np.float32)


def build_density_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "density_pipeline",
        [
            (0.00, "#020617"),
            (0.20, "#1d4ed8"),
            (0.48, "#38bdf8"),
            (0.75, "#fb7185"),
            (1.00, "#7f1d1d"),
        ],
    )


def build_probability_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "sampling_pipeline",
        [
            (0.00, "#f8fafc"),
            (0.18, "#bfdbfe"),
            (0.45, "#60a5fa"),
            (0.72, "#2563eb"),
            (1.00, "#0f172a"),
        ],
    )


def build_voxel_facecolors(values: np.ndarray, cmap: LinearSegmentedColormap) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(values, 0.0, 1.0)
    occupancy = clipped > 0.0
    facecolors = cmap(clipped)
    alpha = 0.50 + 0.40 * clipped
    facecolors[..., 3] = np.where(occupancy, alpha, 0.0)
    return occupancy, facecolors


def aggregate_scalar_field_for_display(
    values: np.ndarray,
    occupancy_mask: np.ndarray,
    block_shape: tuple[int, int, int] = (4, 4, 2),
) -> np.ndarray:
    bx, by, bz = block_shape
    sx, sy, sz = values.shape
    pad_x = (-sx) % bx
    pad_y = (-sy) % by
    pad_z = (-sz) % bz

    padded_values = np.pad(values, ((0, pad_x), (0, pad_y), (0, pad_z)), mode="constant")
    padded_mask = np.pad(occupancy_mask, ((0, pad_x), (0, pad_y), (0, pad_z)), mode="constant")

    reshaped_values = padded_values.reshape(
        padded_values.shape[0] // bx,
        bx,
        padded_values.shape[1] // by,
        by,
        padded_values.shape[2] // bz,
        bz,
    )
    reshaped_mask = padded_mask.reshape(
        padded_mask.shape[0] // bx,
        bx,
        padded_mask.shape[1] // by,
        by,
        padded_mask.shape[2] // bz,
        bz,
    )

    active_sum = (reshaped_values * reshaped_mask.astype(np.float32)).sum(axis=(1, 3, 5))
    active_count = reshaped_mask.sum(axis=(1, 3, 5))
    aggregated = np.divide(
        active_sum,
        np.maximum(active_count, 1),
        out=np.zeros_like(active_sum, dtype=np.float32),
        where=active_count > 0,
    )
    return aggregated.astype(np.float32)


def configure_3d_axes(
    ax: plt.Axes,
    rendered_shape: tuple[int, int, int],
    title: str,
    original_shape: tuple[int, int, int] | None = None,
) -> None:
    source_shape = rendered_shape if original_shape is None else original_shape
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(0, rendered_shape[0])
    ax.set_ylim(0, rendered_shape[1])
    ax.set_zlim(0, rendered_shape[2])
    ax.set_box_aspect(source_shape)
    ax.view_init(elev=21, azim=38)

    x_ticks = np.linspace(0, rendered_shape[0], 5)
    y_ticks = np.linspace(0, rendered_shape[1], 5)
    z_ticks = np.linspace(0, rendered_shape[2], 5)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)
    ax.set_xticklabels([str(int(round(v))) for v in np.linspace(0, source_shape[0], 5)])
    ax.set_yticklabels([str(int(round(v))) for v in np.linspace(0, source_shape[1], 5)])
    ax.set_zticklabels([str(int(round(v))) for v in np.linspace(0, source_shape[2], 5)])


def build_region_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "voronoi_regions",
        [
            (0.00, "#0f172a"),
            (0.16, "#0ea5e9"),
            (0.33, "#22c55e"),
            (0.50, "#facc15"),
            (0.68, "#f97316"),
            (0.84, "#ef4444"),
            (1.00, "#a21caf"),
        ],
    )


def build_region_values(num_seeds: int, rng_seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return rng.random(num_seeds, dtype=np.float32)


def assign_points_to_nearest_seed(points: np.ndarray, seed_points: np.ndarray) -> np.ndarray:
    tree = cKDTree(seed_points)
    _, nearest_seed_indices = tree.query(points, k=1)
    return nearest_seed_indices.astype(np.int32)


def build_cylinder_surface_mesh(
    center_xy: np.ndarray,
    radius: float,
    z_min: float,
    z_max: float,
    theta_steps: int,
    z_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, theta_steps, endpoint=False, dtype=np.float32)
    z_coords = np.linspace(z_min, z_max, z_steps, dtype=np.float32)
    theta_grid, z_grid = np.meshgrid(theta, z_coords, indexing="ij")

    points = np.column_stack(
        (
            center_xy[0] + radius * np.cos(theta_grid).ravel(),
            center_xy[1] + radius * np.sin(theta_grid).ravel(),
            z_grid.ravel(),
        )
    ).astype(np.float32)

    triangles: list[tuple[int, int, int]] = []
    for theta_index in range(theta_steps):
        next_theta_index = (theta_index + 1) % theta_steps
        for z_index in range(z_steps - 1):
            index00 = theta_index * z_steps + z_index
            index01 = theta_index * z_steps + z_index + 1
            index10 = next_theta_index * z_steps + z_index
            index11 = next_theta_index * z_steps + z_index + 1
            triangles.append((index00, index10, index11))
            triangles.append((index00, index11, index01))

    return points, np.asarray(triangles, dtype=np.int32)


def build_annulus_cap_mesh(
    center_xy: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    z_value: float,
    radial_steps: int,
    theta_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    radial_edges = np.linspace(inner_radius**2, outer_radius**2, radial_steps + 1, dtype=np.float32)
    radial_coords = np.sqrt(0.5 * (radial_edges[:-1] + radial_edges[1:])).astype(np.float32)
    theta = np.linspace(0.0, 2.0 * np.pi, theta_steps, endpoint=False, dtype=np.float32)
    radial_grid, theta_grid = np.meshgrid(radial_coords, theta, indexing="ij")

    points = np.column_stack(
        (
            center_xy[0] + radial_grid.ravel() * np.cos(theta_grid).ravel(),
            center_xy[1] + radial_grid.ravel() * np.sin(theta_grid).ravel(),
            np.full(radial_grid.size, z_value, dtype=np.float32),
        )
    ).astype(np.float32)

    triangles: list[tuple[int, int, int]] = []
    for radial_index in range(radial_steps - 1):
        for theta_index in range(theta_steps):
            next_theta_index = (theta_index + 1) % theta_steps
            index00 = radial_index * theta_steps + theta_index
            index01 = radial_index * theta_steps + next_theta_index
            index10 = (radial_index + 1) * theta_steps + theta_index
            index11 = (radial_index + 1) * theta_steps + next_theta_index
            triangles.append((index00, index10, index11))
            triangles.append((index00, index11, index01))

    return points, np.asarray(triangles, dtype=np.int32)


def append_surface_mesh(
    surfaces: list[np.ndarray],
    facecolors: list[np.ndarray],
    points: np.ndarray,
    triangles: np.ndarray,
    region_indices: np.ndarray,
    region_values: np.ndarray,
    cmap: LinearSegmentedColormap,
) -> None:
    triangle_regions = region_indices[triangles]
    triangle_values = region_values[triangle_regions[:, 0]]

    for triangle, value in zip(triangles, triangle_values, strict=False):
        surfaces.append(points[triangle])
        color = np.array(cmap(float(value)), dtype=np.float32)
        color[3] = 0.90
        facecolors.append(color)


def build_voronoi_boundary_surfaces(
    seed_points: np.ndarray,
    xy_size: int,
    z_size: int,
    outer_radius: float,
    inner_radius: float,
    cmap: LinearSegmentedColormap,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    center_xy = np.array([(xy_size - 1) / 2.0, (xy_size - 1) / 2.0], dtype=np.float32)
    region_values = build_region_values(len(seed_points))
    surfaces: list[np.ndarray] = []
    facecolors: list[np.ndarray] = []

    outer_points, outer_triangles = build_cylinder_surface_mesh(
        center_xy=center_xy,
        radius=outer_radius,
        z_min=0.5,
        z_max=z_size - 0.5,
        theta_steps=220,
        z_steps=96,
    )
    outer_regions = assign_points_to_nearest_seed(outer_points, seed_points)
    append_surface_mesh(surfaces, facecolors, outer_points, outer_triangles, outer_regions, region_values, cmap)

    inner_points, inner_triangles = build_cylinder_surface_mesh(
        center_xy=center_xy,
        radius=inner_radius,
        z_min=0.5,
        z_max=z_size - 0.5,
        theta_steps=220,
        z_steps=96,
    )
    inner_regions = assign_points_to_nearest_seed(inner_points, seed_points)
    append_surface_mesh(surfaces, facecolors, inner_points, inner_triangles, inner_regions, region_values, cmap)

    top_points, top_triangles = build_annulus_cap_mesh(
        center_xy=center_xy,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        z_value=z_size - 0.5,
        radial_steps=44,
        theta_steps=220,
    )
    top_regions = assign_points_to_nearest_seed(top_points, seed_points)
    append_surface_mesh(surfaces, facecolors, top_points, top_triangles, top_regions, region_values, cmap)

    bottom_points, bottom_triangles = build_annulus_cap_mesh(
        center_xy=center_xy,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        z_value=0.5,
        radial_steps=44,
        theta_steps=220,
    )
    bottom_regions = assign_points_to_nearest_seed(bottom_points, seed_points)
    append_surface_mesh(surfaces, facecolors, bottom_points, bottom_triangles, bottom_regions, region_values, cmap)

    return surfaces, facecolors, region_values


def render_sampling_overview(
    density_npz: Path,
    seed_npz: Path,
    output_png: Path,
) -> None:
    density_result = load_npz(density_npz)
    seed_result = load_npz(seed_npz)

    density = density_result["density_milli"].astype(np.float32) / 1000.0
    xy_size = int(density_result["xy_size"].item())
    z_size = int(density_result["z_size"].item())
    outer_radius = float(density_result["outer_radius"].item())
    inner_radius = float(density_result["inner_radius"].item())
    seed_points = seed_result["seed_points"].astype(np.float32)
    gamma = float(seed_result["gamma"].item())
    num_seeds = int(seed_result["num_seeds"].item())
    probability = density_to_probability_intensity(density, gamma)
    occupancy_mask = density > 0.0
    display_density = aggregate_scalar_field_for_display(density, occupancy_mask)
    display_probability = aggregate_scalar_field_for_display(probability, occupancy_mask)

    density_cmap = build_density_cmap()
    probability_cmap = build_probability_cmap()
    region_cmap = build_region_cmap()
    density_occ, density_facecolors = build_voxel_facecolors(display_density, density_cmap)
    probability_occ, probability_facecolors = build_voxel_facecolors(display_probability, probability_cmap)
    voronoi_surfaces, voronoi_facecolors, region_values = build_voronoi_boundary_surfaces(
        seed_points=seed_points,
        xy_size=xy_size,
        z_size=z_size,
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        cmap=region_cmap,
    )

    fig = plt.figure(figsize=(26.0, 6.6))
    ax0 = fig.add_subplot(1, 4, 1, projection="3d")
    ax1 = fig.add_subplot(1, 4, 2, projection="3d")
    ax2 = fig.add_subplot(1, 4, 3, projection="3d")
    ax3 = fig.add_subplot(1, 4, 4, projection="3d")
    fig.subplots_adjust(left=0.03, right=0.988, top=0.88, bottom=0.10, wspace=0.18)

    ax0.voxels(
        density_occ,
        facecolors=density_facecolors,
        edgecolor=(1.0, 1.0, 1.0, 0.28),
        linewidth=0.09,
        shade=False,
    )
    configure_3d_axes(ax0, display_density.shape, "1) Fake Topopt Density", density.shape)
    density_mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=1.0), cmap=density_cmap)
    density_mappable.set_array([])
    fig.colorbar(density_mappable, ax=ax0, fraction=0.040, pad=0.06, label="density")

    ax1.voxels(
        probability_occ,
        facecolors=probability_facecolors,
        edgecolor=(1.0, 1.0, 1.0, 0.28),
        linewidth=0.09,
        shade=False,
    )
    configure_3d_axes(ax1, display_probability.shape, f"2) Probability Field (gamma={gamma:.1f})", density.shape)
    probability_mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=1.0), cmap=probability_cmap)
    probability_mappable.set_array([])
    fig.colorbar(probability_mappable, ax=ax1, fraction=0.040, pad=0.06, label="probability")

    scatter = ax2.scatter(
        seed_points[:, 0],
        seed_points[:, 1],
        seed_points[:, 2],
        c=seed_points[:, 2],
        s=7,
        cmap="viridis",
        alpha=0.65,
        linewidths=0.0,
    )
    configure_3d_axes(ax2, density.shape, f"3) Random Seeds (n={num_seeds})", density.shape)
    fig.colorbar(scatter, ax=ax2, fraction=0.040, pad=0.06, label="seed z")

    voronoi_mesh = Poly3DCollection(
        voronoi_surfaces,
        facecolors=np.asarray(voronoi_facecolors),
        edgecolors=(1.0, 1.0, 1.0, 0.08),
        linewidths=0.04,
    )
    ax3.add_collection3d(voronoi_mesh)
    configure_3d_axes(ax3, density.shape, "4) Voronoi Boundary Mesh in Continuous Hollow Cylinder", density.shape)
    region_mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=1.0), cmap=region_cmap)
    region_mappable.set_array(region_values)
    fig.colorbar(region_mappable, ax=ax3, fraction=0.040, pad=0.06, label="region id (shuffled)")

    fig.suptitle(
        "Topology density -> probability -> random seed sampling -> Voronoi partition",
        fontsize=16,
    )
    fig.text(
        0.5,
        0.01,
        f"density grid={density.shape[0]}x{density.shape[1]}x{density.shape[2]} | Voronoi boundary mesh",
        ha="center",
        fontsize=10,
        color="#334155",
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved overview: {output_png}")
