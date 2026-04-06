from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np


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
    z_size: int = 200,
    outer_radius: float = 72.0,
    inner_radius: float = 36.0,
    chunk_depth: int = 8,
) -> None:
    if xy_size <= 0 or z_size <= 0:
        raise ValueError("xy_size and z_size must be positive.")
    if outer_radius <= 0 or inner_radius < 0:
        raise ValueError("outer_radius must be positive and inner_radius must be non-negative.")
    if inner_radius >= outer_radius:
        raise ValueError("inner_radius must be smaller than outer_radius.")
    if outer_radius >= xy_size / 2:
        raise ValueError("outer_radius must be smaller than xy_size / 2 to fit inside the XY plane.")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a voxel annular cylinder and save it as an NPZ file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/voxel/voxel_annular_cylinder_200x200x200.npz"),
        help="Output NPZ path.",
    )
    parser.add_argument(
        "--xy-size",
        type=int,
        default=200,
        help="Grid size in x and y.",
    )
    parser.add_argument(
        "--z-size",
        type=int,
        default=200,
        help="Grid size in z.",
    )
    parser.add_argument(
        "--outer-radius",
        type=float,
        default=72.0,
        help="Outer radius of the annular cylinder in voxel units.",
    )
    parser.add_argument(
        "--inner-radius",
        type=float,
        default=36.0,
        help="Inner radius of the annular cylinder in voxel units.",
    )
    parser.add_argument(
        "--chunk-depth",
        type=int,
        default=8,
        help="How many z slices to generate per chunk.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_annular_cylinder_npz(
        output_path=args.output,
        xy_size=args.xy_size,
        z_size=args.z_size,
        outer_radius=args.outer_radius,
        inner_radius=args.inner_radius,
        chunk_depth=args.chunk_depth,
    )
