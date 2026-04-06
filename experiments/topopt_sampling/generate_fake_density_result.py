from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np


def load_voxels(npz_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(npz_path) as data:
        voxels = data["voxels"]
        metadata = {key: data[key] for key in data.files if key != "voxels"}
    return voxels, metadata


def build_density_chunk(
    voxels_chunk: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    center_xy: tuple[float, float],
    outer_radius: float,
    inner_radius: float,
) -> np.ndarray:
    cx, cy = center_xy
    x, y, z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

    radial = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    annulus_thickness = max(outer_radius - inner_radius, 1e-6)
    radial_norm = np.clip((radial - inner_radius) / annulus_thickness, 0.0, 1.0)
    z_norm = z / max(z_coords[-1], 1.0)
    theta = np.arctan2(y - cy, x - cx)

    field = (
        0.52
        + 0.23 * np.sin(3.0 * theta + 5.5 * z_norm)
        + 0.18 * np.cos(2.0 * np.pi * radial_norm)
        + 0.11 * np.sin(7.0 * radial_norm - 4.0 * z_norm)
        + 0.08 * np.cos(9.0 * theta * radial_norm + 2.5 * z_norm)
    )

    field = (field - field.min()) / max(field.max() - field.min(), 1e-6)
    density = 0.001 + 0.999 * field
    density = np.round(density, 3)
    density = np.where(voxels_chunk > 0, density, 0.0)
    return density.astype(np.float32)


def generate_fake_density_result(
    source_npz: Path,
    output_npz: Path,
    chunk_depth: int = 8,
) -> None:
    voxels, metadata = load_voxels(source_npz)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a fake FEA-like density result NPZ.")
    parser.add_argument(
        "source_npz",
        type=Path,
        nargs="?",
        default=Path("datasets/voxel/voxel_annular_cylinder_200x200x200.npz"),
        help="Source voxel geometry NPZ.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/topopt/fake_density_annular_cylinder_200x200x200.npz"),
        help="Output fake density result NPZ.",
    )
    parser.add_argument(
        "--chunk-depth",
        type=int,
        default=8,
        help="How many z slices to process per chunk.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_fake_density_result(
        source_npz=args.source_npz,
        output_npz=args.output,
        chunk_depth=args.chunk_depth,
    )
