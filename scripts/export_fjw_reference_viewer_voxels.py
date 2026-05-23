from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "datasets" / "topopt" / "fjw_reference_fem_voxels.npz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "viewer" / "public" / "data" / "fjw_reference_voxels"


def _surface_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    prev_x = np.zeros_like(mask)
    prev_x[1:, :, :] = mask[:-1, :, :]
    next_x = np.zeros_like(mask)
    next_x[:-1, :, :] = mask[1:, :, :]
    prev_y = np.zeros_like(mask)
    prev_y[:, 1:, :] = mask[:, :-1, :]
    next_y = np.zeros_like(mask)
    next_y[:, :-1, :] = mask[:, 1:, :]
    prev_z = np.zeros_like(mask)
    prev_z[:, :, 1:] = mask[:, :, :-1]
    next_z = np.zeros_like(mask)
    next_z[:, :, :-1] = mask[:, :, 1:]

    interior = prev_x & next_x & prev_y & next_y & prev_z & next_z
    return mask & (~interior)


def _write_uint16_xyz(path: Path, coords: np.ndarray) -> None:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3).")
    path.write_bytes(np.asarray(coords, dtype=np.uint16).reshape(-1).tobytes())


def export_fjw_reference_viewer_voxels(source_npz: Path, output_dir: Path) -> Path:
    with np.load(source_npz) as data:
        material_id = np.asarray(data["material_id"], dtype=np.int8)
        origin_mm = (np.asarray(data["origin_m"], dtype=np.float32) * 1e3).tolist()
        voxel_size_mm = (np.asarray(data["voxel_size_xyz_m"], dtype=np.float32) * 1e3).tolist()
        grid_shape = np.asarray(data["grid_shape_xyz"], dtype=np.int32).tolist()
        all_voxel_count = int(np.asarray(data["voxels"], dtype=np.uint8).sum())

    output_dir.mkdir(parents=True, exist_ok=True)
    parts = [
        ("cortical_bone", material_id == 0, [0.96, 0.78, 0.39], 1.0, 0.86, 0.0),
        ("trabecular_bone", material_id == 1, [0.63, 0.43, 0.24], 1.0, 0.94, 0.0),
        ("cage", material_id == 2, [0.04, 0.56, 0.63], 1.0, 0.42, 0.08),
    ]

    manifest_parts: list[dict[str, object]] = []
    visible_voxel_count = 0
    for name, mask, color, opacity, roughness, metalness in parts:
        coords = np.argwhere(_surface_mask(mask))
        visible_voxel_count += int(coords.shape[0])
        filename = f"{name}.bin"
        _write_uint16_xyz(output_dir / filename, coords)
        manifest_parts.append(
            {
                "name": name,
                "url": f"/data/fjw_reference_voxels/{filename}",
                "count": int(coords.shape[0]),
                "color": color,
                "opacity": opacity,
                "roughness": roughness,
                "metalness": metalness,
            }
        )

    manifest = {
        "title": "FJW reference structure",
        "representation": "instanced_surface_voxels",
        "origin_mm": origin_mm,
        "voxel_size_mm": voxel_size_mm,
        "grid_shape": grid_shape,
        "all_voxel_count": all_voxel_count,
        "visible_voxel_count": visible_voxel_count,
        "parts": manifest_parts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export FJW reference surface voxels for viewer instancing.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Input FJW voxel NPZ path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output viewer data directory.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(
            f"Missing source NPZ: {source}. Run scripts/export_fjw_reference_voxels.py first."
        )
    output = export_fjw_reference_viewer_voxels(source, Path(args.output_dir))
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
