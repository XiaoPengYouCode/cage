from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import rerun as rr

from cage.helix_stl import export_helix_edges_to_stl
from cage.models import EdgeSegment


@dataclass(frozen=True)
class RerunHelixSummary:
    triangle_count: int
    edge_count: int
    stl_path: Path | None
    rrd_path: Path | None


def log_stl_asset(
    stl_path: Path,
    app_id: str,
    spawn: bool = True,
    save_path: Path | None = None,
) -> None:
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file does not exist: {stl_path}")
    if stl_path.suffix.lower() != ".stl":
        raise ValueError(f"Expected an .stl file, got: {stl_path.name}")

    rr.init(app_id, spawn=spawn)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(save_path)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/stl",
        rr.Asset3D(
            contents=stl_path.read_bytes(),
            media_type="model/stl",
        ),
        static=True,
    )
    rr.disconnect()


def log_helix_edges_to_rerun(
    edges: list[EdgeSegment],
    radius: float,
    app_id: str,
    spawn: bool = True,
    save_path: Path | None = None,
    stl_path: Path | None = None,
) -> RerunHelixSummary:
    temp_stl_path: Path | None = None
    final_stl_path = stl_path

    if final_stl_path is None:
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as handle:
            temp_stl_path = Path(handle.name)
        final_stl_path = temp_stl_path

    try:
        export_summary = export_helix_edges_to_stl(
            edges,
            radius=radius,
            output_path=final_stl_path,
        )
        log_stl_asset(
            stl_path=final_stl_path,
            app_id=app_id,
            spawn=spawn,
            save_path=save_path,
        )
        return RerunHelixSummary(
            triangle_count=export_summary.triangle_count,
            edge_count=export_summary.edge_count,
            stl_path=None if temp_stl_path is not None else final_stl_path,
            rrd_path=save_path,
        )
    finally:
        if temp_stl_path is not None:
            temp_stl_path.unlink(missing_ok=True)
