from __future__ import annotations

from pathlib import Path

import numpy as np

from helix_voronoi.voronoi import build_voronoi_cells, extract_unique_edges


def aggregate_to_blocks(density_milli: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cropped_shape = tuple((size // 3) * 3 for size in density_milli.shape)
    if any(size == 0 for size in cropped_shape):
        raise ValueError("density_milli shape must be at least 3x3x3 for block aggregation.")

    cropped = density_milli[
        : cropped_shape[0],
        : cropped_shape[1],
        : cropped_shape[2],
    ]
    density = cropped.astype(np.float32) / 1000.0
    block_shape = (cropped_shape[0] // 3, cropped_shape[1] // 3, cropped_shape[2] // 3)
    reshaped = density.reshape(
        block_shape[0],
        3,
        block_shape[1],
        3,
        block_shape[2],
        3,
    ).transpose(0, 2, 4, 1, 3, 5)
    block_density = reshaped.mean(axis=(3, 4, 5))
    block_active = reshaped.max(axis=(3, 4, 5)) > 0.0
    return block_density.astype(np.float32), block_active


def assign_template_ids(
    block_density: np.ndarray,
    block_active: np.ndarray,
) -> np.ndarray:
    template_ids = np.full(block_density.shape, -1, dtype=np.int8)
    active_density = np.clip(block_density[block_active], 0.0, 0.999999)
    template_ids[block_active] = np.floor(active_density * 10.0).astype(np.int8)
    return template_ids


def template_seed_count_map() -> np.ndarray:
    return np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 15], dtype=np.int32)


def generate_template(
    seed_count: int,
    rng_seed: int,
) -> dict[str, np.ndarray | list[tuple[np.ndarray, np.ndarray]]]:
    rng = np.random.default_rng(rng_seed)
    seeds = np.clip(rng.random((seed_count, 3)), 1e-5, 1.0 - 1e-5)
    cells, halfspace_sets = build_voronoi_cells(seeds)
    edges = extract_unique_edges(cells, halfspace_sets)
    return {"seeds": seeds.astype(np.float32), "edges": edges}


def build_backfilled_templates(
    density_milli: np.ndarray,
    seed_counts: np.ndarray | None = None,
    template_seed_base: int = 1000,
    progress: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, np.ndarray | list[tuple[np.ndarray, np.ndarray]]]],
]:
    block_density, block_active = aggregate_to_blocks(density_milli)
    template_ids = assign_template_ids(block_density, block_active)
    resolved_seed_counts = template_seed_count_map() if seed_counts is None else seed_counts.astype(np.int32)

    templates = []
    for template_index, seed_count in enumerate(resolved_seed_counts):
        if progress:
            print(f"building template T{template_index} with {int(seed_count)} seeds...")
        templates.append(
            generate_template(seed_count=int(seed_count), rng_seed=template_seed_base + template_index)
        )

    return block_density, block_active, template_ids, resolved_seed_counts, templates


def save_backfilled_templates(
    output_npz: Path,
    block_density: np.ndarray,
    block_active: np.ndarray,
    template_ids: np.ndarray,
    seed_counts: np.ndarray,
    templates: list[dict[str, np.ndarray | list[tuple[np.ndarray, np.ndarray]]]],
    representative_seeds: np.ndarray | None = None,
    candidate_points: np.ndarray | None = None,
    target_shape: tuple[int, int, int] | None = None,
) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    edge_offsets = []
    edge_starts = []
    edge_ends = []
    count = 0
    for template in templates:
        edges = template["edges"]
        edge_offsets.append(count)
        for start, end in edges:
            edge_starts.append(start)
            edge_ends.append(end)
            count += 1
    edge_offsets.append(count)

    payload = dict(
        block_density=np.round(block_density, 4).astype(np.float32),
        block_active=block_active.astype(np.uint8),
        template_ids=template_ids.astype(np.int8),
        template_seed_counts=seed_counts.astype(np.int32),
        template_edge_offsets=np.array(edge_offsets, dtype=np.int32),
        template_edge_starts=np.asarray(edge_starts, dtype=np.float32),
        template_edge_ends=np.asarray(edge_ends, dtype=np.float32),
    )
    if representative_seeds is not None:
        payload["representative_seeds"] = representative_seeds.astype(np.float32)
        payload["num_representative_seeds"] = np.array(len(representative_seeds), dtype=np.int32)
    if candidate_points is not None:
        payload["candidate_points"] = candidate_points.astype(np.float32)
        payload["num_candidate_points"] = np.array(len(candidate_points), dtype=np.int32)
    if target_shape is not None:
        payload["target_shape"] = np.array(target_shape, dtype=np.int32)

    np.savez_compressed(output_npz, **payload)
