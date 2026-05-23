from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fjw_workflow_models import FJWReferenceMeshContext, FJWWorkflowState
from .fjw_workflow_results import FJWNodalDisplacementResult, load_abaqus_u1_result


@dataclass(frozen=True, slots=True)
class FJWElementDisplacementVectorCache:
    vectors_2d: np.ndarray
    element_ids: np.ndarray
    cache_name: str | None = None
    source_result_path: Path | None = None

    def __post_init__(self) -> None:
        vectors_2d = np.asarray(self.vectors_2d, dtype=np.float64)
        element_ids = np.asarray(self.element_ids, dtype=np.int32).reshape(-1)
        if vectors_2d.ndim != 2 or vectors_2d.shape[1] != 24:
            raise ValueError("vectors_2d must have shape (N, 24).")
        if vectors_2d.shape[0] != element_ids.shape[0]:
            raise ValueError("vectors_2d and element_ids must have the same length.")
        if element_ids.size == 0:
            raise ValueError("Element displacement cache is empty.")
        if np.any(element_ids <= 0):
            raise ValueError("element_ids must be positive 1-based integers.")
        if np.unique(element_ids).size != element_ids.size:
            raise ValueError("element_ids contains duplicates.")
        sort_order = np.argsort(element_ids, kind="stable")
        object.__setattr__(self, "vectors_2d", vectors_2d[sort_order])
        object.__setattr__(self, "element_ids", element_ids[sort_order])

    @property
    def vectors_flat(self) -> np.ndarray:
        return self.vectors_2d.reshape(-1)

    @property
    def num_elements(self) -> int:
        return int(self.element_ids.shape[0])

    def slice_by_element_ids(self, element_ids: np.ndarray | list[int] | tuple[int, ...]) -> np.ndarray:
        element_ids = np.asarray(element_ids, dtype=np.int32).reshape(-1)
        if element_ids.size == 0:
            return np.zeros((0, 24), dtype=np.float64)
        zero_based = element_ids - 1
        if np.any(zero_based < 0) or np.any(zero_based >= self.num_elements):
            raise ValueError("Requested element_ids fall outside the cache range.")
        if not np.array_equal(self.element_ids, np.arange(1, self.num_elements + 1, dtype=np.int32)):
            raise ValueError("slice_by_element_ids() requires a dense full-mesh cache with sequential element ids.")
        return self.vectors_2d[zero_based]

    def design_slices(self, workflow_or_mesh: FJWWorkflowState | FJWReferenceMeshContext) -> np.ndarray:
        return design_element_displacement_slices(self, workflow_or_mesh)

    def objective_slices(self, workflow_or_mesh: FJWWorkflowState | FJWReferenceMeshContext) -> np.ndarray:
        return objective_element_displacement_slices(self, workflow_or_mesh)


def _coerce_mesh_context(workflow_or_mesh: FJWWorkflowState | FJWReferenceMeshContext) -> FJWReferenceMeshContext:
    if isinstance(workflow_or_mesh, FJWWorkflowState):
        return workflow_or_mesh.mesh
    return workflow_or_mesh


def _coerce_dense_node_displacements(
    node_displacements: FJWNodalDisplacementResult | np.ndarray,
    *,
    expected_node_count: int,
    strict: bool = False,
) -> np.ndarray:
    if isinstance(node_displacements, FJWNodalDisplacementResult):
        return node_displacements.to_dense_matrix(
            expected_node_count=expected_node_count,
            strict=strict,
        )

    node_displacements = np.asarray(node_displacements, dtype=np.float64)
    if node_displacements.ndim != 2 or node_displacements.shape[1] != 3:
        raise ValueError("node_displacements must have shape (N, 3).")
    if node_displacements.shape[0] != expected_node_count:
        raise ValueError(
            f"node_displacements row count {node_displacements.shape[0]} does not match expected_node_count "
            f"{expected_node_count}."
        )
    return node_displacements


def assemble_element_displacement_vectors(
    node_displacements: FJWNodalDisplacementResult | np.ndarray,
    element_nodes: np.ndarray,
    *,
    expected_node_count: int | None = None,
    strict: bool = False,
) -> np.ndarray:
    element_nodes = np.asarray(element_nodes, dtype=np.int32)
    if element_nodes.ndim != 2 or element_nodes.shape[1] != 8:
        raise ValueError("element_nodes must have shape (N, 8).")
    if np.any(element_nodes <= 0):
        raise ValueError("element_nodes must contain positive 1-based node ids.")

    inferred_node_count = int(element_nodes.max())
    dense_node_displacements = _coerce_dense_node_displacements(
        node_displacements,
        expected_node_count=max(expected_node_count or 0, inferred_node_count),
        strict=strict,
    )
    if inferred_node_count > dense_node_displacements.shape[0]:
        raise ValueError("element_nodes references node ids outside node_displacements.")

    per_element = dense_node_displacements[element_nodes - 1]
    return per_element.reshape(per_element.shape[0], 24)


def build_element_displacement_cache(
    node_displacements: FJWNodalDisplacementResult | np.ndarray,
    workflow_or_mesh: FJWWorkflowState | FJWReferenceMeshContext,
    *,
    cache_name: str | None = None,
    source_result_path: str | Path | None = None,
    strict: bool = False,
) -> FJWElementDisplacementVectorCache:
    mesh = _coerce_mesh_context(workflow_or_mesh)
    vectors_2d = assemble_element_displacement_vectors(
        node_displacements,
        mesh.element_nodes,
        expected_node_count=mesh.node_coordinates.shape[0],
        strict=strict,
    )

    if source_result_path is None and isinstance(node_displacements, FJWNodalDisplacementResult):
        source_result_path = node_displacements.source_path

    return FJWElementDisplacementVectorCache(
        vectors_2d=vectors_2d,
        element_ids=np.arange(1, vectors_2d.shape[0] + 1, dtype=np.int32),
        cache_name=cache_name,
        source_result_path=Path(source_result_path) if source_result_path is not None else None,
    )


def build_element_displacement_cache_from_u1(
    u1_path: str | Path,
    workflow_or_mesh: FJWWorkflowState | FJWReferenceMeshContext,
    *,
    cache_name: str | None = None,
    strict: bool = False,
) -> FJWElementDisplacementVectorCache:
    result = load_abaqus_u1_result(u1_path)
    return build_element_displacement_cache(
        result,
        workflow_or_mesh,
        cache_name=cache_name,
        source_result_path=result.source_path,
        strict=strict,
    )


def save_element_displacement_cache(
    path: str | Path,
    cache: FJWElementDisplacementVectorCache,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vectors_2d=cache.vectors_2d,
        element_ids=cache.element_ids,
        cache_name=np.array(cache.cache_name or ""),
        source_result_path=np.array("" if cache.source_result_path is None else str(cache.source_result_path)),
    )
    return path


def load_element_displacement_cache(path: str | Path) -> FJWElementDisplacementVectorCache:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        cache_name_raw: Any = data["cache_name"]
        source_result_path_raw: Any = data["source_result_path"]
        cache_name = str(cache_name_raw.tolist()) or None
        source_result_path_value = str(source_result_path_raw.tolist())
        source_result_path = Path(source_result_path_value) if source_result_path_value else None
        return FJWElementDisplacementVectorCache(
            vectors_2d=np.asarray(data["vectors_2d"], dtype=np.float64),
            element_ids=np.asarray(data["element_ids"], dtype=np.int32),
            cache_name=cache_name,
            source_result_path=source_result_path,
        )


def _coerce_cache(
    cache_or_vectors: FJWElementDisplacementVectorCache | np.ndarray,
) -> FJWElementDisplacementVectorCache:
    if isinstance(cache_or_vectors, FJWElementDisplacementVectorCache):
        return cache_or_vectors

    vectors_2d = np.asarray(cache_or_vectors, dtype=np.float64)
    if vectors_2d.ndim == 1:
        if vectors_2d.size % 24 != 0:
            raise ValueError("Flat element displacement vectors must have a length divisible by 24.")
        vectors_2d = vectors_2d.reshape(-1, 24)
    if vectors_2d.ndim != 2 or vectors_2d.shape[1] != 24:
        raise ValueError("cache_or_vectors must be a cache object or an array with shape (N, 24).")
    return FJWElementDisplacementVectorCache(
        vectors_2d=vectors_2d,
        element_ids=np.arange(1, vectors_2d.shape[0] + 1, dtype=np.int32),
    )


def design_element_displacement_slices(
    cache_or_vectors: FJWElementDisplacementVectorCache | np.ndarray,
    workflow_or_mesh: FJWWorkflowState | FJWReferenceMeshContext,
) -> np.ndarray:
    cache = _coerce_cache(cache_or_vectors)
    mesh = _coerce_mesh_context(workflow_or_mesh)
    return cache.slice_by_element_ids(mesh.design_elements)


def objective_element_displacement_slices(
    cache_or_vectors: FJWElementDisplacementVectorCache | np.ndarray,
    workflow_or_mesh: FJWWorkflowState | FJWReferenceMeshContext,
) -> np.ndarray:
    cache = _coerce_cache(cache_or_vectors)
    mesh = _coerce_mesh_context(workflow_or_mesh)
    return cache.slice_by_element_ids(mesh.objective_elements)


def load_design_element_displacements_from_u1(
    u1_path: str | Path,
    workflow_or_mesh: FJWWorkflowState | FJWReferenceMeshContext,
    *,
    strict: bool = False,
) -> np.ndarray:
    cache = build_element_displacement_cache_from_u1(u1_path, workflow_or_mesh, strict=strict)
    return design_element_displacement_slices(cache, workflow_or_mesh)


def load_objective_element_displacements_from_u1(
    u1_path: str | Path,
    workflow_or_mesh: FJWWorkflowState | FJWReferenceMeshContext,
    *,
    strict: bool = False,
) -> np.ndarray:
    cache = build_element_displacement_cache_from_u1(u1_path, workflow_or_mesh, strict=strict)
    return objective_element_displacement_slices(cache, workflow_or_mesh)


__all__ = [
    "FJWElementDisplacementVectorCache",
    "assemble_element_displacement_vectors",
    "build_element_displacement_cache",
    "build_element_displacement_cache_from_u1",
    "design_element_displacement_slices",
    "load_design_element_displacements_from_u1",
    "load_element_displacement_cache",
    "load_objective_element_displacements_from_u1",
    "objective_element_displacement_slices",
    "save_element_displacement_cache",
]
