from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class FJWNodalDisplacementResult:
    node_ids: np.ndarray
    displacements_xyz: np.ndarray
    source_path: Path | None = None

    def __post_init__(self) -> None:
        node_ids = np.asarray(self.node_ids, dtype=np.int32).reshape(-1)
        displacements_xyz = np.asarray(self.displacements_xyz, dtype=np.float64)
        if displacements_xyz.ndim != 2 or displacements_xyz.shape[1] != 3:
            raise ValueError("displacements_xyz must have shape (N, 3).")
        if node_ids.shape[0] != displacements_xyz.shape[0]:
            raise ValueError("node_ids and displacements_xyz must have the same length.")
        if node_ids.size == 0:
            raise ValueError("Node displacement result is empty.")
        if np.any(node_ids <= 0):
            raise ValueError("node_ids must be positive 1-based integers.")
        if np.unique(node_ids).size != node_ids.size:
            raise ValueError("node_ids contains duplicates.")
        sort_order = np.argsort(node_ids, kind="stable")
        object.__setattr__(self, "node_ids", node_ids[sort_order])
        object.__setattr__(self, "displacements_xyz", displacements_xyz[sort_order])

    @property
    def node_count(self) -> int:
        return int(self.node_ids.shape[0])

    @property
    def max_node_id(self) -> int:
        return int(self.node_ids[-1])

    def to_abaqus_table(self) -> np.ndarray:
        return np.column_stack((self.node_ids, self.displacements_xyz))

    def to_dense_matrix(
        self,
        *,
        expected_node_count: int | None = None,
        fill_value: float = 0.0,
        strict: bool = False,
    ) -> np.ndarray:
        if expected_node_count is None:
            expected_node_count = self.max_node_id
        expected_node_count = int(expected_node_count)
        if expected_node_count <= 0:
            raise ValueError("expected_node_count must be positive.")
        if self.max_node_id > expected_node_count:
            raise ValueError(
                f"Observed node id {self.max_node_id} exceeds expected_node_count {expected_node_count}."
            )

        dense = np.full((expected_node_count, 3), float(fill_value), dtype=np.float64)
        dense[self.node_ids - 1] = self.displacements_xyz
        if strict and self.node_ids.shape[0] != expected_node_count:
            missing_count = expected_node_count - self.node_ids.shape[0]
            raise ValueError(f"Missing {missing_count} node displacement rows in Abaqus export.")
        return dense


def parse_abaqus_u1_rows(rows: np.ndarray, *, source_path: Path | None = None) -> FJWNodalDisplacementResult:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] != 4:
        raise ValueError("Abaqus U1 rows must have shape (N, 4): node_id, u1, u2, u3.")

    raw_node_ids = rows[:, 0]
    rounded_node_ids = np.rint(raw_node_ids)
    if not np.allclose(raw_node_ids, rounded_node_ids):
        raise ValueError("Abaqus U1 rows contain non-integer node ids.")

    return FJWNodalDisplacementResult(
        node_ids=rounded_node_ids.astype(np.int32),
        displacements_xyz=rows[:, 1:4],
        source_path=source_path,
    )


def load_abaqus_u1_result(path: str | Path) -> FJWNodalDisplacementResult:
    path = Path(path)
    rows = np.loadtxt(path, dtype=np.float64, ndmin=2)
    return parse_abaqus_u1_rows(rows, source_path=path)


def load_abaqus_u1_dense_matrix(
    path: str | Path,
    *,
    expected_node_count: int | None = None,
    fill_value: float = 0.0,
    strict: bool = False,
) -> np.ndarray:
    return load_abaqus_u1_result(path).to_dense_matrix(
        expected_node_count=expected_node_count,
        fill_value=fill_value,
        strict=strict,
    )


__all__ = [
    "FJWNodalDisplacementResult",
    "load_abaqus_u1_dense_matrix",
    "load_abaqus_u1_result",
    "parse_abaqus_u1_rows",
]
