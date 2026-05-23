from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FJW_ABAQUS_INPUTS_PATH = PROJECT_ROOT / "datasets" / "fjw_abaqus_inputs.json"
ABAQUS_NEWLINE = "\r\n"
ELSET_VALUES_PER_LINE = 10


@dataclass(frozen=True)
class ForwardLoad:
    target: str
    dof: int
    magnitude: float
    op: str = "NEW"


@dataclass(frozen=True)
class ForwardLoadCaseTemplate:
    name: str
    template_path: Path
    loads: tuple[ForwardLoad, ...]


@dataclass(frozen=True)
class AdjointInjectionRule:
    target_format: str
    components: tuple[int, ...]
    magnitude_source: str
    description: str


@dataclass(frozen=True)
class AdjointTemplate:
    prefix_template_path: Path
    suffix_template_path: Path
    injection_rule: AdjointInjectionRule


@dataclass(frozen=True)
class FJWWorkflowTemplateCatalog:
    spec_path: Path
    forward_load_cases: Mapping[str, ForwardLoadCaseTemplate]
    adjoint_template: AdjointTemplate
    simulation_constants: Mapping[str, object]
    section_assignments: tuple[Mapping[str, object], ...]
    discretization_rules: Mapping[str, object]


def _resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_fjw_workflow_template_catalog(
    spec_path: Path = DEFAULT_FJW_ABAQUS_INPUTS_PATH,
) -> FJWWorkflowTemplateCatalog:
    spec_path = Path(spec_path)
    raw_data = json.loads(spec_path.read_text(encoding="utf-8"))

    forward_load_cases: dict[str, ForwardLoadCaseTemplate] = {}
    for name, payload in raw_data["forward_load_cases"].items():
        forward_load_cases[name] = ForwardLoadCaseTemplate(
            name=name,
            template_path=_resolve_project_path(payload["template"]),
            loads=tuple(
                ForwardLoad(
                    target=load["target"],
                    dof=int(load["dof"]),
                    magnitude=float(load["magnitude"]),
                    op=load.get("op", "NEW"),
                )
                for load in payload["loads"]
            ),
        )

    adjoint_payload = raw_data["adjoint_load_template"]
    injection_payload = adjoint_payload["injection_rule"]
    adjoint_template = AdjointTemplate(
        prefix_template_path=_resolve_project_path(adjoint_payload["prefix_template"]),
        suffix_template_path=_resolve_project_path(adjoint_payload["suffix_template"]),
        injection_rule=AdjointInjectionRule(
            target_format=injection_payload["target_format"],
            components=tuple(int(component) for component in injection_payload["components"]),
            magnitude_source=injection_payload["magnitude_source"],
            description=injection_payload["description"],
        ),
    )

    return FJWWorkflowTemplateCatalog(
        spec_path=spec_path,
        forward_load_cases=forward_load_cases,
        adjoint_template=adjoint_template,
        simulation_constants=raw_data["simulation_constants"],
        section_assignments=tuple(raw_data["section_assignments"]),
        discretization_rules=raw_data["discretization_rules"],
    )


def read_text(path: Path) -> str:
    return Path(path).read_bytes().decode("utf-8")


def format_abaqus_number(value: float | int) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.15g}"


def coerce_int_sequence(values: Sequence[int | float] | np.ndarray, *, label: str) -> list[int]:
    coerced: list[int] = []
    for value in values:
        numeric = float(value)
        if not numeric.is_integer():
            raise ValueError(f"{label} contains non-integer values.")
        coerced.append(int(numeric))
    return coerced


def render_comma_block(
    values: Iterable[int | float],
    *,
    items_per_line: int = ELSET_VALUES_PER_LINE,
    newline: str = ABAQUS_NEWLINE,
) -> str:
    numbers = coerce_int_sequence(list(values), label="comma_block")
    if not numbers:
        return ""
    lines: list[str] = []
    for start in range(0, len(numbers), items_per_line):
        chunk = numbers[start : start + items_per_line]
        prefix = "".join(f"{value:7d},\t" for value in chunk[:-1])
        suffix = f"{chunk[-1]:7d},"
        lines.append(prefix + suffix)
    return newline.join(lines)


def format_elset_block(
    elset_name: str,
    element_ids: Sequence[int | float] | np.ndarray,
    *,
    newline: str = ABAQUS_NEWLINE,
) -> str:
    body = render_comma_block(element_ids, newline=newline)
    if not body:
        return f"*Elset,elset={elset_name}{newline}"
    return f"*Elset,elset={elset_name}{newline}{body}{newline}"


def render_elset_block(name: str, values: Iterable[int | float]) -> str:
    return format_elset_block(name, list(values), newline="\n")


def format_generate_set_block(
    keyword: str,
    set_name: str,
    start: int,
    stop: int,
    step: int = 1,
    *,
    newline: str = ABAQUS_NEWLINE,
) -> str:
    lines = [
        f"*{keyword}, {keyword.lower()}={set_name}, generate",
        f"{start},{stop},{step}",
    ]
    return newline.join(lines) + newline


def render_nset_generate(name: str, start: int, stop: int, step: int = 1) -> str:
    return format_generate_set_block("Nset", name, start, stop, step, newline="\n")


def format_node_block(
    node_coordinates: Sequence[Sequence[int | float]] | np.ndarray,
    *,
    voxel_size_mm: float,
    newline: str = ABAQUS_NEWLINE,
) -> str:
    if voxel_size_mm <= 0.0:
        raise ValueError("voxel_size_mm must be positive.")

    lines = ["*Node"]
    for index, coordinate in enumerate(node_coordinates, start=1):
        if len(coordinate) != 3:
            raise ValueError("Each node coordinate must contain exactly 3 values.")
        scaled = [(float(value) - 1.0) * voxel_size_mm for value in coordinate]
        lines.append(
            ",".join(
                [
                    str(index),
                    format_abaqus_number(scaled[0]),
                    format_abaqus_number(scaled[1]),
                    format_abaqus_number(scaled[2]),
                ]
            )
        )
    return newline.join(lines) + newline


def render_node_block(node_coordinates_mm: np.ndarray) -> str:
    lines = ["*Node"]
    for index, xyz in enumerate(node_coordinates_mm, start=1):
        lines.append(
            ",".join(
                [
                    str(index),
                    format_abaqus_number(float(xyz[0])),
                    format_abaqus_number(float(xyz[1])),
                    format_abaqus_number(float(xyz[2])),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def format_element_block(
    element_nodes: Sequence[Sequence[int | float]] | np.ndarray,
    *,
    newline: str = ABAQUS_NEWLINE,
) -> str:
    lines = ["*Element, type=C3D8R"]
    for index, connectivity in enumerate(element_nodes, start=1):
        if len(connectivity) != 8:
            raise ValueError("Each hexahedral element must contain exactly 8 node ids.")
        ids = coerce_int_sequence(connectivity, label=f"element_nodes[{index - 1}]")
        lines.append(",".join([str(index), *(str(node_id) for node_id in ids)]))
    return newline.join(lines) + newline


def render_element_block(element_nodes: np.ndarray) -> str:
    return format_element_block(element_nodes, newline="\n")


def render_cload_entries(node_ids: Iterable[int], fv_vector: np.ndarray) -> str:
    lines: list[str] = []
    for node_index, node_id in enumerate(node_ids, start=1):
        base = (int(node_id) - 1) * 3
        lines.extend(
            [
                f"** Name: CFORCE-{node_index}Type: Concentrated force",
                "*Cload, op=NEW",
                f"VERT-1.{int(node_id)}, 1, {format_abaqus_number(float(fv_vector[base]))}",
                f"VERT-1.{int(node_id)}, 2, {format_abaqus_number(float(fv_vector[base + 1]))}",
                f"VERT-1.{int(node_id)}, 3, {format_abaqus_number(float(fv_vector[base + 2]))}",
            ]
        )
    return "\n".join(lines) + ("\n" if lines else "")


def bucket_element_ids(
    element_ids: Sequence[int | float] | np.ndarray,
    values: Sequence[int | float] | np.ndarray,
    *,
    multiplier: int,
    label: str,
    denominator: float = 1.0,
) -> dict[int, list[int]]:
    ids = coerce_int_sequence(element_ids, label=f"{label}_element_ids")
    numeric_values = [float(value) for value in values]
    if len(ids) != len(numeric_values):
        raise ValueError(f"{label} requires element_ids and values to have the same length.")
    if multiplier < 0:
        raise ValueError("multiplier must be non-negative.")
    if denominator <= 0.0:
        raise ValueError("denominator must be positive.")

    buckets = {index: [] for index in range(multiplier + 1)}
    for element_id, value in zip(ids, numeric_values, strict=True):
        bucket_index = int(round(value / denominator * multiplier))
        if bucket_index < 0 or bucket_index > multiplier:
            raise ValueError(
                f"{label} produced out-of-range bucket {bucket_index} for element {element_id}."
            )
        buckets[bucket_index].append(element_id)
    return buckets


def format_bucketed_elsets(
    prefix: str,
    buckets: Mapping[int, Sequence[int]],
    *,
    start: int,
    stop: int,
    newline: str = ABAQUS_NEWLINE,
) -> str:
    return "".join(
        format_elset_block(f"{prefix}{index}", buckets.get(index, ()), newline=newline)
        for index in range(start, stop + 1)
    )


__all__ = [
    "ABAQUS_NEWLINE",
    "DEFAULT_FJW_ABAQUS_INPUTS_PATH",
    "ELSET_VALUES_PER_LINE",
    "AdjointInjectionRule",
    "AdjointTemplate",
    "FJWWorkflowTemplateCatalog",
    "ForwardLoad",
    "ForwardLoadCaseTemplate",
    "bucket_element_ids",
    "coerce_int_sequence",
    "format_abaqus_number",
    "format_bucketed_elsets",
    "format_elset_block",
    "format_element_block",
    "format_generate_set_block",
    "format_node_block",
    "load_fjw_workflow_template_catalog",
    "read_text",
    "render_cload_entries",
    "render_comma_block",
    "render_element_block",
    "render_elset_block",
    "render_node_block",
    "render_nset_generate",
]
