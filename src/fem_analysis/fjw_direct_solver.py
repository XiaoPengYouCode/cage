from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Callable, Literal

import numpy as np

# SfePy prints optional JAX import notices to stdout during import.
with redirect_stdout(StringIO()):
    from sfepy.base.base import output as sfepy_output
    from sfepy.discrete import Equation, Equations, FieldVariable, Integral, Material, Problem
    from sfepy.discrete.conditions import Conditions, EssentialBC, LinearCombinationBC
    from sfepy.discrete.fem import FEDomain, Field, Mesh
    from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
    from sfepy.solvers.ls import PETScKrylovSolver, ScipyDirect, ScipyIterative
    from sfepy.solvers.nls import Newton
    from sfepy.terms import Term

from .fjw_solver_config import petsc_mumps_options, scipy_iterative_options
from .fjw_workflow_loaders import compute_modulus_buckets, load_fjw_workflow_state
from .fjw_workflow_models import FJWLoadCase, FJWWorkflowState


LinearSolverKind = Literal["scipy_direct", "scipy_iterative", "petsc_mumps"]


@dataclass(frozen=True, slots=True)
class FJWDirectSolverConfig:
    linear_solver_kind: LinearSolverKind = "scipy_iterative"
    linear_solver_options: dict[str, object] = field(default_factory=dict)
    nonlinear_solver_options: dict[str, object] = field(
        default_factory=lambda: {"i_max": 1, "eps_a": 1e-10, "eps_r": 1.0}
    )
    store_nodal_displacements: bool = True
    store_rigid_control: bool = True


@dataclass(frozen=True, slots=True)
class FJWDirectMaterialGroup:
    group_id: int
    region_name: str
    material_name: str
    youngs_modulus: float
    poisson_ratio: float
    element_ids: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "element_ids", np.asarray(self.element_ids, dtype=np.int32).reshape(-1))

    @property
    def element_count(self) -> int:
        return int(self.element_ids.size)


@dataclass(frozen=True, slots=True)
class FJWDirectProblemSetup:
    workflow_state: FJWWorkflowState
    load_case: FJWLoadCase
    mesh_coordinates_mm: np.ndarray
    mesh_connectivity: np.ndarray
    element_group_ids: np.ndarray
    material_groups: tuple[FJWDirectMaterialGroup, ...]
    top_vertex_ids: np.ndarray
    bottom_vertex_ids: np.ndarray
    top_rp_vertex_id: int
    bottom_rp_vertex_id: int
    load_vector: np.ndarray
    voxel_size_mm: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mesh_coordinates_mm", np.asarray(self.mesh_coordinates_mm, dtype=np.float64))
        object.__setattr__(self, "mesh_connectivity", np.asarray(self.mesh_connectivity, dtype=np.int32))
        object.__setattr__(self, "element_group_ids", np.asarray(self.element_group_ids, dtype=np.int32).reshape(-1))
        object.__setattr__(self, "top_vertex_ids", np.asarray(self.top_vertex_ids, dtype=np.int32).reshape(-1))
        object.__setattr__(self, "bottom_vertex_ids", np.asarray(self.bottom_vertex_ids, dtype=np.int32).reshape(-1))
        object.__setattr__(self, "load_vector", np.asarray(self.load_vector, dtype=np.float64).reshape(1, 6))


@dataclass(frozen=True, slots=True)
class FJWDirectSolveResult:
    setup: FJWDirectProblemSetup
    nodal_displacements: np.ndarray | None
    rigid_control: np.ndarray | None
    max_displacement_mm: float
    top_rp_displacement: np.ndarray
    top_rp_rotation: np.ndarray


@contextmanager
def quiet_sfepy_output() -> Any:
    sfepy_output.set_output(quiet=True)
    try:
        yield
    finally:
        sfepy_output.set_output(quiet=False)


def _voxel_size_mm(workflow_state: FJWWorkflowState) -> float:
    return float(np.cbrt(workflow_state.material_constants.voxel_volume))


def _mesh_coordinates_mm(workflow_state: FJWWorkflowState) -> np.ndarray:
    return (np.asarray(workflow_state.mesh.node_coordinates, dtype=np.float64) - 1.0) * _voxel_size_mm(workflow_state)


def _find_load_case(workflow_state: FJWWorkflowState, load_case_name: str) -> FJWLoadCase:
    for load_case in workflow_state.load_cases:
        if load_case.name == load_case_name:
            return load_case
    available = ", ".join(load_case.name for load_case in workflow_state.load_cases)
    raise KeyError(f"Unknown load case {load_case_name!r}. Available: {available}.")


def _build_load_vector(load_case: FJWLoadCase) -> np.ndarray:
    load_vector = np.zeros(6, dtype=np.float64)
    for load in load_case.loads:
        dof_index = int(load.dof) - 1
        if dof_index < 0 or dof_index >= 6:
            raise ValueError(f"Unsupported FJW rigid-control dof: {load.dof}.")
        load_vector[dof_index] += float(load.magnitude)
    return load_vector.reshape(1, 6)


def _reference_point_targets_mm(workflow_state: FJWWorkflowState) -> tuple[np.ndarray, np.ndarray]:
    reference_nodes = workflow_state.structured_inputs["assembly_controls"]["reference_nodes"]
    bottom = np.asarray(reference_nodes[0]["coordinates"], dtype=np.float64)
    top = np.asarray(reference_nodes[1]["coordinates"], dtype=np.float64)
    return bottom, top


def _closest_vertex_id(vertex_ids_1_based: np.ndarray, coordinates_mm: np.ndarray, target_mm: np.ndarray) -> int:
    zero_based = np.asarray(vertex_ids_1_based, dtype=np.int32).reshape(-1) - 1
    points = coordinates_mm[zero_based]
    return int(zero_based[int(np.argmin(np.linalg.norm(points - target_mm[None, :], axis=1)))])


def _coerce_dense_nodal_point_loads(
    nodal_point_loads: np.ndarray,
    *,
    expected_node_count: int,
) -> np.ndarray:
    dense = np.asarray(nodal_point_loads, dtype=np.float64)
    if dense.ndim == 1:
        if dense.size != expected_node_count * 3:
            raise ValueError(
                "Flattened nodal_point_loads size must equal expected_node_count * 3: "
                f"{dense.size} != {expected_node_count * 3}."
            )
        dense = dense.reshape(expected_node_count, 3)
    if dense.ndim != 2 or dense.shape != (expected_node_count, 3):
        raise ValueError(
            "nodal_point_loads must have shape (expected_node_count, 3) "
            f"or flattened length {expected_node_count * 3}."
        )
    return dense


def _build_material_groups(
    workflow_state: FJWWorkflowState,
    *,
    design_cage: np.ndarray,
    obj_bo: np.ndarray,
) -> tuple[tuple[FJWDirectMaterialGroup, ...], np.ndarray]:
    mesh = workflow_state.mesh
    constants = workflow_state.material_constants
    buckets = compute_modulus_buckets(
        design_cage,
        obj_bo,
        constants,
        cage_material_buckets=workflow_state.cage_material_buckets,
        bone_material_buckets=workflow_state.bone_material_buckets,
    )

    num_elements = int(mesh.element_nodes.shape[0])
    element_group_ids = np.full(num_elements, -1, dtype=np.int32)
    groups: list[FJWDirectMaterialGroup] = []
    next_group_id = 0

    def add_group(
        *,
        material_name: str,
        youngs_modulus: float,
        poisson_ratio: float,
        element_ids_1_based: np.ndarray,
    ) -> None:
        nonlocal next_group_id
        element_ids = np.asarray(element_ids_1_based, dtype=np.int32).reshape(-1)
        if element_ids.size == 0:
            return
        group_id = next_group_id
        next_group_id += 1
        element_group_ids[element_ids - 1] = group_id
        groups.append(
            FJWDirectMaterialGroup(
                group_id=group_id,
                region_name=f"group_{group_id}",
                material_name=material_name,
                youngs_modulus=float(youngs_modulus),
                poisson_ratio=float(poisson_ratio),
                element_ids=element_ids,
            )
        )

    objective_ids = np.asarray(mesh.objective_elements, dtype=np.int32).reshape(-1)
    cor_elements = np.setdiff1d(mesh.cor_elements, objective_ids, assume_unique=False)
    tra_elements = np.setdiff1d(mesh.tra_elements, objective_ids, assume_unique=False)

    add_group(
        material_name="BONE_COR",
        youngs_modulus=18000.0,
        poisson_ratio=0.3,
        element_ids_1_based=cor_elements,
    )
    add_group(
        material_name="BONE_TRA",
        youngs_modulus=1200.0,
        poisson_ratio=0.3,
        element_ids_1_based=tra_elements,
    )

    design_ids = np.asarray(mesh.design_elements, dtype=np.int32).reshape(-1)
    for bucket_index in np.unique(buckets.cage_bucket_indices):
        mask = buckets.cage_bucket_indices == int(bucket_index)
        add_group(
            material_name=f"CAGE_BUCKET_{int(bucket_index)}",
            youngs_modulus=float(buckets.cage_bucket_moduli[mask][0]),
            poisson_ratio=0.3,
            element_ids_1_based=design_ids[mask],
        )

    for bucket_index in np.unique(buckets.obj_bucket_indices):
        mask = buckets.obj_bucket_indices == int(bucket_index)
        add_group(
            material_name=f"BONE_BUCKET_{int(bucket_index)}",
            youngs_modulus=float(buckets.obj_bucket_moduli[mask][0]),
            poisson_ratio=0.3,
            element_ids_1_based=objective_ids[mask],
        )

    non_design_cage_ids = np.setdiff1d(
        mesh.cage_elements,
        design_ids,
        assume_unique=False,
    )
    if non_design_cage_ids.size:
        add_group(
            material_name="CAGE_FIXED",
            youngs_modulus=float(constants.cage_modulus_0),
            poisson_ratio=0.3,
            element_ids_1_based=non_design_cage_ids,
        )

    if np.any(element_group_ids < 0):
        raise ValueError("Not all FJW elements were assigned a direct-solver material group.")

    return tuple(groups), element_group_ids


def build_fjw_direct_problem_setup(
    workflow_state: FJWWorkflowState,
    *,
    load_case_name: str,
    design_cage: np.ndarray | None = None,
    obj_bo: np.ndarray | None = None,
) -> FJWDirectProblemSetup:
    load_case = _find_load_case(workflow_state, load_case_name)
    design_values = (
        workflow_state.initial_state.design_cage.copy()
        if design_cage is None
        else np.asarray(design_cage, dtype=np.float64).reshape(-1).copy()
    )
    obj_values = (
        workflow_state.initial_state.obj_bo.copy()
        if obj_bo is None
        else np.asarray(obj_bo, dtype=np.float64).reshape(-1).copy()
    )

    material_groups, element_group_ids = _build_material_groups(
        workflow_state,
        design_cage=design_values,
        obj_bo=obj_values,
    )
    coordinates_mm = _mesh_coordinates_mm(workflow_state)
    bottom_target_mm, top_target_mm = _reference_point_targets_mm(workflow_state)
    top_vertex_ids_1_based = workflow_state.mesh.top_node_ids
    bottom_vertex_ids_1_based = workflow_state.mesh.bottom_node_ids

    return FJWDirectProblemSetup(
        workflow_state=workflow_state,
        load_case=load_case,
        mesh_coordinates_mm=coordinates_mm,
        mesh_connectivity=np.asarray(workflow_state.mesh.element_nodes, dtype=np.int32) - 1,
        element_group_ids=element_group_ids,
        material_groups=material_groups,
        top_vertex_ids=np.asarray(top_vertex_ids_1_based, dtype=np.int32) - 1,
        bottom_vertex_ids=np.asarray(bottom_vertex_ids_1_based, dtype=np.int32) - 1,
        top_rp_vertex_id=_closest_vertex_id(top_vertex_ids_1_based, coordinates_mm, top_target_mm),
        bottom_rp_vertex_id=_closest_vertex_id(bottom_vertex_ids_1_based, coordinates_mm, bottom_target_mm),
        load_vector=_build_load_vector(load_case),
        voxel_size_mm=_voxel_size_mm(workflow_state),
    )


def build_fjw_direct_problem(
    setup: FJWDirectProblemSetup,
    *,
    config: FJWDirectSolverConfig | None = None,
    nodal_point_loads: np.ndarray | None = None,
    rigid_point_load: np.ndarray | None = None,
) -> Problem:
    resolved_config = config or FJWDirectSolverConfig()
    dense_nodal_point_loads = None
    active_nodal_load_vertex_ids = np.zeros(0, dtype=np.int32)
    active_nodal_load_values = np.zeros((0, 3), dtype=np.float64)
    if nodal_point_loads is not None:
        dense_nodal_point_loads = _coerce_dense_nodal_point_loads(
            nodal_point_loads,
            expected_node_count=setup.mesh_coordinates_mm.shape[0],
        )
        active_mask = np.any(np.abs(dense_nodal_point_loads) > 0.0, axis=1)
        active_nodal_load_vertex_ids = np.flatnonzero(active_mask).astype(np.int32)
        active_nodal_load_values = dense_nodal_point_loads[active_mask]
    resolved_rigid_point_load = (
        setup.load_vector if rigid_point_load is None else np.asarray(rigid_point_load, dtype=np.float64).reshape(1, 6)
    )

    mesh = Mesh.from_data(
        "fjw_direct_model",
        setup.mesh_coordinates_mm,
        np.zeros(setup.mesh_coordinates_mm.shape[0], dtype=np.int32),
        [setup.mesh_connectivity],
        [setup.element_group_ids],
        ["3_8"],
    )
    domain = FEDomain("domain", mesh)

    selector_functions: dict[str, Callable[..., np.ndarray]] = {
        "get_top_vertices": lambda coors, domain=None: setup.top_vertex_ids.copy(),
        "get_bottom_vertices": lambda coors, domain=None: setup.bottom_vertex_ids.copy(),
        "get_top_rp_vertex": lambda coors, domain=None: np.array([setup.top_rp_vertex_id], dtype=np.int32),
        "get_top_non_rp_vertices": lambda coors, domain=None: np.setdiff1d(
            setup.top_vertex_ids,
            np.array([setup.top_rp_vertex_id], dtype=np.int32),
            assume_unique=False,
        ),
        "get_active_nodal_load_vertices": lambda coors, domain=None: active_nodal_load_vertex_ids.copy(),
    }

    omega = domain.create_region("Omega", "all")
    top_region = domain.create_region(
        "Top",
        "vertices by get_top_vertices",
        "vertex",
        functions=selector_functions,
    )
    top_control_region = domain.create_region(
        "TopControl",
        "vertices by get_top_vertices",
        "facet",
        functions=selector_functions,
    )
    bottom_region = domain.create_region(
        "Bottom",
        "vertices by get_bottom_vertices",
        "vertex",
        functions=selector_functions,
    )
    rp_top_region = domain.create_region(
        "RPTop",
        "vertices by get_top_rp_vertex",
        "vertex",
        functions=selector_functions,
    )
    active_nodal_load_region = None
    if active_nodal_load_vertex_ids.size:
        active_nodal_load_region = domain.create_region(
            "ActiveNodalLoad",
            "vertices by get_active_nodal_load_vertices",
            "vertex",
            functions=selector_functions,
        )
    top_control_not_rp_region = domain.create_region(
        "TopControlNotRP",
        "vertices by get_top_non_rp_vertices",
        "vertex",
        functions=selector_functions,
    )

    field_u = Field.from_args("displacement", np.float64, "vector", omega, approx_order=1)
    field_ur = Field.from_args("rigid_control", np.float64, 6, top_control_region, approx_order=1)
    u = FieldVariable("u", "unknown", field_u)
    v = FieldVariable("v", "test", field_u, primary_var_name="u")
    ur = FieldVariable("ur", "unknown", field_ur)
    vr = FieldVariable("vr", "test", field_ur, primary_var_name="ur")

    integral = Integral("i", order=2)
    elasticity_terms = []
    for material_group in setup.material_groups:
        region = domain.create_region(material_group.region_name, f"cells of group {material_group.group_id}")
        material = Material(
            f"mat_{material_group.group_id}",
            D=stiffness_from_youngpoisson(
                dim=3,
                young=material_group.youngs_modulus,
                poisson=material_group.poisson_ratio,
            ),
        )
        elasticity_terms.append(
            Term.new(
                f"dw_lin_elastic(mat_{material_group.group_id}.D, v, u)",
                integral,
                region,
                **{f"mat_{material_group.group_id}": material, "v": v, "u": u},
            )
        )

    balance_term = elasticity_terms[0]
    for term in elasticity_terms[1:]:
        balance_term = balance_term + term

    balance_expression = balance_term
    if active_nodal_load_region is not None:
        nodal_load_material = Material("nodal_load", values={".val": active_nodal_load_values})
        balance_expression = balance_expression - Term.new(
            "dw_point_load(nodal_load.val, v)",
            integral,
            active_nodal_load_region,
            nodal_load=nodal_load_material,
            v=v,
        )

    load_material = Material("load", values={".val": resolved_rigid_point_load})
    balance = Equation("elasticity", balance_expression)
    rigid_register = Equation(
        "register_rigid_control",
        Term.new("dw_zero(vr, ur)", integral, top_control_region, vr=vr, ur=ur)
        - Term.new("dw_point_load(load.val, vr)", integral, rp_top_region, load=load_material, vr=vr)
    )
    problem = Problem("fjw_direct_problem", equations=Equations([balance, rigid_register]))
    problem.set_bcs(
        ebcs=Conditions(
            [
                EssentialBC("fix_bottom", bottom_region, {"u.all": 0.0}),
                EssentialBC("zero_unused_rigid_control", top_control_not_rp_region, {"ur.all": 0.0}),
            ]
        ),
        lcbcs=Conditions(
            [
                LinearCombinationBC(
                    "top_rigid2",
                    [top_region, rp_top_region],
                    {"u.all": "ur.all"},
                    None,
                    "rigid2",
                ),
            ]
        ),
    )
    if resolved_config.linear_solver_kind == "scipy_direct":
        linear_solver = ScipyDirect(resolved_config.linear_solver_options)
    elif resolved_config.linear_solver_kind == "scipy_iterative":
        iterative_options = {**scipy_iterative_options(), **resolved_config.linear_solver_options}
        linear_solver = ScipyIterative(iterative_options)
    elif resolved_config.linear_solver_kind == "petsc_mumps":
        petsc_options = {**petsc_mumps_options(), **resolved_config.linear_solver_options}
        try:
            linear_solver = PETScKrylovSolver(petsc_options)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The petsc_mumps SfePy backend requires petsc4py and a PETSc build "
                "with MUMPS support. Install that optional solver stack or use "
                "scipy_iterative for the Python-only path."
            ) from exc
    else:
        raise ValueError(f"Unsupported linear_solver_kind: {resolved_config.linear_solver_kind}")
    problem.set_solver(
        Newton(
            resolved_config.nonlinear_solver_options,
            lin_solver=linear_solver,
        )
    )
    return problem


def solve_fjw_direct_case(
    workflow_state: FJWWorkflowState,
    *,
    load_case_name: str,
    design_cage: np.ndarray | None = None,
    obj_bo: np.ndarray | None = None,
    config: FJWDirectSolverConfig | None = None,
    setup: FJWDirectProblemSetup | None = None,
) -> FJWDirectSolveResult:
    resolved_setup = setup or build_fjw_direct_problem_setup(
        workflow_state,
        load_case_name=load_case_name,
        design_cage=design_cage,
        obj_bo=obj_bo,
    )
    problem = build_fjw_direct_problem(resolved_setup, config=config)
    with quiet_sfepy_output():
        state = problem.solve(save_results=False, verbose=False)

    parts = state.get_state_parts()
    nodal_displacements = parts["u"].reshape((-1, 3), order="C")
    rigid_control = parts["ur"].reshape((-1, 6), order="C")
    top_control_index = int(np.argmax(np.linalg.norm(rigid_control, axis=1)))
    rp_top = rigid_control[top_control_index]
    return FJWDirectSolveResult(
        setup=resolved_setup,
        nodal_displacements=nodal_displacements if (config or FJWDirectSolverConfig()).store_nodal_displacements else None,
        rigid_control=rigid_control if (config or FJWDirectSolverConfig()).store_rigid_control else None,
        max_displacement_mm=float(np.max(np.linalg.norm(nodal_displacements, axis=1))),
        top_rp_displacement=nodal_displacements[resolved_setup.top_rp_vertex_id].copy(),
        top_rp_rotation=rp_top[3:].copy(),
    )


def solve_fjw_direct_adjoint_case(
    workflow_state: FJWWorkflowState,
    *,
    load_case_name: str,
    nodal_forces_flat: np.ndarray,
    design_cage: np.ndarray | None = None,
    obj_bo: np.ndarray | None = None,
    config: FJWDirectSolverConfig | None = None,
    setup: FJWDirectProblemSetup | None = None,
) -> FJWDirectSolveResult:
    resolved_setup = setup or build_fjw_direct_problem_setup(
        workflow_state,
        load_case_name=load_case_name,
        design_cage=design_cage,
        obj_bo=obj_bo,
    )
    problem = build_fjw_direct_problem(
        resolved_setup,
        config=config,
        nodal_point_loads=nodal_forces_flat,
        rigid_point_load=np.zeros((1, 6), dtype=np.float64),
    )
    with quiet_sfepy_output():
        state = problem.solve(save_results=False, verbose=False)

    parts = state.get_state_parts()
    nodal_displacements = parts["u"].reshape((-1, 3), order="C")
    rigid_control = parts["ur"].reshape((-1, 6), order="C")
    top_control_index = int(np.argmax(np.linalg.norm(rigid_control, axis=1)))
    rp_top = rigid_control[top_control_index]
    return FJWDirectSolveResult(
        setup=resolved_setup,
        nodal_displacements=nodal_displacements if (config or FJWDirectSolverConfig()).store_nodal_displacements else None,
        rigid_control=rigid_control if (config or FJWDirectSolverConfig()).store_rigid_control else None,
        max_displacement_mm=float(np.max(np.linalg.norm(nodal_displacements, axis=1))),
        top_rp_displacement=nodal_displacements[resolved_setup.top_rp_vertex_id].copy(),
        top_rp_rotation=rp_top[3:].copy(),
    )


__all__ = [
    "FJWDirectMaterialGroup",
    "FJWDirectProblemSetup",
    "FJWDirectSolveResult",
    "FJWDirectSolverConfig",
    "build_fjw_direct_problem",
    "build_fjw_direct_problem_setup",
    "load_fjw_workflow_state",
    "solve_fjw_direct_adjoint_case",
    "solve_fjw_direct_case",
]
