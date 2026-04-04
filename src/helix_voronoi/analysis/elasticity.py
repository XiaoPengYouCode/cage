from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from sfepy.base.base import output as sfepy_output
from sfepy.discrete import (
    Equation,
    Equations,
    FieldVariable,
    Integral,
    Material,
    Problem,
)
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.discrete.fem import FEDomain, Field, Mesh
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.terms import Term

from helix_voronoi.analysis.config import CompressionConfig, MaterialConfig
from helix_voronoi.analysis.voxel import HexMesh

try:
    from pyparsing.exceptions import PyparsingDeprecationWarning
except ImportError:  # pragma: no cover
    PyparsingDeprecationWarning = None

if PyparsingDeprecationWarning is not None:
    warnings.filterwarnings("ignore", category=PyparsingDeprecationWarning)


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
AXIS_TO_NAME = {0: "x", 1: "y", 2: "z"}


@dataclass(frozen=True)
class ElasticitySystem:
    mean_stress_pa: float
    reaction_force_n: float
    solid_volume: float
    active_node_count: int
    active_element_count: int
    top_contact_nodes: int
    bottom_contact_nodes: int


@contextmanager
def quiet_sfepy_output():
    sfepy_output.set_output(quiet=True)
    try:
        yield
    finally:
        sfepy_output.set_output(quiet=False)


def boundary_select(axis: int, is_top: bool, tol: float = 1e-8) -> str:
    axis_name = AXIS_TO_NAME[axis]
    if is_top:
        return f"vertices in {axis_name} > {1.0 - tol}"
    return f"vertices in {axis_name} < {tol}"


def top_plate_ebc(axis: int, compression: CompressionConfig) -> dict[str, float]:
    tangential_axes = [index for index in range(3) if index != axis]
    ebc = {f"u.{axis}": -compression.applied_strain}
    if tangential_axes:
        tangential = ",".join(str(index) for index in tangential_axes)
        ebc[f"u.[{tangential}]"] = 0.0
    return ebc


def build_problem(
    mesh_data: HexMesh,
    material: MaterialConfig,
    compression: CompressionConfig,
) -> tuple[Problem, Integral, np.ndarray, np.ndarray]:
    mesh = Mesh.from_data(
        "modulus_mesh",
        mesh_data.coordinates,
        np.zeros(len(mesh_data.coordinates), dtype=np.int32),
        [mesh_data.connectivity],
        [mesh_data.material_ids],
        [mesh_data.descriptor],
    )
    domain = FEDomain("domain", mesh)
    omega = domain.create_region("Omega", "all")

    axis = AXIS_TO_INDEX[compression.loaded_axis]
    bottom = domain.create_region(
        "Bottom", boundary_select(axis, is_top=False), "vertex"
    )
    top = domain.create_region("Top", boundary_select(axis, is_top=True), "vertex")
    if len(bottom.vertices) == 0 or len(top.vertices) == 0:
        raise ValueError("The selected geometry does not touch both loading plates.")

    field = Field.from_args("displacement", np.float64, "vector", omega, approx_order=1)
    u = FieldVariable("u", "unknown", field)
    v = FieldVariable("v", "test", field, primary_var_name="u")

    solid = Material(
        "solid",
        D=stiffness_from_youngpoisson(
            dim=3,
            young=material.youngs_modulus_pa,
            poisson=material.poisson_ratio,
        ),
    )
    integral = Integral("i", order=2)
    balance = Equation(
        "balance",
        Term.new(
            "dw_lin_elastic(solid.D, v, u)", integral, omega, solid=solid, v=v, u=u
        ),
    )
    problem = Problem("elasticity", equations=Equations([balance]))
    problem.set_bcs(
        ebcs=Conditions(
            [
                EssentialBC("fix_bottom", bottom, {"u.all": 0.0}),
                EssentialBC("load_top", top, top_plate_ebc(axis, compression)),
            ]
        )
    )
    problem.set_solver(
        Newton(
            {"i_max": 1, "eps_a": 1e-10, "eps_r": 1.0},
            lin_solver=ScipyDirect({}),
        )
    )
    return problem, integral, bottom.vertices, top.vertices


def solve_linear_elasticity(
    mesh_data: HexMesh,
    material: MaterialConfig,
    compression: CompressionConfig,
) -> ElasticitySystem:
    with quiet_sfepy_output():
        problem, integral, bottom_nodes, top_nodes = build_problem(
            mesh_data, material, compression
        )
        problem.solve(save_results=False)

        axis = AXIS_TO_INDEX[compression.loaded_axis]
        stress = problem.evaluate(
            "ev_cauchy_stress.i.Omega(solid.D, u)",
            mode="el_avg",
            copy_materials=False,
            verbose=False,
            integrals={"i": integral},
        )
        element_volumes = problem.evaluate(
            "ev_volume.i.Omega(u)",
            mode="el_avg",
            verbose=False,
            integrals={"i": integral},
        )
    stress_component = stress[..., axis, 0].reshape(-1)
    cell_volumes = element_volumes.reshape(-1)
    integrated_stress = float(np.sum(stress_component * cell_volumes))
    solid_volume = float(np.sum(cell_volumes))

    return ElasticitySystem(
        mean_stress_pa=integrated_stress,
        reaction_force_n=abs(integrated_stress),
        solid_volume=solid_volume,
        active_node_count=mesh_data.active_node_count,
        active_element_count=mesh_data.active_element_count,
        top_contact_nodes=len(top_nodes),
        bottom_contact_nodes=len(bottom_nodes),
    )
