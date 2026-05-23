from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sfepy.base.conf import ProblemConf
from sfepy.base.base import output as sfepy_output
from sfepy.discrete import Problem
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.mesh.mesh_generators import gen_block_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = PROJECT_ROOT / "datasets" / "sfepy_validation"
MESH_PATH = VALIDATION_DIR / "rigid2_block.mesh"
REPORT_PATH = VALIDATION_DIR / "rigid2_report.json"

BLOCK_DIMS = np.array([1.0, 1.0, 1.0], dtype=np.float64)
BLOCK_SHAPE = np.array([5, 5, 5], dtype=np.int32)
BLOCK_CENTRE = np.array([0.0, 0.0, 0.0], dtype=np.float64)
TOL = 1e-8
LOAD_VECTOR = np.array([[0.0], [0.0], [-1.0]], dtype=np.float64)


def _prepare_mesh() -> str:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    mesh = gen_block_mesh(
        BLOCK_DIMS,
        BLOCK_SHAPE,
        BLOCK_CENTRE,
        name="rigid2_block",
        verbose=False,
    )
    mesh.write(str(MESH_PATH), io="auto")
    return str(MESH_PATH)


filename_mesh = _prepare_mesh()

options = {
    "ls": "ls",
    "nls": "newton",
}


def _target_top_rp_coordinate() -> np.ndarray:
    return np.array(
        [
            BLOCK_CENTRE[0],
            BLOCK_CENTRE[1],
            BLOCK_CENTRE[2] + BLOCK_DIMS[2] / 2.0,
        ],
        dtype=np.float64,
    )


def get_rp_vertices(coors: np.ndarray, domain=None) -> np.ndarray:
    target = _target_top_rp_coordinate()
    distances = np.linalg.norm(coors - target[None, :], axis=1)
    return np.array([int(np.argmin(distances))], dtype=np.int32)


def get_not_rp_vertices(coors: np.ndarray, domain=None) -> np.ndarray:
    rp = get_rp_vertices(coors, domain=domain)
    all_vertices = np.arange(coors.shape[0], dtype=np.int32)
    mask = np.ones(coors.shape[0], dtype=bool)
    mask[rp] = False
    return all_vertices[mask]


functions = {
    "get_rp_vertices": (get_rp_vertices,),
    "get_not_rp_vertices": (get_not_rp_vertices,),
}

fields = {
    "displacement": ("real", 3, "Omega", 1),
    "rigid_control": ("real", 6, "Omega", 1),
}

materials = {
    "solid": ({
        "D": stiffness_from_youngpoisson(dim=3, young=18_000.0, poisson=0.3),
    },),
    "load": ({
        "val": LOAD_VECTOR,
    },),
}

variables = {
    "u": ("unknown field", "displacement", 0),
    "v": ("test field", "displacement", "u"),
    "ur": ("unknown field", "rigid_control", 1),
    "vr": ("test field", "rigid_control", "ur"),
}

regions = {
    "Omega": "all",
    "Bottom": ("vertices in (z < -0.49999999)", "facet"),
    "Top": ("vertices in (z > 0.49999999)", "facet"),
    "TopLoad": ("vertices in (z > 0.49999999) & (x > 0.0)", "facet"),
    "RP": ("vertices by get_rp_vertices", "vertex"),
    "NotRP": ("vertices by get_not_rp_vertices", "vertex"),
}

ebcs = {
    "fix_bottom": ("Bottom", {"u.all": 0.0}),
    "zero_unused_rigid_control": ("NotRP", {"ur.all": 0.0}),
}

lcbcs = {
    "top_rigid2": (
        ["Top", "RP"],
        {"u.all": "ur.all"},
        None,
        "rigid2",
    ),
}

equations = {
    "elasticity": (
        "dw_lin_elastic.2.Omega(solid.D, v, u)"
        " = - dw_surface_ltr.2.TopLoad(load.val, v)"
    ),
    "register_rigid_control": "dw_zero.2.Omega(vr, ur) = 0",
}

solvers = {
    "ls": ("ls.scipy_direct", {}),
    "newton": ("nls.newton", {
        "i_max": 1,
        "eps_a": 1e-10,
    }),
}


def build_problem() -> Problem:
    conf = ProblemConf.from_dict(globals(), sys.modules[__name__])
    return Problem.from_conf(conf)


def _expected_rigid_displacement(
    coordinates: np.ndarray,
    reference_coordinate: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    relative = coordinates - reference_coordinate[None, :]
    return translation[None, :] + np.cross(rotation[None, :], relative)


def run_validation() -> dict[str, object]:
    problem = build_problem()

    sfepy_output.set_output(quiet=True)
    try:
        state = problem.solve(save_results=False, verbose=False)
    finally:
        sfepy_output.set_output(quiet=False)

    parts = state.get_state_parts()
    displacement = parts["u"].reshape((-1, 3), order="C")
    rigid_control = parts["ur"].reshape((-1, 6), order="C")

    top_vertices = problem.domain.regions["Top"].vertices
    rp_vertex = int(problem.domain.regions["RP"].vertices[0])
    coordinates = problem.domain.mesh.coors
    reference_coordinate = coordinates[rp_vertex]

    translation = rigid_control[rp_vertex, :3]
    rotation = rigid_control[rp_vertex, 3:]
    expected_top = _expected_rigid_displacement(
        coordinates[top_vertices],
        reference_coordinate,
        translation,
        rotation,
    )
    actual_top = displacement[top_vertices]
    top_error = actual_top - expected_top
    top_error_norm = np.linalg.norm(top_error, axis=1)

    report = {
        "mesh_path": str(MESH_PATH),
        "block_dims": BLOCK_DIMS.tolist(),
        "block_shape_nodes": BLOCK_SHAPE.tolist(),
        "top_vertex_count": int(len(top_vertices)),
        "reference_point_vertex": rp_vertex,
        "reference_point_coordinate": reference_coordinate.tolist(),
        "load_vector": LOAD_VECTOR.reshape(-1).tolist(),
        "solved_translation": translation.tolist(),
        "solved_rotation": rotation.tolist(),
        "solved_rotation_norm": float(np.linalg.norm(rotation)),
        "max_top_error": float(np.max(top_error_norm)),
        "mean_top_error": float(np.mean(top_error_norm)),
        "top_center_displacement": displacement[rp_vertex].tolist(),
        "verification_passed": bool(
            (np.max(top_error_norm) < 1e-10)
            and (np.linalg.norm(rotation) > 1e-12)
        ),
    }

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    report = run_validation()
    print(json.dumps(report, indent=2))
    return 0 if report["verification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
