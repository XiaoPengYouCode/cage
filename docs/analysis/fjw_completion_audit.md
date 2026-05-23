# FJW Completion Audit

Audit date: 2026-05-02

## Objective

Implement the FJW workflow in `docs/goal.md` as a Python-only production path:
SfePy/SciPy replaces Abaqus executable/license for the main runtime, with no
stub or mock production implementation.

## Verdict

Status: Python-only target restored; local implementation and tests pass.

The previous audit treated Abaqus and PETSc/MUMPS as hard completion gates. That
was the wrong target for this project. The current target is:

- SfePy/SciPy is the production solver path.
- `scipy_iterative` is the default Python-only solver profile.
- Abaqus remains optional historical comparison tooling.
- PETSc/MUMPS remains optional acceleration tooling.
- Default preflight checks only Python dependencies and reference inputs.

Do not use missing Abaqus, missing `petsc4py`, or missing historical runtime
golden files as blockers for Python-only completion. They are optional evidence
sources.

## Evidence Captured

Commands run during this audit:

```bash
rtk uv run python -m unittest discover -s tests -v
```

Result: `Ran 72 tests in 13.671s` and `OK`.

```bash
rtk uv run fem-analysis fjw-preflight
```

Result: status `pass`. Checks included:

- reference directory exists
- `numpy` importable
- `scipy` importable
- `sfepy` importable

```bash
rtk uv run fem-analysis fjw-direct --load-case force_1
rtk uv run fem-analysis fjw-direct --load-case force_1 --build-problem
```

Result: both commands exit successfully on the full reference inputs. The setup
reports `593790` nodes, `544112` elements, `3936` top nodes, `5189` bottom
nodes, and the SfePy problem exposes `elasticity` plus `register_rigid_control`
equations.

```bash
rtk rg -n "stub|mock|placeholder|TODO|FIXME|NotImplemented|pass$|zero vector|zero-vector" \
  src/fem_analysis tests docs README.md
```

Result: no production `TODO`, `NotImplemented`, or `pass` placeholder matches.
Remaining matches are tests, documentation, CLI output fields, and the
`has_placeholder_adjoint` detector used to fail unsafe states.

## Prompt To Artifact Checklist

| Requirement from `docs/goal.md` | Evidence | Status |
| --- | --- | --- |
| Load FJW static inputs: `nod_coo`, `ele_nod`, `B`, `D`, regions, design and objective elements | `src/fem_analysis/fjw_reference.py`, `src/fem_analysis/fjw_workflow_loaders.py`, `datasets/fjw_input_inventory.json`, FJW regression tests | Implemented |
| Initialize `ini_str = 0.36` and three-load `ini_cage = 0.3` | workflow state loaders and optimizer tests | Implemented |
| Run `force_1`, `force_2`, `force_3` per optimization iteration | `src/fem_analysis/fjw_workflow_driver.py`, `src/fem_analysis/fjw_workflow_three_force.py`, regression tests | Implemented |
| Run `P = 3` forward remodeling steps per force case | single-case runtime and dynamic pipeline tests | Implemented |
| Run reverse adjoint chain and generate dynamic `Fv` | `src/fem_analysis/fjw_workflow_single_case.py`, `src/fem_analysis/fjw_workflow_adjoint.py`, `tests/test_fjw_dynamic_adjoint_pipeline.py` | Implemented |
| Solve forward and adjoint through Python-only SfePy/SciPy | `src/fem_analysis/fjw_direct_solver.py`, `src/fem_analysis/fjw_workflow_sfepy_solver_adapters.py`, direct solver tests | Implemented locally |
| Keep SfePy rigid-control runtime friendly | `src/fem_analysis/fjw_direct_solver.py` limits the 6-DOF control field to the top control surface instead of the full mesh | Implemented |
| Combine three-force terminal bone mass objective | `src/fem_analysis/fjw_workflow_three_force.py`, workflow regression tests | Implemented |
| Combine three-force gradient and update `design_cage` through MMA | `src/fem_analysis/fjw_mma.py`, `src/fem_analysis/fjw_workflow_optimizer.py`, `tests/test_fjw_mma.py` | Implemented |
| Repeat until `delta <= 1e-4` | `src/fem_analysis/fjw_workflow_optimize.py`, optimizer tests | Implemented |
| Persist checkpoints and validation artifacts | checkpoint IO, validation module, validation tests | Implemented |
| Deep-check checkpoint contents in validation report | validation reads MMA, aggregate, case history, forward step, `Fv`, `Fai`, and `fv_manifest.json` artifacts | Implemented |
| Keep Abaqus optional | preflight defaults, docs, CLI `--real-run` gates | Implemented |
| Keep PETSc/MUMPS optional | preflight defaults, solver profile handling, missing-runtime test | Implemented |

## Phase Checklist

| Phase | Deliverable | Evidence | Status |
| --- | --- | --- | --- |
| Python input baseline | Static input inventory and small golden fixtures | `datasets/fjw_golden/`, `scripts/build_fjw_golden_case.py`, `tests/test_fjw_golden_cases.py` | Implemented |
| SfePy/SciPy backend | Real direct and adjoint solve adapters | `fjw_direct_solver.py`, `fjw_workflow_sfepy_solver_adapters.py`, direct/adapter tests | Implemented locally |
| Dynamic adjoint | Runtime `Fv` from forward displacement and `Fai_next` | `tests/test_fjw_dynamic_adjoint_pipeline.py` | Implemented |
| Real MMA | Python/NumPy `mmasub.m` and `subsolv.m` port | `src/fem_analysis/fjw_mma.py`, `tests/test_fjw_mma.py` | Implemented |
| Optimization loop | `fjw-optimize`, checkpoint/resume, stable output | `src/fem_analysis/fjw_workflow_optimize.py`, optimizer tests | Implemented |
| Validation | run validation, checkpoint content checks, golden capture, checksum compare | `src/fem_analysis/fjw_validation.py`, validation tests | Implemented |
| Runtime preflight | Python-only default; optional strict gates | `src/fem_analysis/fjw_environment.py`, environment tests, CLI smoke | Implemented |
| Documentation | Python-only goal/runbook/start docs | `docs/goal.md`, `README.md`, `docs/how_to_start.md`, `docs/fjw_workflow_runbook.md` | Updated |

## CLI Checklist

| CLI | Expected role | Status |
| --- | --- | --- |
| `uv run fem-analysis fjw-preflight` | Python-only runtime check | Passes locally |
| `uv run fem-analysis fjw-direct --load-case force_1 --build-problem` | Full reference SfePy problem setup | Passes locally |
| `uv run fem-analysis fjw-direct --load-case force_1 --solve` | SfePy direct solve entry | Implemented; full reference solve remains resource-dependent |
| `uv run fem-analysis fjw-sfepy-iterate --num-time-steps 1` | One SfePy workflow iteration | Covered by CLI/unit tests |
| `uv run fem-analysis fjw-optimize --backend sfepy --mode three-force --max-iterations 1` | Python-only optimization loop | Covered by optimizer tests |
| `uv run fem-analysis fjw-validate --run-directory runs/fjw_optimize` | Structural and optional golden validation | Implemented |
| `uv run fem-analysis fjw-capture-golden --run-directory runs/fjw_optimize` | Python/SfePy golden capture | Implemented |
| `uv run fem-analysis fjw-workflow --mode three-force --execute-jobs --real-run` | Optional Abaqus comparison | Optional |
| `uv run fem-analysis fjw-direct --sfepy-linear-solver petsc_mumps --solve` | Optional large-model acceleration | Optional |

## Definition Of Done Audit

| DoD item | Evidence | Status |
| --- | --- | --- |
| Default Python-only preflight passes | `fjw-preflight` status `pass` | Complete |
| SfePy/SciPy produces real forward displacement vectors | direct solver and SfePy adapter tests | Complete locally |
| Three-force mode runs real forward + adjoint + MMA from `design_cage = 0.3` | SfePy iteration and optimizer tests | Complete locally |
| Every forward time step has real displacement vectors | SfePy adapter tests cover returned arrays | Complete locally |
| Every adjoint time step `Fv` comes from forward displacement and `Fai_next` | dynamic adjoint pipeline test | Complete locally |
| Every adjoint time step has real adjoint displacement | SfePy adjoint adapter tests | Complete locally |
| MMA update comes from real `mmasub/subsolv` equivalent implementation | `src/fem_analysis/fjw_mma.py`, MMA tests | Complete |
| Outer loop stops on `delta <= 1e-4` | optimizer config and tests | Complete |
| Key intermediates are persisted, resumable, and auditable | checkpoint IO and validation tests | Complete |
| Validation report covers run structure, checkpoint arrays, solver artifacts, and optional golden comparison | validation tests | Complete |
| Abaqus is not required for the main path | preflight defaults and docs | Complete |
| PETSc/MUMPS is not required for the base path | preflight defaults and docs | Complete |

## Remaining Risk

Reference-scale solving is still resource-sensitive. The code can load the full
reference model and construct the SfePy setup, but a full `544112`-element solve
was not run in this audit. That is a performance and resource validation item,
not an Abaqus license blocker.

Historical MATLAB/Abaqus runtime outputs are still absent from
`references/fjw_work/`. If those outputs become available, they should be added
as optional comparison goldens. Their absence does not invalidate the Python-only
implementation.

## Next Runtime Actions

For a local Python-only smoke run:

```bash
rtk uv run fem-analysis fjw-preflight
rtk uv run fem-analysis fjw-optimize \
  --backend sfepy \
  --mode three-force \
  --max-iterations 1 \
  --num-time-steps 1 \
  --run-directory runs/fjw_optimize
rtk uv run fem-analysis fjw-validate \
  --run-directory runs/fjw_optimize \
  --output runs/fjw_optimize/validation_report.json
```

For reference-scale resource validation:

```bash
rtk uv run fem-analysis fjw-direct \
  --load-case force_1 \
  --solve \
  --sfepy-linear-solver scipy_iterative
```

For optional acceleration:

```bash
rtk uv run fem-analysis fjw-direct \
  --load-case force_1 \
  --solve \
  --sfepy-linear-solver petsc_mumps
```

For optional historical comparison:

```bash
rtk uv run fem-analysis fjw-optimize \
  --backend abaqus \
  --mode three-force \
  --max-iterations 1 \
  --num-time-steps 3 \
  --real-run \
  --run-directory runs/fjw_optimize_abaqus
```
