# FJW Workflow Runbook

This runbook covers the upstream FJW pseudo-density generation node in the full cage pipeline. The downstream stages consume the resulting density/design field through the repository's standard density NPZ flow, then continue through `topopt_sampling`, Voronoi geometry, and STL/GLB export.

## Runtime Preflight

```bash
uv run fem-analysis fjw-preflight
```

This checks the Python-only runtime path: reference inputs plus `numpy`, `scipy`,
and `sfepy`. This is the default supported execution path and does not require
Abaqus, `petsc4py`, or PETSc/MUMPS.

Optional historical/large-model gates can be requested explicitly:

```bash
uv run fem-analysis fjw-preflight \
  --require-golden \
  --golden-directory datasets/fjw_golden/captured_run
```

## Optional Archived Input Generation

```bash
uv run fem-analysis fjw-workflow --mode three-force --time-steps 3
```

This writes generated Abaqus inputs and manifests under `runs/fjw_workflow/`.
Use it to audit compatibility with archived `.inp` templates. Static adjoint
inputs in this mode use zero `Fv` and are template artifacts only; they are not
the Python-only solver path.

## SfePy Optimization Smoke Run

```bash
uv run fem-analysis fjw-optimize \
  --backend sfepy \
  --mode three-force \
  --max-iterations 1 \
  --num-time-steps 1 \
  --runtime-profile local \
  --run-directory runs/fjw_optimize
```

Use this for Python workflow validation. `local` keeps the Python-only
`scipy_iterative` solver and runs force cases sequentially.

For the remote production machine, use the default `wuyinyun` profile:

```bash
uv run fem-analysis fjw-preflight --require-petsc-mumps
uv run fem-analysis fjw-optimize \
  --backend sfepy \
  --mode three-force \
  --resume \
  --runtime-profile wuyinyun \
  --run-directory runs/fjw_optimize
```

The `wuyinyun` profile selects `petsc_mumps`, `case_parallelism=2`, and
`solver_threads=12` for a 48 GB host. Override individual knobs only when a run
shows memory pressure or PETSc/MUMPS is unavailable:

```bash
uv run fem-analysis fjw-optimize \
  --backend sfepy \
  --runtime-profile wuyinyun \
  --case-parallelism 1 \
  --sfepy-linear-solver scipy_iterative
```

Each completed iteration writes `iter_###/timing.json`. The timing tree records
the outer iteration, force-case batch, each forward solve, each adjoint solve,
the aggregate step, and checkpoint write time.

SfePy setup caching is enabled by default. Disable it only for debugging:

```bash
uv run fem-analysis fjw-optimize --disable-sfepy-setup-cache
```

## Optional Abaqus Comparison

```bash
uv run fem-analysis fjw-optimize \
  --backend abaqus \
  --mode three-force \
  --max-iterations 1 \
  --num-time-steps 3 \
  --real-run \
  --abaqus-executable abaqus
```

The Abaqus optimizer refuses dry-run mode because real forward and adjoint displacement vectors are required. Each job gets its own run directory under `runs/fjw_optimize/abaqus_jobs/`.

## Checkpoints And Resume

Each run writes:

- `workflow_manifest.json`
- `iter_000/design_cage.npz`
- `iter_###/iteration_state.json`
- `iter_###/timing.json`
- `iter_###/design_cage.npz`
- `iter_###/mma_state.npz`
- `iter_###/aggregate_terms.npz`
- per-case `forward_t*` and `adjoint_t*` artifacts

Resume from the newest complete checkpoint:

```bash
uv run fem-analysis fjw-optimize --backend sfepy --resume --run-directory runs/fjw_optimize
```

## Validation

```bash
uv run fem-analysis fjw-validate --run-directory runs/fjw_optimize
```

Without historical golden data, the report is allowed to validate artifact completeness but must say historical MATLAB + Abaqus equivalence is not proven.

## Capturing A Golden Source

After a successful Python/SfePy run, capture a compact golden manifest:

```bash
uv run fem-analysis fjw-capture-golden \
  --run-directory runs/fjw_optimize \
  --golden-directory datasets/fjw_golden/captured_run
```

The capture writes `golden_manifest.json` with file sizes and SHA256 checksums.
Files up to `--copy-max-bytes` are copied into the golden directory; larger
solver outputs are checksum-only unless the limit is raised intentionally.
By default, capture refuses run directories that fail structural validation;
`--allow-invalid-run` is only for manual salvage of partial artifacts.

## Troubleshooting

- Abaqus missing: pass `--abaqus-executable` or load the Abaqus environment before running.
- ODB missing: inspect the job directory `.log`, `.dat`, `.sta`, and `.msg` files.
- `U1.txt` missing: inspect the generated `*_odb_export.py` script and Abaqus/CAE export logs.
- SfePy memory pressure: keep `scipy_iterative` as the Python-only default, use `scipy_direct` only for small debugging models, and use `petsc_mumps` only when a PETSc/MUMPS environment is intentionally available.
- MMA not converging: inspect `iter_###/iteration_state.json` and `mma_state.npz`, especially `delta`, `g2`, `low`, and `up`.
- Resume failure: ensure the latest `iter_###` directory has `iteration_state.json`, `design_cage.npz`, and `mma_state.npz`.
