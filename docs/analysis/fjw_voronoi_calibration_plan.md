# FJW Voronoi Calibration Plan

Date: 2026-05-26

## Objective

This document defines the missing calibration layer between the upstream
FJW pseudo-density design field and the downstream titanium Voronoi rod
network.

The calibration target is not geometric resemblance.

The calibration target is:

```text
local pseudo-density x
  -> target equivalent modulus E_target(x)
  -> Voronoi geometry parameters theta(x)
  -> local equivalent modulus E_eff(theta)
  -> E_eff(theta) ≈ E_target(x)
```

Until this layer exists, any direct binary replacement or fill-fraction
replacement can only be treated as an exploratory dry run.

## Physical Baseline

The current FJW implementation already interprets pseudo-density as a target
mechanical field.

In `src/fem_analysis/fjw_workflow_forward.py`, the cage modulus interpolation
is:

```text
E_c(x) = E_c,min + E_c,0 * x^3
```

with clipping to `E_c,0`.

Therefore:

- `x` is not a direct rod-radius field;
- `x` is not a direct seed-density field;
- `x` first defines the target equivalent stiffness demand inside the design
  domain.

The Voronoi stage must be calibrated against that target demand.

## First Practical Choice

For the first repository-native calibration pass, use one primary geometry
variable:

- `r_eff(x)`: effective rod radius

and hold the following fixed:

- seed distribution rule: existing density-guided sampling with fixed `gamma`
- CVT relaxation rule: existing Lloyd relaxation
- local topology family: current box-restricted Voronoi skeleton workflow
- local cell size: fixed physical unit-cell size
- base subdivision rule: current fine-grid subdivision

This is the simplest path that matches the current codebase.

The reason for choosing rod radius first is practical:

1. the current geometry chain already exposes a radius-like control through
   `dilation_radius_fine_voxels`;
2. seed density and topology changes are more entangled and would enlarge the
   calibration search space immediately;
3. a one-variable calibration pass is enough to establish the academic logic:
   pseudo-density is mapped to target modulus first, then geometry is tuned to
   hit that modulus.

This does not claim rod radius alone is sufficient for the final thesis. It
defines the first executable calibration baseline.

## Calibration Strategy

### Step 1. Build target-modulus bands from `iter_017`

Compute the target cage modulus values from the real `iter_017` design field.

Then compress the continuous field into a small number of representative bands,
for example:

- very low
- low
- medium
- medium-high
- high

Each band should record:

- pseudo-density range
- target modulus range
- representative pseudo-density
- representative target modulus
- number of design elements in the band

The goal is not to quantize the final design permanently. The goal is to
choose a tractable set of calibration targets for the first FE campaign.

### Step 2. Define a local calibration specimen

For each representative target modulus, build a local Voronoi specimen with:

- fixed physical bounding box
- fixed seed count or fixed seed-generation rule
- fixed CVT rule
- variable effective rod radius

The specimen should be large enough that boundary locking does not dominate the
response, but small enough that repeated FE runs remain cheap.

For the first pass, a periodic-specimen idealization is acceptable in the
analysis document even if the first code implementation uses a simpler bounded
block.

### Step 3. Sweep rod radius

For each target-modulus band, sweep a radius set such as:

```text
r = {r1, r2, ..., rk}
```

using the current Voronoi geometry path.

For each radius:

1. generate local skeleton geometry;
2. voxelize / solidify using repository-native code;
3. run local FE;
4. measure the specimen-level effective stiffness or modulus.

### Step 4. Fit the inverse map

Fit or tabulate:

```text
E_eff = f(r | fixed seed rule, fixed topology family)
```

and then invert to obtain:

```text
r_eff = f^-1(E_target)
```

The inversion can be implemented initially as:

- nearest-neighbor lookup on a monotone table, or
- 1D interpolation over the sampled radius-modulus curve.

### Step 5. Push calibrated parameters back to the full structure

After the local map exists, use the target-modulus bands in the full `iter_017`
field to assign a calibrated radius field.

That radius field then drives regenerated Voronoi skeleton geometry.

Only after that regeneration should the full FE replacement validation be run.

## Metrics For The Local Calibration FE

For each local specimen and radius candidate, record:

- solid volume fraction
- apparent stiffness
- apparent modulus
- directional response if measured in multiple axes
- mesh / voxel metadata
- geometry-generation parameters

For the first pass, one scalar modulus aligned with the dominant loading
direction is sufficient.

The final thesis may need directional stiffness tensors later. That is a second
step, not the first executable baseline.

## What The Current Repository Already Supports

Existing code that can be reused:

- target modulus evaluation
  - `src/fem_analysis/fjw_workflow_forward.py`
- Voronoi seed and geometry generation
  - `src/matlab2stl_pipeline/`
  - `src/helix_voronoi/`
- voxel solidification with radius-like control
  - `src/matlab2stl_pipeline/skeleton_voxelizer.py`
  - via `dilation_radius_fine_voxels`
- STL voxelization
  - `src/ct_reconstruction/voxelizer.py`

What is still missing:

- a dedicated calibration specimen generator
- a local FE evaluator for those specimens
- a stored calibration table from radius to effective modulus
- a regeneration path that attaches calibrated geometry parameters to the full
  Voronoi skeleton

The current turn adds two concrete first-pass tools around this gap:

- `Post process/analysis/build_iter017_target_modulus_bands.py`
  - summarizes the real `iter_017` target-modulus field into representative
    calibration bands;
- `Post process/analysis/generate_voronoi_radius_calibration_specs.py`
  - generates a first-pass STL specimen family for radius sweeps at fixed seed
    and topology settings, with an explicit physical `cell_size_mm`.

The current turn also adds:

- `Post process/analysis/evaluate_voronoi_radius_calibration.py`
  - voxelizes those STL specimens and runs a local compression FE solve to
    measure apparent stiffness and apparent modulus.

## First Real FE Results

The current real FE calibration baseline has now been executed on `wuyinyun`
using:

- target-modulus bands from real `iter_017`;
- one fixed local Voronoi seed realization in `radius_only` layout;
- fixed physical cell size `4.0 mm`;
- radius sweep `{0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.30} mm`;
- voxel size `0.1 mm`;
- material `TC4`, `E = 110 GPa`, `nu = 0.34`;
- compression load `200 N`.

Outputs:

- `Post process/analysis/output/iter017_target_modulus_bands.json`
- `outputs/voronoi_radius_calibration_radius_only_remote_wide_v2/calibration_spec_manifest.json`
- `outputs/voronoi_radius_calibration_radius_only_remote_wide_v2/calibration_fe_results.json`
- `Post process/analysis/output/voronoi_radius_calibration_summary.json`
- `Post process/analysis/output/iter017_band_radius_lookup.json`

Three important facts are now established by real FE data:

1. the local specimen generation -> STL -> voxelization -> FE evaluation chain
   is physically well-scaled and executable;
2. increasing rod radius gives a strong first-pass `r -> E_eff` calibration
   signal over the stable part of the sweep;
3. the sweep already shows that some radii are mechanically unstable for this
   specimen family and must be excluded from inverse calibration.

Current stable support is:

| radius | solid volume fraction | apparent modulus | status |
|---|---:|---:|---|
| `0.08 mm` | `0.0537` | `5.01 GPa` | stable |
| `0.10 mm` | `0.0764` | `16.74 GPa` | stable |
| `0.12 mm` | `0.0950` | `22.44 GPa` | stable |
| `0.20 mm` | `0.1999` | `47.47 GPa` | stable |
| `0.24 mm` | `0.2582` | `57.33 GPa` | stable |
| `0.30 mm` | `0.3183` | `64.99 GPa` | stable |

Two sampled radii are currently unusable for inverse lookup:

| radius | apparent modulus | max displacement | interpretation |
|---|---:|---:|---|
| `0.06 mm` | `5.12e-08 GPa` | `1.09e10 mm` | loss of stable load path |
| `0.16 mm` | `4.19e-11 GPa` | `3.79e12 mm` | loss of stable load path |

Therefore the current measured stable baseline is:

```text
0.08 mm -> 5.01 GPa
0.10 mm -> 16.74 GPa
0.12 mm -> 22.44 GPa
0.20 mm -> 47.47 GPa
0.24 mm -> 57.33 GPa
0.30 mm -> 64.99 GPa
```

## What These Results Prove

These results prove that the calibration layer is no longer hypothetical.

The repository now has a real local homogenization-style baseline:

```text
Voronoi radius r
  -> STL specimen
  -> voxel solid
  -> local FE compression
  -> apparent modulus E_eff
```

This is enough to justify the thesis statement that pseudo-density must first
be converted into target equivalent stiffness and only then mapped to geometry
parameters through calibration.

## What These Results Do Not Yet Prove

The current specimen family still uses the same local Voronoi topology for all
target-modulus bands.

So the six bands currently differ only in their recorded
`representative_target_modulus`, while the generated geometry depends only on
the chosen radius.

Therefore the current outputs do **not** yet provide a complete map

```text
E_target(x) -> r_eff(x)
```

for the real `iter_017` field.

What exists now is the first measured basis needed to build that map.

## Immediate Next Step

The next implementation step is now sharply defined:

1. choose a lookup or interpolation rule on the stable measured `r -> E_eff`
   curve;
2. expand the radius sweep so the reachable modulus range covers the intended
   `E_target` bands more completely at both low and high ends;
3. assign each `iter_017` target-modulus band a calibrated radius, even if the
   first pass is only bandwise and piecewise-constant;
4. regenerate the full Voronoi skeleton with that calibrated radius field;
5. only then run the full three-force FE replacement validation.

## Bandwise Radius Lookup

The repository now contains a bandwise inverse lookup:

- `Post process/analysis/summarize_voronoi_radius_calibration.py`
- `Post process/analysis/build_iter017_band_radius_lookup.py`
- `Post process/analysis/output/voronoi_radius_calibration_summary.json`
- `Post process/analysis/output/iter017_band_radius_lookup.json`
- `Post process/analysis/output/iter017_band_radius_lookup_combined_seed55_plus_lowmid.json`

The lookup uses piecewise-linear inverse interpolation on the monotone frontier
of the stable measured mean `r -> E_eff` curve, with clamping outside the
stable support. Non-monotone or unstable radius rows are kept as diagnostics
but excluded from inverse assignment.

The first wide sweep established this stable support:

```text
0.08 mm -> 5.01 GPa
0.10 mm -> 16.74 GPa
0.12 mm -> 22.44 GPa
0.20 mm -> 47.47 GPa
0.24 mm -> 57.33 GPa
0.30 mm -> 64.99 GPa
```

The first lookup was too narrow at the low end:

| band | representative `E_target` | assigned radius | status |
|---|---:|---:|---|
| 0 | `0.0113 GPa` | `0.08 mm` | `clamped_low` |
| 1 | `0.0123 GPa` | `0.08 mm` | `clamped_low` |
| 2 | `0.0209 GPa` | `0.08 mm` | `clamped_low` |
| 3 | `2.3763 GPa` | `0.08 mm` | `clamped_low` |
| 4 | `36.0443 GPa` | `0.1635 mm` | `interpolated` |
| 5 | `106.0697 GPa` | `0.30 mm` | `clamped_high` |

That table was useful because it made the inadequacy explicit:

- low-modulus bands all collapse to the minimum sampled radius;
- the highest-modulus band saturates at the maximum sampled radius.

The same-family low/mid sweep then added stable points around the dominant
`band 3` target:

```text
0.077 mm -> 2.06 GPa
0.078 mm -> 2.10 GPa
0.079 mm -> 5.24 GPa
0.081 mm -> 9.46 GPa
0.082 mm -> 10.48 GPa
```

The updated combined lookup assigns:

| band | representative `E_target` | assigned radius | status |
|---|---:|---:|---|
| 0 | `0.0113 GPa` | `0.077 mm` | `clamped_low` |
| 1 | `0.0123 GPa` | `0.077 mm` | `clamped_low` |
| 2 | `0.0209 GPa` | `0.077 mm` | `clamped_low` |
| 3 | `2.3763 GPa` | `0.07809 mm` | `interpolated` |
| 4 | `36.0443 GPa` | `0.1635 mm` | `interpolated` |
| 5 | `106.0697 GPa` | `0.30 mm` | `clamped_high` |

This is a real improvement at the edge-assignment level. In the updated
`iter_017` variable-radius edge field, only `0.50%` of edges remain low-clamped
and `99.46%` of edges use interpolated radii. The dominant `band 3` no longer
collapses to the minimum stable radius.

This does not yet prove structure-level equivalence. The high-modulus band
still clamps at the maximum sampled radius, and the coarse replacement field
still loses most of the target modulus when the sparse skeleton is aggregated
back to the original design grid.

## Current Repository Decision

The first calibration implementation in this repository is now fixed as:

1. primary geometry control variable: rod radius `r(x)`;
2. current inverse map carrier:
   `iter017_band_radius_lookup_combined_seed55_plus_lowmid.json`;
3. first full-structure FE input: `design_cage_modulus_weighted`;
4. next required expansion: improve the coarse homogenization / aggregation
   rule and broaden high-modulus support before claiming full-structure
   equivalence.

## First Full-Structure Radius Attachment

The repository now also has a first executable path that pushes the current
bandwise lookup back onto the real `iter_017` Voronoi skeleton.

Artifacts:

- `outputs/fjw_optimize_real_iter017/fjw_iter017_voronoi_edges_variable_radius.npz`
- `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_voxels_variable_radius.npz`
- `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_voxels_variable_radius_smoke.npz`
- `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_variable_radius_smoke.glb`
- `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_variable_radius_smoke.stl`

Supporting scripts:

- `Post process/analysis/build_iter017_variable_radius_edges.py`
- `Post process/analysis/build_iter017_variable_radius_skeleton.py`

Current rule:

1. take each Voronoi edge in `fjw_iter017_voronoi_edges_density.npz`;
2. sample the aligned density field at the edge midpoint;
3. convert sampled pseudo-density to `E_target` using the same cage modulus
   interpolation used by `fem_analysis`;
4. assign the nearest current bandwise calibrated radius from
   `iter017_band_radius_lookup.json`;
5. voxelize the whole skeleton with that per-edge physical radius field.

What this already proves:

- the repository now has a real end-to-end path from
  `rho(x) -> E_target(x) -> bandwise radius -> full-structure skeleton voxels`;
- the calibration layer is no longer only a local specimen FE table;
- the downstream geometry chain can already consume a spatially varying radius
  field instead of one global dilation radius.

What it does **not** prove yet:

- midpoint sampling is only the first attachment rule, not yet the final local
  homogenization rule;
- the updated inverse lookup fixes the dominant low-end edge assignment, but
  the highest target band still clamps to the sampled high end;
- the resulting variable-radius skeleton has only been converted into a first
  FE-ready proxy field, not yet the final homogenized replacement tensor
  required by the three-force forward replacement validation.

## First FE-Ready Proxy Replacement

The repository now also contains a first FE-ready proxy replacement design:

- `outputs/fjw_optimize_real_iter017/fjw_iter017_replacement_design_variable_radius.npz`
- `Post process/analysis/build_iter017_variable_radius_replacement_design.py`

Current rule:

1. read per-fine-voxel physical radius from the variable-radius skeleton;
2. convert radius to calibrated apparent modulus using the stable support in
   `iter017_band_radius_lookup_combined_seed55_plus_lowmid.json`;
3. aggregate each occupied coarse design cell by the mean calibrated apparent
   modulus of its occupied fine voxels;
4. invert the proxy modulus through the upstream cage design interpolation to
   obtain `design_cage_modulus_weighted`.

This is enough to define the first real full-structure FE input generated
through the calibration layer.

The older `fill_fraction * mean(E_eff(r))` proxy is still saved as a diagnostic
field, but it is not the primary replacement field because it penalizes porosity
twice: once inside the calibrated local Voronoi response and once again during
coarse-grid aggregation.

It is still only a first-pass proxy because:

- it remains scalar and isotropic;
- it ignores direction-dependent local stiffness tensors;
- it still clamps the high-modulus band;
- its first remote `force_1` full-structure comparison is not equivalent to the
  reference pseudo-density design.

The updated `force_1` comparison with
`fjw_iter017_replacement_design_variable_radius_seed55_plus_lowmid.npz` returned:

| metric | value |
|---|---:|
| replacement design sum | `260.90` |
| max displacement ratio | `1.9769` |
| `bo_sum_next` ratio | `0.9279` |
| `bone_s` mean ratio | `2.6569` |
| `bone_density_delta_sum` ratio | `-1.1277` |
| `bone_s` correlation | `0.3565` |
| `bone_density_delta` correlation | `-0.1132` |

This is a useful negative result. It shows that the edge-level inverse map is no
longer the main low-end bottleneck, but the first coarse scalar replacement
still does not reproduce the reference structure-level response.

```text
rho(x)
  -> E_target(x)
  -> target-modulus bands
  -> calibrated effective rod radius r_eff
  -> regenerated Voronoi skeleton
  -> full FE replacement validation
```

Seed density and joint calibration of `r` plus `n` are valid future
extensions, but they are not the first step.

## Deliverables

To consider the calibration layer minimally implemented, the repository should
gain:

1. a script that summarizes the `iter_017` target modulus field into
   representative calibration bands;
2. a script or module that generates local calibration specimens;
3. a script or module that measures local FE effective response;
4. a machine-readable calibration table;
5. a script that converts the variable-radius skeleton into an FE-ready
   replacement design field;
6. thesis text that explicitly states this layer and uses it as the basis for
   full-structure replacement validation.
