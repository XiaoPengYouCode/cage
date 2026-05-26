# FJW Voronoi Skeleton Validation Plan

Date: 2026-05-26

## Objective

This document defines the validation experiment for the manufacturable
Voronoi skeleton generated from the FJW pseudo-density result.

The question is simple:

Can the final Voronoi skeleton preserve the same design intent as the original
pseudo-density field when we put it back into finite-element analysis?

The required conclusion is also simple:

- if the skeleton-based design produces similar displacement and target-bone
  stimulus responses, then the full pipeline is successful as a
  pseudo-density-to-manufacturable-geometry workflow;
- if not, the geometry conversion stage is introducing unacceptable drift.

## Current Evidence

What already exists in the repository:

1. Stable upstream pseudo-density checkpoint
   - `runs/fjw_optimize_real/iter_017`

2. Real downstream geometry artifacts
   - `outputs/fjw_optimize_real_iter017/fjw_iter017_density.npz`
   - `outputs/fjw_optimize_real_iter017/fjw_iter017_aligned_density_gamma1.npz`
   - `outputs/fjw_optimize_real_iter017/fjw_iter017_seeds_200_gamma1.npz`
   - `outputs/fjw_optimize_real_iter017/fjw_iter017_seeds_200_gamma1_cvt500.npz`
   - `outputs/fjw_optimize_real_iter017/fjw_iter017_voronoi_density.npz`
   - `outputs/fjw_optimize_real_iter017/fjw_iter017_voronoi_edges_density.npz`
   - `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_voxels_density.npz`
   - `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_density.stl`
   - `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_density.glb`

3. Existing capabilities relevant to the validation chain
   - `src/ct_reconstruction/voxelizer.py`
     - closed STL -> voxel occupancy
   - `src/fem_analysis/`
     - voxel/hexahedral workflow and three-force FE + remodeling solve
   - `src/matlab2stl_pipeline/`
     - pseudo-density -> CVT -> Voronoi -> skeleton geometry

## Missing Link

The missing link is not geometry generation.

The missing link is not only a comparison script either.

The real missing link is a calibration layer between the upstream
pseudo-density field and the downstream titanium Voronoi rod network.

The workflow we actually need is:

```text
pseudo-density x
  -> target equivalent modulus E_target(x)
  -> Voronoi geometry parameters theta(x)
  -> homogenized / calibrated equivalent modulus E_eff(theta)
  -> final skeleton STL/voxels
  -> aligned voxel occupancy in the original design domain
  -> calibrated FE replacement design
  -> rerun three-force forward solve
  -> compare displacement / bone_s / density update / objective-region response
  against the original pseudo-density design
```

At the moment, the repository does not yet contain this full calibrated loop.
The repository currently contains only the geometry loop and an exploratory
direct-replacement FE loop.

## Current Calibration Evidence

The repository now also has a first real local calibration FE pass executed on
`wuyinyun`.

Artifacts:

- `Post process/analysis/output/iter017_target_modulus_bands.json`
- `outputs/voronoi_radius_calibration_radius_only_remote_wide_v2/calibration_spec_manifest.json`
- `outputs/voronoi_radius_calibration_radius_only_remote_wide_v2/calibration_fe_results.json`
- `Post process/analysis/output/voronoi_radius_calibration_summary.json`
- `Post process/analysis/output/iter017_band_radius_lookup.json`

What this pass proves:

1. the local Voronoi specimen generation chain is physically scaled and can be
   executed end to end;
2. the local FE evaluator produces stable apparent moduli in a realistic range
   over the usable part of the sweep;
3. rod radius already gives a strong monotone calibration signal.

Representative measured values:

| radius | solid volume fraction | apparent modulus | status |
|---|---:|---:|---|
| `0.08 mm` | `0.0537` | `5.01 GPa` | stable |
| `0.10 mm` | `0.0764` | `16.74 GPa` | stable |
| `0.12 mm` | `0.0950` | `22.44 GPa` | stable |
| `0.20 mm` | `0.1999` | `47.47 GPa` | stable |
| `0.24 mm` | `0.2582` | `57.33 GPa` | stable |
| `0.30 mm` | `0.3183` | `64.99 GPa` | stable |

This is enough to justify the statement that the calibration layer is now an
implemented FE workflow, not only a thesis placeholder.

The same sweep also shows two unstable radii that must not be used for inverse
lookup:

| radius | apparent modulus | max displacement |
|---|---:|---:|
| `0.06 mm` | `5.12e-08 GPa` | `1.09e10 mm` |
| `0.16 mm` | `4.19e-11 GPa` | `3.79e12 mm` |

What it does **not** prove yet:

- the current six target-modulus bands still reuse one fixed local Voronoi
  topology and one fixed seed realization;
- therefore the current results establish `r -> E_eff`, but do not yet
  establish the full inverse map `E_target -> r_eff`.

The repository now also contains a first executable bandwise inverse lookup:

- `Post process/analysis/output/iter017_band_radius_lookup.json`

That lookup is intentionally simple:

- it uses piecewise-linear inverse interpolation on the current stable
  measured `r -> E_eff` curve;
- it clamps outside the stable sampled modulus range.

Current behavior is informative:

- bands `0` through `3` collapse to the minimum stable sampled radius
  `0.08 mm`;
- band `4` interpolates to about `0.1635 mm`;
- band `5` clamps to the current maximum stable sampled radius `0.30 mm`.

This means the current calibration support is still too narrow for the final
full-structure replacement experiment. The next FE campaign should therefore
expand the sampled radius range before full-skeleton regeneration.

## First Full-Structure Variable-Radius Skeleton

The repository now also has a first executable full-structure attachment pass
that pushes the current bandwise radius lookup back onto the real `iter_017`
Voronoi edge network.

Artifacts:

- `outputs/fjw_optimize_real_iter017/fjw_iter017_voronoi_edges_variable_radius.npz`
- `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_voxels_variable_radius.npz`
- `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_voxels_variable_radius_smoke.npz`
- `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_variable_radius_smoke.glb`
- `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_variable_radius_smoke.stl`

Supporting scripts:

- `Post process/analysis/build_iter017_variable_radius_edges.py`
- `Post process/analysis/build_iter017_variable_radius_skeleton.py`

Current attachment rule:

1. take each edge midpoint from `fjw_iter017_voronoi_edges_density.npz`;
2. sample `fjw_iter017_aligned_density_gamma1.npz` at that midpoint;
3. convert pseudo-density to `E_target` with the same modulus interpolation
   used by the upstream FJW workflow;
4. assign the current bandwise calibrated radius from
   `iter017_band_radius_lookup.json`;
5. voxelize the entire skeleton with one physical radius per edge.

This is already stronger than the earlier global-radius baseline because the
downstream skeleton is no longer controlled by one single dilation radius.

It still remains only a first-pass attachment rule:

- midpoint sampling is a practical first map, not yet a final homogenization
  rule over a finite local support volume;
- most edges still collapse to the current low-end radius because the measured
  `r -> E_eff` support is too narrow;
- the current replacement field is still a first-pass proxy because it uses
  `fill_fraction * mean(E_eff(r))` on occupied fine voxels, not yet a final
  local homogenization tensor.

## First FE-Ready Replacement Design

The repository now also contains a first FE-ready coarse-grid replacement
design built from the variable-radius skeleton:

- `outputs/fjw_optimize_real_iter017/fjw_iter017_replacement_design_variable_radius.npz`

Supporting script:

- `Post process/analysis/build_iter017_variable_radius_replacement_design.py`

Current construction rule:

1. map occupied fine skeleton voxels back to the original coarse FJW design
   grid;
2. convert each occupied fine voxel radius to calibrated equivalent modulus
   using the stable support stored in
   `Post process/analysis/output/iter017_band_radius_lookup.json`;
3. aggregate each coarse cell by
   `proxy_modulus = fill_fraction * mean(E_eff(r))`;
4. invert that proxy modulus back through the upstream cage design law to form
   `design_cage_modulus_weighted`.

This is already stronger than the earlier direct occupancy and fill-fraction
replacements because the FE input now carries a measured radius-to-modulus
signal.

What it still does **not** prove:

- the coarse proxy is scalar and isotropic, so it does not yet reproduce the
  full local stiffness tensor of the rod network;
- the lookup support is still heavily clamped outside the currently measured
  radius range;
- the real three-force comparison result for this variable-radius replacement
  has not yet been generated on `wuyinyun`.

## Physical Interpretation Baseline

In the current FJW implementation, pseudo-density is not just a gray-value
field for geometry sampling.

It already encodes a target stiffness demand through the cage interpolation:

```text
E_c(x) = E_c,min + E_c,0 * x^3
```

with clipping to `E_c,0` in the numerical implementation.

Therefore the downstream Voronoi stage cannot be justified by saying
"the skeleton was sampled from the density field".

The academically correct statement must be:

> the pseudo-density field defines a target equivalent mechanical field, and
> Voronoi geometry parameters must be calibrated so that the local equivalent
> stiffness of the titanium rod network matches that target field.

## Validation Hypothesis

The hypothesis to test is:

> After introducing a calibration layer from target equivalent modulus to
> Voronoi geometry parameters, the manufacturable Voronoi skeleton preserves
> the main mechanical stimulus pattern intended by the original pseudo-density
> optimization result.

This should be tested under the same three load cases already used in the main
FJW workflow.

## Experimental Inputs

### Reference design

Use the original `iter_017` pseudo-density / design-field result as the
reference.

Primary reference sources:

- `runs/fjw_optimize_real/iter_017/design_cage.npz`
- `runs/fjw_optimize_real/iter_017/*/forward_t0/forward_step.npz`
- `runs/fjw_optimize_real/iter_017/*/case_history.npz`

### Geometry replacement design

Use one of the two geometry representations below:

1. preferred for FE replacement:
   - `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_voxels_density.npz`

2. optional independent check:
   - `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_density.stl`
   - voxelized again by `ct_reconstruction.voxelizer`

The first path reduces one conversion step and is better for a stable baseline.
The second path is useful as an external consistency check on the STL export.

## Required Alignment Work

The skeleton geometry must be compared inside the same spatial frame as the
original FJW design domain.

This requires:

1. confirming origin and voxel size for the skeleton voxel grid;
2. mapping the skeleton occupancy back into the original design-domain frame;
3. defining how occupied skeleton voxels become FE design material values.

That last point needs a clear rule, but we now know a simple occupancy rule is
not enough.

The direct rules

- occupied skeleton voxel -> solid design value `1.0`
- empty voxel -> minimum design value `x_min`

and

- occupied coarse voxel -> local fill fraction

are both only exploratory replacement rules. They do not establish physical
equivalence unless the geometry-to-modulus calibration layer has already been
defined.

## Proposed Workflow

### Step 1. Define target modulus field from pseudo-density

Use the actual cage interpolation law from `fem_analysis` to convert the
reference pseudo-density field into a target equivalent modulus field.

### Step 2. Build a Voronoi calibration database

For a fixed family of local Voronoi rod networks, sweep:

- rod radius `r`
- local cell size `l`
- local seed density `n`
- topology family / orientation descriptors where needed

Then run local FE homogenization to obtain:

```text
E_eff = f(r, l, n, topology)
```

### Step 3. Build FE-ready replacement design

Generate an FE-ready, physically interpretable design array on the original
design grid from the calibrated skeleton geometry.

Outputs:

- `replacement_design_cage.npz`
- metadata recording source geometry, voxel alignment, target modulus rule,
  calibration rule, and mapping rule

### Step 4. Run three-force forward solve

Run the same three load cases using:

- original pseudo-density design
- replacement Voronoi skeleton design

The first comparison can focus on forward solve only.
Adjoint and optimization reruns are not required for the first equivalence test.

### Step 5. Collect comparison metrics

For each load case, compute:

- max displacement
- 95th percentile displacement
- mean displacement in the target region
- `bone_s` mean / median / 95th percentile / max
- target-region `bone_s` spatial map
- one-step `bone_density_delta` mean / total / 95th percentile
- objective-region total bone mass after one update

### Step 6. Summarize similarity

Compare skeleton-vs-reference by:

- absolute error
- relative error
- Pearson correlation for spatial maps where meaningful
- overlap / sign consistency for density increment maps

## Acceptance Criteria

The paper needs explicit judgment rules. A reasonable first-pass acceptance
criterion is:

1. local calibration specimens show `E_eff(theta)` close to `E_target(x)`;
2. displacement scale remains in the same order for all three load cases;
3. target-region `bone_s` spatial pattern remains strongly correlated;
4. one-step target-bone total response stays within a stated tolerance band;
5. no load case exhibits a qualitative failure such as complete stimulus loss,
   disconnected support, or singular-like behavior.

Exact numerical tolerances still need to be set after a dry run.
They should be stated in the experiment section, not guessed informally.

## Suggested Report Tables

### Table A. Geometry replacement metadata

- source geometry
- voxel size
- alignment transform
- occupied voxel count
- design-domain fill ratio
- connectivity status

### Table B. Load-case response comparison

- load case
- reference max displacement
- skeleton max displacement
- relative error
- reference target `bone_s` mean
- skeleton target `bone_s` mean
- relative error
- reference target bone total after one update
- skeleton target bone total after one update
- relative error

### Table C. Similarity summary

- load case
- target `bone_s` map correlation
- density increment map correlation
- pass / warn / fail

## Risks

### Risk 1. Spatial misalignment

If the skeleton occupancy is not aligned back to the original FE frame exactly,
the comparison becomes meaningless.

### Risk 2. Missing calibration layer

The original design is a continuous pseudo-density field interpreted through a
target-modulus interpolation law.

The replacement skeleton is a rod network with its own geometry-dependent
equivalent stiffness.

Without a calibration layer, any direct replacement rule is physically
ambiguous and can generate misleading FE conclusions.

### Risk 3. Resolution mismatch

The skeleton voxel grid is finer than the original design grid.
Downsampling or remapping may introduce error before FE even starts.

### Risk 4. Over-simplified replacement variables

Fill fraction alone is not the target quantity.
Two Voronoi structures with similar volume fraction can have very different
effective stiffness tensors.

The experiment must therefore avoid treating fill fraction as a complete
substitute for calibrated equivalent stiffness.

## Required Deliverables

To consider this validation actually completed, the repository should end up
with at least:

1. one script that builds the FE-ready replacement design from the skeleton;
2. one script or notebook family that builds the local calibration database;
3. one script that runs or reuses the three-force forward comparison;
4. one machine-readable summary file for the comparison metrics;
5. one figure or table family tied into `Post process/figure/`;
6. thesis text updated to state the calibration layer explicitly.

## Current Status

Status on 2026-05-26:

- geometry generation: implemented
- final skeleton artifacts: implemented
- direct binary replacement dry run: executed, but not physically sufficient
- local `r -> E_eff` calibration baseline: implemented
- bandwise inverse lookup: implemented
- variable-radius full-structure attachment: implemented
- first FE-ready `modulus_weighted` replacement design: implemented
- measured equivalence result: not yet generated

## Exploratory Dry Run Already Observed

An exploratory direct binary replacement run was executed for `force_1`.

Observed outcome:

- reference max displacement: `0.08379 mm`
- direct-binary replacement max displacement: `0.15290 mm`
- displacement ratio: `1.8248`
- reference `bone_density_delta_sum`: `+37.3364`
- direct-binary replacement `bone_density_delta_sum`: `-35.0384`
- `bone_s` correlation: `0.3576`
- `bone_density_delta` correlation: `-0.0818`

Interpretation:

This result should not be reported as "the Voronoi skeleton failed".
It should be reported as:

> direct occupancy replacement without a modulus-calibration layer causes
> severe physical mismatch and therefore cannot serve as the final academic
> validation protocol.

This boundary must stay explicit in the thesis until the experiment is run.
