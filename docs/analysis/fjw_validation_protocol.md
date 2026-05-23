# FJW Validation Protocol

Validation is split into three layers.

## 1. Input Equivalence

Compare generated `.inp` files after normalizing:

- newline style
- empty lines
- comma spacing
- keyword case
- set ordering where the source format allows stable sorting

This layer checks mesh blocks, node sets, element sets, material buckets, boundary conditions, and load blocks.

## 2. Local Math Equivalence

Compare compact numeric fixtures:

- `bone_delta`
- `d_bone_delta`
- MMA `mmasub/subsolv`
- three-force objective aggregation
- `g2` and `d_g2`
- `Fv` active node ids, forces, dense checksum, and CLOAD text

The current committed fixture is `datasets/fjw_golden/minimal_mma/`.

## 3. Iteration Equivalence

For a completed run, compare:

- displacement vectors
- element energies
- `bone_s`
- `obj_bo`
- `bo_sum`
- `d_ob`
- `xmma`
- `delta`

If historical MATLAB + Abaqus outputs are unavailable, use Python-managed Abaqus real-run outputs as a new golden source and label the report accordingly. SfePy results should be compared against Abaqus results before any claim about replacing the historical solver.

## Report Rule

Every validation report must include pass/fail/warn status, max error for numeric arrays, and the exact file or array key that failed. Without golden data, the report must state that historical equivalence is not proven.
