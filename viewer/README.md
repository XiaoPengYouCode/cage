# Three.js Viewer

## Setup

```bash
cd viewer
pnpm install
pnpm dev --host 127.0.0.1
```

Default local URL:

- `http://127.0.0.1:5173/`

Default data file:

- `public/data/hybrid_exact_shell_2000.glb`

## Export scene data

From repo root:

```bash
uv run topopt-sampling export-threejs-shell-glb \
  datasets/topopt/seed_probability_mapping_2000.npz \
  --xy-size 200 \
  --z-size 80 \
  --outer-radius 100 \
  --inner-radius 50 \
  --output-json viewer/public/data/hybrid_exact_shell_2000.glb
```

## Notes

Current GLB export includes seam handling for cylindrical shell faces:

- choose a more stable cylinder seam atlas before triangulation
- snap shared seam-edge samples across neighboring faces
- add seam strips on cylinder / plane / cap transitions

This is the recommended path to inspect seam continuity and exploded shell cells interactively.
