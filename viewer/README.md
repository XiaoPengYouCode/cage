# Three.js Viewer

`viewer/` 用来交互查看 pipeline 导出的 GLB。它当前可查看 annular-cylinder restricted Voronoi shell、681 downstream geometry、FJW reference voxel blocks，以及其他已注册模型。

## Setup

```bash
cd viewer
pnpm install
pnpm dev --host 127.0.0.1
```

默认地址：

- `http://127.0.0.1:5173/`

常用模型参数：

- `?model=voronoi`
- `?model=681_raw`
- `?model=681_skeleton_density`
- `?model=681_skeleton_cvt500`
- `?model=fjw_reference`

## Export annular-cylinder shell GLB

从仓库根目录执行：

```bash
uv run topopt-sampling export-threejs-shell-glb \
  datasets/topopt/seed_probability_mapping_2000.npz \
  --xy-size 200 \
  --z-size 80 \
  --outer-radius 100 \
  --inner-radius 50 \
  --output-json viewer/public/data/hybrid_exact_shell_2000.glb
```

## Export 681 geometry GLB

```bash
uv run matlab2stl-pipeline run-pipeline \
  --mat datasets/681.mat \
  --output-dir outputs/matlab2stl_pipeline_seed500_cvt500 \
  --viewer-dir viewer/public/data \
  --num-seeds 500 \
  --cvt-iters 500
```

导出后可打开：

- `http://127.0.0.1:5173/?model=681_raw`
- `http://127.0.0.1:5173/?model=681_skeleton_density`
- `http://127.0.0.1:5173/?model=681_skeleton_cvt500`

## Notes

`topopt_sampling` 的 annular-cylinder shell GLB 导出包含圆柱接缝处理：

- cylinder face 会选择更稳定的 seam atlas 再展开三角化
- shell GLB 导出会对共享 seam 边做 canonical snapping
- cylinder / plane / cap 交界会生成 seam strip，降低接缝裂缝与破面风险
