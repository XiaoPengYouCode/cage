# matlab2stl_pipeline how-to-start

在仓库根目录执行。

## 最小用法

```bash
uv run matlab2stl-pipeline run-pipeline \
  --mat datasets/681.mat \
  --output-dir outputs/matlab2stl_pipeline
```

默认会完成 `.mat -> seed points -> Voronoi -> skeleton voxels -> STL/GLB` 全流程，并在 `output-dir` 下生成中间结果和最终网格。

## 500 个种子点，CVT 50 次，导出 STL

```bash
uv run matlab2stl-pipeline run-pipeline \
  --mat datasets/681.mat \
  --output-dir outputs/matlab2stl_pipeline_seed500_cvt50 \
  --num-seeds 500 \
  --cvt-iters 50
```

运行完成后，重点看这两个文件：

- `outputs/matlab2stl_pipeline_seed500_cvt50/681_skeleton_density.stl`
- `outputs/matlab2stl_pipeline_seed500_cvt50/681_skeleton_cvt50.stl`

其中 `681_skeleton_cvt50.stl` 是经过 50 次 Lloyd CVT 松弛后的结果。

## 常用参数

- `--num-seeds`：采样种子点数量。
- `--cvt-iters`：Lloyd CVT 松弛次数，设为 `0` 表示关闭。
- `--gamma`：密度映射到采样概率时的指数权重，默认 `1.0`。
- `--subdivision`：骨架体素化时的细分倍数，默认 `10`。
- `--dilation-radius`：细体素网格中的膨胀半径，默认 `3.0`。
- `--mc-smooth-sigma`：Marching Cubes 前的高斯平滑强度，默认 `1.0`。
- `--viewer-dir`：如果需要在 `viewer` 中预览，可把 `.glb` 复制到指定目录，例如 `viewer/public/data`。

## 查看命令帮助

```bash
uv run matlab2stl-pipeline run-pipeline --help
```
