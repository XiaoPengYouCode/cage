# matlab2stl_pipeline How To Start

`matlab2stl_pipeline` 是完整 cage pipeline 的下游几何节点。它从 681 `.mat` 伪密度快照出发，跑通：

```text
.mat -> density NPZ -> OBB aligned density -> seeds -> Voronoi -> skeleton voxels -> STL / GLB
```

在仓库根目录执行。

## 最小用法

```bash
uv run matlab2stl-pipeline run-pipeline \
  --mat datasets/681.mat \
  --output-dir outputs/matlab2stl_pipeline
```

默认会生成中间结果和最终网格。

## 500 个种子点，CVT 500 次，复制 GLB 到 viewer

```bash
uv run matlab2stl-pipeline run-pipeline \
  --mat datasets/681.mat \
  --output-dir outputs/matlab2stl_pipeline_seed500_cvt500 \
  --viewer-dir viewer/public/data \
  --num-seeds 500 \
  --cvt-iters 500
```

重点看：

- `outputs/matlab2stl_pipeline_seed500_cvt500/681_skeleton_density.stl`
- `outputs/matlab2stl_pipeline_seed500_cvt500/681_skeleton_cvt500.stl`
- `viewer/public/data/681_skeleton_density.glb`
- `viewer/public/data/681_skeleton_cvt500.glb`

## 常用参数

- `--num-seeds`：采样种子点数量
- `--cvt-iters`：Lloyd CVT 松弛次数，设为 `0` 表示关闭
- `--gamma`：密度映射到采样概率时的指数权重，默认 `1.0`
- `--subdivision`：骨架体素化时的细分倍数，默认 `10`
- `--dilation-radius`：细体素网格中的膨胀半径，默认 `3.0`
- `--mc-smooth-sigma`：Marching Cubes 前的高斯平滑强度，默认 `1.0`
- `--viewer-dir`：需要在 viewer 中预览时设为 `viewer/public/data`

## 查看命令帮助

```bash
uv run matlab2stl-pipeline run-pipeline --help
```
