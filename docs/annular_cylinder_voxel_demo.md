# Annular Cylinder Voxel Demo

这个 demo 是 `topopt_sampling` 节点的 smoke input。它生成 `200 x 200 x 80` 中空圆柱体素域，再由假密度生成、seed sampling 和 overview 渲染继续验证下游链路。

它不代表完整 pipeline 的上游密度来源。正式密度来源应来自 `fem_analysis` 的 FJW workflow 输出。

## 生成体素输入

```bash
uv run topopt-sampling generate-voxels \
  --output datasets/voxel/voxel_annular_cylinder_200x200x80.npz \
  --xy-size 200 \
  --z-size 80 \
  --outer-radius 100 \
  --inner-radius 50
```

默认输出：

```text
datasets/voxel/voxel_annular_cylinder_200x200x80.npz
```

## 说明

- 空白体素值为 `0`
- 几何体素值为 `1`
- 几何定义是在 XY 平面生成二维圆环，再沿 Z 方向整体拉伸
- 默认参数下外径等于 `200`，会贴满整个 XY 范围
- 生成脚本使用 `np.memmap + 分块`，避免一次性构建完整三维数组
