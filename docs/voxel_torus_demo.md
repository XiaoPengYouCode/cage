# Voxel Annular Cylinder Demo

当前体素 demo 默认使用 `200 x 200 x 80` 的中空圆柱体，只负责生成拓扑采样链路的输入几何。

相关命令：

- `topopt-sampling generate-voxels`

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
- 几何定义是在 XY 平面先生成一个二维圆环，再沿 Z 方向整体拉伸
- 默认参数下外径正好等于 `200`，也就是会贴满整个 XY 范围
- 生成脚本使用 `np.memmap + 分块`，避免一次性构建完整三维数组
