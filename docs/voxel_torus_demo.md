# Voxel Annular Cylinder Demo

当前体素 demo 默认使用 `200 x 200 x 80` 的中空圆柱体。

相关脚本：

- `experiments/voxel_demos/generate_voxel_torus_npz.py`
- `experiments/voxel_demos/visualize_voxel_torus_npz.py`

## 生成体素输入

```bash
uv run python experiments/voxel_demos/generate_voxel_torus_npz.py \
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

## 可视化体素输入

```bash
uv run python experiments/voxel_demos/visualize_voxel_torus_npz.py \
  datasets/voxel/voxel_annular_cylinder_200x200x80.npz \
  --show
```

## 说明

- 空白体素值为 `0`
- 几何体素值为 `1`
- 几何定义是在 XY 平面先生成一个二维圆环，再沿 Z 方向整体拉伸
- 默认参数下外径正好等于 `200`，也就是会贴满整个 XY 范围
- 生成脚本使用 `np.memmap + 分块`，避免一次性构建完整三维数组
- 可视化脚本会自动降采样，再用 `ax.voxels(...)` 画三维体素块
