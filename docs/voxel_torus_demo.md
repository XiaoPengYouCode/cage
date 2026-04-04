# Voxel Annular Cylinder Demo

新增了两个独立脚本：

- `experiments/voxel_demos/generate_voxel_torus_npz.py`：生成 `1000 x 1000 x 1000` 的体素圆环柱（XY 平面圆环，沿 Z 轴拉伸），并保存为 `npz`
- `experiments/voxel_demos/visualize_voxel_torus_npz.py`：从 `npz` 读取体素数据并做三维体素块可视化

## 生成 npz

```bash
uv run python experiments/voxel_demos/generate_voxel_torus_npz.py
```

默认输出：

```text
datasets/voxel/voxel_annular_cylinder_1000.npz
```

也可以调参数：

```bash
uv run python experiments/voxel_demos/generate_voxel_torus_npz.py \
  --grid-size 1000 \
  --outer-radius 360 \
  --inner-radius 180 \
  --chunk-depth 8
```

## 可视化 npz

```bash
uv run python experiments/voxel_demos/visualize_voxel_torus_npz.py --show
```

默认会额外保存一张图片：

```text
docs/assets/voxel_annular_cylinder_1000.png
```

如果担心显示太慢，可以降低显示分辨率：

```bash
uv run python experiments/voxel_demos/visualize_voxel_torus_npz.py \
  datasets/voxel/voxel_annular_cylinder_1000.npz \
  --max-display-size 80 \
  --show
```

## 说明

- 数组中空白体素为 `0`
- 圆环柱体素为 `1`
- 几何定义是：先在 XY 平面生成一个 2D 圆环，再沿 Z 轴方向贯通整个高度
- 生成脚本使用 `np.memmap + 分块`，避免直接在内存里一次性构建完整 `1000^3` 数组
- 可视化脚本使用降采样后的 `ax.voxels(...)` 进行体素块显示，不再是点云散点图
