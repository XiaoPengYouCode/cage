# FJW Reference Voxel Export

这次补的是一件很具体的事情：把 `references/fjw_work/` 里的历史 Abaqus 六面体参考网格，整理成当前仓库更容易复用的标准体素 `.npz`。

脚本入口：

- [scripts/export_fjw_reference_voxels.py](/Users/flamingo/Projects/cage/scripts/export_fjw_reference_voxels.py)

核心实现：

- [src/fem_analysis/fjw_reference.py](/Users/flamingo/Projects/cage/src/fem_analysis/fjw_reference.py)

## 为什么能直接转成体素

这份 reference 不是任意非结构网格。

从 `nod_coo.mat` 和 `ele_nod.mat` 读出来以后，可以验证出：

- 节点坐标是规则整数晶格
- 范围是 `1..153 x 1..132 x 1..135`
- 每个单元都是跨度恰好为 `1 x 1 x 1` 的八节点六面体
- 每个单元都能唯一对应到一个 0-based 体素坐标

所以这里做的不是重采样，也不是几何近似，而是单元到体素的一一重建。

输出网格尺寸是：

- `grid_shape_xyz = [152, 131, 134]`

## 输出字段

导出的 `.npz` 以仓库现有体素格式为主，包含：

- `voxels`
- `density_milli`
- `material_id`
  - `0 = cor`
  - `1 = tra`
  - `2 = cage`
  - `-1 = void`
- `design_mask`
- `objective_mask`
- `element_id`
- `origin_m`
- `voxel_size_xyz_m`
- `grid_shape_xyz`

其中：

- `design_mask` 对应 `desi_ele`
- `objective_mask` 对应 `obj_ele`
- `element_id` 保留原始 Abaqus 单元编号，方便后续把 Python 结果映射回历史 reference

## 默认物理尺寸

默认体素边长是 `0.6 mm`。

这个默认值来自历史脚本 [references/fjw_work/edit_nodesi_inp.m](/Users/flamingo/Projects/cage/references/fjw_work/edit_nodesi_inp.m)，里面当前实际启用的是：

- `nod_coo_mm = (nod_coo - 1) * 0.6`

如果后面你要做“同一拓扑、不同物理尺度”的实验，可以在导出时改 `--voxel-size-mm`，但默认先保持和 reference 一致。

## 用法

```bash
uv run python scripts/export_fjw_reference_voxels.py
```

也可以显式指定输出：

```bash
uv run python scripts/export_fjw_reference_voxels.py \
  --output datasets/topopt/fjw_reference_fem_voxels.npz
```
