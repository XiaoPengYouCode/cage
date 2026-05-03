# 681 Downstream Geometry Pipeline

这份文档说明当前已经实现的 681 `.mat` 到 Voronoi 骨架 STL / GLB 流程。它是完整 cage pipeline 的下游几何节点，用来验证“伪密度场 -> seed points -> Voronoi 骨架 -> mesh 导出”这段链路。

完整项目主线仍然是：

```text
fem_analysis FJW workflow
  -> final pseudo-density / design density
  -> standardized density NPZ
  -> topopt_sampling probability + seeds
  -> Voronoi / restricted Voronoi
  -> helix rods or scaffold mesh
  -> STL / GLB / viewer / analysis
```

`datasets/681.mat` 是历史或实验密度快照。它可以直接进入下游几何节点，方便调试 OBB 对齐、seed sampling、Voronoi 划分和 mesh 导出。

## Command

最小用法：

```bash
uv run matlab2stl-pipeline run-pipeline \
  --mat datasets/681.mat \
  --output-dir outputs/matlab2stl_pipeline
```

带 viewer 复制、500 个 seed 和 CVT 500 次：

```bash
uv run matlab2stl-pipeline run-pipeline \
  --mat datasets/681.mat \
  --output-dir outputs/matlab2stl_pipeline_seed500_cvt500 \
  --viewer-dir viewer/public/data \
  --num-seeds 500 \
  --cvt-iters 500
```

查看参数：

```bash
uv run matlab2stl-pipeline run-pipeline --help
```

## Implemented Data Flow

当前实现位于 `src/matlab2stl_pipeline/cli.py`，运行 `run-pipeline` 会依次执行：

```text
datasets/681.mat
  |
  v
681_raw_density.npz
  |
  v
681_raw.glb
  |
  v
681_obb.npz
  |
  v
681_aligned_density_gamma{gamma}.npz
  |
  v
681_seeds_{N}_gamma{gamma}.npz
  |
  +--> optional 681_seeds_{N}_gamma{gamma}_cvt{iters}.npz
  |
  v
681_voronoi_{tag}.npz
  |
  v
681_voronoi_cells_{tag}.glb
  |
  v
681_voronoi_edges_{tag}.npz
  |
  v
681_skeleton_voxels_{tag}.npz
  |
  v
681_skeleton_{tag}.glb + 681_skeleton_{tag}.stl
```

`tag` 通常是：

- `density`：直接由密度概率采样得到的 seed set
- `cvt{iters}`：经过 Lloyd CVT 松弛后的 seed set

## Step Details

### Step 1. Import `.mat` to standard density NPZ

实现：`src/matlab2stl_pipeline/mat_importer.py`

输入：

- `datasets/681.mat`

默认读取变量：

- `cage_3D1`

输出：

- `681_raw_density.npz`

核心字段：

- `density_milli`
- `voxels`
- `grid_shape_xyz`
- `origin_m`
- `voxel_size_xyz_m`
- `shape_name`
- `result_type`
- `density_kind`

当前 681 数据默认体素尺寸为 `0.4 mm`。

### Step 2. Raw density GLB preview

实现：`src/matlab2stl_pipeline/cli.py` 中的 `_export_raw_glb`

输出：

- `681_raw.glb`

如果传入 `--viewer-dir viewer/public/data`，该文件会复制到 viewer 数据目录。当前 viewer 已经注册 `?model=681_raw`。

### Step 3. OBB fitting

实现：`src/matlab2stl_pipeline/obb_aligner.py`

输出：

- `681_obb.npz`

核心字段：

- `center_voxel`
- `axes`
- `half_extents_voxel`
- `half_extents_m`
- `rotation_matrix`
- `voxel_size_xyz_m`

这一步用 `density_milli > 0` 的体素点云做 PCA，拟合 681 密度域的有向包围盒。

### Step 4. Density alignment and probability field

实现：`src/matlab2stl_pipeline/obb_aligner.py`

输出：

- `681_aligned_density_gamma{gamma}.npz`

核心字段：

- `density_milli`
- `voxels`
- `probability_field`
- `grid_shape_xyz`
- `origin_m`
- `voxel_size_xyz_m`
- `restore_R`
- `restore_t`

这一步把斜向密度场重采样到 OBB 对齐坐标系，并同步保存概率场。`restore_R` 和 `restore_t` 用于把后续 mesh 变回原始姿态。

### Step 5. Seed sampling

实现：`src/matlab2stl_pipeline/seed_sampler.py`

输出：

- `681_seeds_{N}_gamma{gamma}.npz`

核心字段：

- `seed_points`
- `seed_points_m`
- `num_seeds`
- `gamma`

采样逻辑和主 `topopt_sampling` 节点一致：

```text
p = rho^gamma / sum(rho^gamma)
```

### Step 6. Optional CVT relaxation

实现：`src/matlab2stl_pipeline/cvt_relaxation.py`

输出：

- `681_seeds_{N}_gamma{gamma}_cvt{iters}.npz`

`--cvt-iters 0` 会关闭这一步。

### Step 7. Box-restricted Voronoi

实现：`src/matlab2stl_pipeline/box_voronoi.py`

输出：

- `681_voronoi_{tag}.npz`
- `681_voronoi_cells_{tag}.glb`

当前 681 pipeline 使用 OBB 对齐后的长方体域，因此这里构建的是 box-restricted Voronoi。环柱体的 exact restricted Voronoi 能力仍在 `src/topopt_sampling/` 中维护。

### Step 8. Edge extraction

实现：`src/matlab2stl_pipeline/box_voronoi.py`

输出：

- `681_voronoi_edges_{tag}.npz`

这一步从每个 Voronoi cell 的 face loops 中收集唯一边，作为后续骨架体素化的输入。

### Step 9. Skeleton voxelization

实现：`src/matlab2stl_pipeline/skeleton_voxelizer.py`

输出：

- `681_skeleton_voxels_{tag}.npz`

核心参数：

- `--subdivision`：细体素网格细分倍数
- `--dilation-radius`：细体素中的形态学膨胀半径

### Step 10. Mesh reconstruction and export

实现：`src/matlab2stl_pipeline/skeleton_voxelizer.py`

输出：

- `681_skeleton_{tag}.glb`
- `681_skeleton_{tag}.stl`

这一步通过 marching cubes 从骨架体素重建三角面网格，并使用 `restore_R` / `restore_t` 还原到原始姿态。

## Common Parameters

- `--num-seeds`：采样种子点数量，默认 `200`
- `--gamma`：密度映射到采样概率时的指数权重，默认 `1.0`
- `--cvt-iters`：Lloyd CVT 松弛次数，默认 `500`
- `--subdivision`：骨架体素化细分倍数，默认 `10`
- `--dilation-radius`：细体素网格中的膨胀半径，默认 `3.0`
- `--mc-smooth-sigma`：Marching Cubes 前的高斯平滑强度，默认 `1.0`
- `--viewer-dir`：需要复制 GLB 给 viewer 时设为 `viewer/public/data`

## Viewer

启动 viewer：

```bash
cd viewer
pnpm install
pnpm dev
```

常用模型：

- `http://127.0.0.1:5173/?model=681_raw`
- `http://127.0.0.1:5173/?model=681_skeleton_density`
- `http://127.0.0.1:5173/?model=681_skeleton_cvt500`

## Relationship To Main Pipeline

681 pipeline 是当前下游几何实现的稳定入口。它直接从 `.mat` 读伪密度，适合验证 OBB 对齐、seed sampling、box Voronoi、骨架体素化和 STL / GLB 导出。

完整 cage pipeline 的上游密度来源应来自 `fem_analysis` 的 FJW workflow。把 FJW 输出接入下游时，优先整理成同样的标准密度 NPZ，再复用这里已经跑通的几何节点。
