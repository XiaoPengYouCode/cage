# 681 完整处理 Pipeline

本文档定义从原始 MATLAB `.mat` 文件出发，最终生成可在 viewer 中交互显示的泰森多边形骨架 mesh 的端到端工作流。

---

## 数据流总览

```
681.mat
  │
  ▼ Step 1
NPZ 体素网格（occupancy + 伪密度）
  │
  ▼ Step 2
GLB 文件 → viewer 预览（原始斜向长方体）
  │
  ▼ Step 3
概率密度场（density_milli → probability field）
  │
  ▼ Step 4
包围盒拟合 → 对齐长方体域（AABB-aligned bounding box）
  │
  ▼ Step 5
伪密度场空间变换 → 映射至对齐长方体
  │
  ▼ Step 6
种子点撒播（基于概率密度场）
  │
  ▼ Step 7
限制性泰森多边形划分（在对齐长方体域内）
  │
  ▼ Step 8
提取所有棱（内部 + 面边界 + 长方体棱）
  │
  ▼ Step 9
体素化形态学膨胀（每条棱 → 实体管道） + GLB 导出 → viewer
  │
  ▼ Step 10
连通实体的 mesh 重建（marching cubes）
  │
  ▼ Step 11（最终输出）
泰森多边形骨架 mesh（GLB + viewer 注册）
```

---

## 详细步骤

### Step 1 — 读取 681.mat，转换为标准 NPZ 体素格式

**输入：** `datasets/681.mat`

**目标：** 读取 MATLAB 格式的体素化拓扑优化结果，提取伪密度（pseudo-density）字段，转换为本仓库通用的 NPZ 格式。

**NPZ 标准字段（参照 `ct_reconstruction.npz_writer`）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `density_milli` | `uint16` (0..1000) | 伪密度，归一化到整数千分比 |
| `voxels` | `uint8` (0/1) | 二值占用网格（density > 阈值） |
| `grid_shape_xyz` | `int32[3]` | 体素网格维度 `[nx, ny, nz]` |
| `origin_m` | `float32[3]` | 原点坐标（米） |
| `voxel_size_xyz_m` | `float32[3]` | 各轴体素尺寸（米） |
| `shape_name` | `str` | `"681"` |
| `result_type` | `str` | `"mat_imported"` |
| `density_kind` | `str` | `"pseudo_density"` |

**输出：** `datasets/topopt/681_raw_density.npz`

**技术要点：**
- 使用 `scipy.io.loadmat` 读取 `.mat` 文件
- 检查 `.mat` 内变量名，通常为 `xPhys`、`x` 或 `rho` 等拓扑优化密度字段
- 体素尺寸、原点需从 `.mat` 元数据或约定参数中提取
- 密度值乘以 1000 后 clip 到 `[0, 1000]` 存为 `uint16`

**新增 CLI 子命令（建议）：**
```
topopt-sampling import-mat \
  --mat datasets/681.mat \
  --output datasets/topopt/681_raw_density.npz
```

---

### Step 2 — NPZ 体素 → GLB，在 viewer 中预览原始几何

**输入：** `datasets/topopt/681_raw_density.npz`

**目标：** 将体素占用网格导出为 GLB，注册到 viewer，用于确认几何正确性。

**现有工具：** `ct_reconstruction.glb_export`（exposed-face mesh 提取 → GLB）

**输出：**
- `viewer/public/data/681_raw.glb`

**viewer 注册：** 在 `viewer/src/main.js` 的模型列表中添加 `681_raw` 入口。

**CLI：**
```
ct-reconstruction voxelize \
  --npz datasets/topopt/681_raw_density.npz \
  --output-glb viewer/public/data/681_raw.glb
```

---

### Step 3 — 伪密度场 → 概率密度函数场

**输入：** `datasets/topopt/681_raw_density.npz`（字段 `density_milli`）

**目标：** 将伪密度（0..1）通过幂函数变换，生成用于种子点撒播的概率密度函数（PDF）场。

**变换方式（参照 `topopt_sampling.probability`）：**

```
p(x) = ρ(x)^γ / Σ ρ(x)^γ
```

其中 γ（gamma）为集中度参数，γ > 1 使高密度区域更集中，γ < 1 使分布更均匀。

**输出：** 保存在 `datasets/topopt/681_probability_field_gamma{γ}.npz`，字段为 `probability`（`float32`，归一化使总和为 1）。

**CLI：**
```
topopt-sampling compute-probability \
  --input datasets/topopt/681_raw_density.npz \
  --gamma 3.0 \
  --output datasets/topopt/681_probability_field_gamma3.npz
```

> 现有 `sample-seeds` 命令内含此步骤；如需独立保存概率场，需拆分或新增子命令。

---

### Step 4 — 斜向长方体包围盒拟合 → 生成对齐长方体域

**输入：** `datasets/topopt/681_raw_density.npz`（`voxels` 字段 + `origin_m` + `voxel_size_xyz_m`）

**目标：** 681 几何体是一个在体素空间中斜向放置的长方体。本步骤：

1. 在体素占用网格中提取实体区域的点云
2. 用主成分分析（PCA）或最小有向包围盒（OBB）算法拟合出该斜向长方体的局部坐标系（三个正交轴 `u, v, w` 和质心 `c`）
3. 确定 OBB 的三个半轴长度 `(L_u, L_v, L_w)`，对应真实的长宽高

**输出：** `datasets/topopt/681_obb.npz`，字段包括：

| 字段 | 类型 | 说明 |
|------|------|------|
| `center_m` | `float32[3]` | OBB 质心（米） |
| `axes` | `float32[3,3]` | 局部坐标系三个单位轴向量（行向量） |
| `half_extents_m` | `float32[3]` | 三个方向半轴长度（米） |
| `rotation_matrix` | `float32[3,3]` | 从 OBB 局部坐标到世界坐标的旋转矩阵 |

**技术要点：**
- 使用 `scipy` 或 `sklearn` 的 PCA，或 `scipy.spatial.ConvexHull` 后做旋转卡壳（rotating calipers）
- OBB 比 AABB 小，保证拟合紧密
- 保存旋转矩阵供 Step 5 使用

**CLI（建议新增）：**
```
topopt-sampling fit-obb \
  --input datasets/topopt/681_raw_density.npz \
  --output datasets/topopt/681_obb.npz
```

---

### Step 5 — 伪密度场空间变换 → 映射至对齐长方体

**输入：**
- `datasets/topopt/681_raw_density.npz`（原始体素密度）
- `datasets/topopt/681_obb.npz`（OBB 拟合结果）

**目标：** 将斜向放置的伪密度场通过逆旋转变换，重采样到一个轴对齐（AABB）的规整长方体体素网格中。变换后的长方体：
- 长宽高 = OBB 的 `2 * half_extents_m`
- 坐标轴与世界坐标系完全对齐（即 X/Y/Z 轴平行）
- 体素分辨率保持与原始 `voxel_size_xyz_m` 一致

**变换方法：**

```
对于新网格中每个体素位置 p_new（米）：
  p_local = p_new - center_m        # 平移到 OBB 中心
  p_old = R^T · p_local             # 逆旋转到原始空间
  p_voxel = (p_old - origin_m) / voxel_size_xyz_m  # 转到体素坐标
  用三线性插值 从原始 density_milli[p_voxel] 采样
```

**输出：** `datasets/topopt/681_aligned_density.npz`，格式与 Step 1 相同，但 `origin_m` 和 `axes` 已对齐世界坐标系。

**技术要点：**
- 使用 `scipy.ndimage.affine_transform` 实现高效体素重采样
- 需处理边界（插值超出范围的区域填 0）

**CLI（建议新增）：**
```
topopt-sampling align-density \
  --input datasets/topopt/681_raw_density.npz \
  --obb datasets/topopt/681_obb.npz \
  --output datasets/topopt/681_aligned_density.npz
```

---

### Step 6 — 种子点撒播（基于概率密度场）

**输入：** `datasets/topopt/681_aligned_density.npz`（对齐后的密度场）

**目标：** 在对齐后的长方体域内，根据概率密度函数场进行三维种子点的随机撒播，并可选地运行 CVT（Centroidal Voronoi Tessellation）迭代以优化种子点分布。

**现有工具：** `topopt_sampling.probability.sample_seeds_from_density` + `topopt_sampling.cli sample-seeds`

**参数：**
- `--num-seeds N`：种子点数量（典型值：16、32、50、100、200、300、500、2000）
- `--gamma γ`：PDF 幂次（典型值：1.0 或 3.0）
- `--cvt-iters K`：CVT 迭代次数（0 = 纯随机，> 0 = Lloyd 松弛）

**输出：** `datasets/topopt/681_aligned_seed_points_{N}_gamma{γ}_cvt{K}.npz`

**CLI：**
```
topopt-sampling sample-seeds \
  --input datasets/topopt/681_aligned_density.npz \
  --num-seeds 100 \
  --gamma 3.0 \
  --cvt-iters 500 \
  --output datasets/topopt/681_aligned_seed_points_100_gamma3_cvt500.npz
```

---

### Step 7 — 限制性泰森多边形划分（Restricted Voronoi）

**输入：**
- `datasets/topopt/681_aligned_density.npz`（定义域：对齐长方体）
- `datasets/topopt/681_aligned_seed_points_100_gamma3_cvt500.npz`（种子点）

**目标：** 在对齐长方体域内构建限制性泰森多边形图，每个细胞为凸多面体，域边界上的细胞需与长方体面正确裁剪。

**实现方式：直接使用 `scipy.spatial.Voronoi`**，不复用现有环柱体实现。

**方法：**

```
1. 调用 scipy.spatial.Voronoi(seed_points) 得到全局 Voronoi 图
   - 输出：vertices（Voronoi 顶点坐标）、ridge_vertices（每条脊线的两顶点）
            ridge_points（每条脊线对应的两个种子点）、point_region、regions

2. 用 "镜像点" 技巧处理开放区域（无穷远顶点 index = -1）：
   - 在长方体 6 个面上对所有种子点做镜像，将镜像点加入 Voronoi 输入
   - 这样边界附近的细胞不再有无穷远顶点，所有 ridge 均有限

3. 对每个种子点对应的 Voronoi 区域：
   - 收集该区域所有 Voronoi 顶点
   - 用长方体 6 个半空间约束（scipy.spatial.ConvexHull 或 HalfspaceIntersection）
     对区域顶点做裁剪，得到最终凸多面体细胞

4. 保存每个细胞的 (vertices, faces) 对
```

**输出：** `datasets/topopt/681_voronoi_diagram.npz`，字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `cell_vertices` | `object array` | 每个细胞的顶点坐标 `float32[V_i, 3]` |
| `cell_faces` | `object array` | 每个细胞的面（顶点索引列表） |
| `seed_points` | `float32[N, 3]` | 种子点（与细胞一一对应） |
| `box_min` | `float32[3]` | 长方体最小角 |
| `box_max` | `float32[3]` | 长方体最大角 |

**关键依赖：** `scipy.spatial.Voronoi`、`scipy.spatial.ConvexHull`

**CLI（新增）：**
```
topopt-sampling build-box-voronoi \
  --density-npz datasets/topopt/681_aligned_density.npz \
  --seeds-npz datasets/topopt/681_aligned_seed_points_100_gamma3_cvt500.npz \
  --output datasets/topopt/681_voronoi_diagram.npz
```

---

### Step 8 — 提取泰森多边形所有棱

**输入：** 上一步的泰森多边形图（`681_voronoi_diagram.npz` 或内存对象）

**目标：** 从 Voronoi 图中提取所有唯一的棱（line segments），包括三类：

| 类型 | 说明 |
|------|------|
| **内部棱** | 三个或三个以上 Voronoi 细胞共享的棱，位于长方体内部 |
| **面边界棱** | 位于长方体某一面上，两个细胞在该面上的分界线段 |
| **长方体棱上的棱** | 长方体 12 条棱各自被 Voronoi 细胞分割成若干线段 |

**输出：** 棱集合，每条棱为两个端点 `(p0, p1)`（`float32[3]`），保存为：
- `datasets/topopt/681_voronoi_edges.npz`，字段 `edges`（`float32[E, 2, 3]`）

**技术要点：**
- 遍历所有 Voronoi 细胞的面，收集所有边，去重（按端点排序后哈希）
- 区分三类棱时依据端点是否落在长方体面/棱上（数值容差判断）
- 现有 `hybrid_exact_brep.py` 中有类似的边提取逻辑可参考

**CLI（建议新增）：**
```
topopt-sampling extract-voronoi-edges \
  --voronoi datasets/topopt/681_voronoi_diagram.npz \
  --output datasets/topopt/681_voronoi_edges.npz
```

---

### Step 9 — 体素化形态学膨胀（棱 → 实体管道）+ GLB 导出

**输入：**
- `datasets/topopt/681_voronoi_edges.npz`（棱集合）
- `datasets/topopt/681_aligned_density.npz`（用于获取体素网格参数）

**目标：**
1. 在对齐长方体对应的体素网格上，将每条棱光栅化为细线（Bresenham 三维直线算法）
2. 对所有棱的体素集合执行形态学膨胀（`scipy.ndimage.binary_dilation` 或球形结构元素），膨胀半径对应设计的杆件直径
3. 最终得到一个连通的实体体素网格（长方体域内的泰森多边形骨架实体）
4. 将体素网格导出为 GLB，注册到 viewer

**输出：**
- `datasets/topopt/681_skeleton_voxels.npz`（体素骨架）
- `viewer/public/data/681_skeleton.glb`（viewer 可视化）

**viewer 注册：** 在 `viewer/src/main.js` 添加 `681_skeleton` 模型入口。

**技术要点：**
- 使用 `ct_reconstruction.glb_export` 中的 exposed-face 方法导出体素 GLB
- 膨胀半径（杆件半径）应作为可配置参数（单位：体素数或毫米）
- 棱的光栅化使用三维 Bresenham 或沿棱均匀采样体素坐标

**CLI（建议新增）：**
```
topopt-sampling voxelize-skeleton \
  --edges datasets/topopt/681_voronoi_edges.npz \
  --density-npz datasets/topopt/681_aligned_density.npz \
  --dilation-radius 2 \
  --output-npz datasets/topopt/681_skeleton_voxels.npz \
  --output-glb viewer/public/data/681_skeleton.glb
```

---

### Step 10 — 骨架体素 mesh 重建

**输入：** `datasets/topopt/681_skeleton_voxels.npz`（连通实体体素网格）

**目标：** 对实体体素骨架执行等值面提取（marching cubes），生成光滑的三角面网格，再进行网格简化和平滑处理。

**方法：**
- 等值面提取：`skimage.measure.marching_cubes`（level = 0.5）
- 可选后处理：
  - `open3d` 或 `trimesh` 做 Laplacian 平滑（减少体素阶梯感）
  - `open3d.geometry.TriangleMesh.simplify_quadric_decimation` 做面片简化

**输出：**
- `datasets/topopt/681_scaffold_mesh.npz`（顶点 + 面片）
- `viewer/public/data/681_scaffold.glb`（最终结果 GLB）

**viewer 注册：** 在 `viewer/src/main.js` 添加 `681_scaffold` 模型入口。

**CLI（建议新增）：**
```
topopt-sampling reconstruct-mesh \
  --skeleton-npz datasets/topopt/681_skeleton_voxels.npz \
  --output-npz datasets/topopt/681_scaffold_mesh.npz \
  --output-glb viewer/public/data/681_scaffold.glb \
  --smooth-iters 5
```

---

## 中间文件汇总

| 文件 | 步骤 | 说明 |
|------|------|------|
| `681_raw_density.npz` | Step 1 | MAT 导入的原始体素密度 |
| `viewer/public/data/681_raw.glb` | Step 2 | 原始斜向几何预览 |
| `681_probability_field_gamma3.npz` | Step 3 | 概率密度场 |
| `681_obb.npz` | Step 4 | OBB 拟合结果（旋转矩阵 + 尺寸） |
| `681_aligned_density.npz` | Step 5 | 变换至对齐长方体的密度场 |
| `681_aligned_seed_points_100_gamma3_cvt500.npz` | Step 6 | 种子点 |
| `681_voronoi_diagram.npz` | Step 7 | 泰森多边形图 |
| `681_voronoi_edges.npz` | Step 8 | 所有棱 |
| `681_skeleton_voxels.npz` | Step 9 | 骨架体素网格 |
| `viewer/public/data/681_skeleton.glb` | Step 9 | 骨架体素 GLB 预览 |
| `681_scaffold_mesh.npz` | Step 10 | 最终 mesh（顶点+面片） |
| `viewer/public/data/681_scaffold.glb` | Step 10 | 最终骨架 mesh GLB |

---

## 新增 CLI 子命令汇总

以下命令需在 `src/topopt_sampling/cli.py` 中新增（或适配现有命令）：

| 命令 | 状态 | 对应步骤 |
|------|------|---------|
| `import-mat` | **待实现** | Step 1 |
| `compute-probability` | 现有逻辑拆分 | Step 3 |
| `fit-obb` | **待实现** | Step 4 |
| `align-density` | **待实现** | Step 5 |
| `sample-seeds` | **现有**（需适配 aligned NPZ） | Step 6 |
| `build-box-voronoi` | **待实现**（现有针对环形柱体） | Step 7 |
| `extract-voronoi-edges` | **待实现** | Step 8 |
| `voxelize-skeleton` | **待实现** | Step 9 |
| `reconstruct-mesh` | **待实现** | Step 10 |

---

## 实现优先级建议

1. **Step 1**（`import-mat`）：解锁所有后续步骤
2. **Step 2**（GLB 预览）：快速验证导入结果
3. **Step 4 + Step 5**（OBB 拟合 + 密度变换）：核心几何变换，需仔细验证
4. **Step 6**（种子点撒播）：复用现有 `sample-seeds`，适配即可
5. **Step 7**（Voronoi）：适配长方体域（最复杂）
6. **Step 8–10**（骨架体素化 → mesh）：顺序推进

---

## 技术依赖参考

| 功能 | 推荐库 |
|------|--------|
| MAT 文件读取 | `scipy.io.loadmat` |
| OBB 拟合 | `sklearn.decomposition.PCA` 或 `scipy.spatial` |
| 体素重采样 | `scipy.ndimage.affine_transform` |
| Voronoi（凸多面体） | `scipy.spatial.Voronoi` + 镜像点技巧 + `ConvexHull` 裁剪 |
| 棱光栅化 | 三维 Bresenham（自实现） or 均匀采样 |
| 形态学膨胀 | `scipy.ndimage.binary_dilation` |
| Marching Cubes | `skimage.measure.marching_cubes` |
| 网格后处理 | `trimesh` 或 `open3d` |
| GLB 导出 | `ct_reconstruction.glb_export`（现有） |
