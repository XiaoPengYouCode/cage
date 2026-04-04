# Helix Voronoi + Topopt Backfill Workspace

![Voronoi Mixed Cage](docs/assets/voronoi_mixed_preview.png)

这个仓库现在改成了 **uv workspace + 两个独立 editable package**。

## 这个仓库现在在做什么

整个仓库围绕两层能力组织：

1. **`helix-voronoi`**
   - 面向几何生成与分析
   - 负责 Voronoi 单胞、螺旋杆/直杆实体化、渲染、STL 导出、模量分析

2. **`topopt-backfill`**
   - 面向工作流编排
   - 负责把拓扑优化得到的 density NPZ 转成 seed cloud，并进一步生成 template backfill 数据

你可以把它理解成：
- `topopt-backfill` 解决“**从拓扑优化结果到可用 Voronoi 填充输入**”
- `helix-voronoi` 解决“**Voronoi / helix 几何如何生成、渲染、分析**”

## 包划分

### 1) `helix-voronoi`
语义：**螺旋杆 Voronoi 单胞生成与分析工具包**

位置：`packages/helix-voronoi`

职责：
- Voronoi 几何生成与边提取
- 圆柱杆 / 螺旋杆实体化
- 3D 渲染
- STL 导出
- 单胞模量分析

主要源码：
- `packages/helix-voronoi/src/helix_voronoi/pipeline.py`
- `packages/helix-voronoi/src/helix_voronoi/voronoi.py`
- `packages/helix-voronoi/src/helix_voronoi/helix.py`
- `packages/helix-voronoi/src/helix_voronoi/rods.py`
- `packages/helix-voronoi/src/helix_voronoi/rendering.py`
- `packages/helix-voronoi/src/helix_voronoi/analysis/`
- `packages/helix-voronoi/src/helix_voronoi/cli.py`

CLI：
- `uv run helix-voronoi`
- `uv run helix-voronoi modulus ...`
- `uv run helix-voronoi export-helix ...`
- `uv run helix-voronoi export-mixed ...`

### 2) `topopt-backfill`
语义：**拓扑优化密度结果到 Voronoi 模板回填的工作流包**

位置：`packages/topopt-backfill`

职责：
- density NPZ 概率化
- 从密度场采样 Voronoi seeds
- 候选点压缩与 FPS 代表点选择
- 3x3x3 block 聚合
- 模板 ID 回填
- 输出 backfill NPZ

主要源码：
- `packages/topopt-backfill/src/topopt_backfill/probability.py`
- `packages/topopt-backfill/src/topopt_backfill/selection.py`
- `packages/topopt-backfill/src/topopt_backfill/templates.py`
- `packages/topopt-backfill/src/topopt_backfill/workflows.py`
- `packages/topopt-backfill/src/topopt_backfill/cli.py`

CLI：
- `uv run topopt-backfill sample-seeds ...`
- `uv run topopt-backfill backfill-templates ...`

## 二者关系

可以把它理解成：

- `topopt-backfill` 是**上游工作流 package**
- `helix-voronoi` 是**下游几何/分析 package**
- `topopt-backfill` 会复用 `helix-voronoi` 的 Voronoi 几何能力

也就是说，**螺旋杆 Voronoi 生成** 是整体链路中的一个子工具，而不是整个项目唯一主语。

## 运行

默认渲染：

```bash
uv run helix-voronoi
```

或：

```bash
uv run python main.py
```

弹出 matplotlib 窗口：

```bash
uv run python main.py --show
```

调整种子点数量和三组随机种子：

```bash
uv run python main.py --num-seeds 10 --row-seeds 116 55 49
```

输出图片默认保存为：

```text
docs/assets/voronoi_cube_3d.png
```

## `topopt-backfill` 示例

### 1. 从拓扑优化密度结果采样种子

```bash
uv run topopt-backfill sample-seeds \
  datasets/topopt/fake_density_annular_cylinder_full.npz \
  --num-seeds 100000 \
  --output-npz datasets/topopt/seed_probability_mapping_100k.npz
```

### 2. 做模板化 Voronoi 回填

```bash
uv run topopt-backfill backfill-templates \
  datasets/topopt/fake_density_annular_cylinder_full.npz \
  datasets/topopt/seed_probability_mapping_100k.npz \
  --output-npz datasets/topopt/template_backfilled_helix_voronoi.npz
```

## `helix-voronoi` 示例

导出 STL：

```bash
uv run helix-voronoi export-helix --seed 55 --stl-output docs/assets/voronoi_helix_seed55.stl
uv run helix-voronoi export-mixed --seed 55 --stl-output docs/assets/voronoi_mixed.stl
```

模量分析：

```bash
uv run helix-voronoi modulus --seed 55 --style both
```

只验证配置、不启动求解：

```bash
uv run helix-voronoi modulus --seed 55 --style both --dry-run
```

当前模量分析后端为 `SfePy`，流程是：
- 先把直杆或螺旋杆单胞体素化
- 再转成规则 `Hex8` 六面体网格
- 最后在 `SfePy` 中施加上下完全粘结压板位移边界，求 `Z` 向等效模量

## 测试

```bash
uv run python -m unittest discover -s tests -v
```

## 仓库结构

```text
packages/
  helix-voronoi/      # 螺旋杆 Voronoi 单胞生成与分析 package
  topopt-backfill/    # 拓扑优化结果到 backfill 的 workflow package
experiments/          # 实验脚本，不承载正式主流程
  topopt_backfill/
  voxel_demos/
datasets/             # 可复用 npz 数据资产
  topopt/
  voxel/
docs/assets/          # 文档展示图片与少量展示产物
```

## 其它

- `packages/` 是正式交付边界
- `experiments/` 只保留还没有产品化的实验脚本
- `datasets/` 存放可复用 `.npz` 数据，不再把这类数据混放在 `docs/assets/`
