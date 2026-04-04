# Helix Voronoi + Topopt Backfill

![Voronoi mixed preview](docs/assets/voronoi_mixed_preview.png)

一个围绕 **Voronoi 螺旋杆结构生成** 与 **拓扑优化结果回填** 的 Python 项目。

当前仓库采用标准 `src/` layout，在同一个代码库里维护两个边界清晰的 package：

- `helix_voronoi`：负责几何生成、渲染、STL 导出、模量分析
- `topopt_backfill`：负责把拓扑优化密度场转成 seed cloud，并进一步生成 backfill 数据

---

## 这两个 package 分别做什么？

### 1) `helix_voronoi`
面向 **几何与分析**。

它负责：
- Voronoi 单胞生成与边提取
- 直杆 / 螺旋杆实体化
- 3D 渲染与预览图输出
- STL 导出
- 基于 `SfePy` 的单胞压缩模量分析

源码位置：
- `src/helix_voronoi/`

常用命令：
```bash
uv run helix-voronoi
uv run helix-voronoi export-helix --seed 55
uv run helix-voronoi export-mixed --seed 55
uv run helix-voronoi modulus --seed 55 --style both
```

---

### 2) `topopt_backfill`
面向 **工作流编排**。

它负责：
- 读取拓扑优化输出的 density NPZ
- 把密度场映射成概率分布
- 按目标 seed count 采样 Voronoi seeds
- 做 candidate pool 压缩与 FPS 代表点选择
- 生成 template backfill 数据

源码位置：
- `src/topopt_backfill/`

常用命令：
```bash
uv run topopt-backfill sample-seeds \
  datasets/topopt/fake_density_annular_cylinder_full.npz \
  --num-seeds 100000 \
  --output-npz datasets/topopt/seed_probability_mapping_100k.npz

uv run topopt-backfill backfill-templates \
  datasets/topopt/fake_density_annular_cylinder_full.npz \
  datasets/topopt/seed_probability_mapping_100k.npz \
  --output-npz datasets/topopt/template_backfilled_helix_voronoi.npz
```

---

## 二者关系

可以把整个项目理解成一条上下游链路：

1. `topopt_backfill` 先把拓扑优化结果整理成可用的 seed / backfill 数据
2. `helix_voronoi` 再负责 Voronoi / helix 几何生成、渲染、导出与分析

也就是说：
- `topopt_backfill` 偏 **输入处理与工作流**
- `helix_voronoi` 偏 **几何核心与分析能力**

---

## 快速开始

### 环境
使用 `uv` 管理依赖：

```bash
uv sync
```

### 默认渲染
生成默认 Voronoi 预览图：

```bash
uv run helix-voronoi
```

或直接从根目录入口运行：

```bash
uv run python main.py
```

如果需要弹出 matplotlib 窗口：

```bash
uv run python main.py --show
```

默认输出：

```text
docs/assets/voronoi_cube_3d.png
```

---

## 典型工作流

### A. 几何生成 / STL 导出

导出 helix STL：

```bash
uv run helix-voronoi export-helix \
  --seed 55 \
  --stl-output docs/assets/voronoi_helix_seed55.stl
```

导出 mixed STL：

```bash
uv run helix-voronoi export-mixed \
  --seed 55 \
  --stl-output docs/assets/voronoi_mixed.stl
```

### B. 模量分析

```bash
uv run helix-voronoi modulus --seed 55 --style both
```

只检查配置、不启动求解：

```bash
uv run helix-voronoi modulus --seed 55 --style both --dry-run
```

当前模量分析后端为 `SfePy`，流程是：
- 体素化直杆或螺旋杆单胞
- 转换为规则 `Hex8` 六面体网格
- 施加上下压板位移边界
- 求 `Z` 向等效模量

### C. 拓扑优化结果 -> seed -> backfill

第一步，生成 seed mapping：

```bash
uv run topopt-backfill sample-seeds \
  datasets/topopt/fake_density_annular_cylinder_full.npz \
  --num-seeds 100000 \
  --output-npz datasets/topopt/seed_probability_mapping_100k.npz
```

第二步，生成 backfill 数据：

```bash
uv run topopt-backfill backfill-templates \
  datasets/topopt/fake_density_annular_cylinder_full.npz \
  datasets/topopt/seed_probability_mapping_100k.npz \
  --output-npz datasets/topopt/template_backfilled_helix_voronoi.npz
```

---

## 目录结构

```text
src/
  helix_voronoi/      # 几何生成、渲染、STL、分析
  topopt_backfill/    # density -> seeds -> backfill 工作流

experiments/
  topopt_backfill/    # 还未产品化的实验脚本
  voxel_demos/        # toy voxel demo

datasets/
  topopt/             # 拓扑优化链路用到的 npz 数据
  voxel/              # voxel demo 数据

docs/assets/          # 文档图片与展示产物
tests/                # 回归测试
```

---

## 开发约定

- 正式功能放 `src/`
- 实验脚本放 `experiments/`
- 可复用 `.npz` 数据放 `datasets/`
- 文档插图和展示图片放 `docs/assets/`

---

## 测试

```bash
uv run python -m unittest discover -s tests -v
```

---

## 相关文档

- 模量分析方案：`docs/analysis/modulus-plan.md`
- 体素 demo：`docs/voxel_torus_demo.md`
