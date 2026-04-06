# How To Start

这份文档给第一次接手仓库的人用，目标是尽快把环境跑起来，并知道几个最常见的工作流怎么走。

## 1. 安装 uv

这个项目用 `uv` 管理 Python、虚拟环境和依赖。

macOS / Linux：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装完成后确认一下：

```bash
uv --version
```

如果你本机还没有 Python 3.13，可以直接让 `uv` 安装：

```bash
uv python install 3.13
```

这个仓库根目录有 `.python-version`，当前要求的版本是 `3.13`。

## 2. 克隆仓库

```bash
git clone <your-repo-url>
cd cage
```

如果你的目录名不是 `cage`，把后面的命令都放到仓库根目录执行就行。

## 3. 同步环境

第一次进入仓库后执行：

```bash
uv sync
```

这一步会根据 `pyproject.toml` 和 `uv.lock` 创建 `.venv` 并安装依赖。

如果想确认解释器和环境已经就绪，可以执行：

```bash
uv run python --version
uv run python -m unittest discover -s tests -v
```

## 4. 先了解项目里有什么

仓库主要有两条主线：

- `helix_voronoi`
  - 负责 Voronoi 单胞生成、预览渲染、STL 导出、模量分析
- `topopt_sampling`
  - 负责把拓扑优化三维密度场转成概率分布，并随机采样 seed points

几个常看目录：

- `src/helix_voronoi/`
- `src/topopt_sampling/`
- `datasets/`
- `docs/assets/`
- `tests/`

## 5. 典型工作流

### 工作流 A：先跑一个默认预览图

这是最快的冒烟路径，适合确认环境、渲染链路和 CLI 都正常。

```bash
uv run helix-voronoi
```

默认会生成：

```text
docs/assets/voronoi_cube_3d.png
```

如果你想换输出文件或者 seed 数量：

```bash
uv run helix-voronoi \
  --output docs/assets/my_preview.png \
  --num-seeds 12 \
  --row-seeds 116 55 49
```

### 工作流 B：从 voxel demo 走完整条 density -> seed sampling 链路

这条链路适合做 `topopt_sampling` 的主流程验证。

1. 生成体素输入：

```bash
uv run topopt-sampling generate-voxels \
  --output datasets/voxel/voxel_annular_cylinder_200x200x80.npz \
  --xy-size 200 \
  --z-size 80 \
  --outer-radius 100 \
  --inner-radius 50
```

2. 生成假的拓扑优化密度结果：

```bash
uv run topopt-sampling generate-fake-density \
  datasets/voxel/voxel_annular_cylinder_200x200x80.npz \
  --output datasets/topopt/fake_density_annular_cylinder_200x200x80.npz
```

3. 从密度场采样 seed points：

```bash
uv run topopt-sampling sample-seeds \
  datasets/topopt/fake_density_annular_cylinder_200x200x80.npz \
  --num-seeds 2000 \
  --output-npz datasets/topopt/seed_probability_mapping_2000.npz
```

4. 生成总览图：

```bash
uv run topopt-sampling render-overview \
  --density-npz datasets/topopt/fake_density_annular_cylinder_200x200x80.npz \
  --seed-npz datasets/topopt/seed_probability_mapping_2000.npz \
  --output docs/assets/topopt_sampling_pipeline_overview.png
```

这张总览图和 `README.md` 保持一致，包含 4 个 panel：

- 1) 密度场
- 2) 概率场
- 3) 2000 个随机种子点
- 4) 连续中空圆柱边界上的 3D Voronoi 表面分块图

这条链路里最常改的是：

- `--num-seeds`：采样点数量
- `--gamma`：密度转概率时的权重指数
- `--rng-seed`：随机采样种子

比如：

```bash
uv run topopt-sampling sample-seeds \
  datasets/topopt/fake_density_annular_cylinder_200x200x80.npz \
  --num-seeds 3000 \
  --gamma 1.5 \
  --rng-seed 7 \
  --output-npz datasets/topopt/seed_probability_mapping_3000_gamma15.npz
```

### 工作流 C：导出 STL

如果你要把几何送去打印、仿真或者下游 CAD/mesh 工具，通常会从这里开始。

导出纯 helix 风格：

```bash
uv run helix-voronoi export-helix \
  --seed 55 \
  --num-seeds 10 \
  --stl-output docs/assets/voronoi_helix_seed55.stl
```

导出 mixed 风格：

```bash
uv run helix-voronoi export-mixed \
  --seed 55 \
  --num-seeds 64 \
  --radius 0.012 \
  --helix-cycles 1.0 \
  --helix-amplitude 0.06 \
  --stl-output docs/assets/voronoi_mixed.stl
```

什么时候改哪些参数：

- 想让杆更粗，改 `--radius`
- 想让内部螺旋更明显，改 `--helix-cycles` 和 `--helix-amplitude`
- 想换一个几何实例，改 `--seed`
- 想控制 Voronoi 复杂度，改 `--num-seeds`

### 工作流 D：做模量分析

模量分析走 `SfePy`，计算会比渲染和导出更重，建议先 dry run。

先检查配置：

```bash
uv run helix-voronoi modulus \
  --seed 55 \
  --style both \
  --resolutions 96 128 160 \
  --dry-run
```

确认没问题后再正式运行：

```bash
uv run helix-voronoi modulus \
  --seed 55 \
  --style both \
  --resolutions 96 128 160 \
  --output-markdown docs/analysis/modulus_seed55.md \
  --output-json docs/analysis/modulus_seed55.json
```

常用参数：

- `--style cylinder|helix|both`
- `--resolutions`
- `--seed`
- `--num-seeds`

输出通常会写到：

- `docs/analysis/*.md`
- `docs/analysis/*.json`

### 工作流 E：日常开发

改代码时，最常用的是下面这几个命令：

跑单元测试：

```bash
uv run python -m unittest discover -s tests -v
```

快速看 CLI 是否还能正常启动：

```bash
uv run helix-voronoi --help
uv run topopt-sampling --help
```

一个简单的习惯是：

1. 先跑相关测试
2. 再跑一次你改到的那条主流程
3. 如果改到了输出格式或图像，重新生成产物，并自己检查 `docs/assets/` 或 `datasets/` 里的结果

## 6. 常见约定

- 正式工作流优先放在 `src/` 下面
- 可复用的 `.npz` 数据放 `datasets/`
- 文档展示用图片和 STL 放 `docs/assets/`

## 7. 遇到问题先看哪里

- 项目总览：`README.md`
- voxel demo 说明：`docs/voxel_torus_demo.md`
- 模量分析规划：`docs/analysis/modulus-plan.md`
- 3D 体块规划：`docs/plan/restricted-voronoi-3d-blocks-plan.md`
- 数据目录说明：`datasets/README.md`

如果你只是想先确认仓库是活的，最短路径就是：

```bash
uv sync
uv run helix-voronoi
uv run python -m unittest discover -s tests -v
```
