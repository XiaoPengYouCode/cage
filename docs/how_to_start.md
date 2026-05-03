# How To Start

这份文档给第一次接手仓库的人用，目标是尽快把环境跑起来，并理解整条 pipeline 上每个节点的职责。

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

安装完成后确认：

```bash
uv --version
```

如果本机还没有 Python 3.13，可以直接让 `uv` 安装：

```bash
uv python install 3.13
```

这个仓库根目录有 `.python-version`，当前要求的版本是 `3.13`。

## 2. 同步环境

```bash
uv sync
uv run python --version
uv run python -m unittest discover -s tests -v
```

## 3. 项目全貌

仓库只有一条主线：

```text
fem_analysis
  -> final pseudo-density / design density
  -> topopt_sampling
  -> Voronoi / restricted Voronoi
  -> helix_voronoi or matlab2stl_pipeline
  -> STL / GLB / viewer / analysis
```

常看目录：

- `src/fem_analysis/`
- `src/topopt_sampling/`
- `src/helix_voronoi/`
- `src/matlab2stl_pipeline/`
- `src/ct_reconstruction/`
- `datasets/`
- `docs/assets/`
- `viewer/`
- `tests/`

## 4. 最短 smoke path

如果只是确认仓库能跑：

```bash
uv sync
uv run fem-analysis fjw-preflight
uv run python -m unittest discover -s tests -v
```

如果只想快速生成一个 Voronoi 预览图：

```bash
uv run helix-voronoi
```

默认输出：

```text
docs/assets/voronoi_cube_3d.png
```

## 5. Pipeline 节点说明

### 节点 A：FJW workflow 生成伪密度

先检查 Python-only runtime：

```bash
uv run fem-analysis fjw-preflight
```

跑一轮 SfePy/Python smoke optimization：

```bash
uv run fem-analysis fjw-optimize \
  --backend sfepy \
  --mode three-force \
  --max-iterations 1 \
  --num-time-steps 1 \
  --sfepy-linear-solver scipy_iterative \
  --run-directory runs/fjw_optimize
```

检查输出结构：

```bash
uv run fem-analysis fjw-validate --run-directory runs/fjw_optimize
```

如果这次运行要作为新的 Python/SfePy golden 来源：

```bash
uv run fem-analysis fjw-capture-golden \
  --run-directory runs/fjw_optimize \
  --golden-directory datasets/fjw_golden/captured_run
```

历史 `.inp` 生成只用于审计输入语义和对照旧资料：

```bash
uv run fem-analysis fjw-workflow --mode three-force --time-steps 3
```

有 Abaqus 的机器上，真实历史对照路径必须显式打开：

```bash
uv run fem-analysis fjw-optimize \
  --backend abaqus \
  --mode three-force \
  --max-iterations 1 \
  --num-time-steps 3 \
  --real-run
```

### 节点 B：密度场采样 seed points

正式 pipeline 中，这一步消费 FJW workflow 产出的最终伪密度。日常 smoke 可以用解析中空圆柱假密度场验证 `topopt_sampling` 节点。

生成体素输入：

```bash
uv run topopt-sampling generate-voxels \
  --output datasets/voxel/voxel_annular_cylinder_200x200x80.npz \
  --xy-size 200 \
  --z-size 80 \
  --outer-radius 100 \
  --inner-radius 50
```

生成假密度结果：

```bash
uv run topopt-sampling generate-fake-density \
  datasets/voxel/voxel_annular_cylinder_200x200x80.npz \
  --output datasets/topopt/fake_density_annular_cylinder_200x200x80.npz
```

从密度场采样 seed points：

```bash
uv run topopt-sampling sample-seeds \
  datasets/topopt/fake_density_annular_cylinder_200x200x80.npz \
  --num-seeds 2000 \
  --gamma 1.0 \
  --rng-seed 42 \
  --output-npz datasets/topopt/seed_probability_mapping_2000.npz
```

这一步最常调的参数：

- `--num-seeds`：采样点数量
- `--gamma`：密度转概率时的权重指数
- `--rng-seed`：随机采样种子

生成总览图：

```bash
uv run topopt-sampling render-overview \
  --density-npz datasets/topopt/fake_density_annular_cylinder_200x200x80.npz \
  --seed-npz datasets/topopt/seed_probability_mapping_2000.npz \
  --output docs/assets/topopt_sampling_pipeline_overview.png
```

### 节点 C：restricted Voronoi / shell GLB

```bash
uv run topopt-sampling exact-summary \
  datasets/topopt/seed_probability_mapping_2000.npz \
  --xy-size 200 \
  --z-size 80 \
  --outer-radius 100 \
  --inner-radius 50
```

导出 Three.js viewer 使用的 shell GLB：

```bash
uv run topopt-sampling export-threejs-shell-glb \
  datasets/topopt/seed_probability_mapping_2000.npz \
  --xy-size 200 \
  --z-size 80 \
  --outer-radius 100 \
  --inner-radius 50 \
  --output-json viewer/public/data/hybrid_exact_shell_2000.glb
```

### 节点 D：螺旋杆 / Voronoi 杆系导出

`helix_voronoi` 当前用于验证 Voronoi 杆系和螺旋杆实体化。

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

### 节点 E：681 `.mat` 到骨架 STL / GLB

`matlab2stl_pipeline` 是已实现的下游几何链路，用于验证伪密度场到 Voronoi 骨架 mesh 的全流程。

```bash
uv run matlab2stl-pipeline run-pipeline \
  --mat datasets/681.mat \
  --output-dir outputs/matlab2stl_pipeline_seed500_cvt500 \
  --viewer-dir viewer/public/data \
  --num-seeds 500 \
  --cvt-iters 500
```

重点输出：

- `outputs/matlab2stl_pipeline_seed500_cvt500/681_skeleton_density.stl`
- `outputs/matlab2stl_pipeline_seed500_cvt500/681_skeleton_cvt500.stl`
- `viewer/public/data/681_skeleton_density.glb`
- `viewer/public/data/681_skeleton_cvt500.glb`

## 6. Viewer

```bash
cd viewer
pnpm install
pnpm dev
```

默认地址：

- `http://127.0.0.1:5173/`

常用模型参数：

- `?model=voronoi`
- `?model=681_raw`
- `?model=681_skeleton_density`
- `?model=681_skeleton_cvt500`
- `?model=fjw_reference`

## 7. 开发检查

快速看 CLI 是否正常启动：

```bash
uv run fem-analysis fjw-optimize --help
uv run topopt-sampling --help
uv run helix-voronoi --help
uv run matlab2stl-pipeline run-pipeline --help
```

常用测试：

```bash
uv run python -m unittest discover -s tests -v
uv run python -m unittest tests/test_topopt_sampling.py -v
uv run python -m unittest tests/test_helix.py -v
uv run python -m unittest tests/test_matlab2stl_pipeline_transform.py -v
```

一个简单的习惯：

1. 先跑相关测试
2. 再跑一次改到的 pipeline 节点
3. 如果改到输出格式或图像，重新生成产物，并检查 `docs/assets/`、`datasets/`、`viewer/public/data/` 或 `outputs/` 中的结果

## 8. 常见约定

- 主实现放在 `src/`
- 可复用输入和中间标准数据放在 `datasets/`
- 文档展示图片放在 `docs/assets/`
- 分析报告放在 `docs/analysis/`
- 大体积运行结果放在 `runs/` 或 `outputs/`

## 9. 继续阅读

- 项目背景：`docs/background.md`
- FJW workflow：`docs/fjw_workflow_runbook.md`
- 681 downstream geometry：`docs/681_pipeline.md`
- annular-cylinder demo：`docs/annular_cylinder_voxel_demo.md`
- matlab2stl 快速说明：`src/matlab2stl_pipeline/HOW_TO_START.md`
