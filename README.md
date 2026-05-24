# Cage Pipeline

![Topology Sampling Overview](docs/assets/topopt_sampling_pipeline_overview.png)

这个仓库围绕一条完整 pipeline 组织：先用 `fem_analysis` 中的 FJW workflow 得到 cage 的最终伪密度分布，再把伪密度场转换为 seed points，构建 Voronoi / restricted Voronoi 结构，最后生成螺旋杆或实体骨架几何，并导出为 STL、GLB、图片和验证产物。

```text
fem_analysis FJW workflow
  -> final pseudo-density / design density
  -> standardized density NPZ
  -> topopt_sampling probability + seed sampling
  -> Voronoi / restricted Voronoi geometry
  -> helix_voronoi or matlab2stl_pipeline geometry export
  -> STL / GLB / viewer / analysis artifacts
```

`fem_analysis`、`topopt_sampling`、`helix_voronoi`、`matlab2stl_pipeline` 和 `ct_reconstruction` 是同一条 pipeline 上的节点。阅读和维护时应按阶段理解这些模块，避免把它们拆成多条主线。

## Pipeline Nodes

### 1. `fem_analysis`

FJW workflow 负责把 `references/fjw_work/` 的三工况骨改建优化流程迁移到 Python。当前主路径是 SfePy + SciPy 的 Python-only 求解、动态伴随、MMA 外层优化和 validation report。Abaqus 保留为历史输入生成和可选对照后端。

常用命令：

```bash
uv run fem-analysis fjw-preflight
uv run fem-analysis fjw-optimize --backend sfepy --mode three-force --max-iterations 1 --num-time-steps 1
uv run fem-analysis fjw-validate --run-directory runs/fjw_optimize
uv run fem-analysis fjw-capture-golden --run-directory runs/fjw_optimize --golden-directory datasets/fjw_golden/captured_run
```

边界说明：

- `fjw-preflight` 默认检查 Python-only 运行栈：`numpy`、`scipy`、`sfepy` 和 reference 输入。
- Abaqus、PETSc/MUMPS、captured golden 都是显式 `--require-*` 或专门参数打开的额外 gate。
- `scipy_iterative` 是 SfePy 的 Python-only 默认求解 profile。
- Abaqus backend 用于历史对照；真实优化入口必须使用 `--real-run`。
- 大体积运行产物默认放在 `runs/`，不作为应提交数据。

### 2. `topopt_sampling`

`topopt_sampling` 负责把三维密度场转换成概率场，并采样 Voronoi seed points。它也包含中空圆柱域上的 exact restricted Voronoi / hybrid B-rep / Three.js shell GLB 导出能力。

标准 `.npz` 密度输入的核心字段：

- `density_milli`：`uint16` 三维密度场，范围 `0..1000`
- `voxels`：`uint8` 三维占据体素，`0/1`
- `grid_shape_xyz` 或 `xy_size` / `z_size`：体素域尺寸
- `voxel_size_xyz_m`：体素物理尺寸，若该数据来自物理模型
- `shape_name`、`result_type`：数据来源标识

采样公式：

```text
w(i,j,k) = rho(i,j,k)^gamma
p(i,j,k) = w(i,j,k) / sum(w)
```

命令示例：

```bash
uv run topopt-sampling sample-seeds \
  datasets/topopt/fake_density_annular_cylinder_200x200x80.npz \
  --num-seeds 2000 \
  --output-npz datasets/topopt/seed_probability_mapping_2000.npz
```

### 3. `helix_voronoi`

`helix_voronoi` 是 Voronoi 杆系几何节点，负责 Voronoi 单胞生成、边提取、直杆 / 螺旋杆实体化、预览渲染和 STL 导出。

常用命令：

```bash
uv run helix-voronoi
uv run helix-voronoi export-helix --seed 55
uv run helix-voronoi export-mixed --seed 55
```

当前 CLI 主要用于单位 Voronoi 杆系和螺旋杆样式验证。下游把 `topopt_sampling` 的 density-derived seeds 接进螺旋杆生成时，应复用这个节点里的杆件实体化逻辑。

### 4. `matlab2stl_pipeline`

`matlab2stl_pipeline` 是 681 `.mat` 数据到骨架 STL / GLB 的已实现端到端几何节点。它适合验证“伪密度场到 Voronoi 骨架 mesh”的下游几何链路。

```bash
uv run matlab2stl-pipeline run-pipeline \
  --mat datasets/681.mat \
  --output-dir outputs/matlab2stl_pipeline \
  --viewer-dir viewer/public/data
```

默认步骤包括：

```text
.mat -> raw density NPZ -> raw GLB
  -> OBB fitting -> aligned density + probability
  -> seed sampling -> optional CVT
  -> box-restricted Voronoi
  -> edge extraction
  -> skeleton voxelization
  -> marching-cubes mesh
  -> STL / GLB
```

### 5. `ct_reconstruction` and viewer

`ct_reconstruction` 负责 STL 到体素 / NPZ / GLB 的辅助转换。`viewer/` 负责本地交互查看 GLB 结果。

```bash
cd viewer
pnpm install
pnpm dev
```

默认地址：

- `http://127.0.0.1:5173/`

## Smoke Workflows

### A. 环境检查

```bash
uv sync
uv run python --version
uv run python -m unittest discover -s tests -v
```

### B. FJW Python-only 节点

```bash
uv run fem-analysis fjw-preflight
uv run fem-analysis fjw-optimize \
  --backend sfepy \
  --mode three-force \
  --max-iterations 1 \
  --num-time-steps 1 \
  --runtime-profile local \
  --run-directory runs/fjw_optimize
uv run fem-analysis fjw-validate --run-directory runs/fjw_optimize
```

正式远端优化默认按 `wuyinyun` profile 执行：`petsc_mumps`、`case_parallelism=2`、`solver_threads=12`。每个完成 checkpoint 会额外写入 `iter_###/timing.json`，用于查看 force case、forward、adjoint 和 checkpoint 写入耗时。

### C. Annular-cylinder density sampling demo

这个 demo 用解析中空圆柱生成假密度场，主要用于快速验证 `topopt_sampling` 节点。

```bash
uv run topopt-sampling generate-voxels \
  --output datasets/voxel/voxel_annular_cylinder_200x200x80.npz \
  --xy-size 200 \
  --z-size 80 \
  --outer-radius 100 \
  --inner-radius 50

uv run topopt-sampling generate-fake-density \
  datasets/voxel/voxel_annular_cylinder_200x200x80.npz \
  --output datasets/topopt/fake_density_annular_cylinder_200x200x80.npz

uv run topopt-sampling sample-seeds \
  datasets/topopt/fake_density_annular_cylinder_200x200x80.npz \
  --num-seeds 2000 \
  --output-npz datasets/topopt/seed_probability_mapping_2000.npz

uv run topopt-sampling render-overview \
  --density-npz datasets/topopt/fake_density_annular_cylinder_200x200x80.npz \
  --seed-npz datasets/topopt/seed_probability_mapping_2000.npz \
  --output docs/assets/topopt_sampling_pipeline_overview.png
```

### D. Shell GLB viewer export

```bash
uv run topopt-sampling export-threejs-shell-glb \
  datasets/topopt/seed_probability_mapping_2000.npz \
  --xy-size 200 \
  --z-size 80 \
  --outer-radius 100 \
  --inner-radius 50 \
  --output-json viewer/public/data/hybrid_exact_shell_2000.glb
```

### E. 681 downstream geometry node

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

## Repo Layout

```text
src/
  fem_analysis/          # FJW optimization, solver, adjoint, MMA, validation
  topopt_sampling/       # density -> probability -> seeds -> restricted Voronoi
  helix_voronoi/         # Voronoi rod and helix-rod geometry export
  matlab2stl_pipeline/   # 681 .mat -> Voronoi skeleton STL/GLB pipeline
  ct_reconstruction/     # STL/voxel/NPZ/GLB conversion helpers

datasets/
  topopt/                # reusable density, seed, and intermediate data
  voxel/                 # generated demo voxel geometry

docs/
  assets/                # tracked figures and visual artifacts
  analysis/              # audit reports and analysis notes

viewer/
  public/data/           # GLB data loaded by the Three.js viewer
```

## Testing

```bash
uv run python -m unittest discover -s tests -v
```

For targeted checks:

```bash
uv run python -m unittest tests/test_topopt_sampling.py -v
uv run python -m unittest tests/test_helix.py -v
uv run python -m unittest tests/test_matlab2stl_pipeline_transform.py -v
uv run python -m unittest tests/test_fjw_workflow_optimize.py -v
```

## Related Docs

- `docs/background.md`
- `docs/how_to_start.md`
- `docs/fjw_workflow_runbook.md`
- `docs/681_pipeline.md`
- `docs/annular_cylinder_voxel_demo.md`
- `src/matlab2stl_pipeline/HOW_TO_START.md`
