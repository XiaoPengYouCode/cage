# FJW Work Full Goal

这份文档定义当前仓库要交付的 FJW 工作流目标。它对应完整 cage pipeline 的上游伪密度生成节点。目标已经收束为一件事：用 SfePy/SciPy 完全替代 Abaqus executable/license，形成 Python-only 的生产实现。

完整仓库主线是：`fem_analysis` 生成最终伪密度，`topopt_sampling` 转成概率场和 seed points，Voronoi 几何节点生成结构，`helix_voronoi` 或 `matlab2stl_pipeline` 导出螺旋杆 / 骨架 mesh。

这里的 Python-only 指运行主流程不需要 Abaqus、MATLAB、商业 license，也不要求 PETSc/MUMPS。基础依赖是仓库的 Python 依赖栈，核心求解路径使用 SfePy + SciPy。PETSc/MUMPS 可以作为大模型加速 profile，Abaqus 只保留为历史资料对照和可选比较工具。

## Scope

完整 FJW workflow 需要覆盖三工况历史主流程：

1. 读取 `nod_coo / ele_nod / B / D / cor_ele / tra_ele / cage_ele / desi_ele / obj_ele`
2. 初始化 `ini_str = 0.36`、三工况 `ini_cage = 0.3`
3. 每个外层优化迭代运行 `force_1 / force_2 / force_3`
4. 每个工况做 `P = 3` 个正向骨改建时间步
5. 每个工况做反向伴随链路，动态生成 `Fv` 并求解伴随位移
6. 合并三工况终态骨量作为目标函数
7. 合并三工况梯度，使用 MMA 更新 `design_cage`
8. 重复直到 `delta <= 1e-4` 或达到显式迭代上限
9. 输出设计、目标函数、骨量历史、梯度、`Fv`、`Fai`、checkpoint 和 validation report

生产路径不能使用 zero vector、placeholder、mock solver 或静默 fallback。缺少真实位移、伴随载荷或 MMA 状态时，代码必须显式失败。

## Source Of Truth

当前主实现以这些文件为准：

- `src/fem_analysis/fjw_reference.py`
- `src/fem_analysis/fjw_direct_solver.py`
- `src/fem_analysis/fjw_workflow_sfepy_solver_adapters.py`
- `src/fem_analysis/fjw_workflow_single_case.py`
- `src/fem_analysis/fjw_workflow_adjoint.py`
- `src/fem_analysis/fjw_workflow_three_force.py`
- `src/fem_analysis/fjw_mma.py`
- `src/fem_analysis/fjw_workflow_optimize.py`
- `src/fem_analysis/fjw_validation.py`

`references/fjw_work/` 是历史资料归档，用来理解输入格式、材料参数、载荷、集合命名和旧流程。它不再决定当前运行时必须依赖 Abaqus。

## Target Architecture

最终主路径是一条 SfePy/SciPy 后端驱动的 Python workflow：

```text
FJWWorkflowState
  |
  +-- SfePy/SciPy forward and adjoint solver backend
  |     +-- scipy_iterative  (Python-only default)
  |     +-- scipy_direct     (small-model debugging)
  |     +-- petsc_mumps      (optional acceleration)
  |
  +-- Biology update
  +-- Dynamic adjoint update
  +-- Three-force aggregation
  +-- Real MMA optimizer
  +-- Checkpoint / resume
  +-- Validation / golden capture
```

Abaqus backend can remain in the repository with clear labeling:

- optional historical comparison
- optional `.inp` generation compatibility check
- optional external validation when a licensed machine exists

Abaqus availability is not a Definition of Done item for the Python implementation.

## Current State

已经落地的能力：

- 读取 FJW `.mat` 网格、区域集合、材料常数、载荷和初始状态。
- 生成历史兼容 `.inp`，用于审计输入语义和对照旧资料。
- 用 SfePy 构建六面体线弹性直接求解模型。
- 支持 `scipy_iterative`、`scipy_direct`、`petsc_mumps` 三种 SfePy linear solver profile。
- `scipy_iterative` 是 CLI 和优化配置的 Python-only 默认 profile。
- rigid-control unknown 只挂在顶面控制区域，避免给整个参考模型额外扩展 6-DOF 控制场。
- 用 SfePy backend 跑正向位移和伴随位移，进入骨改建和伴随链路。
- 动态生成每个伴随时间步的 `Fv`，并保存可审计 artifact。
- 纯 Python/NumPy 迁移 `mmasub.m` 和 `subsolv.m`。
- 外层优化支持 `delta` 收敛、checkpoint、resume 和稳定输出目录。
- validation framework 支持结构检查、输入 diff、checkpoint 深度检查、关键 `.npz` 数组检查和 golden manifest。
- preflight 默认只检查 Python runtime：reference dir、`numpy`、`scipy`、`sfepy`。
- Abaqus、PETSc/MUMPS、captured golden 都是显式 `--require-*` 的可选 gate。

已经验证过的事实：

- 完整参考模型能被加载，规模是 `593790` 个节点、`544112` 个六面体单元。
- 三工况 dry-run 能生成完整 job/input 清单。
- 小模型 SfePy 后端能跑通三工况一轮迭代。
- 小模型 SfePy optimizer 能跑外层 checkpoint/resume。
- `scipy_iterative` 能在单六面体测试上完成真实 SfePy 求解。
- MMA 小向量 fixture 能稳定回归。
- 本机 `fjw-preflight` 默认 Python-only 检查通过。

仍需要持续验证的内容：

- reference-scale 完整求解的耗时和内存边界。
- `force_2 / force_3` 转动载荷在 SfePy rigid coupling 中的数值稳定性。
- 如果未来拿到历史运行输出，可以增加可选的 Abaqus/MATLAB 数值对照。

这些内容影响性能置信度和历史对照置信度，不把 Abaqus license 带回主路径。

## Implementation Phases

### Phase 1: Python Input Baseline And Golden Fixtures

目标：建立 Python 可回归输入和小模型 golden fixture。

已完成：

- 整理历史 reference 目录中可复用的静态输入和运行期输出缺口。
- 明确当前没有 `Force_*.mat`、`U1_ele_nod_dir*.mat`、`obj_bo*.mat`、`ob*.mat` 等历史运行输出。
- 建立小模型 golden case，覆盖 FJW 数据结构、B/D 矩阵、骨改建公式和 MMA 接口。
- 建立 reference-scale input dry-run baseline，用来比对 `.inp`、节点集、元素集、材料桶和载荷块。
- 建立 `datasets/fjw_golden/`，只放小体积、可回归的 golden 数据。

验收标准：

- 小模型 golden case 在单元测试里完整回归。
- reference-scale 输入生成能稳定 diff。
- 文档明确历史 runtime 输出缺失，不能把缺失数据描述成已验证一致。

### Phase 2: SfePy/SciPy Solver Backend

目标：用 SfePy/SciPy 产出真实位移向量，替代 Abaqus runtime。

已完成：

- 构建 FJW 六面体网格、材料桶、区域和载荷。
- 实现顶底 reference point coupling 的 SfePy 表达。
- 把 6-DOF rigid-control field 限定在顶面控制区域，非参考点控制自由度显式固定。
- 支持正向和伴随 nodal load。
- 支持 `scipy_iterative` Python-only 求解 profile。
- 保留 `scipy_direct` 作为小模型调试 profile。
- 保留 `petsc_mumps` 作为可选加速 profile，并在缺少 runtime 时明确报错。

验收标准：

- `fjw-direct --solve` 通过 SfePy 返回真实位移向量。
- 小模型 `force_1/2/3` 都能走同一 solver backend。
- 求解失败时保留明确错误，不产生伪结果。

### Phase 3: Dynamic Adjoint Pipeline

目标：伴随 job 使用正向结果动态生成的 `Fv`，形成真实反向链路。

已完成：

- 正向 `ti=0..P-1` 后，反向 `tn=P-1..0`。
- 每个 `tn` 用对应正向位移和 `Fai_next` 构造 `Fv`。
- 每个伴随时间步通过 SfePy 求解真实伴随位移。
- `Fv` 保存 active node ids、active forces、dense checksum 和 CLOAD text。
- `Fv_set` 稳定排序，输出可 diff。

验收标准：

- 三工况、三时间步共 9 个伴随步都来自对应时间步正向结果。
- `Fv` 非零性、active node 数和文本输出可检查。
- 小模型伴随链路和手算 fixture 对齐。

### Phase 4: Real MMA Optimizer

目标：用纯 Python 实现历史 MMA 更新，移除 placeholder optimizer。

已完成：

- 迁移 `mmasub.m` 和 `subsolv.m`。
- 保留历史参数 `m / c / d / a0 / a / xmin / xmax`。
- 保留 `xold1 / xold2 / low / up` 状态。
- 实现 `delta = mean(abs(xmma - design_cage))`。
- 生产 driver 默认使用 real MMA。

验收标准：

- 小规模 MMA fixture 与历史公式在数值容差内一致。
- `run_fjw_workflow_iteration` 默认更新 `design_cage`。
- diagnostics 记录目标函数、约束、delta、move/asymptote。

### Phase 5: Optimization State Machine

目标：实现可恢复的 `while(delta > 1e-4)` Python 主循环。

已完成：

- `uv run fem-analysis fjw-optimize`
- `--backend sfepy`
- `--mode three-force`
- `--max-iterations`
- `--delta-tol`
- `--num-time-steps`
- `--run-directory`
- `--resume`
- `--checkpoint-every`
- 每轮保存设计、目标、骨量、梯度、`Fv`、`Fai`、MMA state 和 `delta`

验收标准：

- 小模型能跑到 `delta <= tol` 或 `max_iterations`。
- `--resume` 能从最后一个完整 checkpoint 继续。
- 输出目录能解释每一轮优化发生了什么。

### Phase 6: Validation

目标：证明 Python-only 实现内部一致、可审计、可复现。

已完成：

- input equivalence：`.inp` 结构和关键集合一致。
- local math equivalence：`bone_delta`、`d_bone_delta`、MMA、梯度聚合一致。
- solve equivalence：同一 Python fixture 的位移、能量、骨量更新一致。
- checkpoint completeness：逐项检查 MMA、aggregate、case history、forward step、`Fv`、`Fai` 和 `fv_manifest.json`。
- golden capture：从完成的 Python/SfePy run 生成 manifest、SHA256 和小文件副本。
- optional historical comparison：如果有 Abaqus/MATLAB 输出，可以作为额外 golden source。

验收标准：

- 每次完整运行能输出 validation report。
- report 明确列出 pass/fail、最大误差、误差位置。
- 没有历史 Abaqus 输出时，报告只说明历史对照不可用，不影响 Python-only 完成状态。

### Phase 7: Documentation And Operations

目标：让后续协作者能安装、运行、验证和调试 Python-only workflow。

已完成：

- README 中记录 Python-only SfePy/SciPy 主路径。
- `docs/how_to_start.md` 中加入最小 FJW smoke path。
- `docs/fjw_workflow_runbook.md` 中区分主路径、可选 Abaqus comparison 和可选 PETSc/MUMPS acceleration。
- preflight 默认检查 Python dependencies。
- artifact policy 说明哪些进 git、哪些放 `runs/`、哪些放 `outputs/`。

验收标准：

- 新人能按文档完成 Python-only preflight。
- 新人能按文档跑 SfePy smoke iteration。
- 失败时能根据 validation report、manifest 和 checkpoint 定位问题。

## Final CLI

主路径：

```bash
uv run fem-analysis fjw-preflight
uv run fem-analysis fjw-direct --load-case force_1 --solve
uv run fem-analysis fjw-sfepy-iterate --num-time-steps 1
uv run fem-analysis fjw-optimize --backend sfepy --mode three-force --max-iterations 1
uv run fem-analysis fjw-validate --run-directory runs/fjw_optimize
uv run fem-analysis fjw-capture-golden --run-directory runs/fjw_optimize
```

显式 profile：

```bash
uv run fem-analysis fjw-direct --load-case force_1 --solve --sfepy-linear-solver scipy_iterative
uv run fem-analysis fjw-optimize --backend sfepy --sfepy-linear-solver scipy_iterative
```

可选加速：

```bash
uv run fem-analysis fjw-direct --load-case force_1 --solve --sfepy-linear-solver petsc_mumps
```

可选历史对照：

```bash
uv run fem-analysis fjw-workflow --mode three-force
uv run fem-analysis fjw-workflow --mode three-force --execute-jobs --real-run
uv run fem-analysis fjw-optimize --backend abaqus --mode three-force --max-iterations 1 --real-run
```

## Runtime Environment

基础完成环境：

- Python 3.13
- `uv sync`
- `numpy`
- `scipy`
- `sfepy`

默认 preflight：

```bash
uv run fem-analysis fjw-preflight
```

可选严格检查：

```bash
uv run fem-analysis fjw-preflight --require-golden
uv run fem-analysis fjw-preflight --require-petsc-mumps
uv run fem-analysis fjw-preflight --require-abaqus
```

严格检查用于额外对照或加速环境，不属于 Python-only 主路径完成条件。

## Definition Of Done

同时满足下面条件，才能说 Python codebase 完成 FJW Python-only 替代实现：

- [ ] `fjw-preflight` 默认 Python-only 检查通过。
- [ ] `fjw-direct --solve` 能通过 SfePy/SciPy 产出真实位移向量。
- [ ] 三工况模式能从 `design_cage = 0.3` 自动跑完至少一轮真实正向 + 伴随 + MMA。
- [ ] 每个正向时间步都有真实位移向量，不接受 zero vector 或 placeholder。
- [ ] 每个伴随时间步的 `Fv` 来自对应正向位移和 `Fai_next`。
- [ ] 每个伴随时间步都有真实伴随位移。
- [ ] MMA 更新来自真实 `mmasub/subsolv` 等价实现。
- [ ] 外层循环能根据 `delta <= 1e-4` 停止。
- [ ] 所有关键中间量都可落盘、可恢复、可审计。
- [ ] validation report 能覆盖输入、公式、求解结果、checkpoint 和可用 golden。
- [ ] 文档明确说明 Python-only 环境、命令、产物、验证和已知限制。
- [ ] Abaqus 不作为主流程运行依赖。
- [ ] PETSc/MUMPS 不作为基础运行依赖。

可选增强项：

- [ ] 在大内存机器上跑通 reference-scale `force_1 t0`。
- [ ] 在 PETSc/MUMPS 环境上记录 reference-scale 性能数据。
- [ ] 在有 Abaqus/MATLAB 输出时增加历史数值对照报告。

## Priority Map

### P0: Python-only Closure

- SfePy/SciPy backend 真实正向求解。
- SfePy/SciPy backend 真实伴随求解。
- 动态 `Fv`。
- Real MMA。
- checkpoint/resume。
- validation report。

### P1: Reference-scale Confidence

- 完整参考模型 setup 稳定。
- reference-scale `force_1` 的内存和耗时记录。
- `force_2 / force_3` 转动载荷数值检查。

### P2: Optional Acceleration

- PETSc/MUMPS profile。
- 大模型运行建议和资源记录。

### P3: Optional Historical Comparison

- Abaqus `.inp` 对照。
- Abaqus/MATLAB golden capture。
- 历史输出数值比较。

## Final Tightening Rules

- 生产路径不能静默使用 placeholder optimizer。
- 生产路径不能静默使用 placeholder adjoint fields。
- 静态 zero-Fv 伴随 `.inp` 只能作为模板检查，不能描述成真实伴随任务。
- 没有 validation report 时，不能声称运行已被验证。
- 没有历史 Abaqus/MATLAB 输出时，不能声称历史逐项数值一致。
- 缺少 Abaqus license 不能阻塞 Python-only 完成状态。
- 缺少 PETSc/MUMPS 不能阻塞基础 Python-only 完成状态。
