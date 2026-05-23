# AGENTS

## Purpose

这个仓库用于推进 cage / Voronoi / topology optimization 的完整 pipeline：从 FJW / `fem_analysis` workflow 得到最终伪密度分布，再采样 seed points，生成 Voronoi / restricted Voronoi 结构，最后形成螺旋杆或骨架 mesh 并导出可视化、制造和分析产物。

协作目标很明确：

- 保持唯一主流程可复现
- 保持代码入口清楚
- 让历史资料和当前实现分层清晰
- 优先交付简单、稳定、容易维护的方案

## First Read

在开始修改代码、文档或数据之前，先阅读 [docs/background.md](docs/background.md)。

这个文档说明了项目背景，以及 `references/fjw_work/` 的来源。涉及背景说明、历史来历、资料出处时，优先引用该文档，不要凭印象改写。

## Source Of Truth

当前仓库里，以下内容是主实现和主说明：

- `src/`
- `pyproject.toml`
- `README.md`
- `docs/`
- `tests/`

以下内容是参考资料或实验产物，不应默认当作当前主入口：

- `references/fjw_work/`
- `outputs/`
- `experiment_output/`

如果历史资料和当前 Python 实现不一致，以当前仓库里的 Python 代码、CLI 和 `docs/` 为准；只有当任务明确要求整理、迁移或复现实验时，才把 `references/fjw_work/` 当成工作对象。

## Repository Shape

当前主要入口模块对应同一条 pipeline 的不同节点：

- `src/fem_analysis/`
- `src/topopt_sampling/`
- `src/helix_voronoi/`
- `src/matlab2stl_pipeline/`
- `src/ct_reconstruction/`

辅助或较底层模块包括：

- `src/cvt/`
- `src/sampling/`
- `src/visualization/`
- `src/voronoi/`
- `src/topopt_backfill/`

处理任务时，优先从完整 pipeline 的调用链理解问题：`fem_analysis -> topopt_sampling -> Voronoi geometry -> helix_voronoi / matlab2stl_pipeline -> viewer / export`。不要先从历史脚本或归档资料入手。

## Environment

- Python 版本要求：`3.13`
- Python 依赖管理工具：`uv`
- 前端 viewer 使用：`pnpm`

首次进入仓库或依赖变更后，优先执行：

```bash
uv sync
```

常用检查命令：

```bash
uv run python --version
uv run python -m unittest discover -s tests -v
```

如果任务涉及 viewer，再进入 `viewer/` 目录执行前端命令。

## Main CLI Entrypoints

`pyproject.toml` 中登记的主 CLI 如下：

- `uv run helix-voronoi`
- `uv run topopt-sampling`
- `uv run ct-reconstruction`
- `uv run fem-analysis`
- `uv run matlab2stl-pipeline`

新增功能时，优先复用这些现有入口；只有当行为边界已经明显超出现有命令职责时，才新增子命令或新入口。

## Working Style

- 优先简单、清楚、适合上线和后续维护的实现
- API 保持小而明确，命名直接，不堆抽象层
- 先接入当前 pipeline 数据流，再考虑扩展能力
- 优先修改现有模块和现有 CLI，不轻易分叉出平行实现
- 对于实验性质代码，尽量放在 `experiments/`，不要污染稳定入口

如果多个方案都可行，优先选择：

1. 与当前目录结构一致
2. 与当前 CLI 行为一致
3. 最少引入新依赖
4. 最容易测试和回归验证

## 远程开发机 Skill

涉及远程开发机或 wuyinyun 的任何操作，加载 `.agents/skills/remote-dev-cage/SKILL.md`。

触发条件：

- 提到 `wuyinyun`、远程机器、远端运行、SSH、云机器
- 需要在远程机器执行 `uv sync`、`git pull`、`fjw-preflight`、`fjw-optimize` 或 `fjw-validate`
- 需要跑 FJW / SfePy 一轮迭代优化

机器特定参数放在 `.agents/skills/remote-dev-cage/machines/`。当前已注册：`wuyinyun`。

## References Policy

`references/fjw_work/` 中保存的是你同门师兄 **方嘉纬学长** 的历史工作资料。

这部分资料的用途主要是：

- 对照历史 `.mat`、`.m`、`.inp` 文件理解前期流程
- 为当前数据格式、边界条件、输入输出关系提供背景
- 在需要复现旧工作时作为参考来源

使用这些资料时遵守下面的边界：

- 不把其中的 MATLAB / Abaqus 脚本直接视为当前主实现
- 不在没有明确需求的情况下把历史脚本大规模搬进当前 Python 主流程
- 如果从这些资料中提炼规则、参数或流程，务必把结论写回 `docs/` 或当前代码注释，而不是只停留在一次性口头说明里

## Data And Artifact Policy

目录职责保持清楚：

- `datasets/`：可复用输入、样例数据、中间标准化数据
- `docs/assets/`：文档要展示的图片或可跟踪展示产物
- `docs/analysis/`：分析结果、摘要、报告
- `outputs/`：临时输出，不默认视为应提交内容
- `experiment_output/`：实验过程产物，不默认视为应提交内容

处理产物时遵守这些原则：

- 只有在文档、测试或功能验证真正需要时，才提交新的二进制产物
- 大体积中间文件优先放在 `outputs/` 或 `experiment_output/`
- 不要把一次性调试产物混入 `datasets/` 或 `docs/assets/`

## Documentation Policy

当代码行为、数据约定或工作流发生变化时，同步更新相关文档：

- 用户如何开始，用 `README.md` 和 `docs/how_to_start.md`
- 背景和资料来源，用 `docs/background.md`
- 具体流程，用对应的 `docs/*.md`
- 分析结论，用 `docs/analysis/`
- 过期计划、快照和被当前实现替代的文档应及时删除，避免和当前 pipeline 口径冲突

如果改动依赖 `references/fjw_work/` 的背景资料，最好在文档中明确写出“这是从历史资料整理出来的结论”，避免后续读者误以为它是当前实现天然自带的规则。

## Testing And Validation

改动完成后，至少做与任务相称的验证。

默认优先级：

1. 跑最小相关测试
2. 跑现有单元测试
3. 如果改动 CLI 或数据流，跑一次对应命令链路

常用命令：

```bash
uv run python -m unittest discover -s tests -v
```

如果只改某一块，优先补充或执行对应测试，例如：

- `tests/test_annular_cylinder.py`
- `tests/test_box_voronoi.py`
- `tests/test_helix.py`
- `tests/test_matlab2stl_pipeline_transform.py`
- `tests/test_topopt_sampling.py`

如果没法验证，要明确说明没跑什么、为什么没跑。

## Branch And Change Hygiene

当前仓库默认以 `main` 作为主分支和主要协作面。

协作时遵守这些约定：

- 不随意保留过期分支、备份分支或无主分支
- 不把“已经归档的旧历史”继续当成活跃实现来源
- 清理分支前先确认对应工作是否已经合入或不再需要
- 没有明确要求时，不重写共享历史

## Decision Heuristics

当信息不完整时，按下面顺序决策：

1. 看当前 CLI 和 `src/` 里的现有实现怎么组织
2. 看 `README.md`、`docs/how_to_start.md`、流程文档怎么描述
3. 必要时参考 `references/fjw_work/` 理解历史背景
4. 仍然有歧义时，选择最小、最清楚、最容易回滚的方案

## Expected Agent Behavior

进入这个仓库工作的 agent 应该：

- 先建立代码上下文，再修改
- 修改时优先维护现有主流程，而不是额外造一条平行链路
- 对历史资料保持尊重，但不过度绑定历史实现
- 在交付结果时说清楚改了什么、为什么这么改、如何验证

如果任务涉及背景解释、历史材料整理、MATLAB/Abaqus 旧流程迁移，先读 [docs/background.md](docs/background.md) 和 `references/fjw_work/`，再动手。
