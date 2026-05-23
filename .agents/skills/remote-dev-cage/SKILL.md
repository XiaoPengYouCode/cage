---
name: remote-dev-cage
description: cage / FJW workflow 远程开发机运维。管理 wuyinyun SSH、远程 checkout、uv sync、FJW Python-only 一轮优化、结果验证和日志检查。当用户提到 wuyinyun、远程机器、远端优化、一轮迭代优化、FJW optimize、SSH 执行时触发。
when_to_use: wuyinyun / 远程开发 / SSH / cage 远端运行 / FJW 优化 / fjw-optimize / 一轮迭代 / uv sync / 远程日志 / GPU 机器
user-invocable: true
---

# cage 远程开发机运维

这个 skill 只服务当前 `cage` 仓库。目标是让远程运行能追溯到明确 git commit，并且优先跑 Python-only FJW smoke optimization。

每台机器的具体参数放在 `machines/` 子目录。当前已注册：

| SSH 别名 | 用途 | 说明 |
|---|---|---|
| `wuyinyun` | FJW / SfePy 远程计算 | `machines/wuyinyun.md` |

## 基本规则

- 源码同步只走 git：本地提交、推送，远程 `git fetch` / `git pull --ff-only` 更新。
- 不用 `rsync` / `scp` 覆盖源码文件。
- 远程产物默认写入 `runs/`，不提交一次性运行输出。
- Python-only 验证优先用 `sfepy` backend 和 `scipy_iterative` solver。
- 需要依赖下载时，先检查远程网络；需要代理时按机器文档建立隧道。

## 标准远程流程

### 1. 确认机器和仓库

```bash
ssh wuyinyun "whoami && hostname && nvidia-smi --query-gpu=name --format=csv,noheader || true"
ssh wuyinyun "test -d ~/project/cage/.git && echo cage-present || echo cage-missing"
```

如果远程没有仓库：

```bash
ssh wuyinyun "mkdir -p ~/project && cd ~/project && git clone https://github.com/XiaoPengYouCode/cage.git"
```

如果远程已有仓库：

```bash
ssh wuyinyun "cd ~/project/cage && git fetch origin && git switch main && git pull --ff-only"
```

### 2. 安装或更新依赖

```bash
ssh wuyinyun "source ~/.local/bin/env 2>/dev/null || true; cd ~/project/cage && uv sync"
```

### 3. 运行一轮 FJW smoke optimization

标准命令：

```bash
ssh wuyinyun "source ~/.local/bin/env 2>/dev/null || true; cd ~/project/cage && uv run fem-analysis fjw-optimize --backend sfepy --mode three-force --max-iterations 1 --num-time-steps 1 --sfepy-linear-solver scipy_iterative --run-directory runs/fjw_optimize"
```

输出成功时应包含：

- `"backend": "sfepy"`
- `"iteration_count": 1`
- `"run_directory": ".../runs/fjw_optimize"`
- `"manifest_path": ".../workflow_manifest.json"`

### 4. 验证输出结构

```bash
ssh wuyinyun "source ~/.local/bin/env 2>/dev/null || true; cd ~/project/cage && uv run fem-analysis fjw-validate --run-directory runs/fjw_optimize"
```

验证通过时，报告会确认 checkpoint、`iteration_state.json`、`design_cage.npz`、`mma_state.npz` 和 `aggregate_terms.npz` 等结构存在。没有历史 golden 时，可以只作为结构验证。

## 常用排障

### uv 找不到

先加载 uv 的 shell env：

```bash
ssh wuyinyun "source ~/.local/bin/env && uv --version"
```

### git pull 失败

先看远程状态，避免覆盖远程手工改动：

```bash
ssh wuyinyun "cd ~/project/cage && git status --short --branch"
```

如果远程只有未跟踪运行产物，保持它们在 `runs/` 或清理后再 `git pull --ff-only`。

### SfePy 或 SciPy 求解失败

先跑 preflight：

```bash
ssh wuyinyun "source ~/.local/bin/env 2>/dev/null || true; cd ~/project/cage && uv run fem-analysis fjw-preflight"
```

再单独构建 direct problem：

```bash
ssh wuyinyun "source ~/.local/bin/env 2>/dev/null || true; cd ~/project/cage && uv run fem-analysis fjw-direct --load-case force_1 --build-problem --sfepy-linear-solver scipy_iterative"
```
