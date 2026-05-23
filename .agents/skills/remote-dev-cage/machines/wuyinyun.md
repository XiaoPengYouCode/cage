# wuyinyun

## 基本信息

| 项目 | 值 |
|---|---|
| SSH 别名 | `wuyinyun` |
| 用户 | `zhongjin_lu` |
| 主机名 | `pntlgaolhyg69qw` |
| 推荐仓库路径 | `~/project/cage` |
| 本地仓库远端 | `https://github.com/XiaoPengYouCode/cage.git` |
| Python 环境 | `uv`，必要时先 `source ~/.local/bin/env` |
| 主要用途 | FJW / SfePy Python-only 一轮优化和远程验证 |

## 连接检查

```bash
ssh wuyinyun "whoami && hostname && pwd"
```

## 仓库初始化

```bash
ssh wuyinyun "mkdir -p ~/project && cd ~/project && git clone https://github.com/XiaoPengYouCode/cage.git"
```

## 更新到本地已推送提交

```bash
ssh wuyinyun "cd ~/project/cage && git fetch origin && git switch main && git pull --ff-only"
```

## 一轮优化命令

```bash
ssh wuyinyun "source ~/.local/bin/env 2>/dev/null || true; cd ~/project/cage && uv run fem-analysis fjw-optimize --backend sfepy --mode three-force --max-iterations 1 --num-time-steps 1 --sfepy-linear-solver scipy_iterative --run-directory runs/fjw_optimize"
```

## 输出验证命令

```bash
ssh wuyinyun "source ~/.local/bin/env 2>/dev/null || true; cd ~/project/cage && uv run fem-analysis fjw-validate --run-directory runs/fjw_optimize"
```
