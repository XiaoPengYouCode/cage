# datasets/

这个目录存放 **可复用的数据资产**，主要是 `.npz`。

## 目录结构

- `topopt/`
  - 拓扑优化到 Voronoi backfill 链路使用的输入/中间数据
- `voxel/`
  - 体素 demo 与体素几何输入数据

## 放置原则

放这里的文件应该满足至少一条：
- 会被多个脚本或 package 反复消费
- 是正式工作流的输入或中间结果
- 不是单纯为了文档展示而存在

## 不放这里的内容

- 文档插图：放 `docs/assets/`
- 一次性实验图：优先放 `docs/assets/`
- 代码：放 `packages/` 或 `experiments/`
