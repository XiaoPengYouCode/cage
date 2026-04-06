# datasets/

这个目录存放 **可复用的数据资产**，主要是 `.npz`，也允许放正式工作流会消费的 `.mat` 示例输入。

## 目录结构

- `topopt/`
  - 拓扑优化概率采样链路使用的输入数据
- `voxel/`
  - 仅在需要时临时生成的体素几何输入

## 放置原则

放这里的文件应该满足至少一条：
- 会被多个脚本或 package 反复消费
- 是正式工作流的输入或中间结果
- 不是单纯为了文档展示而存在

## 当前保留的数据

- `datasets/topopt/fake_density_annular_cylinder_200x200x80.npz`
  - 当前拓扑采样主线默认使用的伪拓扑优化密度输入
- `datasets/topopt/example_fake_density_annular_cylinder_200x200x80.mat`
  - 由同名 `.npz` 中的 `density_milli` 三维矩阵直接导出，作为 Matlab `.mat` 输入示例

## 不放这里的内容

- 文档插图：放 `docs/assets/`
- 一次性实验图：优先放 `docs/assets/`
- 代码：放 `packages/` 或 `experiments/`
