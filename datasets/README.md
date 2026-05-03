# datasets/

这个目录存放完整 pipeline 会复用的数据资产，主要是 `.npz`，也允许放正式工作流会消费的 `.mat` 示例输入。

## 目录结构

- `topopt/`
  - 密度场、seed points、Voronoi 中间结果和可复用示例数据
- `voxel/`
  - 需要时临时生成的解析 demo 体素几何
- `fjw_golden/`
  - 小体积、可审查的 FJW 回归 fixture 和 golden manifest

## 放置原则

放这里的文件应该满足至少一条：

- 会被多个脚本或 package 反复消费
- 是完整 pipeline 的输入或中间标准数据
- 是测试、验证或文档需要的可复用 fixture

## 当前常用数据

- `datasets/topopt/fake_density_annular_cylinder_200x200x80.npz`
  - `topopt_sampling` smoke demo 使用的假密度输入
- `datasets/topopt/example_fake_density_annular_cylinder_200x200x80.mat`
  - 由同名 `.npz` 的 `density_milli` 导出，用于验证 `.mat` 输入读取
- `datasets/topopt/seed_probability_mapping_2000.npz`
  - annular-cylinder demo 的 seed sampling 输出
- `datasets/topopt/fjw_reference_fem_voxels.npz`
  - 从 FJW reference 网格整理出的标准体素数据
- `datasets/681.mat`
  - 681 downstream geometry pipeline 使用的历史或实验伪密度快照

## 不放这里的内容

- 文档插图：放 `docs/assets/`
- 一次性实验输出：优先放 `outputs/`、`experiment_output/` 或 `runs/`
- 代码：放 `src/`
