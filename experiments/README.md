# experiments/

这个目录只放 **实验性脚本**，不承载正式 package 的主流程。

## 目录结构

- `topopt_backfill/`
  - 围绕拓扑优化密度场、seed cloud、template backfill 的预览/调参/可视化实验
- `voxel_demos/`
  - 体素 toy demo 与说明性脚本

## 约束

- 正式工作流已经进入：
  - `src/helix_voronoi`
  - `src/topopt_backfill`
- 如果某个实验脚本已经被正式 CLI 覆盖，就应当删除，而不是继续在这里保留一份平行实现
- 如果脚本产生的是可复用输入数据，请把 `.npz` 放到 `datasets/`
- 如果脚本产生的是文档图片，请把 `.png/.jpg` 放到 `docs/assets/`
