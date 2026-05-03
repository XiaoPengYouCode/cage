# Background

这个仓库围绕一条 cage 设计 pipeline 组织：先用 FJW / `fem_analysis` workflow 得到 cage 的最终伪密度分布，再基于伪密度采样 seed points，生成 Voronoi / restricted Voronoi 几何，最后把几何实体化为螺旋杆或骨架 mesh，并导出 STL、GLB、图片和验证报告。

完整数据流可以概括为：

```text
FJW optimization workflow
  -> final pseudo-density / design density
  -> standardized density NPZ
  -> probability field
  -> seed points
  -> Voronoi / restricted Voronoi structure
  -> helix rods or scaffold mesh
  -> viewer / STL / GLB / analysis
```

仓库中的模块是这条 pipeline 的不同节点：

- `src/fem_analysis/`：FJW 三工况骨改建优化、SfePy/SciPy 求解、动态伴随、MMA 和 validation。
- `src/topopt_sampling/`：密度场到概率场、seed points、restricted Voronoi 和 shell GLB。
- `src/helix_voronoi/`：Voronoi 杆系、螺旋杆样式、预览和 STL 导出。
- `src/matlab2stl_pipeline/`：681 `.mat` 到 Voronoi 骨架 STL / GLB 的端到端几何节点。
- `src/ct_reconstruction/`：STL、体素、NPZ 和 GLB 之间的辅助转换。

## References

`references/fjw_work/` 目录中保存的是我同门师兄 **方嘉纬学长** 的工作资料。

这些文件作为背景参考材料保留在仓库中，主要用途是：

- 了解前期工作的输入输出格式和脚本组织方式
- 对照已有的 `.mat`、`.inp`、`.m` 等文件理解历史流程
- 为后续整理、迁移和复现实验提供参考

这部分内容是参考资料归档，不代表当前主实现入口。当前可维护、可继续演进的实现以仓库现有 Python 代码、CLI、`README.md` 和 `docs/` 中的当前说明为准。

## FJW Algorithm Summary

这套算法先根据当前 cage 材料分布、骨区网格、初始骨状态、材料参数和受力条件，向后推演若干步骨改建响应。它回答的问题是：如果今天这样设计，未来骨组织会受到什么刺激，哪些区域更容易长骨，最终骨量会怎样变化。

随后算法反向计算未来骨量变化和当前 cage 各位置材料分布之间的关系。它会给出每个设计位置的敏感性：哪些位置增加材料更有收益，哪些位置收益有限，哪些位置可以减少材料。

优化器再根据敏感性小步更新 cage 材料分布，重新做正向推演和反向分析。这个闭环持续运行，直到设计逐渐收敛到更利于最终骨生长的伪密度分布。这个最终伪密度分布就是后续 Voronoi seed sampling 和螺旋杆 cage 生成的上游输入。

历史接触统计口径是：统计 cage 和周围骨之间的相邻接触面中，有多少落在松质骨上，有多少落在皮质骨上。当前整理到的数值是：

- cage-tra 接触面：2269
- cage-cor 接触面：315

可以表述为：cage 接触层以松质骨为主，约 87.8%；同时有一小部分接触到皮质骨，约 12.2%。
