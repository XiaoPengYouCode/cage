# Post process analysis

这个目录放论文结果整理、指标汇总和后续计算脚本。

当前内容：

- `summarize_iter017.py`：汇总 `runs/fjw_optimize_real/iter_017` 与相关验证产物，输出可直接给论文引用的 JSON 摘要
- `summarize_iter017_case_histories.py`：汇总 `iter_017` 三工况 `case_history.npz` 中当前可用的时间步骨量历史与分位数统计
- `summarize_iter017_environment.py`：汇总 `iter_017` 对应运行目录中保留下来的远端执行环境、线程配置、运行时段与资源观测信息
- `summarize_iteration_completeness.py`：扫描 `runs/fjw_optimize_real/iter_*` 的关键 checkpoint、forward 和 adjoint 产物是否存在，输出多轮完整性摘要
- `summarize_thesis_gaps.py`：扫描论文主文档中的 `#todo`、待填占位符、附录待填项和剩余图表缺口，输出按章节整理的论文缺口摘要
- `summarize_thesis_readiness.py`：基于论文缺口摘要生成按章节的完成度报告，区分已完成、部分完成和未完成
- `summarize_chapter1_literature.py`：按第 1 章建议的四个主题整理可直接进入研究现状的文献底稿
- `summarize_load_case_interpretation.py`：整理 `force_1/2/3` 的参考点载荷向量、自由度含义、物理解释和 `iter_017` 响应量级
- `summarize_iter017_solver_traces.py`：从真实 heartbeat 日志和 runtime 事件中提取 `iter_017` 单工况残差序列、matrix/solve 时间和 forward/adjoint 用时
- `generate_iter017_cvt_seeds.py`：基于 `iter_017` 的初始 seeds 重新执行 500 轮 Lloyd，生成 `fjw_iter017_seeds_200_gamma1_cvt500.npz`
- `summarize_available_iterations.py`：扫描 `runs/fjw_optimize_real` 中当前足够完整的 iteration 目录，输出可用于趋势图与正文说明的 `available_iterations_summary.json`
- `build_iter017_variable_radius_edges.py`：把 `iter_017` 的 Voronoi 边中点回采样到对齐密度场，按当前 `iter017_band_radius_lookup_combined_seed55_plus_lowmid.json` 为每条边附上 `target_modulus`、`band_index` 和 `assigned_radius_mm`
- `build_iter017_variable_radius_skeleton.py`：基于带半径场的边工件生成整结构 variable-radius 骨架体素，并可直接导出 smoke 级 GLB / STL
- `run_remote_radius_calibration_wide_v3.sh`：在 `wuyinyun` 上启动下一轮高模量端半径扩展标定
- `build_iter017_variable_radius_replacement_design.py`：把 variable-radius 骨架细体素聚合回 coarse 设计网格，按 `iter017_band_radius_lookup_combined_seed55_plus_lowmid.json` 中的稳定 `r -> E_eff` 支撑曲线生成 FE-ready `replacement_design_cage`，并支持 `mean_only`、`fill_scaled`、`local_support` 三种 coarse 聚合模式
- `run_remote_variable_radius_forward_compare.sh`：在 `wuyinyun` 上对指定 load case 直接跑 `modulus_weighted` replacement forward compare
- `run_remote_variable_radius_pipeline_and_compare.sh`：在 `wuyinyun` 上串行执行 variable-radius edges -> skeleton -> replacement design -> forward compare 整条链

当前输出：

- `output/iter017_summary.json`
- `output/iter017_case_history_summary.json`
- `output/iter017_environment_summary.json`
- `output/iteration_completeness_summary.json`
- `output/thesis_gap_summary.json`
- `output/thesis_readiness_summary.json`
- `output/chapter1_literature_summary.json`
- `output/load_case_interpretation_summary.json`
- `output/iter017_solver_trace_summary.json`
- `output/available_iterations_summary.json`
- `output/voronoi_radius_calibration_summary.json`
- `output/iter017_band_radius_lookup.json`
- `output/iter017_band_radius_lookup_combined_seed55_plus_lowmid.json`
- `output/iter017_modulus_proxy_gap_seed55_plus_lowmid_summary.json`

当前与标定层直接相关的整结构工件：

- `outputs/fjw_optimize_real_iter017/fjw_iter017_voronoi_edges_variable_radius.npz`
- `outputs/fjw_optimize_real_iter017/fjw_iter017_voronoi_edges_variable_radius_seed55_plus_lowmid.npz`
- `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_voxels_variable_radius.npz`
- `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_voxels_variable_radius_smoke.npz`
- `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_variable_radius_smoke.glb`
- `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_variable_radius_smoke.stl`
- `outputs/fjw_optimize_real_iter017/fjw_iter017_replacement_design_variable_radius.npz`
- `outputs/fjw_optimize_real_iter017/fjw_iter017_replacement_design_variable_radius_seed55_plus_lowmid.npz`
- `outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_vs_density_modulus_weighted_force_1_seed55_plus_lowmid_comparison.json`

当前这条链路的物理口径：

- 上游 `design_cage` 先通过 `E_target = E_min + E_0 x^3` 转成目标等效模量场
- 中间用 `iter017_band_radius_lookup_combined_seed55_plus_lowmid.json` 把 `E_target` 映射成 bandwise 杆半径
- 下游再把 variable-radius 骨架聚合回 coarse FE 设计网格
- `design_cage_fill_fraction` 只是探索性几何指标
- `design_cage_modulus_weighted` 才是当前第一版 FE-ready 代理场，用来承载按杆径标定得到的局部等效模量

当前结论也要写死：

- `fill fraction` 不能当成最终学术验证对象
- `seed55_plus_lowmid` lookup 已经把 dominant `band 3` 从低端夹紧修正为真实插值，边级 low-clamp 比例降到约 `0.50%`
- `modulus_weighted` 只是第一版可运行代理，不是最终均匀化闭环；当前 `force_1` 远端比较仍然不等效，最大位移比约 `1.9769`，骨密度增量总和仍然反号
- 新增的 `local_support` 聚合模式用于验证 coarse 网格支撑体积是否是当前失败主因；它是下一轮远端对照候选，不是最终均匀化张量
- 下一步重点不是继续证明 fill fraction，而是改进 coarse 聚合、高模量支撑和局部各向异性等效刚度
