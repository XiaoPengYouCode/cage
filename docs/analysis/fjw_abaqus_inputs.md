# FJW Abaqus Structured Inputs

这次我把 `references/fjw_work/` 里的 Abaqus 模板尾段和 MATLAB 里的离散化规则整理成了一份机器可读输入：

- [datasets/fjw_abaqus_inputs.json](/Users/flamingo/Projects/cage/datasets/fjw_abaqus_inputs.json)

生成脚本在这里：

- [scripts/extract_fjw_inp_inputs.py](/Users/flamingo/Projects/cage/scripts/extract_fjw_inp_inputs.py)

## 为什么先用 JSON

我建议当前阶段先用 `JSON`，不要先上 `YAML`。

原因很直接：

- 仓库现有机器输出本来就以 `json` 为主，风格一致。
- Python 直接用标准库就能读写，零新依赖。
- 这份数据是给后续 Python 求解器吃的，机器稳定性比手工编辑体验更重要。
- 这里面有很多规则化的重复材料桶，`JSON` 更适合当“编译后的标准化产物”。

如果后面你确定要让人频繁手改载荷工况、材料表、边界条件，再加一层 `YAML -> JSON` 的作者态配置也不晚。现在先把可执行输入定下来，推进速度更好。

## 已经结构化进去的内容

`datasets/fjw_abaqus_inputs.json` 现在已经包含：

- `assembly_controls`
  - 两个参考点结点
  - `M_SET-1` / `M_SET-2`
  - 顶底面耦合 surface
  - 两个 kinematic coupling
- `boundary_conditions`
  - `M_SET-1` 的 `ENCASTRE`
- `forward_load_cases`
  - `force_1`: `M_SET-2`, dof `3`, magnitude `-1200`
  - `force_2`: `M_SET-2`, dof `4`, magnitude `-7500`
  - `force_3`: `M_SET-2`, dof `5`, magnitude `-7500`
- `adjoint_load_template`
  - `end_Fv_p1.inp` 和 `end_Fv_p2.inp`
  - 中间动态插入 `Fv_set` 的规则
- `materials`
  - `BONE_COR`
  - `CAGE_0..100`
  - `BONE_0..10`
- `section_assignments`
  - `NODESI_ELE_COR -> BONE_COR`
  - `NODESI_ELE_TRA -> BONE_1`
  - `DESI_E_ELE{0..100} -> CAGE_{0..100}`
  - `OBJ_E_ELE{0..10} -> BONE_{0..10}`
- `simulation_constants`
  - `dt`
  - `P`
  - `b_max`
  - `E0_bo`, `Emin_bo`
  - `E0_cage`, `Emin_cage`
  - 初始 `design_cage` / `obj_bo`
- `discretization_rules`
  - `E_cage -> CAGE_i` 的分桶规则
  - `E_obj -> BONE_i` 的分桶规则
  - `obj_bo -> OBJ_BO_ELEi` 的分组规则

## 这份输入还没覆盖的部分

当前还需要补三类运行期数据：

- `TOP_NOD` / `BOT_NOD` 的完整结点成员
  - 这部分来自更早的网格与基础 `.inp` 生成阶段
  - 当前 JSON 只保留了耦合关系和引用名
- `DESI_E_ELE*` / `OBJ_E_ELE*` / `NODESI_ELE_*` 的具体单元成员
  - 它们来自 `.mat` 数据和当前时刻状态量
  - 它们属于运行期状态，不适合手写死在配置里
- 伴随载荷 `Fv`
  - 它是每个时间步反向链路里算出来的动态向量
  - 只能记录“怎么插入”，不能静态列出数值

## 对替代 Abaqus 的意义

这份 JSON 已经把“尾部模板语法”剥干净了，后面如果我们自己写 Python 求解器，至少这几层不用再猜：

- 固定边界怎么施加
- 三个正向工况怎么定义
- 伴随工况怎么拼
- 材料桶和 section 怎么对应
- MATLAB 里的状态量怎样离散到材料桶

下一步真正该补的是两块：

1. 把 `BOT_NOD` / `TOP_NOD` 的结点集合，从基础网格阶段也提成标准化输入。
2. 把 `desi_ele` / `obj_ele` / `cor_ele` / `tra_ele` 这些 `.mat` 集合，整理成 Python 侧统一数据模型。

做到这一步以后，Abaqus 在数据接口层面就基本没有黑箱了。
