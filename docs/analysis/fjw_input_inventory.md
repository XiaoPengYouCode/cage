# FJW Input Inventory

这份清单对应的是“把所有输入整理好”这件事。

机器可读版本在这里：

- [datasets/fjw_input_inventory.json](/Users/flamingo/Projects/cage/datasets/fjw_input_inventory.json)

生成脚本在这里：

- [scripts/build_fjw_input_inventory.py](/Users/flamingo/Projects/cage/scripts/build_fjw_input_inventory.py)

## 结论先说

如果只看这套 MATLAB + Abaqus 历史流程，静态输入已经齐了。

现在仓库里已经有：

- 网格坐标 `nod_coo.mat`
- 单元连接 `ele_nod.mat`
- 单元矩阵 `B_3d.mat`、`D_3d.mat`
- 各个区域的单元集合
- 顶底结点集 `TOP_NOD` / `BOT_NOD`
- 三个正向工况模板
- 伴随工况模板
- 材料分桶和 section 规则
- 初始骨状态、初始 cage 状态、时间步、材料参数

所以“还缺输入文件吗”这个问题，答案是：当前没有缺静态外部输入文件。

## 输入分层

这套算法的输入可以分成三层。

### 1. 固定外部输入

这部分是你在运行前就要准备好的，仓库里现在都在：

- 网格与区域文件
  - `nod_coo.mat`
  - `ele_nod.mat`
  - `cage_ele.mat`
  - `desi_ele.mat`
  - `obj_ele.mat`
  - `cor_ele.mat`
  - `tra_ele.mat`
- 力学核
  - `B_3d.mat`
  - `D_3d.mat`
- 模板和工况
  - `ini.inp`
  - `end1.inp`
  - `end2.inp`
  - `end3.inp`
  - `end_Fv_p1.inp`
  - `end_Fv_p2.inp`
- 结点集和边界条件
  - `TOP_NOD`
  - `BOT_NOD`
  - `M_SET-1` 固定
  - `M_SET-2` 施加载荷

### 2. 初始状态输入

这部分由固定常量和区域集合直接生成：

- 初始 `design_cage(0)`
  - 单工况默认 `0.2`
  - 三工况默认 `0.3`
- 初始 `obj_bo(0)`
  - 默认 `0.36`
- 网格尺寸
  - `152 x 131 x 134`
- 时间步参数
  - `dt = 1`
  - `P = 3`

### 3. 运行期状态量

这部分属于算法在推演中生成的运行期状态：

- `E_obj(t)`
- `E_cage(t)`
- `U(t)`
- `bone_s(t)`
- `obj_bo(t+1)`
- `Fv(t)`
- `V(t)`

你之前一直觉得“是不是还有输入没凑齐”，很多困惑就卡在这里：这些量看起来像输入，实际身份是中间状态。

## 已经整理出来的关键数字

目前已经确认：

- `nod_coo`: `593790 x 3`
- `ele_nod`: `544112 x 8`
- `cage_ele`: `23807`
- `desi_ele`: `18734`
- `obj_ele`: `5073`
- `cor_ele`: `90207`
- `tra_ele`: `430098`
- `TOP_NOD` 结点数：`3936`
- `BOT_NOD` 结点数：`5189`

## 现在真正还差什么

现在缺的是 Python 化落地工作：

1. 把这些 `.mat` 和结点集读进统一的数据结构。
2. 在 Python 里重建当前 Abaqus 模型的装配关系。
3. 用未来的 Python 求解器对齐 Abaqus 基线结果。

所以如果你问我一句最直白的话：

当前这套历史算法，输入已经整理齐了；接下来该做的是“怎么把这些输入接进新的 Python 实现”。  
