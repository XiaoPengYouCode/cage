# Cube Voronoi Demo

使用 `uv` 管理依赖，并通过 `matplotlib` 绘制一个限定在正方体边界内的三维 Voronoi 图。种子点默认在单位正方体 `[0, 1]^3` 内做均匀采样。

当前默认输出是一个 `3 x 5` 图网格：

- 三行：三组不同的随机种子
- 第一列：表层染色的 Voronoi 单元
- 第二、三列：圆柱杆版本的杆件单元与 `3 x 3 x 3` 平铺阵列
- 第四、五列：螺旋杆版本的杆件单元与 `3 x 3 x 3` 平铺阵列

代码结构现在按轻量 pipeline 拆分，方便切换不同的杆件实体化风格，以及后续继续接模量分析：

- `src/cage/pipeline.py`：流程节点，负责采样、Voronoi 构建、边提取
- `src/cage/voronoi.py`：Voronoi 几何和边/面提取
- `src/cage/helix.py`：螺旋中心线、连续截面框架、管状网格生成
- `src/cage/rods.py`：杆件实体化风格，当前实现包含圆柱杆和螺旋杆
- `src/cage/analysis/`：单胞模量分析子系统，包含几何、体素化/六面体网格、`SfePy` 线弹性求解和报告输出
- `src/cage/rendering.py`：三列视图和整张图网格渲染
- `src/cage/cli.py`：命令行入口和配置组装
- `tests/`：最小几何回归测试

运行：

```bash
uv run python main.py
```

或者使用包入口：

```bash
uv run cage
```

输出图片默认保存为：

```text
docs/assets/voronoi_cube_3d.png
```

如果需要直接弹出 matplotlib 窗口：

```bash
uv run python main.py --show
```

调整种子点数量和三组随机种子：

```bash
uv run python main.py --num-seeds 10 --row-seeds 116 55 49
```

运行测试：

```bash
uv run python -m unittest discover -s tests -v
```

模量分析命令：

```bash
uv run cage modulus --seed 55 --style both
```

当前模量分析后端为 `SfePy`，流程是：

- 先把直杆或螺旋杆单胞体素化
- 再转成规则 `Hex8` 六面体网格
- 最后在 `SfePy` 中施加上下完全粘结压板位移边界，求 `Z` 向等效模量

如果只想验证参数接线、不启动实际求解：

```bash
uv run cage modulus --seed 55 --style both --dry-run
```
