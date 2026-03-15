# Cube Voronoi Demo

使用 `uv` 管理依赖，并通过 `matplotlib` 绘制一个限定在正方体边界内的三维 Voronoi 图。种子点默认在单位正方体 `[0, 1]^3` 内做均匀采样。

当前默认输出是一个 `3 x 5` 图网格：

- 三行：三组不同的随机种子
- 第一列：表层染色的 Voronoi 单元
- 第二、三列：圆柱杆版本的杆件单元与 `3 x 3 x 3` 平铺阵列
- 第四、五列：螺旋杆版本的杆件单元与 `3 x 3 x 3` 平铺阵列

代码结构现在按轻量 pipeline 拆分，方便切换不同的杆件实体化风格：

- `cage/pipeline.py`：流程节点，负责采样、Voronoi 构建、边提取
- `cage/voronoi.py`：Voronoi 几何和边/面提取
- `cage/helix.py`：螺旋中心线、连续截面框架、管状网格生成
- `cage/rods.py`：杆件实体化风格，当前实现包含圆柱杆和螺旋杆
- `cage/rendering.py`：三列视图和整张图网格渲染
- `cage/cli.py`：命令行入口和配置组装
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
