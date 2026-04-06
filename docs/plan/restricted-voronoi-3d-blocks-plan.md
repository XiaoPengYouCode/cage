# Restricted Voronoi 3D Blocks Plan

## Goal

把当前“中空圆柱边界上的表面 Voronoi 分块”推进成“中空圆柱域内部的 3D restricted Voronoi 体块分解”。

目标块定义为：

```text
Cell_i = { x ∈ Ω | ||x - s_i|| <= ||x - s_j||, for all j }
```

其中 `Ω` 是中空圆柱域，`s_i` 是第 `i` 个 seed。

## Desired Output

- 每个 seed 对应一个真实的 3D 受限 Voronoi 体块
- 体块边界允许同时包含：
  - Voronoi 平分平面
  - 外圆柱面
  - 内圆柱面
  - 顶面
  - 底面
- 可用于：
  - 可视化
  - 体块统计
  - 后续导出或分解

## Key Observation

- 在全空间里，Voronoi 分界仍然是平面
- 难点不在 Voronoi 本身，而在 `Voronoi cell ∩ annular cylinder domain`
- 由于域 `Ω` 不是凸集，单个受限 cell 可能不是简单凸多面体
- 与内孔相交后，单个 cell 可能出现带洞或多连通现象

## Recommended Phases

### Phase 1. Keep Surface Blocks as Validation Layer

- 保留当前外壁、内壁、顶面、底面的表面分块
- 用它检查 3D 体块和边界投影是否一致
- 不把表面图当成最终 3D 分解

### Phase 2. Build Full Voronoi Halfspace Description Per Seed

- 对每个 seed `i`，收集所有 `j != i` 的 Voronoi 半空间约束
- 形式为：

```text
2 (s_j - s_i) · x <= ||s_j||^2 - ||s_i||^2
```

- 得到全空间 Voronoi cell 的半空间表示

### Phase 3. Encode Domain Boundary Analytically

中空圆柱域 `Ω` 由以下约束组成：

- `z_min <= z <= z_max`
- `r <= outer_radius`
- `r >= inner_radius`

其中：

```text
r^2 = (x - cx)^2 + (y - cy)^2
```

注意：

- 外圆柱约束是凸的
- 内孔约束不是凸半空间，不能直接并入普通 halfspace intersection

### Phase 4. Split the Domain Into Convex Pieces

推荐不要直接对整个非凸域做一次布尔求交。

更稳的路线是：

1. 先把中空圆柱域按角度或按参考切平面切成若干无洞凸块
2. 在每个凸块里做：
   - Voronoi halfspaces
   - 外圆柱近似或解析边界
   - 顶底平面
3. 再把属于同一 seed 的子块合并

这样做的原因：

- 凸域内的半空间裁剪更稳定
- 数据结构简单
- 更容易调试每一步失败点

### Phase 5. Mesh Representation

体块表示建议分两层：

1. 分析表示
   - 保存 seed id
   - 保存约束来源
   - 保存拓扑邻接

2. 渲染表示
   - 三角网格或四边形网格
   - 用于预览和导出

不要一开始只做渲染 mesh，否则很难检查数学正确性。

### Phase 6. Validation

至少做这几类校验：

- 每个表面分块应与当前 surface Voronoi blocks 一致
- 任意体素采样点归属应与最近 seed 一致
- 所有体块并集应覆盖 `Ω`
- 不同 seed 体块内部不应重叠
- 总体积应接近 `Ω` 的体积

## Suggested Implementation Order

1. 先建立每个 seed 的 Voronoi halfspace 数据结构
2. 再把中空圆柱切成可处理的凸子域
3. 在子域内做单 seed 裁剪
4. 合并同 seed 子块
5. 做表面一致性检查
6. 最后接渲染与导出

## Risks

- 直接处理非凸内孔会让布尔和裁剪逻辑变得很脆
- 只保留三角网格会丢失“为什么这个块是这样”的可解释性
- 2000 个 seed 的全对全关系很重，需要邻域裁剪或候选加速

## Practical Recommendation

第一版 3D restricted blocks 不要追求一次到位的完美解析曲面布尔。

更合适的是：

- 先得到数学归属正确的子块拓扑
- 再逐步提高几何边界的解析程度和渲染质量

这样能更快得到可验证、可迭代的 3D 分块主链路。
