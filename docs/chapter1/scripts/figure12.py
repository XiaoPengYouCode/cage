"""
Figure 1-12: 算法全流程 Pipeline 总览图
输出: docs/chapter1/figures/1-12-pipeline-overview.png
"""

import _font_setup  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(18, 7))
ax.set_xlim(0, 18)
ax.set_ylim(0, 7)
ax.axis("off")

# ── Module definitions ──────────────────────────────────────────────────────
modules = [
    {
        "title": "密度场\n预处理",
        "subtitle": "概率场构建\n逆变换采样",
        "params": r"$\gamma=1,\ N$",
        "color": "#AED6F1",
        "x": 0.3,
    },
    {
        "title": "Lloyd\nCVT 迭代",
        "subtitle": "镜像边界\n质心更新",
        "params": r"$k_{\max}=500$",
        "color": "#A9DFBF",
        "x": 3.3,
    },
    {
        "title": "受限 Voronoi\n图构建",
        "subtitle": "HalfspaceIntersection\n共面面合并",
        "params": r"$\mathcal{B}$",
        "color": "#F9E79F",
        "x": 6.3,
    },
    {
        "title": "Voronoi\n脊线提取",
        "subtitle": "ridge_vertices\n有序顶点环",
        "params": r"$\mathcal{E}_{\mathrm{ridge}}$",
        "color": "#FAD7A0",
        "x": 9.3,
    },
    {
        "title": "骨架体素化\n形态学膨胀",
        "subtitle": "线段光栅化\n球形膨胀",
        "params": r"$r_{\mathrm{dil}}=3.0$",
        "color": "#D7BDE2",
        "x": 12.3,
    },
    {
        "title": "Marching\nCubes",
        "subtitle": "Gaussian 预平滑\n等值面提取",
        "params": r"$\sigma=1.0,\ \ell=0.5$",
        "color": "#F1948A",
        "x": 15.3,
    },
]

BOX_W = 2.6
BOX_H = 3.2
BOX_Y = 1.8

for m in modules:
    x = m["x"]
    # Box
    box = FancyBboxPatch((x, BOX_Y), BOX_W, BOX_H,
                          boxstyle="round,pad=0.12",
                          facecolor=m["color"], edgecolor="#555", linewidth=1.2)
    ax.add_patch(box)
    # Title
    ax.text(x + BOX_W/2, BOX_Y + BOX_H - 0.38,
            m["title"], ha="center", va="center",
            fontsize=11, fontweight="bold", color="#222")
    # Subtitle
    ax.text(x + BOX_W/2, BOX_Y + BOX_H/2,
            m["subtitle"], ha="center", va="center",
            fontsize=8.5, color="#444", linespacing=1.5)
    # Params
    ax.text(x + BOX_W/2, BOX_Y + 0.28,
            m["params"], ha="center", va="center",
            fontsize=9, color="#666",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none"))

# Arrows between boxes
for m in modules[:-1]:
    x_start = m["x"] + BOX_W + 0.02
    x_end   = x_start + (3.0 - BOX_W) + 0.1
    y_mid = BOX_Y + BOX_H / 2
    ax.annotate("",
        xy=(x_end - 0.08, y_mid),
        xytext=(x_start, y_mid),
        arrowprops=dict(arrowstyle="-|>", color="#333",
                        lw=1.5, mutation_scale=16))

# Input / Output labels
ax.text(0.05, BOX_Y + BOX_H / 2, "SIMP\n伪密度场\n$\\rho$",
        ha="center", va="center", fontsize=9, color="#333",
        style="italic")
ax.annotate("", xy=(0.3, BOX_Y + BOX_H/2), xytext=(0.7, BOX_Y + BOX_H/2),
            arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5, mutation_scale=14))

last_x = modules[-1]["x"] + BOX_W
ax.annotate("", xy=(last_x + 0.65, BOX_Y + BOX_H/2), xytext=(last_x + 0.05, BOX_Y + BOX_H/2),
            arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5, mutation_scale=14))
ax.text(last_x + 0.92, BOX_Y + BOX_H/2, "GLB\nSTL",
        ha="center", va="center", fontsize=9, color="#333", style="italic")

# Step numbers
for k, m in enumerate(modules):
    ax.text(m["x"] + BOX_W/2, BOX_Y + BOX_H + 0.22,
            f"Step {k+1}", ha="center", va="bottom",
            fontsize=8, color="#666")

ax.set_title("图 1-12  算法全流程总览（Pipeline Overview）",
             fontsize=13, pad=14)
fig.tight_layout()

out_dir = __file__.replace("scripts/figure12.py", "figures")
import os; os.makedirs(out_dir, exist_ok=True)
fig.savefig(f"{out_dir}/1-12-pipeline-overview.png", dpi=150, bbox_inches="tight")
print("Saved 1-12-pipeline-overview.png")
