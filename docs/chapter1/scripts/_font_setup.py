"""中文字体配置，在所有 figure 脚本开头 import 即可。"""
import matplotlib
import matplotlib.pyplot as plt

# Try macOS system fonts first, then fallback
for font in ["STHeiti", "PingFang SC", "Heiti SC", "Arial Unicode MS",
             "Hiragino Sans GB", "SimHei", "WenQuanYi Micro Hei"]:
    try:
        matplotlib.font_manager.findfont(font, fallback_to_default=False)
        plt.rcParams["font.family"] = font
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue
