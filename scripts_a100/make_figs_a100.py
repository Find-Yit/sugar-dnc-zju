import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

P = "/scratch/users/nus/e1351071/test_zju"
FD = f"{P}/repo/sugar-dnc-zju/results_a100/figures"
FONT = f"{P}/assets/fonts/NotoSansCJKsc-Regular.otf"
font_manager.fontManager.addfont(FONT)
fp = font_manager.FontProperties(fname=FONT)
plt.rcParams["font.family"] = fp.get_name()
plt.rcParams["axes.unicode_minus"] = False
VIEWS = ["00", "08", "16", "24"]
NAMES = {"00": "000001", "08": "000065", "16": "000129", "24": "000193"}

# ---------- (b) 2x4 碎片对比 ----------
tags = [("base_d10_q01", "原版 SuGaR (d10, q=0.1)\n2184 分量 / 2094 碎片"),
        ("m2p95_bg9_q0", "最优组合 M2-B+M1a (bg9, q=0)\n1216 分量 / 1194 碎片")]
fig, axes = plt.subplots(2, 4, figsize=(20, 6.2))
for r, (tag, lab) in enumerate(tags):
    for c, v in enumerate(VIEWS):
        ax = axes[r, c]
        ax.imshow(np.array(Image.open(f"{FD}/fragments_{tag}_view{v}.png")))
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(f"视角 {v} (测试图 {NAMES[v]})", fontproperties=fp, fontsize=13)
        if c == 0:
            ax.set_ylabel(lab, fontproperties=fp, fontsize=11)
fig.suptitle("连通分量着色对比：红=碎片(<100面)，灰=最大分量，浅色=其余分量", fontproperties=fp, fontsize=16)
fig.tight_layout(rect=[0, 0.02, 1, 0.95])
fig.savefig(f"{FD}/fig_fragments_compare_base_d10_q01_vs_m2p95_bg9_q0.png", dpi=300)
plt.close(fig)
print("OK b")

# ---------- (c) 天空封口证据图 ----------
cols = [("m_base_d10_q01", "原版 q=0.1\n天空/树冠处白斑 = 真实孔洞\n覆盖率 89.05%"),
        ("m_base_d10_q0", "q=0（不做分位清洗）\n大平面三角形把天空封住\n覆盖率 99.94%"),
        ("m_m2p95_bg9_q0", "最优组合 M2-B+M1a(bg9,q0)\n封口保留，碎片显著减少\n覆盖率 99.97%")]
fig, axes = plt.subplots(1, 3, figsize=(18, 4.4))
for ax, (run, lab) in zip(axes, cols):
    im = np.array(Image.open(f"{P}/outputs/vis/{run}/mesh_normal_view00_000001.png"))
    ax.imshow(im[0:int(im.shape[0] * 0.5), :])   # 视角 00 上半幅
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(lab, fontproperties=fp, fontsize=12)
fig.suptitle("天空封口证据（视角 00 / 测试图 000001 上半幅，mesh 面法向着色，白=无面片）",
             fontproperties=fp, fontsize=15)
fig.text(0.5, 0.015,
         "结论：q=0 使边界边由 38450 降到 11339（−70.5%），其下降的一大部分来自 Poisson 在天空区域生成的大平面封口面片，"
         "而非真正补全了几何；\n覆盖率 92.0%→99.97% 同理主要反映孔洞被封住。该指标不能证明重建质量提升。",
         ha="center", fontproperties=fp, fontsize=11)
fig.tight_layout(rect=[0, 0.10, 1, 0.93])
fig.savefig(f"{FD}/fig_sky_closure_evidence.png", dpi=300)
plt.close(fig)
print("OK c")

# ---------- (d) 扫参图 ----------
rows = list(csv.DictReader(open(f"{P}/outputs/metrics/summary_m.csv")))
x = [r["TAG"] for r in rows]
xi = np.arange(len(x))
specs = [("n_fragments_lt100", "碎片分量数 (<100 面)", "{:.0f}", 1.0),
         ("G1_median_abs", "G1 稀疏点→网格 中位距离", "{:.5f}", 1.0),
         ("largest_component_ratio", "最大分量面数占比 (%)", "{:.1f}", 100.0)]
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
for ax, (k, title, fmt, sc) in zip(axes, specs):
    y = [float(r[k]) * sc for r in rows]
    bars = ax.bar(xi, y, color="#4a7fb5")
    bars[x.index("base_d10_q01")].set_color("#b0b0b0")
    bars[x.index("m2p95_bg9_q0")].set_color("#c23b22")
    for i, v in enumerate(y):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontproperties=fp, fontsize=9)
    ax.set_title(title, fontproperties=fp, fontsize=13)
    ax.set_ylim(0, max(y) * 1.18)
    ax.grid(axis="y", alpha=0.3)
axes[-1].set_xticks(xi)
axes[-1].set_xticklabels(x, rotation=30, ha="right", fontproperties=fp, fontsize=10)
axes[-1].set_xlabel("配置名（灰=原版 q0.1 基线，红=最优组合）", fontproperties=fp, fontsize=12)
fig.suptitle("提取端扫参：碎片数 / G1 中位 / 最大分量占比 随配置变化", fontproperties=fp, fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{FD}/fig_sweep_fragments_G1_largest.png", dpi=300)
plt.close(fig)
print("OK d")
