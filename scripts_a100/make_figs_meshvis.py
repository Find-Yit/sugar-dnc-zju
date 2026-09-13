#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段 M 视觉对比图：法向/深度矩阵、失败案例背景放大、拓扑柱状图。"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

P = os.environ.get("PROJ_ROOT", "/scratch/users/nus/e1351071/test_zju")
FONT = os.path.join(P, "assets/fonts/NotoSansCJKsc-Regular.otf")
font_manager.fontManager.addfont(FONT)
FNAME = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.family": FNAME, "axes.unicode_minus": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
print("字体 =", FNAME)

FIG = os.path.join(P, "outputs/figures/m_nscc"); os.makedirs(FIG, exist_ok=True)
VIS = os.path.join(P, "outputs/vis")
MET = os.path.join(P, "outputs/metrics")

METHODS = [("base_d10_q01", "原版 SuGaR\n(D=10, q=0.1)"),
           ("base_d10_q0",  "基线 q=0\n(D=10)"),
           ("m1a_q0",       "M1+a\n(D_bg=9, q=0)"),
           ("m2p95_q0",     "M2-B\n(DBSCAN p95, q=0)"),
           ("m2p95_bg9_q0", "M2-B + M1+a\n(p95, D_bg=9, q=0)"),
           ("m2largest_q0", "失败案例\n(只留最大簇)")]

meta = {t: json.load(open(os.path.join(MET, f"meshvis_m_{t}.json"))) for t, _ in METHODS}
VIEWS = sorted(int(k) for k in meta["base_d10_q0"]["mesh_pixel_coverage"])
NAMES = {int(k): v for k, v in meta["base_d10_q0"]["image_names"].items()}

def img_path(tag, kind, v):
    return os.path.join(VIS, f"m_{tag}", f"mesh_{kind}_view{v:02d}_{NAMES[v]}.png")

def matrix_fig(kind, title, outname):
    nr, nc = len(METHODS), len(VIEWS)
    fig, axes = plt.subplots(nr, nc, figsize=(4.0 * nc, 2.30 * nr))
    for i, (tag, label) in enumerate(METHODS):
        cov = {int(k): v for k, v in meta[tag]["mesh_pixel_coverage"].items()}
        for j, v in enumerate(VIEWS):
            ax = axes[i, j]
            ax.imshow(np.asarray(Image.open(img_path(tag, kind, v))))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{tag} · 视角{v:02d}({NAMES[v]}) · 覆盖率 {cov[v]*100:.2f}%", fontsize=8.5, pad=3)
            if j == 0:
                ax.set_ylabel(label, fontsize=10.5)
    fig.suptitle(title, fontsize=15, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(FIG, outname); fig.savefig(out, dpi=110); plt.close(fig)
    print("->", out)

matrix_fig("normal", "图 M-1  网格法向着色图：6 种提取配置 × 4 个固定测试视角（Truck，同一 coarse 模型 seed 0）",
           "fig_mesh_normal_compare.png")
matrix_fig("depth", "图 M-2  网格深度图（turbo 伪彩，逐视角 p1–p99 归一化）：6 种提取配置 × 4 个固定测试视角",
           "fig_mesh_depth_compare.png")

# ---- c. 失败案例背景放大 ----
FV = int(os.environ.get("FAIL_VIEW", "0"))
fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
for ax, (tag, lab) in zip(axes, [("base_d10_q0", "基线 q=0（背景高斯保留）"),
                                 ("m2largest_q0", "2D-SuGaR 原样：只留最大簇")]):
    im = np.asarray(Image.open(img_path(tag, "normal", FV)))
    H, W = im.shape[:2]
    ax.imshow(im[: int(H * 0.55), :])          # 上半幅 = 背景（树冠/天空）区域
    ax.set_xticks([]); ax.set_yticks([])
    cov = meta[tag]["mesh_pixel_coverage"][str(FV)]
    ax.set_title(f"{lab}\n{tag} · 视角{FV:02d}({NAMES[FV]}) · 像素覆盖率 {cov*100:.2f}%", fontsize=11)
axes[1].annotate("2D-SuGaR 原样只留最大簇：背景被删除 80.7%\n树冠细节整片消失 → Poisson 用大三角面片强行封口，\n像素覆盖率却仍 ~100%（几何指标与覆盖率均无法捕捉）",
                 xy=(0.5, 0.30), xycoords="axes fraction", ha="center", va="center", fontsize=12.5,
                 color="#b00020", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#b00020", alpha=0.92))
fig.suptitle("图 M-3  失败案例：背景区域并排对比（法向着色，画面上半部）", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(FIG, "fig_failure_background.png"); fig.savefig(out, dpi=115); plt.close(fig)
print("->", out)

# ---- d. 拓扑柱状图 ----
rows = {r["TAG"]: r for r in csv.DictReader(open(os.path.join(MET, "summary_m.csv")))}
tags = [t for t, _ in METHODS]
xl = [l.replace("\n", " ") for _, l in METHODS]
specs = [("n_components", "连通分量数（越少越好）", "{:.0f}", 1.0),
         ("n_boundary_edges", "边界边数（越少越好）", "{:.0f}", 1.0),
         ("largest_component_ratio", "最大连通分量面积占比（越大越好）", "{:.3f}", 1.0)]
colors = ["#9e9e9e", "#607d8b", "#2e7d32", "#1565c0", "#6a1b9a", "#b00020"]
fig, axes = plt.subplots(1, 3, figsize=(20, 5.9))
for ax, (key, ttl, fmt, sc) in zip(axes, specs):
    vals = [float(rows[t][key]) * sc for t in tags]
    b = ax.bar(range(len(tags)), vals, color=colors)
    ax.set_title(ttl, fontsize=13)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(xl, fontsize=8.5, rotation=20, ha="right")
    ax.set_ylim(0, max(vals) * 1.18)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + max(vals) * 0.02, fmt.format(v),
                ha="center", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("图 M-4  网格拓扑指标对比（Truck，同一 coarse 模型 seed 0，数据来自 outputs/metrics/summary_m.csv）", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(FIG, "fig_topology_bars.png"); fig.savefig(out, dpi=115); plt.close(fig)
print("->", out)
