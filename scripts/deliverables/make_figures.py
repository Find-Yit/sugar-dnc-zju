#!/usr/bin/env python3
"""阶段 5 / S5.1：从评测产物生成报告用的全中文科研图表（fig1–fig6）。

只读 outputs/metrics/*.{csv,json}、outputs/vis/<run>/*.png、outputs/runs/<run>/dnc_log.csv、
outputs/runs/<run>/dnc_vis/*.png 与 logs/*.log；**不生成任何实验数值**，缺数据就优雅跳过并提示。

用法（必须用文档 venv，不要 activate）：
    tools/doc_env/bin/python scripts/deliverables/make_figures.py                      # 默认三组
    tools/doc_env/bin/python scripts/deliverables/make_figures.py --runs coarse_baseline --labels "基线(阶段2 留档)"
    tools/doc_env/bin/python scripts/deliverables/make_figures.py --only fig1 fig4
    tools/doc_env/bin/python scripts/deliverables/make_figures.py \
        --crop "coarse_baseline,0,40,0,400,200,背景树冠：网格空洞"

产物：outputs/figures/fig{1..6}_*.png（300 dpi）+ 同名 .pdf（矢量）
      outputs/figures/figures_manifest.json（生成状态 / 数据来源，供 make_report.py 读取）
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")      # noqa: E402  与 GPU 训练并行时不抢 CPU
os.environ.setdefault("MPLBACKEND", "Agg")         # noqa: E402

import argparse                                     # noqa: E402
import json                                         # noqa: E402
import sys                                          # noqa: E402
from pathlib import Path                            # noqa: E402

import numpy as np                                  # noqa: E402
from PIL import Image                               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _deliv_common import (                         # noqa: E402
    COLORS, DEFAULT_LABELS, DEFAULT_RUNS, FALLBACK_COLOR_KEYS, FIGURE_DIR, METRICS_DIR,
    RunData, build_runs, configure_matplotlib, dig, hexc, load_json, rel, save_figure, to_float,
)

plt = configure_matplotlib()
from matplotlib import cm, colors as mcolors        # noqa: E402
from matplotlib.gridspec import GridSpec            # noqa: E402

# 与 scripts/eval/_common.py 的 ERROR_HEATMAP_VMAX 一致；实际取值以 render_*.json 的
# config.error_heatmap_vmax 为准（下面会从 JSON 读，读不到才用这个兜底）。
ERR_VMAX_FALLBACK = 0.25

MUTED = hexc("muted")
INK = hexc("ink")
NAVY = hexc("navy")
LINE = hexc("line")


# =========================================================================== 工具


def _img(path) -> np.ndarray | None:
    if path is None:
        return None
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
    except OSError:
        return None


def _blank_ax(ax, text: str, fontsize: float = 10):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=fontsize,
            color=MUTED, transform=ax.transAxes, wrap=True)


def _footnote(fig, text: str, y: float = 0.035):
    """图下方的数据来源说明。按图宽自动折行，避免 bbox_inches='tight' 为了容纳长文本
    把画布横向撑开（会在图片两侧留出大片空白）。"""
    import textwrap
    w_in = float(fig.get_size_inches()[0])
    ncols = max(40, int(w_in / 0.118))          # 8.2pt 中文字宽约 0.114 英寸，留 3% 余量
    fig.text(0.5, y, textwrap.fill(text, ncols), ha="center", va="top",
             fontsize=8.2, color=MUTED, linespacing=1.55)


def _smooth(y: list[float], win: int = 5) -> np.ndarray:
    a = np.asarray(y, dtype=np.float64)
    if len(a) < 2:
        return a
    win = max(1, min(win, len(a)))
    kernel = np.ones(win) / win
    pad = np.concatenate([np.full(win - 1, a[0]), a])
    return np.convolve(pad, kernel, mode="valid")


# =========================================================================== fig1

# 每个子图：(summary.csv 列名, 从 JSON 兜底取值的函数, 中文标题, 单位, 方向, 小数位)
def _v_render(key):
    return lambda r: dig(r.render, key)


def _v_geom(section, key, scale=1.0):
    def f(r):
        v = to_float(dig(r.geometry, section, key))
        return None if v is None else v * scale
    return f


def _v_frag(r):
    topo = dig(r.geometry, "G4_topology", default={}) or {}
    for k, v in topo.items():
        if k.startswith("n_fragment_components_lt_"):
            return to_float(v)
    return None


PANELS_RENDER = [
    ("PSNR_dB_up", _v_render("psnr_db_mean"), "测试视角 PSNR", "dB", "up", 3),
    ("SSIM_up", _v_render("ssim_mean"), "测试视角 SSIM", "无量纲", "up", 4),
    ("LPIPS_VGG_down", _v_render("lpips_vgg_mean"), "测试视角 LPIPS-VGG", "无量纲", "down", 4),
]
PANELS_GEOM = [
    ("G1_median_rel_pct_down", _v_geom("G1_sparse_to_mesh_distance", "median_rel", 100.0),
     "G1 稀疏点→网格距离 中位数", "% 相机空间尺度", "down", 4),
    ("G2_ratio_lt_1pct_up", _v_geom("G2_precision_ratio", "ratio_below_1pct", 100.0),
     "G2 精度比例（距离 < 1% 尺度）", "%", "up", 3),
    ("G4_n_fragments_lt100faces_down", _v_frag,
     "G4 漂浮碎片分量数（< 100 面）", "分量个数", "down", 0),
    ("G4_largest_comp_face_ratio_pct_up",
     _v_geom("G4_topology", "largest_component_face_ratio", 100.0),
     "G4 最大连通分量面数占比", "%", "up", 3),
]


def _panel_values(runs: list[RunData], col: str, getter) -> list[float | None]:
    out = []
    for r in runs:
        v = r.s(col)
        if v is None:
            v = to_float(getter(r))
        out.append(v)
    return out


def _bar_panel(ax, runs, values, title, unit, better, nd):
    arrow = "↑ 越高越好" if better == "up" else "↓ 越低越好"
    ax.set_title(f"{title}\n（{arrow}）", fontsize=10.5, color=NAVY,
                 fontweight="bold", pad=8, linespacing=1.4)
    ax.set_ylabel(unit, fontsize=9)
    xs = np.arange(len(runs))
    good = [(i, v) for i, v in enumerate(values) if v is not None]
    if not good:
        _blank_ax(ax, "无数据\n（对应 run 的指标文件尚未生成）")
        ax.set_title(f"{title}\n（{arrow}）", fontsize=10.5, color=NAVY,
                     fontweight="bold", pad=8, linespacing=1.4)
        return
    vals = [v for _, v in good]
    vmin, vmax = min(vals), max(vals)
    span = vmax - vmin
    # 纵轴是否截断：差异过小时放大局部，但必须显式标注（style-guide 要求）
    zoom = bool(span > 0 and vmin > 0 and span / max(abs(vmax), 1e-12) < 0.12)
    if zoom:
        lo = vmin - span * 0.9
        hi = vmax + span * 0.9
        if lo < 0:
            lo = 0.0
            zoom = False
    if not zoom:
        lo = 0.0
        hi = vmax * 1.22 if vmax > 0 else 1.0
    ax.set_ylim(lo, hi)

    for i, r in enumerate(runs):
        v = values[i]
        if v is None:
            ax.text(i, lo + (hi - lo) * 0.5, "无数据", ha="center", va="center",
                    fontsize=9, color=MUTED)
            continue
        ax.bar(i, v - lo, bottom=lo, color=r.color, width=0.58,
               edgecolor="white", linewidth=1.0)
        ax.text(i, v + (hi - lo) * 0.035, f"{v:.{nd}f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=INK)
        base = values[0]
        if i > 0 and base is not None:
            d = v - base
            good_dir = (d > 0) if better == "up" else (d < 0)
            col = hexc("teal") if good_dir else hexc("red")
            if abs(d) < 10 ** (-nd) / 2:
                col = MUTED
            ax.text(i, v + (hi - lo) * 0.125, f"Δ{d:+.{nd}f}", ha="center", va="bottom",
                    fontsize=8.2, color=col)
    if zoom:
        ax.text(0.985, 0.03, "纵轴起点非 0（放大局部差异）", ha="right", va="bottom",
                fontsize=7.6, color=hexc("red"), transform=ax.transAxes)
    ax.set_xlim(-0.65, len(runs) - 0.35)
    ax.set_xticks(xs)
    n = len(runs)
    fs = 9 if n <= 3 else (8 if n <= 5 else 7)
    rot = 0 if n <= 3 else (12 if n <= 5 else 20)
    ax.set_xticklabels([r.label.replace("(", "\n(") if n > 4 else r.label for r in runs],
                       fontsize=fs, rotation=rot,
                       ha="center" if rot == 0 else "right")
    ax.grid(axis="y", color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)


def fig1_metrics(runs, args) -> dict:
    fig = plt.figure(figsize=(max(15.4, 2.6 * len(runs) + 6.0), 9.0))
    gs = GridSpec(2, 12, figure=fig, hspace=0.58, wspace=0.85,
                  left=0.055, right=0.985, top=0.845, bottom=0.13)
    for j, (col, getter, title, unit, better, nd) in enumerate(PANELS_RENDER):
        ax = fig.add_subplot(gs[0, j * 4:(j + 1) * 4])
        _bar_panel(ax, runs, _panel_values(runs, col, getter), title, unit, better, nd)
    for j, (col, getter, title, unit, better, nd) in enumerate(PANELS_GEOM):
        ax = fig.add_subplot(gs[1, j * 3:(j + 1) * 3])
        _bar_panel(ax, runs, _panel_values(runs, col, getter), title, unit, better, nd)

    n_views = None
    n_ref = None
    for r in runs:
        n_views = n_views or to_float(dig(r.render, "n_test_views"))
        n_ref = n_ref or to_float(dig(r.geometry, "G1_sparse_to_mesh_distance", "n_reference_points"))
    fig.suptitle("各组对照的渲染质量与网格几何指标（Tanks&Temples / Truck）",
                 fontsize=15, fontweight="bold", color=NAVY, y=0.985)
    fig.text(0.5, 0.938,
             "上排＝渲染质量（测试视角，llffhold=8 划分）；下排＝Poisson 网格几何与拓扑"
             "（surface_level=0.3，decimation 目标：前景 20 万面 + 背景 20 万面）",
             ha="center", va="top", fontsize=9.5, color=MUTED)
    note = "数据来源：outputs/metrics/summary.csv 及 render_*.json / geometry_*.json"
    if n_views:
        note += f"；样本量：测试视角 {int(n_views)} 个"
    if n_ref:
        note += f"，G1 参考点 {int(n_ref):,} 个（COLMAP 稀疏点，track≥3 且落在前景包围盒内）"
    _footnote(fig, note + "。", y=0.055)
    return {"fig": fig, "stem": "fig1_指标对比"}


# =========================================================================== fig2


def _common_views(runs) -> list[tuple[int, str]]:
    lists = [r.vis_views() for r in runs if r.vis_views()]
    if not lists:
        return []
    common = set(lists[0])
    for l in lists[1:]:
        common &= set(l)
    return sorted(common) if common else sorted(lists[0])


def fig2_render_montage(runs, args) -> dict | None:
    views = _common_views(runs)
    if not views:
        return {"skip": "没有任何 run 的 outputs/vis/<run>/render_view*.png，跳过 fig2"}
    views = views[:args.max_views]
    err_vmax = None
    for r in runs:
        err_vmax = err_vmax or to_float(dig(r.render, "config", "error_heatmap_vmax"))
    err_vmax = err_vmax or ERR_VMAX_FALLBACK

    n_rows = len(views)
    n_cols = 1 + 2 * len(runs)
    tile_w = 2.45
    tile_h = tile_w * 546.0 / 979.0                 # 与 Truck 图像宽高比一致，减少留白
    fig = plt.figure(figsize=(max(9.8, tile_w * n_cols + 1.25), tile_h * n_rows + 1.55))
    gs = GridSpec(n_rows, n_cols + 1, figure=fig,
                  width_ratios=[1] * n_cols + [0.085],
                  hspace=0.07, wspace=0.035,
                  left=0.052, right=0.955, top=0.862, bottom=0.085)

    col_titles = ["真值 GT"] + [f"{r.label}\n渲染" for r in runs] + [f"{r.label}\n|误差|" for r in runs]
    for i, (vidx, vname) in enumerate(views):
        gt = None
        for r in runs:          # GT 各组相同，取第一个能读到的；不能写成 `gt = gt or ...`
            if gt is None:      # （numpy 数组用 or 会抛 ambiguous truth value）
                gt = _img(r.vis_png("gt", vidx, vname))
        cells = [gt]
        cells += [_img(r.vis_png("render", vidx, vname)) for r in runs]
        cells += [_img(r.vis_png("errmap", vidx, vname)) for r in runs]
        for j, arr in enumerate(cells):
            ax = fig.add_subplot(gs[i, j])
            if arr is None:
                _blank_ax(ax, "缺图", 8)
            else:
                ax.imshow(arr)
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color(LINE)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=9.5, color=NAVY,
                             fontweight="bold", pad=6, linespacing=1.3)
            if j == 0:
                ax.set_ylabel(f"测试视角 #{vidx}\n{vname}", fontsize=8.6, color=MUTED,
                              linespacing=1.4)

    cax = fig.add_subplot(gs[:, n_cols])
    sm = cm.ScalarMappable(norm=mcolors.Normalize(0.0, err_vmax), cmap="turbo")
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(f"平均绝对误差 |渲染−真值|（RGB 三通道均值，0–1 尺度，上限 {err_vmax:g} 截断）",
                 fontsize=8.6, color=INK)
    cb.ax.tick_params(labelsize=8, color=MUTED)
    cb.outline.set_edgecolor(LINE)

    fig.suptitle("固定测试视角的渲染结果与误差热图对比（↓ 误差越小越好）",
                 fontsize=14.5, fontweight="bold", color=NAVY, y=0.982)
    _footnote(fig,
              "图像来源：outputs/vis/<run>/{gt,render,errmap}_view*.png（由 scripts/eval/eval_render.py 生成，"
              "turbo 伪彩色，各组共用同一色条与同一上限）；每组渲染均使用各自 coarse SuGaR "
              "15000 迭代的高斯，相机与测试划分完全一致。",
              y=0.055)
    return {"fig": fig, "stem": "fig2_渲染对比"}


# =========================================================================== fig3


def fig3_mesh_normals(runs, args) -> dict | None:
    have = [r for r in runs if r.vis_dir and list(r.vis_dir.glob("mesh_normal_view*.png"))]
    if not have:
        return {"skip": "没有任何 run 的 outputs/vis/<run>/mesh_normal_view*.png，跳过 fig3"}
    views = _common_views(runs)
    if not views:
        import re as _re
        found = {}
        for p in have[0].vis_dir.glob("mesh_normal_view*.png"):
            m = _re.match(r"mesh_normal_view(\d+)_(.+)\.png$", p.name)
            if m:
                found[int(m.group(1))] = m.group(2)
        views = sorted(found.items())
    views = views[:args.max_views]

    n_rows, n_cols = len(views), len(runs)
    tile_w = 3.15
    fig = plt.figure(figsize=(max(9.8, tile_w * n_cols + 0.9),
                              tile_w * 546.0 / 979.0 * n_rows + 1.55))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.07, wspace=0.035,
                  left=0.085, right=0.985, top=0.858, bottom=0.095)
    for i, (vidx, vname) in enumerate(views):
        for j, r in enumerate(runs):
            ax = fig.add_subplot(gs[i, j])
            arr = _img(r.vis_png("mesh_normal", vidx, vname))
            if arr is None:
                _blank_ax(ax, f"缺图\n{r.label}", 8)
            else:
                ax.imshow(arr)
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color(LINE)
            if i == 0:
                ax.set_title(r.label, fontsize=11, color=NAVY, fontweight="bold", pad=6)
            if j == 0:
                ax.set_ylabel(f"测试视角 #{vidx}\n{vname}", fontsize=8.6, color=MUTED,
                              linespacing=1.4)
    fig.suptitle("Poisson 网格的法向着色图对比（颜色突变越少 = 表面法向越平滑）",
                 fontsize=14.5, fontweight="bold", color=NAVY, y=0.982)
    _footnote(fig,
              "图像来源：outputs/vis/<run>/mesh_normal_view*.png（scripts/eval/render_mesh_views.py，"
              "pytorch3d 光栅化）。颜色编码：世界系面法向朝向相机翻转后按 rgb=(n×0.5+0.5) 映射；"
              "白色为背景（该像素没有网格命中，即网格空洞）。各组使用完全相同的相机与提取参数。",
              y=0.062)
    return {"fig": fig, "stem": "fig3_网格法向对比"}


# =========================================================================== fig4


def _curve_panel(ax, runs, xs_ys, title, ylabel, note=None, smooth_win=5):
    drew = False
    for r, (xs, ys) in zip(runs, xs_ys):
        if not xs:
            continue
        ax.plot(xs, ys, color=r.color, linewidth=0.9, alpha=0.35)
        ax.plot(xs, _smooth(ys, smooth_win), color=r.color, linewidth=2.1, label=r.label)
        drew = True
    ax.set_title(title, fontsize=11, color=NAVY, fontweight="bold", pad=8)
    ax.set_xlabel("训练迭代数", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if drew:
        ax.legend(frameon=False, fontsize=8.8)
        if note:
            ax.text(0.985, 0.03, note, ha="right", va="bottom", fontsize=7.8,
                    color=MUTED, transform=ax.transAxes)
    return drew


def _dnc_series(runs, column) -> list[tuple[list[int], list[float]]]:
    out = []
    for r in runs:
        rows = r.read_dnc_log()
        xs, ys = [], []
        for row in rows:
            it = to_float(row.get("iteration"))
            v = to_float(row.get(column))
            if it is not None and v is not None:
                xs.append(int(it))
                ys.append(v)
        out.append((xs, ys))
    return out


def fig4_training_curves(runs, args) -> dict | None:
    l_dnc = _dnc_series(runs, "l_dnc")
    sdf_n = _dnc_series(runs, "sdf_better_normal_loss")
    ratio = _dnc_series(runs, "valid_pixel_ratio")
    total = [r.read_train_loss_curve() for r in runs]

    has_dnc = any(xs for xs, _ in l_dnc)
    has_total = any(xs for xs, _ in total)
    if not has_dnc and not has_total:
        return {"skip": "所选 run 既没有 dnc_log.csv 也没有可解析的训练日志，跳过 fig4"}

    no_dnc = [r.label for r, (xs, _) in zip(runs, l_dnc) if not xs]
    fig = plt.figure(figsize=(13.6, 8.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.24,
                  left=0.07, right=0.98, top=0.87, bottom=0.115)

    ax = fig.add_subplot(gs[0, 0])
    if has_dnc:
        _curve_panel(ax, runs, l_dnc, "(a) 深度-法向一致性损失 L_dnc（↓ 越低越好）",
                     "L_dnc = 1 − |cos(N, N_d)|",
                     note="细线=每 100 迭代单帧采样，粗线=5 点滑动平均")
    else:
        _blank_ax(ax, "所选 run 均未记录 L_dnc\n（λ=0 时不进入 DNC 分支，不生成 dnc_log.csv）")
        ax.set_title("(a) 深度-法向一致性损失 L_dnc", fontsize=11, color=NAVY,
                     fontweight="bold", pad=8)

    ax = fig.add_subplot(gs[0, 1])
    drew = _curve_panel(ax, runs, sdf_n, "(b) SuGaR 原有 SDF 法向损失（↓ 越低越好，用于检验 H3）",
                        "sdf_better_normal_loss",
                        note="细线=原始值，粗线=5 点滑动平均")
    if not drew:
        _blank_ax(ax, "所选 run 未记录 sdf_better_normal_loss\n（该逐迭代记录随 dnc_log.csv 一起写出）")
        ax.set_title("(b) SuGaR 原有 SDF 法向损失", fontsize=11, color=NAVY,
                     fontweight="bold", pad=8)

    ax = fig.add_subplot(gs[1, 0])
    drew = _curve_panel(ax, runs, ratio, "(c) L_dnc 有效像素占比（掩码通过率）",
                        "有效像素 / 全图像素",
                        note="掩码=前景腐蚀 ∧ 深度梯度阈值 ∧ 边界剔除 ∧ ‖N_raw‖>0.1")
    if not drew:
        _blank_ax(ax, "所选 run 未记录有效像素占比")
        ax.set_title("(c) L_dnc 有效像素占比", fontsize=11, color=NAVY, fontweight="bold", pad=8)

    ax = fig.add_subplot(gs[1, 1])
    drew = _curve_panel(ax, runs, total, "(d) 训练总损失（每 200 迭代打印一次）",
                        "total loss",
                        note="λ>0 组的总损失已含 λ·L_dnc 项，各组之间不可直接比较绝对值")
    if drew and has_dnc:
        starts = [min(xs) for xs, _ in l_dnc if xs]
        if starts:
            ax.axvline(min(starts) - 100, color=hexc("red"), linewidth=1.1,
                       linestyle="--", alpha=0.75)
            ax.text(min(starts) - 100, ax.get_ylim()[1], " DNC 生效起点", ha="left",
                    va="top", fontsize=8, color=hexc("red"))
    if not drew:
        _blank_ax(ax, "训练日志中未找到可解析的 loss 行")
        ax.set_title("(d) 训练总损失", fontsize=11, color=NAVY, fontweight="bold", pad=8)

    fig.suptitle("coarse SuGaR 训练过程中的损失曲线（DNC 从第 9000 迭代起生效）",
                 fontsize=14.5, fontweight="bold", color=NAVY, y=0.968)
    tail = ("；未记录 L_dnc 的 run：" + "、".join(no_dnc)) if no_dnc else ""
    _footnote(fig,
              "数据来源：outputs/runs/<run>/dnc_log.csv（每 100 迭代一行）与 logs/*coarse*<run>*.log "
              "中的 `loss: ... [iter/15000]` 行" + tail + "。",
              y=0.055)
    return {"fig": fig, "stem": "fig4_训练曲线"}


# =========================================================================== fig5


def fig5_dnc_intermediate(runs, args) -> dict | None:
    cands = [r for r in runs if r.dnc_vis_iterations()]
    if args.dnc_vis_run:
        cands = [r for r in cands if r.key == args.dnc_vis_run]
    if not cands:
        return {"skip": "所选 run 的 outputs/runs/<run>/dnc_vis/ 下没有 iter*_depth.png，跳过 fig5"
                        "（λ=0 组不产生 DNC 中间量；λ>0 组每 1000 迭代才保存一次）"}
    run = max(cands, key=lambda r: (r.dnc_factor or 0.0))
    it = run.dnc_vis_iterations()[-1]
    tag = f"iter{it:06d}"
    panels = [
        (f"{tag}_depth.png", "(a) 渲染深度图 D",
         "灰度按有效像素的 1%–99% 分位归一化；越亮＝离相机越远"),
        (f"{tag}_normal_rendered.png", "(b) 渲染法向图 N",
         "高斯最短轴法向（朝向相机翻转）经 α 合成，rgb=(n×0.5+0.5)"),
        (f"{tag}_normal_from_depth.png", "(c) 深度差分法向 N_d",
         "对 D 反投影得点图后中心差分叉乘，rgb=(n×0.5+0.5)"),
        (f"{tag}_mask.png", "(d) 有效像素掩码",
         "白＝参与 L_dnc 计算；黑＝被背景/边界/深度跳变/低不透明度剔除"),
    ]
    fig = plt.figure(figsize=(12.2, 8.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.06,
                  left=0.03, right=0.97, top=0.855, bottom=0.10)
    for k, (fname, title, desc) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        arr = _img(run.dnc_vis_dir / fname)
        if arr is None:
            _blank_ax(ax, f"缺图：{fname}", 9)
        else:
            ax.imshow(arr)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(LINE)
        ax.set_title(title, fontsize=11.5, color=NAVY, fontweight="bold", pad=6)
        ax.set_xlabel(desc, fontsize=8.4, color=MUTED, labelpad=6)

    lam = run.dnc_factor
    fig.suptitle(f"DNC 中间量可视化（{run.label}，第 {it} 迭代，单个训练视角）",
                 fontsize=14.5, fontweight="bold", color=NAVY, y=0.965)
    fig.text(0.5, 0.905,
             "L_dnc = 掩码内像素的 mean(1 − |cos(N, N_d)|)；取绝对值是为了消除三维椭球最短轴的法向符号歧义",
             ha="center", va="top", fontsize=9.5, color=MUTED)
    _footnote(fig,
              f"图像来源：{rel(run.dnc_vis_dir)}/{tag}_*.png（训练中每 1000 迭代由 "
              f"sugar_utils/dnc_utils.save_dnc_visualization 直接落盘，λ={lam:g}）。",
              y=0.048)
    return {"fig": fig, "stem": "fig5_DNC中间量"}


def fig5b_dnc_lambda_compare(runs, args) -> dict | None:
    """各 λ 组在**同一迭代**上的 DNC 中间量对比（行=λ 组，列=D/N/N_d/mask）。

    当只有一组有 dnc_vis 时自动跳过（此时 fig5 已经把它画完了）。
    """
    cands = [r for r in runs if r.dnc_vis_iterations()]
    if len(cands) < 2:
        return {"skip": "有 dnc_vis 的 run 少于 2 个，跳过 fig5b（λ 之间没有可对比的中间量）"}
    cands = sorted(cands, key=lambda r: (r.dnc_factor or 0.0))
    common = set(cands[0].dnc_vis_iterations())
    for r in cands[1:]:
        common &= set(r.dnc_vis_iterations())
    if not common:
        return {"skip": "各 λ 组的 dnc_vis 没有公共迭代，跳过 fig5b"}
    it = max(common)
    tag = f"iter{it:06d}"
    cols = [("depth", "渲染深度图 D"), ("normal_rendered", "渲染法向图 N"),
            ("normal_from_depth", "深度差分法向 N_d"), ("mask", "有效像素掩码")]

    n_rows, n_cols = len(cands), len(cols)
    tile_w = 3.0
    fig = plt.figure(figsize=(max(9.8, tile_w * n_cols + 1.0),
                              tile_w * 546.0 / 979.0 * n_rows + 1.75))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.08, wspace=0.035,
                  left=0.085, right=0.985, top=0.845, bottom=0.10)
    for i, r in enumerate(cands):
        for j, (kind, title) in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            arr = _img(r.dnc_vis_dir / f"{tag}_{kind}.png")
            if arr is None:
                _blank_ax(ax, "缺图", 8)
            else:
                ax.imshow(arr)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color(LINE)
            if i == 0:
                ax.set_title(title, fontsize=10.5, color=NAVY, fontweight="bold", pad=6)
            if j == 0:
                lam = r.dnc_factor
                lab = r.label
                if lam is not None and "λ" not in lab:
                    lab = f"{lab}\nλ={lam:g}"
                ax.set_ylabel(lab, fontsize=9.5, color=MUTED, linespacing=1.5)
    fig.suptitle(f"不同 λ 下 DNC 中间量的对比（同为第 {it} 迭代，各自的随机训练视角）",
                 fontsize=14.5, fontweight="bold", color=NAVY, y=0.975)
    fig.text(0.5, 0.9,
             "每行各自取该迭代时抽到的训练视角，视角不同属正常现象；关注的是深度图里"
             "是否还保留场景结构、法向图是否还有物体轮廓",
             ha="center", va="top", fontsize=9.2, color=MUTED)
    _footnote(fig,
              "图像来源：outputs/runs/<run>/dnc_vis/" + tag + "_*.png。"
              "深度按各自有效像素的 1%–99% 分位独立归一化，因此行与行之间的绝对灰度不可比，"
              "可比的是结构是否清晰。",
              y=0.055)
    return {"fig": fig, "stem": "fig5b_DNC中间量_各λ对比"}


# =========================================================================== fig6

# 默认失败案例（在 979×546 的 Truck 测试视角上人工选定，只作示例；
# 可用 --crop "run,view,x0,y0,x1,y1[,说明]" 覆盖或追加）
DEFAULT_CROPS = [
    (0, (40, 0, 400, 205), "背景树冠与远景栏杆：深度差分法向噪声大，网格出现空洞"),
    (8, (600, 110, 979, 430), "右侧细杆、遮阳伞与栏杆：线状细结构易被压平或丢失"),
    (24, (150, 0, 860, 135), "顶部电线与树枝：极细结构，Poisson 重建后碎片化"),
]
CROP_KINDS = [("gt", "真值 GT"), ("render", "渲染结果"),
              ("errmap", "|误差| 热图"), ("mesh_normal", "网格法向")]


def _parse_crops(args, runs) -> list[tuple[RunData, int, tuple[int, int, int, int], str]]:
    by_key = {r.key: r for r in runs}
    specs = []
    if args.crop:
        for raw in args.crop:
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) < 6:
                raise SystemExit(f"[ERROR] --crop 格式应为 run,view,x0,y0,x1,y1[,说明]，收到：{raw!r}")
            rk, v = parts[0], int(parts[1])
            box = tuple(int(x) for x in parts[2:6])
            desc = parts[6] if len(parts) > 6 else ""
            targets = runs if rk.upper() == "ALL" else [by_key.get(rk)]
            for t in targets:
                if t is None:
                    raise SystemExit(f"[ERROR] --crop 里的 run={rk!r} 不在 --runs 列表中")
                specs.append((t, v, box, desc))
    else:
        ref = runs[0]
        avail = dict(ref.vis_views())
        for v, box, desc in DEFAULT_CROPS:
            if v in avail:
                specs.append((ref, v, box, desc))
    return specs


def fig6_failure_crops(runs, args) -> dict | None:
    specs = _parse_crops(args, runs)
    if not specs:
        return {"skip": "没有可用的裁剪区域（--crop 为空且默认视角不在 outputs/vis 中），跳过 fig6"}
    rows = []
    for run, vidx, box, desc in specs:
        names = dict(run.vis_views())
        vname = names.get(vidx)
        if vname is None:
            print(f"  [WARN] fig6：{run.key} 没有视角 #{vidx} 的可视化，跳过该行")
            continue
        cells = []
        for kind, _ in CROP_KINDS:
            arr = _img(run.vis_png(kind, vidx, vname))
            if arr is None:
                cells.append(None)
                continue
            h, w = arr.shape[:2]
            x0, y0, x1, y1 = box
            x0, x1 = max(0, min(x0, w - 2)), max(1, min(x1, w))
            y0, y1 = max(0, min(y0, h - 2)), max(1, min(y1, h))
            cells.append(arr[y0:y1, x0:x1])
        rows.append((run, vidx, vname, box, desc, cells))
    if not rows:
        return {"skip": "所有裁剪区域对应的可视化图都不存在，跳过 fig6"}

    n_rows, n_cols = len(rows), len(CROP_KINDS)
    fig = plt.figure(figsize=(max(9.8, 3.55 * n_cols + 0.4), 2.55 * n_rows + 1.65))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.36, wspace=0.05,
                  left=0.032, right=0.985, top=0.852, bottom=0.10)
    for i, (run, vidx, vname, box, desc, cells) in enumerate(rows):
        row_axes = []
        for j, arr in enumerate(cells):
            ax = fig.add_subplot(gs[i, j])
            row_axes.append(ax)
            if arr is None:
                _blank_ax(ax, "缺图", 8)
            else:
                ax.imshow(arr, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color(LINE)
            if i == 0:
                ax.set_title(CROP_KINDS[j][1], fontsize=11, color=NAVY,
                             fontweight="bold", pad=6)
            if j == 0:
                ax.set_ylabel(f"{run.label}\n视角 #{vidx}", fontsize=9, color=MUTED,
                              linespacing=1.5)
            if j == n_cols - 1 and desc:
                ax.set_xlabel(desc, fontsize=8.6, color=hexc("red"), labelpad=6)
        row_axes[0].set_xlabel(f"裁剪框 (x0,y0,x1,y1)=({box[0]},{box[1]},{box[2]},{box[3]})",
                               fontsize=7.8, color=MUTED, labelpad=6)

    fig.suptitle("失败案例特写：细结构、背景与物体边缘（原图 979×546，按像素坐标裁剪放大）",
                 fontsize=14.5, fontweight="bold", color=NAVY, y=0.978)
    _footnote(fig,
              "图像来源：outputs/vis/<run>/{gt,render,errmap,mesh_normal}_view*.png 的同一像素区域；"
              "未做任何增强或重采样（最近邻放大）。误差热图色标与图 2 相同。",
              y=0.062)
    return {"fig": fig, "stem": "fig6_失败案例特写"}


def _poisson_params(run: RunData) -> tuple[str, str]:
    """从 provenance_<run>.json 读 Poisson 深度与密度分位；读不到就用 SuGaR 默认值。"""
    prov = load_json(METRICS_DIR / f"provenance_{run.key}.json") or {}
    st = prov.get("extract_stats") or {}
    d_arg = st.get("poisson_depth_arg")
    d_used = st.get("poisson_depth_used")
    q = st.get("vertices_density_quantile")
    if d_arg is None and d_used is None:
        d = "10（默认）"
    elif str(d_arg) == "auto":
        d = f"auto→{d_used}"
    else:
        d = str(d_used if d_used is not None else d_arg)
    qs = "0.1（默认）" if q is None else f"{float(q):g}"
    return d, qs


FIG7_PANELS = [
    ("G4_n_fragments_lt100faces_down", _v_frag,
     "漂浮碎片分量数（< 100 面）", "分量个数", "down", 0),
    ("G1_median_rel_pct_down", _v_geom("G1_sparse_to_mesh_distance", "median_rel", 100.0),
     "G1 稀疏点→网格距离 中位数", "% 相机空间尺度", "down", 4),
    ("G5_dihedral_abs_deg_mean_down", _v_geom("G5_normal_smoothness", "dihedral_abs_deg_mean"),
     "G5 相邻面二面角绝对值均值", "度", "down", 2),
    ("G4_largest_comp_face_ratio_pct_up",
     _v_geom("G4_topology", "largest_component_face_ratio", 100.0),
     "最大连通分量面数占比", "%", "up", 2),
]


def fig7_poisson_sweep(runs, args) -> dict | None:
    """提取端扫参：同一个 coarse 模型，只改 Poisson 深度 D 与密度分位 quantile。"""
    keys = args.sweep_runs or []
    if not keys:
        return {"skip": "未指定 --sweep_runs，跳过 fig7（提取端扫参图）"}
    sweep = build_runs(keys, [args.sweep_labels[i] if args.sweep_labels and
                              i < len(args.sweep_labels) else k
                              for i, k in enumerate(keys)])
    sweep = [r for r in sweep if r.geometry]
    if not sweep:
        return {"skip": "--sweep_runs 里没有一个 run 有 geometry_*.json，跳过 fig7"}
    for i, r in enumerate(sweep):          # 用 D / quantile 当显示名，柱子颜色按顺序取
        d, q = _poisson_params(r)
        r.label = f"{r.label}\nD={d}\nq={q}"
        r.color = hexc(FALLBACK_COLOR_KEYS[i % len(FALLBACK_COLOR_KEYS)])

    fig = plt.figure(figsize=(max(13.5, 2.9 * len(sweep) + 4.0), 9.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.62, wspace=0.24,
                  left=0.075, right=0.98, top=0.83, bottom=0.115)
    for k, (col, getter, title, unit, better, nd) in enumerate(FIG7_PANELS):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        _bar_panel(ax, sweep, _panel_values(sweep, col, getter), title, unit, better, nd)

    fig.suptitle("提取端扫参：Poisson 深度 D 与顶点密度分位 quantile 对网格的影响",
                 fontsize=14.5, fontweight="bold", color=NAVY, y=0.982)
    fig.text(0.5, 0.925,
             "所有组共用同一个 coarse SuGaR 模型（λ=0，15000 迭代），只改网格提取参数，"
             "因此渲染指标完全相同，差异全部来自提取端；Δ 为相对第一组（原版参数）的变化",
             ha="center", va="top", fontsize=9.5, color=MUTED)
    _footnote(fig,
              "数据来源：outputs/metrics/summary.csv 与 geometry_<run>.json；D 与 quantile 取自 "
              "outputs/metrics/provenance_<run>.json（没有该文件的组用 SuGaR 默认值 D=10、quantile=0.1）。"
              "auto 深度的计算方式移植自 Anttwo/Frosting 的 frosting_extractors/coarse_shell.py 第 17–49 行。",
              y=0.048)
    return {"fig": fig, "stem": "fig7_提取端扫参"}


# =========================================================================== main

BUILDERS = {
    "fig1": fig1_metrics,
    "fig2": fig2_render_montage,
    "fig3": fig3_mesh_normals,
    "fig4": fig4_training_curves,
    "fig5": fig5_dnc_intermediate,
    "fig5b": fig5b_dnc_lambda_compare,
    "fig6": fig6_failure_crops,
    "fig7": fig7_poisson_sweep,
}


def main():
    p = argparse.ArgumentParser(description="生成报告用的全中文科研图表 fig1–fig6",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                   help=f"run 名（outputs/runs 与 outputs/vis 下的目录名），默认 {' '.join(DEFAULT_RUNS)}")
    p.add_argument("--labels", nargs="+", default=None,
                   help="与 --runs 一一对应的中文显示名；不给则用内置映射")
    p.add_argument("--only", nargs="+", default=None, choices=sorted(BUILDERS),
                   help="只生成指定的图，默认全部")
    p.add_argument("--outdir", type=Path, default=FIGURE_DIR)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--max_views", type=int, default=4, help="拼图里最多放几个测试视角")
    p.add_argument("--dnc_vis_run", type=str, default=None,
                   help="fig5 用哪个 run 的 dnc_vis，默认选 λ 最大且有中间量的那组")
    p.add_argument("--crop", action="append", default=None,
                   metavar="run,view,x0,y0,x1,y1[,说明]",
                   help="fig6 的裁剪区域，可重复；run 可写 ALL 表示所有 run 各出一行")
    p.add_argument("--sweep_runs", nargs="+", default=None,
                   help="fig7 用的提取端扫参 run 列表（同一 coarse 模型、不同 Poisson 参数）")
    p.add_argument("--sweep_labels", nargs="+", default=None,
                   help="与 --sweep_runs 一一对应的中文显示名")
    p.add_argument("--manifest", type=Path, default=None,
                   help="生成状态 JSON，默认 <outdir>/figures_manifest.json")
    args = p.parse_args()

    labels = args.labels
    if labels is None:
        labels = [DEFAULT_LABELS.get(k, k) for k in args.runs]
    runs = build_runs(args.runs, labels)

    print("=" * 78)
    print("make_figures.py —— 输入清单")
    for r in runs:
        print(f"  run={r.key:<20s} 显示名={r.label:<16s} "
              f"render={'有' if r.render else '缺'} geometry={'有' if r.geometry else '缺'} "
              f"vis={'有' if r.vis_dir else '缺'} dnc_log={'有' if r.dnc_log else '缺'} "
              f"dnc_vis={len(r.dnc_vis_iterations())} 张 train_log={r.train_log.name if r.train_log else '缺'}")
    print("=" * 78)

    wanted = args.only or list(BUILDERS)
    mpath = args.manifest or (args.outdir / "figures_manifest.json")
    # 增量：只重画 --only 指定的图时，保留清单里其它图的记录（否则 make_report 会以为它们没生成）
    manifest = {"figures": {}}
    if mpath.exists() and args.only:
        try:
            with open(mpath) as f:
                manifest = json.load(f)
            manifest.setdefault("figures", {})
        except (OSError, json.JSONDecodeError):
            manifest = {"figures": {}}
    manifest["runs"] = [{"key": r.key, "label": r.label, "sources": r.sources} for r in runs]
    manifest["dpi"] = args.dpi
    args.outdir.mkdir(parents=True, exist_ok=True)

    for name in wanted:
        print(f"\n>>> {name}")
        try:
            res = BUILDERS[name](runs, args)
        except Exception as exc:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            manifest["figures"][name] = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
            print(f"  [ERROR] {name} 生成失败：{exc}")
            continue
        if res is None or "skip" in res:
            msg = res["skip"] if res else "未生成"
            manifest["figures"][name] = {"status": "skipped", "message": msg}
            print(f"  [SKIP] {msg}")
            continue
        paths = save_figure(res["fig"], args.outdir, res["stem"], dpi=args.dpi)
        plt.close(res["fig"])
        manifest["figures"][name] = {
            "status": "ok",
            "stem": res["stem"],
            "png": rel(paths["png"]),
            "pdf": rel(paths["pdf"]),
            "png_bytes": paths["png"].stat().st_size,
        }
        print(f"  [OK] {rel(paths['png'])}  ({paths['png'].stat().st_size/1024:.0f} KB)")
        print(f"       {rel(paths['pdf'])}")

    with open(mpath, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 清单 -> {rel(mpath)}")

    req = {k: v for k, v in manifest["figures"].items() if k in wanted}
    ok = sum(1 for v in req.values() if v["status"] == "ok")
    n_skip = sum(1 for v in req.values() if v["status"] == "skipped")
    n_err = sum(1 for v in req.values() if v["status"] == "error")
    print(f"[SUMMARY] 本次请求 {len(wanted)} 张：成功 {ok}，跳过 {n_skip}，失败 {n_err}；"
          f"清单中共有 {len(manifest['figures'])} 张记录")


if __name__ == "__main__":
    main()
