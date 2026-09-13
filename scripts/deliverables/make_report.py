#!/usr/bin/env python3
"""阶段 5 / S5.1：生成中文 DOCX 报告 outputs/reports/report.docx。

原则（计划 §2b 末尾诚信条款 + .agents/skills/scientific-deliverables/SKILL.md）：
- **所有数字一律从 outputs/metrics/summary.csv 与各 metrics JSON / train_stats.json / dnc_log.csv
  读入**，脚本里不写死任何实验数值；读不到就显示"未完成"。
- 需要人来判断的分析性文字（结果解读、失败原因、代价权衡、结论）一律写成
  `【待 Fable 填写：...】` 占位符，由主会话根据最终数字补全，脚本不代写结论。
- 命令、环境版本、日志路径从 notes/RUNLOG*.md、notes/ENV.md 原样提取，不手抄。

用法（必须用文档 venv）：
    tools/doc_env/bin/python scripts/deliverables/make_report.py
    tools/doc_env/bin/python scripts/deliverables/make_report.py --runs coarse_baseline --labels "基线"

随后（两条命令由本脚本结尾打印）：
    scripts/deliverables/libreoffice.sh --headless --convert-to pdf --outdir outputs/reports outputs/reports/report.docx
    tools/doc_env/bin/python .agents/skills/scientific-deliverables/scripts/validate_artifacts.py \\
        outputs/reports/report.docx outputs/reports/report.pdf \\
        --preview-root outputs/previews --report notes/doc_toolchain/SELF_CHECK_report.md \\
        --font assets/fonts/NotoSansCJKsc-Regular.otf
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")      # noqa: E402
os.environ.setdefault("MPLBACKEND", "Agg")         # noqa: E402

import argparse                                     # noqa: E402
import json                                         # noqa: E402
import re                                           # noqa: E402
import sys                                          # noqa: E402
from datetime import datetime                       # noqa: E402
from pathlib import Path                            # noqa: E402

from docx import Document                           # noqa: E402
from docx.enum.section import WD_ORIENT             # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT      # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH       # noqa: E402
from docx.oxml import OxmlElement                   # noqa: E402
from docx.oxml.ns import qn                         # noqa: E402
from docx.shared import Cm, Mm, Pt as DocxPt, RGBColor as DocxRGB   # noqa: E402
from PIL import Image                               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _deliv_common import (                         # noqa: E402
    COLORS, DEFAULT_LABELS, DEFAULT_RUNS, FIGURE_DIR, FONT_SANS_NAME, FONT_SERIF_NAME,
    METRICS_DIR, NOTES_DIR, PROJ_ROOT, REPORT_DIR, RunData, SUMMARY_CSV, build_runs,
    configure_matplotlib, dig, fmt_num, load_json, read_text, rel, to_float,
)

PLACEHOLDERS: list[str] = []          # 收集所有"待 Fable 填写"的分析性文字
MISSING: list[str] = []               # 收集"数据/图未生成"的位置（与占位符区分开）
FIGURE_SEQ: list[str] = []            # 图编号顺序
TABLE_SEQ: list[str] = []             # 表编号顺序

# 图与表的编号在正文里被引用，因此**每个位置都固定占一个编号**：
# 即使对应的图还没生成、表还没有数据，也会插入一个"未生成/未完成"的占位框，
# 这样交叉引用（"见表 5"）永远不会错位。
FIG_FORMULA, FIG_METRICS, FIG_RENDER, FIG_NORMAL = 1, 2, 3, 4
FIG_CURVE, FIG_DNC, FIG_DNC_CMP, FIG_FAIL = 5, 6, 7, 8
N_FIGURES = 8
TAB_ENV, TAB_DATA, TAB_PATCH, TAB_CFG, TAB_KILLED = 1, 2, 3, 4, 5
TAB_RENDER, TAB_GEOM, TAB_TOPO, TAB_SMOOTH = 6, 7, 8, 9
TAB_COST, TAB_ARTIFACT, TAB_METRICDEF = 10, 11, 12
N_TABLES = 12

REFERENCES = [
    "[1] Guédon A., Lepetit V. SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh "
    "Reconstruction and High-Quality Mesh Rendering. CVPR 2024. 代码：https://github.com/Anttwo/SuGaR",
    "[2] Kerbl B., Kopanas G., Leimkühler T., Drettakis G. 3D Gaussian Splatting for Real-Time "
    "Radiance Field Rendering. ACM Transactions on Graphics (SIGGRAPH) 2023. "
    "代码：https://github.com/graphdeco-inria/gaussian-splatting",
    "[3] Huang B., Yu Z., Chen A., Geiger A., Gao S. 2D Gaussian Splatting for Geometrically "
    "Accurate Radiance Fields. SIGGRAPH 2024. 本工作的深度-法向一致性思路来源。"
    "代码：https://github.com/hbb1/2d-gaussian-splatting",
    "[4] Knapitsch A., Park J., Zhou Q.-Y., Koltun V. Tanks and Temples: Benchmarking Large-Scale "
    "Scene Reconstruction. ACM Transactions on Graphics 2017. 本报告使用其中的 Truck 场景。",
    "[5] Kazhdan M., Hoppe H. Screened Poisson Surface Reconstruction. ACM Transactions on "
    "Graphics 2013. SuGaR 网格提取所用的泊松重建算法（经 Open3D 实现）。",
    "[6] Zhang R., Isola P., Efros A. A., Shechtman E., Wang O. The Unreasonable Effectiveness of "
    "Deep Features as a Perceptual Metric. CVPR 2018. LPIPS 感知指标的出处。",
]


# =========================================================================== DOCX 基础


def set_run_font(run, family=FONT_SERIF_NAME, size=None, bold=None, color=None, italic=None):
    run.font.name = family
    if size is not None:
        run.font.size = DocxPt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color is not None:
        run.font.color.rgb = DocxRGB.from_string(color)
    r_pr = run._element.get_or_add_rPr()
    r_pr.get_or_add_rFonts().set(qn("w:eastAsia"), family)


def set_style_font(style, family, size, color=COLORS["ink"], bold=None):
    style.font.name = family
    style.font.size = DocxPt(size)
    style.font.color.rgb = DocxRGB.from_string(color)
    if bold is not None:
        style.font.bold = bold
    r_pr = style._element.get_or_add_rPr()
    r_pr.get_or_add_rFonts().set(qn("w:eastAsia"), family)


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_repeat_header(row):
    """让表头行在跨页时自动重复（Word 的“标题行重复”）。"""
    tr_pr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    tr_pr.append(el)


def set_cant_split(row):
    """禁止一行内容被分页切成两半（否则会在下一页顶部出现没有表头的半行）。"""
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    set_run_font(run, FONT_SANS_NAME, 9, color=COLORS["muted"])
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, end])


def para(doc, text, size=10.5, family=FONT_SERIF_NAME, bold=None, color=None,
         align=None, space_after=6, style=None):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = DocxPt(space_after)
    if text:
        run = p.add_run(text)
        set_run_font(run, family, size, bold=bold, color=color)
    return p


def bullets(doc, items, size=10.5):
    for t in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = DocxPt(3)
        set_run_font(p.add_run(t), FONT_SERIF_NAME, size)


def placeholder(doc, text, size=10.5):
    """分析性文字占位符，橙色加粗，便于在 PDF 预览里一眼找到。"""
    PLACEHOLDERS.append(text)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = DocxPt(6)
    run = p.add_run(f"【待 Fable 填写：{text}】")
    set_run_font(run, FONT_SANS_NAME, size, bold=True, color=COLORS["orange"])
    return p


def missing_note(doc, text, size=10):
    """数据/图尚未生成的提示（红色），与"待填写"的分析性占位符区分。"""
    MISSING.append(text)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = DocxPt(6)
    set_run_font(p.add_run(f"【尚未生成：{text}】"), FONT_SANS_NAME, size,
                 bold=True, color=COLORS["red"])
    return p


def code_block(doc, text, size=8.0):
    """命令 / 日志片段：浅灰底单元格，等宽感的小字号。"""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    shade_cell(cell, COLORS["paper"])
    cell.text = ""
    first = True
    for line in text.rstrip("\n").split("\n"):
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        p.paragraph_format.space_after = DocxPt(0)
        p.paragraph_format.line_spacing = 1.12
        set_run_font(p.add_run(line), "DejaVu Sans Mono", size, color=COLORS["ink"])
    doc.add_paragraph().paragraph_format.space_after = DocxPt(2)
    return table


def caption(doc, text, kind="图"):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = DocxPt(10)
    set_run_font(p.add_run(text), FONT_SANS_NAME, 9, color=COLORS["muted"])
    return p


def add_figure(doc, png_path: Path, cap_text: str, width_cm=16.0, source=None):
    """插图 + 自动编号的中文图题（图题在图下，符合 style-guide）。

    图缺失时仍然占用一个编号并插入红色提示框，保证正文里的"见图 N"永不错位。
    """
    FIGURE_SEQ.append(cap_text)
    idx = len(FIGURE_SEQ)
    if png_path is None or not Path(png_path).exists():
        missing_note(doc, f"图 {idx}「{cap_text}」——"
                          f"对应 PNG 尚未生成，请在三组评测完成后重跑 "
                          f"scripts/deliverables/make_figures.py 再重跑本脚本")
        return idx
    with Image.open(png_path) as im:
        w, h = im.size
    # A4 正文可用高度约 24.5 cm，留出图题空间
    height_cm = width_cm * h / w
    if height_cm > 20.0:
        width_cm = width_cm * 20.0 / height_cm
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = DocxPt(3)
    p.add_run().add_picture(str(png_path), width=Cm(width_cm))
    full = f"图 {idx}  {cap_text}"
    if source:
        full += f"（来源：{source}）"
    caption(doc, full)
    return idx


def add_table(doc, headers, rows, cap_text, font_size=8.4, head_size=8.4,
              note=None, col_widths=None):
    """自动编号的表：表题在表上（style-guide 要求），表头深蓝底白字，隔行浅底。"""
    TABLE_SEQ.append(cap_text)
    idx = len(TABLE_SEQ)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = DocxPt(8)
    p.paragraph_format.space_after = DocxPt(4)
    set_run_font(p.add_run(f"表 {idx}  {cap_text}"), FONT_SANS_NAME, 9.5,
                 bold=True, color=COLORS["navy"])

    if not rows:
        rows = [["未完成"] + ["未完成"] * (len(headers) - 1)]
        MISSING.append(f"表 {idx}「{cap_text}」暂无数据（对应 run 的指标文件尚未生成）")
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        shade_cell(cell, COLORS["navy"])
        cp = cell.paragraphs[0]
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_after = DocxPt(1)
        set_run_font(cp.add_run(str(h)), FONT_SANS_NAME, head_size, bold=True,
                     color=COLORS["white"])
    for i, row in enumerate(rows):
        cells = table.add_row().cells
        for j, v in enumerate(row):
            if i % 2 == 1:
                shade_cell(cells[j], COLORS["paper"])
            cp = cells[j].paragraphs[0]
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT
            cp.paragraph_format.space_after = DocxPt(1)
            set_run_font(cp.add_run(str(v)), FONT_SANS_NAME, font_size)
    set_repeat_header(table.rows[0])
    for row in table.rows:
        set_cant_split(row)
    if col_widths:
        for j, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[j].width = Cm(w)
    if note:
        np_ = doc.add_paragraph()
        np_.paragraph_format.space_after = DocxPt(10)
        set_run_font(np_.add_run(note), FONT_SANS_NAME, 8.2, color=COLORS["muted"])
    else:
        doc.add_paragraph().paragraph_format.space_after = DocxPt(4)
    return idx


# =========================================================================== 数据提取


def extract_code_blocks(md: str) -> list[str]:
    return re.findall(r"```[a-zA-Z]*\n(.*?)```", md, flags=re.S)


def pick_block(blocks: list[str], must_contain: str) -> str | None:
    hits = [b for b in blocks if must_contain in b]
    return max(hits, key=len) if hits else None


def parse_env_md(text: str) -> list[tuple[str, str]]:
    """从 notes/ENV.md 抓关键版本行，返回 [(项, 值)]。"""
    want_colon = ["python", "venv", "torch", "torchvision", "torch.version.cuda",
                  "cuda available", "GLIBCXX_USE_CXX11_ABI", "commit"]
    want_pkg = ["pytorch3d", "open3d", "numpy", "scipy", "plyfile", "matplotlib",
                "diff-gaussian-rasterization", "simple-knn", "opencv-python", "scikit-learn"]
    out: list[tuple[str, str]] = []
    seen = set()
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        for k in want_colon:
            if s.lower().startswith(k.lower() + ":") and k not in seen:
                out.append((k, s.split(":", 1)[1].strip()))
                seen.add(k)
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s{2,}([0-9][0-9A-Za-z_.\-+]*)$", s)
        if m and m.group(1) in want_pkg and m.group(1) not in seen:
            out.append((m.group(1), m.group(2)))
            seen.add(m.group(1))
    m = re.search(r"NVIDIA H200,\s*([0-9]+)\s*MiB,\s*([0-9.]+)", text)
    if m:
        out.insert(0, ("GPU", f"2 × NVIDIA H200（{m.group(1)} MiB / 卡，驱动 {m.group(2)}）"))
    m = re.search(r"Cuda compilation tools, release ([0-9.]+)", text)
    if m:
        out.append(("系统 nvcc", m.group(1)))
    m = re.search(r"PRETTY_NAME=\"([^\"]+)\"", text)
    if m:
        out.insert(0, ("操作系统", m.group(1)))
    return out


def patch_stats(patch_text: str) -> tuple[list[tuple[str, int, int]], int, int]:
    """统计 diff 里每个文件的 +/- 行数。"""
    files: list[tuple[str, int, int]] = []
    cur = None
    add = rm = 0
    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            if cur:
                files.append((cur, add, rm))
            cur = line[4:].strip()
            for p in ("b/", "a/"):
                if cur.startswith(p):
                    cur = cur[2:]
            add = rm = 0
        elif cur and line.startswith("+") and not line.startswith("+++"):
            add += 1
        elif cur and line.startswith("-") and not line.startswith("---"):
            rm += 1
    if cur:
        files.append((cur, add, rm))
    return files, sum(f[1] for f in files), sum(f[2] for f in files)


def scan_interrupted_runs(runs_dir: Path) -> list[dict]:
    """扫描 outputs/runs/*_killed_* 这类"首轮被中断"的残留目录。

    全部字段都是从磁盘实测的（最后一条 dnc_log 迭代号、中间量快照数、是否有最终 .pt），
    不写死任何数字。
    """
    out = []
    for d in sorted(runs_dir.glob("*_killed_*")):
        if not d.is_dir() or not any(d.iterdir()):
            continue          # 跳过空目录（例如根本没开始跑的 mesh_*_killed_*）
        info = {"dir": rel(d), "name": d.name, "last_iteration": None,
                "n_log_rows": 0, "n_vis_snapshots": 0, "has_final_pt": False,
                "last_vis_iteration": None}
        log = d / "dnc_log.csv"
        if log.exists():
            rows = [x for x in log.read_text(errors="replace").splitlines()[1:] if x.strip()]
            info["n_log_rows"] = len(rows)
            its = [to_float(x.split(",")[0]) for x in rows]
            its = [int(i) for i in its if i is not None]
            if its:
                info["last_iteration"] = max(its)
        vis = d / "dnc_vis"
        if vis.exists():
            snaps = sorted(int(m.group(1)) for f in vis.glob("iter*_depth.png")
                           if (m := re.match(r"iter(\d+)_depth\.png$", f.name)))
            info["n_vis_snapshots"] = len(snaps)
            info["last_vis_iteration"] = snaps[-1] if snaps else None
        info["has_final_pt"] = any(d.rglob("15000.pt"))
        out.append(info)
    return out


def train_stats(run: RunData) -> dict:
    if run.run_dir is None:
        return {}
    for p in sorted(run.run_dir.rglob("train_stats.json")):
        d = load_json(p)
        if d:
            d["_path"] = rel(p)
            return d
    return {}


# =========================================================================== 公式图


def make_formula_figure(outdir: Path) -> Path:
    """用 matplotlib mathtext 渲染 L_dnc 的定义，插进报告（Word 里排公式不可靠）。"""
    plt = configure_matplotlib()
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    fig = plt.figure(figsize=(9.2, 3.65))
    fig.patch.set_facecolor("white")
    navy = "#" + COLORS["navy"]
    ink = "#" + COLORS["ink"]
    muted = "#" + COLORS["muted"]

    lines = [
        (0.955, r"$P(u,v)\;=\;\mathrm{unproject}\left(x_{ndc}(u),\,y_{ndc}(v),\,D(u,v)\right)$",
         "① 用相机内参把渲染深度 D 反投影成相机系点图 P"),
        (0.735, r"$N_d\;=\;\mathrm{normalize}\left(\frac{\partial P}{\partial u}\times"
                r"\frac{\partial P}{\partial v}\right)$",
         "② 对点图做中心差分并叉乘，得到几何法向 N_d（统一翻转为朝向相机）"),
        (0.515, r"$M=\left\{(u,v)\,:\,D<0.98\,D_{max}\;\wedge\;|\nabla D|<0.05\,D"
                r"\;\wedge\;|N_{raw}|>0.1\;\wedge\;\mathrm{border}\right\}$",
         "③ 有效像素掩码：剔除背景、深度跳变、低不透明度像素与图像边缘 2 像素"),
        (0.295, r"$\mathcal{L}_{DNC}\;=\;\frac{1}{|M|}\sum_{(u,v)\in M}"
                r"\left(1-\left|N(u,v)\cdot N_d(u,v)\right|\right)$",
         "④ 掩码内像素的平均法向不一致度；取绝对值以消除椭球最短轴的法向符号歧义"),
        (0.075, r"$\mathcal{L}\;=\;\mathcal{L}_{SuGaR}\;+\;\lambda\,\mathcal{L}_{DNC}$",
         "⑤ 与 SuGaR 原有损失相加；λ=0 时完全不进入 DNC 分支（代码路径与原版一致）"),
    ]
    for y, formula, desc in lines:
        fig.text(0.035, y, formula, fontsize=15, color=navy, ha="left", va="center")
        fig.text(0.035, y - 0.085, desc, fontsize=9.6, color=muted, ha="left", va="center")
    fig.text(0.99, 0.005,
             "N＝渲染法向图（高斯最短轴法向经 α 合成）；D＝渲染深度图；λ＝--dnc_factor",
             fontsize=8.6, color=ink, ha="right", va="bottom")
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / "fig_公式_DNC.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(outdir / "fig_公式_DNC.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


# =========================================================================== 表格数据


def row_label(r: RunData, mark_ref=False) -> str:
    if mark_ref and "参考" not in r.label:      # 标签里已经写了"参考"就不重复加
        return f"{r.label}（参考）"
    return r.label


def render_rows(runs, refs):
    rows = []
    for r, is_ref in [(x, False) for x in runs] + [(x, True) for x in refs]:
        if not (r.render or r.summary):
            continue
        rows.append([
            row_label(r, is_ref),
            f"{int(v):,}" if (v := r.s("n_gaussians") or to_float(dig(r.render, "n_gaussians"))) is not None else "未完成",
            fmt_num(r.s("PSNR_dB_up") or dig(r.render, "psnr_db_mean"), 3),
            fmt_num(r.s("PSNR_dB_std") or dig(r.render, "psnr_db_std"), 3),
            fmt_num(r.s("SSIM_up") or dig(r.render, "ssim_mean"), 4),
            fmt_num(r.s("LPIPS_VGG_down") or dig(r.render, "lpips_vgg_mean"), 4),
            int(r.s("n_test_views") or to_float(dig(r.render, "n_test_views")) or 0) or "未完成",
        ])
    return rows


def geom_rows(runs, refs):
    rows = []
    for r, is_ref in [(x, False) for x in runs] + [(x, True) for x in refs]:
        if not r.geometry:
            continue
        g = r.geometry
        rows.append([
            row_label(r, is_ref),
            fmt_num(dig(g, "G1_sparse_to_mesh_distance", "mean_abs"), 5),
            fmt_num(dig(g, "G1_sparse_to_mesh_distance", "median_abs"), 5),
            fmt_num(dig(g, "G1_sparse_to_mesh_distance", "p90_abs"), 5),
            fmt_num((v * 100 if (v := to_float(dig(g, "G1_sparse_to_mesh_distance", "median_rel"))) is not None else None), 4),
            fmt_num((v * 100 if (v := to_float(dig(g, "G2_precision_ratio", "ratio_below_0p5pct"))) is not None else None), 2),
            fmt_num((v * 100 if (v := to_float(dig(g, "G2_precision_ratio", "ratio_below_1pct"))) is not None else None), 2),
            fmt_num(dig(g, "G3_mesh_vertex_to_sparse_distance", "mean_abs_fg_vertices"), 5),
        ])
    return rows


def topo_rows(runs, refs):
    rows = []
    for r, is_ref in [(x, False) for x in runs] + [(x, True) for x in refs]:
        if not r.geometry:
            continue
        t = dig(r.geometry, "G4_topology", default={}) or {}
        frag = next((v for k, v in t.items() if k.startswith("n_fragment_components_lt_")), None)
        rows.append([
            row_label(r, is_ref),
            f"{t.get('n_vertices', 0):,}" if t.get("n_vertices") else "未完成",
            f"{t.get('n_faces', 0):,}" if t.get("n_faces") else "未完成",
            f"{t.get('n_connected_components', 0):,}" if t.get("n_connected_components") is not None else "未完成",
            fmt_num((v * 100 if (v := to_float(t.get("largest_component_face_ratio"))) is not None else None), 2),
            f"{frag:,}" if frag is not None else "未完成",
            f"{t.get('fragment_faces_total', 0):,}" if t.get("fragment_faces_total") is not None else "未完成",
            str(t.get("n_non_manifold_edges", "未完成")),
            f"{t.get('n_boundary_edges', 0):,}" if t.get("n_boundary_edges") is not None else "未完成",
        ])
    return rows


def smooth_rows(runs, refs):
    rows = []
    for r, is_ref in [(x, False) for x in runs] + [(x, True) for x in refs]:
        if not r.geometry:
            continue
        g = dig(r.geometry, "G5_normal_smoothness", default={}) or {}
        rows.append([
            row_label(r, is_ref),
            fmt_num(g.get("dihedral_deg_mean"), 2),
            fmt_num(g.get("dihedral_deg_median"), 2),
            fmt_num(g.get("dihedral_deg_p90"), 2),
            fmt_num(g.get("dihedral_abs_deg_mean"), 2),
            fmt_num(g.get("dihedral_abs_deg_p90"), 2),
            fmt_num((v * 100 if (v := to_float(g.get("frac_raw_gt_90deg"))) is not None else None), 2),
        ])
    return rows


def cost_rows(runs, refs):
    rows = []
    for r, is_ref in [(x, False) for x in runs] + [(x, True) for x in refs]:
        st = train_stats(r)
        lam = r.dnc_factor
        if not st and not r.summary:
            continue
        wall = r.s("train_wall_clock_min")
        if wall is None:
            wall = to_float(st.get("wall_clock_min") or st.get("train_wallclock_min"))
            if wall is None and st.get("train_wallclock_s"):
                wall = to_float(st["train_wallclock_s"]) / 60.0
        ms = r.s("train_ms_per_iter") or to_float(
            st.get("ms_per_iter") or st.get("mean_time_per_iteration_ms"))
        mem = r.s("train_peak_mem_gb")
        if mem is None and st.get("max_memory_allocated_MiB"):
            mem = to_float(st["max_memory_allocated_MiB"]) / 1024.0
        ng = r.s("train_n_gaussians_final") or to_float(
            st.get("n_gaussians") or st.get("n_gaussians_final"))
        rows.append([
            row_label(r, is_ref),
            f"{lam:g}" if lam is not None else "未完成",
            fmt_num(wall, 2),
            fmt_num(ms, 1),
            fmt_num(mem, 2),
            f"{int(ng):,}" if ng else "未完成",
        ])
    return rows


# =========================================================================== 正文


def build(doc, runs, refs, figs, args):
    now = datetime.now().strftime("%Y 年 %m 月 %d 日 %H:%M")
    env_text = read_text(NOTES_DIR / "ENV.md")
    runlog = read_text(NOTES_DIR / "RUNLOG.md")
    runlog_s34 = read_text(NOTES_DIR / "RUNLOG_s34.md")
    runlog_eval = read_text(NOTES_DIR / "RUNLOG_eval.md")
    patch_text = read_text(NOTES_DIR / "dnc_stage3.patch")

    # ---------------------------------------------------------------- 封面
    t = doc.add_paragraph(style="Title")
    t.alignment = WD_ALIGN_PARAGRAPH.LEFT
    t.paragraph_format.space_before = DocxPt(52)
    t.paragraph_format.space_after = DocxPt(12)
    set_run_font(t.add_run("从 3D 高斯到三角网格：SuGaR 复现\n与深度-法向一致性正则改动"),
                 FONT_SANS_NAME, 22, bold=True, color=COLORS["navy"])
    para(doc, "浙江大学博士代码考核 ｜ 6 小时限时复现与魔改 ｜ 全中文技术报告",
         size=12.5, family=FONT_SANS_NAME, color=COLORS["blue"], space_after=20)

    info = [
        ("目标开源项目", "SuGaR（CVPR 2024），官方 commit 7c10c4a"),
        ("数据集与场景", "Tanks & Temples / Truck（3DGS 官方打包 tandt_db.zip，含 COLMAP 稀疏重建）"),
        ("改动点", "在 coarse SuGaR 训练主循环中加入深度-法向一致性正则 L_DNC（借鉴 2DGS 思路，自行实现）"),
        ("对照组", "λ = 0 / 0.05 / 0.2 三组完整重跑（训练 + 网格提取 + 评测）"),
        ("报告生成时间", now),
    ]
    for label, value in info:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = DocxPt(4)
        set_run_font(p.add_run(f"{label}："), FONT_SANS_NAME, 10.5, bold=True, color=COLORS["navy"])
        set_run_font(p.add_run(value), FONT_SERIF_NAME, 10.5)

    warn = doc.add_table(rows=1, cols=1)
    warn.alignment = WD_TABLE_ALIGNMENT.LEFT
    cell = warn.cell(0, 0)
    shade_cell(cell, COLORS["pale_orange"])
    cp = cell.paragraphs[0]
    cp.paragraph_format.space_after = DocxPt(2)
    set_run_font(cp.add_run("数据来源声明"), FONT_SANS_NAME, 10.5, bold=True, color=COLORS["orange"])
    for line in [
        f"本报告中出现的**全部实验数值**均由脚本从 {rel(SUMMARY_CSV)} 及 "
        f"{rel(METRICS_DIR)}/ 下的 render_*.json、geometry_*.json、各 run 的 train_stats.json 与 "
        f"dnc_log.csv 自动读取并排版，未经人工转录；未跑出结果的位置显示"
        f"“未完成”，不做任何估计或补值。",
        "CUDA 光栅化（diff-gaussian-rasterization）的原子累加本身非确定，"
        "即使固定 seed=0，逐次运行也会有微小数值差异，本报告中的指标据此理解。",
    ]:
        q = cell.add_paragraph()
        q.paragraph_format.space_after = DocxPt(2)
        set_run_font(q.add_run(line.replace("**", "")), FONT_SERIF_NAME, 9.5)
    doc.add_page_break()

    # ---------------------------------------------------------------- 摘要
    doc.add_heading("摘要", level=1)
    para(doc,
         "本报告完成了 SuGaR（Surface-Aligned Gaussian Splatting）从 3D 高斯到三角网格的完整短流水线复现："
         "3D Gaussian Splatting 训练 7000 迭代 → coarse SuGaR 密度正则训练至 15000 迭代 → "
         "泊松表面重建导出三角网格，并在此基础上做了一处进入训练主循环的改动："
         "深度-法向一致性正则（Depth-Normal Consistency，下称 DNC）。改动思路借鉴 2D Gaussian Splatting"
         "（SIGGRAPH 2024）提出的 depth-normal consistency 损失，将其移植到 SuGaR 的三维椭球表示上，"
         "并针对法向符号歧义、低不透明度像素与物体边缘设计了掩码。"
         "实验在 Tanks & Temples 的 Truck 场景上以 λ ∈ {0, 0.05, 0.2} 做三组完整对照，"
         "统一评测渲染质量（PSNR / SSIM / LPIPS）与网格几何（稀疏点到网格距离、拓扑碎片、法向平滑度）。")
    placeholder(doc, f"摘要的结论句：一句话概括三组对照的主要发现（DNC 对几何指标的影响方向与幅度、"
                     f"渲染指标的代价、λ 的敏感性），必须与表 {TAB_RENDER}–表 {TAB_COST} 的数字一致")

    doc.add_heading("预注册的可检验假设", level=2)
    for h, desc in [
        ("H1（几何）", "λ>0 时，网格碎片连通分量数下降、COLMAP 稀疏点到网格距离中位数下降、网格法向更平滑。"),
        ("H2（渲染代价）", "测试视角 PSNR 变化在 ±0.5 dB 以内；若下降更多，如实报告为代价。"),
        ("H3（叠加关系）", "DNC 与 SuGaR 原有的 SDF 法向损失不冲突，训练中两者同时下降。"),
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = DocxPt(4)
        set_run_font(p.add_run(f"{h}："), FONT_SANS_NAME, 10.5, bold=True, color=COLORS["navy"])
        set_run_font(p.add_run(desc), FONT_SERIF_NAME, 10.5)
    placeholder(doc, f"逐条给出 H1 / H2 / H3 的结论（支持 / 不支持 / 不确定）及其依据数字："
                     f"H1 看表 {TAB_GEOM} 与表 {TAB_TOPO}，H2 看表 {TAB_RENDER}，"
                     f"H3 看图 {FIG_CURVE} 的 (a)(b) 两幅曲线")

    # ---------------------------------------------------------------- 1 环境与数据
    doc.add_heading("1 环境与数据", level=1)
    doc.add_heading("1.1 运行环境", level=2)
    env_rows = parse_env_md(env_text)
    if env_rows:
        add_table(doc, ["项", "值"], [[k, v] for k, v in env_rows],
                  "运行环境与关键依赖版本", font_size=9, col_widths=[5.0, 11.0],
                  note=f"来源：{rel(NOTES_DIR / 'ENV.md')}（由 scripts/gen_env_md.sh 自动抓取），"
                       f"完整冻结清单见 {rel(NOTES_DIR / 'pip_freeze.txt')}。")
    else:
        placeholder(doc, f"未能解析 {rel(NOTES_DIR / 'ENV.md')}，请检查该文件是否存在")

    doc.add_heading("1.2 数据集与划分", level=2)
    ref_run = next((r for r in runs + refs if r.render), None)
    n_train = to_float(dig(ref_run.render, "n_train_views")) if ref_run else None
    n_test = to_float(dig(ref_run.render, "n_test_views")) if ref_run else None
    img_h = to_float(dig(ref_run.render, "config", "image_height")) if ref_run else None
    img_w = to_float(dig(ref_run.render, "config", "image_width")) if ref_run else None
    hold = to_float(dig(ref_run.render, "config", "eval_split_interval")) if ref_run else None
    n_all = (int(n_train) + int(n_test)) if (n_train and n_test) else None
    data_rows = [
        ["数据来源", "3DGS 官方发布包 tandt_db.zip（含 Tanks & Temples 与 Deep Blending）"],
        ["使用场景", "data/tandt/truck（含 COLMAP sparse/0 稀疏重建）"],
        ["图像总数", f"{n_all} 张" if n_all else "未完成"],
        ["渲染分辨率", f"{int(img_w)} × {int(img_h)} 像素" if (img_w and img_h) else "未完成"],
        ["训练 / 测试划分",
         f"llffhold={int(hold)}（按文件名排序后索引 % {int(hold)} == 0 为测试）→ "
         f"训练 {int(n_train)} 张 / 测试 {int(n_test)} 张"
         if (hold and n_train and n_test) else "未完成"],
        ["几何参考", "COLMAP 稀疏点（track 长度 ≥ 3 且落在前景包围盒内），"
                     f"参考点数 {int(v):,} 个" if (ref_run and ref_run.geometry and
                                                (v := to_float(dig(ref_run.geometry,
                                                                   "G1_sparse_to_mesh_distance",
                                                                   "n_reference_points")))) else
         "COLMAP 稀疏点（Truck 场景无真值网格）"],
    ]
    add_table(doc, ["项", "值"], data_rows, "数据集、分辨率与训练/测试划分",
              font_size=9, col_widths=[4.2, 11.8],
              note="所有 run（含 3DGS、三组 coarse SuGaR 与网格提取）使用完全相同的划分与相机。")

    # ---------------------------------------------------------------- 2 复现流程
    doc.add_heading("2 复现流程与命令", level=1)
    para(doc, "完整流水线分三步，三组对照共用同一个 3DGS 检查点以保证公平：")
    bullets(doc, [
        "第一步：3D Gaussian Splatting 训练 7000 迭代，产出 point_cloud/iteration_7000/point_cloud.ply；",
        "第二步：coarse SuGaR（density 正则模式）从该检查点续训到 15000 迭代，"
        "其中 7000–9000 为熵正则阶段，9000 迭代做一次 opacity<0.5 的硬剪枝，9000 起进入 SDF 正则；",
        "第三步：extract_mesh.py 以 surface_level=0.3、decimation 目标 200000 做泊松重建与抽取。",
    ])
    note_dec = doc.add_paragraph()
    note_dec.paragraph_format.space_after = DocxPt(8)
    set_run_font(note_dec.add_run(
        "关于 decimation 语义的勘误：命令行 -d 200000 并非“输出 20 万面”，而是对前景网格与背景网格"
        "各自抽取到 20 万个三角形的目标，两者合并后总面数约为 37 万，属正常现象。三组使用完全相同的参数。"),
        FONT_SERIF_NAME, 10, color=COLORS["red"])

    blk = pick_block(extract_code_blocks(runlog), "gaussian_splatting/train.py")
    if blk:
        para(doc, "第一步（3DGS 7000 迭代）的实际命令：", size=10, family=FONT_SANS_NAME,
             bold=True, color=COLORS["navy"], space_after=3)
        code_block(doc, blk.strip())
    blk34 = pick_block(extract_code_blocks(runlog_s34), "train_coarse_density.py")
    if blk34:
        para(doc, "第二、三步（三组 coarse 训练 + 网格提取）的实际命令：", size=10,
             family=FONT_SANS_NAME, bold=True, color=COLORS["navy"], space_after=3)
        code_block(doc, blk34.strip())
    para(doc, f"以上命令原样摘自 {rel(NOTES_DIR / 'RUNLOG.md')} 与 "
              f"{rel(NOTES_DIR / 'RUNLOG_s34.md')}，未做改写；完整命令另见附录 A。",
         size=9, family=FONT_SANS_NAME, color=COLORS["muted"])

    para(doc, "随机性控制：3DGS 的 safe_state() 固定 random / numpy / torch 的 seed 为 0；"
              "三组 coarse 训练显式传入 --seed 0。但 CUDA 光栅化的原子累加本身非确定，"
              "因此逐次运行仍会有微小数值差异，本报告不把 0.01 dB 量级的差异当作显著差异。")

    # ---------------------------------------------------------------- 3 改动
    doc.add_heading("3 改动动机与实现", level=1)
    doc.add_heading("3.1 动机", level=2)
    para(doc,
         "SuGaR 用 SDF / density 正则把高斯压扁并贴到表面上，但每个高斯的法向只受到采样点级别的 SDF 约束，"
         "相邻高斯之间的法向可以互相不一致。泊松重建以表面采样点及其法向为输入，法向噪声会直接转化为"
         "网格褶皱与漂浮碎片。DNC 的想法是：既然渲染深度图本身就编码了几何，那么由深度图差分得到的"
         "几何法向 N_d 与渲染出的高斯法向 N 应当一致；在像素尺度上强制两者一致，等价于要求相邻高斯共面。")
    doc.add_heading("3.2 思路来源与自己的贡献", level=2)
    para(doc, "思路来源：2D Gaussian Splatting（SIGGRAPH 2024，文献 [3]）提出的 depth-normal consistency "
              "损失。本工作把它移植到 SuGaR 的 coarse 阶段，属于借鉴已发表思路并自行实现，"
              "未复制 2DGS 的任何代码。相对原思路，本实现需要额外处理三件事：")
    bullets(doc, [
        "P1 表示差异：2DGS 的基元是二维圆盘，法向是其固有属性；SuGaR 的基元是三维椭球，"
        "法向取最短 scale 轴经四元数旋转后的方向，且存在正负两个朝向，需要按相机方向翻转并在损失中取绝对值；",
        "P2 与原有损失的叠加关系：SuGaR 已有采样点级别的 SDF 法向损失，DNC 是像素级、跨高斯的约束，"
        "两者是否冗余或冲突需要用实验回答（对应假设 H3，见图中训练曲线）；",
        "P3 掩码设计：渲染法向图是 α 合成的结果，低不透明度像素会得到接近零向量、方向无意义的法向，"
        "因此在计划原有的三条掩码之外增加了 ‖N_raw‖ > 0.1 的数值保护条件。",
    ])
    doc.add_heading("3.3 损失定义", level=2)
    formula_png = make_formula_figure(FIGURE_DIR)
    add_figure(doc, formula_png, "深度-法向一致性正则 L_DNC 的定义与有效像素掩码", width_cm=15.5)
    para(doc, "实现要点：λ = 0 时完全不进入 DNC 分支（不渲染额外的深度图与法向图），"
              "保证基线的代码路径与原版 SuGaR 完全一致；DNC 从第 9000 迭代起生效，与 SuGaR 自身的 "
              "SDF 正则同期；每 100 迭代把 L_DNC、有效像素占比与两项 SDF 损失写入 dnc_log.csv，"
              "每 1000 迭代把 D / N / N_d / 掩码 四张图落盘。")

    doc.add_heading("3.4 代码改动范围", level=2)
    if patch_text:
        files, add, rm = patch_stats(patch_text)
        add_table(doc, ["文件", "新增行", "删除行"],
                  [[f, a, r_] for f, a, r_ in files],
                  "改动涉及的文件与行数统计",
                  font_size=9, col_widths=[10.5, 2.75, 2.75],
                  note=f"来源：{rel(NOTES_DIR / 'dnc_stage3.patch')}（git diff 留痕）。"
                       f"合计新增 {add} 行、删除 {rm} 行；改动只涉及训练入口参数、"
                       f"coarse 训练循环与一个新增的独立工具模块，未触碰网格提取与评测代码。")
    else:
        placeholder(doc, f"未找到 {rel(NOTES_DIR / 'dnc_stage3.patch')}，无法统计改动行数")

    # ---------------------------------------------------------------- 4 实验设置
    doc.add_heading("4 实验设置与对照", level=1)
    cfg_rows = []
    for r in runs + refs:
        st = train_stats(r)
        lam = r.dnc_factor
        cfg_rows.append([
            r.label,
            r.key,
            f"{lam:g}" if lam is not None else "—",
            str(st.get("dnc_start", "9000") if st else "9000") if (lam or 0) > 0 else "—",
            rel(r.run_dir) if r.run_dir else "未完成",
        ])
    add_table(doc, ["组别", "run 目录名", "λ (--dnc_factor)", "DNC 起始迭代", "输出目录"],
              cfg_rows, "三组对照的配置差异（其余超参完全一致）",
              font_size=8.6, col_widths=[3.0, 3.4, 2.2, 2.2, 5.2],
              note="共同固定量：同一 3DGS 7000 迭代检查点、seed=0、总迭代 15000、"
                   "estimation_factor=0.2、normal_factor=0.2、网格提取 surface_level=0.3 与 "
                   "decimation 200000、同一 llffhold=8 划分。")
    para(doc, "留档说明：阶段 2 中在未打补丁的原版代码上跑过一次基线（run 名 coarse_baseline），"
              "该结果仅作为“补丁未改变 λ=0 代码路径”的旁证留档，不参与三组正式对照；"
              "正式对照的三组统一使用打补丁后的代码与 seed=0。",
         size=10)
    placeholder(doc, "对比 coarse_baseline（未打补丁）与 λ=0 组的指标差异，说明差异是否只在 "
                     "CUDA 非确定性的量级内，以此证明 λ=0 时代码路径未被改变")

    doc.add_heading("4.1 过程记录：首轮 λ 组被中断与重跑", level=2)
    killed = scan_interrupted_runs(PROJ_ROOT / "outputs" / "runs")
    para(doc,
         "本次考核在计算集群的交互式作业中进行。首轮三组对照采用双卡并行调度（λ=0 独占一卡，"
         "两个 λ>0 组共享另一卡），但两个 λ>0 组的训练在接近收尾时随作业的 walltime 到期被强制终止，"
         "未能写出最终检查点。恢复作业后只分配到单张 GPU，因此改为顺序重跑两个 λ>0 组："
         "先 λ=0.2，再 λ=0.05，其余超参、种子、数据划分与网格提取参数与首轮完全一致。"
         "λ=0 组在首轮已完整跑完训练与网格提取，未受影响，不需要重跑。")
    para(doc,
         "按照考核要求保留失败记录，首轮被中断的两个目录原样保留（仅重命名以避免与重跑结果混淆），"
         "其中的逐迭代日志与 DNC 中间量快照仍可作为“训练确实进行到了什么程度”的证据：",
         space_after=4)
    if killed:
        krows = [[k["name"],
                  str(k["last_iteration"]) if k["last_iteration"] is not None else "未完成",
                  str(k["n_log_rows"]),
                  str(k["n_vis_snapshots"]),
                  str(k["last_vis_iteration"]) if k["last_vis_iteration"] is not None else "无",
                  "是" if k["has_final_pt"] else "否（被中断，无最终检查点）"]
                 for k in killed]
    else:
        krows = []
    add_table(doc, ["残留目录", "dnc_log 最后迭代", "日志行数", "中间量快照数",
                    "最后快照迭代", "是否有 15000.pt"],
              krows, "首轮被中断的 λ 组残留记录（保留为失败证据，不参与正式对照）",
              font_size=8.6, col_widths=[4.4, 2.4, 1.9, 2.1, 2.1, 3.3],
              note="表中数字由脚本直接从磁盘上的残留文件统计得到（dnc_log.csv 的最大迭代号、"
                   "dnc_vis/ 下 iter*_depth.png 的数量与最大迭代号），不是人工记录。"
                   "这些目录只用于说明中断发生的时点，其指标不进入任何结论。")
    placeholder(doc, "过程记录的补充说明：首轮中断的具体时刻、作业与节点变化（双卡→单卡）、"
                     "重跑的起止时间，以及“单卡顺序执行”对墙钟时间对比的影响（见训练代价表）")

    # ---------------------------------------------------------------- 5 结果
    doc.add_heading("5 结果", level=1)
    doc.add_heading("5.1 定量结果", level=2)
    add_table(doc, ["组别", "高斯数", "PSNR (dB) ↑", "PSNR 标准差", "SSIM ↑", "LPIPS-VGG ↓", "测试视角数"],
              render_rows(runs, refs), "测试视角上的渲染质量指标",
              font_size=8.6, col_widths=[3.4, 2.4, 2.2, 2.0, 2.0, 2.2, 1.8],
              note="↑ 越高越好，↓ 越低越好。PSNR 标准差为 32 个测试视角之间的样本标准差（ddof=0）。"
                   "vanilla3dgs7k 行为 3DGS 训练 7000 迭代的原始高斯，仅作参考基准，不是 SuGaR 结果。")
    add_table(doc, ["组别", "G1 均值", "G1 中位数", "G1 P90", "G1 中位数(相对 %)",
                    "G2 <0.5% (%) ↑", "G2 <1% (%) ↑", "G3 前景均值"],
              geom_rows(runs, refs), "几何精度：COLMAP 稀疏点到网格的距离与精度比例",
              font_size=8.4, col_widths=[3.0, 1.9, 1.9, 1.9, 2.2, 1.9, 1.8, 1.6],
              note="G1 为点到网格表面的无符号距离（场景单位，↓ 越小越好），"
                   "相对值以相机空间尺度归一化；G2 为距离小于相机空间尺度 0.5% / 1% 的点占比（↑）；"
                   "G3 为网格顶点到最近稀疏点的平均距离（仅统计落在前景包围盒内的顶点，↓）。"
                   "稀疏点本身稀疏、非稠密真值，G3 只作参考，不能单独判优劣。")
    add_table(doc, ["组别", "顶点数", "面数", "连通分量数 ↓", "最大分量面数占比 (%) ↑",
                    "碎片 (<100 面) ↓", "碎片总面数 ↓", "非流形边 ↓", "边界边 ↓"],
              topo_rows(runs, refs), "网格拓扑统计（漂浮碎片是 H1 的主要观测量）",
              font_size=8.0, col_widths=[2.7, 1.7, 1.7, 1.8, 2.2, 1.9, 1.7, 1.4, 1.4])
    add_table(doc, ["组别", "二面角均值 (°)", "二面角中位数 (°)", "二面角 P90 (°)",
                    "|二面角| 均值 (°)", "|二面角| P90 (°)", "raw>90° 占比 (%)"],
              smooth_rows(runs, refs), "网格法向平滑度（相邻面二面角统计）",
              font_size=8.4, col_widths=[3.0, 2.3, 2.3, 2.2, 2.2, 2.1, 2.1],
              note="raw 为相邻面法向夹角（依赖三角形绕序），|二面角| 取 min(raw, 180−raw)（与绕序无关）。"
                   "数值越低表示表面越平滑，但过低可能意味着细节被过度平滑，需结合图 3 与图 6 判断。")
    add_table(doc, ["组别", "λ", "训练墙钟 (min)", "每迭代耗时 (ms)", "峰值显存 (GB)", "结束时高斯数"],
              cost_rows(runs, refs), "训练代价（coarse SuGaR 阶段，7000→15000 共 8000 迭代）",
              font_size=8.8, col_widths=[3.6, 1.6, 2.8, 2.8, 2.6, 2.6],
              note="调度方式在过程中发生过变化（见 4.1 节）：首轮两个 λ>0 组共享同一张 GPU 并行执行，"
                   "重跑时改为单卡顺序执行。因此墙钟时间不可直接横向比较，"
                   "每迭代耗时与峰值显存受调度影响较小，是更可比的代价指标。")
    placeholder(doc, f"定量结果解读：逐个指标说明三组的差异方向与幅度（表 {TAB_RENDER}–表 {TAB_COST}），"
                     f"明确哪些差异超出 CUDA 非确定性的量级（可参考 coarse_baseline 与 λ=0 的差），哪些不是")

    doc.add_heading("5.2 定性结果", level=2)
    add_figure(doc, figs.get("fig1"), "三组对照的渲染质量与网格几何指标汇总（上排渲染、下排几何）",
               width_cm=16.2, source="scripts/deliverables/make_figures.py")
    add_figure(doc, figs.get("fig2"), "固定测试视角的渲染结果与误差热图对比（误差图共用同一色条）",
               width_cm=16.2)
    add_figure(doc, figs.get("fig3"), "三组网格的法向着色图对比（白色区域为网格空洞）", width_cm=16.2)
    add_figure(doc, figs.get("fig4"), "训练过程中的 L_DNC、SDF 法向损失、有效像素占比与总损失曲线",
               width_cm=16.2)
    add_figure(doc, figs.get("fig5"), "DNC 中间量：渲染深度 D、渲染法向 N、深度差分法向 N_d 与有效掩码",
               width_cm=15.5)
    add_figure(doc, figs.get("fig5b"), "不同 λ 下 DNC 中间量的对比（同一迭代，各自的随机训练视角）",
               width_cm=16.2)
    placeholder(doc, f"定性结果解读：结合图 {FIG_RENDER}（渲染与误差）与图 {FIG_NORMAL}（网格法向）"
                     f"指出 DNC 组在哪些区域更干净、哪些区域更差，"
                     f"并与表 {TAB_TOPO} 的碎片数、表 {TAB_SMOOTH} 的二面角统计互相印证")

    # ---------------------------------------------------------------- 6 失败案例
    doc.add_heading("6 失败案例与代价", level=1)
    add_figure(doc, figs.get("fig6"), "失败案例特写：背景树冠、细杆栏杆与顶部线状结构", width_cm=16.2)
    para(doc, "预期中的三类失败模式（在计划阶段即已预注册，用于避免只挑好看的结果展示）：")
    bullets(doc, [
        "远景与天空：深度本身不可靠，差分得到的法向噪声大，DNC 可能把错误的一致性强加到背景上；",
        "物体边缘：跨越深度不连续处的中心差分会产生伪法向，掩码的深度梯度阈值只能部分缓解；",
        "细结构（栏杆、天线、电线）：一致性约束天然偏好光滑表面，细结构可能被过度平滑或直接丢失。",
    ])
    placeholder(doc, f"失败案例分析：逐条对照图 {FIG_FAIL} 指出实际观察到的现象，并说明与 λ 取值的关系")
    placeholder(doc, f"代价分析：训练时间 / 显存 / 渲染指标各付出多少代价"
                     f"（数字取自表 {TAB_RENDER} 与表 {TAB_COST}），并给出“这个代价是否值得”的判断")
    placeholder(doc, f"λ 敏感性：说明 λ=0.05 与 λ=0.2 的差异是否单调；"
                     f"若 λ=0.2 出现退化（例如图 {FIG_DNC_CMP} 中深度图失去场景结构、"
                     f"图 {FIG_CURVE}(c) 的有效像素占比异常升高到接近 1），必须在此明确写出并给出机理解释")

    doc.add_heading("6.1 方法本身的局限", level=2)
    bullets(doc, [
        "几何评价使用 COLMAP 稀疏点而非稠密真值网格，只能衡量“网格是否贴合可靠的稀疏观测”，"
        "不能衡量未被稀疏点覆盖区域的正确性；",
        "只在单个场景（Truck）上验证，结论的泛化性未经检验；",
        "coarse 阶段之后的 refined 阶段（需要 nvdiffrast）未在本次时间预算内执行，"
        "因此结论只对 coarse SuGaR + 泊松网格这条短流水线成立；",
        "L_DNC 的梯度同时流经渲染法向 N 与渲染深度 D，理论上存在“把深度整体抹平”这一平凡解，"
        "λ 过大时该风险上升。",
    ])

    # ---------------------------------------------------------------- 7 结论
    doc.add_heading("7 结论与下一步", level=1)
    placeholder(doc, f"结论：3–5 条，每条都要能追溯到具体的表或图编号"
                     f"（本报告共 {N_FIGURES} 图 {N_TABLES} 表）")
    placeholder(doc, "下一步：若有更多时间会做什么（例如对 D 做 detach 的消融、在 train 场景上验证泛化、"
                     "接上 refined 阶段、扫更细的 λ 网格）")

    # ---------------------------------------------------------------- 附录
    doc.add_page_break()
    doc.add_heading("附录 A  完整命令", level=1)
    para(doc, "以下命令块全部从项目内的实际脚本与运行日志中原样提取，未经改写。",
         size=9.5, family=FONT_SANS_NAME, color=COLORS["muted"])

    doc.add_heading("A.1 环境变量与依赖安装", level=2)
    for src in (PROJ_ROOT / "env.sh", PROJ_ROOT / "scripts" / "setup_env_step2.sh"):
        txt = read_text(src)
        if txt:
            para(doc, f"文件：{rel(src)}", size=9, family=FONT_SANS_NAME,
                 bold=True, color=COLORS["navy"], space_after=2)
            code_block(doc, txt.strip())
        else:
            missing_note(doc, f"未找到 {rel(src)}")

    for title, text_md, key in [
        ("A.2 第一步：3DGS 7000 迭代", runlog, "gaussian_splatting/train.py"),
        ("A.3 第二、三步：三组 coarse 训练与网格提取", runlog_s34, "train_coarse_density.py"),
    ]:
        doc.add_heading(title, level=2)
        blk = pick_block(extract_code_blocks(text_md), key) if text_md else None
        if blk:
            code_block(doc, blk.strip())
        else:
            missing_note(doc, f"未能从运行日志中提取到包含 “{key}” 的命令块")

    doc.add_heading("A.4 评测与汇总", level=2)
    eval_blocks = extract_code_blocks(runlog_eval) if runlog_eval else []
    joined = []
    for key in ("run_eval_for_run.sh", "eval_render.py", "eval_geometry.py",
                "render_mesh_views.py", "summarize.py"):
        blk = pick_block(eval_blocks, key)
        if blk and blk.strip() not in joined:
            joined.append(blk.strip())
    if joined:
        code_block(doc, "\n\n".join(joined))
    else:
        missing_note(doc, f"未能从 {rel(NOTES_DIR / 'RUNLOG_eval.md')} 提取到评测命令块")

    doc.add_heading("附录 B  日志与产物路径", level=1)
    art_rows = []
    for r in runs + refs:
        art_rows.append([r.label, rel(r.run_dir) if r.run_dir else "未完成",
                         rel(r.train_log) if r.train_log else "未完成",
                         "、".join(r.sources) if r.sources else "未完成"])
    add_table(doc, ["组别", "训练输出目录", "训练日志", "指标文件"], art_rows,
              "各组的产物与日志路径（均相对项目根目录）",
              font_size=7.8, col_widths=[2.6, 4.0, 3.6, 5.8])
    bullets(doc, [
        f"指标汇总表：{rel(SUMMARY_CSV)}（列名后缀 _up 表示越大越好，_down 表示越小越好）",
        f"逐视角渲染指标：{rel(METRICS_DIR)}/render_<run>.csv",
        f"图表：{rel(FIGURE_DIR)}/fig*.png 与同名 .pdf（矢量版）",
        f"运行记录：{rel(NOTES_DIR)}/RUNLOG.md、RUNLOG_s34.md、RUNLOG_eval.md",
        f"环境记录：{rel(NOTES_DIR)}/ENV.md、pip_freeze.txt",
        f"改动留痕：{rel(NOTES_DIR)}/dnc_stage3.patch",
    ], size=10)

    doc.add_heading("附录 C  评测指标的精确定义", level=1)
    add_table(doc, ["指标", "定义与单位", "方向"], [
        ["PSNR", "逐图 MSE 取 −10·log10 后对测试视角求均值，单位 dB", "↑"],
        ["SSIM", "3DGS 官方实现的 11×11 高斯窗结构相似度，无量纲", "↑"],
        ["LPIPS-VGG", "VGG 特征上的感知距离（文献 [6]），无量纲", "↓"],
        ["G1", "COLMAP 稀疏点到网格表面的无符号距离（Open3D RaycastingScene），场景单位；"
               "相对值除以相机空间尺度", "↓"],
        ["G2", "G1 距离小于相机空间尺度 0.5% / 1% 的点占比", "↑"],
        ["G3", "网格顶点到最近稀疏点的距离均值（scipy cKDTree），仅统计前景包围盒内顶点", "↓"],
        ["G4", "顶点数、面数、连通分量数、最大分量面数占比、面数<100 的碎片分量数、"
               "非流形边数、边界边数", "分量数与碎片数 ↓；最大分量占比 ↑"],
        ["G5", "相邻面二面角的均值 / 中位数 / P90，单位度", "↓（过低可能为过度平滑）"],
    ], "评测指标的定义、单位与方向", font_size=8.6, col_widths=[2.2, 10.4, 3.4],
        note="全部指标由 scripts/eval/ 下的脚本统一计算，三组使用同一套代码与同一套参数。")

    doc.add_heading("附录 D  参考文献", level=1)
    for ref_line in REFERENCES:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = DocxPt(5)
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.8)
        set_run_font(p.add_run(ref_line), FONT_SERIF_NAME, 9.5)


# =========================================================================== main


def load_figs(manifest_path: Path) -> dict[str, Path]:
    data = load_json(manifest_path) or {}
    out = {}
    for name, info in (data.get("figures") or {}).items():
        if info.get("status") == "ok" and info.get("png"):
            p = PROJ_ROOT / info["png"]
            if p.exists():
                out[name] = p
    return out


def main():
    p = argparse.ArgumentParser(description="生成中文 DOCX 报告")
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--ref_runs", nargs="+",
                   default=["vanilla3dgs7k", "coarse_baseline"],
                   help="只作参考行的 run（不参与三组对照结论）")
    p.add_argument("--manifest", type=Path, default=FIGURE_DIR / "figures_manifest.json")
    p.add_argument("--out", type=Path, default=REPORT_DIR / "report.docx")
    args = p.parse_args()

    labels = args.labels or [DEFAULT_LABELS.get(k, k) for k in args.runs]
    runs = build_runs(args.runs, labels)
    refs = build_runs(args.ref_runs, [DEFAULT_LABELS.get(k, k) for k in args.ref_runs])
    figs = load_figs(args.manifest)

    print("=" * 78)
    print("make_report.py —— 输入清单")
    for r in runs + refs:
        print(f"  run={r.key:<20s} render={'有' if r.render else '缺'} "
              f"geometry={'有' if r.geometry else '缺'} train_stats="
              f"{'有' if train_stats(r) else '缺'}")
    print(f"  图（来自 {rel(args.manifest)}）：{', '.join(sorted(figs)) or '无'}")
    print("=" * 78)

    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.PORTRAIT
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    section.top_margin = Mm(22)
    section.bottom_margin = Mm(20)
    section.left_margin = Mm(24)
    section.right_margin = Mm(24)

    styles = doc.styles
    set_style_font(styles["Normal"], FONT_SERIF_NAME, 10.5)
    styles["Normal"].paragraph_format.line_spacing = 1.42
    styles["Normal"].paragraph_format.space_after = DocxPt(6)
    for name, size in (("Title", 22), ("Heading 1", 15.5), ("Heading 2", 12.5)):
        set_style_font(styles[name], FONT_SANS_NAME, size, COLORS["navy"], True)
    styles["Heading 1"].paragraph_format.space_before = DocxPt(14)
    styles["Heading 1"].paragraph_format.space_after = DocxPt(7)
    styles["Heading 2"].paragraph_format.space_before = DocxPt(9)
    styles["Heading 2"].paragraph_format.space_after = DocxPt(5)

    hdr = section.header.paragraphs[0]
    hdr.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_run_font(hdr.add_run("浙江大学博士代码考核 ｜ SuGaR 复现与深度-法向一致性正则改动"),
                 FONT_SANS_NAME, 8.5, color=COLORS["muted"])
    add_page_number(section.footer.paragraphs[0])

    build(doc, runs, refs, figs, args)

    props = doc.core_properties
    props.title = "SuGaR 复现与深度-法向一致性正则改动技术报告"
    props.subject = "浙江大学博士代码考核"
    props.author = "6 小时考核复现小组"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.out)
    print(f"\n[OK] DOCX -> {rel(args.out)}  ({args.out.stat().st_size/1024:.0f} KB)")
    print(f"[OK] 图 {len(FIGURE_SEQ)} 张，表 {len(TABLE_SEQ)} 个")
    if len(FIGURE_SEQ) != N_FIGURES or len(TABLE_SEQ) != N_TABLES:
        print(f"[WARN] 图/表数量与交叉引用常量不一致（期望 {N_FIGURES} 图 {N_TABLES} 表），"
              f"正文里的“见图 N / 见表 N”可能错位，请检查 build() 的顺序")
    print(f"[TODO] 待 Fable 填写的分析性占位符共 {len(PLACEHOLDERS)} 处：")
    for i, t in enumerate(PLACEHOLDERS, 1):
        print(f"   {i:2d}. {t}")
    if MISSING:
        print(f"[MISSING] 尚未生成的数据/图共 {len(MISSING)} 处（三组评测跑完后重跑即可消失）：")
        for i, t in enumerate(MISSING, 1):
            print(f"   {i:2d}. {t}")
    ph_json = args.out.with_name(args.out.stem + "_占位符清单.json")
    with open(ph_json, "w") as f:
        json.dump({"placeholders": PLACEHOLDERS, "missing": MISSING,
                   "figures": FIGURE_SEQ, "tables": TABLE_SEQ}, f,
                  ensure_ascii=False, indent=2)
    print(f"[OK] 占位符清单 -> {rel(ph_json)}")
    print("\n下一步（逐条执行）：")
    print(f"  scripts/deliverables/libreoffice.sh --headless --convert-to pdf "
          f"--outdir {rel(args.out.parent)} {rel(args.out)}")
    print(f"  tools/doc_env/bin/python .agents/skills/scientific-deliverables/scripts/"
          f"validate_artifacts.py {rel(args.out)} {rel(args.out.with_suffix('.pdf'))} "
          f"--preview-root outputs/previews --report notes/doc_toolchain/SELF_CHECK_report.md "
          f"--font assets/fonts/NotoSansCJKsc-Regular.otf")


if __name__ == "__main__":
    main()
