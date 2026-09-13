#!/usr/bin/env python3
"""交付物脚本的公共工具：路径 / 中文字体 / 配色 / 指标读取。

被 make_figures.py 与 make_report.py 共用。

设计约束（计划 §2b 末尾的诚信条款）：
- 所有数字只能来自 outputs/metrics/summary.csv 与各 metrics JSON（以及训练日志 / dnc_log.csv
  这两类逐迭代记录），**不得在代码里硬编码任何实验数值**；
- 缺失的数据返回 None，由调用方显式标注"未完成 / 无数据"，不得用占位数字冒充；
- 只读，不修改 repo/SuGaR、repo/SuGaR_dev、scripts/eval 下任何文件。

只在 tools/doc_env 的 Python 里运行（python-docx / matplotlib 等只装在该 venv）。
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- 路径

PROJ_ROOT = Path(os.environ.get("PROJ_ROOT", "/scratch/e1351071/zju_test"))
METRICS_DIR = PROJ_ROOT / "outputs" / "metrics"
VIS_DIR = PROJ_ROOT / "outputs" / "vis"
RUNS_DIR = PROJ_ROOT / "outputs" / "runs"
FIGURE_DIR = PROJ_ROOT / "outputs" / "figures"
REPORT_DIR = PROJ_ROOT / "outputs" / "reports"
PREVIEW_DIR = PROJ_ROOT / "outputs" / "previews"
LOG_DIR = PROJ_ROOT / "logs"
NOTES_DIR = PROJ_ROOT / "notes"
FONT_DIR = PROJ_ROOT / "assets" / "fonts"
SUMMARY_CSV = METRICS_DIR / "summary.csv"

FONT_SANS_OTF = FONT_DIR / "NotoSansCJKsc-Regular.otf"
FONT_SANS_BOLD_OTF = FONT_DIR / "NotoSansCJKsc-Bold.otf"
FONT_SERIF_OTF = FONT_DIR / "NotoSerifCJKsc-Regular.otf"
FONT_SANS_NAME = "Noto Sans CJK SC"
FONT_SERIF_NAME = "Noto Serif CJK SC"

# --------------------------------------------------------------------------- 配色
# 取自 .agents/skills/scientific-deliverables/references/style-guide.md

COLORS = {
    "navy": "123B5D",
    "blue": "2F6B9A",
    "teal": "138A8A",
    "orange": "D97706",
    "red": "B33A3A",
    "ink": "263238",
    "muted": "60717D",
    "paper": "F4F7F9",
    "white": "FFFFFF",
    "line": "D5E0E7",
    "pale_blue": "EAF1F6",
    "pale_teal": "E8F4F3",
    "pale_orange": "FBF0E3",
}


def hexc(name: str) -> str:
    return "#" + COLORS[name]


# run -> 绘图颜色（基线冷色、DNC 组暖色，λ 越大越暖）
RUN_COLOR_KEYS = {
    "coarse_base_seed0": "blue",
    "coarse_baseline": "navy",
    "coarse_dnc005": "teal",
    "coarse_dnc02": "orange",
    "vanilla3dgs7k": "muted",
}
FALLBACK_COLOR_KEYS = ["blue", "teal", "orange", "navy", "red", "muted"]

DEFAULT_RUNS = ["coarse_base_seed0", "coarse_dnc005", "coarse_dnc02"]
DEFAULT_LABELS = {
    "coarse_base_seed0": "基线(λ=0)",
    "coarse_dnc005": "DNC λ=0.05",
    "coarse_dnc02": "DNC λ=0.2",
    "coarse_baseline": "基线(阶段2 留档)",
    "vanilla3dgs7k": "3DGS 7k(参考)",
}

# --------------------------------------------------------------------------- 通用小工具


def normalize_run(name: str) -> str:
    """与 scripts/eval/summarize.py 的 normalize() 一致：去掉 coarse_/mesh_/meshvis_ 前缀。

    summary.csv 的 run 列写的是归一化后的名字（coarse_dnc02 -> dnc02），
    而 metrics JSON / outputs/vis / outputs/runs 用的是原始名字，这里做双向查表。
    """
    for prefix in ("coarse_", "mesh_", "meshvis_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_summary(path: Path = SUMMARY_CSV) -> dict[str, dict[str, str]]:
    """读 summary.csv，返回 {run 名: {列名: 字符串值}}；原名与归一化名都做键。"""
    out: dict[str, dict[str, str]] = {}
    if not Path(path).exists():
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            run = row.get("run", "")
            if not run:
                continue
            out[run] = row
            for prefix in ("coarse_", "mesh_"):
                out.setdefault(prefix + run, row)
    return out


def to_float(value: Any) -> float | None:
    """把 CSV/JSON 里的值转成 float；'NA' / '' / None / 非数字 -> None。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return None if v != v else v  # NaN -> None
    s = str(value).strip()
    if s in ("", "NA", "N/A", "nan", "None", "--"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return None if v != v else v


def fmt_num(value: Any, nd: int = 3, na: str = "未完成") -> str:
    v = to_float(value)
    if v is None:
        return na
    return f"{v:.{nd}f}"


def dig(data: dict | None, *keys, default=None):
    """按路径取嵌套字典的值，任一层缺失返回 default。"""
    cur: Any = data
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# --------------------------------------------------------------------------- Run 数据


@dataclass
class RunData:
    """一组 run 的全部可读产物（缺失项一律为 None / 空，调用方负责优雅降级）。"""

    key: str                      # 原始 run 名，如 coarse_dnc02
    label: str                    # 图表 / 报告里显示的中文名
    color: str                    # '#RRGGBB'
    render: dict | None = None
    geometry: dict | None = None
    meshvis: dict | None = None
    summary: dict[str, str] = field(default_factory=dict)
    vis_dir: Path | None = None
    run_dir: Path | None = None
    dnc_log: Path | None = None
    dnc_vis_dir: Path | None = None
    train_log: Path | None = None
    sources: list[str] = field(default_factory=list)   # 用过的文件（溯源）

    # ---- 取数 ----
    @property
    def dnc_factor(self) -> float | None:
        v = to_float(self.summary.get("dnc_factor"))
        if v is not None:
            return v
        rows = self.read_dnc_log()
        if rows:
            return to_float(rows[0].get("dnc_factor"))
        return None

    def s(self, column: str) -> float | None:
        """summary.csv 里的一列（float）。"""
        return to_float(self.summary.get(column))

    def read_dnc_log(self) -> list[dict[str, str]]:
        if self.dnc_log is None or not self.dnc_log.exists():
            return []
        with open(self.dnc_log, newline="") as f:
            return list(csv.DictReader(f))

    def read_train_loss_curve(self) -> tuple[list[int], list[float]]:
        """从训练日志解析 `loss: 0.049807  [ 9000/15000] computed in ...` 行。"""
        if self.train_log is None or not self.train_log.exists():
            return [], []
        pat = re.compile(r"^loss:\s*([0-9.eE+-]+)\s*\[\s*(\d+)/\s*(\d+)\]")
        it, val = [], []
        with open(self.train_log, errors="replace") as f:
            for line in f:
                m = pat.match(line.strip())
                if m:
                    val.append(float(m.group(1)))
                    it.append(int(m.group(2)))
        return it, val

    def dnc_vis_iterations(self) -> list[int]:
        if self.dnc_vis_dir is None or not self.dnc_vis_dir.exists():
            return []
        its = set()
        for p in self.dnc_vis_dir.glob("iter*_depth.png"):
            m = re.match(r"iter(\d+)_depth\.png$", p.name)
            if m:
                its.add(int(m.group(1)))
        return sorted(its)

    def vis_views(self) -> list[tuple[int, str]]:
        """可视化目录里有哪些固定视角，返回 [(view_idx, image_name), ...]。"""
        if self.vis_dir is None or not self.vis_dir.exists():
            return []
        found = {}
        for p in self.vis_dir.glob("render_view*.png"):
            m = re.match(r"render_view(\d+)_(.+)\.png$", p.name)
            if m:
                found[int(m.group(1))] = m.group(2)
        return sorted(found.items())

    def vis_png(self, kind: str, view_idx: int, image_name: str) -> Path | None:
        """kind ∈ {gt, render, errmap, mesh_normal, mesh_depth}。"""
        if self.vis_dir is None:
            return None
        p = self.vis_dir / f"{kind}_view{view_idx:02d}_{image_name}.png"
        return p if p.exists() else None


def _find_train_log(key: str) -> Path | None:
    """按 run 名在 logs/ 里找训练日志（排除 mesh / eval / wrapper / 失败留档）。

    被中断后重命名的目录（如 coarse_dnc02_killed_0649）对应的日志仍是原名
    （logs/s34_coarse_dnc02.log），所以找不到时去掉 `_killed_*` 后缀再找一次。
    """
    def _search(k: str):
        cands = []
        for p in LOG_DIR.glob(f"*{k}*.log"):
            n = p.name.lower()
            if any(bad in n for bad in ("mesh", "eval", "wrap", "failed", "pid", "verify")):
                continue
            cands.append(p)
        return max(cands, key=lambda p: p.stat().st_size) if cands else None

    hit = _search(key)
    if hit is None:
        base = re.sub(r"_killed_.*$", "", key)
        if base != key:
            hit = _search(base)
    return hit


def build_run(key: str, label: str | None = None, color: str | None = None,
              summary: dict[str, dict[str, str]] | None = None,
              index: int = 0) -> RunData:
    summary = summary if summary is not None else load_summary()
    norm = normalize_run(key)
    row = summary.get(key) or summary.get(norm) or {}

    if color is None:
        ck = RUN_COLOR_KEYS.get(key, FALLBACK_COLOR_KEYS[index % len(FALLBACK_COLOR_KEYS)])
        color = hexc(ck)

    render = load_json(METRICS_DIR / f"render_{key}.json")
    geometry = load_json(METRICS_DIR / f"geometry_{key}.json")
    meshvis = load_json(METRICS_DIR / f"meshvis_{key}.json")
    vis_dir = VIS_DIR / key
    run_dir = RUNS_DIR / key
    dnc_log = run_dir / "dnc_log.csv"
    dnc_vis = run_dir / "dnc_vis"

    sources = []
    for p in (METRICS_DIR / f"render_{key}.json", METRICS_DIR / f"geometry_{key}.json",
              METRICS_DIR / f"meshvis_{key}.json", dnc_log):
        if p.exists():
            sources.append(str(p.relative_to(PROJ_ROOT)))
    if row:
        sources.append(str(SUMMARY_CSV.relative_to(PROJ_ROOT)))

    return RunData(
        key=key,
        label=label or DEFAULT_LABELS.get(key, key),
        color=color,
        render=render,
        geometry=geometry,
        meshvis=meshvis,
        summary=row,
        vis_dir=vis_dir if vis_dir.exists() else None,
        run_dir=run_dir if run_dir.exists() else None,
        dnc_log=dnc_log if dnc_log.exists() else None,
        dnc_vis_dir=dnc_vis if dnc_vis.exists() else None,
        train_log=_find_train_log(key),
        sources=sources,
    )


def build_runs(keys: list[str], labels: list[str] | None = None) -> list[RunData]:
    summary = load_summary()
    if labels and len(labels) != len(keys):
        raise SystemExit(f"[ERROR] --labels 个数({len(labels)}) 与 --runs 个数({len(keys)}) 不一致")
    out = []
    for i, k in enumerate(keys):
        out.append(build_run(k, labels[i] if labels else None, summary=summary, index=i))
    return out


# --------------------------------------------------------------------------- matplotlib


def configure_matplotlib():
    """注册项目内中文字体并设置科研图表基调。必须在任何绘图之前调用。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for path in (FONT_SANS_OTF, FONT_SANS_BOLD_OTF, FONT_SERIF_OTF):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        "font.family": FONT_SANS_NAME,
        "axes.unicode_minus": False,
        "axes.edgecolor": hexc("line"),
        "axes.labelcolor": hexc("ink"),
        "text.color": hexc("ink"),
        "xtick.color": hexc("muted"),
        "ytick.color": hexc("muted"),
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })
    return plt


def save_figure(fig, out_dir: Path, stem: str, dpi: int = 300) -> dict[str, Path]:
    """同时存 PNG（位图，插报告用）与 PDF（矢量，投影/打印用）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    return {"png": png, "pdf": pdf}


def read_text(path: Path, limit: int | None = None) -> str:
    try:
        txt = Path(path).read_text(errors="replace")
    except OSError:
        return ""
    return txt if limit is None else txt[:limit]


def rel(path: Path | str) -> str:
    """相对 PROJ_ROOT 的路径字符串（报告里写路径用）。"""
    try:
        return str(Path(path).resolve().relative_to(PROJ_ROOT))
    except ValueError:
        return str(path)


__all__ = [n for n in dir() if not n.startswith("_")]
