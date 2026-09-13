#!/usr/bin/env python
"""S3.f — 从 outputs/metrics/geometry_*.json 生成提取端扫参的并排对照表（Markdown）。
基准 = base_seed0（原版提取：D=10, quantile=0.1）。所有数字直接取自 JSON，不做任何加工。"""
import json, os, sys

PROJ = "/scratch/e1351071/zju_test"
M = os.path.join(PROJ, "outputs", "metrics")
BASE = "coarse_base_seed0"
ORDER = [("coarse_base_seed0", "原版提取 D=10, q=0.1（基准）"),
         ("base_seed0_pdauto", "pdauto: D=auto(=10), q=0.1"),
         ("base_seed0_pdauto_q0", "pdauto_q0: D=auto(=10), q=0"),
         ("base_seed0_q0", "q0: D=10, q=0"),
         ("base_seed0_q005", "q005: D=10, q=0.05"),
         ("base_seed0_pd9", "pd9: D=9, q=0.1"),
         ("base_seed0_pd8", "pd8: D=8, q=0.1")]


def load(run):
    p = os.path.join(M, f"geometry_{run}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_vis(run):
    """meshvis_<run>.json：4 个固定测试视角下 mesh 光栅化的命中率（越高 = 洞越少）。"""
    p = os.path.join(M, f"meshvis_{run}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    cov = d.get("mesh_pixel_coverage") or {}
    vals = [float(v) for v in cov.values()]
    return sum(vals) / len(vals) if vals else None


def frag(t):
    for k, v in t.items():
        if k.startswith("n_fragment_components_lt_"):
            return v
    return None


def row(d):
    g1, g2, g4, g5 = (d["G1_sparse_to_mesh_distance"], d["G2_precision_ratio"],
                      d["G4_topology"], d["G5_normal_smoothness"])
    return dict(
        G1_median_rel_pct=g1["median_rel"] * 100, G1_mean_rel_pct=g1["mean_rel"] * 100,
        G2_1pct=g2["ratio_below_1pct"] * 100, G2_05pct=g2["ratio_below_0p5pct"] * 100,
        n_vertices=g4["n_vertices"], n_faces=g4["n_faces"],
        n_components=g4["n_connected_components"],
        largest_pct=g4["largest_component_face_ratio"] * 100,
        n_frag=frag(g4), frag_faces=g4["fragment_faces_total"],
        boundary=g4["n_boundary_edges"],
        G5_abs_mean=g5["dihedral_abs_deg_mean"], G5_abs_p90=g5["dihedral_abs_deg_p90"])


COLS = [("G1_median_rel_pct", "G1 median rel %↓", 4), ("G1_mean_rel_pct", "G1 mean rel %↓", 4),
        ("G2_1pct", "G2 <1% ↑", 3), ("G2_05pct", "G2 <0.5% ↑", 3),
        ("n_vertices", "顶点数", 0), ("n_faces", "面数", 0),
        ("n_components", "连通分量↓", 0), ("largest_pct", "最大分量面占比%↑", 3),
        ("n_frag", "碎片(<100面)↓", 0), ("frag_faces", "碎片总面数↓", 0),
        ("boundary", "边界边↓", 0), ("hit_pct", "mesh 命中率%↑", 3),
        ("G5_abs_mean", "G5 abs mean°↓", 4), ("G5_abs_p90", "G5 abs P90°↓", 4)]


def main():
    data = {}
    for run, _ in ORDER:
        d = load(run)
        if d:
            data[run] = row(d)
            hv = load_vis(run)
            data[run]["hit_pct"] = hv * 100 if hv is not None and hv <= 1.5 else (hv if hv is not None else float("nan"))
    if BASE not in data:
        sys.exit("no baseline")
    b = data[BASE]
    hdr = "| 组 | " + " | ".join(c[1] for c in COLS) + " |"
    print(hdr)
    print("|" + "---|" * (len(COLS) + 1))
    for run, label in ORDER:
        if run not in data:
            continue
        r = data[run]
        cells = []
        for k, _, nd in COLS:
            v = r[k]
            cells.append(f"{v:.{nd}f}" if nd else f"{int(v)}")
        print(f"| {label} | " + " | ".join(cells) + " |")
    print()
    print("相对基准（base_seed0）的变化百分比：")
    print(hdr)
    print("|" + "---|" * (len(COLS) + 1))
    for run, label in ORDER:
        if run == BASE or run not in data:
            continue
        r = data[run]
        cells = []
        for k, _, _ in COLS:
            bv, v = float(b[k]), float(r[k])
            cells.append("—" if bv == 0 else f"{(v - bv) / bv * 100:+.2f}%")
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
