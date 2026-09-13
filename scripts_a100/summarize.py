#!/usr/bin/env python
"""M — 汇总 outputs/metrics/geometry_m_*.json + provenance_m_*.json
输出 outputs/metrics/summary_m.csv 并打印 markdown 表（不依赖渲染指标）。
  python scripts/m/summarize.py [--metrics_dir ...] [--out ...]
"""
import argparse, csv, glob, json, os, re

PROJ = os.environ.get("PROJ_ROOT", "/scratch/users/nus/e1351071/test_zju")

COLS = ["TAG", "D_fg", "D_bg", "quantile", "seed", "prune_ratio", "n_components",
        "n_fragments_lt100", "largest_component_ratio", "n_boundary_edges",
        "G1_median_abs", "G2_ratio_below_1pct", "G5_dihedral_abs_mean",
        "n_faces", "extract_wallclock_s"]


def g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d if d is not None else default


def first(d, keys, default=None):
    """在 dict 里按候选 key 列表取第一个存在的值（兼容另一代理新加字段的命名）。"""
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def from_cmd(cmd, flag):
    if not cmd:
        return None
    m = re.search(re.escape(flag) + r"[= ]+([^\s]+)", cmd)
    return m.group(1) if m else None


def row_for(tag, metrics_dir):
    geo = json.load(open(os.path.join(metrics_dir, "geometry_m_%s.json" % tag)))
    pp = os.path.join(metrics_dir, "provenance_m_%s.json" % tag)
    prov = json.load(open(pp)) if os.path.exists(pp) else {}
    es = prov.get("extract_stats") or {}
    ps = prov.get("prune_stats") or {}
    cmd = first(es, ["command", "cmd", "argv"])
    if isinstance(cmd, list):
        cmd = " ".join(cmd)
    top = g(geo, "G4_topology", default={})
    return {
        "TAG": tag,
        "D_fg": first(es, ["poisson_depth_fg_used", "poisson_depth_used", "poisson_depth_fg",
                           "poisson_depth_arg", "poisson_depth"], from_cmd(cmd, "--poisson_depth")),
        "D_bg": first(es, ["poisson_depth_bg_used", "poisson_depth_bg", "poisson_depth_bg_arg"], from_cmd(cmd, "--poisson_depth_bg")),
        "quantile": first(es, ["vertices_density_quantile"], from_cmd(cmd, "--vertices_density_quantile")),
        "seed": first(es, ["extract_seed", "seed"], from_cmd(cmd, "--extract_seed")),
        "prune_ratio": first(ps, ["prune_ratio", "pruned_ratio", "ratio_pruned", "keep_ratio"]),
        "n_components": top.get("n_connected_components"),
        "n_fragments_lt100": top.get("n_fragment_components_lt_100_faces"),
        "largest_component_ratio": top.get("largest_component_face_ratio"),
        "n_boundary_edges": top.get("n_boundary_edges"),
        "G1_median_abs": g(geo, "G1_sparse_to_mesh_distance", "median_abs"),
        "G2_ratio_below_1pct": g(geo, "G2_precision_ratio", "ratio_below_1pct"),
        "G5_dihedral_abs_mean": g(geo, "G5_normal_smoothness", "dihedral_abs_deg_mean"),
        "n_faces": top.get("n_faces"),
        "extract_wallclock_s": first(es, ["extraction_wall_clock_s", "extract_wallclock_s",
                                          "wall_clock_s"], prov.get("extract_wallclock_s_shell")),
    }


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return ("%.6g" % v) if abs(v) < 1 else ("%.4g" % v)
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default=os.path.join(PROJ, "outputs", "metrics"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.metrics_dir, "summary_m.csv")
    tags = sorted(os.path.basename(p)[len("geometry_m_"):-len(".json")]
                  for p in glob.glob(os.path.join(a.metrics_dir, "geometry_m_*.json")))
    rows = []
    for t in tags:
        try:
            rows.append(row_for(t, a.metrics_dir))
        except Exception as e:
            print("[warn] %s: %s" % (t, e))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("| " + " | ".join(COLS) + " |")
    print("|" + "|".join(["---"] * len(COLS)) + "|")
    for r in rows:
        print("| " + " | ".join(fmt(r[c]) for c in COLS) + " |")
    print("\ncsv -> %s  (%d rows)" % (out, len(rows)))


if __name__ == "__main__":
    main()
