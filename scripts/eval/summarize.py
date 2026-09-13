#!/usr/bin/env python
"""S4.4 汇总表（计划 §2b / 阶段 4 S4.4）。

扫描 outputs/metrics/ 下的 render_<run>.json、geometry_<run>.json、meshvis_<run>.json，
以及各 run 目录下的 train_stats.json（逻辑字段：wall_clock_min / ms_per_iter / peak_mem_gb /
n_gaussians；不同执行者写的键名不同，由 TRAIN_ALIASES 归一，例如
train_wallclock_min / mean_time_per_iteration_ms / max_memory_allocated_MiB / n_gaussians_final；
找不到就填 NA），汇总成 outputs/metrics/summary.csv。

表头一律带单位与方向后缀：_up 表示越大越好，_down 表示越小越好。
报告里的所有数字只能来自本文件与各 metrics JSON（计划 §2b 末尾的诚信条款）。
"""
import argparse
import csv
import glob
import json
import os

PROJ_ROOT = os.environ.get("PROJ_ROOT", "/scratch/e1351071/zju_test")
NA = "NA"

# (列名, 取值函数)。列名后缀 _up / _down 标明方向。
COLUMNS = [
    ("run", lambda r: r["run"]),
    ("n_gaussians", lambda r: r.get("n_gaussians", NA)),
    # ---- 渲染（32 个测试视角）----
    ("PSNR_dB_up", lambda r: fmt(r["render"].get("psnr_db_mean"), 4)),
    ("PSNR_dB_std", lambda r: fmt(r["render"].get("psnr_db_std"), 4)),
    ("SSIM_up", lambda r: fmt(r["render"].get("ssim_mean"), 5)),
    ("LPIPS_VGG_down", lambda r: fmt(r["render"].get("lpips_vgg_mean"), 5)),
    ("n_test_views", lambda r: r["render"].get("n_test_views", NA)),
    # ---- 几何 G1 / G2 ----
    ("G1_mean_abs_down", lambda r: fmt(g(r, "G1_sparse_to_mesh_distance", "mean_abs"), 6)),
    ("G1_median_abs_down", lambda r: fmt(g(r, "G1_sparse_to_mesh_distance", "median_abs"), 6)),
    ("G1_p90_abs_down", lambda r: fmt(g(r, "G1_sparse_to_mesh_distance", "p90_abs"), 6)),
    ("G1_mean_rel_pct_down", lambda r: fmt(pct(g(r, "G1_sparse_to_mesh_distance", "mean_rel")), 4)),
    ("G1_median_rel_pct_down", lambda r: fmt(pct(g(r, "G1_sparse_to_mesh_distance", "median_rel")), 4)),
    ("G1_p90_rel_pct_down", lambda r: fmt(pct(g(r, "G1_sparse_to_mesh_distance", "p90_rel")), 4)),
    ("G1_n_reference_points", lambda r: gi(r, "G1_sparse_to_mesh_distance", "n_reference_points")),
    ("G2_ratio_lt_0p5pct_up", lambda r: fmt(pct(g(r, "G2_precision_ratio", "ratio_below_0p5pct")), 3)),
    ("G2_ratio_lt_1pct_up", lambda r: fmt(pct(g(r, "G2_precision_ratio", "ratio_below_1pct")), 3)),
    # ---- 几何 G3 ----
    ("G3_mean_abs_all_verts_down", lambda r: fmt(g(r, "G3_mesh_vertex_to_sparse_distance", "mean_abs_all_vertices"), 6)),
    ("G3_mean_abs_fg_verts_down", lambda r: fmt(g(r, "G3_mesh_vertex_to_sparse_distance", "mean_abs_fg_vertices"), 6)),
    # ---- 几何 G4 拓扑 ----
    ("G4_n_vertices", lambda r: gi(r, "G4_topology", "n_vertices")),
    ("G4_n_faces", lambda r: gi(r, "G4_topology", "n_faces")),
    ("G4_n_components_down", lambda r: gi(r, "G4_topology", "n_connected_components")),
    ("G4_largest_comp_face_ratio_pct_up", lambda r: fmt(pct(g(r, "G4_topology", "largest_component_face_ratio")), 3)),
    ("G4_n_fragments_lt100faces_down", lambda r: frag(r)),
    ("G4_fragment_faces_total_down", lambda r: gi(r, "G4_topology", "fragment_faces_total")),
    ("G4_n_non_manifold_edges_down", lambda r: gi(r, "G4_topology", "n_non_manifold_edges")),
    ("G4_n_boundary_edges_down", lambda r: gi(r, "G4_topology", "n_boundary_edges")),
    # ---- 几何 G5 ----
    ("G5_dihedral_deg_mean_down", lambda r: fmt(g(r, "G5_normal_smoothness", "dihedral_deg_mean"), 4)),
    ("G5_dihedral_deg_p90_down", lambda r: fmt(g(r, "G5_normal_smoothness", "dihedral_deg_p90"), 4)),
    ("G5_dihedral_abs_deg_mean_down", lambda r: fmt(g(r, "G5_normal_smoothness", "dihedral_abs_deg_mean"), 4)),
    ("G5_dihedral_abs_deg_p90_down", lambda r: fmt(g(r, "G5_normal_smoothness", "dihedral_abs_deg_p90"), 4)),
    ("G5_frac_raw_gt_90deg_pct", lambda r: fmt(pct(g(r, "G5_normal_smoothness", "frac_raw_gt_90deg")), 3)),
    # ---- 训练代价 ----
    ("train_wall_clock_min", lambda r: fmt(t(r, "wall_clock_min"), 2)),
    ("train_ms_per_iter", lambda r: fmt(t(r, "ms_per_iter"), 3)),
    ("train_peak_mem_gb", lambda r: fmt(t(r, "peak_mem_gb"), 3)),
    ("train_n_gaussians_final", lambda r: t(r, "n_gaussians") if t(r, "n_gaussians") is not None else NA),
    ("train_end_iteration", lambda r: t(r, "end_iteration") if t(r, "end_iteration") is not None else NA),
    ("dnc_factor", lambda r: r["train"].get("dnc_factor", NA)),
    # ---- 溯源 ----
    ("cameras_extent", lambda r: fmt(r["geometry"].get("cameras_extent"), 6)),
    ("render_json", lambda r: r.get("render_src", NA)),
    ("geometry_json", lambda r: r.get("geometry_src", NA)),
    ("train_stats_json", lambda r: r.get("train_src", NA)),
    # [ADDED] 仅做「提取端」消融的 run（同一个 coarse 模型、只改 Poisson 参数）在这里标出
    # 它复用的是哪一组 coarse 训练结果；其渲染指标 / 训练代价直接继承自该 run（高斯完全相同）。
    ("source_coarse", lambda r: r.get("source_coarse", NA)),
]


def fmt(v, nd):
    if v is None or v == "":
        return NA
    try:
        return format(float(v), f".{nd}f")
    except (TypeError, ValueError):
        return NA


def pct(v):
    return None if v is None else float(v) * 100.0


def g(row, section, key):
    return (row.get("geometry", {}).get(section) or {}).get(key)


def gi(row, section, key):
    """整数类指标：只有真的缺失才返回 NA。
    注意不能写成 `g(...) or NA` —— 0 是 falsy，会把「0 条非流形边」这种好结果误显示成 NA。"""
    v = g(row, section, key)
    return NA if v is None else v


def frag(row):
    """碎片分量数。键名里带阈值（默认 n_fragment_components_lt_100_faces），这里按前缀找。"""
    topo = row.get("geometry", {}).get("G4_topology") or {}
    for k, v in topo.items():
        if k.startswith("n_fragment_components_lt_"):
            return v
    return NA


# train_stats.json 的字段名各执行者写法不一，这里做别名归一。
# 值为 (候选键名, 换算函数) 的列表，按顺序取第一个存在的。
TRAIN_ALIASES = {
    "wall_clock_min": [("wall_clock_min", None), ("train_wallclock_min", None),
                       ("total_wallclock_min", None),
                       ("train_wallclock_s", lambda v: v / 60.0),
                       ("wall_clock_sec", lambda v: v / 60.0)],
    "ms_per_iter": [("ms_per_iter", None), ("mean_time_per_iteration_ms", None),
                    ("ms_per_iteration", None)],
    "peak_mem_gb": [("peak_mem_gb", None),
                    ("max_memory_allocated_MiB", lambda v: v / 1024.0),
                    ("max_memory_allocated_bytes", lambda v: v / (1024.0 ** 3)),
                    ("peak_memory_gb", None)],
    "n_gaussians": [("n_gaussians", None), ("n_gaussians_final", None)],
    # coarse 训练从 3DGS 的 7000 iter 续训到 15000，这里记的是「结束时的迭代号」（如 15000），
    # 不是本次实际跑的步数（那是 n_iterations_done = 15000 - 7000）。
    "end_iteration": [("last_iteration", None), ("num_iterations", None), ("end_iteration", None)],
}


def t(row, logical_key):
    """从 train_stats.json 里按别名表取值，取不到返回 None。"""
    stats = row.get("train", {})
    for key, conv in TRAIN_ALIASES.get(logical_key, [(logical_key, None)]):
        if key in stats and stats[key] is not None:
            v = stats[key]
            try:
                return conv(v) if conv else v
            except (TypeError, ValueError):
                return None
    return None


def normalize(name, enabled=True):
    if not enabled:
        return name
    for prefix in ("coarse_", "mesh_", "meshvis_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def main():
    p = argparse.ArgumentParser(description="汇总所有 run 的渲染 / 几何 / 训练代价指标")
    p.add_argument("--metrics_dir", type=str, default=os.path.join(PROJ_ROOT, "outputs", "metrics"))
    p.add_argument("--runs_dir", type=str, default=os.path.join(PROJ_ROOT, "outputs", "runs"))
    p.add_argument("--out_csv", type=str, default=os.path.join(PROJ_ROOT, "outputs", "metrics", "summary.csv"))
    p.add_argument("--no_normalize", action="store_true",
                   help="不把 coarse_/mesh_ 前缀归并成同一个 run")
    p.add_argument("--run_order", type=str, default="vanilla3dgs7k,baseline,dnc005,dnc02",
                   help="输出行顺序（逗号分隔）；未列出的 run 追加在后面并按字母序")
    args = p.parse_args()
    norm = not args.no_normalize

    rows = {}

    def slot(run):
        return rows.setdefault(run, {"run": run, "render": {}, "geometry": {}, "train": {}})

    for path in sorted(glob.glob(os.path.join(args.metrics_dir, "render_*.json"))):
        name = normalize(os.path.basename(path)[len("render_"):-len(".json")], norm)
        with open(path) as f:
            data = json.load(f)
        r = slot(name)
        r["render"] = data
        r["render_src"] = os.path.relpath(path, PROJ_ROOT)
        if data.get("n_gaussians") is not None:
            r["n_gaussians"] = data["n_gaussians"]

    for path in sorted(glob.glob(os.path.join(args.metrics_dir, "geometry_*.json"))):
        name = normalize(os.path.basename(path)[len("geometry_"):-len(".json")], norm)
        with open(path) as f:
            data = json.load(f)
        r = slot(name)
        r["geometry"] = data
        r["geometry_src"] = os.path.relpath(path, PROJ_ROOT)

    # train_stats.json：run 目录下任意深度，或 metrics 下 train_stats_<run>.json
    train_candidates = sorted(glob.glob(os.path.join(args.runs_dir, "**", "train_stats.json"), recursive=True))
    for path in train_candidates:
        rel = os.path.relpath(path, args.runs_dir)
        name = normalize(rel.split(os.sep)[0], norm)
        with open(path) as f:
            data = json.load(f)
        r = slot(name)
        r["train"] = data
        r["train_src"] = os.path.relpath(path, PROJ_ROOT)
    for path in sorted(glob.glob(os.path.join(args.metrics_dir, "train_stats_*.json"))):
        name = normalize(os.path.basename(path)[len("train_stats_"):-len(".json")], norm)
        with open(path) as f:
            data = json.load(f)
        r = slot(name)
        r["train"] = data
        r["train_src"] = os.path.relpath(path, PROJ_ROOT)

    # [ADDED] provenance_<run>.json = {"source_coarse": "<coarse run 名>", ...}
    # 用于「只做 mesh 提取端消融」的 run：它们没有自己的 coarse 训练，渲染指标与训练代价
    # 与源 coarse run 完全相同（同一批高斯），这里直接继承并在 source_coarse 列标注来源。
    for path in sorted(glob.glob(os.path.join(args.metrics_dir, "provenance_*.json"))):
        name = normalize(os.path.basename(path)[len("provenance_"):-len(".json")], norm)
        with open(path) as f:
            data = json.load(f)
        src = data.get("source_coarse")
        if not src:
            continue
        r = slot(name)
        r["source_coarse"] = src
        src_key = normalize(src, norm)
        parent = rows.get(src_key)
        if parent is None:
            continue
        if not r["render"] and parent.get("render"):
            r["render"] = parent["render"]
            r["render_src"] = parent.get("render_src", NA) + " (继承自 " + src + ")"
            if parent.get("n_gaussians") is not None:
                r["n_gaussians"] = parent["n_gaussians"]
        if not r["train"] and parent.get("train"):
            r["train"] = parent["train"]
            r["train_src"] = parent.get("train_src", NA) + " (继承自 " + src + ")"

    if not rows:
        raise SystemExit(f"[ERROR] 在 {args.metrics_dir} 下没有找到任何 render_*.json / geometry_*.json")

    preferred = [x for x in args.run_order.split(",") if x]
    ordered = [x for x in preferred if x in rows] + sorted(k for k in rows if k not in preferred)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in COLUMNS])
        for name in ordered:
            r = rows[name]
            w.writerow([c[1](r) for c in COLUMNS])

    print(f"[OK] {len(ordered)} 个 run 写入 {args.out_csv}")
    print("     列名后缀 _up = 越大越好，_down = 越小越好；缺失填 NA")
    with open(args.out_csv) as f:
        for line in f:
            cells = line.rstrip("\n").split(",")
            print("  " + " | ".join(cells[:8]))
    missing = []
    for name in ordered:
        r = rows[name]
        if not r["render"]:
            missing.append(f"{name}: 缺 render_*.json")
        if not r["geometry"]:
            missing.append(f"{name}: 缺 geometry_*.json")
        if not r["train"]:
            missing.append(f"{name}: 缺 train_stats.json")
    inherited = [n for n in ordered if rows[n].get("source_coarse")]
    if inherited:
        print("\n[INFO] 以下 run 只做了 mesh 提取端消融，渲染指标 / 训练代价继承自 source_coarse 列所指的 run：")
        for n in inherited:
            print(f"  - {n} <- {rows[n]['source_coarse']}")
    if missing:
        print("\n[WARN] 缺失项（对应单元格为 NA）：")
        for m in missing:
            print("  -", m)


if __name__ == "__main__":
    main()
