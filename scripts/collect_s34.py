"""S3.4 结果收集：三组 train_stats.json + dnc_log.csv 走势 + mesh .ply 独立核验。

用法:  python scripts/collect_s34.py
输出:  stdout 的 JSON + outputs/metrics/s34_collect.json
只读，不修改任何 run 产物。
"""
import os, sys, json, glob, csv
import numpy as np

PROJ = "/scratch/e1351071/zju_test"
RUNS = [
    ("coarse_base_seed0", "mesh_base_seed0", 0.0),
    ("coarse_dnc005",     "mesh_dnc005",     0.05),
    ("coarse_dnc02",      "mesh_dnc02",      0.2),
]
STAT_KEYS = [
    "gpu_index", "seed", "dnc_factor", "dnc_start", "dnc_enabled",
    "iteration_to_load", "num_iterations", "first_iteration", "last_iteration",
    "n_iterations_done", "sdf_estimation_factor", "sdf_better_normal_factor",
    "train_wallclock_s", "train_wallclock_min", "total_wallclock_s", "total_wallclock_min",
    "mean_time_per_iteration_ms", "max_memory_allocated_MiB", "max_memory_reserved_MiB",
    "n_gaussians_final", "image_height", "image_width", "n_training_cameras",
    "eval_split", "final_loss", "last_l_dnc", "last_dnc_valid_pixel_ratio",
    "gs_checkpoint_path", "scene_path", "final_model_path", "timestamp",
]


def read_dnc_log(path):
    """L_dnc 首/中/末 + sdf_better_normal_loss 走势（H3）。"""
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return None

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r[name]))
            except (KeyError, TypeError, ValueError):
                out.append(float("nan"))
        return np.array(out, dtype=float)

    it = col("iteration")
    ld = col("l_dnc")
    sn = col("sdf_better_normal_loss")
    se = col("sdf_estimation_loss")
    vr = col("valid_pixel_ratio")
    n = len(rows)
    mid = n // 2

    def trio(a):
        return {"first": float(a[0]), "mid": float(a[mid]), "last": float(a[-1])}

    def fin(a):
        m = np.isfinite(a)
        return a[m], m

    ld_f, ld_m = fin(ld)
    sn_f, sn_m = fin(sn)

    # 线性拟合斜率（每 1000 iter），用于判断"下降/上升"
    def slope_per_1k(x, y):
        m = np.isfinite(y)
        if m.sum() < 3:
            return None
        p = np.polyfit(x[m], y[m], 1)
        return float(p[0] * 1000.)

    # 首末各 10% 的均值，抗噪声
    k = max(1, n // 10)
    return {
        "csv_path": path,
        "n_rows": n,
        "iteration_first": float(it[0]), "iteration_last": float(it[-1]),
        "l_dnc": trio(ld),
        "l_dnc_mean_first10pct": float(np.nanmean(ld[:k])),
        "l_dnc_mean_last10pct": float(np.nanmean(ld[-k:])),
        "l_dnc_min": float(np.nanmin(ld)), "l_dnc_max": float(np.nanmax(ld)),
        "l_dnc_all_finite": bool(np.isfinite(ld).all()),
        "l_dnc_slope_per_1k_iter": slope_per_1k(it, ld),
        "sdf_better_normal_loss": trio(sn),
        "sdf_better_normal_mean_first10pct": float(np.nanmean(sn[:k])),
        "sdf_better_normal_mean_last10pct": float(np.nanmean(sn[-k:])),
        "sdf_better_normal_n_finite": int(np.isfinite(sn).sum()),
        "sdf_better_normal_slope_per_1k_iter": slope_per_1k(it, sn),
        "sdf_estimation_loss": trio(se),
        "sdf_estimation_slope_per_1k_iter": slope_per_1k(it, se),
        "valid_pixel_ratio": trio(vr),
        "valid_pixel_ratio_mean": float(np.nanmean(vr)),
        "any_nan_in_l_dnc": bool(np.isnan(ld).any()),
    }


def check_mesh(path):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(path)
    V = np.asarray(m.vertices); F = np.asarray(m.triangles)
    info = {
        "path": path,
        "file_bytes": os.path.getsize(path),
        "n_vertices": int(V.shape[0]),
        "n_triangles": int(F.shape[0]),
        "vertices_finite": bool(np.isfinite(V).all()),
        "any_nan_vertices": bool(np.isnan(V).any()),
        "any_nan_or_inf_vertices": bool((~np.isfinite(V)).any()),
        "has_vertex_colors": bool(m.has_vertex_colors()),
        "has_vertex_normals": bool(m.has_vertex_normals()),
        "edge_manifold": bool(m.is_edge_manifold()),
        "vertex_manifold": bool(m.is_vertex_manifold()),
        "bbox_min": V.min(0).tolist() if V.size else None,
        "bbox_max": V.max(0).tolist() if V.size else None,
    }
    if m.has_vertex_colors():
        C = np.asarray(m.vertex_colors)
        info["any_nan_vertex_colors"] = bool((~np.isfinite(C)).any())
    if F.size:
        labels, cluster_n_tri, _ = m.cluster_connected_triangles()
        cn = np.asarray(cluster_n_tri)
        info["n_connected_components"] = int(cn.shape[0])
        info["largest_component_tri"] = int(cn.max())
        info["largest_component_frac"] = float(cn.max() / cn.sum())
        info["n_components_lt100_tri"] = int((cn < 100).sum())
    return info


def main():
    out = {"runs": {}}
    for run, mesh_dir, lam in RUNS:
        rec = {"run": run, "lambda": lam}
        od = os.path.join(PROJ, "outputs", "runs", run)

        sp = os.path.join(od, "train_stats.json")
        if os.path.isfile(sp):
            with open(sp) as f:
                st = json.load(f)
            rec["train_stats"] = {k: st.get(k) for k in STAT_KEYS}
            rec["train_stats_path"] = sp
        else:
            rec["train_stats"] = None
            rec["train_stats_path"] = None

        pts = sorted(glob.glob(os.path.join(od, "sugarcoarse_*", "15000.pt")))
        rec["pt_path"] = pts[0] if pts else None
        rec["pt_bytes"] = os.path.getsize(pts[0]) if pts else None

        rec["dnc_log"] = read_dnc_log(os.path.join(od, "dnc_log.csv"))
        vis = sorted(glob.glob(os.path.join(od, "dnc_vis", "*.png")))
        rec["n_dnc_vis_png"] = len(vis)

        plys = sorted(glob.glob(os.path.join(PROJ, "outputs", "runs", mesh_dir, "*.ply")))
        rec["ply_path"] = plys[0] if plys else None
        rec["mesh"] = check_mesh(plys[0]) if plys else None

        out["runs"][run] = rec

    # 三组一致性（C4）
    sts = [out["runs"][r]["train_stats"] for r, _, _ in RUNS]
    if all(sts):
        out["consistency_C4"] = {
            "same_gs_checkpoint": len({s["gs_checkpoint_path"] for s in sts}) == 1,
            "gs_checkpoint": sts[0]["gs_checkpoint_path"],
            "same_scene": len({s["scene_path"] for s in sts}) == 1,
            "scene": sts[0]["scene_path"],
            "same_seed": len({s["seed"] for s in sts}) == 1,
            "seed": sts[0]["seed"],
            "same_iteration_to_load": len({s["iteration_to_load"] for s in sts}) == 1,
            "same_num_iterations": len({s["num_iterations"] for s in sts}) == 1,
            "same_n_iterations_done": len({s["n_iterations_done"] for s in sts}) == 1,
            "n_iterations_done": sts[0]["n_iterations_done"],
            "same_sdf_factors": len({(s["sdf_estimation_factor"], s["sdf_better_normal_factor"]) for s in sts}) == 1,
            "same_eval_split": len({s["eval_split"] for s in sts}) == 1,
            "dnc_factors": [s["dnc_factor"] for s in sts],
            "dnc_starts": [s["dnc_start"] for s in sts],
        }

    os.makedirs(os.path.join(PROJ, "outputs", "metrics"), exist_ok=True)
    dst = os.path.join(PROJ, "outputs", "metrics", "s34_collect.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[saved] {dst}", file=sys.stderr)


if __name__ == "__main__":
    main()
