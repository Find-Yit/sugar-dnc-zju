#!/usr/bin/env python
"""S3.f — 各组提取的命令 / 参数 / 耗时 / 顶点面数表（数字全部取自落盘文件）。"""
import json, os, re, glob

PROJ = "/scratch/e1351071/zju_test"
TAGS = [("(基准, 阶段3.4)", "mesh_base_seed0", "s34_mesh_base_seed0", "coarse_base_seed0"),
        ("pdauto", "mesh_base_seed0_pdauto", "s3f_mesh_pdauto", "base_seed0_pdauto"),
        ("pdauto_q0", "mesh_base_seed0_pdauto_q0", "s3f_mesh_pdauto_q0", "base_seed0_pdauto_q0"),
        ("q0", "mesh_base_seed0_q0", "s3f_mesh_q0", "base_seed0_q0"),
        ("q005", "mesh_base_seed0_q005", "s3f_mesh_q005", "base_seed0_q005"),
        ("pd9", "mesh_base_seed0_pd9", "s3f_mesh_pd9", "base_seed0_pd9"),
        ("pd8", "mesh_base_seed0_pd8", "s3f_mesh_pd8", "base_seed0_pd8")]

print("| 组 | --poisson_depth | --vertices_density_quantile | 实际 D | 提取墙钟(s) | 顶点数 | 面数 | mesh 路径 |")
print("|---|---|---|---|---|---|---|---|")
for label, d, logname, run in TAGS:
    md = os.path.join(PROJ, "outputs", "runs", d)
    st = os.path.join(md, "extract_stats.json")
    pd_arg = vdq = used = "—"
    wall = "—"
    if os.path.exists(st):
        s = json.load(open(st))
        pd_arg, vdq, used = s.get("poisson_depth_arg", "—"), s.get("vertices_density_quantile", "—"), s.get("poisson_depth_used", "—")
        if s.get("extraction_wall_clock_s"):
            wall = f"{s['extraction_wall_clock_s']:.0f}"
    elif "基准" in label:
        pd_arg, vdq, used = "10 (原版硬编码)", "0.1 (原版硬编码)", "10"
    else:
        pd_arg = vdq = used = "（未完成）"
    log = os.path.join(PROJ, "logs", logname + ".log")
    if wall == "—" and os.path.exists(log):
        m = re.search(r"MESH END.*wallclock=(\d+)s", open(log, errors="ignore").read())
        if m:
            wall = m.group(1)
    gj = os.path.join(PROJ, "outputs", "metrics", f"geometry_{run}.json")
    nv = nf = "—"
    if os.path.exists(gj):
        t = json.load(open(gj))["G4_topology"]
        nv, nf = t["n_vertices"], t["n_faces"]
    plys = glob.glob(os.path.join(md, "*.ply"))
    rel = os.path.relpath(plys[0], PROJ) if plys else "（缺）"
    print(f"| {label} | {pd_arg} | {vdq} | {used} | {wall} | {nv} | {nf} | `{rel}` |")
