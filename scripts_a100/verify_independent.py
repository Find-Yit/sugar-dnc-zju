#!/usr/bin/env python
"""独立复算 §5 验收标准（不复用 scripts/eval/eval_geometry.py 的代码路径）。

用法: OMP_NUM_THREADS=1 python scripts/m/verify_independent.py base_d10_q01 m1a_q0 m2p95_q0

复算项：
  (a) 连通分量数 / <100 面碎片分量数 —— 自写 union-find（按边共享做面连通），不用 open3d
  (b) 边界边数 —— 自写 numpy 边表（只被 1 个面使用的无向边）
  (c) G1 中位数 —— scipy cKDTree：COLMAP 稀疏点到 mesh 的近似距离
      口径 A: 只到顶点（纯上界近似）
      口径 B: 到 顶点+边中点+面重心 的增稠点集（更接近点到面片距离，仍是上界）
      与 json 的 RaycastingScene 精确点到三角面距离对比，差异应为「上界 >= 精确值」的正偏。
  相机 extent / 前景 bbox 自己从 3DGS cameras.json 重算（train split i%8!=0，radius=1.1*max||c-mean c||），
  不经 SuGaR 的 CamerasWrapper。
  (d) extract_stats.json 的 poisson_depth_fg/bg
  (e) outputs/m2/pruned_p95/15000_pruned_prune_stats.json 的 pruned_ratio
"""
import glob
import json
import os
import struct
import sys

import numpy as np
from scipy.spatial import cKDTree

PROJ = os.environ.get("PROJ_ROOT", "/scratch/users/nus/e1351071/test_zju")
SCENE = os.path.join(PROJ, "data", "tandt", "truck")
GS_CKPT = os.path.join(PROJ, "outputs", "baseline", "gs_truck")
OUT_JSON = os.path.join(PROJ, "outputs", "metrics", "verify_independent.json")
EVAL_INTERVAL, MIN_TRACK, FG_FACTOR, FRAG_TH = 8, 3, 1.0, 100


# ---------------------------------------------------------------- 自写 PLY 读取（不经 open3d）
def read_ply(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        fmt = f.readline().strip().decode()
        assert "binary_little_endian" in fmt, fmt
        elems, cur = [], None
        while True:
            line = f.readline().decode().strip()
            if line == "end_header":
                break
            t = line.split()
            if t[0] == "element":
                cur = {"name": t[1], "count": int(t[2]), "props": []}
                elems.append(cur)
            elif t[0] == "property":
                cur["props"].append(t[1:])
        maps = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
                "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
                "short": "i2", "ushort": "u2", "int": "i4", "int32": "i4",
                "uint": "u4", "uint32": "u4"}
        verts = faces = None
        for e in elems:
            if all(p[0] != "list" for p in e["props"]):
                dt = np.dtype([(p[-1], "<" + maps[p[0]]) for p in e["props"]])
                arr = np.frombuffer(f.read(dt.itemsize * e["count"]), dtype=dt, count=e["count"])
                if e["name"] == "vertex":
                    verts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
            else:
                p = [q for q in e["props"] if q[0] == "list"][0]
                cnt_dt, idx_dt = np.dtype("<" + maps[p[1]]), np.dtype("<" + maps[p[2]])
                # 三角网格：假定全部为 3 顶点面（读完后校验）
                rec = np.dtype([("n", cnt_dt), ("v", idx_dt, 3)])
                arr = np.frombuffer(f.read(rec.itemsize * e["count"]), dtype=rec, count=e["count"])
                assert (arr["n"] == 3).all(), "存在非三角面，本脚本只支持三角网格"
                faces = arr["v"].astype(np.int64)
    return verts, faces


# ---------------------------------------------------------------- (b) 边表
def edge_table(faces):
    F = faces.shape[0]
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    fidx = np.tile(np.arange(F, dtype=np.int64), 3)
    uniq, inv, counts = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    return uniq, inv.reshape(-1), counts, fidx


# ---------------------------------------------------------------- (a) union-find
class DSU:
    def __init__(self, n):
        self.p = np.arange(n, dtype=np.int64)

    def find(self, x):
        p = self.p
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:      # 路径压缩
            p[x], x = root, p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def connected_components_faces(faces, inv, fidx):
    """面级连通：共享一条无向边的两个面视为连通（与 open3d cluster_connected_triangles 同定义）。"""
    F = faces.shape[0]
    order = np.argsort(inv, kind="stable")
    inv_s, f_s = inv[order], fidx[order]
    starts = np.searchsorted(inv_s, np.arange(inv_s[-1] + 1), side="left")
    ends = np.searchsorted(inv_s, np.arange(inv_s[-1] + 1), side="right")
    dsu = DSU(F)
    for s, t in zip(starts.tolist(), ends.tolist()):
        if t - s >= 2:
            base = f_s[s]
            for k in range(s + 1, t):
                dsu.union(base, f_s[k])
    roots = np.array([dsu.find(i) for i in range(F)], dtype=np.int64)
    _, sizes = np.unique(roots, return_counts=True)
    return sizes


# ---------------------------------------------------------------- 相机 extent（自算）
def cameras_extent():
    cams = json.load(open(os.path.join(GS_CKPT, "cameras.json")))
    cams = sorted(cams, key=lambda c: c["img_name"])
    train = [c for i, c in enumerate(cams) if i % EVAL_INTERVAL != 0]
    C = np.array([c["position"] for c in train], dtype=np.float64)
    mean = C.mean(axis=0)
    return 1.1 * float(np.linalg.norm(C - mean, axis=1).max()), mean, len(cams), len(train)


# ---------------------------------------------------------------- COLMAP 稀疏点（自读）
def read_points3D(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        xyz = np.empty((n, 3), np.float64)
        tl = np.empty(n, np.int64)
        for i in range(n):
            p = struct.unpack("<QdddBBBd", f.read(43))
            xyz[i] = p[1:4]
            t = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * t)
            tl[i] = t
    return xyz, tl


def point_triangle_dist(P, A, B, C):
    """精确点到三角形距离（逐点 vs 其候选面集，向量化 Ericson 区域判别）。"""
    AB, AC, AP = B - A, C - A, P - A
    d1 = np.einsum("ij,ij->i", AB, AP)
    d2 = np.einsum("ij,ij->i", AC, AP)
    BP = P - B
    d3 = np.einsum("ij,ij->i", AB, BP)
    d4 = np.einsum("ij,ij->i", AC, BP)
    CP = P - C
    d5 = np.einsum("ij,ij->i", AB, CP)
    d6 = np.einsum("ij,ij->i", AC, CP)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = 1.0 / np.maximum(va + vb + vc, 1e-30)
    v = np.clip(vb * denom, 0, 1)
    w = np.clip(vc * denom, 0, 1)
    Q = A + v[:, None] * AB + w[:, None] * AC          # 内部区域
    # 顶点区域
    Q = np.where(((d1 <= 0) & (d2 <= 0))[:, None], A, Q)
    Q = np.where(((d3 >= 0) & (d4 <= d3))[:, None], B, Q)
    Q = np.where(((d6 >= 0) & (d5 <= d6))[:, None], C, Q)
    # 边区域
    t = np.clip(d1 / np.where(d1 - d3 == 0, 1e-30, d1 - d3), 0, 1)
    Q = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[:, None], A + t[:, None] * AB, Q)
    t = np.clip(d2 / np.where(d2 - d6 == 0, 1e-30, d2 - d6), 0, 1)
    Q = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[:, None], A + t[:, None] * AC, Q)
    t = np.clip((d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1e-30, (d4 - d3) + (d5 - d6)), 0, 1)
    Q = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[:, None], B + t[:, None] * (C - B), Q)
    return np.linalg.norm(P - Q, axis=1)


def exact_dist_subsample(V, F, ref, n_sample=3000, k=96, seed=0):
    """对随机子样本做精确点到面距离：用面重心 KDTree 取 k 个候选面再精确求距。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(ref.shape[0], size=min(n_sample, ref.shape[0]), replace=False)
    P = ref[idx]
    cents = V[F].mean(axis=1)
    _, cand = cKDTree(cents).query(P, k=min(k, F.shape[0]), workers=1)
    n, kk = cand.shape
    Pr = np.repeat(P, kk, axis=0)
    tri = F[cand.reshape(-1)]
    d = point_triangle_dist(Pr, V[tri[:, 0]], V[tri[:, 1]], V[tri[:, 2]])
    return d.reshape(n, kk).min(axis=1), idx


def find_ply(tag):
    g = sorted(glob.glob(os.path.join(PROJ, "outputs", "m", "mesh_" + tag, "*.ply")))
    assert len(g) == 1, (tag, g)
    return g[0]


def rel(a, b):
    return None if b in (0, None) or a is None else float((a - b) / b)


def main():
    tags = sys.argv[1:] or ["base_d10_q01", "m1a_q0", "m2p95_q0"]
    extent, avg_cam, n_all_cam, n_train = cameras_extent()
    fg_min, fg_max = avg_cam - FG_FACTOR * extent, avg_cam + FG_FACTOR * extent
    xyz, tl = read_points3D(os.path.join(SCENE, "sparse", "0", "points3D.bin"))
    keep = (tl >= MIN_TRACK) & np.all(xyz > fg_min, axis=1) & np.all(xyz < fg_max, axis=1)
    ref = xyz[keep]
    print(f"[extent] {extent:.9f} center={avg_cam.tolist()} cams={n_all_cam}/train={n_train}")
    print(f"[colmap] total={len(xyz)} track_ok={int((tl>=MIN_TRACK).sum())} kept={int(keep.sum())}")

    out = {"recomputed_cameras_extent": extent, "recomputed_camera_center": avg_cam.tolist(),
           "n_cameras_total": n_all_cam, "n_cameras_train": n_train,
           "n_colmap_total": int(len(xyz)), "n_colmap_track_ok": int((tl >= MIN_TRACK).sum()),
           "n_colmap_kept": int(keep.sum()), "method_notes": {
               "components": "自写 union-find（面级，共享无向边）",
               "boundary_edges": "自写 numpy 边表，counts==1",
               "G1": "scipy cKDTree 近似：A=仅顶点，B=顶点+边中点+面重心（均为点到面精确距离的上界）"},
           "runs": {}}

    for tag in tags:
        ply = find_ply(tag)
        jf = os.path.join(PROJ, "outputs", "metrics", f"geometry_m_{tag}.json")
        J = json.load(open(jf))
        V, F = read_ply(ply)
        uniq, inv, counts, fidx = edge_table(F)
        n_bnd = int((counts == 1).sum())
        sizes = connected_components_faces(F, inv, fidx)
        n_cc, n_frag = int(sizes.size), int((sizes < FRAG_TH).sum())

        dA, _ = cKDTree(V).query(ref, k=1, workers=1)
        mids = 0.5 * (V[uniq[:, 0]] + V[uniq[:, 1]])
        cents = V[F].mean(axis=1)
        dB, _ = cKDTree(np.concatenate([V, mids, cents], axis=0)).query(ref, k=1, workers=1)

        dE, sub_idx = exact_dist_subsample(V, F, ref)
        dB_sub = dB[sub_idx]
        el = np.linalg.norm(V[uniq[:, 0]] - V[uniq[:, 1]], axis=1)

        es = json.load(open(os.path.join(PROJ, "outputs", "m", "mesh_" + tag, "extract_stats.json")))
        g4, g1 = J["G4_topology"], J["G1_sparse_to_mesh_distance"]
        r = {"mesh": ply, "json": jf,
             "n_vertices": {"json": g4["n_vertices"], "mine": int(V.shape[0])},
             "n_faces": {"json": g4["n_faces"], "mine": int(F.shape[0])},
             "n_connected_components": {"json": g4["n_connected_components"], "mine": n_cc,
                                        "rel": rel(n_cc, g4["n_connected_components"])},
             "n_fragment_lt100": {"json": g4["n_fragment_components_lt_100_faces"], "mine": n_frag,
                                  "rel": rel(n_frag, g4["n_fragment_components_lt_100_faces"])},
             "n_boundary_edges": {"json": g4["n_boundary_edges"], "mine": n_bnd,
                                  "rel": rel(n_bnd, g4["n_boundary_edges"])},
             "G1_median_abs": {"json": g1["median_abs"],
                               "mine_kdtree_vertices": float(np.median(dA)),
                               "mine_kdtree_enriched": float(np.median(dB)),
                               "rel_vertices": rel(float(np.median(dA)), g1["median_abs"]),
                               "rel_enriched": rel(float(np.median(dB)), g1["median_abs"])},
             "G1_mean_abs": {"json": g1["mean_abs"], "mine_kdtree_vertices": float(dA.mean()),
                             "mine_kdtree_enriched": float(dB.mean()),
                             "rel_enriched": rel(float(dB.mean()), g1["mean_abs"])},
             "G1_median_exact_subsample": {
                 "n_sample": int(dE.size), "mine_exact": float(np.median(dE)),
                 "json_full": g1["median_abs"], "rel": rel(float(np.median(dE)), g1["median_abs"]),
                 "mine_kdtree_enriched_same_subsample": float(np.median(dB_sub)),
                 "note": "精确点到三角面距离（Ericson 区域判别 + 面重心 KDTree 取 96 候选面），"
                         "与 json 的 RaycastingScene 同口径；子样本 vs 全量导致的抽样噪声约 1%"},
             "edge_length": {"median": float(np.median(el)), "mean": float(el.mean())},
             "poisson_depth_fg_used": es.get("poisson_depth_fg_used"),
             "poisson_depth_bg_used": es.get("poisson_depth_bg_used"),
             "vertices_density_quantile": es.get("vertices_density_quantile")}
        out["runs"][tag] = r
        print(f"[{tag}] V={V.shape[0]} F={F.shape[0]} cc={n_cc}(json {g4['n_connected_components']}) "
              f"frag={n_frag}(json {g4['n_fragment_components_lt_100_faces']}) "
              f"bnd={n_bnd}(json {g4['n_boundary_edges']}) "
              f"G1med A={np.median(dA):.6f} B={np.median(dB):.6f} (json {g1['median_abs']:.6f}) "
              f"depth fg/bg={es.get('poisson_depth_fg_used')}/{es.get('poisson_depth_bg_used')}")

    ps = json.load(open(os.path.join(PROJ, "outputs", "m2", "pruned_p95", "15000_pruned_prune_stats.json")))
    out["prune_p95"] = {k: ps[k] for k in ("n_total", "n_pruned", "n_kept", "pruned_ratio")}
    out["prune_p95"]["pruned_ratio_recomputed"] = ps["n_pruned"] / ps["n_total"]
    print(f"[prune] pruned_ratio={ps['pruned_ratio']} recomputed={ps['n_pruned']/ps['n_total']}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print("[written]", OUT_JSON)


if __name__ == "__main__":
    main()
