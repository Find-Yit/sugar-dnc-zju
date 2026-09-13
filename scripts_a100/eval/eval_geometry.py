#!/usr/bin/env python
"""S4.2 几何指标评测（计划 §2b G1-G5）。

Truck 无真值网格，用 COLMAP 稀疏点云作为独立几何参考。

  G1 稀疏点 -> mesh 无符号距离（越小越好）
     参考点集 = COLMAP points3D 中 track 长度 >= 3 且落在前景 bbox 内的点。
     前景 bbox 与 sugar_extractors/coarse_mesh.py 的定义完全一致（center_bbox=True, fg_bbox_factor=1）：
         extent, avg_cam = get_cameras_spatial_extent(训练相机, return_average_xyz=True)
         bbox = [avg_cam - 1.0*extent, avg_cam + 1.0*extent]
     距离用 open3d RaycastingScene.compute_distance。报 mean / median / P90，
     同时给出绝对值与除以 extent 的相对值。
  G2 精度比例（越大越好）：G1 距离 < tau 的点占比，tau = extent 的 0.5% 与 1%。
  G3 mesh 顶点 -> 稀疏点最近距离 mean（越小越好，Chamfer 另一半，scipy cKDTree）。
     稀疏点本身稀疏，该项只作参考。同时报「全部顶点」与「仅前景 bbox 内顶点」两个版本。
  G4 拓扑统计：顶点数、面数、连通分量数、最大分量面数占比（越大越好）、
     面数<100 的碎片分量数（越小越好，即漂浮碎片）、非流形边数、边界边数。
  G5 法向平滑度：相邻面二面角 mean / P90（度，越小越平滑；过度平滑会吞细节，需结合图判断）。
     同时报 raw（面法向夹角，依赖绕序）与 abs（min(raw,180-raw)，与绕序无关）两套口径。

输出：<out_dir>/geometry_<run>.json
只读 repo/SuGaR，不修改任何现有模块。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (cameras_spatial_extent, ensure_trailing_sep,
                     read_points3D_binary_with_tracks, setup_sugar_path)

PROJ_ROOT = os.environ.get("PROJ_ROOT", "/scratch/e1351071/zju_test")


def parse_args():
    p = argparse.ArgumentParser(description="SuGaR mesh 几何指标评测（G1-G5）")
    p.add_argument("--scene_path", type=str, required=True)
    p.add_argument("--mesh_path", type=str, required=True, help="要评测的 mesh .ply")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--gs_checkpoint", type=str, default="",
                   help="3DGS 输出目录（读 cameras.json 以计算相机空间尺度）。给了 --cameras_extent 则可省略")
    p.add_argument("--cameras_extent", type=float, default=None,
                   help="直接给定相机空间尺度；不给则从 cameras.json 按 SuGaR 的定义计算")
    p.add_argument("--cameras_center", type=str, default=None,
                   help="直接给定相机平均中心 'x,y,z'；不给则从 cameras.json 计算")
    p.add_argument("--out_dir", type=str, default=os.path.join(PROJ_ROOT, "outputs", "metrics"))
    p.add_argument("--sugar_dir", type=str, default=None)
    p.add_argument("--eval_split_interval", type=int, default=8)
    p.add_argument("--min_track_length", type=int, default=3)
    p.add_argument("--fg_bbox_factor", type=float, default=1.0)
    p.add_argument("--fragment_face_threshold", type=int, default=100)
    return p.parse_args()


# ------------------------------------------------------------------ 拓扑 / 二面角

def edge_face_table(faces):
    """返回 (uniq_edges, counts, edge_faces_start, faces_by_edge)。

    faces: (F, 3) int64。无向边按顶点索引升序规范化。
    """
    n_faces = faces.shape[0]
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    face_idx = np.tile(np.arange(n_faces, dtype=np.int64), 3)
    uniq, inv, counts = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    inv_sorted = inv[order]
    faces_by_edge = face_idx[order]
    starts = np.searchsorted(inv_sorted, np.arange(uniq.shape[0]), side="left")
    return uniq, counts, starts, faces_by_edge


def face_normals(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    valid = (norm[:, 0] > 1e-20) & np.isfinite(norm[:, 0])
    n = np.where(norm > 1e-20, n / np.maximum(norm, 1e-20), 0.0)
    return n, valid


def dihedral_angles_deg(verts, faces, counts, starts, faces_by_edge):
    """相邻面（共享一条恰好被 2 个面使用的边）的法向夹角，单位度。"""
    n, valid = face_normals(verts, faces)
    sel = np.where(counts == 2)[0]
    if sel.size == 0:
        return np.zeros(0, dtype=np.float64)
    f0 = faces_by_edge[starts[sel]]
    f1 = faces_by_edge[starts[sel] + 1]
    ok = valid[f0] & valid[f1]
    f0, f1 = f0[ok], f1[ok]
    dots = np.clip(np.einsum("ij,ij->i", n[f0], n[f1]), -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def main():
    args = parse_args()
    t_start = time.time()
    setup_sugar_path(args.sugar_dir)
    import open3d as o3d
    from scipy.spatial import cKDTree

    # ---------------- 相机空间尺度 ----------------
    if args.cameras_extent is not None and args.cameras_center is not None:
        extent = float(args.cameras_extent)
        avg_cam = np.array([float(x) for x in args.cameras_center.split(",")], dtype=np.float64)
        extent_source = "cli"
    else:
        if not args.gs_checkpoint:
            raise ValueError("需要 --gs_checkpoint 或同时给出 --cameras_extent 与 --cameras_center")
        import torch
        from sugar_scene.cameras import CamerasWrapper, load_gs_cameras
        cam_list = load_gs_cameras(
            source_path=args.scene_path,
            gs_output_path=ensure_trailing_sep(args.gs_checkpoint),
            load_gt_images=False,
        )
        train_cams = [c for i, c in enumerate(cam_list) if i % args.eval_split_interval != 0]
        wrapper = CamerasWrapper(train_cams)
        extent, avg_cam_t = cameras_spatial_extent(wrapper.p3d_cameras)
        avg_cam = avg_cam_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
        extent_source = f"computed from {len(train_cams)} training cameras (SuGaR get_cameras_spatial_extent)"
        del wrapper, cam_list, train_cams
        torch.cuda.empty_cache()
    print(f"[INFO] cameras_extent = {extent:.6f}  avg_camera_center = {avg_cam.tolist()}  ({extent_source})")

    fg_min = avg_cam - args.fg_bbox_factor * extent
    fg_max = avg_cam + args.fg_bbox_factor * extent

    # ---------------- COLMAP 稀疏点 ----------------
    pts_path = os.path.join(args.scene_path, "sparse", "0", "points3D.bin")
    xyz, rgb, err, track_len = read_points3D_binary_with_tracks(pts_path)
    n_total_pts = xyz.shape[0]
    m_track = track_len >= args.min_track_length
    m_bbox = np.all(xyz > fg_min, axis=1) & np.all(xyz < fg_max, axis=1)
    keep = m_track & m_bbox
    ref_pts = xyz[keep].astype(np.float32)
    print(f"[INFO] COLMAP 点 {n_total_pts} -> track>={args.min_track_length}: {int(m_track.sum())}"
          f" -> 且在前景 bbox 内: {int(keep.sum())}")
    if ref_pts.shape[0] == 0:
        raise RuntimeError("过滤后参考点为空，检查 bbox / track 阈值")

    # ---------------- mesh ----------------
    mesh = o3d.io.read_triangle_mesh(args.mesh_path)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    n_verts, n_faces = verts.shape[0], faces.shape[0]
    n_nan_verts = int((~np.isfinite(verts)).any(axis=1).sum())
    print(f"[INFO] mesh: {n_verts} 顶点 / {n_faces} 面 / NaN 顶点 {n_nan_verts}")
    if n_faces == 0:
        raise RuntimeError("mesh 没有三角面")

    # ---------------- G1 / G2：稀疏点 -> mesh ----------------
    t_scene = o3d.t.geometry.RaycastingScene()
    t_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    d = t_scene.compute_distance(o3d.core.Tensor(ref_pts, dtype=o3d.core.Dtype.Float32)).numpy().astype(np.float64)
    g1 = {
        "n_reference_points": int(ref_pts.shape[0]),
        "mean_abs": float(d.mean()),
        "median_abs": float(np.median(d)),
        "p90_abs": float(np.percentile(d, 90)),
        "mean_rel": float(d.mean() / extent),
        "median_rel": float(np.median(d) / extent),
        "p90_rel": float(np.percentile(d, 90) / extent),
    }
    g2 = {
        "tau_0p5pct_abs": float(0.005 * extent),
        "ratio_below_0p5pct": float((d < 0.005 * extent).mean()),
        "tau_1pct_abs": float(0.01 * extent),
        "ratio_below_1pct": float((d < 0.01 * extent).mean()),
    }

    # ---------------- G3：mesh 顶点 -> 稀疏点 ----------------
    tree = cKDTree(ref_pts.astype(np.float64))
    finite_v = np.isfinite(verts).all(axis=1)
    dv_all, _ = tree.query(verts[finite_v], k=1, workers=-1)
    v_in_fg = finite_v & np.all(verts > fg_min, axis=1) & np.all(verts < fg_max, axis=1)
    dv_fg, _ = tree.query(verts[v_in_fg], k=1, workers=-1) if int(v_in_fg.sum()) > 0 else (np.zeros(0), None)
    g3 = {
        "n_vertices_all": int(finite_v.sum()),
        "mean_abs_all_vertices": float(dv_all.mean()),
        "median_abs_all_vertices": float(np.median(dv_all)),
        "mean_rel_all_vertices": float(dv_all.mean() / extent),
        "n_vertices_in_fg_bbox": int(v_in_fg.sum()),
        "mean_abs_fg_vertices": float(dv_fg.mean()) if dv_fg.size else None,
        "median_abs_fg_vertices": float(np.median(dv_fg)) if dv_fg.size else None,
        "mean_rel_fg_vertices": float(dv_fg.mean() / extent) if dv_fg.size else None,
        "note": "COLMAP 稀疏点本身稀疏（非稠密真值），该指标只作参考，不能单独判优劣",
    }

    # ---------------- G4：拓扑 ----------------
    tri_clusters, cluster_n_tri, cluster_area = mesh.cluster_connected_triangles()
    cluster_n_tri = np.asarray(cluster_n_tri, dtype=np.int64)
    n_components = int(cluster_n_tri.shape[0])
    largest = int(cluster_n_tri.max()) if n_components else 0
    uniq_e, counts, starts, faces_by_edge = edge_face_table(faces)
    n_boundary_edges = int((counts == 1).sum())
    n_nonmanifold_edges = int((counts > 2).sum())
    try:
        o3d_nm = len(mesh.get_non_manifold_edges(allow_boundary_edges=True))
    except Exception:
        o3d_nm = None
    # 诊断：重复顶点会把本应连通的面拆成多个分量，从而虚高连通分量数。
    # 三组 run 走完全相同的提取流水线，该偏差一致，不影响横向比较，但绝对值需要知情。
    _, first_idx = np.unique(np.round(verts, 9), axis=0, return_index=True)
    n_unique_positions = int(first_idx.shape[0])

    g4 = {
        "n_vertices": int(n_verts),
        "n_unique_vertex_positions": n_unique_positions,
        "n_duplicated_vertex_positions": int(n_verts - n_unique_positions),
        "n_faces": int(n_faces),
        "n_nan_vertices": n_nan_verts,
        "n_connected_components": n_components,
        "largest_component_faces": largest,
        "largest_component_face_ratio": float(largest / n_faces) if n_faces else 0.0,
        "n_fragment_components_lt_%d_faces" % args.fragment_face_threshold:
            int((cluster_n_tri < args.fragment_face_threshold).sum()),
        "fragment_faces_total": int(cluster_n_tri[cluster_n_tri < args.fragment_face_threshold].sum()),
        "n_unique_edges": int(uniq_e.shape[0]),
        "n_boundary_edges": n_boundary_edges,
        "n_non_manifold_edges": n_nonmanifold_edges,
        "n_non_manifold_edges_open3d": o3d_nm,
        "n_edges_used_by_2_faces": int((counts == 2).sum()),
    }

    # ---------------- G5：二面角 ----------------
    # 两套口径都报，避免被误读：
    #   raw = 两个面法向的夹角，依赖三角形绕序；若网格朝向不一致会在 180 度附近堆出假峰。
    #   abs = min(raw, 180 - raw)，与绕序无关，纯粹刻画两个面所在平面的夹角（0 度=共面）。
    # baseline 实测 raw 的直方图从 0-10 度单调递减到 170-180 度（无 180 度假峰），
    # 说明没有系统性的绕序翻转，raw 偏大是网格本身粗糙，不是朝向 bug。
    ang = dihedral_angles_deg(verts, faces, counts, starts, faces_by_edge)
    ang_abs = np.minimum(ang, 180.0 - ang) if ang.size else ang
    g5 = {
        "n_adjacent_face_pairs": int(ang.shape[0]),
        "dihedral_deg_mean": float(ang.mean()) if ang.size else None,
        "dihedral_deg_median": float(np.median(ang)) if ang.size else None,
        "dihedral_deg_p90": float(np.percentile(ang, 90)) if ang.size else None,
        "dihedral_abs_deg_mean": float(ang_abs.mean()) if ang.size else None,
        "dihedral_abs_deg_median": float(np.median(ang_abs)) if ang.size else None,
        "dihedral_abs_deg_p90": float(np.percentile(ang_abs, 90)) if ang.size else None,
        "frac_raw_gt_90deg": float((ang > 90.0).mean()) if ang.size else None,
        "histogram_deg_bin10_counts": np.histogram(ang, bins=18, range=(0.0, 180.0))[0].tolist() if ang.size else None,
        "note": "raw=面法向夹角（依赖绕序）；abs=min(raw,180-raw)（与绕序无关）。"
                "histogram 是 raw 的 18 个 10 度分箱计数，用来看有没有 180 度假峰",
    }

    out = {
        "run_name": args.run_name,
        "cameras_extent": float(extent),
        "cameras_average_center": avg_cam.tolist(),
        "fg_bbox_min": fg_min.tolist(),
        "fg_bbox_max": fg_max.tolist(),
        "G1_sparse_to_mesh_distance": g1,
        "G2_precision_ratio": g2,
        "G3_mesh_vertex_to_sparse_distance": g3,
        "G4_topology": g4,
        "G5_normal_smoothness": g5,
        "metric_directions": {
            "G1_*": "down", "G2_*": "up", "G3_*": "down",
            "G4_largest_component_face_ratio": "up",
            "G4_n_connected_components": "down",
            "G4_n_fragment_components": "down",
            "G5_dihedral_deg_mean": "down (但过低可能是过度平滑)",
        },
        "config": {
            "scene_path": os.path.abspath(args.scene_path),
            "mesh_path": os.path.abspath(args.mesh_path),
            "mesh_file_bytes": os.path.getsize(args.mesh_path),
            "gs_checkpoint": os.path.abspath(args.gs_checkpoint) if args.gs_checkpoint else None,
            "eval_split_interval": args.eval_split_interval,
            "min_track_length": args.min_track_length,
            "fg_bbox_factor": args.fg_bbox_factor,
            "center_bbox": True,
            "fragment_face_threshold": args.fragment_face_threshold,
            "extent_source": extent_source,
            "n_colmap_points_total": int(n_total_pts),
            "n_colmap_points_track_ok": int(m_track.sum()),
            "n_colmap_points_kept": int(keep.sum()),
        },
        "wall_clock_sec": None,
    }
    out["wall_clock_sec"] = round(time.time() - t_start, 2)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"geometry_{args.run_name}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n===== 几何指标 =====")
    print(f"run                       : {args.run_name}")
    print(f"cameras_extent            : {extent:.6f}")
    print(f"G1 median abs / rel       : {g1['median_abs']:.6f} / {g1['median_rel']*100:.4f}%")
    print(f"G1 mean   abs / rel       : {g1['mean_abs']:.6f} / {g1['mean_rel']*100:.4f}%")
    print(f"G1 P90    abs / rel       : {g1['p90_abs']:.6f} / {g1['p90_rel']*100:.4f}%")
    print(f"G2 <0.5% / <1%            : {g2['ratio_below_0p5pct']*100:.2f}% / {g2['ratio_below_1pct']*100:.2f}%")
    print(f"G3 mean (all / fg verts)  : {g3['mean_abs_all_vertices']:.6f} / {g3['mean_abs_fg_vertices']}")
    print(f"G4 顶点/面                : {g4['n_vertices']} / {g4['n_faces']}")
    print(f"G4 连通分量 / 最大占比    : {g4['n_connected_components']} / {g4['largest_component_face_ratio']*100:.2f}%")
    print(f"G4 碎片(<{args.fragment_face_threshold}面) / 非流形边 / 边界边 : "
          f"{g4['n_fragment_components_lt_%d_faces' % args.fragment_face_threshold]} / "
          f"{g4['n_non_manifold_edges']} / {g4['n_boundary_edges']}")
    print(f"G5 二面角 raw mean/P90 (度): {g5['dihedral_deg_mean']:.3f} / {g5['dihedral_deg_p90']:.3f}")
    print(f"G5 二面角 abs mean/P90 (度): {g5['dihedral_abs_deg_mean']:.3f} / {g5['dihedral_abs_deg_p90']:.3f}"
          f"   (raw>90 度占比 {g5['frac_raw_gt_90deg']*100:.2f}%)")
    print(f"JSON -> {json_path}")
    print(f"耗时 {out['wall_clock_sec']} s")


if __name__ == "__main__":
    main()
