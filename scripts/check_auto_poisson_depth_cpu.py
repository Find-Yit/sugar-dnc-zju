#!/usr/bin/env python
"""独立 CPU 复算 Frosting 自动 Poisson 深度（不依赖 SuGaR / GPU / pytorch3d）。

用途：交叉验证 sugar_extractors/coarse_mesh.py 里移植进来的 compute_optimal_poisson_depth
      的 GPU 结果。直接读 coarse 模型的 .pt（state_dict 里的 _points / all_densities）
      与 3DGS 输出的 cameras.json，用 scipy cKDTree 做 K=2 最近邻。

公式（与 Frosting 一致，注意 knn 距离用的是**平方**距离，照抄不开方）：
    extent, avg = 1.1 * max||c_i - mean(c)||, mean(c)          # 训练相机
    mask  = (|p - avg| < extent 各轴) & (sigmoid(density) > 0.5)
    bbox  = 1.1 * max(p[mask].max(0) - p[mask].min(0))
    d_q   = quantile_q( nn_dist2(p[mask]) / bbox )
    D     = min(floor(-log2(ratio * d_q)), 10)
"""
import argparse, json, os
import numpy as np
import torch
from scipy.spatial import cKDTree


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coarse_pt", required=True)
    p.add_argument("--gs_checkpoint", required=True, help="含 cameras.json 的 3DGS 输出目录")
    p.add_argument("--eval_split_interval", type=int, default=8, help="llffhold：索引 %% N == 0 的是测试图")
    p.add_argument("--ratio", type=float, default=100.)
    p.add_argument("--quantile", type=float, default=0.1)
    p.add_argument("--opacity_threshold", type=float, default=0.5)
    p.add_argument("--fg_bbox_factor", type=float, default=1.)
    p.add_argument("--max_poisson_depth", type=int, default=10)
    p.add_argument("--out_json", default="")
    a = p.parse_args()

    with open(os.path.join(a.gs_checkpoint, "cameras.json")) as f:
        cams = json.load(f)
    # SuGaR 的 CamerasWrapper 只装训练相机（eval_split 时剔除 idx % 8 == 0）
    train_centers = np.array([c["position"] for i, c in enumerate(cams)
                              if i % a.eval_split_interval != 0], dtype=np.float64)
    avg = train_centers.mean(axis=0, keepdims=True)
    extent = 1.1 * float(np.linalg.norm(train_centers - avg, axis=-1).max())

    sd = torch.load(a.coarse_pt, map_location="cpu")["state_dict"]
    pts = sd["_points"].float().numpy().astype(np.float64)
    op = torch.sigmoid(sd["all_densities"].view(-1)).numpy().astype(np.float64)

    lo, hi = avg - a.fg_bbox_factor * extent, avg + a.fg_bbox_factor * extent
    fg = np.all(pts > lo, axis=-1) & np.all(pts < hi, axis=-1)
    opq = op > a.opacity_threshold
    mask = fg & opq
    sel = pts[mask]

    bbox_size = 1.1 * float((sel.max(axis=0) - sel.min(axis=0)).max())
    d, _ = cKDTree(sel).query(sel, k=2, workers=-1)
    nn2 = d[:, 1] ** 2                      # 平方距离，与 pytorch3d knn_points 一致
    dq = float(np.quantile(nn2 / bbox_size, a.quantile))
    raw = -np.log2(a.ratio * dq)
    depth = int(min(int(np.floor(raw)), a.max_poisson_depth))

    res = dict(n_total=int(pts.shape[0]), n_fg=int(fg.sum()), n_opaque=int(opq.sum()),
               n_used=int(mask.sum()), cameras_spatial_extent=extent,
               camera_average_xyz=avg.ravel().tolist(), bbox_size=bbox_size,
               quantile_dist_normalized_SQUARED=dq, raw_depth_before_floor=float(raw),
               poisson_depth=depth, ratio=a.ratio, quantile=a.quantile,
               n_train_cameras=int(train_centers.shape[0]))
    print(json.dumps(res, indent=2))
    # ratio 敏感性：多少 ratio 会改变 D
    for r in (25, 50, 100, 200, 400, 800):
        rr = -np.log2(r * dq)
        print(f"  ratio={r:>5}: raw={rr:8.4f} -> D={min(int(np.floor(rr)), a.max_poisson_depth)}")
    if a.out_json:
        os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
        with open(a.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print("JSON ->", a.out_json)


if __name__ == "__main__":
    main()
