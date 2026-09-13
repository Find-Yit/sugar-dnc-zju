#!/usr/bin/env python
"""S4.3 mesh 可视化（计划 §2b V2）。

用 pytorch3d 的 MeshRasterizer（headless，无需显示环境）在与 eval_render.py 完全相同的
4 个固定测试视角上渲染 mesh 的：
  * 法向着色图：世界系面法向 -> RGB，(n * 0.5 + 0.5) * 255
  * 深度图：pytorch3d fragments.zbuf（单位与场景一致，即相机系 z，不是 NDC）-> turbo 伪彩色

相机取自 nerfmodel.test_cameras.p3d_cameras（与 sugar_scene/cameras.py 的 p3d_cameras 一致），
mesh 用 open3d 读 .ply 后转成 pytorch3d Meshes。

法向朝向说明：mesh 面法向的符号取决于三角形绕序，直接上色会出现同一平面正反两色。
本脚本把每个面的法向翻转成「朝向相机」（n · (cam_center - face_center) > 0），
这样同一视角下的着色是稳定可比的；三组 run 用同一规则，可横向比较。

输出：<vis_dir>/<run>/mesh_normal_view*.png, mesh_depth_view*.png
      <out_dir>/meshvis_<run>.json（记录每个视角的深度归一化区间，供阶段 5 画色条）
只读 repo/SuGaR，不修改任何现有模块。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (FIXED_TEST_VIEW_INDICES, colorize, ensure_trailing_sep,
                     save_png, setup_sugar_path)

PROJ_ROOT = os.environ.get("PROJ_ROOT", "/scratch/e1351071/zju_test")


def parse_args():
    p = argparse.ArgumentParser(description="用 pytorch3d 渲染 mesh 的法向图与深度图")
    p.add_argument("--scene_path", type=str, required=True)
    p.add_argument("--gs_checkpoint", type=str, required=True)
    p.add_argument("--mesh_path", type=str, required=True)
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--vis_dir", type=str, default=os.path.join(PROJ_ROOT, "outputs", "vis"))
    p.add_argument("--out_dir", type=str, default=os.path.join(PROJ_ROOT, "outputs", "metrics"))
    p.add_argument("--sugar_dir", type=str, default=None)
    p.add_argument("--eval_split_interval", type=int, default=8)
    p.add_argument("--view_indices", type=str, default=",".join(str(i) for i in FIXED_TEST_VIEW_INDICES),
                   help="测试集内部的视角顺序号，逗号分隔")
    p.add_argument("--depth_range_from", type=str, default="",
                   help="复用另一个 run 的 meshvis_*.json 的深度归一化区间，便于三组并排可比")
    p.add_argument("--max_faces_per_bin", type=int, default=200000)
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    setup_sugar_path(args.sugar_dir)

    import open3d as o3d
    import torch
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.structures import Meshes
    from sugar_scene.cameras import CamerasWrapper, load_gs_cameras

    view_indices = [int(x) for x in args.view_indices.split(",") if x.strip() != ""]

    # ---------------- 相机（与 eval_render.py 的测试划分与分辨率完全一致）----------------
    # 注意：必须 load_gt_images=True。3DGS 写出的 cameras.json 里 width/height 是 COLMAP 内参
    # 分辨率 1957x1091，而 images/ 里的图是 979x546；load_gt_images=False 时 SuGaR 会按
    # cameras.json 的尺寸再按 max_img_size=1920 缩放，得到 1920x1070，与 eval_render.py 渲染出的
    # RGB 图对不上。load_gt_images=True 时尺寸取自真实图片（979x546），两者才能逐像素对齐。
    cam_list = load_gs_cameras(
        source_path=args.scene_path,
        gs_output_path=ensure_trailing_sep(args.gs_checkpoint),
        load_gt_images=True,
    )
    test_cams = [c for i, c in enumerate(cam_list) if i % args.eval_split_interval == 0]
    test_cameras = CamerasWrapper(test_cams)
    H = int(test_cams[0].image_height)
    W = int(test_cams[0].image_width)
    print(f"[INFO] 测试相机 {len(test_cams)} 个，分辨率 {W}x{H}，渲染视角 {view_indices}")

    # ---------------- mesh ----------------
    o3d_mesh = o3d.io.read_triangle_mesh(args.mesh_path)
    verts_np = np.asarray(o3d_mesh.vertices, dtype=np.float32)
    faces_np = np.asarray(o3d_mesh.triangles, dtype=np.int64)
    print(f"[INFO] mesh: {verts_np.shape[0]} 顶点 / {faces_np.shape[0]} 面")
    verts = torch.from_numpy(verts_np).to(device)
    faces = torch.from_numpy(faces_np).to(device)
    meshes = Meshes(verts=[verts], faces=[faces])

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = torch.nn.functional.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
    fc = (v0 + v1 + v2) / 3.0

    raster_settings = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1,
        max_faces_per_bin=args.max_faces_per_bin,
    )

    reuse_ranges = {}
    if args.depth_range_from and os.path.isfile(args.depth_range_from):
        with open(args.depth_range_from) as f:
            prev = json.load(f)
        reuse_ranges = {int(k): v for k, v in prev.get("depth_ranges", {}).items()}
        print(f"[INFO] 复用深度色标区间：{args.depth_range_from}")

    vis_run_dir = os.path.join(args.vis_dir, args.run_name)
    os.makedirs(vis_run_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    depth_ranges = {}
    coverage = {}
    with torch.no_grad():
        for idx in view_indices:
            p3d_cam = test_cameras.p3d_cameras[idx]
            rasterizer = MeshRasterizer(cameras=p3d_cam, raster_settings=raster_settings)
            frags = rasterizer(meshes)
            pix_to_face = frags.pix_to_face[0, ..., 0]        # (H, W)
            zbuf = frags.zbuf[0, ..., 0]                      # (H, W)，相机系 z（场景单位）
            hit = pix_to_face >= 0

            # --- 法向图（世界系，翻转成朝向相机）---
            cam_center = p3d_cam.get_camera_center()[0]       # (3,)
            safe_idx = torch.where(hit, pix_to_face, torch.zeros_like(pix_to_face))
            n_pix = fn[safe_idx]                              # (H, W, 3)
            c_pix = fc[safe_idx]
            to_cam = torch.nn.functional.normalize(cam_center[None, None, :] - c_pix, dim=-1)
            sign = torch.where((n_pix * to_cam).sum(-1, keepdim=True) < 0, -1.0, 1.0)
            n_pix = n_pix * sign
            normal_rgb = ((n_pix * 0.5 + 0.5).clamp(0, 1) * 255.0).round().to(torch.uint8)
            normal_rgb = torch.where(hit[..., None], normal_rgb,
                                     torch.full_like(normal_rgb, 255))  # 背景白
            normal_np = normal_rgb.cpu().numpy()

            # --- 深度图 ---
            z = zbuf.cpu().numpy()
            hit_np = hit.cpu().numpy()
            valid_z = z[hit_np]
            if idx in reuse_ranges:
                vmin, vmax = float(reuse_ranges[idx][0]), float(reuse_ranges[idx][1])
            elif valid_z.size:
                vmin = float(np.percentile(valid_z, 1))
                vmax = float(np.percentile(valid_z, 99))
            else:
                vmin, vmax = 0.0, 1.0
            depth_rgb = colorize(np.where(hit_np, z, vmax), vmin, vmax)
            depth_rgb[~hit_np] = 255  # 背景白
            depth_ranges[idx] = [vmin, vmax]
            coverage[idx] = float(hit_np.mean())

            name = test_cams[idx].image_name
            tag = f"view{idx:02d}_{name}"
            save_png(os.path.join(vis_run_dir, f"mesh_normal_{tag}.png"), normal_np)
            save_png(os.path.join(vis_run_dir, f"mesh_depth_{tag}.png"), depth_rgb)
            print(f"  [{tag}] 命中率 {coverage[idx]*100:.2f}%  深度区间 [{vmin:.3f}, {vmax:.3f}]")
            del frags, rasterizer
            torch.cuda.empty_cache()

    out = {
        "run_name": args.run_name,
        "view_indices": view_indices,
        "image_names": {i: test_cams[i].image_name for i in view_indices},
        "depth_ranges": {str(k): v for k, v in depth_ranges.items()},
        "mesh_pixel_coverage": {str(k): v for k, v in coverage.items()},
        "config": {
            "mesh_path": os.path.abspath(args.mesh_path),
            "scene_path": os.path.abspath(args.scene_path),
            "gs_checkpoint": os.path.abspath(args.gs_checkpoint),
            "image_height": H, "image_width": W,
            "n_vertices": int(verts_np.shape[0]), "n_faces": int(faces_np.shape[0]),
            "normal_encoding": "world-space face normal, flipped toward camera, rgb=(n*0.5+0.5)*255, background=white",
            "depth_source": "pytorch3d fragments.zbuf (camera-space z, scene units), background=white",
            "depth_colormap": "matplotlib turbo (256-entry LUT)",
            "depth_range_rule": "per-view [p1, p99] of hit pixels" if not reuse_ranges else f"reused from {args.depth_range_from}",
            "faces_per_pixel": 1, "blur_radius": 0.0,
            "eval_split_interval": args.eval_split_interval,
            "gpu": args.gpu,
        },
        "wall_clock_sec": None,
    }
    out["wall_clock_sec"] = round(time.time() - t_start, 2)
    json_path = os.path.join(args.out_dir, f"meshvis_{args.run_name}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nPNG  -> {vis_run_dir}")
    print(f"JSON -> {json_path}")
    print(f"耗时 {out['wall_clock_sec']} s")


if __name__ == "__main__":
    main()
