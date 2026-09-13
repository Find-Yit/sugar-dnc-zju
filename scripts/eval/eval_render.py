#!/usr/bin/env python
"""S4.1 渲染指标评测（计划 §2b M1-M3）。

在 32 张测试视角（3DGS --eval 划分，llffhold=8）上评测：
  M1 PSNR (dB, 越高越好) —— gaussian_splatting/utils/image_utils.psnr，逐图 MSE 取 -10log10 后对视角求均值
  M2 SSIM (越高越好)     —— gaussian_splatting/utils/loss_utils.ssim（11x11 高斯窗）
  M3 LPIPS-VGG (越低越好)—— gaussian_splatting/lpipsPyTorch（VGG 权重落到 TORCH_HOME）

被评模型：
  * 给了 --coarse_pt  -> coarse SuGaR（加载方式与 sugar_extractors/coarse_mesh.py 完全一致），
                        用 sugar.render_image_gaussian_rasterizer 渲染（与 SuGaR 官方 metrics.py 一致）。
  * 没给 --coarse_pt  -> vanilla 3DGS 参考行，默认用 3DGS 官方渲染器 nerfmodel.render_image
                        （--vanilla_render_mode sugar 可切换成 SuGaR 包装器路径做交叉验证）。

输出：
  <out_dir>/render_<run>.csv   逐视角
  <out_dir>/render_<run>.json  均值 + 配置
  <vis_dir>/<run>/            4 个固定测试视角的 render / gt / 误差热图 PNG

只读 repo/SuGaR，不修改任何现有模块。
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (ERROR_HEATMAP_VMAX, FIXED_TEST_VIEW_INDICES, colorize,
                     ensure_trailing_sep, save_png, setup_sugar_path, to_uint8)


def parse_args():
    p = argparse.ArgumentParser(description="SuGaR / 3DGS 渲染指标评测（PSNR / SSIM / LPIPS-VGG）")
    p.add_argument("--scene_path", type=str, required=True, help="COLMAP 场景目录，例如 data/tandt/truck")
    p.add_argument("--gs_checkpoint", type=str, required=True, help="vanilla 3DGS 输出目录（含 cameras.json 与 point_cloud/）")
    p.add_argument("--iteration", type=int, default=7000, help="要加载的 3DGS 迭代数，默认 7000")
    p.add_argument("--coarse_pt", type=str, default="", help="coarse SuGaR 的 .pt；留空则评 vanilla 3DGS")
    p.add_argument("--run_name", type=str, required=True, help="run 名，用于输出文件名")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=os.path.join(os.environ.get("PROJ_ROOT", "/scratch/e1351071/zju_test"), "outputs", "metrics"))
    p.add_argument("--vis_dir", type=str, default=os.path.join(os.environ.get("PROJ_ROOT", "/scratch/e1351071/zju_test"), "outputs", "vis"))
    p.add_argument("--sugar_dir", type=str, default=None, help="SuGaR 仓库路径（默认 $PROJ_ROOT/repo/SuGaR）")
    p.add_argument("--eval_split_interval", type=int, default=8, help="llffhold，默认 8")
    p.add_argument("--skip_lpips", action="store_true", help="跳过 LPIPS（权重下载失败时用）")
    p.add_argument("--no_vis", action="store_true", help="不保存可视化 PNG")
    p.add_argument("--vanilla_render_mode", type=str, default="gs", choices=["gs", "sugar"],
                   help="仅当未给 --coarse_pt 时生效：gs=3DGS 官方渲染器；sugar=SuGaR 包装器")
    return p.parse_args()


def build_sugar_from_coarse_pt(SuGaR, SH2RGB, nerfmodel, coarse_pt, torch):
    """与 sugar_extractors/coarse_mesh.py 第 165-182 行完全一致的加载方式。"""
    checkpoint = torch.load(coarse_pt, map_location=nerfmodel.device)
    colors = SH2RGB(checkpoint["state_dict"]["_sh_coordinates_dc"][:, 0, :])
    sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=checkpoint["state_dict"]["_points"],
        colors=colors,
        initialize=True,
        sh_levels=nerfmodel.gaussians.active_sh_degree + 1,
        keep_track_of_knn=False,
        knn_to_track=16,
        beta_mode="average",
        primitive_types="diamond",
        surface_mesh_to_bind=None,
    )
    sugar.load_state_dict(checkpoint["state_dict"])
    sugar.eval()
    return sugar


def build_sugar_from_vanilla(SuGaR, SH2RGB, nerfmodel, torch):
    """与 coarse_mesh.py 的 use_vanilla_3dgs 分支一致。"""
    with torch.no_grad():
        points = nerfmodel.gaussians.get_xyz.detach().float().cuda()
        colors = SH2RGB(nerfmodel.gaussians.get_features[:, 0].detach().float().cuda())
    sugar = SuGaR(
        nerfmodel=nerfmodel, points=points, colors=colors, initialize=True,
        sh_levels=nerfmodel.gaussians.active_sh_degree + 1,
        keep_track_of_knn=False, knn_to_track=16, beta_mode="average",
        primitive_types="diamond", surface_mesh_to_bind=None,
    )
    with torch.no_grad():
        sugar._scales[...] = nerfmodel.gaussians._scaling.detach()
        sugar._quaternions[...] = nerfmodel.gaussians._rotation.detach()
        sugar.all_densities[...] = nerfmodel.gaussians._opacity.detach()
        sugar._sh_coordinates_dc[...] = nerfmodel.gaussians._features_dc.detach()
        sugar._sh_coordinates_rest[...] = nerfmodel.gaussians._features_rest.detach()
    sugar.eval()
    return sugar


def main():
    args = parse_args()
    t_start = time.time()

    sugar_dir = setup_sugar_path(args.sugar_dir)
    import torch
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(0)
    np.random.seed(0)

    from sugar_scene.gs_model import GaussianSplattingWrapper
    from sugar_scene.sugar_model import SuGaR
    from sugar_utils.spherical_harmonics import SH2RGB
    from gaussian_splatting.utils.image_utils import psnr as psnr_fn
    from gaussian_splatting.utils.loss_utils import ssim as ssim_fn

    lpips_criterion = None
    lpips_error = None
    if not args.skip_lpips:
        try:
            from gaussian_splatting.lpipsPyTorch.modules.lpips import LPIPS
            lpips_criterion = LPIPS(net_type="vgg", version="0.1").to(f"cuda:{args.gpu}").eval()
        except Exception as exc:  # 权重下载失败等
            lpips_error = repr(exc)
            print(f"[WARN] LPIPS 初始化失败，将跳过 M3: {lpips_error}")

    use_coarse = bool(args.coarse_pt)
    print(f"[INFO] SuGaR 仓库: {sugar_dir}")
    print(f"[INFO] 模式: {'coarse SuGaR' if use_coarse else 'vanilla 3DGS (' + args.vanilla_render_mode + ')'}")

    nerfmodel = GaussianSplattingWrapper(
        source_path=args.scene_path,
        output_path=ensure_trailing_sep(args.gs_checkpoint),
        iteration_to_load=args.iteration,
        load_gt_images=True,
        eval_split=True,
        eval_split_interval=args.eval_split_interval,
    )
    n_test = len(nerfmodel.test_cameras)
    n_train = len(nerfmodel.training_cameras)
    print(f"[INFO] 训练视角 {n_train} / 测试视角 {n_test}")

    sh_deg = nerfmodel.gaussians.active_sh_degree
    renderer_kind = "3dgs_official"
    sugar = None
    if use_coarse:
        sugar = build_sugar_from_coarse_pt(SuGaR, SH2RGB, nerfmodel, args.coarse_pt, torch)
        renderer_kind = "sugar_gaussian_rasterizer"
    elif args.vanilla_render_mode == "sugar":
        sugar = build_sugar_from_vanilla(SuGaR, SH2RGB, nerfmodel, torch)
        renderer_kind = "sugar_gaussian_rasterizer"
    n_gaussians = int(sugar.n_points) if sugar is not None else int(nerfmodel.gaussians.get_xyz.shape[0])
    print(f"[INFO] 高斯数量 {n_gaussians}，SH degree {sh_deg}，渲染器 {renderer_kind}")

    vis_run_dir = os.path.join(args.vis_dir, args.run_name)
    if not args.no_vis:
        os.makedirs(vis_run_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    with torch.no_grad():
        for cam_idx in range(n_test):
            gt_img = nerfmodel.get_test_gt_image(cam_idx, to_cuda=True).permute(2, 0, 1).unsqueeze(0)
            if sugar is not None:
                img = sugar.render_image_gaussian_rasterizer(
                    nerf_cameras=nerfmodel.test_cameras,
                    camera_indices=cam_idx,
                    verbose=False,
                    bg_color=None,
                    sh_deg=sh_deg,
                    compute_color_in_rasterizer=True,
                ).clamp(min=0, max=1).permute(2, 0, 1).unsqueeze(0)
            else:
                img = nerfmodel.render_image(
                    nerf_cameras=nerfmodel.test_cameras,
                    camera_indices=cam_idx,
                ).clamp(min=0, max=1).permute(2, 0, 1).unsqueeze(0)

            row = {
                "test_view_idx": cam_idx,
                "image_name": nerfmodel.test_cam_list[cam_idx].image_name,
                "psnr_db": float(psnr_fn(img, gt_img).mean().item()),
                "ssim": float(ssim_fn(img, gt_img).item()),
            }
            if lpips_criterion is not None:
                row["lpips_vgg"] = float(lpips_criterion(img, gt_img).item())
            else:
                row["lpips_vgg"] = ""
            rows.append(row)

            if (not args.no_vis) and cam_idx in FIXED_TEST_VIEW_INDICES:
                r = img[0].permute(1, 2, 0).cpu().numpy()
                g = gt_img[0].permute(1, 2, 0).cpu().numpy()
                err = np.abs(r - g).mean(axis=-1)
                tag = f"view{cam_idx:02d}_{row['image_name']}"
                save_png(os.path.join(vis_run_dir, f"render_{tag}.png"), to_uint8(r))
                save_png(os.path.join(vis_run_dir, f"gt_{tag}.png"), to_uint8(g))
                save_png(os.path.join(vis_run_dir, f"errmap_{tag}.png"),
                         colorize(err, 0.0, ERROR_HEATMAP_VMAX))
                row_err = {"view": cam_idx, "err_mean": float(err.mean()), "err_max": float(err.max())}
                print(f"[VIS] {tag} err_mean={row_err['err_mean']:.4f} err_max={row_err['err_max']:.4f}")

            if cam_idx % 8 == 0:
                print(f"  [{cam_idx + 1}/{n_test}] PSNR={row['psnr_db']:.3f} SSIM={row['ssim']:.4f}")

    # ---------------- 落盘 ----------------
    csv_path = os.path.join(args.out_dir, f"render_{args.run_name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test_view_idx", "image_name", "psnr_db", "ssim", "lpips_vgg"])
        w.writeheader()
        w.writerows(rows)

    psnrs = np.array([r["psnr_db"] for r in rows], dtype=np.float64)
    ssims = np.array([r["ssim"] for r in rows], dtype=np.float64)
    summary = {
        "run_name": args.run_name,
        "n_test_views": n_test,
        "n_train_views": n_train,
        "psnr_db_mean": float(psnrs.mean()),
        "psnr_db_std": float(psnrs.std(ddof=0)),
        "ssim_mean": float(ssims.mean()),
        "ssim_std": float(ssims.std(ddof=0)),
        "n_gaussians": n_gaussians,
        "config": {
            "scene_path": os.path.abspath(args.scene_path),
            "gs_checkpoint": os.path.abspath(args.gs_checkpoint),
            "iteration_to_load": args.iteration,
            "coarse_pt": os.path.abspath(args.coarse_pt) if args.coarse_pt else None,
            "eval_split_interval": args.eval_split_interval,
            "renderer": renderer_kind,
            "sh_degree": int(sh_deg),
            "gpu": args.gpu,
            "image_height": int(nerfmodel.image_height),
            "image_width": int(nerfmodel.image_width),
            "fixed_vis_test_view_indices": FIXED_TEST_VIEW_INDICES,
            "error_heatmap_vmax": ERROR_HEATMAP_VMAX,
            "error_heatmap_colormap": "matplotlib turbo (256-entry LUT)",
            "low_opacity_pruning_applied": False,
            "note": "coarse .pt 按训练结束状态直接评测，不再做 opacity<0.5 剪枝（该剪枝在训练 9000 iter 已执行）",
        },
        "metric_directions": {"psnr_db": "up", "ssim": "up", "lpips_vgg": "down"},
        "wall_clock_sec": None,
    }
    if lpips_criterion is not None:
        lp = np.array([r["lpips_vgg"] for r in rows], dtype=np.float64)
        summary["lpips_vgg_mean"] = float(lp.mean())
        summary["lpips_vgg_std"] = float(lp.std(ddof=0))
    else:
        summary["lpips_vgg_mean"] = None
        summary["lpips_vgg_std"] = None
        summary["lpips_error"] = lpips_error
    summary["wall_clock_sec"] = round(time.time() - t_start, 2)

    json_path = os.path.join(args.out_dir, f"render_{args.run_name}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== 结果 =====")
    print(f"run           : {args.run_name}")
    print(f"PSNR (dB, up) : {summary['psnr_db_mean']:.4f}")
    print(f"SSIM (up)     : {summary['ssim_mean']:.5f}")
    print(f"LPIPS-VGG(dn) : {summary['lpips_vgg_mean'] if summary['lpips_vgg_mean'] is None else format(summary['lpips_vgg_mean'], '.5f')}")
    print(f"n_gaussians   : {n_gaussians}")
    print(f"CSV  -> {csv_path}")
    print(f"JSON -> {json_path}")
    if not args.no_vis:
        print(f"VIS  -> {vis_run_dir}")
    print(f"耗时 {summary['wall_clock_sec']} s")


if __name__ == "__main__":
    main()
