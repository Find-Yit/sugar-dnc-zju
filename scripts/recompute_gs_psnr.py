"""独立复算 3DGS ckpt 在 test 视角上的 PSNR（阶段2 验收条目 B1 的交叉验证）。
不修改仓库任何文件，只 import gaussian_splatting 的官方模块。
用法: python scripts/recompute_gs_psnr.py
"""
import os, sys, json, csv
GS = "/scratch/e1351071/zju_test/repo/SuGaR/gaussian_splatting"
sys.path.insert(0, GS)
os.chdir(GS)

import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr
from utils.loss_utils import l1_loss

parser = ArgumentParser()
lp = ModelParams(parser)  # 用默认值（与 cfg_args 一致: resolution=-1, white_background=False, images=images）
pp = PipelineParams(parser)
args = parser.parse_args([
    "-s", "/scratch/e1351071/zju_test/data/tandt/truck",
    "-m", "/scratch/e1351071/zju_test/outputs/baseline/gs_truck",
    "--eval",
])
dataset = lp.extract(args)
pipe = pp.extract(args)

gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=7000, shuffle=False)
bg = torch.tensor([0., 0., 0.], dtype=torch.float32, device="cuda")

rows = []
for split, cams in (("test", scene.getTestCameras()), ("train", scene.getTrainCameras())):
    ps, l1s = [], []
    for i, cam in enumerate(cams):
        with torch.no_grad():
            img = torch.clamp(render(cam, gaussians, pipe, bg)["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
            p = psnr(img, gt).mean().double().item()
            l = l1_loss(img, gt).mean().double().item()
        ps.append(p); l1s.append(l)
        rows.append({"split": split, "idx": i, "image_name": cam.image_name, "psnr": p, "l1": l})
    print(f"{split}: n={len(cams)}  PSNR={sum(ps)/len(ps):.6f}  L1={sum(l1s)/len(l1s):.6f}")

out = "/scratch/e1351071/zju_test/outputs/metrics/gs7000_psnr_recompute.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["split", "idx", "image_name", "psnr", "l1"])
    w.writeheader(); w.writerows(rows)
print("逐视角结果写入:", out)
print("n_gaussians:", gaussians.get_xyz.shape[0])
