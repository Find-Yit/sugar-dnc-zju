#!/usr/bin/env python
"""独立复算工具（验收用）：从 outputs/vis/<run>/ 里保存的 8bit PNG 重新计算 PSNR，
与 outputs/metrics/render_<run>.csv 里的数值比对。

这条路径完全不经过 torch / 3DGS 的 psnr 实现（只用 PIL + numpy），
因此可以独立验证 eval_render.py 的 M1 没有算错。
两者唯一的系统差异是 PNG 的 8bit 量化，实测差值在 0.003 dB 以内。

用法：
    python scripts/eval/check_png_psnr.py --runs vanilla3dgs7k,coarse_baseline
"""
import argparse
import csv
import os

import numpy as np
from PIL import Image

PROJ_ROOT = os.environ.get("PROJ_ROOT", "/scratch/e1351071/zju_test")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=str, required=True, help="逗号分隔的 run 名")
    p.add_argument("--vis_dir", type=str, default=os.path.join(PROJ_ROOT, "outputs", "vis"))
    p.add_argument("--metrics_dir", type=str, default=os.path.join(PROJ_ROOT, "outputs", "metrics"))
    p.add_argument("--tol_db", type=float, default=0.05, help="允许的最大差值（8bit 量化导致）")
    args = p.parse_args()

    worst = 0.0
    n_checked = 0
    for run in [r for r in args.runs.split(",") if r]:
        vis = os.path.join(args.vis_dir, run)
        csv_path = os.path.join(args.metrics_dir, f"render_{run}.csv")
        if not (os.path.isdir(vis) and os.path.isfile(csv_path)):
            print(f"[SKIP] {run}: 缺 {vis} 或 {csv_path}")
            continue
        rows = {int(r["test_view_idx"]): r for r in csv.DictReader(open(csv_path))}
        print(f"--- {run} ---")
        for fn in sorted(os.listdir(vis)):
            if not fn.startswith("render_"):
                continue
            tag = fn[len("render_"):-len(".png")]
            idx = int(tag[4:6])
            r = np.asarray(Image.open(os.path.join(vis, fn)), dtype=np.float64) / 255.0
            g = np.asarray(Image.open(os.path.join(vis, "gt_" + tag + ".png")), dtype=np.float64) / 255.0
            psnr_png = -10.0 * np.log10(((r - g) ** 2).mean())
            psnr_csv = float(rows[idx]["psnr_db"])
            diff = abs(psnr_png - psnr_csv)
            worst = max(worst, diff)
            n_checked += 1
            flag = "OK " if diff <= args.tol_db else "FAIL"
            print(f"  [{flag}] view{idx:02d} PNG={psnr_png:.4f} dB  CSV={psnr_csv:.4f} dB  diff={diff:.4f} dB")
    print(f"\n检查 {n_checked} 个视角，最大差值 {worst:.4f} dB，阈值 {args.tol_db} dB -> "
          f"{'PASS' if worst <= args.tol_db and n_checked else 'FAIL'}")


if __name__ == "__main__":
    main()
