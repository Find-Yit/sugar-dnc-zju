"""
train_coarse_official_dnc.py
============================
Entry point for the **official SuGaR** depth-normal-consistency coarse trainer, used as
the control group `coarse_official_dnc` of plan section 3e.

Why this file exists
--------------------
The official SuGaR repository already ships a 2DGS-style depth-normal consistency
regularization in `sugar_trainers/coarse_density_and_dn_consistency.py`
(`dn_consistency_factor = 0.05`, started at iteration 9000, no validity mask, no
absolute value, gradients flowing through BOTH the rendered depth and the rendered
normals).  Upstream only exposes it through `train.py -r dn_consistency`, which is a
full end-to-end script (coarse + refine + textured mesh) that we do not want to run.

This file is a *thin* entry point: it exposes exactly the same CLI as
`train_coarse_density.py` (plus `--seed`) and calls
`coarse_training_with_density_regularization_and_dn_consistency(args)` directly, so the
official regularizer can be compared head-to-head with our own DNC implementation under
the identical scene / 3DGS checkpoint / iteration budget / evaluation protocol.

NOTHING in the official trainer is modified: this module only
  (1) seeds random / numpy / torch before the call,
  (2) measures wall-clock time and peak CUDA memory around the call,
  (3) counts the final number of Gaussians from the saved checkpoint, and
  (4) dumps `<output_dir>/train_stats.json` with the SAME field names used by our
      `sugar_trainers/coarse_density.py` runs, so the runs are directly comparable.

Usage (note the REQUIRED trailing slash on -c):
    python train_coarse_official_dnc.py \
        -s $PROJ_ROOT/data/tandt/truck \
        -c $PROJ_ROOT/outputs/baseline/gs_truck/ \
        -i 7000 \
        -o $PROJ_ROOT/outputs/runs/coarse_official_dnc \
        --eval True --gpu 0 --seed 0
"""
import argparse
import json
import os
import random
import subprocess
import time

import numpy as np
import torch

from sugar_utils.general_utils import str2bool
from sugar_trainers.coarse_density_and_dn_consistency import (
    coarse_training_with_density_regularization_and_dn_consistency,
)

# Constants hard-coded inside the official trainer (kept here only for the stats file;
# they are NOT passed to the trainer, which owns its own copies).
OFFICIAL_DN_CONSISTENCY_FACTOR = 0.05
OFFICIAL_DN_CONSISTENCY_START = 9000
OFFICIAL_NUM_ITERATIONS = 15_000


def _git_info(repo_dir):
    info = {'commit': None, 'dirty': None}
    try:
        info['commit'] = subprocess.check_output(
            ['git', '-C', repo_dir, 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
        info['dirty'] = subprocess.check_output(
            ['git', '-C', repo_dir, 'status', '--short'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass
    return info


def _count_gaussians(model_path, key='_points'):
    """Number of Gaussians of the final coarse SuGaR model, read back from the saved .pt."""
    try:
        ckpt = torch.load(model_path, map_location='cpu')
        sd = ckpt['state_dict']
        for k in (key, '_scales', '_sh_coordinates_dc'):
            if k in sd:
                return int(sd[k].shape[0])
    except Exception as e:  # pragma: no cover - diagnostics only
        print(f'[STATS] WARNING: could not count gaussians from {model_path}: {e}')
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Optimize a coarse SuGaR model with the OFFICIAL SuGaR '
                    'depth-normal-consistency regularization (control group).')
    parser.add_argument('-c', '--checkpoint_path',
                        type=str,
                        help='path to the vanilla 3D Gaussian Splatting Checkpoint to load. '
                             'MUST end with a "/".')
    parser.add_argument('-s', '--scene_path',
                        type=str,
                        help='path to the scene data to use.')
    parser.add_argument('-o', '--output_dir',
                        type=str, default=None,
                        help='path to the output directory.')
    parser.add_argument('-i', '--iteration_to_load',
                        type=int, default=7000,
                        help='iteration to load.')

    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')
    parser.add_argument('--white_background', type=str2bool, default=False,
                        help='Use a white background instead of black.')

    parser.add_argument('-e', '--estimation_factor', type=float, default=0.2,
                        help='factor to multiply the estimation loss by.')
    parser.add_argument('-n', '--normal_factor', type=float, default=0.2,
                        help='factor to multiply the normal loss by.')

    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')

    # -----[REPRO] Random seed (same convention as our train_coarse_density.py) -----
    parser.add_argument('--seed', type=int, default=0,
                        help='[REPRO] Seed for random / numpy / torch.')

    args = parser.parse_args()

    if args.checkpoint_path is not None and not args.checkpoint_path.endswith('/'):
        print('[WARNING] -c/--checkpoint_path does not end with "/". '
              'SuGaR\'s GaussianSplattingWrapper needs the trailing slash; '
              'the run will most likely fail.')

    # ----------------------[REPRO] seeding----------------------
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f'[REPRO] random / numpy / torch seeded with {seed}.')

    # ----------------------[STATS] timers / memory----------------------
    device_index = int(args.gpu)
    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    t0 = time.time()

    model_path = coarse_training_with_density_regularization_and_dn_consistency(args)

    wallclock_s = time.time() - t0
    max_mem_bytes = int(torch.cuda.max_memory_allocated(device_index))
    max_mem_reserved_bytes = int(torch.cuda.max_memory_reserved(device_index))

    # ----------------------[STATS] dump----------------------
    n_iterations_counted = OFFICIAL_NUM_ITERATIONS - int(args.iteration_to_load)  # 8000
    n_gaussians_final = _count_gaussians(model_path)
    gi = _git_info(os.path.dirname(os.path.abspath(__file__)))

    train_stats = {
        'run_output_dir': args.output_dir,
        'final_model_path': model_path,
        'scene_path': args.scene_path,
        'gs_checkpoint_path': args.checkpoint_path,
        'iteration_to_load': int(args.iteration_to_load),
        'gpu_index': device_index,
        'gpu_name': torch.cuda.get_device_name(device_index),
        'seed': seed,
        'regularization': 'official_dn_consistency',
        'dn_consistency_factor': OFFICIAL_DN_CONSISTENCY_FACTOR,
        'start': OFFICIAL_DN_CONSISTENCY_START,
        'sdf_estimation_factor': float(args.estimation_factor),
        'sdf_better_normal_factor': float(args.normal_factor),
        'num_iterations': OFFICIAL_NUM_ITERATIONS,
        'n_iterations_counted': n_iterations_counted,
        'train_wallclock_s': float(wallclock_s),
        'train_wallclock_min': float(wallclock_s / 60.),
        'mean_time_per_iteration_ms': float(1000. * wallclock_s / n_iterations_counted),
        'max_memory_allocated_bytes': max_mem_bytes,
        'max_memory_allocated_MiB': float(max_mem_bytes / 1024. ** 2),
        'max_memory_reserved_MiB': float(max_mem_reserved_bytes / 1024. ** 2),
        'n_gaussians_final': n_gaussians_final,
        'eval_split': bool(args.eval),
        'white_background': bool(args.white_background),
        'trainer_module': 'sugar_trainers.coarse_density_and_dn_consistency',
        'entry_script': os.path.basename(__file__),
        'git_commit': gi['commit'],
        'git_dirty': gi['dirty'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'note': ('Wall-clock is measured around the whole trainer call (data loading '
                 'included), unlike coarse_density.py runs where train_wallclock_s starts '
                 'after data loading; mean_time_per_iteration_ms therefore divides by '
                 f'{n_iterations_counted} iterations and is a slight OVER-estimate '
                 '(~40 s of setup) compared with the other runs.'),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    stats_path = os.path.join(args.output_dir, 'train_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(train_stats, f, indent=2)
    print(f'[STATS] Training statistics saved to {stats_path}')
    print(f'[STATS] {n_iterations_counted} iterations in '
          f'{train_stats["train_wallclock_min"]:.2f} min '
          f'({train_stats["mean_time_per_iteration_ms"]:.1f} ms/iter), '
          f'peak CUDA memory {train_stats["max_memory_allocated_MiB"]:.0f} MiB, '
          f'{n_gaussians_final} gaussians.')
