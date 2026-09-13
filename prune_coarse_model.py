"""
[ADDED - M2-B] Stand-alone pre-processing step: DBSCAN cluster pruning of a trained
coarse SuGaR model.

    coarse .pt  ->  load SuGaR  ->  DBSCAN on the Gaussian centers  ->  prune isolated
    small clusters  ->  save a new .pt  ->  feed it to the UNMODIFIED extract_mesh.py
                                            via `-m <out>.pt`.

Zero coupling with the extraction side (nothing in extract_mesh.py /
sugar_extractors/coarse_mesh.py is touched), which also makes the ablation trivial:
extract once from the original .pt and once from the pruned .pt.

Example:
    python prune_coarse_model.py \
        -s data/tandt/truck -c outputs/baseline/gs_truck/ -i 7000 \
        -m outputs/runs/coarse_base_seed0/.../15000.pt --gpu 2 \
        -o outputs/m2/pruned_default/15000_pruned.pt
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from rich.console import Console

from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_scene.sugar_model import SuGaR
from sugar_utils.general_utils import str2bool
from sugar_utils.spherical_harmonics import SH2RGB
from sugar_utils.cluster_prune import cluster_prune_mask


# The loading path below is copied from sugar_extractors/coarse_mesh.py so that the
# model we write back out is byte-compatible with what extract_mesh.py expects.
def load_coarse_sugar(args, CONSOLE):
    n_skip_images_for_eval_split = 8
    CONSOLE.print(f"Loading the initial 3DGS model from path {args.checkpoint_path}...")
    nerfmodel = GaussianSplattingWrapper(
        source_path=args.scene_path,
        output_path=args.checkpoint_path,
        iteration_to_load=args.iteration_to_load,
        load_gt_images=False,
        eval_split=args.eval,
        eval_split_interval=n_skip_images_for_eval_split,
        )
    CONSOLE.print(f'{len(nerfmodel.training_cameras)} training images detected.')

    CONSOLE.print(f"\nLoading the coarse SuGaR model from path {args.coarse_model_path}...")
    checkpoint = torch.load(args.coarse_model_path, map_location=nerfmodel.device)
    colors = SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :])
    sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=checkpoint['state_dict']['_points'],
        colors=colors,
        initialize=True,
        sh_levels=nerfmodel.gaussians.active_sh_degree + 1,
        keep_track_of_knn=True,
        knn_to_track=16,
        beta_mode='average',
        primitive_types='diamond',
        surface_mesh_to_bind=None,
        )
    sugar.load_state_dict(checkpoint['state_dict'])
    sugar.eval()
    return nerfmodel, sugar, checkpoint


def main():
    parser = argparse.ArgumentParser(
        description='[M2-B] Prune isolated small DBSCAN clusters from a coarse SuGaR model.')
    # --- same flags as extract_mesh.py for the model to load ---
    parser.add_argument('-s', '--scene_path', type=str, required=True,
                        help='Path to the scene data (COLMAP dataset).')
    parser.add_argument('-c', '--checkpoint_path', type=str, required=True,
                        help='Path to the vanilla 3DGS checkpoint directory.')
    parser.add_argument('-i', '--iteration_to_load', type=int, default=7000,
                        help='3DGS iteration to load.')
    parser.add_argument('-m', '--coarse_model_path', type=str, required=True,
                        help='Path to the coarse SuGaR .pt model to prune.')
    parser.add_argument('--eval', type=str2bool, default=True,
                        help='Use train/test split (must match the training run).')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    # --- output ---
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help='Path of the pruned .pt to write. '
                             '<out>_prune_stats.json is written next to it.')
    # --- pruning hyper-parameters ---
    parser.add_argument('--rule', type=str, default='keep_large',
                        choices=['largest', 'keep_large'],
                        help="'largest': 2D-SuGaR original, keep only the biggest cluster "
                             "(failure-case control: deletes whole backgrounds). "
                             "'keep_large' (default): keep every cluster of size >= "
                             "keep_cluster_min_frac * N.")
    parser.add_argument('--keep_cluster_min_frac', type=float, default=0.005,
                        help="Minimum cluster size, as a fraction of the number of points "
                             "in the group, for rule=keep_large.")
    parser.add_argument('--separate_fg_bg', type=str2bool, default=True,
                        help='Estimate eps and run DBSCAN separately inside/outside the '
                             'foreground camera bbox (fg_bbox_factor=1, as in coarse_mesh.py).')
    parser.add_argument('--fg_bbox_factor', type=float, default=1.,
                        help='Foreground bbox factor (coarse_mesh.py uses 1.).')
    parser.add_argument('--min_samples', type=int, default=6, help='DBSCAN min_samples.')
    parser.add_argument('--knn_percentile', type=float, default=90,
                        help='Percentile of the k-th NN distance used as eps.')
    parser.add_argument('--eps_estimate_subsample', type=int, default=200000,
                        help='Subsample size used when estimating eps (0 = no subsample).')
    parser.add_argument('--opacity_min', type=float, default=0.,
                        help='Points with opacity below this do not take part in the '
                             'clustering, but are never pruned.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verify_reload', type=str2bool, default=True,
                        help='Self-check: reload the written .pt with coarse_mesh.py code path.')
    args = parser.parse_args()

    CONSOLE = Console(width=120)
    torch.cuda.set_device(args.gpu)
    t0 = time.time()

    CONSOLE.print("-----[M2-B] Cluster pruning parameters-----")
    for k, v in sorted(vars(args).items()):
        CONSOLE.print(f"  {k}: {v}")
    CONSOLE.print("-------------------------------------------")

    nerfmodel, sugar, checkpoint = load_coarse_sugar(args, CONSOLE)
    n_before = int(sugar.n_points)
    CONSOLE.print(f"Coarse model loaded: {n_before} gaussians.")

    # Foreground bbox, same convention as sugar_extractors/coarse_mesh.py
    extent, avg_xyz = sugar.get_cameras_spatial_extent(return_average_xyz=True)
    fg_min = (avg_xyz - args.fg_bbox_factor * extent * torch.ones(1, 3, device=sugar.device))
    fg_max = (avg_xyz + args.fg_bbox_factor * extent * torch.ones(1, 3, device=sugar.device))

    t_prune = time.time()
    keep_mask, stats = cluster_prune_mask(
        points=sugar.points,
        fg_bbox_min=fg_min.flatten().cpu().numpy(),
        fg_bbox_max=fg_max.flatten().cpu().numpy(),
        opacities=sugar.strengths[..., 0],
        rule=args.rule,
        separate_fg_bg=args.separate_fg_bg,
        min_samples=args.min_samples,
        knn_percentile=args.knn_percentile,
        keep_cluster_min_frac=args.keep_cluster_min_frac,
        eps_estimate_subsample=(args.eps_estimate_subsample if args.eps_estimate_subsample > 0 else None),
        opacity_min=args.opacity_min,
        seed=args.seed,
        verbose=True,
        )
    prune_seconds = time.time() - t_prune

    # prune_points() takes a KEEP mask (sugar_scene/sugar_model.py:819, cf.
    # drop_low_opacity_points which passes `strengths > threshold`).
    sugar.prune_points(torch.tensor(keep_mask, dtype=torch.bool, device=sugar.device))
    n_after = int(sugar.n_points)
    CONSOLE.print(f"Pruned model: {n_before} -> {n_after} gaussians.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    # Same save format as sugar_trainers/coarse_density.py (SuGaR.save_model ->
    # {'state_dict': ...}), which is what coarse_mesh.py loads.
    sugar.save_model(
        path=args.output_path,
        pruned_from=args.coarse_model_path,
        prune_stats=stats,
        iteration=checkpoint.get('iteration', None),
        epoch=checkpoint.get('epoch', None),
        )
    CONSOLE.print(f"Pruned model saved to {args.output_path}")

    stats.update({
        'coarse_model_path': args.coarse_model_path,
        'output_path': args.output_path,
        'scene_path': args.scene_path,
        'gs_checkpoint_path': args.checkpoint_path,
        'iteration_to_load': int(args.iteration_to_load),
        'gpu_index': int(args.gpu),
        'fg_bbox_factor': float(args.fg_bbox_factor),
        'cameras_spatial_extent': float(extent),
        'camera_average_xyz': [float(v) for v in avg_xyz.flatten().tolist()],
        'n_gaussians_before': n_before,
        'n_gaussians_after': n_after,
        'prune_seconds': float(prune_seconds),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    })

    # ---- self-check: can the freshly written .pt be reloaded by coarse_mesh.py's path?
    if args.verify_reload:
        CONSOLE.print("\n[self-check] Reloading the pruned model with the coarse_mesh.py code path...")
        ck = torch.load(args.output_path, map_location=nerfmodel.device)
        colors = SH2RGB(ck['state_dict']['_sh_coordinates_dc'][:, 0, :])
        sugar2 = SuGaR(
            nerfmodel=nerfmodel,
            points=ck['state_dict']['_points'],
            colors=colors,
            initialize=True,
            sh_levels=nerfmodel.gaussians.active_sh_degree + 1,
            keep_track_of_knn=True, knn_to_track=16,
            beta_mode='average', primitive_types='diamond', surface_mesh_to_bind=None,
            )
        sugar2.load_state_dict(ck['state_dict'])
        sugar2.eval()
        ok = int(sugar2.n_points) == n_after
        CONSOLE.print(f"[self-check] reloaded {int(sugar2.n_points)} gaussians "
                      f"(expected {n_after}) -> {'OK' if ok else 'MISMATCH'}")
        stats['reload_self_check_ok'] = bool(ok)
        stats['reload_n_points'] = int(sugar2.n_points)

    stats['total_seconds_script'] = float(time.time() - t0)
    stats_path = os.path.splitext(args.output_path)[0] + '_prune_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    CONSOLE.print(f"[STATS] {stats_path}")
    CONSOLE.print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
