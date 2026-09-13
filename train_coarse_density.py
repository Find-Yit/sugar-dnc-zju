import argparse
from sugar_utils.general_utils import str2bool
from sugar_trainers.coarse_density import coarse_training_with_density_regularization


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to optimize a coarse SuGaR model, i.e. a 3D Gaussian Splatting model with surface regularization losses in density space.')
    parser.add_argument('-c', '--checkpoint_path', 
                        type=str, 
                        help='path to the vanilla 3D Gaussian Splatting Checkpoint to load.')
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
    parser.add_argument('--white_background', type=str2bool, default=False, help='Use a white background instead of black.')
    
    parser.add_argument('-e', '--estimation_factor', type=float, default=0.2, help='factor to multiply the estimation loss by.')
    parser.add_argument('-n', '--normal_factor', type=float, default=0.2, help='factor to multiply the normal loss by.')
    
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')

    # -----[DNC] Depth-Normal Consistency regularization (new) -----
    parser.add_argument('--dnc_factor', type=float, default=0.0,
                        help='[DNC] Weight lambda of the depth-normal consistency loss. '
                             '0.0 (default) completely disables the DNC code path, so the '
                             'baseline behaves exactly like the original SuGaR.')
    parser.add_argument('--dnc_start', type=int, default=9000,
                        help='[DNC] Iteration after which the depth-normal consistency loss '
                             'is applied (same schedule as the SDF regularization by default).')
    parser.add_argument('--dnc_depth_grad_rel_thresh', type=float, default=0.05,
                        help='[DNC] Pixels whose relative depth gradient exceeds this value '
                             '(object silhouettes) are excluded from the loss.')
    parser.add_argument('--dnc_border', type=int, default=2,
                        help='[DNC] Number of image border pixels excluded from the loss.')
    parser.add_argument('--dnc_min_normal_norm', type=float, default=0.1,
                        help='[DNC] Pixels whose alpha-composited normal has a norm below '
                             'this value are excluded (numerical guard for low-opacity pixels).')
    parser.add_argument('--dnc_detach_depth', type=str2bool, default=False,
                        help='[DNC] If True, the geometric normal N_d is computed from a '
                             'DETACHED depth map, so the DNC gradient flows only through the '
                             'Gaussian normal map N and cannot flatten the rendered depth '
                             '(the trivial solution of L_dnc). Default False keeps the '
                             'behaviour of the first three reported runs unchanged.')
    parser.add_argument('--dnc_vis_every', type=int, default=1000,
                        help='[DNC] Save D / N / N_d / mask PNGs every N iterations (0 = off).')
    parser.add_argument('--dnc_log_every', type=int, default=100,
                        help='[DNC] Print and log L_dnc every N iterations.')

    # -----[SMOKE] Total number of iterations override (for the smoke test only) -----
    parser.add_argument('--num_iterations', type=int, default=None,
                        help='[SMOKE] Override the total number of training iterations '
                             '(default None = keep the original 15000). Only meant for '
                             'smoke tests; the three reported runs all use the default.')

    # -----[REPRO] Random seed -----
    parser.add_argument('--seed', type=int, default=0,
                        help='[REPRO] Seed for random / numpy / torch.')

    args = parser.parse_args()
    
    # Call function
    coarse_training_with_density_regularization(args)
    