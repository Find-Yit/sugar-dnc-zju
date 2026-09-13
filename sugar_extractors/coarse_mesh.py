import os
import json  # [ADDED] for extract_stats.json
import time  # [ADDED] for extract_stats.json
import numpy as np
import open3d as o3d
import torch
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer
from pytorch3d.ops import knn_points
from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_scene.sugar_model import SuGaR
from sugar_utils.general_utils import str2bool
from sugar_utils.spherical_harmonics import SH2RGB

from rich.console import Console

# ---------------------------------------------------------------------------
# [ADDED - ported from Gaussian Frosting]
# Automatic selection of the Poisson octree depth.
#
# Source (body copied verbatim):
#   Anttwo/Frosting -> frosting_extractors/coarse_shell.py, lines 17-49
#   https://github.com/Anttwo/Frosting/blob/main/frosting_extractors/coarse_shell.py
#   raw: https://raw.githubusercontent.com/Anttwo/Frosting/main/frosting_extractors/coarse_shell.py
#   Paper: Guedon & Lepetit, "Gaussian Frosting: Editable Complex Radiance Fields
#          with Real-Time Rendering", ECCV 2024 (Oral) -- same authors as SuGaR.
#          Supplementary material, section 7 "Improving surface reconstruction":
#          SuGaR hard-codes a large depth D=10 for every scene; when the octree
#          resolution is too high w.r.t. the scene's level of detail, the ellipsoidal
#          shape of the Gaussians shows up as bumps on the surface and holes appear.
#
# Idea: take the 10% quantile of the nearest-neighbour distance of the foreground,
#       opaque Gaussians (normalised by the bounding-box size), ask the Poisson cell
#       size to be about 1/100 of it, and solve for
#           D = min(floor(-log2(100 * d_10%)), 10).
#       Denser Gaussians -> larger D; sparser Gaussians -> automatically smaller D.
#
# NOTE: pytorch3d's knn_points(...).dists returns *SQUARED* distances. Frosting uses
#       them as-is and the constant cell_size_nn_distance_ratio=100 was tuned with that
#       convention, so the body below is copied verbatim and is deliberately NOT
#       "fixed" by taking a square root.
#
# The ONLY modification w.r.t. the original is the optional `return_details` keyword
# (default False -> identical signature/behaviour), which additionally returns the
# intermediate quantities so that they can be logged to extract_stats.json.
# ---------------------------------------------------------------------------
def compute_optimal_poisson_depth(
    sugar:SuGaR,
    cell_size_nn_distance_ratio:float=100,
    max_poisson_depth:int=10,
    fg_bbox_factor=1.,
    opacity_threshold=0.5,
    quantile_to_use=0.1,
    return_details:bool=False,
    ):

    # Compute foreground bbox
    _cameras_spatial_extent, _camera_average_xyz = sugar.get_cameras_spatial_extent(return_average_xyz=True)
    fg_bbox_min_tensor = _camera_average_xyz - fg_bbox_factor * _cameras_spatial_extent * torch.ones(1, 3, device=sugar.device)
    fg_bbox_max_tensor = _camera_average_xyz + fg_bbox_factor * _cameras_spatial_extent * torch.ones(1, 3, device=sugar.device)
    fg_mask = (sugar.points > fg_bbox_min_tensor).all(dim=-1) * (sugar.points < fg_bbox_max_tensor).all(dim=-1)

    # Remove transparent gaussians
    opacity_mask = sugar.strengths[..., 0] > opacity_threshold
    mask = opacity_mask * fg_mask

    # Compute poisson bbox
    bbox_size = 1.1 * (sugar.points[mask].max(dim=0)[0] - sugar.points[mask].min(dim=0)[0]).max().item()

    # Compute nearest neighbor distances and normalize by bbox size
    nn_dists = knn_points(sugar.points[mask][None], sugar.points[mask][None], K=2).dists[0, ..., 1]
    norm_nn_dists = nn_dists / bbox_size
    quantile_dist = norm_nn_dists.quantile(quantile_to_use).item()

    # Compute optimal poisson depth
    poisson_depth = -np.log2(cell_size_nn_distance_ratio * quantile_dist)
    poisson_depth = np.floor(poisson_depth).astype(int)
    poisson_depth = min(poisson_depth, max_poisson_depth)

    if return_details:
        details = {
            'cell_size_nn_distance_ratio': float(cell_size_nn_distance_ratio),
            'max_poisson_depth': int(max_poisson_depth),
            'fg_bbox_factor': float(fg_bbox_factor),
            'opacity_threshold': float(opacity_threshold),
            'quantile_to_use': float(quantile_to_use),
            'cameras_spatial_extent': float(_cameras_spatial_extent),
            'camera_average_xyz': [float(v) for v in _camera_average_xyz.flatten().tolist()],
            'n_gaussians_total': int(sugar.points.shape[0]),
            'n_gaussians_fg': int(fg_mask.sum().item()),
            'n_gaussians_opaque': int(opacity_mask.sum().item()),
            'n_gaussians_used': int(mask.sum().item()),
            'bbox_size': float(bbox_size),
            'quantile_dist_normalized_SQUARED': float(quantile_dist),
            'nn_dist_note': 'knn_points returns SQUARED distances (pytorch3d); copied verbatim from Frosting, not square-rooted',
            'raw_depth_before_floor': float(-np.log2(cell_size_nn_distance_ratio * quantile_dist)),
            'poisson_depth': int(poisson_depth),
        }
        return int(poisson_depth), details
    return poisson_depth


# ---------------------------------------------------------------------------
# [M1+ : 自有改动，区别于 Frosting 原版]
# Frosting 原版只估计 *一个* D（前景高斯中心的最近邻分位数），前景/背景共用，且样本
# 永远是高斯中心。实测 Truck 上 raw=10.87 被 max_poisson_depth=10 截断 => 零收益。
# M1+a：前景 / 背景各自估 D（背景点更稀疏，应得到更小的 D）。
# M1+b：`--depth_estimate_source surface` 改用真正送进 Poisson 的表面采样点来估计，
#       而不是高斯中心；表面点密度才是 Poisson 八叉树真正面对的密度。
# 公式与 Frosting 完全一致（knn_points K=2 的 *平方* 距离、不开方、除以 bbox 尺寸、
# D = min(floor(-log2(ratio * d_q)), max_poisson_depth)），只是换了样本集合。
# ---------------------------------------------------------------------------
def compute_poisson_depth_from_points(
    points: torch.Tensor,
    cell_size_nn_distance_ratio: float = 100.,
    max_poisson_depth: int = 10,
    quantile_to_use: float = 0.1,
    max_points_for_knn: int = 500_000,
    tag: str = '',
    ):
    """[自有改动] 对任意一组点（高斯中心 或 表面采样点）套用 Frosting 的 D 公式。

    points: (N, 3) tensor. 超过 max_points_for_knn 时随机子采样（受 --extract_seed 控制）。
    返回 (poisson_depth:int, details:dict)。
    """
    n_points_total = int(points.shape[0])
    if n_points_total < 2:
        return None, {'tag': tag, 'n_points_total': n_points_total, 'error': 'not enough points'}

    pts = points
    n_subsampled = n_points_total
    if n_points_total > max_points_for_knn:
        idx = torch.randperm(n_points_total, device=pts.device)[:max_points_for_knn]
        pts = pts[idx]
        n_subsampled = int(pts.shape[0])

    # bbox 用这组点自己的包围盒（Frosting 用 1.1 * 最大边长）
    bbox_size = 1.1 * (pts.max(dim=0)[0] - pts.min(dim=0)[0]).max().item()

    # 与 Frosting 一致：knn_points 返回的是 *平方* 距离，不开方
    nn_dists = knn_points(pts[None].float(), pts[None].float(), K=2).dists[0, ..., 1]
    norm_nn_dists = nn_dists / bbox_size
    quantile_dist = norm_nn_dists.quantile(quantile_to_use).item()

    raw_depth = -np.log2(cell_size_nn_distance_ratio * quantile_dist)
    poisson_depth = int(min(int(np.floor(raw_depth)), int(max_poisson_depth)))

    details = {
        'tag': tag,
        'n_points_total': n_points_total,
        'n_points_used_for_knn': n_subsampled,
        'max_points_for_knn': int(max_points_for_knn),
        'bbox_size': float(bbox_size),
        'quantile_to_use': float(quantile_to_use),
        'quantile_dist_normalized_SQUARED': float(quantile_dist),
        'cell_size_nn_distance_ratio': float(cell_size_nn_distance_ratio),
        'max_poisson_depth': int(max_poisson_depth),
        'raw_depth_before_floor': float(raw_depth),
        'poisson_depth': int(poisson_depth),
        'nn_dist_note': 'knn_points returns SQUARED distances (pytorch3d), kept as-is like Frosting',
    }
    return poisson_depth, details


def compute_optimal_poisson_depth_bg(
    sugar: SuGaR,
    cell_size_nn_distance_ratio: float = 100.,
    max_poisson_depth: int = 10,
    fg_bbox_factor: float = 1.,
    bg_bbox_factor: float = 4.,
    opacity_threshold: float = 0.5,
    quantile_to_use: float = 0.1,
    max_points_for_knn: int = 500_000,
    ):
    """[自有改动 M1+a] 用 *背景* 高斯中心估 D_bg。

    背景集合 = (bg bbox 内) AND (fg bbox 外) AND (opacity > 0.5)，
    与 extract_mesh_from_coarse_sugar 里划分 fg_mask / bg_mask 的规则一致；
    bbox_size 用背景点自己的包围盒。Frosting 原版没有这一步。
    """
    _cameras_spatial_extent, _camera_average_xyz = sugar.get_cameras_spatial_extent(return_average_xyz=True)
    fg_bbox_min_tensor = _camera_average_xyz - fg_bbox_factor * _cameras_spatial_extent * torch.ones(1, 3, device=sugar.device)
    fg_bbox_max_tensor = _camera_average_xyz + fg_bbox_factor * _cameras_spatial_extent * torch.ones(1, 3, device=sugar.device)
    fg_mask = (sugar.points > fg_bbox_min_tensor).all(dim=-1) * (sugar.points < fg_bbox_max_tensor).all(dim=-1)
    bg_mask = ((sugar.points - _camera_average_xyz).abs().max(dim=-1)[0] < bg_bbox_factor * _cameras_spatial_extent) * ~fg_mask
    opacity_mask = sugar.strengths[..., 0] > opacity_threshold
    mask = opacity_mask * bg_mask

    depth, details = compute_poisson_depth_from_points(
        sugar.points[mask],
        cell_size_nn_distance_ratio=cell_size_nn_distance_ratio,
        max_poisson_depth=max_poisson_depth,
        quantile_to_use=quantile_to_use,
        max_points_for_knn=max_points_for_knn,
        tag='bg_centers',
        )
    details.update({
        'fg_bbox_factor': float(fg_bbox_factor),
        'bg_bbox_factor': float(bg_bbox_factor),
        'opacity_threshold': float(opacity_threshold),
        'cameras_spatial_extent': float(_cameras_spatial_extent),
        'n_gaussians_total': int(sugar.points.shape[0]),
        'n_gaussians_bg_bbox': int(bg_mask.sum().item()),
        'n_gaussians_used': int(mask.sum().item()),
    })
    return depth, details


def extract_mesh_from_coarse_sugar(args):
    CONSOLE = Console(width=120)
    
    all_sugar_mesh_paths = []

    # ========== Parameters ==========

    use_train_test_split = True
    n_skip_images_for_eval_split = 8
    low_opacity_gaussian_pruning_threshold = 0.5

    # Surface level extraction parameters
    n_total_points = 10_000_000
    use_gaussian_depth_for_surface_levels = False  # False until now
    surface_level_triangle_scale = 2.  # 2.
    # surface_level_triangle_scale = -2 * np.log(surface_level)
    surface_level_primitive_types = 'diamond'  # 'diamond'
    surface_level_splat_mesh = True  # True
    surface_level_n_points_in_range = 21  # 21
    surface_level_range_size = 3.0  # 3.0
    surface_level_n_points_per_pass = 2_000_000  # '2_000_000'
    surface_level_knn_to_track = 16  # 16
    flat_surface_level_normals = False  # False

    use_fast_method = True  # TODO: Was False before, but True seems better

    # Mesh computation parameters
    fg_bbox_factor = 1.  # 1.
    bg_bbox_factor = 4.  # 4.
    # [MODIFIED] These two used to be hard-coded here; they are now overridable from the
    # command line (extract_mesh.py --poisson_depth / --vertices_density_quantile).
    # The values below are the original SuGaR defaults and are kept as fallbacks, so
    # running extract_mesh.py without the new flags reproduces the original behaviour exactly.
    poisson_depth = 10  # 10 for most real scenes. 6 or 7 work well for most synthetic scenes
    vertices_density_quantile = 0.1  # 0.1 for most real scenes. 0. works well for most synthetic scenes
    decimate_mesh = True
    clean_mesh = True
    project_mesh_on_surface_points = args.project_mesh_on_surface_points

    # [ADDED] Poisson depth / cleaning quantile now come from the command line.
    # --poisson_depth is a *string* so that it accepts either an integer ("10") or "auto"
    # ("-1" is also accepted, that is the spelling Frosting uses for "auto").
    poisson_depth_arg = str(getattr(args, 'poisson_depth', '10')).strip()
    use_auto_poisson_depth = poisson_depth_arg.lower() in ('auto', '-1')
    if not use_auto_poisson_depth:
        poisson_depth = int(poisson_depth_arg)
    poisson_depth_str = 'auto' if use_auto_poisson_depth else str(poisson_depth)
    vertices_density_quantile = float(getattr(args, 'vertices_density_quantile', 0.1))
    cell_size_nn_distance_ratio = float(getattr(args, 'cell_size_nn_distance_ratio', 100.))
    only_report_depth = bool(getattr(args, 'only_report_depth', False))

    # ---- [M1+ 自有改动，区别于 Frosting 原版] ----
    # --max_poisson_depth   : Frosting 写死 10 的上限，现在可调（Truck raw=10.87 被 10 截断）。
    # --poisson_depth_bg    : 背景单独的 D。'same'(默认)=原版行为，与前景共用；整数或 'auto'。
    # --depth_estimate_source: 'centers'(默认, Frosting 原样，用高斯中心) 或 'surface'
    #                          (用真正送进 Poisson 的表面采样点，前景/背景各自估)。
    # --extract_seed        : >=0 时给 torch/np/random 播种，使表面点 randperm 可复现。
    max_poisson_depth = int(getattr(args, 'max_poisson_depth', 10))
    poisson_depth_bg_arg = str(getattr(args, 'poisson_depth_bg', 'same')).strip()
    _bg_low = poisson_depth_bg_arg.lower()
    use_same_depth_for_bg = _bg_low in ('same', '')
    use_auto_poisson_depth_bg = _bg_low in ('auto', '-1')
    depth_estimate_source = str(getattr(args, 'depth_estimate_source', 'centers')).strip().lower()
    if depth_estimate_source not in ('centers', 'surface'):
        raise ValueError(f"--depth_estimate_source must be 'centers' or 'surface', got {depth_estimate_source}")
    use_surface_depth_estimate = (depth_estimate_source == 'surface')
    extract_seed = int(getattr(args, 'extract_seed', -1))
    if extract_seed >= 0:
        import random as _random
        torch.manual_seed(extract_seed)
        torch.cuda.manual_seed_all(extract_seed)
        np.random.seed(extract_seed)
        _random.seed(extract_seed)

    # 实际用于 fg / bg Poisson 的两个深度（bg 默认跟随 fg = 原版行为）
    poisson_depth_fg = poisson_depth
    poisson_depth_bg = poisson_depth
    if not (use_same_depth_for_bg or use_auto_poisson_depth_bg):
        poisson_depth_bg = int(poisson_depth_bg_arg)

    extract_stats = {
        'poisson_depth_arg': poisson_depth_arg,
        'poisson_depth_is_auto': bool(use_auto_poisson_depth),
        'poisson_depth_bg_arg': poisson_depth_bg_arg,
        'poisson_depth_bg_is_auto': bool(use_auto_poisson_depth_bg),
        'poisson_depth_bg_is_same_as_fg': bool(use_same_depth_for_bg),
        'depth_estimate_source': depth_estimate_source,
        'max_poisson_depth': max_poisson_depth,
        'extract_seed': extract_seed,
        'vertices_density_quantile': vertices_density_quantile,
        'cell_size_nn_distance_ratio': cell_size_nn_distance_ratio,
        'only_report_depth': only_report_depth,
        'coarse_model_path': args.coarse_model_path,
        'scene_path': args.scene_path,
        'gs_checkpoint_path': args.checkpoint_path,
        'surface_level_arg': args.surface_level,
        'decimation_target_arg': args.decimation_target,
        'eval_split': bool(args.eval),
    }
    _extract_t0 = time.time()
    
    # Vanilla 3DGS data
    source_path = args.scene_path
    gs_checkpoint_path = args.checkpoint_path
    iteration_to_load = args.iteration_to_load
    use_train_test_split = args.eval
    
    # Coarse model path
    sugar_checkpoint_path = args.coarse_model_path
            
    # Surface levels to extract
    if args.surface_level is None:
        surface_levels = [0.1, 0.3, 0.5]
    else:
        surface_levels = [args.surface_level]
        
    # Decimation targets
    if args.decimation_target is None:
        decimation_targets = [200_000, 1_000_000]
    else:
        decimation_targets = [args.decimation_target]
    
    # Mesh output dir
    if args.mesh_output_dir is None:
        if len(args.scene_path.split("/")[-1]) > 0:
            args.mesh_output_dir = os.path.join("./output/coarse_mesh", args.scene_path.split("/")[-1])
        else:
            args.mesh_output_dir = os.path.join("./output/coarse_mesh", args.scene_path.split("/")[-2])
    mesh_output_dir = args.mesh_output_dir
    os.makedirs(mesh_output_dir, exist_ok=True)
            
    # Bounding box
    if (args.bboxmin is None) or (args.bboxmin == 'None'):
        use_custom_bbox = False
    else:
        if (args.bboxmax is None) or (args.bboxmax == 'None'):
            raise ValueError("You need to specify both bboxmin and bboxmax.")
        use_custom_bbox = True
        
        # Parse bboxmin
        if args.bboxmin[0] == '(':
            args.bboxmin = args.bboxmin[1:]
        if args.bboxmin[-1] == ')':
            args.bboxmin = args.bboxmin[:-1]
        args.bboxmin = tuple([float(x) for x in args.bboxmin.split(",")])
        
        # Parse bboxmax
        if args.bboxmax[0] == '(':
            args.bboxmax = args.bboxmax[1:]
        if args.bboxmax[-1] == ')':
            args.bboxmax = args.bboxmax[:-1]
        args.bboxmax = tuple([float(x) for x in args.bboxmax.split(",")])
        
        fg_bbox_min = args.bboxmin
        fg_bbox_max = args.bboxmax
    center_bbox = args.center_bbox
        
    use_centers_to_extract_mesh = args.use_centers_to_extract_mesh
    use_marching_cubes = args.use_marching_cubes
    use_vanilla_3dgs = args.use_vanilla_3dgs
            
    CONSOLE.print("-----Parameters-----")
    CONSOLE.print("Source path:", source_path)
    CONSOLE.print("Gaussian Splatting Checkpoint path:", gs_checkpoint_path)
    CONSOLE.print("Coarse model Checkpoint path:", sugar_checkpoint_path)
    CONSOLE.print("Mesh output path:", mesh_output_dir)
    CONSOLE.print("Surface levels:", surface_levels)
    CONSOLE.print("Decimation targets:", decimation_targets)
    CONSOLE.print("Project mesh on surface points:", project_mesh_on_surface_points)
    CONSOLE.print("Use custom bbox:", use_custom_bbox)
    CONSOLE.print("Use eval split:", use_train_test_split)
    CONSOLE.print("GPU:", args.gpu)
    CONSOLE.print("Use centers to extract mesh:", use_centers_to_extract_mesh)
    CONSOLE.print("Use marching cubes:", use_marching_cubes)
    CONSOLE.print("Use vanilla 3DGS:", use_vanilla_3dgs)
    CONSOLE.print("Poisson depth:", poisson_depth_str)  # [ADDED]
    CONSOLE.print("Vertices density quantile (cleaning quantile):", vertices_density_quantile)  # [ADDED]
    CONSOLE.print("Cell size / NN distance ratio (auto depth only):", cell_size_nn_distance_ratio)  # [ADDED]
    CONSOLE.print("Only report depth (no extraction):", only_report_depth)  # [ADDED]
    CONSOLE.print("[M1+] Background poisson depth arg:", poisson_depth_bg_arg)
    CONSOLE.print("[M1+] Depth estimate source:", depth_estimate_source)
    CONSOLE.print("[M1+] Max poisson depth:", max_poisson_depth)
    CONSOLE.print("[M1+] Extract seed:", extract_seed)
    CONSOLE.print("--------------------")
    
    # Set the GPU
    torch.cuda.set_device(args.gpu)
    
    # Load the initial 3DGS model
    CONSOLE.print(f"Loading the initial 3DGS model from path {gs_checkpoint_path}...")
    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_checkpoint_path,
        iteration_to_load=iteration_to_load,
        load_gt_images=False,
        eval_split=use_train_test_split,
        eval_split_interval=n_skip_images_for_eval_split,
        )
    
    CONSOLE.print(f'{len(nerfmodel.training_cameras)} training images detected.')
    CONSOLE.print(f'The model has been trained for {iteration_to_load} steps.')
    
    # Load the coarse model
    if use_vanilla_3dgs:
        CONSOLE.print(f"\nUsing the vanilla 3DGS model for meshing...")
        with torch.no_grad():    
            print("Initializing model from trained 3DGS...")
            points = nerfmodel.gaussians.get_xyz.detach().float().cuda()
            colors = SH2RGB(nerfmodel.gaussians.get_features[:, 0].detach().float().cuda())
        sugar = SuGaR(
            nerfmodel=nerfmodel,
            points=points,
            colors=colors,
            initialize=True,
            sh_levels=nerfmodel.gaussians.active_sh_degree+1,
            keep_track_of_knn=True,
            knn_to_track=16,
            beta_mode='average',  # 'learnable', 'average', 'weighted_average'
            primitive_types='diamond',  # 'diamond', 'square'
            surface_mesh_to_bind=None,  # Open3D mesh
            )
        with torch.no_grad():
            sugar._scales[...] = nerfmodel.gaussians._scaling.detach()
            sugar._quaternions[...] = nerfmodel.gaussians._rotation.detach()
            sugar.all_densities[...] = nerfmodel.gaussians._opacity.detach()
            sugar._sh_coordinates_dc[...] = nerfmodel.gaussians._features_dc.detach()
            sugar._sh_coordinates_rest[...] = nerfmodel.gaussians._features_rest.detach()
    else:
        CONSOLE.print(f"\nLoading the coarse SuGaR model from path {sugar_checkpoint_path}...")
        checkpoint = torch.load(sugar_checkpoint_path, map_location=nerfmodel.device)
        colors = SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :])
        sugar = SuGaR(
            nerfmodel=nerfmodel,
            points=checkpoint['state_dict']['_points'],
            colors=colors,
            initialize=True,
            sh_levels=nerfmodel.gaussians.active_sh_degree+1,
            keep_track_of_knn=True,
            knn_to_track=16,
            beta_mode='average',  # 'learnable', 'average', 'weighted_average'
            primitive_types='diamond',  # 'diamond', 'square'
            surface_mesh_to_bind=None,  # Open3D mesh
            )
        sugar.load_state_dict(checkpoint['state_dict'])
    sugar.eval()
    
    CONSOLE.print("Coarse model loaded.")
    CONSOLE.print("Coarse model parameters:")
    for name, param in sugar.named_parameters():
        CONSOLE.print(name, param.shape, param.requires_grad)

    # [ADDED] Frosting-style automatic Poisson depth. Computed here, i.e. BEFORE the
    # low-opacity pruning below, exactly like Frosting (coarse_shell.py:234-240) -- the
    # function applies its own opacity_threshold=0.5 mask internally.
    _need_centers_estimate = only_report_depth or (
        (use_auto_poisson_depth or use_auto_poisson_depth_bg) and not use_surface_depth_estimate)
    if _need_centers_estimate:
        # --- fg：Frosting 原样（高斯中心） ---
        CONSOLE.print("Computing optimal poisson depth (fg, centers)...")
        _auto_depth, _auto_details = compute_optimal_poisson_depth(
            sugar,
            cell_size_nn_distance_ratio=cell_size_nn_distance_ratio,
            max_poisson_depth=max_poisson_depth,
            return_details=True,
            )
        CONSOLE.print("Optimal poisson depth (fg, centers):", _auto_depth)
        CONSOLE.print("  -> details:", _auto_details)
        extract_stats['auto_poisson_depth'] = int(_auto_depth)
        extract_stats['auto_depth_details'] = _auto_details
        extract_stats['depth_fg_centers'] = int(_auto_depth)
        extract_stats['depth_fg_centers_details'] = _auto_details
        # --- bg：[M1+a 自有改动] 背景高斯中心单独估 D ---
        CONSOLE.print("[M1+a] Computing optimal poisson depth (bg, centers)...")
        _auto_depth_bg, _auto_details_bg = compute_optimal_poisson_depth_bg(
            sugar,
            cell_size_nn_distance_ratio=cell_size_nn_distance_ratio,
            max_poisson_depth=max_poisson_depth,
            fg_bbox_factor=fg_bbox_factor,
            bg_bbox_factor=bg_bbox_factor,
            )
        CONSOLE.print("[M1+a] Optimal poisson depth (bg, centers):", _auto_depth_bg)
        CONSOLE.print("  -> details:", _auto_details_bg)
        extract_stats['depth_bg_centers'] = None if _auto_depth_bg is None else int(_auto_depth_bg)
        extract_stats['depth_bg_centers_details'] = _auto_details_bg

        if not use_surface_depth_estimate:
            if use_auto_poisson_depth:
                poisson_depth = int(_auto_depth)
                poisson_depth_fg = int(_auto_depth)
                if use_same_depth_for_bg:
                    poisson_depth_bg = int(_auto_depth)
            if use_auto_poisson_depth_bg and _auto_depth_bg is not None:
                poisson_depth_bg = int(_auto_depth_bg)

    extract_stats['poisson_depth_used'] = int(poisson_depth_fg)
    extract_stats['poisson_depth_fg_used'] = int(poisson_depth_fg)
    extract_stats['poisson_depth_bg_used'] = int(poisson_depth_bg)
    _stats_path = os.path.join(mesh_output_dir, 'extract_stats.json')
    with open(_stats_path, 'w') as _f:
        json.dump(extract_stats, _f, indent=2)
    CONSOLE.print("Extraction stats written to", _stats_path)
    if only_report_depth and not use_surface_depth_estimate:
        CONSOLE.print("[only_report_depth=True] Skipping the actual mesh extraction.")
        return []

    # Pruning low opacity gaussians
    with torch.no_grad():
        CONSOLE.print("Number of gaussians:", sugar.n_points)
        CONSOLE.print("Opacities min/max/mean:", sugar.strengths.min(), sugar.strengths.max(), sugar.strengths.mean())
        n_quantiles = 10
        for i in range(n_quantiles):
            CONSOLE.print(f'Quantile {i/n_quantiles}:', sugar.strengths.quantile(i/n_quantiles).item())
            
        CONSOLE.print("\nStarting pruning low opacity gaussians...")
        sugar.drop_low_opacity_points(low_opacity_gaussian_pruning_threshold)

        CONSOLE.print("Number of gaussians left:", sugar.n_points)
        CONSOLE.print("Opacities min/max/mean:", sugar.strengths.min(), sugar.strengths.max(), sugar.strengths.mean())
        n_quantiles = 10
        for i in range(n_quantiles):
            CONSOLE.print(f'Quantile {i/n_quantiles}:', sugar.strengths.quantile(i/n_quantiles).item())
            
    # Build the triangle soup that will be used for splatting
    # sugar.primitive_types = 'square'
    sugar.primitive_types = 'diamond'
    sugar.triangle_scale = 2.
    sugar.update_texture_features()
    mesh = sugar.mesh
    
    # Create a mesh renderer
    faces_per_pixel = 10
    max_faces_per_bin = 50_000

    mesh_raster_settings = RasterizationSettings(
        image_size=(sugar.image_height, sugar.image_width),
        blur_radius=0.0, 
        faces_per_pixel=faces_per_pixel,
        max_faces_per_bin=max_faces_per_bin
    )
    rasterizer = MeshRasterizer(
            cameras=nerfmodel.training_cameras.p3d_cameras[0], 
            raster_settings=mesh_raster_settings,
    )
    
    if not use_marching_cubes:
        if not use_centers_to_extract_mesh:
            # Compute surface levels point clouds
            n_pts_per_frame = int(n_total_points / len(nerfmodel.training_cameras)) + 1
            sugar.knn_to_track = surface_level_knn_to_track

            surface_levels_outputs = {}
            for surface_level in surface_levels:
                surface_levels_outputs[surface_level] = {
                    'points': torch.zeros(0, 3, device=sugar.device),
                    'colors': torch.zeros(0, 3, device=sugar.device),
                    'view_directions': torch.zeros(0, 3, device=sugar.device),
                    'pix_to_gaussians': torch.zeros(0, dtype=torch.long, device=sugar.device),
                    'normals': torch.zeros(0, 3, device=sugar.device),
                }

            with torch.no_grad():
                cameras_to_use = nerfmodel.training_cameras
                    
                for cam_idx in range(len(nerfmodel.training_cameras)):
                    if cam_idx % 30 == 0:
                        CONSOLE.print(f"Processing frame {cam_idx}/{len(nerfmodel.training_cameras)}...")
                        for surface_level in surface_levels:
                            CONSOLE.print(f"Current point cloud for level {surface_level} has {len(surface_levels_outputs[surface_level]['points'])} points.")
                    
                    point_depth = cameras_to_use.p3d_cameras[cam_idx].get_world_to_view_transform().transform_points(sugar.points)[..., 2:].expand(-1, 3)
                    
                    # Render RGB image with Gaussian splatting
                    rgb = sugar.render_image_gaussian_rasterizer(
                        nerf_cameras=cameras_to_use, 
                        camera_indices=cam_idx,
                        bg_color = None,
                        sh_deg=0,  # nerfmodel.gaussians.active_sh_degree,
                        compute_color_in_rasterizer=True,
                        compute_covariance_in_rasterizer=True,
                        return_2d_radii=False,
                        use_same_scale_in_all_directions=False,
                    ).clamp(min=0., max=1.).contiguous()
                    
                    # Compute surface level points for the current frame
                    if cam_idx == 0:
                        sugar.reset_neighbors(knn_to_track=surface_level_knn_to_track)
                    with torch.no_grad():
                        if use_fast_method:
                            frame_surface_level_outputs = sugar.compute_level_surface_points_from_camera_fast(
                                cam_idx=cam_idx,
                                rasterizer=rasterizer,
                                surface_levels=surface_levels, 
                                n_surface_points=2*n_pts_per_frame,  # TODO: 2*n_pts_per_frame is safe to avoid empty pixels
                                primitive_types=surface_level_primitive_types, 
                                triangle_scale=surface_level_triangle_scale,
                                splat_mesh=surface_level_splat_mesh,
                                n_points_in_range=surface_level_n_points_in_range,
                                range_size=surface_level_range_size,
                                n_points_per_pass=surface_level_n_points_per_pass,
                                density_factor=1.,
                                return_pixel_idx=True,
                                return_gaussian_idx=True,
                                return_normals=True,
                                compute_flat_normals=flat_surface_level_normals,
                                use_gaussian_depth=use_gaussian_depth_for_surface_levels,)
                        else:
                            frame_surface_level_outputs = sugar.compute_level_surface_points_from_camera_efficient(
                                cam_idx=cam_idx,
                                rasterizer=rasterizer,
                                surface_levels=surface_levels, 
                                primitive_types=surface_level_primitive_types, 
                                triangle_scale=surface_level_triangle_scale,
                                splat_mesh=surface_level_splat_mesh,
                                n_points_in_range=surface_level_n_points_in_range,
                                range_size=surface_level_range_size,
                                n_points_per_pass=surface_level_n_points_per_pass,
                                density_factor=1.,
                                return_depth=True,
                                return_gaussian_idx=True,
                                return_normals=True,
                                compute_flat_normals=flat_surface_level_normals,
                                use_gaussian_depth=use_gaussian_depth_for_surface_levels,)
                        
                        for surface_level in surface_levels:
                            img_surface_points = frame_surface_level_outputs[surface_level]['intersection_points']
                            surface_gaussian_idx = frame_surface_level_outputs[surface_level]['gaussian_idx']
                            img_surface_normals = frame_surface_level_outputs[surface_level]['normals']
                            
                            if use_fast_method:
                                pixel_idx = frame_surface_level_outputs[surface_level]['pixel_idx']
                                img_surface_colors = rgb.view(-1, 3)[pixel_idx]
                            else:
                                empty_pixels = frame_surface_level_outputs[surface_level]['empty_pixels']
                                img_surface_colors = rgb.view(-1, 3)[~empty_pixels]
                            
                            img_surface_view_directions = torch.nn.functional.normalize(cameras_to_use.p3d_cameras[cam_idx].get_camera_center() - img_surface_points)
                            img_surface_pix_to_gaussians = surface_gaussian_idx.view(-1)
                            
                            idx = torch.randperm(len(img_surface_points), device=sugar.device)[:n_pts_per_frame]
                            
                            surface_levels_outputs[surface_level]['points'] = torch.cat([surface_levels_outputs[surface_level]['points'], img_surface_points[idx]], dim=0)
                            surface_levels_outputs[surface_level]['colors'] = torch.cat([surface_levels_outputs[surface_level]['colors'], img_surface_colors[idx]], dim=0)
                            surface_levels_outputs[surface_level]['view_directions'] = torch.cat([surface_levels_outputs[surface_level]['view_directions'], img_surface_view_directions[idx]], dim=0)
                            surface_levels_outputs[surface_level]['pix_to_gaussians'] = torch.cat([surface_levels_outputs[surface_level]['pix_to_gaussians'], img_surface_pix_to_gaussians[idx]], dim=0)
                            surface_levels_outputs[surface_level]['normals'] = torch.cat([surface_levels_outputs[surface_level]['normals'], img_surface_normals[idx]], dim=0)

            # -----Processing surface levels-----
            for surface_level in surface_levels:
                CONSOLE.print("\n========== Processing surface level", surface_level, "==========")
                CONSOLE.print(f"Final point cloud for level {surface_level} has {len(surface_levels_outputs[surface_level]['points'])} points.")
                surface_points = surface_levels_outputs[surface_level]['points']
                surface_colors = surface_levels_outputs[surface_level]['colors']
                surface_normals = surface_levels_outputs[surface_level]['normals']

                if use_custom_bbox:
                    CONSOLE.print("Using provided bounding box.")
                    fg_bbox_min_tensor = torch.tensor(fg_bbox_min).to(sugar.device)
                    fg_bbox_max_tensor = torch.tensor(fg_bbox_max).to(sugar.device)
                else:
                    CONSOLE.print("Using default, camera based bounding box.")
                    fg_bbox_min_tensor = - fg_bbox_factor * sugar.get_cameras_spatial_extent() * torch.ones(1, 3, device=sugar.device)
                    fg_bbox_max_tensor = fg_bbox_factor * sugar.get_cameras_spatial_extent() * torch.ones(1, 3, device=sugar.device)
                    
                if center_bbox:
                    _cameras_spatial_extent, _camera_average_xyz = sugar.get_cameras_spatial_extent(return_average_xyz=True)
                    with torch.no_grad():
                        CONSOLE.print("Centering bounding box.")
                        fg_bbox_min_tensor = fg_bbox_min_tensor + _camera_average_xyz
                        fg_bbox_max_tensor = fg_bbox_max_tensor + _camera_average_xyz

                points_idx = torch.arange(len(surface_points))
                fg_mask = (surface_points[points_idx] > fg_bbox_min_tensor).all(dim=-1) * (surface_points[points_idx] < fg_bbox_max_tensor).all(dim=-1)
                if center_bbox:
                    bg_mask = ((surface_points[points_idx] - _camera_average_xyz).abs().max(dim=-1)[0]
                               < bg_bbox_factor * _cameras_spatial_extent) * ~fg_mask
                else:
                    bg_mask = (surface_points[points_idx].abs().max(dim=-1)[0] < bg_bbox_factor * sugar.get_cameras_spatial_extent()) * ~fg_mask

                fg_points = surface_points[points_idx][fg_mask]
                fg_colors = surface_colors[points_idx][fg_mask]
                fg_normals = surface_normals[points_idx][fg_mask]

                bg_points = surface_points[points_idx][bg_mask]
                bg_colors = surface_colors[points_idx][bg_mask]
                bg_normals = surface_normals[points_idx][bg_mask]

                CONSOLE.print("Foreground points:", fg_points.shape, fg_colors.shape, fg_normals.shape)
                CONSOLE.print("Background points:", bg_points.shape, bg_colors.shape, bg_normals.shape)

                # ---- [M1+b 自有改动，Frosting 原版没有] ----
                # 用真正送进 Poisson 的表面采样点（fg_points / bg_points，即 fg_pcd / bg_pcd 的
                # 来源 tensor）分别估 D_fg / D_bg：随机子采样 <=50 万点，knn_points K=2 取平方
                # 距离的 10% 分位数，除以各自包围盒尺寸，再套 Frosting 的 D 公式。
                if use_surface_depth_estimate or only_report_depth:
                    _d_fg_s, _det_fg_s = compute_poisson_depth_from_points(
                        fg_points, cell_size_nn_distance_ratio=cell_size_nn_distance_ratio,
                        max_poisson_depth=max_poisson_depth, tag='fg_surface')
                    _d_bg_s, _det_bg_s = compute_poisson_depth_from_points(
                        bg_points, cell_size_nn_distance_ratio=cell_size_nn_distance_ratio,
                        max_poisson_depth=max_poisson_depth, tag='bg_surface')
                    CONSOLE.print("[M1+b] Poisson depth (fg, surface):", _d_fg_s, _det_fg_s)
                    CONSOLE.print("[M1+b] Poisson depth (bg, surface):", _d_bg_s, _det_bg_s)
                    extract_stats['depth_fg_surface'] = None if _d_fg_s is None else int(_d_fg_s)
                    extract_stats['depth_fg_surface_details'] = _det_fg_s
                    extract_stats['depth_bg_surface'] = None if _d_bg_s is None else int(_d_bg_s)
                    extract_stats['depth_bg_surface_details'] = _det_bg_s

                    if use_surface_depth_estimate:
                        if use_auto_poisson_depth and _d_fg_s is not None:
                            poisson_depth_fg = int(_d_fg_s)
                            if use_same_depth_for_bg:
                                poisson_depth_bg = int(_d_fg_s)
                        if use_auto_poisson_depth_bg and _d_bg_s is not None:
                            poisson_depth_bg = int(_d_bg_s)
                    extract_stats['poisson_depth_fg_used'] = int(poisson_depth_fg)
                    extract_stats['poisson_depth_bg_used'] = int(poisson_depth_bg)
                    extract_stats['poisson_depth_used'] = int(poisson_depth_fg)
                    with open(os.path.join(mesh_output_dir, 'extract_stats.json'), 'w') as _f:
                        json.dump(extract_stats, _f, indent=2)
                    if only_report_depth:
                        CONSOLE.print("[only_report_depth=True] Depths reported, skipping mesh extraction.")
                        return []
                CONSOLE.print("[M1+] Poisson depth used -> fg:", poisson_depth_fg, " bg:", poisson_depth_bg)
                
                # ---Compute foreground mesh---
                CONSOLE.print("\n-----Foreground mesh-----")
                if fg_points.shape[0] > 0:
                    CONSOLE.print("Computing points, colors and normals...")
                    fg_pcd = o3d.geometry.PointCloud()
                    fg_pcd.points = o3d.utility.Vector3dVector(fg_points.double().cpu().numpy())
                    fg_pcd.colors = o3d.utility.Vector3dVector(fg_colors.double().cpu().numpy())
                    fg_pcd.normals = o3d.utility.Vector3dVector(fg_normals.double().cpu().numpy())

                    # outliers removal
                    cl, ind = fg_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.)
                    CONSOLE.print("Cleaning Point Cloud...")
                    fg_pcd = fg_pcd.select_by_index(ind)

                    CONSOLE.print("Finished computing points, colors and normals.")

                    CONSOLE.print("Now computing mesh...")
                    o3d_fg_mesh, o3d_fg_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        fg_pcd, depth=poisson_depth_fg) #, width=0, scale=1.1, linear_fit=False)  # depth=10 should be the default value? 11 is good to (but it starts to make a big number of triangles)

                    if vertices_density_quantile > 0.:
                        CONSOLE.print("Removing vertices with low densities...")
                        vertices_to_remove = o3d_fg_densities < np.quantile(o3d_fg_densities, vertices_density_quantile)
                        o3d_fg_mesh.remove_vertices_by_mask(vertices_to_remove)
                else:
                    CONSOLE.print("\n[WARNING] Foreground is empty.")
                    o3d_fg_mesh = None
                
                # ---Compute background mesh---
                CONSOLE.print("\n-----Background mesh-----")
                if bg_points.shape[0] > 0:
                    CONSOLE.print("Computing points, colors and normals...")
                    bg_pcd = o3d.geometry.PointCloud()
                    bg_pcd.points = o3d.utility.Vector3dVector(bg_points.double().cpu().numpy())
                    bg_pcd.colors = o3d.utility.Vector3dVector(bg_colors.double().cpu().numpy())
                    bg_pcd.normals = o3d.utility.Vector3dVector(bg_normals.double().cpu().numpy())

                    # outliers removal
                    cl, ind = bg_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.)
                    CONSOLE.print("Cleaning Point Cloud...")
                    bg_pcd = bg_pcd.select_by_index(ind)

                    CONSOLE.print("Finished computing points, colors and normals.")

                    CONSOLE.print("Now computing mesh...")
                    o3d_bg_mesh, o3d_bg_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        bg_pcd, depth=poisson_depth_bg) #, width=0, scale=1.1, linear_fit=False)  # depth=10 should be the default value? 11 is good to (but it starts to make a big number of triangles)

                    if vertices_density_quantile > 0.:
                        CONSOLE.print("Removing vertices with low densities...")
                        vertices_to_remove = o3d_bg_densities < np.quantile(o3d_bg_densities, vertices_density_quantile)
                        o3d_bg_mesh.remove_vertices_by_mask(vertices_to_remove)
                else:
                    CONSOLE.print("\n[WARNING] Background is empty.")
                    o3d_bg_mesh = None
                
                CONSOLE.print("Finished computing meshes.")
                CONSOLE.print("Foreground mesh:", o3d_fg_mesh)
                CONSOLE.print("Background mesh:", o3d_bg_mesh)
                
                # ---Decimate and clean meshes---
                CONSOLE.print("\n-----Decimating and cleaning meshes-----")
                for decimation_target in decimation_targets:
                    CONSOLE.print("\nProcessing decimation target:", decimation_target)
                    if decimate_mesh:
                        if o3d_fg_mesh is not None:
                            CONSOLE.print("Decimating foreground mesh...")
                            decimated_o3d_fg_mesh = o3d_fg_mesh.simplify_quadric_decimation(decimation_target)
                            CONSOLE.print("Finished decimating foreground mesh.")
                        else:
                            decimated_o3d_fg_mesh = None
                            
                        if o3d_bg_mesh is not None:                            
                            CONSOLE.print("Decimating background mesh...")
                            decimated_o3d_bg_mesh = o3d_bg_mesh.simplify_quadric_decimation(decimation_target)
                            CONSOLE.print("Finished decimating background mesh.")
                        else:
                            decimated_o3d_bg_mesh = None

                    if clean_mesh:
                        CONSOLE.print("Cleaning mesh...")
                        if decimated_o3d_fg_mesh is not None:
                            decimated_o3d_fg_mesh.remove_duplicated_vertices()
                            decimated_o3d_fg_mesh.remove_degenerate_triangles()
                            decimated_o3d_fg_mesh.remove_duplicated_triangles()
                            decimated_o3d_fg_mesh.remove_non_manifold_edges()
                        
                        if decimated_o3d_bg_mesh is not None:
                            decimated_o3d_bg_mesh.remove_duplicated_vertices()
                            decimated_o3d_bg_mesh.remove_degenerate_triangles()
                            decimated_o3d_bg_mesh.remove_duplicated_triangles()
                            decimated_o3d_bg_mesh.remove_non_manifold_edges()
                    
                    if (decimated_o3d_fg_mesh is not None) and (decimated_o3d_bg_mesh is not None):
                        CONSOLE.print("Merging foreground and background meshes.")
                        decimated_o3d_mesh = decimated_o3d_fg_mesh + decimated_o3d_bg_mesh
                    elif decimated_o3d_fg_mesh is not None:
                        CONSOLE.print("Using foreground mesh only, since background mesh is empty.")
                        decimated_o3d_mesh = decimated_o3d_fg_mesh
                    elif decimated_o3d_bg_mesh is not None:
                        CONSOLE.print("Using background mesh only, since foreground mesh is empty.")
                        decimated_o3d_mesh = decimated_o3d_bg_mesh
                    else:
                        raise ValueError("Both foreground and background meshes are empty. Please provide a valid bounding box for the scene.")
                    
                    # Project Mesh on extracted surface points for better details
                    if project_mesh_on_surface_points:
                        CONSOLE.print("Projecting mesh on surface points to recover better details...")
                        mesh_verts = torch.tensor(np.asarray(decimated_o3d_mesh.vertices), device=sugar.device, dtype=torch.float32)
                        mesh_faces = torch.tensor(np.asarray(decimated_o3d_mesh.triangles), device=sugar.device, dtype=torch.int64)

                        proj_knn_idx = knn_points(
                            mesh_verts[None], 
                            surface_points[None], 
                            K=1,
                        ).idx[0][..., 0]
                        
                        new_mesh_verts = surface_points[proj_knn_idx]
                        new_mesh_faces = mesh_faces
                        new_mesh_colors = surface_colors[proj_knn_idx]
                        
                        decimated_o3d_mesh = o3d.geometry.TriangleMesh()
                        decimated_o3d_mesh.vertices = o3d.utility.Vector3dVector(new_mesh_verts.cpu().numpy())
                        decimated_o3d_mesh.triangles = o3d.utility.Vector3iVector(new_mesh_faces.cpu().numpy())
                        decimated_o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(new_mesh_colors.cpu().numpy())
                        decimated_o3d_mesh.compute_vertex_normals()
                        
                        # ADDED
                        CONSOLE.print("Cleaning projected mesh...")
                        decimated_o3d_mesh.remove_duplicated_vertices()
                        decimated_o3d_mesh.remove_degenerate_triangles()
                        decimated_o3d_mesh.remove_duplicated_triangles()
                        decimated_o3d_mesh.remove_non_manifold_edges()
                        CONSOLE.print("Projection done.")
                    
                    if use_vanilla_3dgs:
                        sugar_mesh_path = 'sugarmesh_vanilla3dgs_levelZZ_decimAA.ply'
                    else:
                        sugar_mesh_path = 'sugarmesh_' + sugar_checkpoint_path.split('/')[-2].replace('sugarcoarse_', '') + '_levelZZ_decimAA.ply'
                    sugar_mesh_path = sugar_mesh_path.replace(
                        'ZZ', str(surface_level).replace('.', '')
                        ).replace(
                            'AA', str(decimation_target).replace('.', '')
                            )
                    sugar_mesh_path = os.path.join(mesh_output_dir, sugar_mesh_path)
                    o3d.io.write_triangle_mesh(sugar_mesh_path, decimated_o3d_mesh, write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                    CONSOLE.print("Mesh saved at", sugar_mesh_path)
                    all_sugar_mesh_paths.append(sugar_mesh_path)
                    
        else:
            CONSOLE.print("\nWARNING: Using centers of gaussians to extract mesh.")
            CONSOLE.print("Results will look bad, this is not the best way to extract a mesh.")
            CONSOLE.print("You should use this option only for ablation.")
            
            with torch.no_grad():
                surface_points = sugar.points
                surface_colors = SH2RGB(sugar._sh_coordinates_dc[:, 0, :])
                surface_normals = sugar.get_normals(estimate_from_points=True)
                
                if use_custom_bbox:
                    CONSOLE.print("Using provided bounding box.")
                    fg_bbox_min_tensor = torch.tensor(fg_bbox_min).to(sugar.device)
                    fg_bbox_max_tensor = torch.tensor(fg_bbox_max).to(sugar.device)
                else:
                    CONSOLE.print("Using default, camera based bounding box.")
                    fg_bbox_min_tensor = - fg_bbox_factor * sugar.get_cameras_spatial_extent() * torch.ones(1, 3, device=sugar.device)
                    fg_bbox_max_tensor = fg_bbox_factor * sugar.get_cameras_spatial_extent() * torch.ones(1, 3, device=sugar.device)

                points_idx = torch.arange(len(surface_points))
                fg_mask = (surface_points[points_idx] > fg_bbox_min_tensor).all(dim=-1) * (surface_points[points_idx] < fg_bbox_max_tensor).all(dim=-1)
                bg_mask = (surface_points[points_idx].abs().max(dim=-1)[0] < bg_bbox_factor * sugar.get_cameras_spatial_extent()) * ~fg_mask

                fg_points = surface_points[points_idx][fg_mask]
                fg_colors = surface_colors[points_idx][fg_mask]
                fg_normals = surface_normals[points_idx][fg_mask]

                bg_points = surface_points[points_idx][bg_mask]
                bg_colors = surface_colors[points_idx][bg_mask]
                bg_normals = surface_normals[points_idx][bg_mask]

                CONSOLE.print("Foreground points:", fg_points.shape, fg_colors.shape, fg_normals.shape)
                CONSOLE.print("Background points:", bg_points.shape, bg_colors.shape, bg_normals.shape)

                # ---- [M1+b 自有改动，Frosting 原版没有] ----
                # 用真正送进 Poisson 的表面采样点（fg_points / bg_points，即 fg_pcd / bg_pcd 的
                # 来源 tensor）分别估 D_fg / D_bg：随机子采样 <=50 万点，knn_points K=2 取平方
                # 距离的 10% 分位数，除以各自包围盒尺寸，再套 Frosting 的 D 公式。
                if use_surface_depth_estimate or only_report_depth:
                    _d_fg_s, _det_fg_s = compute_poisson_depth_from_points(
                        fg_points, cell_size_nn_distance_ratio=cell_size_nn_distance_ratio,
                        max_poisson_depth=max_poisson_depth, tag='fg_surface')
                    _d_bg_s, _det_bg_s = compute_poisson_depth_from_points(
                        bg_points, cell_size_nn_distance_ratio=cell_size_nn_distance_ratio,
                        max_poisson_depth=max_poisson_depth, tag='bg_surface')
                    CONSOLE.print("[M1+b] Poisson depth (fg, surface):", _d_fg_s, _det_fg_s)
                    CONSOLE.print("[M1+b] Poisson depth (bg, surface):", _d_bg_s, _det_bg_s)
                    extract_stats['depth_fg_surface'] = None if _d_fg_s is None else int(_d_fg_s)
                    extract_stats['depth_fg_surface_details'] = _det_fg_s
                    extract_stats['depth_bg_surface'] = None if _d_bg_s is None else int(_d_bg_s)
                    extract_stats['depth_bg_surface_details'] = _det_bg_s

                    if use_surface_depth_estimate:
                        if use_auto_poisson_depth and _d_fg_s is not None:
                            poisson_depth_fg = int(_d_fg_s)
                            if use_same_depth_for_bg:
                                poisson_depth_bg = int(_d_fg_s)
                        if use_auto_poisson_depth_bg and _d_bg_s is not None:
                            poisson_depth_bg = int(_d_bg_s)
                    extract_stats['poisson_depth_fg_used'] = int(poisson_depth_fg)
                    extract_stats['poisson_depth_bg_used'] = int(poisson_depth_bg)
                    extract_stats['poisson_depth_used'] = int(poisson_depth_fg)
                    with open(os.path.join(mesh_output_dir, 'extract_stats.json'), 'w') as _f:
                        json.dump(extract_stats, _f, indent=2)
                    if only_report_depth:
                        CONSOLE.print("[only_report_depth=True] Depths reported, skipping mesh extraction.")
                        return []
                CONSOLE.print("[M1+] Poisson depth used -> fg:", poisson_depth_fg, " bg:", poisson_depth_bg)
                
                # ---Compute foreground mesh---
                CONSOLE.print("\n-----Foreground mesh-----")
                CONSOLE.print("Computing points, colors and normals...")
                fg_pcd = o3d.geometry.PointCloud()
                fg_pcd.points = o3d.utility.Vector3dVector(fg_points.double().cpu().numpy())
                fg_pcd.colors = o3d.utility.Vector3dVector(fg_colors.double().cpu().numpy())
                fg_pcd.normals = o3d.utility.Vector3dVector(fg_normals.double().cpu().numpy())

                # outliers removal
                cl, ind = fg_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.)
                CONSOLE.print("Cleaning Point Cloud...")
                fg_pcd = fg_pcd.select_by_index(ind)

                CONSOLE.print("Finished computing points, colors and normals.")

                CONSOLE.print("Now computing mesh...")
                o3d_fg_mesh, o3d_fg_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    fg_pcd, depth=poisson_depth_fg) #, width=0, scale=1.1, linear_fit=False)  # depth=10 should be the default value? 11 is good to (but it starts to make a big number of triangles)

                if vertices_density_quantile > 0.:
                    CONSOLE.print("Removing vertices with low densities...")
                    vertices_to_remove = o3d_fg_densities < np.quantile(o3d_fg_densities, vertices_density_quantile)
                    o3d_fg_mesh.remove_vertices_by_mask(vertices_to_remove)
                
                # ---Compute background mesh---
                if bg_points.shape[0] > 0:
                    CONSOLE.print("\n-----Background mesh-----")
                    CONSOLE.print("Computing points, colors and normals...")
                    bg_pcd = o3d.geometry.PointCloud()
                    bg_pcd.points = o3d.utility.Vector3dVector(bg_points.double().cpu().numpy())
                    bg_pcd.colors = o3d.utility.Vector3dVector(bg_colors.double().cpu().numpy())
                    bg_pcd.normals = o3d.utility.Vector3dVector(bg_normals.double().cpu().numpy())

                    # outliers removal
                    cl, ind = bg_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.)
                    CONSOLE.print("Cleaning Point Cloud...")
                    bg_pcd = bg_pcd.select_by_index(ind)

                    CONSOLE.print("Finished computing points, colors and normals.")

                    CONSOLE.print("Now computing mesh...")
                    o3d_bg_mesh, o3d_bg_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        bg_pcd, depth=poisson_depth_bg) #, width=0, scale=1.1, linear_fit=False)  # depth=10 should be the default value? 11 is good to (but it starts to make a big number of triangles)

                    if vertices_density_quantile > 0.:
                        CONSOLE.print("Removing vertices with low densities...")    
                        vertices_to_remove = o3d_bg_densities < np.quantile(o3d_bg_densities, vertices_density_quantile)
                        o3d_bg_mesh.remove_vertices_by_mask(vertices_to_remove)
                else:
                    o3d_bg_mesh = None
                
                CONSOLE.print("Finished computing meshes.")
                CONSOLE.print("Foreground mesh:", o3d_fg_mesh)
                CONSOLE.print("Background mesh:", o3d_bg_mesh)
                
                # ---Decimate and clean meshes---
                CONSOLE.print("\n-----Decimating and cleaning meshes-----")
                for decimation_target in decimation_targets:
                    CONSOLE.print("\nProcessing decimation target:", decimation_target)
                    if decimate_mesh:
                        CONSOLE.print("Decimating foreground mesh...")
                        decimated_o3d_fg_mesh = o3d_fg_mesh.simplify_quadric_decimation(decimation_target)
                        CONSOLE.print("Finished decimating foreground mesh.")
                        
                        if o3d_bg_mesh is not None:                                                        
                            CONSOLE.print("Decimating background mesh...")
                            decimated_o3d_bg_mesh = o3d_bg_mesh.simplify_quadric_decimation(decimation_target)
                            CONSOLE.print("Finished decimating background mesh.")

                    if clean_mesh:
                        CONSOLE.print("Cleaning mesh...")
                        decimated_o3d_fg_mesh.remove_duplicated_vertices()
                        decimated_o3d_fg_mesh.remove_degenerate_triangles()
                        decimated_o3d_fg_mesh.remove_duplicated_triangles()
                        decimated_o3d_fg_mesh.remove_non_manifold_edges()
                        
                        if decimated_o3d_bg_mesh is not None:
                            decimated_o3d_bg_mesh.remove_duplicated_vertices()
                            decimated_o3d_bg_mesh.remove_degenerate_triangles()
                            decimated_o3d_bg_mesh.remove_duplicated_triangles()
                            decimated_o3d_bg_mesh.remove_non_manifold_edges()
                        
                    if decimated_o3d_bg_mesh is not None:
                        decimated_o3d_mesh = decimated_o3d_fg_mesh + decimated_o3d_bg_mesh
                    else:
                        decimated_o3d_mesh = decimated_o3d_fg_mesh
                    
                    if use_vanilla_3dgs:
                        sugar_mesh_path = 'sugarmesh_vanilla3dgs_poissoncenters_decimAA.ply'
                    else:
                        sugar_mesh_path = 'sugarmesh_' + sugar_checkpoint_path.split('/')[-2].replace('sugarcoarse_', '') + '_poissoncenters_decimAA.ply'
                    sugar_mesh_path = sugar_mesh_path.replace(
                            'AA', str(decimation_target).replace('.', '')
                            )
                    sugar_mesh_path = os.path.join(mesh_output_dir, sugar_mesh_path)
                    o3d.io.write_triangle_mesh(sugar_mesh_path, decimated_o3d_mesh, write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                    CONSOLE.print("Mesh saved at", sugar_mesh_path)
                    all_sugar_mesh_paths.append(sugar_mesh_path)
    else:
        CONSOLE.print("\nWARNING: Using marching cubes to extract mesh.")
        import mcubes
        
        sugar.reset_neighbors(knn_to_track=16)
        
        resolution = 512
        surface_level = surface_levels[0]
        decimation_target = decimation_targets[0]

        # Foreground mesh
        CONSOLE.print("\n-----Foreground mesh-----")
        X = torch.linspace(-1, 1, resolution) * sugar.get_cameras_spatial_extent()
        Y = torch.linspace(-1, 1, resolution) * sugar.get_cameras_spatial_extent()
        Z = torch.linspace(-1, 1, resolution) * sugar.get_cameras_spatial_extent()

        xx, yy, zz = torch.meshgrid(X, Y, Z)
        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(sugar.device)

        xx.shape, yy.shape, zz.shape, pts.shape

        n_pts_per_pass = 2_000_000

        densities = torch.zeros(0, device=sugar.device)

        CONSOLE.print("Computing densities...")
        with torch.no_grad():
            for i in range(0, len(pts), n_pts_per_pass):
                print("\nPts:", i, 'to', i+n_pts_per_pass)
                pts_i = pts[i:i+n_pts_per_pass]
                densities_i = sugar.compute_density(pts_i)
                densities = torch.cat([densities, densities_i], dim=0)
            densities = densities.reshape(resolution, resolution, resolution)
            CONSOLE.print("Finished computing densities.")
            
            density_th = surface_levels[0]  # 1.
            CONSOLE.print(f"Computing mesh for surface level {density_th}...")
            vertices, triangles = mcubes.marching_cubes(densities.cpu().numpy(), density_th)
            verts = -sugar.get_cameras_spatial_extent() + (torch.tensor(vertices) / resolution) * 2 * sugar.get_cameras_spatial_extent()
            faces = torch.tensor(triangles.tolist())
            closest_gaussians = sugar.get_gaussians_closest_to_samples(verts.float().to(sugar.device))
            verts_colors = SH2RGB(sugar._sh_coordinates_dc[closest_gaussians[:, 0]][:, 0, :])
            
            mc_mesh = o3d.geometry.TriangleMesh()
            mc_mesh.vertices = o3d.utility.Vector3dVector(verts.cpu().numpy())
            mc_mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
            mc_mesh.vertex_colors = o3d.utility.Vector3dVector(verts_colors.cpu().numpy())
            mc_mesh.compute_vertex_normals()
            CONSOLE.print("Finished computing mesh.")
         
        # Background mesh   
        CONSOLE.print("\n-----Background mesh-----")
        X = torch.linspace(-1, 1, resolution) * 4 * sugar.get_cameras_spatial_extent()
        Y = torch.linspace(-1, 1, resolution) * 4 * sugar.get_cameras_spatial_extent()
        Z = torch.linspace(-1, 1, resolution) * 4 * sugar.get_cameras_spatial_extent()

        xx, yy, zz = torch.meshgrid(X, Y, Z)
        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(sugar.device)

        xx.shape, yy.shape, zz.shape, pts.shape

        n_pts_per_pass = 2_000_000

        densities = torch.zeros(0, device=sugar.device)

        CONSOLE.print("Computing densities...")
        with torch.no_grad():
            for i in range(0, len(pts), n_pts_per_pass):
                print("\nPts:", i, 'to', i+n_pts_per_pass)
                pts_i = pts[i:i+n_pts_per_pass]
                densities_i = sugar.compute_density(pts_i)
                densities = torch.cat([densities, densities_i], dim=0)
            CONSOLE.print("Finished computing densities.")
            
            # Removing pts in foreground
            densities[(pts > -sugar.get_cameras_spatial_extent()).all(dim=-1) * (pts < sugar.get_cameras_spatial_extent()).all(dim=-1)] = 0.
            densities = densities.reshape(resolution, resolution, resolution)

            density_th = surface_levels[0]  # 1.
            CONSOLE.print(f"Computing mesh for surface level {density_th}...")
            bg_vertices, bg_triangles = mcubes.marching_cubes(densities.cpu().numpy(), density_th)
            bg_verts = - 4 * sugar.get_cameras_spatial_extent() + (torch.tensor(bg_vertices) / resolution) * 2 * 4 * sugar.get_cameras_spatial_extent()
            bg_faces = torch.tensor(bg_triangles.tolist())
            closest_gaussians = sugar.get_gaussians_closest_to_samples(bg_verts.float().to(sugar.device))
            bg_verts_colors = SH2RGB(sugar._sh_coordinates_dc[closest_gaussians[:, 0]][:, 0, :])
            
            bg_mc_mesh = o3d.geometry.TriangleMesh()
            bg_mc_mesh.vertices = o3d.utility.Vector3dVector(bg_verts.cpu().numpy())
            bg_mc_mesh.triangles = o3d.utility.Vector3iVector(bg_faces.cpu().numpy())
            bg_mc_mesh.vertex_colors = o3d.utility.Vector3dVector(bg_verts_colors.cpu().numpy())
            bg_mc_mesh.compute_vertex_normals()
            CONSOLE.print("Finished computing mesh.")
            
        # Decimate and clean meshes
        decimate_mesh = True
        decimation_target = decimation_targets[0]

        if decimate_mesh:
            print(f"Decimating mesh to target {decimation_target}...")
            decimated_o3d_fg_mesh = mc_mesh.simplify_quadric_decimation(decimation_target)
            print("Finished decimating mesh.")
            
            print("Decimating mesh...")
            decimated_o3d_bg_mesh = bg_mc_mesh.simplify_quadric_decimation(decimation_target)
            print("Finished decimating mesh.")
        else:
            decimated_o3d_fg_mesh = mc_mesh
            decimated_o3d_bg_mesh = bg_mc_mesh
            
        clean_mesh = True
        if clean_mesh:
            decimated_o3d_fg_mesh.remove_degenerate_triangles()
            decimated_o3d_fg_mesh.remove_duplicated_triangles()
            decimated_o3d_fg_mesh.remove_duplicated_vertices()
            decimated_o3d_fg_mesh.remove_non_manifold_edges()
            
            decimated_o3d_bg_mesh.remove_degenerate_triangles()
            decimated_o3d_bg_mesh.remove_duplicated_triangles()
            decimated_o3d_bg_mesh.remove_duplicated_vertices()
            decimated_o3d_bg_mesh.remove_non_manifold_edges()
            
        decimated_o3d_mesh = decimated_o3d_fg_mesh + decimated_o3d_bg_mesh
        if use_vanilla_3dgs:
            sugar_mesh_path = 'sugarmesh_vanilla3dgsmarchingcubes_levelZZ_decimAA.ply'
        else:
            sugar_mesh_path = 'sugarmesh_' + sugar_checkpoint_path.split('/')[-2].replace('sugarcoarse_', '') + 'marchingcubes_levelZZ_decimAA.ply'
        sugar_mesh_path = sugar_mesh_path.replace(
            'ZZ', str(surface_level).replace('.', '')
            ).replace(
                'AA', str(decimation_target).replace('.', '')
                )
        sugar_mesh_path = os.path.join(mesh_output_dir, sugar_mesh_path)
        o3d.io.write_triangle_mesh(sugar_mesh_path, decimated_o3d_mesh, write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
        CONSOLE.print("Mesh saved at", sugar_mesh_path)
        all_sugar_mesh_paths.append(sugar_mesh_path)

    # [ADDED] final extraction stats (mesh paths + wall clock)
    extract_stats['mesh_paths'] = list(all_sugar_mesh_paths)
    extract_stats['extraction_wall_clock_s'] = float(time.time() - _extract_t0)
    with open(os.path.join(mesh_output_dir, 'extract_stats.json'), 'w') as _f:
        json.dump(extract_stats, _f, indent=2)

    return all_sugar_mesh_paths