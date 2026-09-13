import argparse
from sugar_utils.general_utils import str2bool
from sugar_extractors.coarse_mesh import extract_mesh_from_coarse_sugar

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to extract a mesh from a coarse SuGaR scene.')
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='path to the scene data to use.')
    parser.add_argument('-c', '--checkpoint_path', 
                        type=str, 
                        help='path to the vanilla 3D Gaussian Splatting Checkpoint to load.')
    parser.add_argument('-i', '--iteration_to_load', 
                        type=int, default=7000, 
                        help='iteration to load.')
    
    parser.add_argument('-m', '--coarse_model_path', type=str, default=None, help='')
    
    parser.add_argument('-l', '--surface_level', type=float, default=None, 
                        help='Surface level to extract the mesh at. If None, will extract levels 0.1, 0.3 and 0.5')
    parser.add_argument('-d', '--decimation_target', type=int, default=None, 
                        help='Target number of vertices to decimate the mesh to. If None, will decimate to 200_000 and 1_000_000.')
    parser.add_argument('--project_mesh_on_surface_points', type=str2bool, default=True, 
                        help='If True, project the mesh on the surface points for better details.')
    
    parser.add_argument('-o', '--mesh_output_dir',
                        type=str, default=None, 
                        help='path to the output directory.')
    
    parser.add_argument('-b', '--bboxmin', type=str, default=None, help='Min coordinates to use for foreground.')
    parser.add_argument('-B', '--bboxmax', type=str, default=None, help='Max coordinates to use for foreground.')
    parser.add_argument('--center_bbox', type=str2bool, default=True, help='If True, center the bounding box. Default is True.')
    
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')
    parser.add_argument('--use_centers_to_extract_mesh', type=str2bool, default=False, 
                        help='If True, just use centers of the gaussians to extract mesh.')
    parser.add_argument('--use_marching_cubes', type=str2bool, default=False, 
                        help='If True, use marching cubes to extract mesh.')
    parser.add_argument('--use_vanilla_3dgs', type=str2bool, default=False, 
                        help='If True, use vanilla 3DGS to extract mesh.')

    # ----- [ADDED] Poisson reconstruction knobs (ported from Gaussian Frosting) -----
    # Frosting: frosting_extractors/coarse_shell.py:17-49 (compute_optimal_poisson_depth)
    #           train_full_pipeline.py:31-33 (--poisson_depth / --cleaning_quantile)
    parser.add_argument('--poisson_depth', type=str, default='10',
                        help="Octree depth of the Poisson surface reconstruction. "
                             "An integer (SuGaR's original hard-coded value is 10), or 'auto' "
                             "(equivalently '-1') to compute it automatically from the SuGaR model "
                             "with Frosting's compute_optimal_poisson_depth.")
    parser.add_argument('--vertices_density_quantile', type=float, default=0.1,
                        help="Quantile of the Poisson vertex densities below which vertices are removed "
                             "(Frosting calls this --cleaning_quantile). 0.1 for most real scenes, "
                             "0. works well for most synthetic scenes. 0. disables the cleaning.")
    parser.add_argument('--cell_size_nn_distance_ratio', type=float, default=100.,
                        help="Constant of Frosting's automatic depth formula "
                             "D = min(floor(-log2(ratio * d_q)), 10). Only used when --poisson_depth auto. "
                             "Larger ratio -> smaller depth.")
    parser.add_argument('--only_report_depth', type=str2bool, default=False,
                        help="If True, only load the model, compute + print + dump the automatic Poisson "
                             "depth to <mesh_output_dir>/extract_stats.json, then exit without extracting.")
    
    args = parser.parse_args()
    
    # Call function
    extract_mesh_from_coarse_sugar(args)
    