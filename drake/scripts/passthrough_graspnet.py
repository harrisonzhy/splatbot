# %%
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import open3d as o3d
import torch

from scipy.spatial import KDTree
from utils.minimal_grasp_utils import *

# modify Python's Path
sys.path.insert(1, f"{Path(__file__).parent.parent.resolve()}")
print("Imports successful")


splat_home = os.getenv('SPLAT_HOME')
splat_img = f"{splat_home}/cloud_grasps"
obj = "vase"

downsample_pcd = False

def passthrough(object_points, object_colors):

    normal_plane = [0, 0, 1]

    cfgs = Args()

    # checkpoint path
    cfgs.checkpoint_path = Path(
        f"{Path(__file__).parent.parent.resolve()}/graspnet/model_checkpoints/checkpoint-rs.tar"
    )

     # number of grasps to visualize
    num_vis_grasp = 5  # default: 50

    # color to display each grasp
    grasp_group_color = np.array([[0, 1, 0]])

    # output directory
    grasp_output_dir = Path(f"{splat_img}/grasp_results/{obj}/")

    # # # # #
    # # # # # Proposed Grasps from GraspNet
    # # # # #

    # run GraspNet inference
    cand_grasps, pcd = demo(object_points, object_colors, cfgs)

    cand_grasps.nms()
    cand_grasps.sort_by_score()
    gg = cand_grasps

    # option to display the axes and gridlines
    showaxes_grid = False

    print("*" * 50)
    print("GraspNet without Affordance")
    print("*" * 50)

    # visualize the grasps
    fig_grasp_wout_aff = visualize_grasps(
        gg,
        pcd=pcd,
        num_vis_grasp=num_vis_grasp,
        grasp_group_color=grasp_group_color,
        showaxes_grid=showaxes_grid,
        image_path=f"{splat_img}/{obj}_grasps_without_affordance_scaled.png"
    )


    # # # # #
    # # # # # Incorporate Affordance
    # # # # #

    print("*" * 50)
    print("GraspNet with Affordance")
    print("*" * 50)

    # number of neighbors in query
    n_gs_neighbors = 1

    # radius for query
    k_tree_rad = 5e-3

    # construct KD-tree
    env_tree = KDTree(data=np.asarray(object_points))

    # affordance for the environment
    env_affordance = torch.tensor(np.ones(len(object_points)))
    env_pcd_mask = None

    if env_pcd_mask is not None:
        # affordance for the environment
        env_affordance = env_affordance[env_pcd_mask]

    # affordance-aware grasps
    ranked_grasps, comp_score = rank_grasps(
        cand_grasps,
        env_tree=env_tree,
        env_affordance=env_affordance,
        n_neighbors=n_gs_neighbors,
    )

    # visualize the grasps
    fig_grasp_wout_aff_wout_pp = visualize_grasps(
        ranked_grasps,
        pcd=pcd,
        num_vis_grasp=num_vis_grasp,
        grasp_group_color=grasp_group_color,
        showaxes_grid=showaxes_grid,
        image_path=f"{splat_img}/{obj}_grasps_with_affordance_scaled.png"
    )

    print("*" * 50)
    print("GraspNet with Affordance and Heuristics")
    print("*" * 50)

    # # # # #
    # # # # # Reorient Proposed Grasps
    # # # # #

    # reoriented grasps
    reor_gp = reorient_grasps(ranked_grasps[:num_vis_grasp], normal_plane)

    fig_grasp_w_aff_w_pp = visualize_grasps(
        reor_gp,
        pcd=pcd,
        num_vis_grasp=num_vis_grasp,
        grasp_group_color=grasp_group_color,
        showaxes_grid=showaxes_grid,
        image_path=f"{splat_img}/{obj}_grasps_with_affordance_and_heuristics_scaled.png"
    )

    # # # # #
    # # # # # Save the Grasps.
    # # # # #

    # filename
    pose_filename = Path(f"{grasp_output_dir}/{obj}_graspnet_scaled.npy")

    # number of grasps to save
    num_grasps_save = num_vis_grasp

    # save the proposed grasps
    save_grasps(
        filename=pose_filename, grasps=cand_grasps, num_grasps_save=num_grasps_save
    )

    # filename
    pose_filename = Path(f"{grasp_output_dir}/{obj}_w_affordance_scaled.npy")

    # save the proposed grasps
    save_grasps(
        filename=pose_filename,
        grasps=ranked_grasps,
        num_grasps_save=num_grasps_save,
    )

    # filename
    pose_filename = Path(f"{grasp_output_dir}/{obj}_w_post_processing_scaled.npy")

    # save the proposed grasps
    save_grasps(
        filename=pose_filename, grasps=reor_gp, num_grasps_save=num_grasps_save
    )

if __name__ == "__main__":

    point_cloud_path = f"{splat_home}/drake/vase_100k.obj"

    mesh = o3d.io.read_triangle_mesh(point_cloud_path)
    
    if not mesh.is_empty():
        scale = 0.1

        points = np.asarray(mesh.vertices) * scale
        colors = np.asarray(mesh.vertex_colors)
        
        print("Point cloud loaded successfully.")

        if downsample_pcd:
            points = mesh.uniform_down_sample(every_k_points=10)

        print(len(points))
        print(len(colors))

        passthrough(points, colors)

