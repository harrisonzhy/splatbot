# # # # #
# # # # # Grasp Generation
# # # # #

import argparse

from graspnet.graspnetAPI.graspnetAPI.grasp import GraspGroup
from graspnet.graspnet_baseline.models.graspnet import GraspNet, pred_decode
from graspnet.graspnet_baseline.utils.collision_detector import (
    ModelFreeCollisionDetector,
)
import scipy.io as scio
from PIL import Image
import open3d as o3d
from pathlib import Path
import numpy as np
import torch
import gc

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class Args(argparse.Namespace):
    # Model checkpoint path
    checkpoint_path = Path("")

    # Point Number [default: 20000]
    num_point = 20000

    # View Number [default: 300]
    num_view = 300

    # Collision Threshold in collision detection [default: 0.01]
    collision_thresh = 0.01

    # Voxel Size to process point clouds before collision detection [default: 0.01
    voxel_size = 0.01


def get_affordance(tree, points, env_affordance, n_neighbors=1):
    """
    Computes the affordance for each point in points in the environment.
    """
    # nearest-neighbors
    dist, ind = tree.query(points, k=n_neighbors)

    # alternative
    # ind = tree.query_ball_point(points, r=k_tree_rad)

    # compute affordance
    afford = env_affordance[ind].detach().cpu().numpy()

    return afford


def rank_grasps(grasps, env_tree, env_affordance, n_neighbors=1):
    """
    Utilizes the composite grasp-affordance metric to rank candidate grasps.
    """
    # grasp points
    grasp_pts = []

    # grasp score from GraspNet
    grasp_score = []

    for gp in grasps:
        # grasp points
        grasp_pts.append(gp.translation)

        # grasp score from grasp net
        grasp_score.append(gp.score)

    grasp_pts = np.array(grasp_pts)
    grasp_score = np.array(grasp_score)

    # compute the affordance
    grasp_affordance = get_affordance(
        env_tree, grasp_pts, env_affordance, n_neighbors=n_neighbors
    )

    # compute the composite metric
    comp_score = grasp_affordance.squeeze()

    # Alternative:
    # comp_score = sigmoid(grasp_score) * grasp_affordance.squeeze()

    # ordering
    rank_order = np.argsort(comp_score)[::-1]

    # rerank grasps
    return grasps[rank_order], comp_score[rank_order]


def get_net(cfgs):
    # Init the model
    net = GraspNet(
        input_feature_dim=0,
        num_view=cfgs.num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint["epoch"]
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net


def get_and_process_data(cloud_masked, color_masked, cfgs):
    # upsample or downsample the point cloud
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(
            len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True
        )
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(float))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device).float()
    end_points["point_clouds"] = cloud_sampled
    end_points["cloud_colors"] = color_sampled

    # point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_masked)
    pcd.colors = o3d.utility.Vector3dVector(color_masked)

    return end_points, pcd


def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg


def collision_detection(gg, cloud, cfgs):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(
        gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh
    )
    gg = gg[~collision_mask]
    return gg


def vis_grasps(gg, cloud, num_vis_grasp, grasp_group_color=np.array([[0, 1, 0]]), image_path=None):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:num_vis_grasp]
    grippers = gg.to_open3d_geometry_list()
    # refine visualization
    grippers_pcd = []
    for gp in grippers:
        gp.paint_uniform_color(grasp_group_color.T)
        grippers_pcd.append(gp.sample_points_uniformly(number_of_points=4000))
    draw_plotly([cloud, *grippers, *grippers_pcd], image_path=image_path)


def demo(
    cloud_masked,
    color_masked,
    cfgs,
    num_vis_grasp=50,
    grasp_group_color=np.array([[0, 1, 0]]),
    image_path=None,
):
    net = get_net(cfgs)
    end_points, cloud = get_and_process_data(cloud_masked, color_masked, cfgs)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points), cfgs)

    del net
    gc.collect()
    torch.cuda.empty_cache()

    return gg, cloud

import plotly.graph_objects as go
import plotly.io as pio


def get_plotly_fig(geometry_list,
                   width=600,
                   height=400,
                   mesh_show_wireframe=False,
                   point_sample_factor=1,
                   front=None,
                   lookat=None,
                   up=None,
                   zoom=4.0):
    graph_objects = get_graph_objects(geometry_list, mesh_show_wireframe,
                                      point_sample_factor)
    geometry_center = get_geometry_center(geometry_list)
    max_bound = get_max_bound(geometry_list)
    # adjust camera to plotly-style
    if up is not None:
        plotly_up = dict(x=up[0], y=up[1], z=up[2])
    else:
        plotly_up = dict(x=0, y=0, z=1)

    if lookat is not None:
        lookat = [
            (i - j) / k for i, j, k in zip(lookat, geometry_center, max_bound)
        ]
        plotly_center = dict(x=lookat[0], y=lookat[1], z=lookat[2])
    else:
        plotly_center = dict(x=0, y=0, z=0)

    if front is not None:
        normalize_factor = np.sqrt(np.abs(np.sum(front)))
        front = [i / normalize_factor for i in front]
        plotly_eye = dict(x=zoom * 5 * front[0] + plotly_center['x'],
                          y=zoom * 5 * front[1] + plotly_center['y'],
                          z=zoom * 5 * front[2] + plotly_center['z'])
    else:
        plotly_eye = None

    camera = dict(up=plotly_up, center=plotly_center, eye=plotly_eye)
    fig = go.Figure(data=graph_objects,
                    layout=dict(
                        showlegend=False,
                        width=width,
                        height=height,
                        margin=dict(
                            l=0,
                            r=0,
                            b=0,
                            t=0,
                        ),
                        scene_camera=camera,
                        yaxis=dict(range=[-1, 1]),
                        xaxis=dict(range=[-1, 1]),
                    ))
    return fig


def get_point_object(geometry, point_sample_factor=1):
    points = np.asarray(geometry.points)
    colors = None
    if geometry.has_colors():
        colors = np.asarray(geometry.colors)
    elif geometry.has_normals():
        colors = (0.5, 0.5, 0.5) + np.asarray(geometry.normals) * 0.5
    else:
        geometry.paint_uniform_color((1.0, 0.0, 0.0))
        colors = np.asarray(geometry.colors)
    if (point_sample_factor > 0 and point_sample_factor < 1):
        indices = np.random.choice(len(points),
                                   (int)(len(points) * point_sample_factor),
                                   replace=False)
        points = points[indices]
        colors = colors[indices]
    scatter_3d = go.Scatter3d(x=points[:, 0],
                              y=points[:, 1],
                              z=points[:, 2],
                              mode='markers',
                              marker=dict(size=1, color=colors))
    return scatter_3d


def get_mesh_object(geometry,):
    pl_mygrey = [0, 'rgb(153, 153, 153)'], [1., 'rgb(255,255,255)']
    triangles = np.asarray(geometry.triangles)
    vertices = np.asarray(geometry.vertices)

    mesh_3d = go.Mesh3d(x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=triangles[:, 0],
                        j=triangles[:, 1],
                        k=triangles[:, 2],
                        flatshading=True,
                        colorscale=pl_mygrey,
                        intensity=vertices[:, 0],
                        lighting=dict(ambient=0.18,
                                      diffuse=1,
                                      fresnel=0.1,
                                      specular=1,
                                      roughness=0.05,
                                      facenormalsepsilon=1e-15,
                                      vertexnormalsepsilon=1e-15),
                        lightposition=dict(x=100, y=200, z=0))
    return mesh_3d


def get_wireframe_object(geometry):
    triangles = np.asarray(geometry.triangles)
    vertices = np.asarray(geometry.vertices)
    x = []
    y = []
    z = []
    tri_points = np.asarray(vertices)[triangles]
    for point in tri_points:
        x.extend([point[k % 3][0] for k in range(4)] + [None])
        y.extend([point[k % 3][1] for k in range(4)] + [None])
        z.extend([point[k % 3][2] for k in range(4)] + [None])
    wireframe = go.Scatter3d(x=x,
                             y=y,
                             z=z,
                             mode='lines',
                             line=dict(color='rgb(70,70,70)', width=1))
    return wireframe


def get_lineset_object(geometry):
    x = []
    y = []
    z = []
    line_points = np.asarray(geometry.points)[np.asarray(geometry.lines)]
    for point in line_points:
        x.extend([point[k % 2][0] for k in range(2)] + [None])
        y.extend([point[k % 2][1] for k in range(2)] + [None])
        z.extend([point[k % 2][2] for k in range(2)] + [None])
    line_3d = go.Scatter3d(x=x, y=y, z=z, mode='lines')
    return line_3d


def get_graph_objects(geometry_list,
                      mesh_show_wireframe=False,
                      point_sample_factor=1):

    graph_objects = []
    for geometry in geometry_list:
        geometry_type = geometry.get_geometry_type()

        if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
            graph_objects.append(get_point_object(geometry,
                                                  point_sample_factor))

        if geometry_type == o3d.geometry.Geometry.Type.TriangleMesh:
            graph_objects.append(get_mesh_object(geometry))
            if (mesh_show_wireframe):
                graph_objects.append(get_wireframe_object(geometry))

        if geometry_type == o3d.geometry.Geometry.Type.LineSet:
            graph_objects.append(get_lineset_object(geometry))

    return graph_objects


def get_max_bound(geometry_list):
    max_bound = [0, 0, 0]

    for geometry in geometry_list:
        bound = np.subtract(geometry.get_max_bound(), geometry.get_min_bound())
        max_bound = np.fmax(bound, max_bound)
    return max_bound


def get_geometry_center(geometry_list):
    center = [0, 0, 0]
    for geometry in geometry_list:
        center += geometry.get_center()
    np.divide(center, len(geometry_list))
    return center

def visualize_grasps(
    grasps,
    pcd,
    num_vis_grasp=50,
    grasp_group_color=np.array([[0, 1, 0]]),
    showaxes_grid=True,
    width=3600,
    height=2400,
    image_path=None,  # Add parameter for image path
):
    """
    Save grasp visualization as an image.
    """
    grippers = grasps[:num_vis_grasp].to_open3d_geometry_list()

    # Refine visualization
    grippers_pcd = []
    for idx, gp in enumerate(grippers):
        if idx == 0:
            # highlight best grasp in red
            gp.paint_uniform_color(np.array([[1, 0, 0]]).T)
        else:
            gp.paint_uniform_color(grasp_group_color.T)
        grippers_pcd.append(gp.sample_points_uniformly(number_of_points=4000))

    fig = get_plotly_fig(
        [pcd, *grippers, *grippers_pcd],
        width=width,
        height=height,
        mesh_show_wireframe=showaxes_grid
    )

    if image_path is not None:
        pio.write_image(fig, image_path)
        print(f"Figure saved as {image_path}")

def draw_plotly(geometry_list,
                width=3600,
                height=2400,
                mesh_show_wireframe=False,
                point_sample_factor=1,
                front=None,
                lookat=None,
                up=None,
                zoom=4.0,
                image_path=None):
    fig = get_plotly_fig(geometry_list, 
                         width, 
                         height, 
                         mesh_show_wireframe,
                         point_sample_factor,
                         front,
                         lookat,
                         up,
                         zoom)
    if image_path is not None:
        pio.write_image(fig, image_path)
        print(f"Figure saved as {image_path}")

def save_grasps(
    filename: Path, grasps, translation: np.ndarray = None, num_grasps_save: int = 10
):
    # create the directory, if necessary
    filename.parent.mkdir(parents=True, exist_ok=True)

    # get transformation matrix
    grasp_pose = []

    for idx in range(num_grasps_save):
        # grasp pose
        gp_T = np.eye(4)

        # rotation
        gp_T[:3, :3] = grasps[idx].rotation_matrix

        # translation
        gp_T[:3, -1] = grasps[idx].translation

        # append
        grasp_pose.append(gp_T)

    # grasp pose
    grasp_pose = np.array(grasp_pose)

    # save file
    np.save(file=filename, arr=grasp_pose)

    # save file
    if translation is not None:
        np.save(file=f"{filename.stem}_translation.npy", arr=translation)


def reorient_grasps(
    grasps, normal_dir, ang_threshold=np.deg2rad(20), side_grasp_desired: bool = True 
):
    import copy
    from scipy.spatial.transform import Rotation

    # modified grasp poses
    mod_grasps = copy.deepcopy(grasps)

    # rotation matrices
    rot_mat = mod_grasps.rotation_matrices

    # x-axis of each grasp
    x_axis = rot_mat[..., 0]

    # z-axis of each grasp
    z_axis = rot_mat[..., 2]

    if not side_grasp_desired:
        # align the x-axis with the negative normal direction

        # negative normal direction
        normal_dir = -np.reshape(normal_dir, (-1, 1))

        # angle between the normal direction and the x-axis of the grasp pose
        ang_x_to_normal = np.arccos(
            (x_axis @ normal_dir)
            / (
                np.linalg.norm(x_axis, axis=-1, keepdims=True)
                * np.linalg.norm(normal_dir)
            )
        )

        # rotation axis
        rot_ax = np.cross(x_axis, normal_dir.reshape(1, -1), axis=-1)

        # normalize
        rot_ax /= np.linalg.norm(rot_ax, axis=-1, keepdims=True)

        # rotate the grasp about the rotation axis of the grasp pose
        rot_obj = Rotation.from_rotvec(ang_x_to_normal * rot_ax)

        # result of the rotation
        prop_rot = rot_obj.as_matrix() @ rot_mat
        mod_grasps.rotation_matrices = prop_rot
    else:
        # align the z-axis with the positive normal direction

        # negative normal direction
        normal_dir = np.reshape(normal_dir, (-1, 1))

        # angle between the normal direction and the z-axis of the grasp pose
        ang_z_to_normal = np.arccos(
            (z_axis @ normal_dir)
            / (
                np.linalg.norm(z_axis, axis=-1, keepdims=True)
                * np.linalg.norm(normal_dir)
            )
        )

        # rotation axis
        rot_ax = np.cross(z_axis, normal_dir.reshape(1, -1), axis=-1)

        # normalize
        rot_ax /= np.linalg.norm(rot_ax, axis=-1, keepdims=True)

        # rotate the grasp about the rotation axis of the grasp pose
        rot_obj = Rotation.from_rotvec(ang_z_to_normal * rot_ax)

        # result of the rotation
        prop_rot = rot_obj.as_matrix() @ rot_mat
        mod_grasps.rotation_matrices = prop_rot

    return mod_grasps

