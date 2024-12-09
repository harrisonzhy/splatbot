import numpy as np
from pydrake.all import (
    RigidTransform,
    RotationMatrix,
)
import os

workspace_drake = "/workspace/drake"

def extract_vase_poses(multistage=False):
    def extract_target_goal_pose(obj):
        directory = os.path.join(workspace_drake, f"cloud_grasps/grasp_results/{obj}")
        goal_pose_path = os.path.join(directory, f"{obj}_w_post_processing_scaled.npy")
        goal_pose = np.load(goal_pose_path)

        # top pose
        goal_pose = goal_pose[1]
        rot = np.array(goal_pose[:3, :3])
        trans = np.array(goal_pose[:3, 3]) + [0.5, 0.575, 0.4] # offset on the table (from scenario yaml)
        
        # shift to account for gripper block
        trans += rot @ [0.05, -0.05, 0]

        return RigidTransform(RotationMatrix(rot), trans)

    goal_pose_target = extract_target_goal_pose("vase")
    desired_rotation = goal_pose_target.rotation()
    target_translation = goal_pose_target.translation()

    goal_poses = []
    # append target
    goal_poses.append(goal_pose_target)
    # above target (in the z-axis)
    goal_poses.append(
        RigidTransform(desired_rotation, 
                       np.array([target_translation[0], target_translation[1], 0]) + np.array([0, 0, 0.6])))
    # append translated final pose (in the z-axis)
    goal_poses.append(
        RigidTransform(desired_rotation, 
                       np.array([-0.2, 0.4, 0.6]))
    )
    # append final pose
    goal_poses.append(
        RigidTransform(desired_rotation, 
                       np.array([goal_poses[-1].translation()[0],
                                 goal_poses[-1].translation()[1],
                                 target_translation[2]]))
    )

    # prepend initial pose (close to target in the y-axis)
    goal_poses.insert(0,
        RigidTransform(desired_rotation, 
                       target_translation + desired_rotation @ np.array([0, -0.15, 0]).T))
    
    if multistage:
        goal_poses.append(goal_poses[4])
        goal_poses.append(goal_poses[3])
        goal_poses.append(goal_poses[2])
        goal_poses.append(goal_poses[1])

    return goal_poses

def extract_mustard_poses(multistage=False):
    def extract_target_goal_pose(obj):
        directory = os.path.join(workspace_drake, f"cloud_grasps/grasp_results/{obj}")
        goal_pose_path = os.path.join(directory, f"{obj}_w_post_processing_scaled.npy")
        goal_pose = np.load(goal_pose_path)

        # top pose
        goal_pose = goal_pose[0]
        rot = np.array(goal_pose[:3, :3])
        trans = np.array(goal_pose[:3, 3]) + [0.5, 0.6, 0.4] # offset on the table (from scenario yaml)
        
        # shift to account for gripper block
        trans += rot @ [0.0205, -0.1, 0]

        return RigidTransform(RotationMatrix(rot), trans)

    goal_pose_target = extract_target_goal_pose("mustard")
    desired_rotation = goal_pose_target.rotation()
    target_translation = goal_pose_target.translation()

    goal_poses = []
    # append target
    goal_poses.append(goal_pose_target)
    # above target (in the z-axis)
    goal_poses.append(
        RigidTransform(desired_rotation, 
                       np.array([target_translation[0], target_translation[1], 0]) + np.array([0, 0, 0.6])))
    # append translated final pose (in the z-axis)
    goal_poses.append(
        RigidTransform(desired_rotation, 
                       np.array([-0.2, 0.4, 0.6]))
    )
    # append final pose
    goal_poses.append(
        RigidTransform(desired_rotation, 
                       np.array([goal_poses[-1].translation()[0],
                                 goal_poses[-1].translation()[1],
                                 target_translation[2]]))
    )

    # prepend initial pose (close to target in the y-axis)
    goal_poses.insert(0,
        RigidTransform(desired_rotation, 
                       target_translation + desired_rotation @ np.array([0, -0.1, 0]).T))
    
    if multistage:
        goal_poses.append(goal_poses[4])
        goal_poses.append(goal_poses[3])
        goal_poses.append(goal_poses[2])
        goal_poses.append(goal_poses[1])

    return goal_poses

def extract_plane_poses(multistage=False):
    def extract_target_goal_pose(obj):
        directory = os.path.join(workspace_drake, f"cloud_grasps/grasp_results/{obj}")
        goal_pose_path = os.path.join(directory, f"{obj}_w_post_processing_scaled.npy")
        goal_pose = np.load(goal_pose_path)

        # frame correction
        angle_rad = np.radians(-90)
        frame_correction = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0,                  0,                 1]
        ])

        # top pose
        goal_pose = goal_pose[0]
        rot = np.array(goal_pose[:3, :3]) @ frame_correction
        trans = np.array(goal_pose[:3, 3]) + [0.5, 0.25, 0.4] # offset on the table (from scenario yaml)
        
        # shift to account for gripper block
        trans += rot @ [0.0205, -0.1, 0]

        return RigidTransform(RotationMatrix(rot), trans)

    goal_pose_target = extract_target_goal_pose("plane")
    desired_rotation = goal_pose_target.rotation()
    target_translation = goal_pose_target.translation()

    goal_poses = []
    # append target
    goal_poses.append(goal_pose_target)
    # above target (in the z-axis)
    goal_poses.append(
        RigidTransform(desired_rotation, 
                       np.array([target_translation[0], target_translation[1], 0]) + np.array([0, 0, 0.7])))
    # append translated final pose (in the z-axis)
    goal_poses.append(
        RigidTransform(desired_rotation, 
                       np.array([-0.15, 0.4, goal_poses[-1].translation()[2]]))
    )
    # append final pose
    goal_poses.append(
        RigidTransform(desired_rotation, 
                       np.array([goal_poses[-1].translation()[0],
                                 goal_poses[-1].translation()[1],
                                 target_translation[2]]))
    )

    # prepend initial pose (close to target in the y-axis)
    goal_poses.insert(0,
        RigidTransform(desired_rotation, 
                       target_translation + desired_rotation @ np.array([0, -0.1, 0]).T))
    
    if multistage:
        goal_poses.append(goal_poses[4])
        goal_poses.append(goal_poses[3])
        goal_poses.append(goal_poses[2]) 
        goal_poses.append(goal_poses[1])

    return goal_poses

def extract_gravy_poses(multistage=False):
    def extract_target_goal_pose(obj):
        directory = os.path.join(workspace_drake, f"cloud_grasps/grasp_results/{obj}")
        goal_pose_path = os.path.join(directory, f"{obj}_w_post_processing_scaled.npy")
        goal_pose = np.load(goal_pose_path)

        # top pose
        goal_pose = goal_pose[0]
        rot = np.array(goal_pose[:3, :3])
        trans = np.array(goal_pose[:3, 3]) + [0.4, 0.4, 0.4] # offset on the table (from scenario yaml)
        
        # shift to account for gripper block
        trans += rot @ [0.015, 0, 0.1]

        return RigidTransform(RotationMatrix(rot), trans)

    goal_pose_target = extract_target_goal_pose("gravy")
    desired_rotation = goal_pose_target.rotation()
    target_translation = goal_pose_target.translation()

    goal_poses = []
    # append target
    goal_poses.append(goal_pose_target)
    # above target (in the z-axis)
    goal_poses.append(
        RigidTransform(desired_rotation, 
                       np.array([target_translation[0], target_translation[1], 0]) + np.array([0, 0, 0.6])))
    # append translated final pose (in the z-axis)
    goal_poses.append(
        RigidTransform(desired_rotation, 
                       np.array([-0.2, 0.4, 0.6]))
    )
    # append final pose
    goal_poses.append(
        RigidTransform(desired_rotation, 
                       np.array([goal_poses[-1].translation()[0],
                                 goal_poses[-1].translation()[1],
                                 target_translation[2]]))
    )

    # prepend initial pose (close to target in the y-axis)
    goal_poses.insert(0,
        RigidTransform(desired_rotation, 
                       target_translation + desired_rotation @ np.array([0, -0.15, 0]).T))

    if multistage:
        goal_poses.append(goal_poses[4])
        goal_poses.append(goal_poses[3])
        goal_poses.append(goal_poses[2])
        goal_poses.append(goal_poses[1])

    return goal_poses

def extract_aff_vase_poses(multistage=False):
    def extract_target_goal_pose(obj):
        directory = os.path.join(workspace_drake, f"cloud_grasps/grasp_results/{obj}")
        goal_pose_path = os.path.join(directory, f"{obj}_w_post_processing_scaled.npy")
        goal_pose = np.load(goal_pose_path)

        # top pose
        goal_pose = goal_pose[0]
        rot = np.array(goal_pose[:3, :3])
        trans = np.array(goal_pose[:3, 3]) + [0.5, 0.2, 0.4] # offset on the table (from scenario yaml)
        
        # shift to account for gripper block
        trans += rot @ [0.05, -0.0925, 0.025]

        return RigidTransform(RotationMatrix(rot), trans)

    goal_pose_target = extract_target_goal_pose("aff_vase")
    desired_rotation = goal_pose_target.rotation()
    target_translation = goal_pose_target.translation()

    goal_poses = []
    # append target
    goal_poses.append(goal_pose_target)
    # above target (in the z-axis)
    goal_poses.append(
        RigidTransform(desired_rotation, 
                       np.array([target_translation[0], target_translation[1], 0]) + np.array([0, 0, 0.6])))
    # append translated final pose (in the z-axis)
    goal_poses.append(
        RigidTransform(desired_rotation, 
                       np.array([0, 0.3, 0.6]))
    )
    # append final pose
    goal_poses.append(
        RigidTransform(desired_rotation, 
                       np.array([goal_poses[-1].translation()[0],
                                 goal_poses[-1].translation()[1],
                                 target_translation[2] + 0.05]))
    )

    # prepend initial pose (close to target in the y-axis)
    goal_poses.insert(0,
        RigidTransform(desired_rotation, 
                       target_translation + desired_rotation @ np.array([0, -0.15, 0]).T))
    
    if multistage:
        # back away to avoid collision
        goal_poses.append(
            RigidTransform(desired_rotation,
                           goal_poses[4].translation() + desired_rotation @ np.array([0, -0.1, 0]).T))
        # go down to pick up
        goal_poses.append(
            RigidTransform(desired_rotation,
                        np.array([goal_poses[-1].translation()[0],
                                  goal_poses[-1].translation()[1],
                                  target_translation[2]])))
            
        # go up to lift object
        goal_poses.append(
            RigidTransform(goal_poses[-2].rotation(),
                          goal_poses[-2].translation() + np.array([0, 0, 0.1])))

        goal_poses.append(
            RigidTransform(goal_poses[1].rotation(),
                           np.array([goal_poses[1].translation()[0], 
                                     goal_poses[1].translation()[1],
                                     target_translation[2]])
                         + np.array(desired_rotation @ np.array([0, -0.025, 0.05]).T)))

    return goal_poses
