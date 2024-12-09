import numpy as np
from pydrake.all import (
    DiagramBuilder,
    InverseKinematics,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    RigidTransform,
    RotationMatrix,
    Solve,
    StartMeshcat,
    Simulator
)
from manipulation.station import LoadScenario, MakeHardwareStation
from manipulation.scenarios import AddMultibodyTriad
from manipulation.meshcat_utils import AddMeshcatTriad
import time
from pydrake.trajectories import PiecewisePolynomial
from pydrake.systems.primitives import TrajectorySource, MatrixGain
from pydrake.systems.framework import LeafSystem, BasicVector
from scipy.optimize import minimize

import os
from extract_poses import *

workspace_drake = "/workspace/drake"

# Start the visualizer
meshcat = StartMeshcat()
print("Meshcat URL:", meshcat.web_url())

def build_env():
    """Build the simulation environment and set up visualization with Meshcat."""
    builder = DiagramBuilder()
    scenario = LoadScenario(filename=final_scenario_path)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")

    MeshcatVisualizer.AddToBuilder(
        builder,
        station.GetOutputPort("query_object"),
        meshcat,
        MeshcatVisualizerParams(delete_on_initialization_event=False),
    )
    AddMultibodyTriad(plant.GetFrameByName("body"), scene_graph)
    diagram = builder.Build()
    context = plant.CreateDefaultContext()
    initial_q = plant.GetPositions(context)
    return diagram, plant, scene_graph, initial_q


def solve_ik(X_WG, max_tries=100, initial_guess=None):
    """Solve the inverse kinematics problem for the given goal pose, including orientation constraints."""
    diagram, plant, scene_graph, initial_q = build_env()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    ik = InverseKinematics(plant, plant_context)
    q_variables = ik.q()
    prog = ik.prog()

    q_nominal = initial_guess if initial_guess is not None else np.zeros(len(q_variables))
    prog.AddQuadraticErrorCost(np.eye(len(q_variables)), q_nominal, q_variables)

    # Add position constraint
    ik.AddPositionConstraint(
        frameB=plant.GetFrameByName("body"),
        p_BQ=np.array([0, 0, 0]),
        frameA=plant.world_frame(),
        p_AQ_lower=X_WG.translation() - 0.01,
        p_AQ_upper=X_WG.translation() + 0.01,
    )

    # Add orientation constraint
    ik.AddOrientationConstraint(
        frameAbar=plant.world_frame(),
        R_AbarA=X_WG.rotation(),
        frameBbar=plant.GetFrameByName("body"),
        R_BbarB=RotationMatrix.Identity(),
        theta_bound=0.1,
    )

    for count in range(max_tries):
        q_initial_guess = (
            initial_guess if initial_guess is not None else np.random.uniform(-1.0, 1.0, len(q_variables))
        )
        prog.SetInitialGuess(q_variables, q_initial_guess)
        result = Solve(prog)

        if result.is_success():
            print(f"Succeeded in {count + 1} tries.")
            return diagram, plant, scene_graph, initial_q, result.GetSolution(q_variables), context

    print("IK failed!")
    return None, None, None, None, None, None


class GripperControlSystem(LeafSystem):
    """Control system for the gripper."""
    def __init__(self, open_position=0.3, close_position=0.0, close_time=10.0, open_time=20.0):
        super().__init__()
        self.open_position = open_position
        self.close_position = close_position
        self.close_time = close_time
        self.open_time = open_time
        self.DeclareVectorOutputPort("wsg.position", BasicVector(1), self.CalcGripperPosition)

    def CalcGripperPosition(self, context, output):
        current_time = context.get_time()
        if current_time >= self.open_time:
            output.SetAtIndex(0, self.open_position)
        elif current_time >= self.close_time:
            output.SetAtIndex(0, self.close_position)
        else:
            output.SetAtIndex(0, self.open_position)


def visualize_goal_frames(meshcat, goal_poses):
    """Visualize multiple goal poses in Meshcat."""
    for idx, pose in enumerate(goal_poses, start=1):
        AddMeshcatTriad(
            meshcat,
            path=f"goal_frame_{idx}",
            X_PT=pose,
            length=0.2,
            radius=0.01,
            opacity=1.0,
        )

def generate_trajectory(initial_q, final_q, num_steps=100, weight_smoothness=10.0, weight_straightness=5000.0):
    """
    Generate a trajectory that minimizes deviation from a straight line between initial and final configurations.
    """
    n_dofs = len(initial_q)
    alphas = np.linspace(0, 1, num_steps)
    straight_line = np.array([(1 - alpha) * initial_q + alpha * final_q for alpha in alphas])

    def trajectory_cost(traj):
        traj = traj.reshape(num_steps, n_dofs)
        # Smoothness cost: penalize changes between consecutive points
        smoothness_cost = sum(np.linalg.norm(traj[i] - traj[i - 1]) ** 2 for i in range(1, num_steps))
        # Straightness cost: penalize deviation from straight line
        straightness_cost = sum(np.linalg.norm(traj[i] - straight_line[i]) ** 2 for i in range(num_steps))
        return weight_smoothness * smoothness_cost + weight_straightness * straightness_cost

    # Minimize the cost function
    result = minimize(
        trajectory_cost,
        straight_line.flatten(),  # Use straight line as initial guess
        method="L-BFGS-B",
        options={"disp": True}
    )

    # Reshape the result into the trajectory
    optimized_trajectory = result.x.reshape(num_steps, n_dofs)
    return optimized_trajectory

import numpy as np
from scipy.optimize import minimize, Bounds

def generate_restricted_trajectory(
    initial_q, final_q, num_steps=100, weight_smoothness=10.0, 
    weight_straightness=100.0, tolerance=0.02, weight_orientation=100.0
):
    """
    Generate a trajectory with explicit constraints for linear interpolation, smoothness, and orientation consistency.
    """
    n_dofs = len(initial_q)
    alphas = np.linspace(0, 1, num_steps)
    straight_line = np.array([(1 - alpha) * initial_q + alpha * final_q for alpha in alphas])

    # Function to calculate orientation alignment cost
    def orientation_cost(traj):
        traj = traj.reshape(num_steps, n_dofs)
        total_cost = 0
        for i in range(1, num_steps):
            delta_orientation = np.arccos(
                np.clip(np.dot(traj[i - 1], traj[i]) / (np.linalg.norm(traj[i - 1]) * np.linalg.norm(traj[i])), -1.0, 1.0)
            )
            total_cost += delta_orientation**2
        return total_cost

    # Define the cost function
    def trajectory_cost(traj):
        traj = traj.reshape(num_steps, n_dofs)
        # Smoothness cost: penalize large changes between consecutive points
        smoothness_cost = sum(np.linalg.norm(traj[i] - traj[i - 1]) ** 2 for i in range(1, num_steps))
        # Straightness cost: penalize deviation from the straight line
        straightness_cost = sum(np.linalg.norm(traj[i] - straight_line[i]) ** 2 for i in range(num_steps))
        # Orientation cost: encourage smooth orientation changes
        orientation_alignment_cost = orientation_cost(traj)
        return (weight_smoothness * smoothness_cost +
                weight_straightness * straightness_cost +
                weight_orientation * orientation_alignment_cost)

    # Bounds: Ensure trajectory points are within tolerance of the straight line
    lower_bounds = (straight_line - tolerance).flatten()
    upper_bounds = (straight_line + tolerance).flatten()
    bounds = Bounds(lower_bounds, upper_bounds)

    # Minimize the cost function
    result = minimize(
        trajectory_cost,
        straight_line.flatten(),  # Use straight line as initial guess
        method="L-BFGS-B",
        bounds=bounds,
        options={"disp": True}
    )

    # Reshape the result into the trajectory
    optimized_trajectory = result.x.reshape(num_steps, n_dofs)
    return optimized_trajectory

def build_and_simulate_trajectory(q_traj, gripper_actions, pause_duration=1.0):
    """
    Simulate the robot's motion along the generated trajectory, 
    with multiple gripper open/close actions and pauses after each action.
    
    Parameters:
    - q_traj: PiecewisePolynomial trajectory for the robot's motion.
    - gripper_actions: List of tuples [(time, position)], where:
        - time: The simulation time to set the gripper position.
        - position: The target position of the gripper (e.g., open or close).
    - pause_duration: Duration of pause after each gripper action.
    """
    builder = DiagramBuilder()
    scenario = LoadScenario(filename=final_scenario_path)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))

    # Add trajectory source
    class PausedTrajectorySource(LeafSystem):
        def __init__(self, trajectory, gripper_actions, pause_duration):
            super().__init__()
            self.trajectory = trajectory
            self.gripper_actions = sorted(gripper_actions, key=lambda x: x[0])
            self.pause_duration = pause_duration
            output_size = trajectory.rows()  # Match trajectory DoF
            self.DeclareVectorOutputPort("position", BasicVector(output_size), self.CalcPosition)
            self.last_position = trajectory.value(0).flatten()  # Initialize with the starting position
            self.holding = False

        def CalcPosition(self, context, output):
            current_time = context.get_time()
            # Check if we are in a pause
            self.holding = any(
                action_time <= current_time < action_time + self.pause_duration
                for action_time, _ in self.gripper_actions
            )
            if self.holding:
                # During pause, hold the last position
                output.SetFromVector(self.last_position)
            else:
                # Update position from the trajectory
                self.last_position = self.trajectory.value(current_time).flatten()
                output.SetFromVector(self.last_position)

    paused_trajectory_source = builder.AddSystem(
        PausedTrajectorySource(q_traj, gripper_actions, pause_duration)
    )

    # Adjust MatrixGain size to match trajectory DoF
    dof = q_traj.rows()  # Number of degrees of freedom
    trim_gain = builder.AddSystem(MatrixGain(np.eye(7, 16)))  # Adjust to correct size
    builder.Connect(paused_trajectory_source.get_output_port(), trim_gain.get_input_port())
    builder.Connect(trim_gain.get_output_port(), station.GetInputPort("iiwa.position"))

    # Add gripper control
    class AdvancedGripperControlSystem(LeafSystem):
        def __init__(self, gripper_actions):
            super().__init__()
            self.gripper_actions = sorted(gripper_actions, key=lambda x: x[0])
            self.DeclareVectorOutputPort("wsg.position", BasicVector(1), self.CalcGripperPosition)

        def CalcGripperPosition(self, context, output):
            current_time = context.get_time()
            # Default to the last position if no action is specified
            gripper_position = self.gripper_actions[-1][1]
            for time, position in self.gripper_actions:
                if current_time < time:
                    break
                gripper_position = position
            output.SetAtIndex(0, gripper_position)

    gripper_control_system = builder.AddSystem(AdvancedGripperControlSystem(gripper_actions))
    builder.Connect(gripper_control_system.get_output_port(0), station.GetInputPort("wsg.position"))

    # Build and simulate the diagram
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    # Extend simulation time to include pauses after gripper actions
    max_simulation_time = max([action[0] for action in gripper_actions]) + pause_duration + 1.0
    meshcat.StartRecording(set_visualizations_while_recording=False)
    simulator.AdvanceTo(max_simulation_time)
    meshcat.PublishRecording()
    print("Simulation completed.")

def add_pauses_to_actions(gripper_actions, pause_duration=3.0):
    """
    Add pauses after each gripper action.

    Parameters:
    - gripper_actions: List of tuples [(time, position)].
    - pause_duration: Duration of the pause after each action.

    Returns:
    - A new list of gripper actions with pauses included.
    """
    adjusted_actions = []
    for i, (time, position) in enumerate(gripper_actions):
        adjusted_actions.append((time, position))
        # Add a pause after each action, except the last one
        if i < len(gripper_actions) - 1:
            pause_time = time + pause_duration
            adjusted_actions.append((pause_time, position))
    return adjusted_actions

def generate_combined_trajectory(pose_nodes, num_steps=100, segment_duration=5):
    trajectories = []
    time_segments = []
    segment_times = []

    # generate trajectories between consecutive pose_nodes
    for i in range(len(pose_nodes) - 1):
        trajectory = generate_restricted_trajectory(pose_nodes[i], pose_nodes[i + 1], num_steps)
        trajectories.append(np.array(trajectory).T)

        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        print(start_time, end_time)

        time_segment = np.linspace(start_time, end_time, num_steps)

        if i > 0:
            time_segment = time_segment[1:]
        
        time_segments.append(time_segment)
        segment_times.append((start_time, end_time))

    # combine time segments and trajectories
    combined_times = np.hstack(time_segments)
    combined_trajectory = np.hstack([trajectory[:, 1:] if i > 0 else trajectory for i, trajectory in enumerate(trajectories)])

    return combined_times, combined_trajectory, segment_times

if __name__ == "__main__":
    # Finalized scenario
    final_scenario_path = os.path.join(workspace_drake, "final_scenario.yaml")
    print("Scenario path:", final_scenario_path)

    # Define goal poses
    goal_poses = extract_aff_vase_poses(multistage=True)
    # goal_poses = extract_gravy_poses(multistage=True)
    # goal_poses = extract_plane_poses(multistage=True)
    # goal_poses = extract_vase_poses(multistage=True)
    # goal_poses = extract_mustard_poses(multistage=True)

    # Visualize goal frames
    visualize_goal_frames(meshcat, goal_poses)

    # Solve IK
    initial_guess = None
    pose_nodes = []    

    for ik_num in range(len(goal_poses)):
        print(f"Solve IK {ik_num}...")
        _, _, _, initial_guess, res, _ = solve_ik(goal_poses[ik_num], initial_guess=initial_guess)
        if res is None:
            exit(code=1)
        if ik_num == 0:
            pose_nodes.append(initial_guess)
        pose_nodes.append(res)
        initial_guess = res

    # Generate interpolated trajectory between poses
    combined_times, combined_trajectory, segment_times = generate_combined_trajectory(pose_nodes)

    q_traj_combined = PiecewisePolynomial.FirstOrderHold(combined_times, combined_trajectory)

    # Simulate
    print(q_traj_combined)

    # Define gripper actions (default)
    # gripper_actions = [
    #     (segment_times[2][0], 0.0),
    #     (segment_times[5][0], 0.2),
    #     (segment_times[6][0], 0.0),
    #     (segment_times[-1][1], 0.2),
    # ]
    
    # Define gripper actions (affordance vase demo)
    gripper_actions = [
        (segment_times[2][0], 0.0), 
        (segment_times[5][0], 0.2),
        (segment_times[7][0], 0.0),
        (segment_times[-1][1], 0.2),
    ]

    # Add pauses to gripper actions
    gripper_actions_with_pauses = add_pauses_to_actions(gripper_actions)

    # Simulate the trajectory with gripper actions
    build_and_simulate_trajectory(q_traj_combined, gripper_actions_with_pauses)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Visualization interrupted.")
