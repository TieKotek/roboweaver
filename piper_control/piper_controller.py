"""
AgileX PiPER Robot Arm Controller

This module provides a comprehensive control interface for the AgileX PiPER robot arm
using MuJoCo simulation environment.
"""

import numpy as np
import mujoco
import time
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass
import ikpy.chain
import warnings
import math
from scipy.spatial.transform import Rotation as R

@dataclass
class RobotState:
    """Robot state data structure"""
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_efforts: np.ndarray
    end_effector_pose: np.ndarray  # [x, y, z, roll, pitch, yaw]
    gripper_position: float
    timestamp: float


class PiperKinematics:
    """Forward and inverse kinematics for PiPER robot"""
    
    def __init__(self, model: mujoco.MjModel, urdf_path: str):
        self.model = model
        self.data = mujoco.MjData(model)
        
        # Define which links are active for IK.
        # The chain from URDF has 9 links: base, 6 joints, 1 fixed link, 1 gripper joint.
        # We set only the 6 arm joints to True to be used in IK calculations.
        active_links_mask = [False, True, True, True, True, True, True, False, False]
        
        # Create ikpy chain from URDF, providing the active links mask to avoid warnings
        self.chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            active_links_mask=active_links_mask
        )
        
        # Joint indices
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.gripper_joint = 'joint7'
        
        # Get joint IDs
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                         for name in self.joint_names]
        self.gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.gripper_joint)
        
        # End-effector body (link6 or the gripper center)
        self.ee_body_name = 'link6'
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)
        
        # Joint limits
        self.joint_limits = np.array([
            [-2.618, 2.618],   # joint1
            [0, 3.14],         # joint2
            [-2.697, 0],       # joint3
            [-1.832, 1.832],   # joint4
            [-1.22, 1.22],     # joint5
            [-3.14, 3.14]      # joint6
        ])

    def _rotation_matrix_to_euler_xyz(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Converts a rotation matrix to euler angles (roll, pitch, yaw) using scipy.
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Euler angles [roll, pitch, yaw] in radians using XYZ convention
        """
        return R.from_matrix(rotation_matrix).as_euler('xyz')

    def _euler_xyz_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """
        Converts euler angles (roll, pitch, yaw) to a rotation matrix using scipy.
        
        Args:
            euler: Euler angles [roll, pitch, yaw] in radians using XYZ convention
            
        Returns:
            3x3 rotation matrix
        """
        return R.from_euler('xyz', euler).as_matrix()
        
    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics to get end-effector pose
        
        Args:
            joint_positions: Joint angles [rad] for joints 1-6
            
        Returns:
            End-effector pose [x, y, z, roll, pitch, yaw]
        """
        if len(joint_positions) != 6:
            raise ValueError("Expected 6 joint positions")
            
        # ikpy expects full joint vector for all active links
        # Chain structure: [Base, joint1, joint2, joint3, joint4, joint5, joint6, fixed_joint, joint7]
        # We need: [0, joint1, joint2, joint3, joint4, joint5, joint6, 0, 0]
        full_joint_positions = np.zeros(9)
        full_joint_positions[0] = 0  # Base link (fixed)
        full_joint_positions[1:7] = joint_positions  # joint1-joint6
        full_joint_positions[7] = 0  # fixed joint (joint6_to_gripper_base)
        full_joint_positions[8] = 0  # gripper joint (set to closed position)
        
        # Compute forward kinematics using ikpy
        transformation_matrix = self.chain.forward_kinematics(full_joint_positions)
        
        # Extract position (translation part)
        position = transformation_matrix[:3, 3]
        
        # Extract rotation matrix and convert to Euler angles
        rotation_matrix = transformation_matrix[:3, :3]
        euler_angles = self._rotation_matrix_to_euler_xyz(rotation_matrix)
        
        # Return pose as [x, y, z, roll, pitch, yaw]
        return np.concatenate([position, euler_angles])
    
    def inverse_kinematics(self, target_pose: np.ndarray, 
                          current_joints: Optional[np.ndarray] = None,
                          orientation_needed: bool = False
                          ) -> Optional[np.ndarray]:
        """
        Compute inverse kinematics to get joint positions for target pose
        
        Args:
            target_pose: Target pose [x, y, z] + [roll, pitch, yaw] if orientation_needed is True
            current_joints: Current joint positions for initialization (optional)
            orientation_needed: Whether to consider orientation in IK
            
        Returns:
            Joint positions [rad] for joints 1-6, or None if no solution found
        """
        if orientation_needed:
            if len(target_pose) != 6:
                raise ValueError("Expected target_pose of length 6 when orientation_needed is True")
        else:
            if len(target_pose) != 3:
                raise ValueError("Expected target_pose of length 3 when orientation_needed is False")
            # Append zero orientation
            target_pose = np.concatenate([target_pose, np.zeros(3)])

        # Extract position and euler angles
        position = target_pose[:3]
        euler_angles = target_pose[3:]
        
        # Convert euler angles to rotation matrix
        rotation_matrix = self._euler_xyz_to_rotation_matrix(euler_angles)
        
        # Create 4x4 transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = position
        
        # Prepare initial joint positions for ikpy
        if current_joints is not None:
            if len(current_joints) != 6:
                raise ValueError("Expected 6 current joint positions")
            # ikpy expects full joint vector for all active links
            initial_position = np.zeros(9)
            initial_position[0] = 0  # Base link (fixed)
            initial_position[1:7] = current_joints  # joint1-joint6
            initial_position[7] = 0  # fixed joint (joint6_to_gripper_base)
            initial_position[8] = 0  # gripper joint
        else:
            # Use zeros as initial guess
            initial_position = np.zeros(9)
            
        try:
            # Solve inverse kinematics using ikpy
            
            ik_solution = self.chain.inverse_kinematics_frame(
                transformation_matrix, 
                initial_position=initial_position,
                orientation_mode="all" if orientation_needed else None,
                # optimizer='fmin_slsqp'
            )
            
            # Extract the 6 joint values (excluding fixed joints and gripper)
            joint_positions = ik_solution[1:7]  # joint1-joint6
            
            # Check joint limits
            if not self._check_joint_limits_ik(joint_positions):
                return None
                
            return joint_positions
            
        except Exception as e:
            print(f"IK computation failed: {e}")
            return None
            
    def _check_joint_limits_ik(self, joint_positions: np.ndarray) -> bool:
        """Check if joint positions are within limits for IK"""
        for i, pos in enumerate(joint_positions):
            if i < len(self.joint_limits):
                lower, upper = self.joint_limits[i]
                if pos < lower or pos > upper:
                    return False
        return True
    

class TrajectoryPlanner:
    """Trajectory planning utilities"""
    
    @staticmethod
    def interpolate_joints(start_joints: np.ndarray, end_joints: np.ndarray, 
                          duration: float, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate smooth joint trajectory using cubic spline
        
        Args:
            start_joints: Starting joint positions
            end_joints: Target joint positions  
            duration: Motion duration [s]
            dt: Time step [s]
            
        Returns:
            time_points, joint_trajectories
        """
        time_points = np.arange(0, duration + dt, dt)
        n_points = len(time_points)
        n_joints = len(start_joints)
        
        # Cubic polynomial: q(t) = a0 + a1*t + a2*t^2 + a3*t^3
        # Boundary conditions: q(0)=q_start, q(T)=q_end, dq(0)=0, dq(T)=0
        
        trajectories = np.zeros((n_points, n_joints))
        
        for j in range(n_joints):
            # Solve for cubic coefficients
            q0 = start_joints[j]
            qf = end_joints[j]
            
            a0 = q0
            a1 = 0  # zero initial velocity
            a2 = 3 * (qf - q0) / (duration ** 2)
            a3 = -2 * (qf - q0) / (duration ** 3)
            
            # Generate trajectory
            for i, t in enumerate(time_points):
                if t <= duration:
                    trajectories[i, j] = a0 + a1*t + a2*(t**2) + a3*(t**3)
                else:
                    trajectories[i, j] = qf
                    
        return time_points, trajectories
    
    @staticmethod
    def interpolate_cartesian(start_pose: np.ndarray, end_pose: np.ndarray,
                            duration: float, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate smooth Cartesian trajectory
        
        Args:
            start_pose: Starting pose [x, y, z, roll, pitch, yaw]
            end_pose: Target pose [x, y, z, roll, pitch, yaw]
            duration: Motion duration [s]
            dt: Time step [s]
            
        Returns:
            time_points, pose_trajectories
        """
        time_points = np.arange(0, duration + dt, dt)
        n_points = len(time_points)
        
        trajectories = np.zeros((n_points, 6))
        
        # Linear interpolation for position
        start_pos = start_pose[:3]
        end_pos = end_pose[:3]
        
        # Linear interpolation for orientation (Euler angles)
        start_orient = start_pose[3:]
        end_orient = end_pose[3:]
        
        for i, t in enumerate(time_points):
            if t <= duration:
                alpha = t / duration
                # Smooth interpolation factor (S-curve)
                alpha_smooth = 3*alpha**2 - 2*alpha**3
            else:
                alpha_smooth = 1.0
                
            # Position interpolation
            pos = start_pos + alpha_smooth * (end_pos - start_pos)
            
            # Orientation interpolation
            orient = start_orient + alpha_smooth * (end_orient - start_orient)
            
            trajectories[i] = np.concatenate([pos, orient])
            
        return time_points, trajectories


class PiperController:
    """Main controller class for AgileX PiPER robot arm"""
    
    def __init__(self, model_path: str = "agilex_piper/scene.xml", urdf_path: str = "agilex_piper/piper_description.urdf"):
        """
        Initialize PiPER controller
        
        Args:
            model_path: Path to MuJoCo model file
        """
        # Load model and data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize kinematics
        self.kinematics = PiperKinematics(self.model, urdf_path)
        
        # Control parameters
        self.control_dt = 0.01  # 100Hz control rate
        self.max_cartesian_velocity = 0.5  # m/s
        
        # Safety parameters
        self.emergency_stop_flag = False
        self.is_moving = False
        
        # Home position
        self.home_joints = np.array([0, 1.57, -1.3485, 0, 0, 0])
        
        # Reset to home position
        self._reset_to_home()
        
    def _reset_to_home(self):
        """Reset robot to home position"""
        self.set_joint_positions(self.home_joints)
        self.set_gripper_position(0.0)
        
        # IMPORTANT: Also set the control targets to match home position
        # This prevents the robot from jumping to zero position when
        # first action is idle, gripper control, etc.
        ctrl_indices = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                        for name in self.kinematics.joint_names]
        for j, ctrl_id in enumerate(ctrl_indices):
            if ctrl_id != -1:  # Check if actuator exists
                self.data.ctrl[ctrl_id] = self.home_joints[j]
        
        # Also set gripper control target
        gripper_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        if gripper_actuator_id != -1:
            self.data.ctrl[gripper_actuator_id] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        
    def get_robot_state(self) -> RobotState:
        """Get current robot state"""
        joint_positions = np.array([self.data.qpos[joint_id] for joint_id in self.kinematics.joint_ids])
        joint_velocities = np.array([self.data.qvel[joint_id] for joint_id in self.kinematics.joint_ids])
        joint_efforts = np.array([self.data.qfrc_applied[joint_id] for joint_id in self.kinematics.joint_ids])
        
        end_effector_pose = self.kinematics.forward_kinematics(joint_positions)
        gripper_position = self.data.qpos[self.kinematics.gripper_id]
        
        return RobotState(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_efforts=joint_efforts,
            end_effector_pose=end_effector_pose,
            gripper_position=gripper_position,
            timestamp=time.time()
        )
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions"""
        return np.array([self.data.qpos[joint_id] for joint_id in self.kinematics.joint_ids])
    
    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities"""
        return np.array([self.data.qvel[joint_id] for joint_id in self.kinematics.joint_ids])
    
    def get_end_effector_pose(self) -> np.ndarray:
        """Get current end-effector pose [x, y, z, roll, pitch, yaw]"""
        joint_positions = self.get_joint_positions()
        return self.kinematics.forward_kinematics(joint_positions)
    
    def get_gripper_position(self) -> float:
        """Get current gripper position"""
        return self.data.qpos[self.kinematics.gripper_id]

    def get_gripper_velocity(self) -> float:
        """Get current gripper joint velocity."""
        return self.data.qvel[self.kinematics.gripper_id]
    
    def set_joint_positions(self, joint_positions: np.ndarray):
        """Set joint positions directly (for initialization)"""
        for i, joint_id in enumerate(self.kinematics.joint_ids):
            self.data.qpos[joint_id] = joint_positions[i]
            
    def set_gripper_position(self, position: float):
        """Set gripper position directly"""
        self.data.qpos[self.kinematics.gripper_id] = np.clip(position, 0.0, 0.035)
        
    def move_to_joint_positions(self, target_joints: np.ndarray, duration: Optional[float] = None, viewer: Optional['mujoco.viewer.Viewer'] = None):
        """
        Move to target joint positions.

        Args:
            target_joints: Target joint angles [rad].
            duration: Motion duration [s]. If None, moves as fast as possible and waits for completion.
            viewer: Optional MuJoCo viewer instance for rendering.
        """
        if self.emergency_stop_flag:
            print("Emergency stop active - motion cancelled")
            return False

        # Validate joint limits
        if not self._check_joint_limits(target_joints):
            print("Target joints exceed limits")
            return False

        # If duration is None, use direct control and wait for completion
        if duration is None:
            return self._move_joints_to_completion(target_joints, viewer)

        # --- Trajectory-based movement ---
        current_joints = self.get_joint_positions()

        # Generate trajectory
        time_points, joint_trajectory = TrajectoryPlanner.interpolate_joints(
            current_joints, target_joints, duration, self.control_dt
        )

        # Execute trajectory
        return self._execute_joint_trajectory(time_points, joint_trajectory, viewer)
    
    def move_to_cartesian_pose(self, target_pose: np.ndarray, duration: Optional[float] = None, orientation_needed: bool = False, viewer: Optional['mujoco.viewer.Viewer'] = None):
        """
        Move to target Cartesian pose
        
        Args:
            target_pose: Target pose [x, y, z] + [roll, pitch, yaw] if orientation_needed is True
            duration: Motion duration [s]
            orientation_needed: Whether to consider orientation in IK
            viewer: Optional MuJoCo viewer instance for rendering.
        """
        if self.emergency_stop_flag:
            print("Emergency stop active - motion cancelled")
            return False
            
        # Solve inverse kinematics
        current_joints = self.get_joint_positions()
        target_joints = self.kinematics.inverse_kinematics(target_pose, current_joints, orientation_needed=orientation_needed)
        
        if target_joints is None:
            print("No IK solution found for target pose")
            return False
            
        return self.move_to_joint_positions(target_joints, duration, viewer)
    
    def open_gripper(self, viewer: Optional['mujoco.viewer.Viewer'] = None):
        """
        Open gripper until it is fully open.
        
        Args:
            viewer: Optional MuJoCo viewer instance for rendering.
        """
        self._move_gripper_to_completion(0.035, viewer=viewer)
        
    def close_gripper(self, viewer: Optional['mujoco.viewer.Viewer'] = None):
        """
        Close gripper until it is fully closed or has gripped an object.
        
        Args:
            viewer: Optional MuJoCo viewer instance for rendering.
        """
        self._move_gripper_to_completion(0.0, viewer=viewer)
        
    def is_pose_reachable(self, target_pose: np.ndarray) -> bool:
        """Check if target pose is reachable"""
        current_joints = self.get_joint_positions()
        target_joints = self.kinematics.inverse_kinematics(target_pose, current_joints)
        return target_joints is not None
    
    def emergency_stop(self):
        """Emergency stop - halt all motion"""
        self.emergency_stop_flag = True
        self.is_moving = False
        print("EMERGENCY STOP ACTIVATED")
        
    def reset_emergency_stop(self):
        """Reset emergency stop"""
        self.emergency_stop_flag = False
        print("Emergency stop reset")
        
    def idle(self, duration: float, viewer: Optional['mujoco.viewer.Viewer'] = None):
        """
        Make the robot arm stay idle for a fixed duration.

        Args:
            duration: The time in seconds to stay idle.
            viewer: Optional MuJoCo viewer instance for rendering.
        """
        if self.emergency_stop_flag:
            print("Emergency stop active - idle cancelled")
            return

        print(f"Robot idling for {duration:.2f} seconds...")
        start_time = time.time()
        while (time.time() - start_time) < duration:
            if self.emergency_stop_flag:
                print("Emergency stop active - idle interrupted")
                break
            
            if viewer and not viewer.is_running():
                print("Viewer closed, stopping idle.")
                break

            # Continue stepping the simulation to keep it active, but without moving
            mujoco.mj_step(self.model, self.data)
            
            # Render
            if viewer:
                viewer.sync()

            time.sleep(self.control_dt)
        print("Idle complete.")
        
    def _check_joint_limits(self, joint_positions: np.ndarray) -> bool:
        """Check if joint positions are within limits"""
        limits = self.kinematics.joint_limits
        return np.all(joint_positions >= limits[:, 0]) and np.all(joint_positions <= limits[:, 1])

    def _wait_for_joints_to_settle(self, target_joints: np.ndarray, viewer: Optional['mujoco.viewer.Viewer'] = None):
        """Waits for the joints to settle at a target position."""
        start_time = time.time()
        timeout = 10.0  # seconds

        pos_tolerance = 0.1  # rad
        vel_tolerance = 0.05  # rad/s
        settle_steps = 5
        settle_counter = 0

        while time.time() - start_time < timeout:
            if self.emergency_stop_flag:
                print("Motion stopped due to emergency stop")
                break
            
            if viewer and not viewer.is_running():
                print("Viewer closed, stopping wait.")
                break

            current_joints = self.get_joint_positions()
            current_velocities = self.get_joint_velocities()

            # Check if target position is reached within tolerance
            position_error = np.abs(current_joints - target_joints)
            if np.all(position_error < pos_tolerance):
                # Check for settling (low velocity)
                if np.all(np.abs(current_velocities) < vel_tolerance):
                    settle_counter += 1
                else:
                    settle_counter = 0

                if settle_counter >= settle_steps:
                    print("Joints have settled at target.")
                    break
                
            # Keep stepping simulation
            mujoco.mj_step(self.model, self.data)
            if viewer:
                viewer.sync()
            time.sleep(self.control_dt)
        else:
            print("Warning: Joint move timed out while waiting to settle.")

    def _move_joints_to_completion(self, target_joints: np.ndarray, viewer: Optional['mujoco.viewer.Viewer'] = None) -> bool:
        """
        Move joints directly to a target and wait for completion.

        Args:
            target_joints: The target joint positions [rad].
            viewer: Optional MuJoCo viewer instance for rendering.
        """
        self.is_moving = True
        try:
            # Get actuator IDs once
            ctrl_indices = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                            for name in self.kinematics.joint_names]

            # Set the actuator to drive towards the target position
            for j, ctrl_id in enumerate(ctrl_indices):
                self.data.ctrl[ctrl_id] = target_joints[j]
            
            # Wait for the movement to complete
            self._wait_for_joints_to_settle(target_joints, viewer)

        except Exception as e:
            print(f"Error during joint move: {e}")
            return False
        finally:
            self.is_moving = False

        return True

    def _move_gripper_to_completion(self, target_position: float, viewer: Optional['mujoco.viewer.Viewer'] = None):
        """
        Move the gripper until it reaches the target or stalls.
        
        Args:
            target_position: The target position for the gripper joint.
            viewer: Optional MuJoCo viewer instance for rendering.
        """
        gripper_actuator_name = "gripper"
        gripper_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_actuator_name)

        if gripper_actuator_id == -1:
            print(f"Warning: Gripper actuator '{gripper_actuator_name}' not found. Setting position directly.")
            self.set_gripper_position(target_position)
            return

        # Set the actuator to drive towards the target position
        self.data.ctrl[gripper_actuator_id] = target_position

        # --- Loop until movement is complete ---
        start_time = time.time()
        timeout = 5.0  # seconds
        
        pos_tolerance = 0.001  # meters
        vel_tolerance = 0.002  # m/s (tighter tolerance)
        stall_steps = 20  # number of consecutive steps with low velocity to detect a stall (more robust)
        stall_counter = 0
        
        # State for settling
        stalled = False
        stall_time = 0.0
        settle_duration = 0.2 # seconds to hold after stall detected

        while time.time() - start_time < timeout:
            if viewer and not viewer.is_running():
                break

            current_pos = self.get_gripper_position()
            current_vel = self.get_gripper_velocity()

            # 1. Check if target position is reached
            if abs(current_pos - target_position) < pos_tolerance:
                # print("Gripper reached target position.")
                break

            # 2. Check for stall (especially when closing)
            if target_position < current_pos:  # Only check for stall when closing
                if abs(current_vel) < vel_tolerance:
                    stall_counter += 1
                else:
                    stall_counter = 0
                
                if stall_counter >= stall_steps:
                    if not stalled:
                        # First time stall detected
                        stalled = True
                        stall_time = time.time()
                        # print("Gripper stall detected, settling...")
                    
                    # Wait for settle duration to ensure stable grasp
                    if time.time() - stall_time > settle_duration:
                        # print("Gripper stalled (likely gripped an object).")
                        break
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
            # Render
            if viewer:
                viewer.sync()
                
            # Sleep to maintain control rate
            time.sleep(self.control_dt)
        else:
            print("Warning: Gripper move timed out.")
    
    def _execute_joint_trajectory(self, time_points: np.ndarray, joint_trajectory: np.ndarray, viewer: Optional['mujoco.viewer.Viewer'] = None) -> bool:
        """
        Execute joint trajectory and wait for completion.
        
        Args:
            time_points: Array of time points for the trajectory.
            joint_trajectory: Array of joint configurations for the trajectory.
            viewer: Optional MuJoCo viewer instance for rendering.
        """
        self.is_moving = True
        
        try:
            start_time = time.time()
            
            # Get actuator IDs once
            ctrl_indices = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) 
                            for name in self.kinematics.joint_names]
            
            for i, target_joints in enumerate(joint_trajectory):
                if self.emergency_stop_flag:
                    print("Motion stopped due to emergency stop")
                    break
                
                if viewer and not viewer.is_running():
                    print("Viewer closed, stopping trajectory.")
                    break
                    
                # Set target position for controllers
                for j, ctrl_id in enumerate(ctrl_indices):
                    self.data.ctrl[ctrl_id] = target_joints[j]
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
                # Render if viewer is provided
                if viewer:
                    viewer.sync()
                
                # Timing control
                elapsed = time.time() - start_time
                expected_time = time_points[i]
                sleep_time = expected_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # After trajectory, wait for joints to settle at the final position
            if not self.emergency_stop_flag:
                final_target_joints = joint_trajectory[-1]
                self._wait_for_joints_to_settle(final_target_joints, viewer)
                    
        except Exception as e:
            print(f"Error during trajectory execution: {e}")
            return False
        finally:
            self.is_moving = False
            
        return True
