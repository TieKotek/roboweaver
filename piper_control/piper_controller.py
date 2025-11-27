"""
AgileX PiPER Specific Controller.
Implements the BaseRobotController interface for the PiPER arm.
"""

import numpy as np
import mujoco
import time
import ikpy.chain
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple, List

# Import the common interface
from common.robot_api import BaseRobotController, RobotState
from dataclasses import dataclass

@dataclass
class PiperState(RobotState):
    joint_positions: Optional[np.ndarray] = None
    end_effector_pose: Optional[np.ndarray] = None
    end_effector_euler: Optional[np.ndarray] = None

class PiperController(BaseRobotController):
    """
    Controller specifically for the AgileX PiPER 6-DOF Arm.
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, robot_name: str = "piper", urdf_path: str = "agilex_piper/piper_description.urdf", base_pos: np.ndarray = None, base_quat: np.ndarray = None, log_dir: str = None):
        super().__init__(model, data, robot_name, base_pos, base_quat, log_dir)
        
        self.prefix = robot_name
        # Initialize Kinematics Helper
        self.kinematics = PiperKinematics(model, urdf_path, prefix=self.prefix)
        
        # --- START: COORDINATE FRAME CORRECTION ---
        # User reported that with Euler angles [0,0,0], the end-effector points along the World Y-axis.
        # This correction re-aligns the "Tool Frame" (logical control frame) so that [0,0,0] Euler
        # angles correspond to the end-effector pointing along the World X-axis, which is a common
        # convention for robot tool frames.
        # If the physical robot's default orientation (Y-forward) is intended, this block can be removed.
        #
        # The physical End Effector (Link6) has its functional axis along Y (based on user report).
        # The User "Tool Frame" expects the functional axis along X.
        # We define a fixed rotation between Tool Frame and Link6 Frame:
        # We need Link6's Y-axis (geometry forward) to align with Tool's X-axis (control forward).
        # This corresponds to a rotation of -90 degrees around the Z-axis.
        self.r_tool_link6 = R.from_euler('z', -90, degrees=True) # Rotation from Tool Frame to Link6 Frame
        self.r_link6_tool = self.r_tool_link6.inv()            # Rotation from Link6 Frame to Tool Frame
        # --- END: COORDINATE FRAME CORRECTION ---

        self.home_joints = np.array([0, 1.57, -1.3485, 0, 0, 0])
        self._reset_to_home()

    # --- Interface Implementation ---

    def get_robot_state(self) -> PiperState:
        # FK returns Pose of Link6 in Base Frame
        ee_pose_link6 = self.kinematics.forward_kinematics(self._get_joint_positions())
        
        # --- START: COORDINATE FRAME CORRECTION ---
        # Convert Link6 Pose -> Tool Pose
        # P_tool = P_link6 (assuming origin is same, just rotation)
        # R_world_tool = R_world_link6 * R_link6_tool
        # This converts the raw FK result (Link6 frame) into the user-friendly Tool Frame
        # (where [0,0,0] Euler means X-forward).
        pos = ee_pose_link6[:3]
        quat_link6 = ee_pose_link6[3:] # [x, y, z, w]
        
        r_world_link6 = R.from_quat(quat_link6)
        r_world_tool = r_world_link6 * self.r_link6_tool
        
        # Reconstruct State with Tool Frame
        ee_pose_tool = np.concatenate([pos, r_world_tool.as_quat()])
        # --- END: COORDINATE FRAME CORRECTION ---
        
        # Use 'xyz' (Extrinsic) which corresponds to the requested RPY (XYZ fixed axis) format
        euler = r_world_tool.as_euler('xyz', degrees=True)
        
        return PiperState(
            timestamp=time.time(),
            joint_positions=self._get_joint_positions(),
            end_effector_pose=ee_pose_tool,
            end_effector_euler=euler
        )

    def format_state(self) -> str:
        """Custom format with quaternion for orientation."""
        state = self.get_robot_state()
        lines = [f"[{self.robot_name}] State:"]
        lines.append(f"  Timestamp: {state.timestamp:.3f}")
        
        if state.joint_positions is not None:
            lines.append(f"  Joints: {np.array2string(state.joint_positions, precision=3, suppress_small=True)}")

        if state.end_effector_pose is not None:
            pos = state.end_effector_pose[:3]
            quat = state.end_effector_pose[3:] # [x, y, z, w]
            lines.append(f"  EE Pos: {np.array2string(pos, precision=3, suppress_small=True)}")
            lines.append(f"  EE Quat (xyzw): {np.array2string(quat, precision=3)}")
        
        if state.end_effector_euler is not None:
            lines.append(f"  EE Euler (xyz deg): {np.array2string(state.end_effector_euler, precision=3)}")
        return "\n".join(lines)

    def print_state(self):
        print(self.format_state())

    # --- Action Handlers (Called by run_robots.py via execute_action) ---

    def action_move_joints(self, joints: List[float], duration: Optional[float] = None):
        """Handler for 'move_joints' action."""
        self._move_to_joints(np.array(joints), duration)

    def action_move_cartesian(self, pose: List[float], 
                              quat: Optional[List[float]] = None, 
                              euler: Optional[List[float]] = None, 
                              duration: Optional[float] = None,
                              **kwargs):
        """
        Handler for 'move_cartesian' action. Pose is in World Frame.
        pose: [x, y, z]
        quat: [x, y, z, w] (Optional)
        euler: [x, y, z] angles (Optional) - interpreted as RPY (Extrinsic xyz in SciPy)
        """
        # Check exclusivity
        if quat is not None and euler is not None:
            print(f"[{self.robot_name}] Error: Cannot provide both 'quat' and 'euler'.")
            return

        # Convert World Pose to Local (Robot Base) Pose
        target_pos_world = np.array(pose[:3])
        
        orientation_needed = False
        r_world_tool = R.identity() # Default to Identity Tool Frame

        if quat is not None:
            orientation_needed = True
            r_world_tool = R.from_quat(quat)
        elif euler is not None:
            orientation_needed = True
            # Use 'xyz' to match requested RPY standard
            r_world_tool = R.from_euler('xyz', euler, degrees=True)
        
        # --- START: COORDINATE FRAME CORRECTION ---
        # Convert User's Tool Frame target -> Link6 Frame target for IK solver.
        # R_world_link6 = R_world_tool * R_tool_link6
        # This transforms the user's desired orientation (in the X-forward Tool Frame)
        # into the corresponding orientation for Link6 (which is Y-forward in its own frame).
        r_world_link6 = r_world_tool * self.r_tool_link6
        # --- END: COORDINATE FRAME CORRECTION ---

        # Base Transform
        r_base = R.from_quat([self.base_quat[1], self.base_quat[2], self.base_quat[3], self.base_quat[0]]) # scalar-last in scipy!
        # Note: self.base_quat is [w, x, y, z] (mujoco default), scipy expects [x, y, z, w]
        
        # T_local = T_base_inv * T_world
        
        # Position: p_local = R_base_inv * (p_world - p_base)
        # Assuming Tool Origin == Link6 Origin
        p_local = r_base.inv().apply(target_pos_world - self.base_pos)
        
        # Rotation: R_local = R_base_inv * R_world_link6
        r_local = r_base.inv() * r_world_link6
        quat_local = r_local.as_quat() # [x, y, z, w]
        
        # Construct target pose for IK
        target_pose_local = np.concatenate([p_local, quat_local])
        
        # Call IK with Local Pose
        current_joints = self._get_joint_positions()
        target_joints = self.kinematics.inverse_kinematics(
            target_pose_local, current_joints, orientation_needed
        )
        
        if target_joints is None:
            print(f"[{self.robot_name}] IK Failed. Target unreachable.")
            return

        self._move_to_joints(target_joints, duration)

    def action_open_gripper(self):
        """Handler for 'open_gripper'."""
        # 0.035 is max open width for Piper
        self._move_gripper(0.035)

    def action_close_gripper(self):
        """Handler for 'close_gripper'."""
        self._move_gripper(0.0)
        
    def action_home(self, duration: float = 3.0):
        """Handler for 'home'."""
        print(f"[{self.robot_name}] Homing...")
        self._move_to_joints(self.home_joints, duration)

    # --- Internal Logic ---

    def _reset_to_home(self):
        """Instant reset (teleport) to home."""
        self._set_joints_direct(self.home_joints)
        self._set_gripper_ctrl(0.0)
        
        # Also set control signals to prevent jumping when physics starts
        self._update_ctrl(self.home_joints)

    def _get_joint_positions(self) -> np.ndarray:
        return np.array([self.data.qpos[self.model.jnt_qposadr[i]] for i in self.kinematics.joint_ids])

    def _set_joints_direct(self, positions: np.ndarray):
        for i, j_id in enumerate(self.kinematics.joint_ids):
            addr = self.model.jnt_qposadr[j_id]
            self.data.qpos[addr] = positions[i]

    def _update_ctrl(self, positions: np.ndarray):
        """Update MuJoCo control inputs."""
        for i, ctrl_id in enumerate(self.kinematics.actuator_ids):
            if ctrl_id != -1:
                self.data.ctrl[ctrl_id] = positions[i]

    def _move_to_joints(self, target_joints: np.ndarray, duration: Optional[float]):
        """Smooth move logic."""
        if self.emergency_stop_flag: return

        if not self._check_limits(target_joints):
            print(f"[{self.robot_name}] Target exceeds joint limits!")
            return

        if duration is None:
            # Move fast (direct control + wait)
            self._update_ctrl(target_joints)
            self._wait_settle(target_joints)
            return

        # Interpolated move
        start_joints = self._get_joint_positions()
        steps = int(duration / self.control_dt)
        
        for t in range(steps):
            if self.emergency_stop_flag: break
            alpha = t / steps
            # Cubic easing (smooth start/stop)
            alpha_smooth = 3*alpha**2 - 2*alpha**3
            
            interp_joints = start_joints + alpha_smooth * (target_joints - start_joints)
            self._update_ctrl(interp_joints)
            time.sleep(self.control_dt)
            
        self._wait_settle(target_joints)

    def _set_gripper_ctrl(self, val: float):
        """Directly set gripper control signal."""
        if self.kinematics.gripper_actuator_id != -1:
            self.data.ctrl[self.kinematics.gripper_actuator_id] = val
        else:
            self.data.qpos[self.kinematics.gripper_id] = val

    def _move_gripper(self, target_position: float):
        """
        Advanced gripper control with stall detection.
        Waits until the gripper reaches the target OR stalls (grips object).
        """
        self._set_gripper_ctrl(target_position)

        start_time = time.time()
        timeout = 2.0
        
        pos_tolerance = 0.001
        vel_tolerance = 0.002
        stall_steps = 10
        stall_counter = 0
        
        stalled = False
        stall_time = 0.0
        settle_duration = 0.5 # Hold force for a bit to stabilize grasp

        while time.time() - start_time < timeout:
            if self.emergency_stop_flag: break

            # Get gripper state
            # Note: For slide joint, qpos is length
            current_pos = self.data.qpos[self.kinematics.gripper_id]
            current_vel = self.data.qvel[self.kinematics.gripper_id]

            # 1. Target Reached?
            if abs(current_pos - target_position) < pos_tolerance:
                break

            # 2. Stall Detection (Only relevant when closing, i.e., target < current)
            if target_position < current_pos: 
                if abs(current_vel) < vel_tolerance:
                    stall_counter += 1
                else:
                    stall_counter = 0
                
                if stall_counter >= stall_steps:
                    if not stalled:
                        stalled = True
                        stall_time = time.time()
                        # print(f"[{self.robot_name}] Gripper stall detected.")
                    
                    if time.time() - stall_time > settle_duration:
                        # print(f"[{self.robot_name}] Grasp stabilized.")
                        break
            
            time.sleep(self.control_dt)

    def _check_limits(self, joints: np.ndarray) -> bool:
        return np.all(joints >= self.kinematics.limits[:, 0]) and \
               np.all(joints <= self.kinematics.limits[:, 1])

    def _wait_settle(self, target_joints: np.ndarray):
        """Wait for physical joints to reach target."""
        timeout = 2.0
        start = time.time()
        while time.time() - start < timeout:
            curr = self._get_joint_positions()
            if np.allclose(curr, target_joints, atol=0.1):
                return
            time.sleep(0.01)


class PiperKinematics:
    """Helper class to isolate IK/FK/ID lookups from logic."""
    def __init__(self, model, urdf_path, prefix=""):
        self.model = model
        
        # 1. Setup Names
        # If prefix is "left_arm", joint is "left_arm_joint1"
        p = f"{prefix}_" if prefix else ""
        self.joint_names = [f"{p}joint{i}" for i in range(1, 7)]
        self.gripper_name = f"{p}joint7"
        self.ee_name = f"{p}link6"

        # 2. Get IDs
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        self.gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.gripper_name)
        
        # Actuator IDs might differ from joint IDs
        self.actuator_ids = []
        for n in self.joint_names:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            if aid == -1:
                print(f"WARNING: Actuator '{n}' not found!")
            self.actuator_ids.append(aid)
        
        gripper_act_name = f"{p}gripper" if prefix else "gripper"
        self.gripper_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_act_name)
        if self.gripper_actuator_id == -1:
             print(f"WARNING: Gripper Actuator '{gripper_act_name}' not found!")

        # 3. IK Chain
        # Note: IKPy chain mask needs to match the URDF structure
        active_links_mask = [False, True, True, True, True, True, True, False, False]
        self.chain = ikpy.chain.Chain.from_urdf_file(urdf_path, active_links_mask=active_links_mask)

        self.limits = np.array([
            [-2.618, 2.618], [0, 3.14], [-2.697, 0],
            [-1.832, 1.832], [-1.22, 1.22], [-3.14, 3.14]
        ])

    def forward_kinematics(self, joints):
        full = np.zeros(9)
        full[1:7] = joints
        mat = self.chain.forward_kinematics(full)
        pos = mat[:3, 3]
        quat = R.from_matrix(mat[:3, :3]).as_quat() # [x, y, z, w]
        return np.concatenate([pos, quat])

    def inverse_kinematics(self, target_pose, current_joints, orientation_needed):
        # target_pose is [x, y, z, qx, qy, qz, qw] or [x, y, z]
        if len(target_pose) == 3:
            target_pose = np.concatenate([target_pose, [0, 0, 0, 1]]) # Identity quat [x,y,z,w]
            
        pos = target_pose[:3]
        # target_pose[3:] should be [x, y, z, w]
        rot = R.from_quat(target_pose[3:]).as_matrix()
        target_mat = np.eye(4)
        target_mat[:3, :3] = rot
        target_mat[:3, 3] = pos

        initial = np.zeros(9)
        initial[1:7] = current_joints

        try:
            sol = self.chain.inverse_kinematics_frame(
                target_mat, initial_position=initial, 
                orientation_mode="all" if orientation_needed else None
            )
            return sol[1:7]
        except:
            return None