"""
AgileX PiPER Specific Controller.
Implements the BaseRobotController interface for the PiPER arm.
"""

import numpy as np
import mujoco
import time
import ikpy.chain
from scipy.spatial.transform import Rotation as R, Slerp
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
        self.r_tool_link6 = R.from_euler('z', -90, degrees=True) # Rotation from Tool Frame to Link6 Frame
        self.r_link6_tool = self.r_tool_link6.inv()            # Rotation from Link6 Frame to Tool Frame
        self.kinematics = PiperKinematics(model, urdf_path, prefix=self.prefix)

        self.home_joints = np.array([0, 1.57, -1.3485, 0, 0, 0])
        self._reset_to_home()

    # --- Interface Implementation ---

    def get_robot_state(self) -> PiperState:
        # FK returns Pose of Link6 in Base Frame
        ee_pose_link6 = self.kinematics.forward_kinematics(self._get_joint_positions())
        pos = ee_pose_link6[:3]
        quat_link6 = ee_pose_link6[3:] # [x, y, z, w]
        
        r_world_link6 = R.from_quat(quat_link6)
        r_world_tool = r_world_link6 * self.r_link6_tool
        
        # Reconstruct State with Tool Frame
        ee_pose_tool = np.concatenate([pos, r_world_tool.as_quat()])
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
    
        r_world_link6 = r_world_tool * self.r_tool_link6

        # Base Transform
        r_base = R.from_quat([self.base_quat[1], self.base_quat[2], self.base_quat[3], self.base_quat[0]]) # scalar-last in scipy!
        # Note: self.base_quat is [w, x, y, z] (mujoco default), scipy expects [x, y, z, w]
        
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
            msg = f"[{self.robot_name}] CRITICAL: Target {target_pos_world} is out of reach or unreachable with orientation. Action aborted."
            print(msg)
            self.log(msg)
            return

        self._move_to_joints(target_joints, duration)

    def action_move_linear(self, pose: List[float], 
                          quat: Optional[List[float]] = None, 
                          euler: Optional[List[float]] = None, 
                          duration: Optional[float] = None,
                          **kwargs):
        """
        Handler for 'move_linear' action. Moves the end-effector in a straight line 
        to the target pose at a constant speed.
        
        Parameters:
        - pose: [x, y, z] target position in World Frame.
        - quat: [x, y, z, w] target orientation (Optional).
        - euler: [x, y, z] target orientation in degrees (Optional).
        - duration: Total time for the movement. If None, estimated based on distance.
        """
        if quat is not None and euler is not None:
            print(f"[{self.robot_name}] Error: Cannot provide both 'quat' and 'euler'.")
            return

        # 1. Setup Target Orientation
        target_pos_world = np.array(pose[:3])
        orientation_constrained = False
        r_world_tool_end = R.identity()

        if quat is not None:
            orientation_constrained = True
            r_world_tool_end = R.from_quat(quat)
        elif euler is not None:
            orientation_constrained = True
            r_world_tool_end = R.from_euler('xyz', euler, degrees=True)
            
        # 2. Get Current Pose
        current_state = self.get_robot_state()
        start_pos_world = current_state.end_effector_pose[:3]
        r_world_tool_start = R.from_quat(current_state.end_effector_pose[3:])

        # If target orientation isn't provided, maintain current orientation to keep motion stable
        if not orientation_constrained:
            r_world_tool_end = r_world_tool_start
            orientation_constrained = True

        # 3. Timing and Steps
        if duration is None:
            dist = np.linalg.norm(target_pos_world - start_pos_world)
            duration = max(dist / 0.1, 0.5) # Default 0.1 m/s, min 0.5s
        
        steps = int(duration / self.control_dt)
        if steps <= 0: steps = 1

        # 4. Interpolation Setup (SLERP for orientation)
        key_times = [0, 1]
        key_rots = R.from_quat([r_world_tool_start.as_quat(), r_world_tool_end.as_quat()])
        slerp_func = Slerp(key_times, key_rots)

        # Base Transform (for local conversion)
        r_base = R.from_quat([self.base_quat[1], self.base_quat[2], self.base_quat[3], self.base_quat[0]])

        # --- PRE-CHECK: Can we even reach the end? ---
        r_world_link6_end = r_world_tool_end * self.r_tool_link6
        p_local_end = r_base.inv().apply(target_pos_world - self.base_pos)
        r_local_end = r_base.inv() * r_world_link6_end
        target_pose_local_end = np.concatenate([p_local_end, r_local_end.as_quat()])
        
        test_joints = self.kinematics.inverse_kinematics(
            target_pose_local_end, current_state.joint_positions, orientation_needed=True
        )
        if test_joints is None:
            msg = f"[{self.robot_name}] CRITICAL: Linear move endpoint {target_pos_world} is unreachable. Action aborted before execution."
            print(msg)
            self.log(msg)
            return

        # 5. Execution Loop
        print(f"[{self.robot_name}] Executing linear move to {target_pos_world}...")
        action_start = time.perf_counter()
        last_joints = current_state.joint_positions

        for t in range(1, steps + 1):
            if self.emergency_stop_flag: break
            
            alpha = t / steps
            # Linear position interpolation
            interp_pos_world = start_pos_world + alpha * (target_pos_world - start_pos_world)
            # Slerp orientation interpolation
            r_world_tool_interp = slerp_func(alpha)
            
            # Convert to Local Frame and Link6 Frame
            r_world_link6_interp = r_world_tool_interp * self.r_tool_link6
            p_local = r_base.inv().apply(interp_pos_world - self.base_pos)
            r_local = r_base.inv() * r_world_link6_interp
            
            target_pose_local = np.concatenate([p_local, r_local.as_quat()])
            
            # Solve IK (passing previous solution as seed for continuity)
            target_joints = self.kinematics.inverse_kinematics(
                target_pose_local, last_joints, orientation_needed=True
            )
            
            if target_joints is None:
                msg = f"[{self.robot_name}] CRITICAL: Linear path blocked or unreachable at step {t}/{steps}. Stopping."
                print(msg)
                self.log(msg)
                return

            # --- High Precision Sync ---
            # Wait BEFORE issuing the command to ensure commands are delivered at exact intervals
            target_step_time = action_start + (t * self.control_dt)
            while True:
                remaining = target_step_time - time.perf_counter()
                if remaining <= 0:
                    break
                if remaining > 0.002:
                    time.sleep(0.001)

            # --- Precision Delivery ---
            self._update_ctrl(target_joints)
            last_joints = target_joints
        
        self._wait_settle(last_joints)

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
        if steps <= 0: steps = 1
        action_start = time.perf_counter()
        
        for t in range(1, steps + 1):
            if self.emergency_stop_flag: break
            
            alpha = t / steps
            # Cubic easing (smooth start/stop)
            alpha_smooth = 3*alpha**2 - 2*alpha**3
            
            interp_joints = start_joints + alpha_smooth * (target_joints - start_joints)
            
            # --- High Precision Sync ---
            target_step_time = action_start + (t * self.control_dt)
            while True:
                remaining = target_step_time - time.perf_counter()
                if remaining <= 0:
                    break
                if remaining > 0.002:
                    time.sleep(0.001)

            # --- Precision Delivery ---
            self._update_ctrl(interp_joints)
            
        self._wait_settle(target_joints)

    def _set_gripper_ctrl(self, val: float):
        """Directly set gripper control signal."""
        if self.kinematics.gripper_actuator_id != -1:
            self.data.ctrl[self.kinematics.gripper_actuator_id] = val
        else:
            qpos_adr = self.model.jnt_qposadr[self.kinematics.gripper_id]
            self.data.qpos[qpos_adr] = val

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
            qpos_adr = self.model.jnt_qposadr[self.kinematics.gripper_id]
            qvel_adr = self.model.jnt_dofadr[self.kinematics.gripper_id]
            
            current_pos = self.data.qpos[qpos_adr]
            current_vel = self.data.qvel[qvel_adr]

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
        
        # Check if we timed out
        if time.time() - start_time >= timeout:
            # print(f"[{self.robot_name}] Gripper timeout! Target: {target_position}")
            pass

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

        # 4. Precompute Seeds for Multi-Seed IK
        self.seeds = []
        self._init_seeds()

    def _init_seeds(self):
        """
        Pre-calculate joint configurations for standard poses (Forward, Left, Back, Right).
        These serve as restart seeds for IK to avoid local minima during large movements.
        """
        # Relative to base (x, y, z)
        seed_points = [
            [0.4, 0.0, 0.4],   # Forward
            [0.0, 0.4, 0.4],   # Left
            [-0.4, 0.0, 0.4],  # Back
            [0.0, -0.4, 0.4]   # Right
        ]
        
        # Neutral orientation (Identity matrix)
        dummy_rot = np.eye(3)
        
        # Initial guess for the solver to find the seeds (Flat zeros)
        initial_guess = np.zeros(9)
        
        for pt in seed_points:
            target = np.eye(4)
            target[:3, 3] = pt
            target[:3, :3] = dummy_rot
            try:
                # Solve for these "Anchor" poses
                sol = self.chain.inverse_kinematics_frame(target, initial_position=initial_guess)
                self.seeds.append(sol)
            except Exception:
                pass

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

        # --- Multi-Seed IK Strategy ---
        # 1. Try from Current Configuration (Fastest, best for continuity)
        # 2. Try from Pre-computed Seeds (Good for large jumps/orientation flips)
        # 3. Try from Zero/Home (Fallback)
        
        candidates_to_try = []
        
        # Seed 1: Current Joints
        current_full = np.zeros(9)
        current_full[1:7] = current_joints
        candidates_to_try.append(current_full)
        
        # Seed 2..5: Cached Seeds
        candidates_to_try.extend(self.seeds)
        
        # Seed 6: Zero
        candidates_to_try.append(np.zeros(9))

        best_sol = None
        best_error = float('inf')
        acceptance_tolerance = 0.01 # 1cm position error acceptable

        for initial_guess in candidates_to_try:
            try:
                sol = self.chain.inverse_kinematics_frame(
                    target_mat, initial_position=initial_guess, 
                    orientation_mode="all" if orientation_needed else None
                )
                
                # Validation: Check Forward Kinematics of the solution
                fk_mat = self.chain.forward_kinematics(sol)
                pos_error = np.linalg.norm(fk_mat[:3, 3] - pos)
                
                if pos_error < best_error:
                    best_error = pos_error
                    best_sol = sol
                
                # Early exit if good enough
                if best_error < acceptance_tolerance:
                    break
            except:
                continue

        if best_sol is not None:
            return best_sol[1:7]
            
        return None