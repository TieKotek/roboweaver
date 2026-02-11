import numpy as np
import mujoco
import time
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple, List, Dict, Any
from common.robot_api import BaseRobotController, RobotState
from dataclasses import dataclass

@dataclass
class StretchState(RobotState):
    base_pose: np.ndarray = None # [x, y, yaw]
    arm_status: np.ndarray = None # [lift, extend, wrist_pitch, wrist_roll]
    gripper_pos: float = 0.0
    ee_pos: np.ndarray = None # [x, y, z] in world frame

class StretchController(BaseRobotController):
    """
    Controller for the Hello Robot Stretch 3.
    """
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, robot_name: str = "stretch", base_pos: np.ndarray = None, base_quat: np.ndarray = None, log_dir: str = None, **kwargs):
        super().__init__(model, data, robot_name, base_pos, base_quat, log_dir)
        
        # 1. Identify Actuators
        # Note: Stretch 3 uses velocity control for wheels and position control for others.
        self.actuator_names = [
            "left_wheel_vel", "right_wheel_vel", "lift", "arm", 
            "wrist_yaw", "wrist_pitch", "wrist_roll", "gripper", 
            "head_pan", "head_tilt"
        ]
        self.act_ids = {}
        for name in self.actuator_names:
            full_name = f"{robot_name}_{name}" if robot_name else name
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full_name)
            if aid == -1: aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.act_ids[name] = aid

        # 2. Identify Bodies and Joints with prefix support
        def get_id(name, obj_type):
            full_name = f"{robot_name}_{name}" if robot_name else name
            oid = mujoco.mj_name2id(model, obj_type, full_name)
            if oid == -1: oid = mujoco.mj_name2id(model, obj_type, name)
            return oid

        self.body_id = get_id("base_link", mujoco.mjtObj.mjOBJ_BODY)
        self.ee_id = get_id("link_grasp_center", mujoco.mjtObj.mjOBJ_BODY)
        self.gripper_joint_id = get_id("joint_gripper_slide", mujoco.mjtObj.mjOBJ_JOINT)
        
        if self.ee_id == -1:
            print(f"[{self.robot_name}] WARNING: 'link_grasp_center' not found. ee_pos will be inaccurate.")

        # 2. Capture Initial State as 'Home' Pose
        # This captures whatever state the model was in when loaded (e.g., from XML or keyframe)
        self.home_ctrl = {}
        for name, aid in self.act_ids.items():
            if aid != -1:
                self.home_ctrl[name] = self.data.ctrl[aid]
        
        # 3. Default Speeds & Home Pose
        self.move_speed = 1.0 
        self.rotate_speed = 1.0
        self.ee_speed = 0.1    # Default end-effector speed in m/s
        self.home_pose = [0.6, 0.1, 0.0, 0.0] # lift, extend, pitch, roll
        
        # 4. Initialize State (Force qpos to match ctrl for "Instant Load")
        full_key_name = f"{self.robot_name}_home" if self.robot_name else "home"
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, full_key_name)
        
        # Fallback to unprefixed 'home' if prefixed not found (for standalone use)
        if key_id == -1:
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        
        # Determine target values
        targets = {}
        if key_id != -1:
            # Get targets from keyframe
            # Use relative index because keyframes in merged XMLs might be padded with zeros at the end,
            # putting this robot's local keyframe values at the beginning of the global key_ctrl vector.
            for i, name in enumerate(self.actuator_names):
                aid = self.act_ids.get(name, -1)
                if aid != -1:
                    targets[name] = self.model.key_ctrl[key_id, i]
        else:
            # Manual defaults
            targets = {"lift": 0.6, "arm": 0.1, "wrist_yaw": 0.0, "wrist_pitch": 0.0, "wrist_roll": 0.0}

        # Apply to BOTH ctrl and qpos
        for name, val in targets.items():
            aid = self.act_ids.get(name, -1)
            if aid == -1: continue
            
            # Set control target
            self.data.ctrl[aid] = val
            
            # Set physical position (qpos)
            trn_type = self.model.actuator_trntype[aid]
            joint_id = self.model.actuator_trnid[aid, 0]
            
            if trn_type == mujoco.mjtTrn.mjTRN_JOINT:
                qpos_addr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_addr] = val
            elif trn_type == mujoco.mjtTrn.mjTRN_TENDON:
                # Arm extension distribution
                for i in range(4):
                    j_name = f"joint_arm_l{i}"
                    jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
                    if jid != -1:
                        self.data.qpos[self.model.jnt_qposadr[jid]] = val / 4.0
        
        mujoco.mj_forward(model, self.data)
        print(f"[{self.robot_name}] Initial state locked (Keyframe: {key_id != -1}).")

    def get_robot_state(self) -> StretchState:
        pos = self.data.xpos[self.body_id]
        quat = self.data.xquat[self.body_id]
        w, x, y, z = quat
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        
        # Get control values for status
        lift = self.data.ctrl[self.act_ids["lift"]] if self.act_ids["lift"] != -1 else 0
        arm = self.data.ctrl[self.act_ids["arm"]] if self.act_ids["arm"] != -1 else 0
        pitch = self.data.ctrl[self.act_ids["wrist_pitch"]] if self.act_ids["wrist_pitch"] != -1 else 0
        roll = self.data.ctrl[self.act_ids["wrist_roll"]] if self.act_ids["wrist_roll"] != -1 else 0
        
        g_pos = 0.0
        if self.gripper_joint_id != -1:
            g_pos = self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_id]]

        ee_pos = self.data.xpos[self.ee_id].copy() if self.ee_id != -1 else np.zeros(3)

        return StretchState(
            timestamp=self.data.time,
            base_pose=np.array([pos[0], pos[1], yaw]),
            arm_status=np.array([lift, arm, pitch, roll]),
            gripper_pos=g_pos,
            ee_pos=ee_pos
        )

    def action_move_base(self, distance: float, speed: float = 5.0):
        if self.emergency_stop_flag: return
        print(f"[{self.robot_name}] Moving base {distance}m...")
        start_pos = self.data.xpos[self.body_id].copy()
        direction = np.sign(distance)
        abs_dist = abs(distance)
        sim_start_time = self.data.time
        step = 0
        
        # Velocity control for base
        while True:
            if self.emergency_stop_flag: break
            curr_pos = self.data.xpos[self.body_id]
            if np.linalg.norm(curr_pos[:2] - start_pos[:2]) >= abs_dist: break
            
            # Differential drive velocity (simplified forward)
            v = direction * speed
            self._update_ctrl_dict({"left_wheel_vel": v, "right_wheel_vel": v})
            
            step += 1
            self._sync_sim(sim_start_time, step)
            
        self._update_ctrl_dict({"left_wheel_vel": 0.0, "right_wheel_vel": 0.0})
        self._wait_for_base_stop()

    def action_rotate_base(self, angle_deg: float, speed: float = 5.0):
        if self.emergency_stop_flag: return
        print(f"[{self.robot_name}] Rotating base {angle_deg} deg...")
        target_rad = np.deg2rad(angle_deg)
        _, _, start_yaw = self._get_base_pose()
        accumulated = 0.0
        last_yaw = start_yaw
        direction = np.sign(angle_deg)
        sim_start_time = self.data.time
        step = 0
        
        while abs(accumulated) < abs(target_rad):
            if self.emergency_stop_flag: break
            _, _, curr_yaw = self._get_base_pose()
            diff = (curr_yaw - last_yaw + np.pi) % (2 * np.pi) - np.pi
            accumulated += diff
            last_yaw = curr_yaw
            
            # Rotation: wheels move in opposite directions
            v_l = -direction * speed
            v_r = direction * speed
            self._update_ctrl_dict({"left_wheel_vel": v_l, "right_wheel_vel": v_r})
            
            step += 1
            self._sync_sim(sim_start_time, step)
            
        self._update_ctrl_dict({"left_wheel_vel": 0.0, "right_wheel_vel": 0.0})
        self._wait_for_base_stop()

    def action_open_gripper(self):
        # Stretch 3 gripper range: -0.02 to 0.04
        self._move_gripper(0.04)

    def action_close_gripper(self):
        self._move_gripper(-0.01)

    def action_move_gripper(self, position: float):
        """Flexible gripper control."""
        # Clamp to physical limits
        pos = max(-0.02, min(0.04, position))
        self._move_gripper(pos)

    def action_move_arm(self, lift: float, arm: float, ee_speed: float = None):
        """
        Move end-effector in a straight line at constant velocity.
        lift: target height (m) [0.0 - 1.1]
        arm: target extension (m) [0.0 - 0.52]
        ee_speed: combined end-effector speed in m/s (safe range: 0.01 - 0.3)
        """
        if self.emergency_stop_flag: return
        
        # 1. Target Clamping
        target_lift = max(0.0, min(1.1, lift))
        target_arm = max(0.0, min(0.52, arm))
        
        # 2. Get Current Positions
        start_lift = self.data.ctrl[self.act_ids["lift"]]
        start_ext = self.data.ctrl[self.act_ids["arm"]]
        
        # 3. Calculate Euclidean Distance
        dist = np.sqrt((target_lift - start_lift)**2 + (target_arm - start_ext)**2)
        
        if dist < 0.001:
            self._update_ctrl_dict({"lift": target_lift, "arm": target_arm})
            return

        # 4. Determine and Validate Speed
        speed = ee_speed if ee_speed is not None else self.ee_speed
        min_s, max_s = 0.01, 0.3
        if not (min_s <= speed <= max_s):
            print(f"[{self.robot_name}] WARNING: ee_speed {speed} is outside safe range [{min_s}, {max_s}]. Clamping.")
            speed = max(min_s, min(max_s, speed))

        # 5. Calculate Duration
        duration = dist / speed

        print(f"[{self.robot_name}] Moving arm: ({start_lift:.2f}, {start_ext:.2f}) -> ({target_lift:.2f}, {target_arm:.2f}) at {speed:.2f} m/s (duration={duration:.2f}s)")
        self._linear_move_arm(target_lift, target_arm, duration)

    def action_move_ee(self, reach: float, height: float, ee_speed: float = None):
        """
        Precisely move end-effector to target reach and height.
        reach: horizontal distance from base center (m) [0.5 - 0.9]
        height: vertical distance from ground (m) [0.3 - 1.0]
        ee_speed: constant velocity in m/s [0.01 - 0.3]
        """
        if self.emergency_stop_flag: return

        # 1. Validate and Clamp Inputs
        r_range = (0.5, 0.9)
        h_range = (0.3, 1.05)
        s_range = (0.01, 0.3)

        target_r = max(r_range[0], min(r_range[1], reach))
        target_h = max(h_range[0], min(h_range[1], height))
        speed = max(s_range[0], min(s_range[1], ee_speed if ee_speed is not None else self.ee_speed))

        if target_r != reach: print(f"[{self.robot_name}] Reach {reach} clamped to {target_r}")
        if target_h != height: print(f"[{self.robot_name}] Height {height} clamped to {target_h}")

        # 2. Solve for (lift, arm) using Numerical IK
        t_lift, t_arm = self._solve_ik_2d(target_r, target_h)
        
        # 3. Execute linear move
        # Calculate Euclidean distance in Reach-Height space for timing
        state = self.get_robot_state()
        curr_base_pos = self.data.xpos[self.body_id]
        curr_r = np.linalg.norm(state.ee_pos[:2] - curr_base_pos[:2])
        curr_h = state.ee_pos[2] - curr_base_pos[2]
        dist = np.sqrt((target_r - curr_r)**2 + (target_h - curr_h)**2)
        
        if dist < 0.001:
            self._update_ctrl_dict({"lift": t_lift, "arm": t_arm})
            return

        duration = dist / speed
        print(f"[{self.robot_name}] EE Move: Reach {curr_r:.3f}->{target_r:.3f}, Height {curr_h:.3f}->{target_h:.3f} (duration={duration:.2f}s)")
        self._linear_move_arm(t_lift, t_arm, duration)

    def action_home(self):
        print(f"[{self.robot_name}] Homing (Stretch 3)...")
        # Reset head and wrist
        self._update_ctrl_dict({
            "wrist_yaw": 0.0, 
            "wrist_pitch": 0.0, 
            "wrist_roll": 0.0,
            "head_pan": 0.0,
            "head_tilt": 0.0
        })
        # Move to home pose with default ee_speed
        self.action_move_arm(self.home_pose[0], self.home_pose[1])

    # --- Internal Helpers ---

    def _solve_ik_2d(self, target_reach, target_height, max_iter=10, tol=0.0005):
        """
        Numerical 2D IK to find (lift, arm) for (reach, height).
        Does not affect self.data until applied.
        """
        # Create a temporary data for probing and copy state manually for compatibility
        tmp_data = mujoco.MjData(self.model)
        tmp_data.qpos[:] = self.data.qpos[:]
        tmp_data.ctrl[:] = self.data.ctrl[:]
        
        # Initial guess from current control
        c_lift = self.data.ctrl[self.act_ids["lift"]]
        c_arm = self.data.ctrl[self.act_ids["arm"]]

        for _ in range(max_iter):
            # Forward kinematics at current guess
            self._set_lift_arm_qpos(tmp_data, c_lift, c_arm)
            mujoco.mj_forward(self.model, tmp_data)
            
            ee_p = tmp_data.xpos[self.ee_id]
            curr_base_p = tmp_data.xpos[self.body_id]
            curr_r = np.linalg.norm(ee_p[:2] - curr_base_p[:2])
            curr_h = ee_p[2] - curr_base_p[2]
            
            err = np.array([target_reach - curr_r, target_height - curr_h])
            if np.linalg.norm(err) < tol: break
            
            # Numerical Jacobian
            eps = 1e-4
            # Perturb lift
            self._set_lift_arm_qpos(tmp_data, c_lift + eps, c_arm)
            mujoco.mj_forward(self.model, tmp_data)
            ee_p1 = tmp_data.xpos[self.ee_id]
            r1 = np.linalg.norm(ee_p1[:2] - curr_base_p[:2])
            h1 = ee_p1[2] - curr_base_p[2]
            
            # Perturb arm
            self._set_lift_arm_qpos(tmp_data, c_lift, c_arm + eps)
            mujoco.mj_forward(self.model, tmp_data)
            ee_p2 = tmp_data.xpos[self.ee_id]
            r2 = np.linalg.norm(ee_p2[:2] - curr_base_p[:2])
            h2 = ee_p2[2] - curr_base_p[2]
            
            J = np.array([
                [(r1 - curr_r)/eps, (r2 - curr_r)/eps],
                [(h1 - curr_h)/eps, (h2 - curr_h)/eps]
            ])
            
            try:
                delta = np.linalg.solve(J, err)
                c_lift = max(0.0, min(1.1, c_lift + delta[0]))
                c_arm = max(0.0, min(0.52, c_arm + delta[1]))
            except np.linalg.LinAlgError:
                break
        
        return c_lift, c_arm

    def _set_lift_arm_qpos(self, data_obj, lift_val, arm_val):
        """Helper to set qpos for probe data."""
        # Lift
        aid_l = self.act_ids["lift"]
        jid_l = self.model.actuator_trnid[aid_l, 0]
        data_obj.qpos[self.model.jnt_qposadr[jid_l]] = lift_val
        
        # Arm (Distributed to 4 joints via tendon)
        for i in range(4):
            j_name = f"joint_arm_l{i}"
            full_j_name = f"{self.robot_name}_{j_name}" if self.robot_name else j_name
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, full_j_name)
            if jid != -1:
                data_obj.qpos[self.model.jnt_qposadr[jid]] = arm_val / 4.0

    def _get_base_pose(self):
        pos = self.data.xpos[self.body_id]
        quat = self.data.xquat[self.body_id]
        w, x, y, z = quat
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return pos[0], pos[1], yaw

    def _update_ctrl_dict(self, vals: Dict[str, float]):
        for name, val in vals.items():
            if name in self.act_ids and self.act_ids[name] != -1:
                self.data.ctrl[self.act_ids[name]] = val

    def _linear_move_arm(self, target_lift, target_ext, duration):
        sim_start_time = self.data.time
        start_lift = self.data.ctrl[self.act_ids["lift"]]
        start_ext = self.data.ctrl[self.act_ids["arm"]]
        steps = int(duration / self.control_dt)
        if steps <= 0: steps = 1
        for t in range(1, steps + 1):
            if self.emergency_stop_flag: break
            alpha = t / steps
            # Linear interpolation for constant velocity
            curr_l = start_lift + alpha * (target_lift - start_lift)
            curr_e = start_ext + alpha * (target_ext - start_ext)
            self._update_ctrl_dict({"lift": curr_l, "arm": curr_e})
            self._sync_sim(sim_start_time, t)
        
        self._wait_arm_settle(target_lift, target_ext)

    def _wait_for_base_stop(self, timeout=2.0):
        """Wait until base velocity is near zero."""
        if self.body_id == -1: return
        sim_start = self.data.time
        while self.data.time - sim_start < timeout:
            # Check velocity of the base link (6 DOF for freejoint)
            dof_adr = self.model.body_dofadr[self.body_id]
            qvel_base = self.data.qvel[dof_adr:dof_adr+6]
            if np.linalg.norm(qvel_base) < 0.01: break
            time.sleep(0.001)

    def _wait_arm_settle(self, target_lift, target_ext, timeout=1.0):
        """Wait until lift and arm joints reach target positions."""
        sim_start = self.data.time
        # Get lift joint address
        aid_l = self.act_ids.get("lift", -1)
        if aid_l == -1: return
        jid_l = self.model.actuator_trnid[aid_l, 0]
        qadr_l = self.model.jnt_qposadr[jid_l]
        
        # Get arm joint address (joint_arm_l0)
        jid_a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.robot_name}_joint_arm_l0" if self.robot_name else "joint_arm_l0")
        qadr_a = self.model.jnt_qposadr[jid_a] if jid_a != -1 else -1
        
        while self.data.time - sim_start < timeout:
            curr_l = self.data.qpos[qadr_l]
            # Each telescopic joint takes 1/4 of the total extension
            curr_a = self.data.qpos[qadr_a] * 4.0 if qadr_a != -1 else target_ext
            
            # Require both lift and arm to be within tolerance before returning
            if abs(curr_l - target_lift) < 0.002 and abs(curr_a - target_ext) < 0.005:
                break
            time.sleep(0.001)

    def _move_gripper(self, target_pos: float):
        self._update_ctrl_dict({"gripper": target_pos})
        sim_start = self.data.time
        timeout = 2.0
        while self.data.time - sim_start < timeout:
            if self.emergency_stop_flag: break
            if self.gripper_joint_id != -1:
                curr = self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_id]]
                if abs(curr - target_pos) < 0.002: break
            time.sleep(0.001)

    def _sync_sim(self, start_sim_time, step_count):
        target_sim_time = start_sim_time + (step_count * self.control_dt)
        while self.data.time < target_sim_time:
            if self.emergency_stop_flag: break
            time.sleep(0.001)
