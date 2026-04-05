import math
import time
from dataclasses import dataclass
from typing import Dict

import mujoco
import numpy as np

from common.robot_api import BaseRobotController, RobotState


@dataclass
class StretchState(RobotState):
    base_pose: np.ndarray = None  # [x, y, yaw]
    arm_status: np.ndarray = None  # [lift, extend, wrist_pitch, wrist_roll]
    gripper_pos: float = 0.0
    ee_pos: np.ndarray = None  # [x, y, z] in world frame


class StretchController(BaseRobotController):
    """Controller for the Hello Robot Stretch 3."""

    # Base speed caps are conservative user-facing limits. Runtime safe limits
    # are derived from wheel actuator capability * SAFETY_SPEED_FACTOR.
    SAFETY_SPEED_FACTOR = 0.8
    DEFAULT_LINEAR_SPEED = 0.08
    MAX_LINEAR_SPEED = 0.10
    DEFAULT_ANGULAR_SPEED_DEG = 20.0
    MAX_ANGULAR_SPEED_DEG = 27.0
    LINEAR_ACCEL = 0.25
    ANGULAR_ACCEL_DEG = 60.0
    TURN_KP = 0.35
    LINEAR_DISTANCE_TOLERANCE = 0.025
    HEADING_TOLERANCE_DEG = 2.5
    MAX_COMPLETION_OVERRUN = 4.0
    LINEAR_SETTLE_SPEED = 0.03
    ANGULAR_SETTLE_SPEED_DEG = 3.0
    WHEEL_RADIUS = 0.05
    WHEEL_TRACK = 0.3407
    WHEEL_GEAR = 3.0
    GRIPPER_MOVE_DURATION = 2.0
    MIN_EE_SPEED = 0.01
    MAX_EE_SPEED = 0.30
    DEFAULT_EE_SPEED = 0.10

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_name: str = "stretch",
        base_pos: np.ndarray = None,
        base_quat: np.ndarray = None,
        log_dir: str = None,
        **kwargs,
    ):
        super().__init__(model, data, robot_name, base_pos, base_quat, log_dir)

        self.actuator_names = [
            "left_wheel_vel",
            "right_wheel_vel",
            "lift",
            "arm",
            "wrist_yaw",
            "wrist_pitch",
            "wrist_roll",
            "gripper",
            "head_pan",
            "head_tilt",
        ]
        self.act_ids = {}
        for name in self.actuator_names:
            full_name = f"{robot_name}_{name}" if robot_name else name
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full_name)
            if aid == -1:
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.act_ids[name] = aid

        def get_id(name, obj_type):
            full_name = f"{robot_name}_{name}" if robot_name else name
            oid = mujoco.mj_name2id(model, obj_type, full_name)
            if oid == -1:
                oid = mujoco.mj_name2id(model, obj_type, name)
            return oid

        self.body_id = get_id("base_link", mujoco.mjtObj.mjOBJ_BODY)
        self.ee_id = get_id("link_grasp_center", mujoco.mjtObj.mjOBJ_BODY)
        self.gripper_joint_id = get_id("joint_gripper_slide", mujoco.mjtObj.mjOBJ_JOINT)

        if self.ee_id == -1:
            print(f"[{self.robot_name}] WARNING: 'link_grasp_center' not found. ee_pos will be inaccurate.")

        self.home_ctrl = {}
        for name, aid in self.act_ids.items():
            if aid != -1:
                self.home_ctrl[name] = self.data.ctrl[aid]

        self.ee_speed = self.DEFAULT_EE_SPEED
        self.home_pose = [0.6, 0.1, 0.0, 0.0]
        self.safe_linear_speed = float(self.MAX_LINEAR_SPEED)
        self.safe_angular_speed_deg = float(self.MAX_ANGULAR_SPEED_DEG)

        full_key_name = f"{self.robot_name}_home" if self.robot_name else "home"
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, full_key_name)
        if key_id == -1:
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

        targets = {}
        if key_id != -1:
            for i, name in enumerate(self.actuator_names):
                aid = self.act_ids.get(name, -1)
                if aid != -1:
                    targets[name] = self.model.key_ctrl[key_id, i]
        else:
            targets = {"lift": 0.6, "arm": 0.1, "wrist_yaw": 0.0, "wrist_pitch": 0.0, "wrist_roll": 0.0}

        for name, val in targets.items():
            aid = self.act_ids.get(name, -1)
            if aid == -1:
                continue

            self.data.ctrl[aid] = val
            trn_type = self.model.actuator_trntype[aid]
            joint_id = self.model.actuator_trnid[aid, 0]

            if trn_type == mujoco.mjtTrn.mjTRN_JOINT:
                qpos_addr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_addr] = val
            elif trn_type == mujoco.mjtTrn.mjTRN_TENDON:
                for i in range(4):
                    j_name = f"joint_arm_l{i}"
                    jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
                    if jid != -1:
                        self.data.qpos[self.model.jnt_qposadr[jid]] = val / 4.0

        mujoco.mj_forward(model, self.data)
        self._refresh_safe_speed_limits()
        print(f"[{self.robot_name}] Initial state locked (Keyframe: {key_id != -1}).")

    def get_robot_state(self) -> StretchState:
        x, y, yaw = self._get_base_pose()
        lift = self.data.ctrl[self.act_ids["lift"]] if self.act_ids["lift"] != -1 else 0.0
        arm = self.data.ctrl[self.act_ids["arm"]] if self.act_ids["arm"] != -1 else 0.0
        pitch = self.data.ctrl[self.act_ids["wrist_pitch"]] if self.act_ids["wrist_pitch"] != -1 else 0.0
        roll = self.data.ctrl[self.act_ids["wrist_roll"]] if self.act_ids["wrist_roll"] != -1 else 0.0

        gripper_pos = 0.0
        if self.gripper_joint_id != -1:
            gripper_pos = self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_id]]

        ee_pos = self.data.xpos[self.ee_id].copy() if self.ee_id != -1 else np.zeros(3)

        return StretchState(
            timestamp=self.data.time,
            base_pose=np.array([x, y, yaw], dtype=float),
            arm_status=np.array([lift, arm, pitch, roll], dtype=float),
            gripper_pos=float(gripper_pos),
            ee_pos=ee_pos,
        )

    def action_move_base(self, distance: float, speed: float = None):
        if self.emergency_stop_flag:
            return
        target_speed = self._resolve_linear_speed(speed)
        profile = self._build_trapezoidal_profile(abs(float(distance)), target_speed, self.LINEAR_ACCEL)
        print(
            f"[{self.robot_name}] Moving base {distance:.2f}m at {target_speed:.2f}m/s "
            f"(Traj Duration: {profile['duration']:.2f}s)..."
        )
        self._execute_base_profile(profile, mode="linear", direction=self._sign(distance), target_value=float(distance))

    def action_rotate_base(self, angle_deg: float, speed: float = None):
        if self.emergency_stop_flag:
            return
        target_speed_deg = self._resolve_angular_speed(speed)
        profile = self._build_trapezoidal_profile(
            abs(math.radians(float(angle_deg))),
            math.radians(target_speed_deg),
            math.radians(self.ANGULAR_ACCEL_DEG),
        )
        print(
            f"[{self.robot_name}] Rotating base {angle_deg:.2f} deg at {target_speed_deg:.2f}deg/s "
            f"(Traj Duration: {profile['duration']:.2f}s)..."
        )
        self._execute_base_profile(
            profile,
            mode="angular",
            direction=self._sign(angle_deg),
            target_value=math.radians(float(angle_deg)),
        )

    def action_open_gripper(self):
        self.action_move_gripper(0.04)

    def action_close_gripper(self):
        self.action_move_gripper(-0.01)

    def action_move_gripper(self, position: float):
        target = max(-0.02, min(0.04, float(position)))
        print(
            f"[{self.robot_name}] Moving gripper to {target:.3f} "
            f"(Traj Duration: {self.GRIPPER_MOVE_DURATION:.2f}s)..."
        )
        self._execute_gripper_move(target, self.GRIPPER_MOVE_DURATION)

    def action_move_arm(self, lift: float, arm: float, ee_speed: float = None):
        if self.emergency_stop_flag:
            return

        target_lift = max(0.0, min(1.1, float(lift)))
        target_arm = max(0.0, min(0.52, float(arm)))
        start_lift = self.data.ctrl[self.act_ids["lift"]]
        start_arm = self.data.ctrl[self.act_ids["arm"]]
        distance = float(np.hypot(target_lift - start_lift, target_arm - start_arm))
        if distance < 0.001:
            self._update_ctrl_dict({"lift": target_lift, "arm": target_arm})
            return

        speed = self._resolve_ee_speed(ee_speed)
        duration = max(self.control_dt, distance / speed)
        print(
            f"[{self.robot_name}] Moving arm: ({start_lift:.2f}, {start_arm:.2f}) -> "
            f"({target_lift:.2f}, {target_arm:.2f}) at {speed:.2f}m/s "
            f"(Traj Duration: {duration:.2f}s)..."
        )
        self._execute_arm_trajectory(target_lift, target_arm, duration)

    def action_move_ee(self, reach: float, height: float, ee_speed: float = None):
        if self.emergency_stop_flag:
            return

        target_r = max(0.5, min(0.9, float(reach)))
        target_h = max(0.3, min(1.05, float(height)))
        speed = self._resolve_ee_speed(ee_speed)

        if target_r != reach:
            print(f"[{self.robot_name}] Reach {reach} clamped to {target_r}")
        if target_h != height:
            print(f"[{self.robot_name}] Height {height} clamped to {target_h}")

        t_lift, t_arm = self._solve_ik_2d(target_r, target_h)
        state = self.get_robot_state()
        curr_base_pos = self.data.xpos[self.body_id]
        curr_r = np.linalg.norm(state.ee_pos[:2] - curr_base_pos[:2])
        curr_h = state.ee_pos[2] - curr_base_pos[2]
        dist = float(np.hypot(target_r - curr_r, target_h - curr_h))

        if dist < 0.001:
            self._update_ctrl_dict({"lift": t_lift, "arm": t_arm})
            return

        duration = max(self.control_dt, dist / speed)
        print(
            f"[{self.robot_name}] Moving EE to reach={target_r:.3f}, height={target_h:.3f} "
            f"at {speed:.2f}m/s (Traj Duration: {duration:.2f}s)..."
        )
        self._execute_arm_trajectory(t_lift, t_arm, duration)

    def action_home(self):
        print(f"[{self.robot_name}] Homing (Stretch 3)...")
        self._update_ctrl_dict(
            {
                "wrist_yaw": 0.0,
                "wrist_pitch": 0.0,
                "wrist_roll": 0.0,
                "head_pan": 0.0,
                "head_tilt": 0.0,
            }
        )
        self.action_move_arm(self.home_pose[0], self.home_pose[1])

    def _resolve_linear_speed(self, speed: float) -> float:
        value = self.DEFAULT_LINEAR_SPEED if speed is None else float(speed)
        return self._clamp_speed(value, 0.01, self.safe_linear_speed, "linear")

    def _resolve_angular_speed(self, speed_deg: float) -> float:
        value = self.DEFAULT_ANGULAR_SPEED_DEG if speed_deg is None else float(speed_deg)
        return self._clamp_speed(value, 1.0, self.safe_angular_speed_deg, "angular")

    def _resolve_ee_speed(self, speed: float) -> float:
        value = self.DEFAULT_EE_SPEED if speed is None else float(speed)
        return self._clamp_speed(value, self.MIN_EE_SPEED, self.MAX_EE_SPEED, "ee")

    def _clamp_speed(self, speed: float, min_speed: float, max_speed: float, label: str) -> float:
        if speed < min_speed:
            print(f"[{self.robot_name}] WARNING: {label} speed {speed} below {min_speed}, clamping.")
            return float(min_speed)
        if speed > max_speed:
            print(f"[{self.robot_name}] WARNING: {label} speed {speed} exceeds safe maximum {max_speed}, Clamping to {max_speed}.")
            return float(max_speed)
        return float(speed)

    def _refresh_safe_speed_limits(self):
        left_id = self.act_ids.get("left_wheel_vel", -1)
        right_id = self.act_ids.get("right_wheel_vel", -1)
        if self.model is None or left_id == -1 or right_id == -1:
            self.safe_linear_speed = float(self.MAX_LINEAR_SPEED)
            self.safe_angular_speed_deg = float(self.MAX_ANGULAR_SPEED_DEG)
            return

        actuator_limits = []
        for actuator_id in (left_id, right_id):
            ctrl_limit = max(abs(float(value)) for value in self.model.actuator_ctrlrange[actuator_id])
            gear = abs(float(self.model.actuator_gear[actuator_id, 0]))
            if gear <= 0.0:
                gear = 1.0
            actuator_limits.append(ctrl_limit / gear)

        theoretical_linear = min(actuator_limits) * float(self.WHEEL_RADIUS)
        safe_linear = theoretical_linear * float(self.SAFETY_SPEED_FACTOR)
        safe_angular_deg = math.degrees((2.0 * safe_linear) / float(self.WHEEL_TRACK))
        self.safe_linear_speed = min(float(self.MAX_LINEAR_SPEED), safe_linear)
        self.safe_angular_speed_deg = min(float(self.MAX_ANGULAR_SPEED_DEG), safe_angular_deg)

    def _build_trapezoidal_profile(self, distance: float, peak_speed: float, accel: float) -> Dict[str, float]:
        distance = abs(float(distance))
        accel = abs(float(accel))
        peak_speed = abs(float(peak_speed))

        if distance == 0.0:
            return {
                "distance": 0.0,
                "accel": accel,
                "accel_time": 0.0,
                "accel_distance": 0.0,
                "cruise_time": 0.0,
                "peak_speed": 0.0,
                "duration": 0.0,
            }

        accel_time = peak_speed / accel
        accel_distance = 0.5 * accel * accel_time * accel_time
        if distance < 2.0 * accel_distance:
            accel_time = math.sqrt(distance / accel)
            peak_speed = accel * accel_time
            cruise_time = 0.0
            duration = 2.0 * accel_time
            accel_distance = 0.5 * accel * accel_time * accel_time
        else:
            cruise_distance = distance - 2.0 * accel_distance
            cruise_time = cruise_distance / peak_speed
            duration = 2.0 * accel_time + cruise_time

        return {
            "distance": distance,
            "accel": accel,
            "accel_time": accel_time,
            "accel_distance": accel_distance,
            "cruise_time": cruise_time,
            "peak_speed": peak_speed,
            "duration": duration,
        }

    def _sample_profile(self, profile: Dict[str, float], elapsed: float):
        accel = profile["accel"]
        accel_time = profile["accel_time"]
        cruise_time = profile["cruise_time"]
        duration = profile["duration"]
        peak_speed = profile["peak_speed"]
        total = profile["distance"]

        if elapsed <= 0.0:
            return 0.0, 0.0
        if elapsed < accel_time:
            speed = accel * elapsed
            traveled = 0.5 * accel * elapsed * elapsed
        elif elapsed < accel_time + cruise_time:
            speed = peak_speed
            traveled = profile["accel_distance"] + peak_speed * (elapsed - accel_time)
        elif elapsed < duration:
            remaining = duration - elapsed
            speed = accel * remaining
            traveled = total - 0.5 * accel * remaining * remaining
        else:
            speed = 0.0
            traveled = total
        return traveled, speed

    def _execute_base_profile(self, profile: Dict[str, float], mode: str, direction: float, target_value: float):
        start_x, start_y, start_yaw = self._get_base_pose()
        sim_start_time = self.data.time
        last_yaw = start_yaw
        accumulated_yaw = 0.0

        while True:
            if self.emergency_stop_flag:
                break

            elapsed = self.data.time - sim_start_time
            traveled, speed_now = self._sample_profile(profile, min(elapsed, profile["duration"]))

            if mode == "linear":
                linear_cmd = direction * speed_now
                self._command_base_twist(linear_cmd, 0.0)
                progress = self._measure_linear_progress(start_x, start_y, start_yaw)
                goal_reached = abs(progress - target_value) <= self.LINEAR_DISTANCE_TOLERANCE
            else:
                current_yaw = self._get_base_pose()[2]
                delta_yaw = self._normalize_angle(current_yaw - last_yaw)
                accumulated_yaw += delta_yaw
                last_yaw = current_yaw
                target_yaw = start_yaw + direction * traveled
                yaw_error = self._normalize_angle(target_yaw - current_yaw)
                angular_cmd = direction * speed_now + self.TURN_KP * yaw_error
                self._command_base_twist(0.0, angular_cmd)
                progress = accumulated_yaw
                goal_reached = abs(progress - target_value) <= math.radians(self.HEADING_TOLERANCE_DEG)

            if elapsed >= profile["duration"] and goal_reached and self._base_motion_is_settled():
                break
            if elapsed >= profile["duration"] + self.MAX_COMPLETION_OVERRUN:
                raise RuntimeError(
                    f"[{self.robot_name}] Base {mode} motion exceeded planned duration by more than "
                    f"{self.MAX_COMPLETION_OVERRUN:.2f}s."
                )

            self._sync_sim(sim_start_time, int(round((elapsed + self.control_dt) / self.control_dt)))

        self._command_base_twist(0.0, 0.0)

    def _command_base_twist(self, linear_m_s: float, angular_rad_s: float):
        half_track = 0.5 * self.WHEEL_TRACK
        left_linear = linear_m_s - half_track * angular_rad_s
        right_linear = linear_m_s + half_track * angular_rad_s
        left_rad_s = left_linear / self.WHEEL_RADIUS
        right_rad_s = right_linear / self.WHEEL_RADIUS
        self._update_ctrl_dict(
            {
                "left_wheel_vel": left_rad_s * self.WHEEL_GEAR,
                "right_wheel_vel": right_rad_s * self.WHEEL_GEAR,
            }
        )

    def _base_motion_is_settled(self) -> bool:
        linear_speed, angular_speed = self._get_base_velocity()
        return (
            abs(linear_speed) <= self.LINEAR_SETTLE_SPEED
            and abs(angular_speed) <= math.radians(self.ANGULAR_SETTLE_SPEED_DEG)
        )

    def _get_base_velocity(self):
        if self.body_id == -1:
            return 0.0, 0.0
        vel = self.data.cvel[self.body_id]
        linear_speed = float(np.linalg.norm(vel[3:]))
        angular_speed = float(np.linalg.norm(vel[:3]))
        return linear_speed, angular_speed

    def _measure_linear_progress(self, start_x: float, start_y: float, start_yaw: float) -> float:
        current_x, current_y, _ = self._get_base_pose()
        forward_x = math.cos(start_yaw)
        forward_y = math.sin(start_yaw)
        return (current_x - start_x) * forward_x + (current_y - start_y) * forward_y

    def _execute_gripper_move(self, target_pos: float, duration: float):
        start_ctrl = float(self.data.ctrl[self.act_ids["gripper"]])
        sim_start_time = self.data.time
        step = 0
        while self.data.time - sim_start_time < duration:
            if self.emergency_stop_flag:
                break
            elapsed = self.data.time - sim_start_time
            alpha = min(1.0, elapsed / duration) if duration > 0.0 else 1.0
            current = start_ctrl + alpha * (target_pos - start_ctrl)
            self._update_ctrl_dict({"gripper": current})
            step += 1
            self._sync_sim(sim_start_time, step)
        self._update_ctrl_dict({"gripper": target_pos})

    def _execute_arm_trajectory(self, target_lift: float, target_arm: float, duration: float):
        start_lift = float(self.data.ctrl[self.act_ids["lift"]])
        start_arm = float(self.data.ctrl[self.act_ids["arm"]])
        sim_start_time = self.data.time
        step = 0
        while self.data.time - sim_start_time < duration:
            if self.emergency_stop_flag:
                break
            elapsed = self.data.time - sim_start_time
            alpha = min(1.0, elapsed / duration) if duration > 0.0 else 1.0
            current_lift = start_lift + alpha * (target_lift - start_lift)
            current_arm = start_arm + alpha * (target_arm - start_arm)
            self._update_ctrl_dict({"lift": current_lift, "arm": current_arm})
            step += 1
            self._sync_sim(sim_start_time, step)
        self._update_ctrl_dict({"lift": target_lift, "arm": target_arm})

    def _solve_ik_2d(self, target_reach, target_height, max_iter=10, tol=0.0005):
        tmp_data = mujoco.MjData(self.model)
        tmp_data.qpos[:] = self.data.qpos[:]
        tmp_data.ctrl[:] = self.data.ctrl[:]

        current_lift = self.data.ctrl[self.act_ids["lift"]]
        current_arm = self.data.ctrl[self.act_ids["arm"]]

        for _ in range(max_iter):
            self._set_lift_arm_qpos(tmp_data, current_lift, current_arm)
            mujoco.mj_forward(self.model, tmp_data)

            ee_pos = tmp_data.xpos[self.ee_id]
            base_pos = tmp_data.xpos[self.body_id]
            current_reach = np.linalg.norm(ee_pos[:2] - base_pos[:2])
            current_height = ee_pos[2] - base_pos[2]
            error = np.array([target_reach - current_reach, target_height - current_height])
            if np.linalg.norm(error) < tol:
                break

            eps = 1e-4
            self._set_lift_arm_qpos(tmp_data, current_lift + eps, current_arm)
            mujoco.mj_forward(self.model, tmp_data)
            ee_pos_lift = tmp_data.xpos[self.ee_id]
            reach_lift = np.linalg.norm(ee_pos_lift[:2] - base_pos[:2])
            height_lift = ee_pos_lift[2] - base_pos[2]

            self._set_lift_arm_qpos(tmp_data, current_lift, current_arm + eps)
            mujoco.mj_forward(self.model, tmp_data)
            ee_pos_arm = tmp_data.xpos[self.ee_id]
            reach_arm = np.linalg.norm(ee_pos_arm[:2] - base_pos[:2])
            height_arm = ee_pos_arm[2] - base_pos[2]

            jacobian = np.array(
                [
                    [(reach_lift - current_reach) / eps, (reach_arm - current_reach) / eps],
                    [(height_lift - current_height) / eps, (height_arm - current_height) / eps],
                ]
            )
            try:
                delta = np.linalg.solve(jacobian, error)
            except np.linalg.LinAlgError:
                break

            current_lift = max(0.0, min(1.1, current_lift + delta[0]))
            current_arm = max(0.0, min(0.52, current_arm + delta[1]))

        return float(current_lift), float(current_arm)

    def _set_lift_arm_qpos(self, data_obj, lift_val, arm_val):
        aid_lift = self.act_ids["lift"]
        joint_id_lift = self.model.actuator_trnid[aid_lift, 0]
        data_obj.qpos[self.model.jnt_qposadr[joint_id_lift]] = lift_val

        for i in range(4):
            joint_name = f"joint_arm_l{i}"
            full_joint_name = f"{self.robot_name}_{joint_name}" if self.robot_name else joint_name
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, full_joint_name)
            if joint_id != -1:
                data_obj.qpos[self.model.jnt_qposadr[joint_id]] = arm_val / 4.0

    def _get_base_pose(self):
        pos = self.data.xpos[self.body_id]
        quat = self.data.xquat[self.body_id]
        w, x, y, z = quat
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return float(pos[0]), float(pos[1]), float(yaw)

    def _update_ctrl_dict(self, vals: Dict[str, float]):
        for name, val in vals.items():
            if name in self.act_ids and self.act_ids[name] != -1:
                self.data.ctrl[self.act_ids[name]] = val

    def _normalize_angle(self, angle_rad: float) -> float:
        return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi

    def _sign(self, value: float) -> float:
        return 1.0 if value >= 0.0 else -1.0

    def _sync_sim(self, start_sim_time, step_count):
        target_sim_time = start_sim_time + (step_count * self.control_dt)
        while self.data.time < target_sim_time:
            if self.emergency_stop_flag:
                break
            time.sleep(0.001)
