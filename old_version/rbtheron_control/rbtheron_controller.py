import time
import numpy as np
import mujoco
from common.robot_api import BaseRobotController, RobotState
from dataclasses import dataclass

@dataclass
class RbtheronState(RobotState):
    global_pose: np.ndarray = None # [x, y, yaw]

class RbtheronController(BaseRobotController):
    """
    Controller for differential drive robots (e.g., RB-Theron).
    """
    def __init__(self, model, data, robot_name, base_pos=None, base_quat=None, log_dir=None, **kwargs):
        super().__init__(model, data, robot_name, base_pos, base_quat, log_dir)
        
        # Identify Actuators
        self.left_actuator_name = f"{robot_name}_left_wheel_vel"
        self.right_actuator_name = f"{robot_name}_right_wheel_vel"
        self.base_body_name = f"{robot_name}_base_link"
        self.rotate_wheel_speed = 20.0  # rad/s
        self.move_wheel_speed = 30.0    # rad/s

        try:
            self.left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.left_actuator_name)
            self.right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.right_actuator_name)
            self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.base_body_name)
        except Exception as e:
            print(f"[{self.robot_name}] Error finding IDs: {e}")
            self.left_id = -1
            self.right_id = -1
            self.body_id = -1

    def get_robot_state(self) -> RbtheronState:
        x, y, yaw = self._get_pose()
        return RbtheronState(
            timestamp=self.data.time,
            global_pose=np.array([x, y, yaw])
        )

    def _get_pose(self):
        """Returns (x, y, yaw) in World Frame."""
        if self.body_id == -1:
            return 0, 0, 0
        pos = self.data.xpos[self.body_id]
        quat = self.data.xquat[self.body_id] # w, x, y, z
        w, x, y, z = quat
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return pos[0], pos[1], yaw

    def _set_velocity(self, left, right):
        if self.left_id != -1:
            self.data.ctrl[self.left_id] = left
        if self.right_id != -1:
            self.data.ctrl[self.right_id] = right

    def _wait_for_stop(self, timeout=5.0):
        """Waits until the base velocity is near zero using simulation time."""
        sim_start_time = self.data.time
        while self.data.time - sim_start_time < timeout:
            vel = self.data.cvel[self.body_id] 
            lin_speed = np.linalg.norm(vel[3:])
            rot_speed = np.linalg.norm(vel[:3])
            if lin_speed < 0.01 and rot_speed < 0.01:
                break
            time.sleep(0.001)

    def action_turn(self, target_yaw: float, direction: str = "auto"):
        if self.emergency_stop_flag: return
        target_rad = np.deg2rad(target_yaw)
        base_speed = self.rotate_wheel_speed
        DECEL_THRESHOLD = np.deg2rad(20.0)
        STOP_THRESHOLD = np.deg2rad(0.05)
        MIN_SPEED = 2.0

        _, _, start_yaw = self._get_pose()
        diff = target_rad - start_yaw
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        if direction == "ccw" and diff < 0: diff += 2 * np.pi
        elif direction == "cw" and diff > 0: diff -= 2 * np.pi

        turn_angle = diff
        turn_abs = abs(turn_angle)
        turn_sign = np.sign(turn_angle)
        accumulated_yaw = 0.0
        last_yaw = start_yaw
        
        print(f"[{self.robot_name}] Turning to {target_yaw} deg...")
        sim_start_time = self.data.time
        step_count = 0
        
        while abs(accumulated_yaw) < turn_abs:
            if self.emergency_stop_flag: 
                self._set_velocity(0, 0)
                return
            _, _, curr_yaw = self._get_pose()
            delta = curr_yaw - last_yaw
            delta = (delta + np.pi) % (2 * np.pi) - np.pi
            accumulated_yaw += delta
            last_yaw = curr_yaw
            remaining = turn_abs - abs(accumulated_yaw)
            if remaining <= 0: break
            current_speed = base_speed
            if remaining < DECEL_THRESHOLD:
                ratio = remaining / DECEL_THRESHOLD
                current_speed = MIN_SPEED + (base_speed - MIN_SPEED) * ratio
            if turn_sign > 0: self._set_velocity(-current_speed, current_speed)
            else: self._set_velocity(current_speed, -current_speed)
            
            step_count += 1
            while self.data.time < sim_start_time + (step_count * self.control_dt):
                time.sleep(0.001)
            
        self._set_velocity(0, 0)
        self._wait_for_stop()

    def action_move_straight(self, distance: float, direction: str = "forward"):
        if self.emergency_stop_flag: return
        base_speed = self.move_wheel_speed
        if direction == "backward": base_speed = -base_speed
        Kp = 20.0 
        start_x, start_y, start_yaw = self._get_pose()
        start_pos = np.array([start_x, start_y])
        target_yaw = start_yaw
        
        print(f"[{self.robot_name}] Moving straight {distance}m...")
        sim_start_time = self.data.time
        step_count = 0
        while True:
            if self.emergency_stop_flag:
                self._set_velocity(0, 0)
                return
            curr_x, curr_y, curr_yaw = self._get_pose()
            curr_pos = np.array([curr_x, curr_y])
            dist_traveled = np.linalg.norm(curr_pos - start_pos)
            if dist_traveled >= distance: break
            yaw_error = curr_yaw - target_yaw
            yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
            correction = Kp * yaw_error
            self._set_velocity(base_speed + correction, base_speed - correction)
            step_count += 1
            while self.data.time < sim_start_time + (step_count * self.control_dt):
                time.sleep(0.001)
        self._set_velocity(0, 0)
        self._wait_for_stop()