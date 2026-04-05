import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np

from common.robot_api import BaseRobotController, RobotState


@dataclass
class DifferentialDriveState(RobotState):
    global_pose: np.ndarray = None
    body_velocity: np.ndarray = None


class DifferentialDriveController(BaseRobotController):
    """Shared differential-drive base controller."""

    SAFETY_SPEED_FACTOR = 0.8
    WHEEL_RADIUS = 0.1
    WHEEL_TRACK = 0.5
    DEFAULT_LINEAR_SPEED = 0.5
    MAX_LINEAR_SPEED = 1.0
    DEFAULT_ANGULAR_SPEED_DEG = 45.0
    MAX_ANGULAR_SPEED_DEG = 90.0
    LINEAR_ACCEL = 0.5
    ANGULAR_ACCEL_DEG = 90.0
    HEADING_KP = 2.0
    TURN_KP = 3.0
    LINEAR_DISTANCE_TOLERANCE = 0.02
    HEADING_TOLERANCE_DEG = 2.0
    MAX_COMPLETION_OVERRUN = 2.0
    LINEAR_SETTLE_SPEED = 0.03
    ANGULAR_SETTLE_SPEED_DEG = 3.0

    left_actuator_name: Optional[str] = None
    right_actuator_name: Optional[str] = None
    base_body_name: Optional[str] = None
    left_sign = 1.0
    right_sign = 1.0

    def __init__(
        self,
        model,
        data,
        robot_name,
        base_pos=None,
        base_quat=None,
        log_dir=None,
        **kwargs,
    ):
        super().__init__(model, data, robot_name, base_pos, base_quat, log_dir)
        self.left_id = -1
        self.right_id = -1
        self.body_id = -1
        self.safe_linear_speed = float(self.MAX_LINEAR_SPEED)
        self.safe_angular_speed_deg = float(self.MAX_ANGULAR_SPEED_DEG)
        if self.left_actuator_name and self.right_actuator_name and self.base_body_name:
            self._lookup_handles()
            self._refresh_safe_speed_limits()

    def _lookup_handles(self):
        """Resolve MuJoCo ids for the configured actuators and base body."""
        if self.model is None:
            return

        self.left_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.left_actuator_name
        )
        self.right_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.right_actuator_name
        )
        self.body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_body_name
        )

    def _refresh_safe_speed_limits(self):
        theoretical_linear = self._theoretical_linear_speed_limit()
        if theoretical_linear is None:
            self.safe_linear_speed = float(self.MAX_LINEAR_SPEED)
            self.safe_angular_speed_deg = float(self.MAX_ANGULAR_SPEED_DEG)
            return

        safe_linear = theoretical_linear * float(self.SAFETY_SPEED_FACTOR)
        safe_angular_deg = math.degrees((2.0 * safe_linear) / float(self.WHEEL_TRACK))
        self.safe_linear_speed = min(float(self.MAX_LINEAR_SPEED), safe_linear)
        self.safe_angular_speed_deg = min(float(self.MAX_ANGULAR_SPEED_DEG), safe_angular_deg)

    def _theoretical_linear_speed_limit(self) -> Optional[float]:
        if self.model is None:
            return None

        actuator_limits = []
        for actuator_id in (self.left_id, self.right_id):
            if actuator_id is None or actuator_id < 0:
                return None
            ctrl_limit = max(abs(float(value)) for value in self.model.actuator_ctrlrange[actuator_id])
            gear = abs(float(self.model.actuator_gear[actuator_id, 0]))
            if gear <= 0.0:
                gear = 1.0
            actuator_limits.append(ctrl_limit / gear)

        return min(actuator_limits) * float(self.WHEEL_RADIUS)

    def _require_motion_handles(self):
        missing = []
        if self.left_id == -1:
            missing.append(f"left actuator '{self.left_actuator_name}'")
        if self.right_id == -1:
            missing.append(f"right actuator '{self.right_actuator_name}'")
        if self.body_id == -1:
            missing.append(f"base body '{self.base_body_name}'")
        if missing:
            raise RuntimeError(
                f"[{self.robot_name}] Cannot execute differential-drive action: "
                f"unresolved handles: {', '.join(missing)}"
            )

    def _validate_straight_direction(self, direction: str) -> float:
        if direction == "forward":
            return 1.0
        if direction == "backward":
            return -1.0
        raise ValueError(
            f"[{self.robot_name}] Invalid straight-motion direction '{direction}'. "
            "Expected 'forward' or 'backward'."
        )

    def _validate_turn_direction(self, direction: str) -> str:
        if direction in {"auto", "cw", "ccw"}:
            return direction
        raise ValueError(
            f"[{self.robot_name}] Invalid turn direction '{direction}'. "
            "Expected 'auto', 'cw', or 'ccw'."
        )

    def get_robot_state(self) -> DifferentialDriveState:
        x, y, yaw = self._get_pose()
        linear_speed, angular_speed = self._get_base_velocity()
        return DifferentialDriveState(
            timestamp=getattr(self.data, "time", 0.0),
            global_pose=np.array([x, y, yaw], dtype=float),
            body_velocity=np.array([linear_speed, angular_speed], dtype=float),
        )

    def _normalize_angle(self, angle_rad: float) -> float:
        return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi

    def _clamp_speed(self, speed: float, min_val: float, max_val: float, label: str) -> float:
        if speed < min_val:
            msg = (
                f"[{self.robot_name}] Warning: {label} speed {speed} is below minimum {min_val}. "
                f"Clamping to {min_val}."
            )
            print(msg)
            self.log(msg)
            return float(min_val)
        if speed > max_val:
            msg = (
                f"[{self.robot_name}] Warning: {label} speed {speed} exceeds safe maximum {max_val}. "
                f"Clamping to {max_val}."
            )
            print(msg)
            self.log(msg)
            return float(max_val)
        return float(speed)

    def _clamp_linear_speed(self, speed: float) -> float:
        max_speed = getattr(self, "safe_linear_speed", float(self.MAX_LINEAR_SPEED))
        return self._clamp_speed(speed, 0.01, max_speed, "Linear")

    def _clamp_angular_speed_deg(self, speed_deg: float) -> float:
        max_speed = getattr(self, "safe_angular_speed_deg", float(self.MAX_ANGULAR_SPEED_DEG))
        return self._clamp_speed(speed_deg, 1.0, max_speed, "Angular")

    def _build_linear_profile(self, distance: float, speed: float) -> Dict[str, float]:
        distance = abs(float(distance))
        if distance == 0.0:
            return {
                "distance": 0.0,
                "accel": self.LINEAR_ACCEL,
                "accel_time": 0.0,
                "accel_distance": 0.0,
                "cruise_time": 0.0,
                "peak_speed": 0.0,
                "duration": 0.0,
            }

        speed = self._clamp_linear_speed(speed)
        accel = float(self.LINEAR_ACCEL)
        accel_time = speed / accel
        accel_distance = 0.5 * accel * accel_time * accel_time

        if distance < 2.0 * accel_distance:
            accel_time = math.sqrt(distance / accel)
            peak_speed = accel * accel_time
            cruise_time = 0.0
            duration = 2.0 * accel_time
            accel_distance = 0.5 * accel * accel_time * accel_time
        else:
            peak_speed = speed
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

    def _build_angular_profile(self, angle_rad: float, speed_deg: float) -> Dict[str, float]:
        angle_rad = abs(float(angle_rad))
        if angle_rad == 0.0:
            return {
                "angle_rad": 0.0,
                "accel": math.radians(self.ANGULAR_ACCEL_DEG),
                "accel_time": 0.0,
                "accel_distance": 0.0,
                "cruise_time": 0.0,
                "peak_speed": 0.0,
                "duration": 0.0,
            }

        speed_deg = self._clamp_angular_speed_deg(speed_deg)
        accel = math.radians(self.ANGULAR_ACCEL_DEG)
        peak_speed = math.radians(speed_deg)
        accel_time = peak_speed / accel
        accel_distance = 0.5 * accel * accel_time * accel_time

        if angle_rad < 2.0 * accel_distance:
            accel_time = math.sqrt(angle_rad / accel)
            peak_speed = accel * accel_time
            cruise_time = 0.0
            duration = 2.0 * accel_time
            accel_distance = 0.5 * accel * accel_time * accel_time
        else:
            cruise_distance = angle_rad - 2.0 * accel_distance
            cruise_time = cruise_distance / peak_speed
            duration = 2.0 * accel_time + cruise_time

        return {
            "angle_rad": angle_rad,
            "accel": accel,
            "accel_time": accel_time,
            "accel_distance": accel_distance,
            "cruise_time": cruise_time,
            "peak_speed": peak_speed,
            "duration": duration,
        }

    def _body_twist_to_wheels(self, linear_m_s: float, angular_rad_s: float) -> Tuple[float, float]:
        half_track = 0.5 * float(self.WHEEL_TRACK)
        left_linear = linear_m_s - half_track * angular_rad_s
        right_linear = linear_m_s + half_track * angular_rad_s
        return left_linear / self.WHEEL_RADIUS, right_linear / self.WHEEL_RADIUS

    def _resolve_turn_delta(self, start_yaw: float, target_yaw: float, direction: str) -> float:
        delta = self._normalize_angle(target_yaw - start_yaw)
        if direction == "ccw" and delta < 0.0:
            delta += 2.0 * math.pi
        elif direction == "cw" and delta > 0.0:
            delta -= 2.0 * math.pi
        return delta

    def _sample_profile(self, profile: Dict[str, float], elapsed: float) -> Tuple[float, float]:
        accel = profile["accel"]
        accel_time = profile["accel_time"]
        cruise_time = profile["cruise_time"]
        duration = profile["duration"]
        peak_speed = profile["peak_speed"]
        total = profile.get("distance", profile.get("angle_rad", 0.0))

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

    def _wait_until(self, target_sim_time: float):
        while getattr(self.data, "time", 0.0) < target_sim_time:
            if self.emergency_stop_flag:
                break
            time.sleep(0.001)

    def _get_pose(self) -> Tuple[float, float, float]:
        if self.body_id == -1 or self.data is None:
            return 0.0, 0.0, 0.0
        pos = self.data.xpos[self.body_id]
        quat = self.data.xquat[self.body_id]
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return float(pos[0]), float(pos[1]), float(yaw)

    def _get_base_velocity(self) -> Tuple[float, float]:
        if self.body_id == -1 or self.data is None:
            return 0.0, 0.0
        vel = self.data.cvel[self.body_id]
        linear_speed = float(np.linalg.norm(vel[3:]))
        angular_speed = float(np.linalg.norm(vel[:3]))
        return linear_speed, angular_speed

    def _is_settled(self, linear_speed: float, angular_speed: float) -> bool:
        return (
            abs(linear_speed) <= float(self.LINEAR_SETTLE_SPEED)
            and abs(angular_speed) <= math.radians(float(self.ANGULAR_SETTLE_SPEED_DEG))
        )

    def _command_wheels(self, left_rad_s: float, right_rad_s: float):
        if self.data is None:
            return
        if self.left_id != -1:
            self.data.ctrl[self.left_id] = self.left_sign * left_rad_s
        if self.right_id != -1:
            self.data.ctrl[self.right_id] = self.right_sign * right_rad_s

    def _stop_motion(self):
        self._command_wheels(0.0, 0.0)

    def _straight_motion_progress(
        self,
        start_x: float,
        start_y: float,
        start_yaw: float,
        current_x: float,
        current_y: float,
        direction_sign: float,
    ) -> float:
        forward_x = math.cos(start_yaw)
        forward_y = math.sin(start_yaw)
        displacement = (current_x - start_x) * forward_x + (current_y - start_y) * forward_y
        return direction_sign * displacement

    def _turn_motion_progress(self, start_yaw: float, current_yaw: float, direction_sign: float) -> float:
        return direction_sign * self._normalize_angle(current_yaw - start_yaw)

    def action_move_straight(
        self,
        distance: float,
        direction: str = "forward",
        speed: Optional[float] = None,
    ):
        if self.emergency_stop_flag:
            return

        self._require_motion_handles()
        direction_sign = self._validate_straight_direction(direction)
        requested_speed = self.DEFAULT_LINEAR_SPEED if speed is None else speed
        profile = self._build_linear_profile(distance, requested_speed)
        target_distance = profile["distance"]
        start_x, start_y, start_yaw = self._get_pose()
        sim_start_time = getattr(self.data, "time", 0.0)
        max_deadline = profile["duration"] + self.MAX_COMPLETION_OVERRUN
        heading_tolerance = math.radians(self.HEADING_TOLERANCE_DEG)

        print(
            f"[{self.robot_name}] Moving {abs(distance):.2f}m {direction} at {profile['peak_speed']:.2f}m/s "
            f"(Traj Duration: {profile['duration']:.2f}s)..."
        )

        try:
            while not self.emergency_stop_flag:
                elapsed = getattr(self.data, "time", 0.0) - sim_start_time
                _, speed_now = self._sample_profile(profile, elapsed)
                current_x, current_y, current_yaw = self._get_pose()
                progress = self._straight_motion_progress(
                    start_x, start_y, start_yaw, current_x, current_y, direction_sign
                )
                yaw_error = self._normalize_angle(start_yaw - current_yaw)
                angular_cmd = self.HEADING_KP * yaw_error
                left, right = self._body_twist_to_wheels(direction_sign * speed_now, angular_cmd)
                self._command_wheels(left, right)

                linear_speed, angular_speed = self._get_base_velocity()
                heading_ok = abs(yaw_error) <= heading_tolerance
                progress_ok = progress >= (target_distance - self.LINEAR_DISTANCE_TOLERANCE)
                settled = self._is_settled(linear_speed, angular_speed)

                if elapsed >= profile["duration"] and heading_ok and progress_ok and settled:
                    break

                if elapsed >= max_deadline:
                    raise RuntimeError(
                        f"[{self.robot_name}] Straight motion did not reach the planned endpoint: "
                        f"progress={progress:.3f}m target={target_distance:.3f}m "
                        f"heading_error={math.degrees(yaw_error):.2f}deg"
                    )

                self._wait_until(getattr(self.data, "time", 0.0) + self.control_dt)
        finally:
            self._stop_motion()

    def action_turn(
        self,
        target_yaw: float,
        direction: str = "auto",
        speed: Optional[float] = None,
    ):
        if self.emergency_stop_flag:
            return

        self._require_motion_handles()
        direction = self._validate_turn_direction(direction)
        requested_speed = self.DEFAULT_ANGULAR_SPEED_DEG if speed is None else speed
        _, _, start_yaw = self._get_pose()
        target_rad = math.radians(target_yaw)
        delta = self._resolve_turn_delta(start_yaw, target_rad, direction)
        profile = self._build_angular_profile(delta, requested_speed)
        turn_sign = 1.0 if delta >= 0.0 else -1.0
        sim_start_time = getattr(self.data, "time", 0.0)
        max_deadline = profile["duration"] + self.MAX_COMPLETION_OVERRUN
        angle_tolerance = math.radians(self.HEADING_TOLERANCE_DEG)
        target_delta = abs(delta)
        accumulated_yaw = 0.0
        last_yaw = start_yaw

        print(
            f"[{self.robot_name}] Turning to {target_yaw:.1f} deg at {math.degrees(profile['peak_speed']):.1f} deg/s "
            f"(Traj Duration: {profile['duration']:.2f}s)..."
        )

        try:
            while not self.emergency_stop_flag:
                elapsed = getattr(self.data, "time", 0.0) - sim_start_time
                traveled, speed_now = self._sample_profile(profile, elapsed)
                current_x, current_y, current_yaw = self._get_pose()
                delta_yaw = self._normalize_angle(current_yaw - last_yaw)
                accumulated_yaw += delta_yaw
                last_yaw = current_yaw
                target = start_yaw + turn_sign * traveled
                yaw_error = self._normalize_angle(target - current_yaw)
                angular_cmd = turn_sign * speed_now + self.TURN_KP * yaw_error
                left, right = self._body_twist_to_wheels(0.0, angular_cmd)
                self._command_wheels(left, right)

                linear_speed, angular_speed = self._get_base_velocity()
                heading_ok = abs(yaw_error) <= angle_tolerance
                progress_ok = abs(accumulated_yaw) >= (target_delta - angle_tolerance)
                settled = self._is_settled(linear_speed, angular_speed)

                if elapsed >= profile["duration"] and heading_ok and progress_ok and settled:
                    break

                if elapsed >= max_deadline:
                    raise RuntimeError(
                        f"[{self.robot_name}] Turn did not reach the planned heading: "
                        f"progress={math.degrees(accumulated_yaw):.2f}deg "
                        f"target={math.degrees(target_delta):.2f}deg "
                        f"heading_error={math.degrees(yaw_error):.2f}deg"
                    )

                self._wait_until(getattr(self.data, "time", 0.0) + self.control_dt)
        finally:
            self._stop_motion()
