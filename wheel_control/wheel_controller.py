import time
import numpy as np
import mujoco
from common.robot_api import BaseRobotController, RobotState

class WheelController(BaseRobotController):
    """
    Controller for differential drive robots (e.g., RB-Theron).
    """
    def __init__(self, model, data, robot_name, base_pos=None, base_quat=None, **kwargs):
        super().__init__(model, data, robot_name, base_pos, base_quat)
        
        # Identify Actuators
        # SceneBuilder prefixes names with "{robot_name}_"
        self.left_actuator_name = f"{robot_name}_left_wheel_vel"
        self.right_actuator_name = f"{robot_name}_right_wheel_vel"
        self.base_body_name = f"{robot_name}_base_link"
        self.rotate_wheel_speed = 10.0  # rad/s
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

        if self.left_id == -1 or self.right_id == -1:
            print(f"[{self.robot_name}] WARNING: Wheel actuators not found!")

    def get_robot_state(self) -> RobotState:
        # For now, just return a timestamp
        return RobotState(timestamp=time.time())

    def _get_pose(self):
        """Returns (x, y, yaw) in World Frame."""
        if self.body_id == -1:
            return 0, 0, 0
        
        pos = self.data.xpos[self.body_id]
        quat = self.data.xquat[self.body_id] # w, x, y, z
        
        # Quat to Yaw (Z-rotation)
        # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
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
        """Waits until the base velocity is near zero."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            vel = self.data.cvel[self.body_id] # 6D velocity (3 rot, 3 lin)
            lin_speed = np.linalg.norm(vel[3:]) # Linear part
            rot_speed = np.linalg.norm(vel[:3]) # Rotational part
            
            if lin_speed < 0.01 and rot_speed < 0.01:
                break
            time.sleep(self.control_dt)

    def action_turn(self, target_yaw: float, direction: str = "auto"):
        """
        Turn in place to an absolute yaw angle with deceleration and fine-tuning.
        Args:
            target_yaw: Target absolute yaw in degrees (-180 to 180).
            direction: "cw" (clockwise), "ccw" (counter-clockwise), or "auto" (shortest path).
        """
        if self.emergency_stop_flag: return

        target_rad = np.deg2rad(target_yaw)
        base_speed = self.rotate_wheel_speed
        
        # Thresholds
        DECEL_THRESHOLD = np.deg2rad(20.0)
        STOP_THRESHOLD = np.deg2rad(0.05)
        MIN_SPEED = 2.0

        _, _, start_yaw = self._get_pose()
        
        # Calculate required delta angle based on direction
        diff = target_rad - start_yaw
        
        # Normalize to [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi

        if direction == "ccw":
            if diff < 0: diff += 2 * np.pi
        elif direction == "cw":
            if diff > 0: diff -= 2 * np.pi
        # "auto" uses the shortest path (diff as is)

        turn_angle = diff
        turn_abs = abs(turn_angle)
        turn_sign = np.sign(turn_angle)
        
        accumulated_yaw = 0.0
        last_yaw = start_yaw
        
        print(f"[{self.robot_name}] Turning to {target_yaw} deg ({direction}). Delta: {np.rad2deg(turn_angle):.2f} deg")
        
        # --- Phase 1: Main Turn with Deceleration ---
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
            
            if remaining <= 0:
                break

            # Speed Control
            current_speed = base_speed
            if remaining < DECEL_THRESHOLD:
                ratio = remaining / DECEL_THRESHOLD
                current_speed = MIN_SPEED + (base_speed - MIN_SPEED) * ratio
            
            # Set Velocity based on turn direction
            if turn_sign > 0: # CCW
                self._set_velocity(-current_speed, current_speed)
            else: # CW
                self._set_velocity(current_speed, -current_speed)
            
            time.sleep(self.control_dt)
            
        self._set_velocity(0, 0)
        self._wait_for_stop()

        # --- Phase 2: Fine-Tuning (Targeting Absolute Yaw) ---
        MAX_RETRIES = 5
        FINE_SPEED = 0.2
        
        for attempt in range(MAX_RETRIES):
            _, _, curr_yaw = self._get_pose()
            
            # Calculate shortest error to target
            error = target_rad - curr_yaw
            error = (error + np.pi) % (2 * np.pi) - np.pi
            
            if abs(error) <= STOP_THRESHOLD:
                break
                
            # Determine correction direction (shortest path)
            if error > 0: # Need CCW
                l_cmd, r_cmd = -FINE_SPEED, FINE_SPEED
            else: # Need CW
                l_cmd, r_cmd = FINE_SPEED, -FINE_SPEED
                
            self._set_velocity(l_cmd, r_cmd)
            
            # Correction Loop
            while True:
                if self.emergency_stop_flag: return
                
                _, _, curr_yaw = self._get_pose()
                
                prev_error = error
                error = target_rad - curr_yaw
                error = (error + np.pi) % (2 * np.pi) - np.pi
                
                if abs(error) <= STOP_THRESHOLD:
                    break
                
                if np.sign(error) != np.sign(prev_error):
                    break
                    
                time.sleep(self.control_dt)
                
            self._set_velocity(0, 0)
            self._wait_for_stop()
            
            # Check final error for this attempt
            _, _, curr_yaw = self._get_pose()
            error = target_rad - curr_yaw
            error = (error + np.pi) % (2 * np.pi) - np.pi

        print(f"[{self.robot_name}] Turn complete. Final Abs Error: {np.rad2deg(error):.4f} deg")

    def action_move_straight(self, distance: float, direction: str = "forward"):
        """
        Move straight with closed-loop heading control.
        Args:
            distance: Meters to move.
            direction: "forward" or "backward".
        """
        if self.emergency_stop_flag: return

        base_speed = self.move_wheel_speed
        if direction == "backward":
            base_speed = -base_speed
            
        # Heading Control Gains
        Kp = 20.0 
            
        start_x, start_y, start_yaw = self._get_pose()
        start_pos = np.array([start_x, start_y])
        target_yaw = start_yaw # Maintain initial heading
        
        print(f"[{self.robot_name}] Moving straight {distance}m ({direction}) with Heading Control...")
        
        # Initial command
        self._set_velocity(base_speed, base_speed)
        
        while True:
            if self.emergency_stop_flag:
                self._set_velocity(0, 0)
                return

            curr_x, curr_y, curr_yaw = self._get_pose()
            curr_pos = np.array([curr_x, curr_y])
            
            # 1. Check Distance
            dist_traveled = np.linalg.norm(curr_pos - start_pos)
            if dist_traveled >= distance:
                break
            
            # 2. Calculate Heading Error
            yaw_error = curr_yaw - target_yaw
            # Normalize to [-pi, pi]
            yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
            
            # 3. Apply P-Control
            # If error > 0 (Left bias), we need CW turn (Left > Right)
            # If error < 0 (Right bias), we need CCW turn (Right > Left)
            correction = Kp * yaw_error
            
            left_cmd = base_speed + correction
            right_cmd = base_speed - correction
            
            self._set_velocity(left_cmd, right_cmd)
                
            time.sleep(self.control_dt)
            
        self._set_velocity(0, 0)
        self._wait_for_stop()
        print(f"[{self.robot_name}] Move complete.")
