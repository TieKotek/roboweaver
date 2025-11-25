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
        self.move_wheel_speed = 15.0    # rad/s

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

    def action_turn(self, angle: float):
        """
        Turn in place.
        Args:
            angle: Degrees (-180 to 180). Positive = CCW (Left), Negative = CW (Right)
        """
        if self.emergency_stop_flag: return

        target_rad = np.deg2rad(angle)
        speed = self.rotate_wheel_speed
        
        # Determine direction
        # Differential drive: 
        # Left turn (positive angle): Left wheel back (-), Right wheel fwd (+)
        # Right turn (negative angle): Left wheel fwd (+), Right wheel back (-)
        if angle > 0:
            left_cmd = -speed
            right_cmd = speed
        else:
            left_cmd = speed
            right_cmd = -speed
            
        _, _, start_yaw = self._get_pose()
        # Handle wrapping for target calculation isn't strictly necessary if we track delta
        # But tracking accumulated delta is safer
        
        accumulated_yaw = 0.0
        last_yaw = start_yaw
        target_abs = abs(target_rad)
        
        print(f"[{self.robot_name}] Turning {angle} deg...")
        self._set_velocity(left_cmd, right_cmd)
        
        while abs(accumulated_yaw) < target_abs:
            if self.emergency_stop_flag: 
                self._set_velocity(0, 0)
                return

            _, _, curr_yaw = self._get_pose()
            
            # Calculate shortest delta
            delta = curr_yaw - last_yaw
            # Normalize delta to [-pi, pi]
            delta = (delta + np.pi) % (2 * np.pi) - np.pi
            
            accumulated_yaw += delta
            last_yaw = curr_yaw
            
            time.sleep(self.control_dt)
            
        # Stop
        self._set_velocity(0, 0)
        self._wait_for_stop()
        print(f"[{self.robot_name}] Turn complete.")

    def action_move_straight(self, distance: float, direction: str = "forward"):
        """
        Move straight.
        Args:
            distance: Meters to move.
            direction: "forward" or "backward".
        """
        if self.emergency_stop_flag: return

        speed = self.move_wheel_speed
        if direction == "backward":
            speed = -speed
            
        start_x, start_y, _ = self._get_pose()
        start_pos = np.array([start_x, start_y])
        
        print(f"[{self.robot_name}] Moving straight {distance}m ({direction})...")
        self._set_velocity(speed, speed)
        
        while True:
            if self.emergency_stop_flag:
                self._set_velocity(0, 0)
                return

            curr_x, curr_y, _ = self._get_pose()
            curr_pos = np.array([curr_x, curr_y])
            
            dist_traveled = np.linalg.norm(curr_pos - start_pos)
            
            if dist_traveled >= distance:
                break
                
            time.sleep(self.control_dt)
            
        self._set_velocity(0, 0)
        self._wait_for_stop()
        print(f"[{self.robot_name}] Move complete.")
