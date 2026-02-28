import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from common.robot_api import BaseRobotController, RobotState
from dataclasses import dataclass

@dataclass
class SkydioState(RobotState):
    pos: np.ndarray = None
    vel: np.ndarray = None
    rpy: np.ndarray = None

class SkydioController(BaseRobotController):
    """
    Controller for the Skydio X2 Drone.
    Features precise position and velocity control via a cascaded PID loop.
    """
    def __init__(self, model, data, robot_name, base_pos=None, base_quat=None, log_dir=None, **kwargs):
        super().__init__(model, data, robot_name, base_pos, base_quat, log_dir)
        
        # 1. Identify Actuators
        self.thrust_ids = []
        for i in range(1, 5):
            name = f"{robot_name}_thrust{i}"
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid == -1:
                print(f"[{self.robot_name}] Warning: Actuator {name} not found.")
            self.thrust_ids.append(aid)
            
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{robot_name}_x2")
        if self.body_id == -1:
            print(f"[{self.robot_name}] Error: Body {robot_name}_x2 not found.")
        
        # 2. Mixer Matrix (Fz, tau_x, tau_y, tau_z) -> (T1, T2, T3, T4)
        self.M = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-0.18, 0.18, 0.18, -0.18],
            [0.14, 0.14, -0.14, -0.14],
            [-0.0201, 0.0201, -0.0201, 0.0201]
        ])
        self.M_inv = np.linalg.pinv(self.M)
        
        self.mass = 1.325
        self.g = 9.81
        
        # 3. Cascaded PID Gains (Tuned for precise, overshoot-free movement)
        self.Kp_p = np.array([2.5, 2.5, 2.5]) # Increased position gain
        
        self.Kp_v = np.array([6.0, 6.0, 8.0]) # Increased velocity gain
        self.Ki_v = np.array([0.5, 0.5, 1.0])
        self.Kd_v = np.array([0.2, 0.2, 0.3])
        
        self.Kp_att = np.array([25.0, 25.0, 15.0])
        self.Kd_att = np.array([5.0, 5.0, 3.0])
        
        self.integral_v = np.zeros(3)
        self.is_landed = True
        
        # 4. Speed Limits
        self.min_lin_speed = 0.1
        self.max_lin_speed = 5.0
        self.max_land_speed = 2.0
        self.min_ang_speed = 5.0
        self.max_ang_speed = 90.0
        
        # 5. Initialize internal state
        self.target_yaw = 0.0
        if self.base_quat is not None:
            r = R.from_quat([self.base_quat[1], self.base_quat[2], self.base_quat[3], self.base_quat[0]])
            self.target_yaw = r.as_euler('xyz')[2]
            
        self.target_pos = self.base_pos.copy()

    def _get_pose(self):
        if self.body_id == -1: return np.zeros(3), np.array([1, 0, 0, 0])
        pos = self.data.xpos[self.body_id].copy()
        quat = self.data.xquat[self.body_id].copy() # w, x, y, z
        return pos, quat
        
    def _get_velocities(self):
        if self.body_id == -1: return np.zeros(3), np.zeros(3)
        jnt_adr = self.model.body_jntadr[self.body_id]
        if jnt_adr == -1: return np.zeros(3), np.zeros(3)
        
        qvel_adr = self.model.jnt_dofadr[jnt_adr]
        lin_vel = self.data.qvel[qvel_adr:qvel_adr+3].copy()
        ang_vel = self.data.qvel[qvel_adr+3:qvel_adr+6].copy()
        return lin_vel, ang_vel

    def _clamp_speed(self, speed: float, min_val: float, max_val: float, speed_type: str = "Linear") -> float:
        if speed < min_val:
            msg = f"[{self.robot_name}] Warning: {speed_type} speed {speed} is below minimum {min_val}. Clamping to {min_val}."
            print(msg)
            self.log(msg)
            return min_val
        elif speed > max_val:
            msg = f"[{self.robot_name}] Warning: {speed_type} speed {speed} exceeds maximum {max_val}. Clamping to {max_val}."
            print(msg)
            self.log(msg)
            return max_val
        return speed

    def get_robot_state(self) -> SkydioState:
        pos, quat = self._get_pose()
        lin_vel, _ = self._get_velocities()
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        rpy = np.rad2deg(r.as_euler('xyz'))
        return SkydioState(
            timestamp=self.data.time,
            pos=pos,
            vel=lin_vel,
            rpy=rpy
        )

    def _control_step(self, target_pos, target_yaw, target_vel=np.zeros(3), dt=0.01):
        """Internal PID Control Step run at high frequency."""
        if self.is_landed:
            for aid in self.thrust_ids:
                if aid != -1: self.data.ctrl[aid] = 0.0
            return

        pos, quat = self._get_pose()
        lin_vel, ang_vel = self._get_velocities()
        
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        roll, pitch, yaw = r.as_euler('xyz')
        
        # 1. Position -> Desired Velocity
        e_p = target_pos - pos
        v_des = self.Kp_p * e_p + target_vel
        
        # 2. Velocity -> Desired Acceleration
        e_v = v_des - lin_vel
        self.integral_v += e_v * dt
        self.integral_v = np.clip(self.integral_v, -2.0, 2.0) # Anti-windup
        
        a_des = self.Kp_v * e_v + self.Ki_v * self.integral_v - self.Kd_v * lin_vel
        
        # 3. Acceleration -> Desired Attitude & Thrust
        Fz_world = self.mass * (a_des[2] + self.g)
        
        r_yaw = R.from_euler('z', yaw).as_matrix()
        a_des_yaw = r_yaw.T @ a_des
        
        roll_des = np.clip(-a_des_yaw[1] / self.g, -0.5, 0.5)
        pitch_des = np.clip(a_des_yaw[0] / self.g, -0.5, 0.5)
        
        # 4. Attitude -> Torques
        e_roll = roll_des - roll
        e_pitch = pitch_des - pitch
        e_yaw = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        
        tau_x = self.Kp_att[0] * e_roll - self.Kd_att[0] * ang_vel[0]
        tau_y = self.Kp_att[1] * e_pitch - self.Kd_att[1] * ang_vel[1]
        tau_z = self.Kp_att[2] * e_yaw - self.Kd_att[2] * ang_vel[2]
        
        Fz = Fz_world / (np.cos(roll) * np.cos(pitch))
        Fz = np.clip(Fz, 0, 50)
        
        # 5. Mixer (Map virtual forces to motor thrusts)
        U = np.array([Fz, tau_x, tau_y, tau_z])
        T = self.M_inv @ U
        T = np.clip(T, 0.0, 13.0) # 13 is the max control limit in XML
        
        for i, aid in enumerate(self.thrust_ids):
            if aid != -1:
                self.data.ctrl[aid] = T[i]

    def action_idle(self, duration: float = 1.0):
        """Override Base idle to maintain PID hover instead of falling."""
        if self.emergency_stop_flag: return
        sim_start_time = self.data.time
        while self.data.time - sim_start_time < duration:
            if self.emergency_stop_flag: break
            self._control_step(self.target_pos, self.target_yaw, dt=self.control_dt)
            
            target_time = self.data.time + self.control_dt
            while self.data.time < target_time:
                time.sleep(0.001)

    def action_takeoff(self, altitude: float = 1.0, speed: float = 1.0):
        if self.emergency_stop_flag: return
        speed = self._clamp_speed(speed, self.min_lin_speed, self.max_lin_speed, "Takeoff")
        self.is_landed = False
        self.integral_v = np.zeros(3)
        
        pos, quat = self._get_pose()
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        self.target_yaw = r.as_euler('xyz')[2]
        self.target_pos = pos.copy()
        
        # Calculate how much we need to move up to reach absolute 'altitude'
        relative_dist = max(0.1, altitude - pos[2])
        self.action_move_distance(relative_dist, "up", speed)

    def action_land(self, speed: float = 0.5):
        if self.emergency_stop_flag or self.is_landed: return
        speed = self._clamp_speed(speed, self.min_lin_speed, self.max_land_speed, "Landing")
        pos, _ = self._get_pose()
        distance = max(0, pos[2] - 0.05) # Assume ground is Z ~ 0
        if distance > 0:
            print(f"[{self.robot_name}] Landing...")
            self.action_move_distance(distance, "down", speed)
        
        self.is_landed = True
        for aid in self.thrust_ids:
            if aid != -1: self.data.ctrl[aid] = 0.0
        print(f"[{self.robot_name}] Landed. Motors off.")

    def action_move_distance(self, distance: float, direction: str = "forward", speed: float = 1.0):
        """Moves exactly `distance` at `speed` m/s using a dynamic PID setpoint."""
        if self.emergency_stop_flag or self.is_landed: return
        speed = self._clamp_speed(speed, self.min_lin_speed, self.max_lin_speed, "Movement")
        
        pos, quat = self._get_pose()
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        yaw = r.as_euler('xyz')[2]
        
        # Calculate local direction vector
        dir_vec_local = np.array([0.0, 0.0, 0.0])
        if direction == "forward": dir_vec_local[0] = 1.0
        elif direction == "backward": dir_vec_local[0] = -1.0
        elif direction == "left": dir_vec_local[1] = 1.0
        elif direction == "right": dir_vec_local[1] = -1.0
        elif direction == "up": dir_vec_local[2] = 1.0
        elif direction == "down": dir_vec_local[2] = -1.0
        else:
            print(f"[{self.robot_name}] Unknown direction: {direction}")
            return
            
        # Rotate local direction to world direction based on current yaw
        r_yaw = R.from_euler('z', yaw)
        dir_vec_world = r_yaw.apply(dir_vec_local)
        
        start_pos = pos.copy()
        
        # Perfect S-Curve (Trapezoidal) Generator
        max_a = 2.0 # Acceleration limit (m/s^2)
        accel_time = speed / max_a
        accel_dist = 0.5 * max_a * accel_time**2
        
        if distance < 2 * accel_dist:
            # Triangular profile
            accel_time = np.sqrt(distance / max_a)
            actual_max_v = max_a * accel_time
            cruise_time = 0.0
            duration = 2 * accel_time
        else:
            # Trapezoidal profile
            actual_max_v = speed
            cruise_dist = distance - 2 * accel_dist
            cruise_time = cruise_dist / actual_max_v
            duration = 2 * accel_time + cruise_time

        target_pos = start_pos + dir_vec_world * distance
        
        sim_start_time = self.data.time
        self.integral_v = np.zeros(3) # Reset integral for clean start
        
        settle_time = 1.0 # Allow 1 second to settle after reaching target
        
        print(f"[{self.robot_name}] Moving {distance:.2f}m {direction} at {speed:.2f}m/s (Traj Duration: {duration:.2f}s + {settle_time:.1f}s settle time)...")
            
        while True:
            if self.emergency_stop_flag: break
            
            t = self.data.time - sim_start_time
            
            # Smooth velocity profile (Trapezoidal)
            if t < accel_time:
                # Accelerating
                current_speed = max_a * t
                traveled = 0.5 * max_a * t**2
            elif t < accel_time + cruise_time:
                # Cruising
                current_speed = actual_max_v
                traveled = accel_dist + actual_max_v * (t - accel_time)
            elif t < duration:
                # Decelerating
                rem_t = duration - t
                current_speed = max_a * rem_t
                traveled = distance - 0.5 * max_a * rem_t**2
            else:
                current_speed = 0.0
                traveled = distance
            
            setpoint_pos = start_pos + dir_vec_world * traveled
            target_vel = dir_vec_world * current_speed
            
            self._control_step(setpoint_pos, yaw, target_vel=target_vel, dt=self.control_dt)
            
            # Stop condition: Trajectory elapsed AND settle time finished (deterministic)
            if t >= duration + settle_time:
                self.target_pos = target_pos
                self.target_yaw = yaw
                break
                    
            target_sim_time = self.data.time + self.control_dt
            while self.data.time < target_sim_time:
                time.sleep(0.001)
                
        # self.target_pos and self.target_yaw are already set

    def action_rotate(self, angle_deg: float, speed: float = 30.0):
        """Rotates exactly `angle_deg` at `speed` deg/s."""
        if self.emergency_stop_flag or self.is_landed: return
        speed = self._clamp_speed(speed, self.min_ang_speed, self.max_ang_speed, "Rotation")
        
        angle_rad = np.deg2rad(angle_deg)
        speed_rad = np.deg2rad(speed)
        
        # Perfect S-Curve (Trapezoidal) Generator for Rotation
        max_a = np.deg2rad(90.0) # Acceleration limit (deg/s^2)
        accel_time = speed_rad / max_a
        accel_dist = 0.5 * max_a * accel_time**2
        
        if abs(angle_rad) < 2 * accel_dist:
            accel_time = np.sqrt(abs(angle_rad) / max_a)
            actual_max_v = max_a * accel_time
            cruise_time = 0.0
            duration = 2 * accel_time
        else:
            actual_max_v = speed_rad
            cruise_dist = abs(angle_rad) - 2 * accel_dist
            cruise_time = cruise_dist / actual_max_v
            duration = 2 * accel_time + cruise_time
        
        start_yaw = self.target_yaw
        target_yaw = start_yaw + angle_rad
        
        settle_time = 1.0 # Extra time allowed to stop spinning completely
        
        sim_start_time = self.data.time
        print(f"[{self.robot_name}] Rotating {angle_deg:.1f} deg at {speed:.1f} deg/s (Traj Duration: {duration:.2f}s + {settle_time:.1f}s settle time)...")
        
        while True:
            if self.emergency_stop_flag: break
            
            t = self.data.time - sim_start_time
            
            if t < accel_time:
                traveled = 0.5 * max_a * t**2
            elif t < accel_time + cruise_time:
                traveled = accel_dist + actual_max_v * (t - accel_time)
            elif t < duration:
                rem_t = duration - t
                traveled = abs(angle_rad) - 0.5 * max_a * rem_t**2
            else:
                traveled = abs(angle_rad)
                
            setpoint_yaw = start_yaw + np.sign(angle_rad) * traveled
            
            # Maintain position while rotating
            self._control_step(self.target_pos, setpoint_yaw, dt=self.control_dt)
            
            # Stop condition: Trajectory elapsed AND settle time finished (deterministic)
            if t >= duration + settle_time:
                self.target_yaw = target_yaw
                break
                    
            target_sim_time = self.data.time + self.control_dt
            while self.data.time < target_sim_time:
                time.sleep(0.001)
        
        # self.target_yaw is already set
