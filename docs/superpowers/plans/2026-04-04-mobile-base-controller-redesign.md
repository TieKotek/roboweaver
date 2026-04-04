# Mobile Base Controller Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the `rbtheron` and `tracer` mobile base controllers around deterministic speed-based motion profiles, shared differential-drive logic, and stabilized MJCF parameters while preserving existing JSON actions.

**Architecture:** Add a shared differential-drive base controller in `common/` that owns pose reading, trapezoidal/triangular profile generation, and action execution for straight motion and rotation. Keep `RbtheronController` and `TracerController` as thin wrappers that only define robot-specific actuator naming, wheel geometry, and tuning. Update each robot's MJCF so the new trajectory-following controller runs on stable wheel-ground contact rather than compensating for model defects.

**Tech Stack:** Python 3, MuJoCo, NumPy, standard-library `unittest`, JSON scenario configs, MJCF XML

---

## File Structure

- Create: `common/differential_drive_controller.py`
  Purpose: Shared differential-drive controller base class, velocity-profile helpers, speed clamping, wheel-command conversion, and deterministic simulation-time execution.
- Create: `tests/test_differential_drive_controller.py`
  Purpose: Unit tests for profile timing, wheel-speed conversion, yaw normalization, and direction handling without needing a live MuJoCo scene.
- Modify: `robots/rbtheron_control/rbtheron_controller.py`
  Purpose: Replace duplicate controller logic with a thin robot-specific subclass of the new shared base class.
- Modify: `robots/tracer_control/tracer_controller.py`
  Purpose: Replace duplicate controller logic with a thin robot-specific subclass of the new shared base class.
- Modify: `robots/rbtheron_control/rbtheron/rbtheron.xml`
  Purpose: Stabilize chassis mass/contact geometry and wheel/caster interaction for deterministic straight and turning motion.
- Modify: `robots/tracer_control/agilex_tracer2/tracer2.xml`
  Purpose: Tune chassis and wheel contact/actuator parameters to reduce drift and improve traction.
- Modify: `examples/rbtheron_mobile_base_demo.json`
  Purpose: Exercise explicit linear and angular speed parameters in the baseline validation scenario.
- Modify: `examples/tracer_mobile_base_demo.json`
  Purpose: Exercise explicit linear and angular speed parameters in the baseline validation scenario.

### Task 1: Add Trajectory Math Tests

**Files:**
- Create: `tests/test_differential_drive_controller.py`

- [ ] **Step 1: Write the failing test file**

```python
import math
import unittest
import numpy as np

from common.differential_drive_controller import DifferentialDriveController


class DummyDrive(DifferentialDriveController):
    WHEEL_RADIUS = 0.1
    WHEEL_TRACK = 0.5
    DEFAULT_LINEAR_SPEED = 0.6
    MAX_LINEAR_SPEED = 1.2
    DEFAULT_ANGULAR_SPEED_DEG = 45.0
    MAX_ANGULAR_SPEED_DEG = 90.0
    LINEAR_ACCEL = 0.5
    ANGULAR_ACCEL_DEG = 90.0

    def __init__(self):
        self.robot_name = "dummy"
        self.control_dt = 0.01
        self.model = None
        self.data = None
        self.base_pos = np.zeros(3)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.emergency_stop_flag = False
        self.log_file = None

    def _lookup_handles(self):
        return None

    def _get_pose(self):
        return 0.0, 0.0, 0.0

    def _get_base_velocity(self):
        return 0.0, 0.0

    def _command_wheels(self, left_rad_s: float, right_rad_s: float):
        self.last_command = (left_rad_s, right_rad_s)

    def _stop_motion(self):
        self.last_command = (0.0, 0.0)


class DifferentialDriveProfileTests(unittest.TestCase):
    def setUp(self):
        self.ctrl = DummyDrive()

    def test_linear_profile_becomes_trapezoid_when_distance_is_long(self):
        profile = self.ctrl._build_linear_profile(distance=1.0, speed=0.5)
        self.assertAlmostEqual(profile["accel_time"], 1.0, places=6)
        self.assertAlmostEqual(profile["cruise_time"], 1.0, places=6)
        self.assertAlmostEqual(profile["duration"], 3.0, places=6)

    def test_linear_profile_becomes_triangle_when_distance_is_short(self):
        profile = self.ctrl._build_linear_profile(distance=0.2, speed=0.8)
        self.assertAlmostEqual(profile["cruise_time"], 0.0, places=6)
        self.assertAlmostEqual(profile["duration"], 2.0 * math.sqrt(0.2 / 0.5), places=6)

    def test_rotation_profile_uses_degree_speed_limit(self):
        profile = self.ctrl._build_angular_profile(angle_rad=math.pi / 2, speed_deg=45.0)
        self.assertGreater(profile["duration"], 2.0)
        self.assertLess(profile["duration"], 4.0)

    def test_diff_drive_mapping_matches_expected_wheel_speeds(self):
        left, right = self.ctrl._body_twist_to_wheels(linear_m_s=0.5, angular_rad_s=0.4)
        self.assertAlmostEqual(left, 4.0, places=6)
        self.assertAlmostEqual(right, 6.0, places=6)

    def test_resolve_turn_delta_honors_forced_direction(self):
        start = math.radians(10.0)
        target = math.radians(350.0)
        self.assertLess(self.ctrl._resolve_turn_delta(start, target, "auto"), 0.0)
        self.assertGreater(self.ctrl._resolve_turn_delta(start, target, "ccw"), 0.0)
        self.assertLess(self.ctrl._resolve_turn_delta(start, target, "cw"), 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_differential_drive_controller -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.differential_drive_controller'`

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_differential_drive_controller.py
git commit -m "test: add diff drive trajectory math coverage"
```

### Task 2: Implement Shared Differential-Drive Base

**Files:**
- Create: `common/differential_drive_controller.py`
- Modify: `tests/test_differential_drive_controller.py`

- [ ] **Step 1: Write the shared base implementation**

```python
import math
import time
from dataclasses import dataclass

import numpy as np

from common.robot_api import BaseRobotController, RobotState


@dataclass
class DifferentialDriveState(RobotState):
    global_pose: np.ndarray = None
    body_velocity: np.ndarray = None


class DifferentialDriveController(BaseRobotController):
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
    SETTLE_DURATION = 0.25

    def get_robot_state(self) -> DifferentialDriveState:
        x, y, yaw = self._get_pose()
        linear, angular = self._get_base_velocity()
        return DifferentialDriveState(
            timestamp=self.data.time,
            global_pose=np.array([x, y, yaw]),
            body_velocity=np.array([linear, angular]),
        )

    def _normalize_angle(self, angle_rad: float) -> float:
        return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi

    def _clamp_linear_speed(self, speed: float) -> float:
        return float(np.clip(speed, 0.01, self.MAX_LINEAR_SPEED))

    def _clamp_angular_speed_deg(self, speed_deg: float) -> float:
        return float(np.clip(speed_deg, 1.0, self.MAX_ANGULAR_SPEED_DEG))

    def _build_linear_profile(self, distance: float, speed: float) -> dict:
        speed = self._clamp_linear_speed(speed)
        accel = self.LINEAR_ACCEL
        accel_time = speed / accel
        accel_dist = 0.5 * accel * accel_time ** 2
        if distance < 2.0 * accel_dist:
            accel_time = math.sqrt(distance / accel)
            cruise_time = 0.0
            peak_speed = accel * accel_time
            duration = 2.0 * accel_time
        else:
            peak_speed = speed
            cruise_dist = distance - 2.0 * accel_dist
            cruise_time = cruise_dist / peak_speed
            duration = 2.0 * accel_time + cruise_time
        return {
            "distance": distance,
            "accel": accel,
            "accel_time": accel_time,
            "cruise_time": cruise_time,
            "peak_speed": peak_speed,
            "duration": duration,
        }

    def _build_angular_profile(self, angle_rad: float, speed_deg: float) -> dict:
        speed_deg = self._clamp_angular_speed_deg(speed_deg)
        accel = math.radians(self.ANGULAR_ACCEL_DEG)
        peak_speed = math.radians(speed_deg)
        angle_abs = abs(angle_rad)
        accel_time = peak_speed / accel
        accel_dist = 0.5 * accel * accel_time ** 2
        if angle_abs < 2.0 * accel_dist:
            accel_time = math.sqrt(angle_abs / accel)
            cruise_time = 0.0
            peak_speed = accel * accel_time
            duration = 2.0 * accel_time
        else:
            cruise_dist = angle_abs - 2.0 * accel_dist
            cruise_time = cruise_dist / peak_speed
            duration = 2.0 * accel_time + cruise_time
        return {
            "angle_rad": angle_rad,
            "accel": accel,
            "accel_time": accel_time,
            "cruise_time": cruise_time,
            "peak_speed": peak_speed,
            "duration": duration,
        }

    def _body_twist_to_wheels(self, linear_m_s: float, angular_rad_s: float) -> tuple[float, float]:
        left_linear = linear_m_s - 0.5 * self.WHEEL_TRACK * angular_rad_s
        right_linear = linear_m_s + 0.5 * self.WHEEL_TRACK * angular_rad_s
        return left_linear / self.WHEEL_RADIUS, right_linear / self.WHEEL_RADIUS

    def _resolve_turn_delta(self, start_yaw: float, target_yaw: float, direction: str) -> float:
        delta = self._normalize_angle(target_yaw - start_yaw)
        if direction == "ccw" and delta < 0.0:
            delta += 2.0 * math.pi
        elif direction == "cw" and delta > 0.0:
            delta -= 2.0 * math.pi
        return delta
```

- [ ] **Step 2: Run the unit tests**

Run: `python -m unittest tests.test_differential_drive_controller -v`
Expected: PASS for all five tests

- [ ] **Step 3: Extend the shared base with deterministic actions**

```python
    def _wait_until(self, target_sim_time: float):
        while self.data.time < target_sim_time:
            if self.emergency_stop_flag:
                break
            time.sleep(0.001)

    def _sample_profile(self, profile: dict, t: float) -> tuple[float, float]:
        accel = profile["accel"]
        accel_time = profile["accel_time"]
        cruise_time = profile["cruise_time"]
        total = profile["duration"]
        peak = profile["peak_speed"]
        total_dist = profile.get("distance", abs(profile["angle_rad"]))
        if t < accel_time:
            speed = accel * t
            traveled = 0.5 * accel * t ** 2
        elif t < accel_time + cruise_time:
            speed = peak
            traveled = 0.5 * accel * accel_time ** 2 + peak * (t - accel_time)
        elif t < total:
            remaining = total - t
            speed = accel * remaining
            traveled = total_dist - 0.5 * accel * remaining ** 2
        else:
            speed = 0.0
            traveled = total_dist
        return traveled, speed

    def action_move_straight(self, distance: float, direction: str = "forward", speed: float | None = None):
        speed = self.DEFAULT_LINEAR_SPEED if speed is None else speed
        profile = self._build_linear_profile(abs(distance), speed)
        sign = -1.0 if direction == "backward" else 1.0
        start_x, start_y, start_yaw = self._get_pose()
        sim_start = self.data.time
        print(
            f"[{self.robot_name}] Moving {distance:.2f}m {direction} at {profile['peak_speed']:.2f}m/s "
            f"(Traj Duration: {profile['duration']:.2f}s + {self.SETTLE_DURATION:.2f}s settle time)..."
        )
        while not self.emergency_stop_flag:
            t = self.data.time - sim_start
            traveled, speed_now = self._sample_profile(profile, t)
            yaw_error = self._normalize_angle(start_yaw - self._get_pose()[2])
            angular_cmd = self.HEADING_KP * yaw_error
            left, right = self._body_twist_to_wheels(sign * speed_now, angular_cmd)
            self._command_wheels(left, right)
            if t >= profile["duration"] + self.SETTLE_DURATION:
                break
            self._wait_until(self.data.time + self.control_dt)
        self._stop_motion()

    def action_turn(self, target_yaw: float, direction: str = "auto", speed: float | None = None):
        speed = self.DEFAULT_ANGULAR_SPEED_DEG if speed is None else speed
        _, _, start_yaw = self._get_pose()
        delta = self._resolve_turn_delta(start_yaw, math.radians(target_yaw), direction)
        profile = self._build_angular_profile(delta, speed)
        sim_start = self.data.time
        print(
            f"[{self.robot_name}] Turning to {target_yaw:.1f} deg at {math.degrees(profile['peak_speed']):.1f} deg/s "
            f"(Traj Duration: {profile['duration']:.2f}s + {self.SETTLE_DURATION:.2f}s settle time)..."
        )
        while not self.emergency_stop_flag:
            t = self.data.time - sim_start
            traveled, speed_now = self._sample_profile(profile, t)
            target = start_yaw + math.copysign(traveled, delta)
            yaw_error = self._normalize_angle(target - self._get_pose()[2])
            angular_cmd = math.copysign(speed_now, delta) + self.TURN_KP * yaw_error
            left, right = self._body_twist_to_wheels(0.0, angular_cmd)
            self._command_wheels(left, right)
            if t >= profile["duration"] + self.SETTLE_DURATION:
                break
            self._wait_until(self.data.time + self.control_dt)
        self._stop_motion()
```

- [ ] **Step 4: Run the unit tests again**

Run: `python -m unittest tests.test_differential_drive_controller -v`
Expected: PASS and no new failures

- [ ] **Step 5: Commit the shared controller**

```bash
git add common/differential_drive_controller.py tests/test_differential_drive_controller.py
git commit -m "feat: add shared differential drive controller"
```

### Task 3: Port `tracer` to the Shared Base

**Files:**
- Modify: `robots/tracer_control/tracer_controller.py`

- [ ] **Step 1: Replace duplicated controller logic with a thin subclass**

```python
import mujoco

from common.differential_drive_controller import DifferentialDriveController


class TracerController(DifferentialDriveController):
    WHEEL_RADIUS = 0.085
    WHEEL_TRACK = 0.5074
    DEFAULT_LINEAR_SPEED = 0.5
    MAX_LINEAR_SPEED = 1.5
    DEFAULT_ANGULAR_SPEED_DEG = 45.0
    MAX_ANGULAR_SPEED_DEG = 120.0
    LINEAR_ACCEL = 0.8
    ANGULAR_ACCEL_DEG = 180.0
    HEADING_KP = 2.5
    TURN_KP = 3.5

    def __init__(self, model, data, robot_name, base_pos=None, base_quat=None, log_dir=None, **kwargs):
        super().__init__(model, data, robot_name, base_pos, base_quat, log_dir)
        self.left_actuator_name = f"{robot_name}_left_drive"
        self.right_actuator_name = f"{robot_name}_right_drive"
        self.base_body_name = f"{robot_name}_base_link"
        self.left_sign = -1.0
        self.right_sign = 1.0
        self._lookup_handles()

    def _lookup_handles(self):
        self.left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.left_actuator_name)
        self.right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.right_actuator_name)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_body_name)

    def _command_wheels(self, left_rad_s: float, right_rad_s: float):
        self.data.ctrl[self.left_id] = self.left_sign * left_rad_s
        self.data.ctrl[self.right_id] = self.right_sign * right_rad_s
```

- [ ] **Step 2: Run the tracer demo headless to ensure the subclass loads**

Run: `python run_robots.py examples/tracer_mobile_base_demo.json --headless`
Expected: scenario completes without `Action 'turn' not implemented` or actuator lookup errors

- [ ] **Step 3: Commit the tracer port**

```bash
git add robots/tracer_control/tracer_controller.py
git commit -m "refactor: port tracer to shared diff drive base"
```

### Task 4: Tune `tracer` MJCF and Demo Inputs

**Files:**
- Modify: `robots/tracer_control/agilex_tracer2/tracer2.xml`
- Modify: `examples/tracer_mobile_base_demo.json`

- [ ] **Step 1: Update actuator and contact parameters for traction**

```xml
<default>
  <default class="collision">
    <geom group="1" friction="4.5 0.02 0.0001" solref="0.003 1" solimp="0.95 0.99 0.001"/>
    <default class="wheel">
      <geom type="cylinder" size="0.085 0.02" friction="5.0 0.02 0.0001"/>
    </default>
    <default class="caster">
      <geom type="sphere" size="0.02" friction="0.0005 0.005 0.0001"/>
    </default>
  </default>
</default>

<actuator>
  <velocity name="right_drive" joint="right_joint" kv="80" ctrlrange="-30 30"/>
  <velocity name="left_drive" joint="left_joint" kv="80" ctrlrange="-30 30"/>
</actuator>
```

- [ ] **Step 2: Add explicit speed parameters to the tracer demo**

```json
{
  "sequence": [
    {"action": "idle", "parameters": {"duration": 1.0}},
    {"action": "turn", "parameters": {"target_yaw": 0.0, "direction": "auto", "speed": 45.0}},
    {"action": "idle", "parameters": {"duration": 1.0}},
    {"action": "move_straight", "parameters": {"distance": 1.0, "direction": "forward", "speed": 0.6}},
    {"action": "idle", "parameters": {"duration": 1.0}},
    {"action": "turn", "parameters": {"target_yaw": 90.0, "direction": "ccw", "speed": 60.0}},
    {"action": "idle", "parameters": {"duration": 1.0}},
    {"action": "move_straight", "parameters": {"distance": 1.0, "direction": "forward", "speed": 0.6}},
    {"action": "idle", "parameters": {"duration": 1.0}},
    {"action": "turn", "parameters": {"target_yaw": 0.0, "direction": "cw", "speed": 60.0}},
    {"action": "idle", "parameters": {"duration": 1.0}},
    {"action": "move_straight", "parameters": {"distance": 1.0, "direction": "backward", "speed": 0.5}}
  ]
}
```

- [ ] **Step 3: Run the tracer demo and inspect timing output**

Run: `python run_robots.py examples/tracer_mobile_base_demo.json --headless`
Expected: controller prints `Traj Duration:` lines for turns and straight moves, and the log file contains matching simulation durations within a small tolerance

- [ ] **Step 4: Commit tracer physical tuning**

```bash
git add robots/tracer_control/agilex_tracer2/tracer2.xml examples/tracer_mobile_base_demo.json
git commit -m "tune tracer diff drive motion profile"
```

### Task 5: Port `rbtheron` to the Shared Base

**Files:**
- Modify: `robots/rbtheron_control/rbtheron_controller.py`

- [ ] **Step 1: Replace duplicated controller logic with a thin subclass**

```python
import mujoco

from common.differential_drive_controller import DifferentialDriveController


class RbtheronController(DifferentialDriveController):
    WHEEL_RADIUS = 0.0762
    WHEEL_TRACK = 0.5032
    DEFAULT_LINEAR_SPEED = 0.45
    MAX_LINEAR_SPEED = 1.2
    DEFAULT_ANGULAR_SPEED_DEG = 40.0
    MAX_ANGULAR_SPEED_DEG = 100.0
    LINEAR_ACCEL = 0.6
    ANGULAR_ACCEL_DEG = 150.0
    HEADING_KP = 3.0
    TURN_KP = 4.0

    def __init__(self, model, data, robot_name, base_pos=None, base_quat=None, log_dir=None, **kwargs):
        super().__init__(model, data, robot_name, base_pos, base_quat, log_dir)
        self.left_actuator_name = f"{robot_name}_left_wheel_vel"
        self.right_actuator_name = f"{robot_name}_right_wheel_vel"
        self.base_body_name = f"{robot_name}_base_link"
        self.left_sign = 1.0
        self.right_sign = 1.0
        self._lookup_handles()

    def _lookup_handles(self):
        self.left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.left_actuator_name)
        self.right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.right_actuator_name)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_body_name)

    def _command_wheels(self, left_rad_s: float, right_rad_s: float):
        self.data.ctrl[self.left_id] = self.left_sign * left_rad_s
        self.data.ctrl[self.right_id] = self.right_sign * right_rad_s
```

- [ ] **Step 2: Run the rbtheron demo headless to ensure the subclass loads**

Run: `python run_robots.py examples/rbtheron_mobile_base_demo.json --headless`
Expected: scenario completes without `Action 'move_straight' not implemented` or actuator lookup errors

- [ ] **Step 3: Commit the rbtheron port**

```bash
git add robots/rbtheron_control/rbtheron_controller.py
git commit -m "refactor: port rbtheron to shared diff drive base"
```

### Task 6: Rebuild `rbtheron` MJCF for Stable Wheel Contact

**Files:**
- Modify: `robots/rbtheron_control/rbtheron/rbtheron.xml`
- Modify: `examples/rbtheron_mobile_base_demo.json`

- [ ] **Step 1: Replace the chassis and support-wheel dynamics with a simpler stable model**

```xml
<body name="base_link" pos="0 0 0.09">
  <freejoint/>
  <inertial pos="0 0 0" mass="42" diaginertia="1.8 1.8 2.4"/>
  <geom type="box" size="0.30 0.24 0.07" rgba="0 0 0 0" friction="1.0 0.005 0.0001"/>
  <geom pos="0 0 -0.01" type="box" size="0.28 0.22 0.02" rgba="0 0 0 0" contype="0" conaffinity="0"/>
  <geom pos="0 0 0.02" quat="1 0 0 0" type="mesh" rgba="0.2 0.2 0.2 1" mesh="theron_base_v4" contype="0" conaffinity="0"/>

  <body name="right_wheel_link" pos="0 -0.2516 -0.0138">
    <inertial pos="0 0 0" mass="2.0" diaginertia="0.01 0.005 0.005"/>
    <joint name="right_wheel_joint" axis="0 1 0" damping="0.2" armature="0.02"/>
    <geom type="cylinder" size="0.0762 0.03" quat="0.707107 0.707107 0 0" friction="5.0 0.02 0.0001"/>
  </body>

  <body name="left_wheel_link" pos="0 0.2516 -0.0138">
    <inertial pos="0 0 0" mass="2.0" diaginertia="0.01 0.005 0.005"/>
    <joint name="left_wheel_joint" axis="0 1 0" damping="0.2" armature="0.02"/>
    <geom type="cylinder" size="0.0762 0.03" quat="0.707107 0.707107 0 0" friction="5.0 0.02 0.0001"/>
  </body>

  <body name="front_support" pos="0.235 0 -0.03">
    <geom type="sphere" size="0.03" friction="0.0005 0.005 0.0001"/>
  </body>

  <body name="rear_support" pos="-0.235 0 -0.03">
    <geom type="sphere" size="0.03" friction="0.0005 0.005 0.0001"/>
  </body>
</body>

<actuator>
  <velocity name="right_wheel_vel" joint="right_wheel_joint" kv="90" ctrlrange="-25 25"/>
  <velocity name="left_wheel_vel" joint="left_wheel_joint" kv="90" ctrlrange="-25 25"/>
</actuator>
```

- [ ] **Step 2: Add explicit speed parameters to the rbtheron demo**

```json
{
  "sequence": [
    {"action": "idle", "parameters": {"duration": 1.0}},
    {"action": "turn", "parameters": {"target_yaw": 0.0, "direction": "auto", "speed": 40.0}},
    {"action": "idle", "parameters": {"duration": 1.0}},
    {"action": "move_straight", "parameters": {"distance": 1.0, "direction": "forward", "speed": 0.5}},
    {"action": "idle", "parameters": {"duration": 1.0}},
    {"action": "turn", "parameters": {"target_yaw": 90.0, "direction": "ccw", "speed": 50.0}},
    {"action": "idle", "parameters": {"duration": 1.0}},
    {"action": "move_straight", "parameters": {"distance": 1.0, "direction": "forward", "speed": 0.5}},
    {"action": "idle", "parameters": {"duration": 1.0}},
    {"action": "turn", "parameters": {"target_yaw": 0.0, "direction": "cw", "speed": 50.0}},
    {"action": "idle", "parameters": {"duration": 1.0}},
    {"action": "move_straight", "parameters": {"distance": 1.0, "direction": "backward", "speed": 0.45}}
  ]
}
```

- [ ] **Step 3: Run the rbtheron demo and inspect timing output**

Run: `python run_robots.py examples/rbtheron_mobile_base_demo.json --headless`
Expected: controller prints `Traj Duration:` lines, final pose drift is materially lower than the baseline run, and the log file shows simulation durations close to the planned values

- [ ] **Step 4: Commit rbtheron physical tuning**

```bash
git add robots/rbtheron_control/rbtheron/rbtheron.xml examples/rbtheron_mobile_base_demo.json
git commit -m "tune rbtheron diff drive model and demo"
```

### Task 7: Final Validation and Cleanup

**Files:**
- Modify: `common/differential_drive_controller.py`
- Modify: `robots/rbtheron_control/rbtheron_controller.py`
- Modify: `robots/tracer_control/tracer_controller.py`
- Modify: `robots/rbtheron_control/rbtheron/rbtheron.xml`
- Modify: `robots/tracer_control/agilex_tracer2/tracer2.xml`

- [ ] **Step 1: Run the full automated and scenario validation set**

Run: `python -m unittest tests.test_differential_drive_controller -v`
Expected: PASS

Run: `python run_robots.py examples/tracer_mobile_base_demo.json --headless`
Expected: completes with printed planned durations and no controller/runtime errors

Run: `python run_robots.py examples/rbtheron_mobile_base_demo.json --headless`
Expected: completes with printed planned durations and no controller/runtime errors

- [ ] **Step 2: Make any final tuning-only adjustments needed to match duration and pose targets**

```python
# Typical final tuning edits should stay inside these constants:
HEADING_KP = 2.0
TURN_KP = 3.0
SETTLE_DURATION = 0.25
```

```xml
<!-- Typical final tuning edits should stay inside these attributes: -->
friction="5.0 0.02 0.0001"
kv="80"
ctrlrange="-30 30"
```

- [ ] **Step 3: Run the validation set one more time after final tuning**

Run: `python -m unittest tests.test_differential_drive_controller -v`
Expected: PASS

Run: `python run_robots.py examples/tracer_mobile_base_demo.json --headless`
Expected: PASS with stable timing output

Run: `python run_robots.py examples/rbtheron_mobile_base_demo.json --headless`
Expected: PASS with improved straight-line and turn behavior

- [ ] **Step 4: Commit the final integrated redesign**

```bash
git add common/differential_drive_controller.py tests/test_differential_drive_controller.py robots/tracer_control/tracer_controller.py robots/rbtheron_control/rbtheron_controller.py robots/tracer_control/agilex_tracer2/tracer2.xml robots/rbtheron_control/rbtheron/rbtheron.xml examples/tracer_mobile_base_demo.json examples/rbtheron_mobile_base_demo.json
git commit -m "feat: redesign mobile base motion control"
```
