import math
import unittest
from types import SimpleNamespace

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


class FakeMotionDrive(DifferentialDriveController):
    WHEEL_RADIUS = 0.1
    WHEEL_TRACK = 0.5
    DEFAULT_LINEAR_SPEED = 0.5
    MAX_LINEAR_SPEED = 1.0
    DEFAULT_ANGULAR_SPEED_DEG = 45.0
    MAX_ANGULAR_SPEED_DEG = 90.0
    LINEAR_ACCEL = 0.5
    ANGULAR_ACCEL_DEG = 90.0
    HEADING_KP = 0.0
    TURN_KP = 0.0
    SETTLE_DURATION = 0.05
    MAX_COMPLETION_OVERRUN = 0.5

    def __init__(self):
        self.robot_name = "fake"
        self.control_dt = 0.05
        self.model = None
        self.data = SimpleNamespace(time=0.0)
        self.base_pos = np.zeros(3)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.emergency_stop_flag = False
        self.log_file = None
        self.left_id = 0
        self.right_id = 1
        self.body_id = 2
        self.left_sign = 1.0
        self.right_sign = 1.0
        self.pose = np.array([0.0, 0.0, 0.0], dtype=float)
        self.last_command = (0.0, 0.0)

    def _lookup_handles(self):
        return None

    def _get_pose(self):
        return tuple(self.pose)

    def _get_base_velocity(self):
        left, right = self.last_command
        linear = self.WHEEL_RADIUS * 0.5 * (left + right)
        angular = self.WHEEL_RADIUS * (right - left) / self.WHEEL_TRACK
        return linear, angular

    def _command_wheels(self, left_rad_s: float, right_rad_s: float):
        self.last_command = (left_rad_s, right_rad_s)

    def _stop_motion(self):
        self.last_command = (0.0, 0.0)

    def _wait_until(self, target_sim_time: float):
        while self.data.time < target_sim_time:
            step = min(self.control_dt, target_sim_time - self.data.time)
            left, right = self.last_command
            linear = self.WHEEL_RADIUS * 0.5 * (left + right)
            angular = self.WHEEL_RADIUS * (right - left) / self.WHEEL_TRACK
            x, y, yaw = self.pose
            x += linear * math.cos(yaw) * step
            y += linear * math.sin(yaw) * step
            yaw = (yaw + angular * step + math.pi) % (2.0 * math.pi) - math.pi
            self.pose[:] = [x, y, yaw]
            self.data.time += step


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

    def test_angular_profile_timing_changes_with_distance_and_speed(self):
        profile = self.ctrl._build_angular_profile(angle_rad=math.pi / 2, speed_deg=30.0)
        self.assertGreater(profile["cruise_time"], 0.0)
        self.assertGreater(profile["duration"], 2.0)
        self.assertLess(profile["duration"], 4.0)

    def test_diff_drive_mapping_matches_expected_wheel_speeds(self):
        left, right = self.ctrl._body_twist_to_wheels(linear_m_s=0.5, angular_rad_s=0.4)
        self.assertAlmostEqual(left, 4.0, places=6)
        self.assertAlmostEqual(right, 6.0, places=6)

    def test_resolve_turn_delta_honors_auto_cw_and_ccw(self):
        start = math.radians(10.0)
        target = math.radians(350.0)
        auto_delta = self.ctrl._resolve_turn_delta(start, target, "auto")
        ccw_delta = self.ctrl._resolve_turn_delta(start, target, "ccw")
        cw_delta = self.ctrl._resolve_turn_delta(start, target, "cw")

        self.assertAlmostEqual(auto_delta, math.radians(-20.0), places=6)
        self.assertGreater(ccw_delta, 0.0)
        self.assertAlmostEqual(ccw_delta, math.radians(340.0), places=6)
        self.assertLess(cw_delta, 0.0)
        self.assertAlmostEqual(cw_delta, math.radians(-20.0), places=6)

    def test_invalid_straight_direction_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Invalid straight-motion direction"):
            self.ctrl._validate_straight_direction("left")

    def test_invalid_turn_direction_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Invalid turn direction"):
            self.ctrl._validate_turn_direction("spin")

    def test_missing_handles_raise_before_motion(self):
        ctrl = DummyDrive()
        ctrl.left_actuator_name = "left"
        ctrl.right_actuator_name = "right"
        ctrl.base_body_name = "base"
        ctrl.left_id = -1
        ctrl.right_id = -1
        ctrl.body_id = -1

        with self.assertRaisesRegex(RuntimeError, "unresolved handles"):
            ctrl.action_move_straight(0.25)

    def test_move_straight_completes_with_fake_motion_loop(self):
        ctrl = FakeMotionDrive()

        ctrl.action_move_straight(distance=0.25, direction="forward", speed=0.5)

        self.assertGreater(ctrl.pose[0], 0.20)
        self.assertAlmostEqual(ctrl.pose[1], 0.0, places=3)
        self.assertEqual(ctrl.last_command, (0.0, 0.0))
        self.assertGreater(ctrl.data.time, 0.0)

    def test_turn_completes_with_fake_motion_loop(self):
        ctrl = FakeMotionDrive()

        ctrl.action_turn(target_yaw=90.0, direction="ccw", speed=45.0)

        self.assertAlmostEqual(math.degrees(ctrl.pose[2]), 90.0, delta=5.0)
        self.assertEqual(ctrl.last_command, (0.0, 0.0))
        self.assertGreater(ctrl.data.time, 0.0)


if __name__ == "__main__":
    unittest.main()
