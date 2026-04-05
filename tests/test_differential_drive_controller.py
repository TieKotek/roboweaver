import math
import io
import struct
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np

from common.differential_drive_controller import DifferentialDriveController
from robots.rbtheron_control.rbtheron_controller import RbtheronController
from robots.tracer_control.tracer_controller import TracerController


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


class EarlyFinishDrive(FakeMotionDrive):
    MAX_COMPLETION_OVERRUN = 0.30
    LINEAR_DISTANCE_TOLERANCE = 0.01
    HEADING_TOLERANCE_DEG = 1.0


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

    def test_move_straight_can_finish_before_full_settle_window_when_already_stable(self):
        ctrl = EarlyFinishDrive()

        target_distance = 0.25
        speed = 0.5
        profile = ctrl._build_linear_profile(target_distance, speed)

        ctrl.action_move_straight(distance=target_distance, direction="forward", speed=speed)

        self.assertGreaterEqual(ctrl.pose[0], target_distance - ctrl.LINEAR_DISTANCE_TOLERANCE)
        self.assertLessEqual(ctrl.data.time, profile["duration"] + ctrl.control_dt)

    def test_move_log_omits_settle_text(self):
        ctrl = FakeMotionDrive()
        buf = io.StringIO()

        with redirect_stdout(buf):
            ctrl.action_move_straight(distance=0.25, direction="forward", speed=0.5)

        self.assertNotIn("settle time", buf.getvalue())

    def test_differential_drive_controllers_no_longer_define_settle_duration(self):
        self.assertNotIn("SETTLE_DURATION", DifferentialDriveController.__dict__)
        self.assertNotIn("SETTLE_DURATION", RbtheronController.__dict__)
        self.assertNotIn("SETTLE_DURATION", TracerController.__dict__)

    def test_mobile_base_motion_hyperparameters_are_aligned(self):
        for attr in (
            "DEFAULT_LINEAR_SPEED",
            "MAX_LINEAR_SPEED",
            "DEFAULT_ANGULAR_SPEED_DEG",
            "MAX_ANGULAR_SPEED_DEG",
            "LINEAR_ACCEL",
            "ANGULAR_ACCEL_DEG",
            "HEADING_KP",
            "TURN_KP",
            "LINEAR_DISTANCE_TOLERANCE",
            "HEADING_TOLERANCE_DEG",
            "MAX_COMPLETION_OVERRUN",
        ):
            self.assertEqual(getattr(RbtheronController, attr), getattr(TracerController, attr), attr)

    def test_tracer_support_plane_stays_close_to_drive_wheel_contact_plane(self):
        root = ET.parse("robots/tracer_control/agilex_tracer2/tracer2.xml").getroot()

        wheel_radius = None
        caster_radius = None
        wheel_bottom = None
        caster_bottoms = []

        for default in root.findall("./default/default/default"):
            if default.attrib.get("class") == "wheel":
                geom = default.find("geom")
                wheel_radius = float(geom.attrib["size"].split()[0])
            elif default.attrib.get("class") == "caster":
                geom = default.find("geom")
                caster_radius = float(geom.attrib["size"])

        right_link = root.find(".//body[@name='right_link']")
        self.assertIsNotNone(right_link)
        wheel_bottom = float(right_link.attrib["pos"].split()[2]) - wheel_radius

        for body_name in ("right_front_link", "right_rear_link", "left_front_link", "left_rear_link"):
            body = root.find(f".//body[@name='{body_name}']")
            self.assertIsNotNone(body)
            caster_bottoms.append(float(body.attrib["pos"].split()[2]) - caster_radius)

        self.assertIsNotNone(wheel_radius)
        self.assertIsNotNone(caster_radius)
        self.assertIsNotNone(wheel_bottom)
        self.assertEqual(len(caster_bottoms), 4)

        highest_caster_bottom = max(caster_bottoms)
        clearance = highest_caster_bottom - wheel_bottom
        self.assertGreaterEqual(clearance, -0.0005)
        self.assertLessEqual(clearance, 0.002)

    def test_tracer_caster_links_use_vertical_compliance_joints(self):
        root = ET.parse("robots/tracer_control/agilex_tracer2/tracer2.xml").getroot()

        caster_links = {
            "right_front_link",
            "right_rear_link",
            "left_front_link",
            "left_rear_link",
        }
        seen = set()
        for body in root.findall(".//body"):
            name = body.attrib.get("name")
            if name in caster_links:
                seen.add(name)
                joint = body.find("joint")
                self.assertIsNotNone(joint)
                self.assertEqual(joint.attrib.get("type"), "slide")
                self.assertEqual(joint.attrib.get("axis"), "0 0 1")

        self.assertEqual(seen, caster_links)

    def test_tracer_base_link_uses_explicit_inertial_and_simple_collision(self):
        root = ET.parse("robots/tracer_control/agilex_tracer2/tracer2.xml").getroot()

        base_body = root.find(".//body[@name='base_link']")
        self.assertIsNotNone(base_body)

        inertial = base_body.find("inertial")
        self.assertIsNotNone(inertial)
        self.assertAlmostEqual(float(inertial.attrib["mass"]), 24.0, places=6)

        chassis_collision = base_body.find("geom[@name='chassis_collision']")
        self.assertIsNotNone(chassis_collision)
        self.assertEqual(chassis_collision.attrib.get("type"), "box")

        collision_mesh = None
        for geom in base_body.findall("geom"):
            if geom.attrib.get("mesh") == "base_link" and geom.attrib.get("type") == "mesh":
                collision_mesh = geom
                break

        self.assertIsNotNone(collision_mesh)
        self.assertEqual(collision_mesh.attrib.get("contype"), "0")
        self.assertEqual(collision_mesh.attrib.get("conaffinity"), "0")

    def test_rbtheron_uses_four_caster_supports_close_to_drive_wheel_contact_plane(self):
        root = ET.parse("robots/rbtheron_control/rbtheron/rbtheron.xml").getroot()

        wheel_radius = None
        support_type = None
        support_half_height = None
        support_bottoms = []
        support_names = set()
        wheel_center_z = None

        for default in root.findall("./default/default"):
            if default.attrib.get("class") == "wheel":
                geom = default.find("geom")
                wheel_radius = float(geom.attrib["size"].split()[0])
            elif default.attrib.get("class") == "support":
                geom = default.find("geom")
                support_type = geom.attrib["type"]
                sizes = [float(value) for value in geom.attrib["size"].split()]
                if support_type == "sphere":
                    support_half_height = sizes[0]
                elif support_type == "box":
                    support_half_height = sizes[2]
                else:
                    support_half_height = sizes[0]

        for body in root.findall(".//body[@name='right_wheel_link']"):
            wheel_center_z = float(body.attrib["pos"].split()[2])

        for body in root.findall(".//body"):
            body_name = body.attrib.get("name", "")
            body_z = float(body.attrib.get("pos", "0 0 0").split()[2])
            for geom in body.findall("geom"):
                if geom.attrib.get("name") in {
                    "front_left_support",
                    "front_right_support",
                    "rear_left_support",
                    "rear_right_support",
                }:
                    support_names.add(geom.attrib["name"])
                    local_z = float(geom.attrib.get("pos", "0 0 0").split()[2])
                    support_bottoms.append(body_z + local_z - support_half_height)

        self.assertIsNotNone(wheel_radius)
        self.assertEqual(support_type, "sphere")
        self.assertIsNotNone(support_half_height)
        self.assertIsNotNone(wheel_center_z)
        self.assertEqual(
            support_names,
            {"front_left_support", "front_right_support", "rear_left_support", "rear_right_support"},
        )
        self.assertEqual(len(support_bottoms), 4)

        wheel_bottom = wheel_center_z - wheel_radius
        highest_support_bottom = max(support_bottoms)
        clearance = highest_support_bottom - wheel_bottom
        self.assertGreaterEqual(clearance, -0.0002)
        self.assertLessEqual(clearance, 0.0002)

    def test_rbtheron_support_wheels_are_visible_for_viewer_feedback(self):
        root = ET.parse("robots/rbtheron_control/rbtheron/rbtheron.xml").getroot()

        support_visual_names = {
            "front_left_support_visual",
            "front_right_support_visual",
            "rear_left_support_visual",
            "rear_right_support_visual",
        }
        found = set()
        support_visual_rgba = None
        for default in root.findall("./default/default"):
            if default.attrib.get("class") == "support_visual":
                geom = default.find("geom")
                support_visual_rgba = [float(value) for value in geom.attrib["rgba"].split()]
                break
        for geom in root.findall(".//geom"):
            name = geom.attrib.get("name")
            if name in support_visual_names:
                found.add(name)

        self.assertEqual(found, support_visual_names)
        self.assertIsNotNone(support_visual_rgba)
        self.assertGreaterEqual(support_visual_rgba[3], 0.8)

    def test_rbtheron_support_links_use_vertical_compliance_joints(self):
        root = ET.parse("robots/rbtheron_control/rbtheron/rbtheron.xml").getroot()

        support_links = {
            "front_left_support_link",
            "front_right_support_link",
            "rear_left_support_link",
            "rear_right_support_link",
        }
        seen = set()
        for body in root.findall(".//body"):
            name = body.attrib.get("name")
            if name in support_links:
                seen.add(name)
                joint = body.find("joint")
                self.assertIsNotNone(joint)
                self.assertEqual(joint.attrib.get("type"), "slide")
                self.assertEqual(joint.attrib.get("axis"), "0 0 1")

        self.assertEqual(seen, support_links)

    def test_rbtheron_visual_shell_does_not_float_far_above_supports(self):
        root = ET.parse("robots/rbtheron_control/rbtheron/rbtheron.xml").getroot()
        visual_z = None
        support_top = None

        chassis_visual = root.find(".//geom[@name='chassis_visual']")
        self.assertIsNotNone(chassis_visual)
        visual_z = float(chassis_visual.attrib["pos"].split()[2])

        support_default = None
        for default in root.findall("./default/default"):
            if default.attrib.get("class") == "support":
                support_default = default.find("geom")
                break

        self.assertIsNotNone(support_default)
        support_size = [float(value) for value in support_default.attrib["size"].split()]
        front_support_body = root.find(".//body[@name='front_left_support_link']")
        self.assertIsNotNone(front_support_body)
        support_top = float(front_support_body.attrib["pos"].split()[2]) + support_size[0]

        stl_path = "robots/rbtheron_control/rbtheron/assets/theron_base_v4.stl"
        with open(stl_path, "rb") as stl_file:
            stl_file.read(80)
            triangle_count = struct.unpack("<I", stl_file.read(4))[0]
            min_z = math.inf
            for _ in range(triangle_count):
                record = stl_file.read(50)
                vertices = struct.unpack("<12fH", record)
                min_z = min(min_z, vertices[5], vertices[8], vertices[11])

        visual_bottom = visual_z + min_z
        self.assertLessEqual(visual_bottom - support_top, 0.004)

    def test_mobile_base_speed_limits_stay_below_model_theoretical_limits(self):
        cases = [
            (
                "robots/rbtheron_control/rbtheron/rbtheron.xml",
                RbtheronController,
                ("left_wheel_vel", "right_wheel_vel"),
            ),
            (
                "robots/tracer_control/agilex_tracer2/tracer2.xml",
                TracerController,
                ("left_drive", "right_drive"),
            ),
        ]

        for xml_path, controller_cls, actuator_names in cases:
            root = ET.parse(xml_path).getroot()
            actuator_limits = []
            for actuator_name in actuator_names:
                actuator = root.find(f".//actuator/velocity[@name='{actuator_name}']")
                self.assertIsNotNone(actuator)
                ctrl_min, ctrl_max = [float(value) for value in actuator.attrib["ctrlrange"].split()]
                actuator_limits.append(min(abs(ctrl_min), abs(ctrl_max)))

            max_joint_rad_s = min(actuator_limits)
            max_linear_speed = max_joint_rad_s * controller_cls.WHEEL_RADIUS
            max_angular_speed_deg = math.degrees((2.0 * max_linear_speed) / controller_cls.WHEEL_TRACK)

            self.assertLessEqual(controller_cls.MAX_LINEAR_SPEED, max_linear_speed + 1e-6)
            self.assertLessEqual(controller_cls.MAX_ANGULAR_SPEED_DEG, max_angular_speed_deg + 1e-6)

    def test_warning_and_clamp_use_safe_speed_limits_for_linear_motion(self):
        ctrl = FakeMotionDrive()
        ctrl.left_id = 0
        ctrl.right_id = 1
        ctrl.body_id = 2
        ctrl.model = SimpleNamespace(
            actuator_ctrlrange=np.array([[-4.0, 4.0], [-4.0, 4.0]], dtype=float),
            actuator_gear=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        )
        ctrl._refresh_safe_speed_limits()
        buf = io.StringIO()

        with redirect_stdout(buf):
            ctrl.action_move_straight(distance=0.25, direction="forward", speed=0.8)

        output = buf.getvalue()
        self.assertIn("exceeds safe maximum", output)
        self.assertIn(f"Clamping to {ctrl.safe_linear_speed:.2f}", output)
        self.assertIn(f"at {ctrl.safe_linear_speed:.2f}m/s", output)


if __name__ == "__main__":
    unittest.main()
