import json
from pathlib import Path
import tempfile
import unittest

from mirobot_demo.generate_mirobot_scene import (
    ParserConfig,
    ParserState,
    build_scene_from_commands,
    convert_command_stream,
    convert_motion_command,
    convert_tool_command,
    parse_command_token,
)


class ParseCommandTokenTests(unittest.TestCase):
    def test_parse_g00_with_embedded_feed_and_axes(self):
        parsed = parse_command_token("M20 G90 G00 X163.5 Y1.9 Z71.8 A0 B0 C0 F2000")
        self.assertEqual(parsed.motion, "G00")
        self.assertEqual(parsed.feed_rate_mm_per_min, 2000.0)
        self.assertEqual(parsed.modal_commands, ["M20", "G90"])
        self.assertEqual(parsed.axes["X"], 163.5)
        self.assertEqual(parsed.axes["Y"], 1.9)
        self.assertEqual(parsed.axes["Z"], 71.8)
        self.assertEqual(parsed.axes["A"], 0.0)
        self.assertEqual(parsed.axes["B"], 0.0)
        self.assertEqual(parsed.axes["C"], 0.0)

    def test_parse_g04_dwell(self):
        parsed = parse_command_token("G04 P1.5")
        self.assertEqual(parsed.motion, "G04")
        self.assertEqual(parsed.dwell_seconds, 1.5)

    def test_parse_m3_pwm(self):
        parsed = parse_command_token("M3 S1000")
        self.assertEqual(parsed.tool_command, "M3")
        self.assertEqual(parsed.tool_value, 1000.0)


class ConversionTests(unittest.TestCase):
    def test_convert_g00_uses_world_pose_and_cartesian_action(self):
        config = ParserConfig()
        state = ParserState(
            robot_name="arm2",
            cartesian_mode=True,
            absolute_mode=True,
            feed_rate_mm_per_min=2000.0,
            current_axes={"X": 0.0, "Y": 0.0, "Z": 0.0, "A": 0.0, "B": 0.0, "C": 0.0},
            base_pos_m=[0.42, 0.23, 0.0],
        )
        parsed = parse_command_token("G00 X100 Y-140 Z205 A0 B-25 C-90 F2000")

        action = convert_motion_command(parsed, state, config)

        self.assertEqual(action["action"], "move_cartesian")
        self.assertEqual(action["parameters"]["pose"], [0.52, 0.09, 0.205])
        self.assertEqual(action["parameters"]["euler"], [0.0, -25.0, -90.0])
        self.assertGreater(action["parameters"]["duration"], 0.2)

    def test_convert_g01_uses_linear_action(self):
        config = ParserConfig()
        state = ParserState(
            robot_name="arm2",
            cartesian_mode=True,
            absolute_mode=True,
            feed_rate_mm_per_min=1000.0,
            current_axes={"X": 0.0, "Y": 0.0, "Z": 0.0, "A": 0.0, "B": 0.0, "C": 0.0},
            base_pos_m=[0.42, 0.23, 0.0],
        )
        parsed = parse_command_token("G01 X250 Y0 Z100 A0 B0 C0")

        action = convert_motion_command(parsed, state, config)

        self.assertEqual(action["action"], "move_linear")
        self.assertEqual(action["parameters"]["pose"], [0.67, 0.23, 0.1])

    def test_convert_m3_on_and_off_to_idle_with_descriptions(self):
        config = ParserConfig(tool_engage_wait_sec=0.8, tool_release_wait_sec=0.5)
        on_action = convert_tool_command(parse_command_token("M3 S1000"), config)
        off_action = convert_tool_command(parse_command_token("M3 S0"), config)

        self.assertEqual(on_action["action"], "idle")
        self.assertEqual(on_action["parameters"]["duration"], 0.8)
        self.assertIn("engage", on_action["description"].lower())
        self.assertEqual(off_action["parameters"]["duration"], 0.5)
        self.assertIn("release", off_action["description"].lower())

    def test_g05_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "G05"):
            parse_command_token("M20 G90 G05 X198.6 Y0 Z165.7 A-20 B-60 C0")


class SceneAssemblyTests(unittest.TestCase):
    def test_build_scene_replaces_only_matching_mirobot_sequences(self):
        template = {
            "scene": {"objects": []},
            "robots": [
                {"name": "belt", "type": "conveyor", "base_pos": [0.25, 0.0, 0.0], "sequence": [{"action": "run"}]},
                {"name": "arm2", "type": "mirobot", "base_pos": [0.42, 0.23, 0.0], "sequence": [{"action": "placeholder"}]},
                {"name": "arm3", "type": "mirobot", "base_pos": [0.8, 0.23, 0.0], "sequence": [{"action": "placeholder"}]},
            ],
        }
        commands = {
            "arm2": "M20 G90 G00 X100 Y0 Z200 A0 B0 C0 F2000,M3 S1000",
            "arm3": "M20 G90 G01 X0 Y-160 Z185 A0 B-25 C-90 F1000",
        }

        scene = build_scene_from_commands(template, commands, ParserConfig())

        robots = {robot["name"]: robot for robot in scene["robots"]}
        self.assertEqual(robots["belt"]["sequence"], [{"action": "run"}])
        self.assertEqual(robots["arm2"]["sequence"][0]["action"], "move_cartesian")
        self.assertEqual(robots["arm2"]["sequence"][1]["action"], "idle")
        self.assertEqual(robots["arm3"]["sequence"][0]["action"], "move_linear")

    def test_relative_axes_keep_unspecified_values(self):
        config = ParserConfig()
        actions = convert_command_stream(
            "arm2",
            "M20 G90 G00 X100 Y0 Z200 A0 B0 C0 F2000,G91 G01 Y-20 Z-10",
            [0.42, 0.23, 0.0],
            config,
        )
        self.assertEqual(actions[1]["parameters"]["pose"], [0.52, 0.21, 0.19])
        self.assertEqual(actions[1]["parameters"]["euler"], [0.0, 0.0, 0.0])

    def test_unknown_robot_name_is_rejected(self):
        template = {
            "scene": {},
            "robots": [{"name": "arm2", "type": "mirobot", "base_pos": [0.42, 0.23, 0.0], "sequence": []}],
        }
        with self.assertRaisesRegex(ValueError, "arm3"):
            build_scene_from_commands(template, {"arm3": "M20 G90 G00 X0 Y0 Z0 A0 B0 C0 F2000"}, ParserConfig())


class GeneratorCliTests(unittest.TestCase):
    def test_generate_scene_file_from_template_and_commands(self):
        template = {
            "description": "template",
            "scene": {"objects": []},
            "robots": [
                {"name": "belt", "type": "conveyor", "base_pos": [0.25, 0.0, 0.0], "sequence": []},
                {"name": "arm2", "type": "mirobot", "base_pos": [0.42, 0.23, 0.0], "sequence": []},
            ],
        }
        commands = {"arm2": "M20 G90 G00 X100 Y0 Z200 A0 B0 C0 F2000,M3 S0"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            template_path = tmp_path / "template.json"
            commands_path = tmp_path / "commands.json"
            output_path = tmp_path / "generated.json"
            template_path.write_text(json.dumps(template), encoding="utf-8")
            commands_path.write_text(json.dumps(commands), encoding="utf-8")

            from mirobot_demo.generate_mirobot_scene import generate_scene_file

            summary = generate_scene_file(template_path, commands_path, output_path, ParserConfig())
            generated = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["arm2"], 2)
        self.assertEqual(generated["robots"][1]["sequence"][0]["action"], "move_cartesian")
        self.assertEqual(generated["robots"][1]["sequence"][1]["action"], "idle")


if __name__ == "__main__":
    unittest.main()
