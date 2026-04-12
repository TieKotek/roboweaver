"""Generate runnable Mirobot demo scenes from grouped G-code commands."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re


AXES = ("X", "Y", "Z", "A", "B", "C")


@dataclass
class ParserConfig:
    angular_speed_deg_per_sec: float = 90.0
    minimum_motion_duration_sec: float = 0.5
    tool_engage_wait_sec: float = 0.8
    tool_release_wait_sec: float = 0.5


@dataclass
class ParsedCommand:
    raw: str
    modal_commands: list[str] = field(default_factory=list)
    motion: str | None = None
    tool_command: str | None = None
    feed_rate_mm_per_min: float | None = None
    dwell_seconds: float | None = None
    tool_value: float | None = None
    axes: dict[str, float] = field(default_factory=dict)


@dataclass
class ParserState:
    robot_name: str
    cartesian_mode: bool = False
    absolute_mode: bool = True
    feed_rate_mm_per_min: float | None = None
    current_axes: dict[str, float] = field(
        default_factory=lambda: {"X": 0.0, "Y": 0.0, "Z": 0.0, "A": 0.0, "B": 0.0, "C": 0.0}
    )
    base_pos_m: list[float] = field(default_factory=list)


def parse_command_token(raw: str) -> ParsedCommand:
    parsed = ParsedCommand(raw=raw.strip())
    tokens = parsed.raw.split()
    axis_pattern = re.compile(r"^([XYZABC])(-?\d+(?:\.\d+)?)$")

    for token in tokens:
        if token in {"M20", "G90", "G91"}:
            parsed.modal_commands.append(token)
            continue
        if token in {"G00", "G01", "G04"}:
            parsed.motion = token
            continue
        if token == "G05":
            raise ValueError(f"Unsupported command: {parsed.raw}")
        if token == "M3":
            parsed.tool_command = token
            continue
        if token == "M21":
            raise ValueError(f"Unsupported command: {parsed.raw}")
        if token.startswith("F"):
            parsed.feed_rate_mm_per_min = float(token[1:])
            continue
        if token.startswith("P"):
            parsed.dwell_seconds = float(token[1:])
            continue
        if token.startswith("S"):
            parsed.tool_value = float(token[1:])
            continue
        match = axis_pattern.match(token)
        if match:
            parsed.axes[match.group(1)] = float(match.group(2))
            continue
        raise ValueError(f"Unsupported command token '{token}' in '{parsed.raw}'")

    return parsed


def apply_modal_updates(parsed: ParsedCommand, state: ParserState) -> None:
    for modal in parsed.modal_commands:
        if modal == "M20":
            state.cartesian_mode = True
        elif modal == "G90":
            state.absolute_mode = True
        elif modal == "G91":
            state.absolute_mode = False
    if parsed.feed_rate_mm_per_min is not None:
        state.feed_rate_mm_per_min = parsed.feed_rate_mm_per_min


def _resolve_target_axes(parsed: ParsedCommand, state: ParserState) -> dict[str, float]:
    target_axes = dict(state.current_axes)
    for axis, value in parsed.axes.items():
        if state.absolute_mode:
            target_axes[axis] = value
        else:
            target_axes[axis] += value
    return target_axes


def convert_motion_command(parsed: ParsedCommand, state: ParserState, config: ParserConfig) -> dict:
    if not state.cartesian_mode:
        raise ValueError(f"[{state.robot_name}] Motion command before M20: {parsed.raw}")
    target_axes = _resolve_target_axes(parsed, state)

    start_xyz = [state.current_axes[axis] for axis in ("X", "Y", "Z")]
    target_xyz = [target_axes[axis] for axis in ("X", "Y", "Z")]
    distance_mm = math.dist(start_xyz, target_xyz)

    feed_rate = parsed.feed_rate_mm_per_min or state.feed_rate_mm_per_min
    if feed_rate is None or feed_rate <= 0:
        raise ValueError(f"[{state.robot_name}] Motion command missing feed rate: {parsed.raw}")

    translation_time = distance_mm / feed_rate * 60.0 if distance_mm > 0 else 0.0
    orientation_delta = max(abs(target_axes[axis] - state.current_axes[axis]) for axis in ("A", "B", "C"))
    orientation_time = orientation_delta / config.angular_speed_deg_per_sec if orientation_delta > 0 else 0.0
    duration = max(translation_time, orientation_time, config.minimum_motion_duration_sec)

    pose = [
        round(state.base_pos_m[0] + target_axes["X"] / 1000.0, 6),
        round(state.base_pos_m[1] + target_axes["Y"] / 1000.0, 6),
        round(state.base_pos_m[2] + target_axes["Z"] / 1000.0, 6),
    ]
    euler = [round(target_axes["A"], 6), round(target_axes["B"], 6), round(target_axes["C"], 6)]

    state.current_axes = target_axes
    if parsed.feed_rate_mm_per_min is not None:
        state.feed_rate_mm_per_min = parsed.feed_rate_mm_per_min

    action = "move_cartesian" if parsed.motion == "G00" else "move_linear"
    return {
        "action": action,
        "parameters": {"pose": pose, "euler": euler, "duration": duration},
        "description": f"Converted {parsed.motion} command",
    }


def convert_tool_command(parsed: ParsedCommand, config: ParserConfig) -> dict:
    duration = config.tool_engage_wait_sec if (parsed.tool_value or 0.0) > 0 else config.tool_release_wait_sec
    description = (
        f"End effector engage wait ({parsed.raw})"
        if (parsed.tool_value or 0.0) > 0
        else f"End effector release wait ({parsed.raw})"
    )
    return {"action": "idle", "parameters": {"duration": duration}, "description": description}


def convert_command_stream(robot_name: str, command_stream: str, base_pos_m: list[float], config: ParserConfig) -> list[dict]:
    state = ParserState(robot_name=robot_name, base_pos_m=list(base_pos_m))
    actions: list[dict] = []

    for raw_token in [token.strip() for token in command_stream.split(",") if token.strip()]:
        parsed = parse_command_token(raw_token)
        apply_modal_updates(parsed, state)

        if parsed.motion in {"G00", "G01"}:
            actions.append(convert_motion_command(parsed, state, config))
        elif parsed.motion == "G04":
            actions.append(
                {
                    "action": "idle",
                    "parameters": {"duration": parsed.dwell_seconds},
                    "description": f"Dwell ({raw_token})",
                }
            )
        elif parsed.tool_command == "M3":
            actions.append(convert_tool_command(parsed, config))

    return actions


def build_scene_from_commands(template_scene: dict, commands_by_robot: dict[str, str], config: ParserConfig) -> dict:
    output_scene = copy.deepcopy(template_scene)
    robots_by_name = {robot["name"]: robot for robot in output_scene["robots"]}

    for robot_name, command_stream in commands_by_robot.items():
        if robot_name not in robots_by_name:
            raise ValueError(f"Robot '{robot_name}' was not found in template scene.")
        robot = robots_by_name[robot_name]
        if robot.get("type") != "mirobot":
            raise ValueError(f"Robot '{robot_name}' is not a mirobot template entry.")
        robot["sequence"] = convert_command_stream(robot_name, command_stream, robot["base_pos"], config)

    return output_scene


def generate_scene_file(template_path: Path, commands_path: Path, output_path: Path, config: ParserConfig) -> dict[str, int]:
    template_scene = json.loads(template_path.read_text(encoding="utf-8"))
    commands_by_robot = json.loads(commands_path.read_text(encoding="utf-8"))
    output_scene = build_scene_from_commands(template_scene, commands_by_robot, config)
    output_path.write_text(json.dumps(output_scene, indent=2), encoding="utf-8")
    return {
        robot["name"]: len(robot.get("sequence", []))
        for robot in output_scene.get("robots", [])
        if robot.get("type") == "mirobot"
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--commands", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = generate_scene_file(Path(args.template), Path(args.commands), Path(args.output), ParserConfig())
    for robot_name, count in summary.items():
        print(f"{robot_name}: generated {count} actions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
