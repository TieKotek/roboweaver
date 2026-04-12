# Mirobot G-code Parser Design

## Goal

Add a reusable offline parser under `mirobot_demo/` that converts a WLKATA Mirobot G-code JSON file such as `cmd.json` into a new runnable scenario JSON for this project.

The generated scenario must:

- keep the scene layout, conveyor settings, object setup, and robot placements from `mirobot_demo/mirobot_arm_demo.json`
- replace the placeholder sequences for `arm2`, `arm3`, and `arm4` with parsed commands from the input G-code JSON
- produce action sequences that the existing `mirobot` controller can execute
- estimate action durations to stay as close as practical to real Mirobot execution timing

## Scope

This feature covers only the current Mirobot demo pipeline inside `mirobot_demo/`.

Included:

- parsing grouped command strings from a JSON file keyed by robot name
- converting supported G-code commands into framework actions
- transforming local millimeter coordinates into world-meter poses
- preserving end-effector command semantics in descriptions while degrading unsupported tool actions to `idle`
- writing a new runnable scene JSON file

Excluded:

- runtime integration into `run_robots.py`
- support for general G-code files outside the current JSON container format
- support for angle mode (`M21`)
- support for `G05`, arcs, or additional M-code families
- adding a physical suction or gripper model to the Mirobot robot

## Existing Context

The current template scene is `mirobot_demo/mirobot_arm_demo.json`. It already defines:

- the conveyor robot and its placement
- the cargo object
- the three Mirobot robots `arm2`, `arm3`, and `arm4`
- the desired base positions to keep unchanged

The current command source is `mirobot_demo/cmd.json`, which stores one comma-delimited G-code command string per robot.

The existing Mirobot controller supports these useful actions:

- `move_cartesian`
- `move_linear`
- `move_joints`
- `home`

The parser should therefore emit `move_cartesian`, `move_linear`, and `idle`, but not new controller actions.

## Command Semantics

### Supported commands

- `M20`
  - enter Cartesian mode
  - required before any Cartesian motion command
- `G90`
  - absolute coordinate mode
- `G91`
  - relative coordinate mode
- `F<number>`
  - update the active feed rate in `mm/min`
- `G00`
  - rapid point-to-point Cartesian motion
  - maps to framework action `move_cartesian`
- `G01`
  - linear interpolation motion
  - maps to framework action `move_linear`
- `G04 P<number>`
  - dwell
  - maps to framework action `idle`
- `M3 S<number>`
  - end-effector PWM command
  - maps to framework action `idle`
  - description preserves semantic meaning such as suction-on wait or suction-off wait

### Unsupported commands

- `G05`
  - parser must fail with a clear error
- `M21`
  - parser must fail with a clear error
- any other unknown command
  - parser must fail with a clear error

Failing hard is preferred over silently generating a wrong scene.

## Coordinate And Orientation Rules

### Position units

The G-code uses millimeters for `X`, `Y`, and `Z`.

The generated scene JSON must use meters.

Conversion rule:

- `meters = millimeters / 1000.0`

### Local vs world frame

The G-code positions for each arm are expressed in that arm's local frame.

The framework scene expects world-frame poses.

Conversion rule for each parsed motion target:

1. parse local `X/Y/Z` in millimeters
2. convert to meters
3. add the target robot's `base_pos` from the template scene
4. write the resulting world-frame `pose`

This design assumes the local Mirobot axes align with the scene axes used by the current demo layout. No extra rotation is applied at the robot base.

### Partial axis updates

For motion commands, omitted axes keep their previous value.

This applies to all six Cartesian fields:

- `X`
- `Y`
- `Z`
- `A`
- `B`
- `C`

For relative mode (`G91`), only the provided axes are incremented; omitted axes remain unchanged.

### Orientation mapping

The design will treat G-code `A/B/C` as roll/pitch/yaw values mapped directly to the framework `euler` field:

- `A -> euler[0]`
- `B -> euler[1]`
- `C -> euler[2]`

This aligns with the current controller implementation, which reads and writes Euler angles with `xyz` ordering.

The design assumes the current understanding is correct for the demo inputs:

- `[0, 0, 90]` keeps the tool facing downward and rotates around the local/world `z` axis by positive 90 degrees

If later inputs show an orientation mismatch, the parser can be extended with configurable sign or order mapping. That configurability is not required for the first version.

## Duration Model

Accurate-enough timing is a primary feature, not an afterthought.

Each generated motion action must include a `duration` computed from the G-code command and the state before that command.

### Feed rate

The active feed rate comes from the most recent `F<number>` value, whether supplied as a standalone command or inside a motion command token.

Feed rate unit:

- `mm/min`

If a motion command is encountered before any feed rate has been established, the parser must fail with a clear error.

### Motion duration

For `G00` and `G01`, compute:

1. translation distance in millimeters between the previous local Cartesian position and the target local Cartesian position
2. translation time as `distance_mm / feed_mm_per_min * 60`
3. orientation delta as the largest absolute change across `A/B/C`
4. orientation time as `orientation_delta_deg / angular_speed_deg_per_sec`
5. final duration as the maximum of:
   - translation time
   - orientation time
   - a configurable minimum motion duration

Initial constants:

- `angular_speed_deg_per_sec = 90`
- `minimum_motion_duration_sec = 0.2`

These constants should live in one obvious configuration block near the top of the script so they can be tuned later.

### Dwell duration

`G04 P<number>` maps directly to:

- `action = "idle"`
- `parameters.duration = P`

### End-effector command duration

`M3 S<number>` maps to `idle`, but descriptions should distinguish likely suction-on versus suction-off intent.

Initial heuristic:

- `S > 0`: description like `End effector engage wait (M3 S1000)`
- `S == 0`: description like `End effector release wait (M3 S0)`

Because the scene does not model the real tool, duration should come from fixed configuration constants rather than the PWM value itself.

Initial constants:

- `tool_engage_wait_sec = 0.8`
- `tool_release_wait_sec = 0.5`

## Output Format

The generated scene JSON should remain structurally compatible with current examples.

Each generated robot action should look like existing example actions:

- `action`
- `parameters`
- `description`

Mapping details:

- `G00`
  - `action = "move_cartesian"`
  - `parameters.pose = [world_x, world_y, world_z]`
  - `parameters.euler = [A, B, C]`
  - `parameters.duration = computed_duration`
- `G01`
  - `action = "move_linear"`
  - `parameters.pose = [world_x, world_y, world_z]`
  - `parameters.euler = [A, B, C]`
  - `parameters.duration = computed_duration`
- `G04`
  - `action = "idle"`
  - `parameters.duration = P`
- `M3`
  - `action = "idle"`
  - `parameters.duration = configured_wait`

The output scene should preserve all non-sequence content from the template unless the script is explicitly told otherwise.

## Script Interface

The parser should be a reusable command-line script placed under `mirobot_demo/`, for example `mirobot_demo/generate_mirobot_scene.py`.

Suggested interface:

```powershell
python mirobot_demo\generate_mirobot_scene.py `
  --template mirobot_demo\mirobot_arm_demo.json `
  --commands mirobot_demo\cmd.json `
  --output mirobot_demo\generated_mirobot_arm_demo.json
```

Expected behavior:

- read template and command input
- replace sequences for matching Mirobot robots
- validate the conversion
- write the new scene JSON to the requested path
- print a short summary of how many commands were converted per robot

## Error Handling

The parser must fail fast with actionable messages for:

- missing input files
- invalid JSON structure
- robot names in `cmd.json` not found in the template scene
- motion commands before `M20`
- motion commands before feed rate is known
- unsupported commands such as `G05`
- malformed numeric fields
- missing initial state needed for relative motion

Error messages should identify the robot name and the original command text whenever possible.

## Verification Strategy

Implementation is complete only after these checks pass:

1. run the generator against the current `mirobot_demo/cmd.json`
2. inspect the generated JSON for all three robots
3. verify the output scene keeps the same conveyor and robot placements as the template
4. run the generated scene with `python run_robots.py <generated-scene> --headless`
5. confirm there are no JSON schema issues, action-dispatch failures, or unsupported-action errors

If headless execution exposes unreachable poses or IK failures, that is a data or mapping issue that must be surfaced explicitly instead of ignored.

## Implementation Notes

Keep the implementation scoped and local to `mirobot_demo/`.

Recommended structure:

- one small parser state object for each robot stream
- one tokenizer/parser for individual command strings
- one conversion function that maps parsed commands to scene actions
- one scene assembly function that copies the template and injects sequences

Do not modify `run_robots.py` for this feature unless a real integration gap is discovered during verification.
