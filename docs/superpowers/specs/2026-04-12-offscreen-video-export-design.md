# Offscreen Video Export Design

## Goal

Add a built-in offscreen video export path to `run_robots.py` that renders simulation frames directly to an MP4 file.

The exported video's time axis must be driven by simulation time rather than wall-clock time:

- `1.0` second of simulation time must equal `1.0` second of final video playback
- export may run faster or slower than real time
- viewer refresh rate and machine performance must not change the final MP4 duration

## Scope

Included:

- MP4 export directly from `run_robots.py`
- offscreen rendering without opening the MuJoCo interactive viewer
- deterministic frame sampling based on simulation time and requested video FPS
- command-line options for output path, resolution, FPS, and named camera selection
- validation for missing or invalid camera names in export mode

Excluded:

- recording the existing interactive viewer window
- auto-follow, mouse-controlled, or free-camera capture
- audio export
- post-processing effects, overlays, or timestamp burn-in
- scene-format changes for camera definitions

## Existing Context

The current runtime has two display modes:

- normal mode launches `mujoco.viewer.launch_passive(model, data)`
- `--headless` mode runs physics without a viewer

The main loop currently advances physics with `mj_step`, optionally syncs the viewer, and sleeps to roughly match real time. This is appropriate for interactive viewing, but it means observed playback is tied to runtime performance and viewer sync cadence.

There is currently no frame capture, offscreen renderer, or video encoding pipeline in the repository.

## User-Facing Interface

Add command-line options to `run_robots.py`:

- `--save-video <path>`
  - enable offscreen export and write an MP4 file to the given path
- `--video-fps <float>`
  - default `30.0`
- `--width <int>`
  - default `1280`
- `--height <int>`
  - default `720`
- `--camera <name>`
  - required when `--save-video` is used

Behavior rules:

- normal runs without `--save-video` keep current behavior
- `--headless` may still be used independently
- when `--save-video` is enabled, the program should not launch the interactive viewer even if `--headless` is not passed
- if the requested camera name is missing, the program must fail with a clear error and list available named cameras

## Time Semantics

This feature is centered on one strict rule:

- final video time is derived from simulation time only

Recommended model:

- frame `0` corresponds to simulation time `0.0`
- frame `n` corresponds to target simulation time `n / video_fps`
- the runtime should keep stepping physics until `data.time >= next_frame_time`, then render exactly one frame

Implications:

- export speed is unconstrained and may be non-real-time
- there is no `sleep` in export mode
- viewer sync cadence does not exist in export mode
- a slow machine may take longer to generate the file, but the resulting MP4 duration remains tied to simulation duration

This is intentionally different from the current interactive loop, where wall-clock pacing is part of the behavior.

## Rendering Model

Use MuJoCo offscreen rendering rather than screen capture.

Recommended implementation:

- construct an offscreen renderer after model/data initialization
- bind rendering to the named camera requested via `--camera`
- render RGB frames at the requested `width` and `height`
- pass frames directly to a video writer during simulation

The export path should be isolated from the interactive viewer path so the logic is easy to reason about:

- interactive mode: physics + optional viewer sync + real-time sleep
- export mode: physics + simulation-time frame sampling + no real-time sleep

## Video Writing

Write frames directly to MP4 during the run.

Preferred behavior:

- initialize a video writer once before the main loop
- append each rendered frame immediately as it is produced
- close the writer cleanly in `finally`

The implementation may use an already-available Python video library if present in project dependencies. If a new dependency is required, it should be added explicitly and used only for the MP4 writing step.

## Camera Resolution Strategy

Because most scenes in this repository rely on viewer orientation defaults rather than reusable named scene cameras, export mode must use explicit named cameras for reproducibility.

Rules:

- `--camera` is mandatory in export mode
- only named MuJoCo cameras are supported
- missing camera name is a hard error
- no fallback to free camera or viewer state

This keeps exported videos scriptable and repeatable across machines.

## Main Loop Behavior

The runtime should split into two loop styles.

### Interactive loop

Keep the current behavior:

- `mj_step`
- optional `viewer.sync()`
- `time.sleep(...)` to roughly honor `model.opt.timestep`

### Export loop

Use simulation-time-based scheduling:

1. initialize `next_frame_time = 0.0`
2. render an initial frame before or at the first simulation step if needed to capture time zero
3. continue stepping physics
4. whenever `data.time >= next_frame_time`, render one frame and increment `next_frame_time += 1.0 / video_fps`
5. continue until robot threads finish
6. if desired, emit a final frame near the terminal simulation time so the last state is represented

The export loop must not use wall-clock sleeps.

## Error Handling

Export mode must fail fast for:

- missing `--camera`
- invalid camera name
- non-positive `--video-fps`
- non-positive output width or height
- inability to initialize the renderer
- inability to initialize the MP4 writer

Errors should identify the invalid parameter value whenever possible.

## Verification Strategy

Implementation is complete only after these checks pass:

1. add focused tests for export-argument validation
2. add focused tests for simulation-time frame scheduling logic
3. run a real scenario in export mode with a known named camera
4. confirm an MP4 file is produced
5. confirm the exported frame count and duration match the expected simulation-time-based schedule

The most important verification point is not whether the export looks correct in motion, but whether the resulting video duration is decoupled from wall-clock runtime.

## Implementation Notes

- keep the scheduling logic in a small helper so it can be tested without needing MuJoCo rendering in unit tests
- keep interactive mode backward compatible
- keep export mode explicit rather than trying to reuse viewer timing code
- prefer clear validation and failure messages over permissive fallback behavior
