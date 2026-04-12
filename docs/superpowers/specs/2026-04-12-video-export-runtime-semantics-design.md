# Video Export Runtime Semantics Design

## Goal

Fix the current `--save-video` behavior so enabling video export does not change the simulation's runtime semantics.

The export path must satisfy both of these requirements at the same time:

- the simulation should execute with the same paced main-loop behavior as normal non-export runs
- the final MP4 should still be sampled from simulation time so video playback length aligns with simulation time

## Scope

Included:

- keep real-time pacing active when `--save-video` is enabled
- continue sampling video frames from simulation time
- preserve the no-export performance path as much as possible
- verify that export no longer uses the "fast as possible" simulation loop

Excluded:

- changing controller internals
- two-pass trajectory recording and offline replay
- new video formats or encoding strategies

## Existing Context

The current implementation disables the main-loop sleep whenever `--save-video` is enabled.

That makes export mode run the physics loop as fast as possible. In this project, that changes the behavior of robot controllers enough to cause visible divergence from ordinary runs, including unstable drone behavior and scenes completing far faster than expected in wall-clock time.

This is not acceptable because `--save-video` should be an observation feature, not a behavior-changing execution mode.

## Recommended Approach

Use the normal paced main loop for both ordinary execution and export execution.

Behavior:

- keep the existing `mujoco.mj_step(...)`
- keep the real-time sleep logic active even when exporting
- if export is enabled, additionally check whether simulation time has crossed the next frame boundary and render frames when needed

This preserves controller/runtime behavior while keeping the output video aligned to simulation time.

## Main Loop Model

Recommended loop structure:

1. advance one physics step
2. if exporting, render any frame whose target simulation timestamp has been reached
3. if viewer is active, sync it as usual
4. if all robot threads are finished, exit
5. sleep for the remaining timestep budget regardless of whether export is enabled

The key change is:

- export should add work to the existing loop, not replace the loop's timing semantics

## Performance Semantics

Required behavior:

- `--save-video` may make wall-clock execution slower because rendering and encoding cost time
- `--save-video` must not make simulation execute in a qualitatively different timing mode
- when `--save-video` is not enabled, the code path should remain as close as possible to the current non-export path

## Verification Strategy

Implementation is complete only after these checks pass:

1. add a focused regression test that covers the paced-loop policy decision
2. run the export-specific test file
3. run a real export command and confirm it still produces a valid MP4
4. confirm the runtime output no longer reflects the previous "fast as possible" execution mode

## Implementation Notes

- keep this fix local to the main loop in `run_robots.py`
- do not bundle unrelated refactors
- preserve the newer default-camera and dimension-normalization behavior
