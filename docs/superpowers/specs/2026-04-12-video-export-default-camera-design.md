# Video Export Default Camera Design

## Goal

Replace the explicit named-camera export workflow with a simpler default-camera export workflow that matches the normal viewer's initial free-camera view, then pulls that view back by a configurable fixed multiplier so exported videos show more of the overall scene.

## Scope

Included:

- remove the requirement to select a named camera for video export
- use the default MuJoCo free camera as the export camera baseline
- add a fixed zoom-out multiplier for export camera distance
- validate the multiplier as a positive number
- update tests for the new export-camera behavior

Excluded:

- named-camera selection during export
- scene-dependent auto-framing
- interactive camera control during export
- changes to the simulation-time-based frame scheduling

## Existing Context

The current export implementation requires `--camera <name>` and resolves that name to a MuJoCo named camera.

That is technically functional, but it is not the user interaction we want. The desired behavior is:

- export should not require users to know model-specific camera names
- export should start from the same overall viewpoint users see when they normally launch the simulation viewer
- export should be slightly farther away than that default viewpoint so more of the scene remains visible

## Recommended Approach

Use a MuJoCo free camera for export rather than a named camera.

Behavior:

- remove `--camera`
- add `--camera-zoom-out <float>`
- default `--camera-zoom-out` to `1.5`
- treat `1.0` as "use the normal initial viewer distance"
- treat values greater than `1.0` as "pull the camera back"

The export camera should preserve the default viewer orientation and look-at target, changing only the distance.

## Camera Model

Recommended implementation:

1. create a `mujoco.MjvCamera()`
2. initialize it with the default free-camera settings for the loaded model
3. multiply `camera.distance` by `args.camera_zoom_out`
4. pass that camera object to `renderer.update_scene(...)`

This keeps exported framing close to the normal viewer initialization while making the shot more global.

## CLI Changes

Update the export CLI to:

- remove `--camera`
- add `--camera-zoom-out`

Validation rules:

- `--camera-zoom-out` must be strictly positive
- export mode no longer requires a named camera

## Error Handling

Export mode must fail fast for:

- non-positive `--camera-zoom-out`
- non-positive `--video-fps`
- non-positive output width or height
- inability to initialize the renderer or writer

## Verification Strategy

Implementation is complete only after these checks pass:

1. add a focused unit test showing export validation no longer requires `camera`
2. add a focused unit test showing non-positive `camera_zoom_out` is rejected
3. run the export-specific test file
4. run a real export command without `--camera`
5. confirm the command succeeds and still produces an MP4

## Implementation Notes

- keep the camera logic small and local to `run_robots.py`
- do not keep dead named-camera export code
- preserve the dimension-normalization behavior that was just added
