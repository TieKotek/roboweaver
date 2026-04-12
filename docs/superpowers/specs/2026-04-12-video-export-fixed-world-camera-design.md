# Video Export Fixed World Camera Design

## Goal

Replace the current export camera initialization with a fixed, stable world-space camera that is independent of the interactive viewer.

The export camera must:

- look toward the world origin
- face the positive `Y` direction
- view the scene from a `45` degree downward angle
- support `--camera-zoom-out` as a distance multiplier

## Scope

Included:

- remove the current attempt to approximate the viewer's default free camera
- define one deterministic export camera in world space
- keep `--camera-zoom-out` as the only user-facing framing control
- update tests for the new fixed-camera behavior

Excluded:

- matching the interactive viewer camera exactly
- named-camera support
- configurable look-at targets
- runtime camera dragging or interactive control

## Existing Context

The current export path uses a free camera initialized from MuJoCo defaults and then scales its distance.

That still does not reliably match the interactive viewer, and exact matching is not possible without actually driving the viewer camera.

The clarified requirement is simpler and better:

- use a known fixed camera
- point it at the origin
- make it globally readable
- let `--camera-zoom-out` control how far back it sits

## Recommended Approach

Build the export camera explicitly rather than deriving it from default viewer state.

Recommended camera parameters:

- `lookat = [0.0, 0.0, 0.0]`
- `azimuth = 90.0`
- `elevation = -45.0`
- `distance = base_distance * camera_zoom_out`

Where `base_distance` is a fixed baseline distance chosen for a sensible global framing. The exact value can be tuned once from a real export check.

## Camera Model

Recommended implementation:

1. create a `mujoco.MjvCamera()`
2. initialize it with `mujoco.mjv_defaultCamera(...)`
3. set:
   - `camera.lookat[:] = (0.0, 0.0, 0.0)`
   - `camera.azimuth = 90.0`
   - `camera.elevation = -45.0`
   - `camera.distance = base_distance * zoom_out`
4. pass this camera object to `renderer.update_scene(...)`

This makes the export framing deterministic and easy to reason about.

## CLI Behavior

Keep:

- `--camera-zoom-out`

Semantics:

- `1.0` means baseline fixed-camera distance
- values greater than `1.0` pull farther back
- values less than `1.0` move closer while staying positive

## Verification Strategy

Implementation is complete only after these checks pass:

1. add a focused unit test covering the fixed camera configuration helper
2. run the export-specific test file
3. run a real export command
4. confirm the export still works without a window

## Implementation Notes

- keep the camera helper local to `run_robots.py`
- do not change the recently-fixed runtime pacing behavior
- keep dimension normalization unchanged
