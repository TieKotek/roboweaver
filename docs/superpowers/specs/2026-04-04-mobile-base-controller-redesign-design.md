# Mobile Base Controller Redesign Design

## Summary

This design covers a focused refactor of the `rbtheron` and `tracer` mobile base controllers. The goal is to replace the current ad hoc wheel-speed logic with a deterministic trajectory-driven motion layer that matches the user-facing behavior of `skydio`: explicit speed parameters, predictable simulated duration, and logs that match the planned trajectory time.

The redesign also includes targeted MJCF cleanup for both platforms because the current controller instability is partly caused by model-side contact and inertial issues. The implementation must keep the exposed JSON actions backward-compatible while adding optional speed control.

## Problem Statement

The current `rbtheron` and `tracer` controllers are nearly duplicated and rely on a mixed strategy:

- fixed wheel-speed commands
- direct heading correction during straight motion
- hand-written deceleration logic near the end of turns
- passive waiting for the base to stop after command completion

That structure has three problems:

1. It is harder to reason about than the motion it controls.
2. Action duration is not planned explicitly from the requested motion profile.
3. The controller is compensating for model-side contact and inertia issues instead of operating on a stable physical base.

Observed behavior in the current demos confirms this:

- `rbtheron` accumulates noticeable heading and lateral error, especially after multiple actions.
- `tracer` performs better but still drifts during turns and does not provide clean, deterministic timing semantics.

## Goals

- Support explicit motion speed parameters for both mobile bases.
- Compute and print a planned motion duration before each move, following a deterministic trapezoidal or triangular velocity profile.
- Make logged simulation duration align with the planned duration.
- Improve straight-line and turn accuracy without retaining the current patchwork control logic.
- Share the majority of logic between `rbtheron` and `tracer`.
- Keep existing JSON actions working.
- Simplify MJCF contact and inertia behavior enough that wheel-ground traction is predictable.

## Non-Goals

- No attempt to build a generic controller for every robot type in the repository.
- No UI or viewer-side tooling changes.
- No broad refactor of unrelated robot controllers.
- No requirement to remove all feedback internally. The user-facing requirement is deterministic motion behavior, not pure open-loop control.

## Constraints

- Existing action names used by scenarios must continue to work.
- The design should remain understandable and maintainable by future contributors.
- Model changes should stay local to each robot's control directory.
- Runtime behavior must remain compatible with the current action dispatch system in `common/robot_api.py`.

## Existing Code Context

- `robots/rbtheron_control/rbtheron_controller.py` and `robots/tracer_control/tracer_controller.py` each implement their own copy of differential-drive motion behavior.
- `robots/skydio_control/skydio_controller.py` already demonstrates the desired user-facing motion semantics: explicit `speed`, deterministic trajectory timing, and logs that describe expected trajectory duration.
- `robots/rbtheron_control/rbtheron/rbtheron.xml` is physically suspect: the main body lacks a clear simplified collision/inertia representation, while support wheels and joints can influence dynamics in ways that make the controller compensate for the model.
- `robots/tracer_control/agilex_tracer2/tracer2.xml` is cleaner but still depends on coarse converted geometry and contact approximations.

## Proposed Architecture

### 1. Shared Differential Drive Base Class

Create a shared differential-drive controller base class in `common/` to hold behavior currently duplicated across `rbtheron` and `tracer`.

Responsibilities:

- resolve body and actuator ids
- read pose and base velocities
- convert body-frame linear/angular velocity targets into left/right wheel targets
- generate trapezoidal or triangular motion profiles for translation and rotation
- advance control deterministically using simulation time
- print planned timing information
- provide a small amount of tracking feedback to reduce accumulated drift

This base class becomes the single implementation of differential-drive action behavior. Robot-specific controllers become thin configuration layers.

### 2. Thin Robot-Specific Controllers

`RbtheronController` and `TracerController` remain as public controller classes registered in `run_robots.py`, but they should mostly define:

- actuator names
- base body name
- left/right command sign convention
- wheel radius
- wheel track width
- default/max linear speed
- default/max angular speed
- linear/angular acceleration limits
- optional tracking gains tuned per robot

They should not contain duplicated trajectory execution logic.

### 3. Deterministic Motion Layer

For each move, the controller computes a complete motion profile before execution:

- straight translation:
  - target distance
  - requested linear speed
  - acceleration limit
  - resulting triangular or trapezoidal profile
  - total planned duration
- pure rotation:
  - target angle
  - requested angular speed
  - angular acceleration limit
  - resulting triangular or trapezoidal profile
  - total planned duration

This planned duration is printed at action start, similar to `skydio`.

Execution then follows the profile in simulation time, rather than commanding a fixed wheel speed until a raw distance threshold happens to be crossed.

### 4. Lightweight Tracking Layer

The design deliberately avoids the current heavy-handed patch style, but it does not force pure open-loop control.

The controller may apply limited feedback for:

- heading error during straight motion
- residual angle error during rotation
- final settle behavior at the end of the commanded trajectory

This tracking must remain simple and subordinate to the planned profile:

- the profile defines duration and nominal speed
- feedback only corrects small execution error
- no hand-written end-stage wheel ramp logic
- no large direct wheel asymmetry terms that dominate the command

### 5. MJCF Stabilization

Both mobile bases require MJCF cleanup so the new controller is not fighting unstable contact dynamics.

#### `rbtheron`

Expected changes:

- add or revise a simple chassis collision representation
- ensure the chassis has coherent mass and inertia
- verify drive wheel radius and contact geometry
- tune wheel-ground friction to avoid frequent slip
- reduce support wheel interference with differential drive motion
- revisit actuator `kv` and `ctrlrange`

Because `rbtheron` currently shows the larger tracking error, this model is the higher-risk part of the work.

#### `tracer`

Expected changes:

- verify wheel collision geometry and friction
- verify chassis collision participation and mass distribution
- revisit actuator `kv` and `ctrlrange`
- reduce positional drift during in-place turns

`tracer` should need fewer structural changes than `rbtheron`, but the same control architecture should apply.

## Public Action Interface

Backward compatibility is required.

### Existing actions to preserve

- `move_straight(distance, direction="forward")`
- `turn(target_yaw, direction="auto")`

### New optional parameters

- `move_straight(distance, direction="forward", speed=<m/s>)`
- `turn(target_yaw, direction="auto", speed=<deg/s>)`

If `speed` is omitted, the controller uses a robot-specific default.

### Optional future additions

The implementation may also expose cleaner internal helpers or new public actions such as:

- `move_distance(...)`
- `rotate(...)`

but examples and existing scenarios must continue to work without requiring these.

## Motion Semantics

### Straight Motion

For `move_straight`:

- the controller captures the start pose
- builds a body-forward or body-backward motion profile
- derives target body linear velocity over time
- converts that to wheel targets
- uses minimal heading tracking to hold the line
- stops at the planned end of trajectory

The primary contract is not “constant wheel speed.” The contract is “execute a planned distance at a requested speed profile with deterministic timing.”

### Rotation

For `turn`:

- the controller resolves the final yaw target exactly as today, including `direction`
- computes the shortest or forced-direction angular displacement
- builds a rotational motion profile
- drives a pure rotation command with limited angular tracking correction
- stops at the planned end of the trajectory

This replaces the current pattern of accumulating angle and manually tapering wheel speed near the end.

## Logging and Timing

Each motion action should print:

- motion type
- requested distance or angle
- requested speed
- planned trajectory duration

The existing `BaseRobotController.execute_action()` timing log should then report a simulation duration close to the planned duration.

Acceptance is based on simulation time, not wall-clock time.

## Validation Plan

Validation must include at least:

- `examples/rbtheron_mobile_base_demo.json`
- `examples/tracer_mobile_base_demo.json`

Those examples should be updated to exercise explicit speed parameters.

Additional validation scenarios should cover:

- straight 1 m forward at multiple speeds
- straight 1 m backward at multiple speeds
- 90-degree turn at multiple angular speeds
- repeated move-turn-move sequences to detect accumulated drift

For each run, capture:

- planned duration printed by the controller
- logged simulation duration
- final pose error after the action sequence

## Risks

### Risk 1: Model instability hides controller quality

If wheel slip and support-wheel interference remain too high, even a good trajectory controller will look inaccurate.

Mitigation:

- stage model tuning before final controller tuning
- prioritize stable simplified contact over visual fidelity

### Risk 2: Over-correction reintroduces complexity

If tracking gains are used aggressively, the new controller could collapse back into a patchy closed-loop design.

Mitigation:

- keep the profile generator authoritative
- limit feedback to narrow error correction
- review final code for duplicated heuristics and threshold logic

### Risk 3: Timing looks correct while geometry is still poor

A controller can hit the planned time yet still miss the target pose due to slip.

Mitigation:

- validate both duration and final pose error
- include repeated sequences, not just isolated commands

## Implementation Outline

1. Introduce a shared differential-drive base controller in `common/`.
2. Port `rbtheron` and `tracer` to the new base class with robot-specific parameters only.
3. Add deterministic straight and rotation trajectory generation with explicit speed parameters.
4. Remove the current direct wheel correction and manual deceleration logic.
5. Clean up `rbtheron` MJCF for stable chassis/wheel contact behavior.
6. Tune `tracer` MJCF parameters for consistent turning and traction.
7. Update demos to include speed parameters and serve as validation cases.
8. Run headless validation and compare planned vs logged simulation duration and final pose accuracy.

## Success Criteria

The redesign is successful if:

- both mobile bases accept explicit `speed` for translation and rotation
- the controller prints planned duration before executing the action
- log duration in simulation time aligns with that planned duration
- `rbtheron` no longer shows obvious route drift in the baseline demo
- `tracer` shows reduced in-place turn drift
- the final controller code is meaningfully simpler than the current duplicated logic
