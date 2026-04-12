# Robot Wait Synchronization Design

## Goal

Add a framework-level synchronization action that allows any robot to wait for a specific action from another robot to complete, then use that mechanism to re-orchestrate the Mirobot conveyor demo so robot motion is staged around conveyor stop-and-go windows instead of starting immediately at simulation start.

## Scope

Included:

- a new universal `wait` action available to all robot controllers
- shared execution-state tracking in the runtime so robots can observe other robots' completed actions
- validation for invalid wait targets
- updating `mirobot_demo/generated_mirobot_arm_demo.json` to use staged conveyor and robot coordination

Excluded:

- a generic message bus, event queue, or named-signal protocol
- changes to the underlying motion controllers for individual robots
- automatic scene timing optimization based on object tracking

## Existing Context

The current execution model is:

- `run_robots.py` creates one thread per robot
- each thread executes its own `sequence` in order
- each controller already supports a universal `idle` action via `BaseRobotController`
- there is no cross-robot synchronization primitive today

This means timing coordination is currently only possible by hard-coding idle durations in scene JSON, which is fragile when action durations change.

## Proposed Synchronization Model

### New action

Add a universal action:

```json
{
  "action": "wait",
  "parameters": {
    "robot": "arm2",
    "action_index": 7
  },
  "description": "Wait for arm2 to finish release handling"
}
```

Semantics:

- the current robot pauses until the target robot has completed the specified action index
- during the pause, the robot remains effectively idle
- once the target action completes, the waiting robot proceeds immediately to its next action

### Indexing

`action_index` is **0-based**.

Examples:

- `action_index: 0` means wait for the first action in the target robot sequence
- `action_index: 7` means wait for the eighth action in the target robot sequence

### Runtime state tracking

`run_robots.py` should maintain a shared execution-state object containing:

- the known action count per robot
- which action index each robot is currently executing
- which actions are already completed

Recommended representation:

- `completed_actions[(robot_name, action_index)] = True`
- optional `current_action_index[robot_name] = int`

This state must be thread-safe because robot threads run concurrently.

### Placement of logic

The synchronization logic should be split like this:

- `run_robots.py`
  - owns the shared execution-state object
  - injects it into every controller or thread
  - updates completion state before and after each action execution
- `robot_common/robot_api.py`
  - provides universal `action_wait`
  - reads the shared execution state and blocks until the target action is done

This keeps robot-specific controllers unchanged.

## Error Handling

The `wait` action must fail fast for:

- unknown target robot name
- negative action index
- action index beyond the target robot's sequence length
- missing runtime synchronization state

Errors should include the waiting robot name and the invalid target reference.

No timeout is required for the first version. If a dependency cannot ever complete because of a scene design bug, the simulation should visibly stall rather than silently skip synchronization.

## Waiting Behavior

`action_wait` should behave like synchronized idle:

- repeatedly check whether the target action has completed
- if not complete, sleep briefly and yield to the physics loop
- exit immediately when the target action becomes complete

This action does not need a `duration` parameter.

## Mirobot Demo Orchestration

After the framework-level wait mechanism exists, update `mirobot_demo/generated_mirobot_arm_demo.json` so the staged process matches the intended workflow.

### Process definition

1. conveyor starts running
2. when the box reaches `arm2`, conveyor stops
3. `arm2` begins its sequence
4. after `arm2` completes the release interaction, but before its final return-to-home move, conveyor resumes
5. when the box reaches the `arm3`/`arm4` area, conveyor stops again
6. `arm3` and `arm4` begin their sequences
7. after both complete their release interaction, conveyor resumes

### Arm sequence split point

For `arm2`, the interaction-complete boundary is defined as:

- after the `M3 S0` release-wait action completes
- before the final return `G00` action begins

This corresponds to action index `7` in the current generated `arm2` sequence:

- index `7` = `idle` for `M3 S0`
- index `8` = final return move

### Belt sequence structure

The belt should be rewritten from one long `run(duration=30)` into staged actions:

1. `run` to the arm2 pickup zone
2. `idle` while arm2 works
3. `run` to the arm3/arm4 zone
4. `idle` while arm3 and arm4 work
5. optional final `run` tail if desired for scene continuity

The exact run durations can be chosen conservatively and tuned from scene observation, but the stop/resume boundaries should be expressed through `wait` dependencies wherever sequencing matters.

### Arm startup gating

The Mirobot arms should no longer start moving immediately.

Recommended pattern:

- `arm2` begins with `wait(robot="belt", action_index=0)`
  - meaning wait for the first conveyor run segment to finish
- `arm3` begins with `wait(robot="belt", action_index=2)`
- `arm4` begins with `wait(robot="belt", action_index=2)`

This makes conveyor stop points the explicit release condition for arm motion.

### Belt resume gating

The conveyor stop windows should also be synchronized with robot completion:

- after stopping for arm2, belt resumes only after `arm2` action index `7` completes
- after stopping for arm3/arm4, belt resumes only after both:
  - `arm3` release-complete action
  - `arm4` release-complete action

For the first version, the belt can express this with sequential waits:

1. `wait(robot="arm3", action_index=<release-index>)`
2. `wait(robot="arm4", action_index=<release-index>)`
3. `run(...)`

That is sufficient and does not require a separate multi-condition wait action.

## Verification Strategy

Implementation is complete only after these checks pass:

1. add focused tests for the shared execution-state bookkeeping
2. add tests for `wait` success and invalid references
3. run the updated Mirobot scene headless
4. confirm logs show:
   - conveyor starts first
   - arm2 does not start before the first conveyor stop
   - arm3 and arm4 do not start before the second conveyor stop
   - conveyor resumes only after the intended robot interaction boundaries

## Implementation Notes

- keep the synchronization state minimal and explicit
- do not add named signal syntax yet
- do not change scene parsing format beyond adding the new `wait` action to JSON sequences
- preserve backward compatibility: scenes that do not use `wait` must continue to run unchanged
