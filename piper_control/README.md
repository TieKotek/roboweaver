# PiPER Robot Arm JSON Configuration Format

This document describes the JSON configuration format used by `run_piper.py` to control the AgileX PiPER robot arm.

## Basic Structure

```json
{
  "description": "Description of the sequence",
  "scene": {
    // Custom scene definition (optional)
  },
  "sequence": [...]
}
```

## Configuration Fields

### Top Level Fields

- **description** (string, optional): Human-readable description of the sequence
- **scene** (object, optional): Definition of the simulation scene (robot base, objects, global physics)
- **sequence** (array, required): Main sequence of actions to execute

## Scene Configuration

The `scene` object allows you to customize the simulation environment, including physics parameters crucial for stable grasping.

```json
"scene": {
  "robot_base": [0, 0, 0],
  "friction": [2.0, 0.005, 0.0001],
  "solimp": [0.95, 0.99, 0.001],
  "solref": [0.004, 1],
  "objects": [
    {
      "name": "target_block",
      "type": "box",
      "size": [0.02, 0.02, 0.02],
      "pos": [0.4, 0, 0.1],
      "rgba": [0, 1, 0, 1],
      "movable": true,
      "friction": [5.0, 0.1, 0.001]
    }
  ]
}
```

### Scene Fields

- **robot_base** (array, optional): [x, y, z] position of the robot base. Default is [0, 0, 0].
- **objects** (array, optional): List of objects to add to the scene.

#### Global Physics Parameters
These parameters apply as defaults to all objects in the scene unless overridden by a specific object. They are tuned to improve grasp stability in simulation.

- **friction** (array, optional): **[sliding, torsional, rolling]** friction coefficients.
  - *Sliding*: Resistance to linear movement (stickiness). Higher values (e.g., 2.0-5.0) help prevent objects from slipping out of the gripper.
  - *Torsional*: Resistance to rotation around the contact normal. Prevents objects from spinning in the grasp.
  - *Rolling*: Resistance to rolling.
  - *Default if unspecified*: `[2.0, 0.005, 0.0001]`

- **solimp** (array, optional): **[dmin, dmax, width, mid, power]** - Solver Impedance.
  - Controls the stiffness/softness of the contact. Values close to 1 (e.g., `0.95`, `0.99`) make contacts harder and more precise, preventing the gripper from "sinking" into the object.
  - *Default if unspecified*: `[0.95, 0.99, 0.001]`

- **solref** (array, optional): **[timeconst, dampratio]** - Solver Reference.
  - *Timeconst*: Time constant for constraint correction. Smaller values (e.g., `0.004`s) mean faster, harder contact correction.
  - *Dampratio*: Damping ratio. `1` indicates critical damping (no oscillation/bouncing upon contact).
  - *Default if unspecified*: `[0.004, 1]`

### Object Definition

Each object in the `objects` list can have the following fields:

- **name** (string, optional): Unique name for the object.
- **type** (string, optional): Geometry type. Common values: "box", "sphere", "cylinder", "capsule". Default is "box".
- **size** (array or number, required): Size of the object.
  - For "box": [x_half_size, y_half_size, z_half_size]
  - For "sphere": radius
  - For "cylinder"/"capsule": [radius, half_length]
- **pos** (array, optional): [x, y, z] position. Default is [0.5, 0, 0.1].
- **quat** (array, optional): [w, x, y, z] quaternion orientation.
- **euler** (array, optional): [roll, pitch, yaw] Euler angle orientation.
- **rgba** (array, optional): [r, g, b, a] color. Default is red [1, 0, 0, 1].
- **movable** (boolean, optional): If true, the object will be a free joint (physics enabled). If false, it is static. Default is false.
- **mass** (number, optional): Mass of the object. For small grasping targets, lower mass (e.g., 0.01) improves stability.
- **friction** (array, optional): Override global friction for this specific object.
- **solimp** (array, optional): Override global solver impedance for this object.
- **solref** (array, optional): Override global solver reference for this object.

## Action Structure

Each action in the sequences has the following structure:

```json
{
  "action": "action_name",
  "description": "Human-readable description",
  "parameters": {
    // Action-specific parameters
  }
}
```

### Common Action Fields

- **action** (string, required): Name of the action to execute
- **description** (string, optional): Human-readable description
- **parameters** (object, optional): Action-specific parameters

## Available Actions

### 1. `move_joints`
Move to specific joint positions.

```json
{
  "action": "move_joints",
  "description": "Move to specific joint configuration",
  "parameters": {
    "joints": [0, 1.57, -1.3485, 0, 0, 0],  // Joint angles in radians
    "duration": 5.0  // Optional: movement duration in seconds
  }
}
```

**Parameters:**
- **joints** (array): Array of 6 joint angles in radians [joint1, joint2, joint3, joint4, joint5, joint6]
- **duration** (number, optional): Movement duration in seconds. If not specified, moves as fast as possible

### 2. `move_cartesian`
Move to a Cartesian pose using inverse kinematics.

```json
{
  "action": "move_cartesian",
  "description": "Move to Cartesian position",
  "parameters": {
    "pose": [0.3, 0.2, 0.15],  // [x, y, z] or [x, y, z, roll, pitch, yaw]
    "duration": 4.0,  // Optional: movement duration in seconds
    "orientation_needed": false  // Whether to control orientation
  }
}
```

**Parameters:**
- **pose** (array): Target pose. Can be [x, y, z] for position-only or [x, y, z, roll, pitch, yaw] for full pose
- **duration** (number, optional): Movement duration in seconds
- **orientation_needed** (boolean, optional): Whether to consider orientation in IK (default: false)

### 3. `open_gripper`
Open the gripper.

```json
{
  "action": "open_gripper",
  "description": "Open gripper"
}
```

**Parameters:** None

### 4. `close_gripper`
Close the gripper.

```json
{
  "action": "close_gripper",
  "description": "Close gripper"
}
```

**Parameters:** None

### 5. `home`
Move to the home position.

```json
{
  "action": "home",
  "description": "Return to home position",
  "parameters": {
    "duration": 3.0  // Optional: movement duration in seconds
  }
}
```

**Parameters:**
- **duration** (number, optional): Movement duration in seconds (default: 3.0)

### 6. `idle`
Stay idle for a specified duration.

```json
{
  "action": "idle",
  "description": "Wait for specified time",
  "parameters": {
    "duration": 2.0  // Duration in seconds
  }
}
```

**Parameters:**
- **duration** (number, required): Idle duration in seconds

### 7. `print_state`
Print the current robot state.

```json
{
  "action": "print_state",
  "description": "Print current robot state"
}
```

**Parameters:** None

### 8. `emergency_stop`
Activate emergency stop.

```json
{
  "action": "emergency_stop",
  "description": "Activate emergency stop"
}
```

**Parameters:** None

### 9. `reset_emergency_stop`
Reset emergency stop.

```json
{
  "action": "reset_emergency_stop",
  "description": "Reset emergency stop"
}
```

**Parameters:** None



## Coordinate System

- **Position coordinates**: [x, y, z] in meters relative to robot base
- **Orientation coordinates**: [roll, pitch, yaw] in radians using XYZ Euler convention
- **Joint angles**: In radians, corresponding to joints 1-6 of the PiPER arm

## Joint Limits

The robot arm has the following joint limits (in radians):

- **joint1**: [-2.618, 2.618] 
- **joint2**: [0, 3.14]
- **joint3**: [-2.697, 0]
- **joint4**: [-1.832, 1.832]
- **joint5**: [-1.22, 1.22]
- **joint6**: [-3.14, 3.14]

## Error Handling

- If inverse kinematics fails for a Cartesian target, the action will fail and execution will continue
- If joint positions exceed limits, the action will be rejected
- Emergency stop can halt execution at any time
- Invalid JSON will be detected during validation

## Usage Examples

### Validate configuration:
```bash
python run_piper.py my_config.json --validate-only
```

### Run with viewer:
```bash
python run_piper.py my_config.json
```

### Run without viewer:
```bash
python run_piper.py my_config.json --headless
```
