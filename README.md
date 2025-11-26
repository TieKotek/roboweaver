# Multi-Robot Simulation Framework

A comprehensive, modular simulation framework based on **MuJoCo**, designed to control multiple robots (starting with AgileX PiPER arms) through a unified JSON configuration interface.

## Features

- **Multi-Robot Support**: Control multiple robots simultaneously in a shared environment.
- **Threaded Execution**: Each robot runs on its own thread for non-blocking control.
- **World Coordinate System**: Motion commands use global world coordinates; the framework automatically handles base-relative transformations.
- **JSON-Driven**: Define scenes, robot placements, and action sequences purely via configuration files.
- **Extensible Architecture**: Easily plug in new robot types (wheeled bases, drones, etc.) by implementing a standard interface.

## Installation

Ensure you have Python 3.8+ and the required dependencies:

```bash
pip install mujoco numpy scipy ikpy
```

## Quick Start

Run the dual-arm pick-and-place demo:

```bash
python run_robots.py example_config.json
```

**Options:**
- `--headless`: Run without the visual viewer (faster, suitable for training/testing).

## Configuration Format (`.json`)

The configuration file controls the scene layout and robot behaviors.

### Basic Structure

```json
{
  "description": "Demo Description",
  "scene": {
    "friction": [2.0, 0.005, 0.0001],
    "objects": [...]
  },
  "robots": [
    {
      "name": "left_arm",
      "type": "piper",
      "base_pos": [0, 0.4, 0],
      "sequence": [...]
    },
    {
      "name": "right_arm",
      "type": "piper",
      "base_pos": [0, -0.4, 0],
      "sequence": [...]
    }
  ]
}
```

### 1. Scene Configuration
Defines the static environment and global physics parameters.
- **friction**: `[sliding, torsional, rolling]` (Global default).
- **objects**: List of static or movable objects (boxes, tables, etc.).


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

#### Object Definition

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



### 2. Robot Configuration
- **name**: Unique identifier for the robot (e.g., "left_arm"). Used for logging and XML naming.
- **type**: The class of robot (e.g., "piper"). Must match a registered type in `run_robots.py`.
- **base_pos**: `[x, y, z]` World position of the robot base.
- **sequence**: List of actions to execute.

### 3. Actions
Actions are dispatched dynamically to the robot controller.

**Common Actions (PiPER):**
*   **`move_cartesian`**: Move end-effector to a target pose.
    *   `pose`: `[x, y, z]` or `[x, y, z, roll, pitch, yaw]` (**World Coordinates**).
    *   `duration`: Time in seconds.
*   **`move_joints`**: Move specific joints.
    *   `joints`: List of joint angles (radians).
*   **`open_gripper` / `close_gripper`**: Control the end effector.
*   **`home`**: Return to default safe position.
*   **`idle`**: Wait for a duration.

## Developer Guide: Adding New Robots

The framework is designed to be agnostic to the robot type. To add a new robot:

### Step 1: Create the Controller Module
Create a directory (e.g., `XXrobot_control/`) and a controller file (e.g., `XXrobot_controller.py`).

Your controller class **must** inherit from `BaseRobotController` and implement the abstract methods.

```python
from common.robot_api import BaseRobotController, RobotState

class XXrobotController(BaseRobotController):
    def __init__(self, model, data, robot_name, base_pos=None, base_quat=None, **kwargs):
        super().__init__(model, data, robot_name, base_pos, base_quat)
        # Initialize your specific kinematics or actuators here

    def get_robot_state(self) -> RobotState:
        # Return current state
        pass

    # Implement specific actions with the prefix 'action_'
    def action_navigate_to(self, x, y, theta):
        # Implement navigation logic
        pass
```

### Step 2: Prepare Assets
Ensure you have the URDF and XML files for your robot.
*   **URDF**: For kinematics (IK/FK).
*   **XML**: For MuJoCo physics and visualization.

### Step 3: Register in `run_robots.py`
Open `run_robots.py` and register your new robot type in the global dictionaries at the top of the file:

```python
# Import your new controller
from XXrobot_control.XXrobot_controller import XXrobotController

ROBOT_CLASSES = {
    "piper": PiperController,
    "XXrobot": XXrobotController,  # <--- Add this
}

# Optional: Only when controller needs URDF files
ROBOT_URDFS = {
    "piper": "piper_control/agilex_piper/piper_description.urdf",
    "XXrobot": "XXrobot_control/assets/XXrobot.urdf", # <--- Add this
}

ROBOT_XML_TEMPLATES = {
    "piper": "piper_control/agilex_piper/piper.xml",
    "XXrobot": "XXrobot_control/assets/XXrobot.xml",  # <--- Add this
}
```

### Step 4: Usage
You can now use `"type": "XXrobot"` in your JSON configuration file!

```json
{
  "name": "my_agv",
  "type": "XXrobot",
  "base_pos": [1, 1, 0],
  "sequence": [
    { "action": "navigate_to", "parameters": { "x": 2, "y": 2, "theta": 1.57 } }
  ]
}
```

## Coordinate Systems

*   **World Frame**: The global MuJoCo frame `(0,0,0)`. All `pose` parameters in the JSON config are interpreted as World Frame coordinates.
*   **Base Frame**: The local frame of the robot, defined by `base_pos` in the config.
*   **Transformation**: The `BaseRobotController` (and its subclasses) automatically transforms World Frame targets into Local Frame targets before calculating Inverse Kinematics. Users do **not** need to manually subtract the base offset.