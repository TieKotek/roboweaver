# Robotsim: Multi-Agent Embodied AI Research Platform

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

**Robotsim** is a high-performance, lightweight, and highly extensible simulation framework built on **MuJoCo**. It is specifically designed as an evaluation and demonstration platform for **Multi-Agent Embodied AI** research, allowing researchers to simulate complex collaborative tasks with heterogeneous robot teams.

### 🌟 Highlights

- **🚀 Lightweight & Fast**: Minimal overhead with high-fidelity physics powered by MuJoCo.
- **🤖 Multi-Agent Ready**: Native support for simultaneous control of multiple heterogeneous robots in a shared environment.
- **🧵 Threaded Execution**: Each robot operates on its own dedicated thread, ensuring non-blocking, real-time control.
- **⚙️ Highly Customizable**: Purely JSON-driven scene and action configuration. No recompilation needed for scene changes.
- **📐 Unified World Frame**: Automatic handling of base-relative transformations; command your robots using global coordinates.
- **🔌 Extensible Architecture**: Modular design makes it easy to plug in new robot types (arms, mobile bases, etc.) by implementing a standard API.

---

### 🛠️ Installation

Ensure you have Python 3.8+ and install the dependencies:

```bash
pip install mujoco numpy scipy ikpy
```

---

### 🚀 Quick Start

Launch the multi-robot pick-and-place evaluation demo:

```bash
python run_robots.py examples/task_config.json
```

**Options:**
- `--headless`: Run without the visual viewer (optimized for training and large-scale testing).

---

### 📝 Configuration Format (`.json`)

The configuration file controls the scene layout and robot behaviors.

#### 1. Scene Configuration
Defines the static environment and global physics parameters.

**Global Physics Parameters:**
These parameters apply as defaults to all objects and are tuned to improve grasp stability in simulation.
- **friction** (array, optional): `[sliding, torsional, rolling]` friction coefficients.
  - *Sliding*: Resistance to linear movement. Higher values (e.g., 2.0-5.0) help prevent objects from slipping.
  - *Torsional*: Resistance to rotation around the contact normal. Prevents objects from spinning in the grasp.
- **solimp** (array, optional): `[dmin, dmax, width, mid, power]` - Solver Impedance. Values close to 1 (e.g., `0.95`, `0.99`) make contacts "harder" and more precise.
- **solref** (array, optional): `[timeconst, dampratio]` - Solver Reference. `dampratio=1` indicates critical damping (no bouncing).

**Object Definition:**
Each object in the `objects` list supports:
- **type**: "box", "sphere", "cylinder", "capsule".
- **size**: [x, y, z] for boxes; [radius] for spheres; [radius, half-length] for others.
- **pos** / **quat** / **euler**: World placement.
- **movable**: Boolean. If true, the object has a free joint (physics enabled).
- **mass**: Mass of the object. Lower mass (e.g., 0.01) often improves stability for small grasping targets.

#### 2. Robot Configuration
- **name**: Unique identifier for the robot.
- **type**: Matches the registered type in `run_robots.py`.
- **base_pos**: `[x, y, z]` World position of the robot base.
- **sequence**: List of actions to execute.

#### 3. Actions
Standardized actions dispatched to robot controllers:
*   **`move_cartesian`**: Move end-effector to target pose in **World Coordinates**.
*   **`move_joints`**: Move specific joints to target angles (radians).
*   **`open_gripper` / `close_gripper`**: End-effector interaction.
*   **`home`**: Return to default safe position.
*   **`idle`**: Wait for a duration.

---

### 👩‍💻 Developer Guide: Adding New Robots

Robotsim is designed to be agnostic to robot kinematics and morphology. To add a new robot:

#### Step 1: Create the Controller Module
Create a directory in `robots/` (e.g., `robots/XXrobot_control/`) and a controller file (e.g., `XXrobot_controller.py`). Your class **must** inherit from `BaseRobotController`.

```python
from common.robot_api import BaseRobotController, RobotState

class XXrobotController(BaseRobotController):
    def __init__(self, model, data, robot_name, base_pos=None, base_quat=None, **kwargs):
        super().__init__(model, data, robot_name, base_pos, base_quat)
        # Initialize your specific kinematics or actuators here

    def get_robot_state(self) -> RobotState:
        # Return current state (timestamp, pose, etc.)
        pass

    # Implement specific actions with the prefix 'action_'
    def action_navigate_to(self, x, y, theta):
        # Implement your custom logic here
        pass
```

#### Step 2: Prepare Assets (URDF vs XML)
- **URDF**: Used exclusively for **Kinematics** (IK/FK calculation). It defines the mathematical joint chain.
- **XML**: Used for **MuJoCo Physics** and visualization. It defines geometries, visual assets (STLs), and actuators.

#### Step 3: Register in `run_robots.py`
Open `run_robots.py` and register your new robot type in the global dictionaries:

```python
# Import your new controller
from robots.XXrobot_control.XXrobot_controller import XXrobotController

ROBOT_CLASSES = {
    "piper": PiperController,
    "XXrobot": XXrobotController,  # <--- Add this
}

# Optional: Only when controller needs URDF files for IK
ROBOT_URDFS = {
    "piper": "robots/piper_control/agilex_piper/piper_description.urdf",
    "XXrobot": "robots/XXrobot_control/assets/XXrobot.urdf", # <--- Add this
}

ROBOT_XML_TEMPLATES = {
    "piper": "robots/piper_control/agilex_piper/piper.xml",
    "XXrobot": "robots/XXrobot_control/assets/XXrobot.xml",  # <--- Add this
}
```

#### Step 4: Usage
You can now use `"type": "XXrobot"` in your JSON configuration file!

---

### 📐 Coordinate Systems

*   **World Frame**: The global MuJoCo frame `(0,0,0)`. All `pose` parameters in the JSON config are interpreted as World Frame coordinates.
*   **Base Frame**: The local frame of the robot, defined by `base_pos` in the config.
*   **Transformation**: The `BaseRobotController` (and its subclasses) automatically transforms World Frame targets into Local Frame targets before calculating Inverse Kinematics. Users do **not** need to manually subtract the base offset.

---

<a name="中文"></a>
## 中文

**Robotsim** 是一个基于 **MuJoCo** 构建的高性能、轻量级且高度可扩展的仿真框架。它专门为 **多智能体具身智能 (Multi-Agent Embodied AI)** 研究而设计，是一个理想的评估与展示平台，能够支持研究人员在共享环境中模拟复杂的异构机器人协作任务。

### 🌟 项目亮点

- **🚀 轻量且快速**：基于 MuJoCo 物理引擎，在保持高精度物理模拟的同时将系统开销降至最低。
- **🤖 原生多智能体支持**：支持在同一场景中同时控制多个不同类型的机器人。
- **🧵 多线程执行**：每个机器人运行在独立的线程中，确保实时、非阻塞的控制响应。
- **⚙️ 高度可自定义**：纯 JSON 驱动的场景和动作配置，无需重新编译即可更改仿真内容。
- **📐 统一世界坐标系**：自动处理相对于基座的坐标转换，直接使用全局坐标指挥机器人。
- **🔌 易于扩展**：模块化设计，通过实现标准 API 即可轻松接入新机器人（机械臂、移动底座等）。

---

### 🛠️ 安装

确保您已安装 Python 3.8+ 并安装以下依赖：

```bash
pip install mujoco numpy scipy ikpy
```

---

### 🚀 快速开始

运行多智能体协作取放评估演示：

```bash
python run_robots.py examples/task_config.json
```

**常用选项：**
- `--headless`：在没有可视化窗口的情况下运行（适用于模型训练和大规模自动化测试）。

---

### 📝 配置格式 (`.json`)

配置文件用于控制场景布局和机器人行为。

#### 1. 场景配置 (Scene)
定义静态环境和全局物理参数。

**全局物理参数：**
这些参数作为所有物体的默认值，经过调优以提高仿真中的抓取稳定性。
- **friction** (数组，可选)：`[滑动, 扭转, 滚动]` 摩擦系数。
  - *滑动*：线性运动阻力。较高的值（如 2.0-5.0）有助于防止物体滑落。
  - *扭转*：绕接触法线的旋转阻力。防止物体在抓取中旋转。
- **solimp** (数组，可选)：`[dmin, dmax, width, mid, power]` - 求解器阻抗。接近 1 的值（如 `0.95`, `0.99`）使接触更硬、更精确。
- **solref** (数组，可选)：`[timeconst, dampratio]` - 求解器参考。`1` 表示临界阻尼（不产生反弹）。

**物体定义：**
`objects` 列表中的每个物体支持：
- **type**：几何类型 ("box", "sphere", "cylinder", "capsule")。
- **size**：box 为 [x, y, z]；其他类型根据定义提供半径或长度。
- **pos** / **quat** / **euler**：在世界坐标系中的位置和姿态。
- **movable**：布尔值。设为 true 则启用物理仿真（添加 freejoint）。
- **mass**：质量。对于小型抓取目标，较低的质量（如 0.01）可提高稳定性。

#### 2. 机器人配置 (Robots)
- **name**：机器人的唯一标识符。
- **type**：匹配 `run_robots.py` 中注册的类型。
- **base_pos**：基座在世界坐标系中的位置 `[x, y, z]`。
- **sequence**：要执行的动作序列。

#### 3. 动作 (Actions)
动作会动态分发给机器人控制器。常见动作包括：
*   **`move_cartesian`**：末端执行器移动到**世界坐标系**下的目标位姿。
*   **`move_joints`**：移动到目标关节角度（弧度）。
*   **`open_gripper` / `close_gripper`**：末端执行器控制。
*   **`home`**：返回默认安全位置。
*   **`idle`**：等待（与仿真时间同步）。

---

### 👩‍💻 开发指南：添加新机器人

Robotsim 的设计与机器人的运动学结构和形态无关。添加新机器人的步骤如下：

#### 第一步：创建控制器模块
在 `robots/` 目录下创建目录（如 `robots/my_robot_control/`）和控制器文件（如 `my_controller.py`）。您的类**必须**继承自 `BaseRobotController`。

```python
from common.robot_api import BaseRobotController, RobotState

class MyRobotController(BaseRobotController):
    def __init__(self, model, data, robot_name, base_pos=None, base_quat=None, **kwargs):
        super().__init__(model, data, robot_name, base_pos, base_quat)
        # 在此处初始化特定的运动学或执行器

    def get_robot_state(self) -> RobotState:
        # 返回当前状态（时间戳、位姿等）
        pass

    # 实现以 'action_' 为前缀的操作
    def action_navigate_to(self, x, y, theta):
        # 在此处实现您的自定义逻辑
        pass
```

#### 第二步：准备资产 (URDF 与 XML)
- **URDF**：专门用于 **运动学 (IK/FK)** 计算。定义数学上的关节链结构。
- **XML**：专门用于 **MuJoCo 物理仿真**。定义几何体、视觉资产 (STL) 和执行器。

#### 第三步：在 `run_robots.py` 中注册
打开 `run_robots.py` 并在全局字典中注册您的新机器人类型：

```python
from robots.my_robot_control.my_controller import MyRobotController

ROBOT_CLASSES = {
    "my_robot": MyRobotController,
}

ROBOT_URDFS = {
    "my_robot": "robots/my_robot_control/assets/my_robot.urdf",
}

ROBOT_XML_TEMPLATES = {
    "my_robot": "robots/my_robot_control/assets/my_robot.xml",
}
```

#### 第四步：使用
在您的 JSON 配置文件中使用 `"type": "my_robot"` 即可！

---

### 📐 坐标系说明

- **世界坐标系 (World Frame)**：全局 MuJoCo 坐标系 `(0,0,0)`。JSON 配置中的所有位姿参数均被视为世界坐标。
- **基座坐标系 (Base Frame)**：机器人的局部坐标系，由配置中的 `base_pos` 定义。
- **自动转换**：`BaseRobotController` 会在计算逆运动学之前自动将世界坐标目标转换为局部坐标目标，用户**无需**手动计算位置偏移。
