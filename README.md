# RoboWeaver: Heterogeneous Multi-Robot Simulation Framework

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

**RoboWeaver** is a high-performance, lightweight, and highly extensible simulation framework built on **MuJoCo**. Unlike static environments, RoboWeaver "weaves" together heterogeneous robots (arms, mobile bases, drones) into a unified simulation fabric on-the-fly using a dynamic scene-building engine. It is specifically designed as an evaluation and demonstration platform for **Multi-Agent Embodied AI** research.

### 🌟 Key Characteristics

- **🕸️ Dynamic Scene Weaving**: Automatically merges individual robot XML templates, meshes, and textures into a single consistent MuJoCo scene at runtime via JSON configuration.
- **🤖 Heterogeneous Teams**: Native support for simultaneous control of diverse robot types (Manipulators, AMRs, UAVs) in a shared physical environment.
- **🧵 Parallel Execution**: Each robot operates on its own dedicated thread with a standardized asynchronous API, ensuring non-blocking, real-time control.
- **⚙️ Configuration-Driven**: Define complex tasks, scene layouts, and robot sequences purely through JSON. No recompilation is required.
- **📐 Universal Coordinate Frame**: Command all robots using global coordinates; the framework handles base-relative transformations automatically.

---

### 🤖 Supported Robot Gallery

| Type (`type`) | Model | Category | Core Capabilities |
| :--- | :--- | :--- | :--- |
| `piper` | AgileX PiPER | 6-DOF Arm | `move_cartesian`, `move_joints`, `gripper` |
| `stretch` | Hello Robot Stretch 3 | Mobile Manipulator | `move_distance`, `rotate`, `move_cartesian` |
| `tracer` | AgileX Tracer 2 | Differential AMR | `move_distance`, `rotate` |
| `rbtheron` | Robotnik RB-Theron | Omnidirectional AMR | `move_distance`, `rotate` |
| `skydio` | Skydio X2 | Quadrotor UAV | `takeoff`, `land`, `move_distance`, `rotate` |

---

### 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install mujoco numpy scipy ikpy
   ```
2. **Launch a Task**:
   ```bash
   python run_robots.py examples/dual_arm_collab.json
   ```
   *Use `--headless` to run without the visual viewer.*

---

### 📝 Detailed Configuration Guide (`.json`)

The configuration file controls the entire "weaving" process.

#### 1. Scene Configuration (`scene`)
Defines the environment and global physics.
*   **`friction`** (Array, optional): `[sliding, torsional, rolling]`. 
    *   *Sliding*: Resistance to linear movement. Higher values (e.g., `2.0`) prevent object slippage.
    *   *Torsional*: Prevents objects from spinning within a gripper.
*   **`solimp` / `solref`**: MuJoCo solver parameters. `solref=[0.02, 1]` ensures critical damping (no bouncing).
*   **`objects`**: List of dynamic objects.
    *   `type`: "box", "sphere", "cylinder", "capsule".
    *   `movable`: Boolean. If `true`, a `freejoint` is added for physics simulation.
    *   `mass`: Lower mass (e.g., `0.01`) often improves grasping stability for small targets.

#### 2. Robot Configuration (`robots`)
*   **`name`**: Unique identifier for the instance.
*   **`type`**: Matches the registered type (e.g., `piper`).
*   **`base_pos` / `base_yaw`**: World placement of the robot base.
*   **`sequence`**: List of action objects (e.g., `{"action": "move_cartesian", "parameters": {...}}`).

#### 3. Standard Actions
*   **`move_cartesian`**: Move end-effector to target pose in **World Coordinates**.
*   **`move_joints`**: Move joints to target angles (radians).
*   **`home` / `idle`**: Return to safe pose or wait for a duration.

---

### 👩‍💻 Developer Guide: Adding New Robots

RoboWeaver is agnostic to robot morphology. Follow these steps to "weave" in a new robot:

#### Step 1: Create the Controller
Create a directory in `robots/` and a controller inheriting from `BaseRobotController`.

```python
from common.robot_api import BaseRobotController, RobotState

class MyRobotController(BaseRobotController):
    def __init__(self, model, data, robot_name, base_pos=None, base_quat=None, **kwargs):
        super().__init__(model, data, robot_name, base_pos, base_quat)
        # Initialize IK or specific actuators

    def get_robot_state(self) -> RobotState:
        return RobotState(timestamp=self.data.time)

    # Implement actions with 'action_' prefix
    def action_custom_move(self, target_val):
        # Your logic here
        pass
```

#### Step 2: Register in `run_robots.py`
Add your robot to the three core registries:
```python
ROBOT_CLASSES = { "my_robot": MyRobotController }
ROBOT_URDFS = { "my_robot": "path/to/model.urdf" } # Optional for IK
ROBOT_XML_TEMPLATES = { "my_robot": "path/to/model.xml" } # Required for Physics
```

---

### 📐 Coordinate Systems & Logic

*   **World Frame**: The global MuJoCo origin `(0,0,0)`. All JSON `pos` values are global.
*   **Base Frame**: The local frame defined by `base_pos` in the config.
*   **Auto-Transformation**: The `BaseRobotController` automatically transforms World Frame targets into the robot's Local Frame before calculating Inverse Kinematics. Users do **not** need to manually handle offsets.

---

<a name="中文"></a>
## 中文

**RoboWeaver** 是一个基于 **MuJoCo** 构建的高性能、轻量级且高度可扩展的仿真框架。与传统的静态环境不同，RoboWeaver 通过动态场景构建引擎，将异构机器人（机械臂、移动底座、无人机）实时“编织”到一个统一的仿真布景中。它专为 **多智能体具身智能 (Multi-Agent Embodied AI)** 研究而设计。

### 🌟 核心特性

- **🕸️ 动态场景编织 (Scene Weaving)**：运行时自动合并各机器人的 XML 模板、模型网格和贴图。
- **🤖 异构机器人集群**：原生支持在同一物理空间内同时控制机械臂、移动底座、无人机等多种机器人。
- **🧵 并行异步控制**：每个机器人运行在独立线程，通过标准化异步 API 确保实时控制。
- **⚙️ 纯配置驱动**：通过 JSON 定义复杂的场景布局和动作序列，无需改动代码或重新编译。
- **📐 统一世界坐标系**：直接使用全局坐标指挥所有机器人，框架自动处理复杂的基座相对变换。

---

### 🤖 已支持机器人库

| 类型 (`type`) | 型号 | 类别 | 核心功能 |
| :--- | :--- | :--- | :--- |
| `piper` | AgileX PiPER (松灵) | 6轴机械臂 | `move_cartesian`, `move_joints`, `gripper` |
| `stretch` | Hello Robot Stretch 3 | 移动操作机器人 | 大范围操作、移动导航 |
| `tracer` | AgileX Tracer 2 (松灵) | 差速移动底盘 | 高负载移动运输 |
| `rbtheron` | Robotnik RB-Theron | 全向移动底盘 | 工业级全向底座 |
| `skydio` | Skydio X2 | 四旋翼无人机 | 航空巡检、自主飞行 |

---

### 🚀 快速开始

1. **安装依赖**:
   ```bash
   pip install mujoco numpy scipy ikpy
   ```
2. **运行任务**:
   ```bash
   python run_robots.py examples/dual_arm_collab.json
   ```
   *使用 `--headless` 可在无界面模式下运行。*

---

### 📝 详细配置指南 (`.json`)

配置文件是“编织”场景的核心蓝图。

#### 1. 场景配置 (`scene`)
定义环境背景与全局物理。
*   **`friction`** (数组, 可选): `[滑动, 扭转, 滚动]`。
    *   *滑动*: 线性运动阻力。建议值 `2.0` 以增强抓取稳定性。
    *   *扭转*: 防止物体在夹爪中发生扭转旋转。
*   **`solimp` / `solref`**: MuJoCo 求解器参数。`solref=[0.02, 1]` 可实现临界阻尼，消除碰撞回弹。
*   **`objects`**: 场景中的动态物体列表。
    *   `type`: "box", "sphere", "cylinder", "capsule"。
    *   `movable`: 布尔值。设为 `true` 将启用物理仿真。
    *   `mass`: 质量。对于小型抓取目标，较低质量（如 `0.01`）可显著提升稳定性。

#### 2. 机器人配置 (`robots`)
*   **`name`**: 实例的唯一名称。
*   **`type`**: 匹配注册的机器人类型（如 `piper`）。
*   **`base_pos` / `base_yaw`**: 机器人在世界坐标系中的初始位姿。
*   **`sequence`**: 动作序列（如 `{"action": "move_cartesian", "parameters": {...}}`）。

#### 3. 标准动作
*   **`move_cartesian`**：末端执行器移动到**世界坐标系**下的目标位姿。
*   **`move_joints`**：移动到目标关节角度（弧度）。
*   **`home` / `idle`**：返回默认位姿或等待一段时间。

---

### 👩‍💻 开发指南：接入新机器人

RoboWeaver 的模块化设计允许您快速接入任何形态的机器人：

#### 第一步：创建控制器
在 `robots/` 下创建目录并编写继承自 `BaseRobotController` 的类。

```python
from common.robot_api import BaseRobotController, RobotState

class MyRobotController(BaseRobotController):
    def __init__(self, model, data, robot_name, base_pos=None, base_quat=None, **kwargs):
        super().__init__(model, data, robot_name, base_pos, base_quat)
        # 初始化运动学或执行器

    def get_robot_state(self) -> RobotState:
        return RobotState(timestamp=self.data.time)

    # 实现以 'action_' 为前缀的方法
    def action_custom_move(self, target_val):
        # 编写您的控制逻辑
        pass
```

#### 第二步：在 `run_robots.py` 中注册
将您的机器人添加到三个核心注册字典中：
```python
ROBOT_CLASSES = { "my_robot": MyRobotController }
ROBOT_URDFS = { "my_robot": "path/to/model.urdf" } # IK 可选
ROBOT_XML_TEMPLATES = { "my_robot": "path/to/model.xml" } # 物理仿真必填
```

---

### 📐 坐标系说明

- **世界坐标系 (World Frame)**：全局 MuJoCo 原点 `(0,0,0)`。JSON 中所有 `pos` 均为全局坐标。
- **基座坐标系 (Base Frame)**：由配置中的 `base_pos` 定义的局部坐标。
- **自动转换逻辑**：`BaseRobotController` 会在计算逆运动学之前自动将世界坐标目标转换为局部坐标目标，用户**无需**手动计算偏移。
