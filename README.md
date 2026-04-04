# RoboWeaver: Heterogeneous Multi-Robot Simulation Framework

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

**RoboWeaver** is a MuJoCo-based framework for composing heterogeneous robots into one shared scene at runtime. A JSON scenario defines scene objects, robot placements, and action sequences; `run_robots.py` merges robot XML assets, builds a temporary MuJoCo model, and launches each robot controller on its own thread.

### Key Characteristics

- Dynamic scene weaving: merge multiple robot XML templates into one MuJoCo scene at runtime.
- Heterogeneous teams: arms, mobile bases, drones, and conveyors can coexist in one world.
- Parallel execution: each robot controller runs on its own thread through a shared API.
- Configuration-driven: scene layouts and action sequences live in JSON.
- World-frame commands: controllers handle base-relative transforms internally.

### Supported Robot Types

| Type (`type`) | Model | Category | Core Capabilities |
| :--- | :--- | :--- | :--- |
| `piper` | AgileX PiPER | 6-DOF arm | `move_cartesian`, `move_joints`, `gripper` |
| `mirobot` | WLKATA Mirobot | 6-DOF arm | `move_cartesian`, `move_linear`, `move_joints` |
| `franka` | Franka Emika Panda | 7-DOF arm | `move_cartesian`, `move_linear`, `move_joints`, `gripper` |
| `stretch` | Hello Robot Stretch 3 | Mobile manipulator | base and arm control |
| `tracer` | AgileX Tracer 2 | Differential AMR | base motion |
| `rbtheron` | Robotnik RB-Theron | Omnidirectional AMR | base motion |
| `skydio` | Skydio X2 | Quadrotor UAV | `takeoff`, `land`, `move_distance` |
| `conveyor` | Parametric Conveyor | Conveyor robot | `run`, `idle` |

### Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run a scenario:
   ```bash
   python run_robots.py examples/dual_arm_collab.json
   ```
3. Run headless:
   ```bash
   python run_robots.py examples/conveyor_robot_demo.json --headless
   ```
4. Run the Franka grasp demo:
   ```bash
   python run_robots.py examples/franka_arm_demo.json --headless
   ```

### Configuration Guide

#### `scene`

- `friction`: default geom friction `[sliding, torsional, rolling]`
- `solimp` / `solref`: MuJoCo contact solver parameters
- `timestep`: optional global simulation timestep
- `objects`: scene objects such as `box`, `sphere`, `cylinder`, and `capsule`

Example movable object:

```json
{
  "name": "red_box",
  "type": "box",
  "size": [0.02, 0.02, 0.02],
  "pos": [0.72, 0.0, 0.19],
  "movable": true,
  "mass": 0.05
}
```

#### `robots`

- `name`: unique instance name
- `type`: registered robot type
- `base_pos` / `base_yaw`: world placement of the robot base
- `sequence`: ordered list of action objects

For `type: "conveyor"`, these extra size fields are supported:

- `length`: overall conveyor length in meters, default `1.04`
- `width`: overall conveyor width in meters, default `0.32`
- `height`: overall conveyor height in meters, default `0.144`

Example:

```json
{
  "name": "belt_1",
  "type": "conveyor",
  "length": 1.4,
  "width": 0.36,
  "height": 0.16,
  "base_pos": [0.8, 0.0, 0.0]
}
```

Common arm actions:

- `move_joints`
- `move_cartesian`
- `move_linear`
- `home`
- `open_gripper` / `close_gripper` for gripper-equipped arms

### Conveyor Behavior

Conveyors now use normal scene-object collision behavior. Any movable object placed on the visible belt surface will be carried by the conveyor without any extra role flag or collision-group setting.

- The visible dark belt defines the supported transport region.
- A hidden driver layer is aligned with that region and provides the motion.
- Objects move while they remain on the belt and fall only after they pass beyond its end.

This keeps scenario JSON simple: ordinary boxes, cylinders, or other movable objects can be placed on the belt directly.

### Franka Notes

Franka runtime support is self-contained under `robots/franka_control/`.

- MuJoCo model: `robots/franka_control/franka_emika_panda/panda.xml`
- IKPy URDF: `robots/franka_control/franka_panda.urdf`
- Demo: `examples/franka_arm_demo.json`

`franka` and `piper` both use fixed gripper timing:

- `open_gripper` always occupies 2 simulated seconds
- `close_gripper` always occupies 2 simulated seconds
- the next action does not start early even if the fingers reach the target sooner

### Developer Notes

To add a new robot:

1. Create a controller under `robots/<robot>_control/` inheriting from `BaseRobotController`.
2. Register it in `ROBOT_CLASSES`.
3. Add XML template and optional URDF paths in `run_robots.py`.
4. Add at least one example JSON under `examples/`.

---

<a name="中文"></a>
## 中文

**RoboWeaver** 是一个基于 MuJoCo 的异构多机器人仿真框架。项目通过 JSON 场景配置，在运行时动态合并不同机器人的 XML 模型与资源，并由 `run_robots.py` 构建临时场景、加载控制器、并行执行动作序列。

### 核心特点

- 动态场景拼装：运行时把多个机器人 XML 模板合并到同一个 MuJoCo 场景中。
- 异构机器人协作：机械臂、移动底盘、无人机、传送带可以同时存在。
- 并行执行：每个机器人控制器在独立线程中运行，共享统一控制接口。
- 配置驱动：场景布局、任务流程、动作序列全部由 JSON 描述。
- 世界坐标控制：用户直接写全局坐标，控制器内部处理基座变换。

### 当前支持的机器人类型

| 类型 (`type`) | 模型 | 类别 | 主要能力 |
| :--- | :--- | :--- | :--- |
| `piper` | AgileX PiPER | 六轴机械臂 | `move_cartesian`, `move_joints`, `gripper` |
| `mirobot` | WLKATA Mirobot | 六轴机械臂 | `move_cartesian`, `move_linear`, `move_joints` |
| `franka` | Franka Emika Panda | 七轴机械臂 | `move_cartesian`, `move_linear`, `move_joints`, `gripper` |
| `stretch` | Hello Robot Stretch 3 | 移动操作机器人 | 底盘与机械臂控制 |
| `tracer` | AgileX Tracer 2 | 差速移动底盘 | 底盘运动 |
| `rbtheron` | Robotnik RB-Theron | 全向移动底盘 | 底盘运动 |
| `skydio` | Skydio X2 | 四旋翼无人机 | `takeoff`, `land`, `move_distance` |
| `conveyor` | 参数化传送带 | 传送带机器人 | `run`, `idle` |

### 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 运行示例：
   ```bash
   python run_robots.py examples/dual_arm_collab.json
   ```
3. 无界面运行：
   ```bash
   python run_robots.py examples/conveyor_robot_demo.json --headless
   ```
4. 运行 Franka 抓取示例：
   ```bash
   python run_robots.py examples/franka_arm_demo.json --headless
   ```

### 配置说明

#### `scene`

- `friction`：默认几何体摩擦参数 `[滑动, 扭转, 滚动]`
- `solimp` / `solref`：MuJoCo 接触求解参数
- `timestep`：可选的全局仿真步长
- `objects`：场景物体，例如 `box`、`sphere`、`cylinder`、`capsule`

可移动物体示例：

```json
{
  "name": "red_box",
  "type": "box",
  "size": [0.02, 0.02, 0.02],
  "pos": [0.72, 0.0, 0.19],
  "movable": true,
  "mass": 0.05
}
```

#### `robots`

- `name`：机器人实例名
- `type`：机器人类型
- `base_pos` / `base_yaw`：机器人基座在世界坐标系中的位置和偏航角
- `sequence`：按顺序执行的动作列表

对于 `type: "conveyor"`，额外支持以下尺寸参数：

- `length`：传送带整体长度，单位米，默认 `1.04`
- `width`：传送带整体宽度，单位米，默认 `0.32`
- `height`：传送带整体高度，单位米，默认 `0.144`

示例：

```json
{
  "name": "belt_1",
  "type": "conveyor",
  "length": 1.4,
  "width": 0.36,
  "height": 0.16,
  "base_pos": [0.8, 0.0, 0.0]
}
```

机械臂常用动作包括：

- `move_joints`
- `move_cartesian`
- `move_linear`
- `home`
- 带夹爪的机械臂支持 `open_gripper` / `close_gripper`

### 传送带行为

传送带现在使用普通场景物体碰撞语义，不再需要 `role: "cargo"` 之类的额外配置。只要物体是可移动的，并且真正放在可见皮带表面上，它就会被传送带带走。

- 深色皮带表面定义了有效输送区域。
- 隐藏驱动层与这段区域对齐，负责提供运动。
- 物体只要还在皮带范围内就会持续运动，超出末端后才会掉落。

这样配置文件会更简单：普通方块、圆柱或其他可移动物体都可以直接放到皮带上，不需要专门的角色字段。

### Franka 说明

Franka 的运行依赖已经收敛到 `robots/franka_control/` 目录内。

- MuJoCo 模型：`robots/franka_control/franka_emika_panda/panda.xml`
- IKPy 使用的 URDF：`robots/franka_control/franka_panda.urdf`
- 示例：`examples/franka_arm_demo.json`

`franka` 和 `piper` 的夹爪动作采用固定时长策略：

- `open_gripper` 固定占用 2 秒仿真时间
- `close_gripper` 固定占用 2 秒仿真时间
- 即使提前到位，也会等满 2 秒才开始下一步动作

### 开发说明

如果要新增一种机器人：

1. 在 `robots/<robot>_control/` 下新增控制器，并继承 `BaseRobotController`。
2. 在 `run_robots.py` 中注册控制器、XML 模板和可选 URDF。
3. 在 `examples/` 下补一个最小可运行示例。
