# RoboWeaver: Heterogeneous Multi-Robot Simulation Framework

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

**RoboWeaver** is a MuJoCo-based simulation framework for composing heterogeneous robots into one shared scene at runtime. A JSON scenario describes the scene, robot placements, and action sequences; `run_robots.py` merges robot XML assets, builds a temporary MuJoCo model, and launches each robot controller on its own thread.

### Key Characteristics

- Dynamic scene weaving: merge multiple robot XML templates into one MuJoCo scene at runtime.
- Heterogeneous teams: arms, mobile bases, drones, and conveyors can coexist in one world.
- Parallel execution: each robot runs through a standardized controller API on its own thread.
- Configuration-driven: tasks, object layouts, and action sequences live in JSON.
- World-frame commands: controllers handle base-relative transforms internally.

### Supported Robot Types

| Type (`type`) | Model | Category | Core Capabilities |
| :--- | :--- | :--- | :--- |
| `piper` | AgileX PiPER | 6-DOF arm | `move_cartesian`, `move_joints`, `gripper` |
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

### Configuration Guide

#### `scene`

- `friction`: default geom friction `[sliding, torsional, rolling]`
- `solimp` / `solref`: MuJoCo contact solver parameters
- `timestep`: optional global simulation timestep
- `objects`: scene objects such as `box`, `sphere`, `cylinder`, `capsule`

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

### Conveyor Cargo Convention

If an object should be transported by a conveyor, use the semantic field `role: "cargo"` instead of exposing low-level MuJoCo collision masks in scenario JSON.

```json
{
  "name": "cargo_red",
  "type": "box",
  "movable": true,
  "role": "cargo"
}
```

`role: "cargo"` automatically maps the object into the conveyor-compatible collision group. Conveyor frame geoms still collide normally, while the hidden drive layer only interacts with cargo objects. This keeps arm-conveyor collaboration more stable.

### Developer Notes

To add a new robot:

1. Create a controller under `robots/<robot>_control/` inheriting from `BaseRobotController`.
2. Register it in `ROBOT_CLASSES`.
3. Add XML template and optional URDF paths in `run_robots.py`.
4. Add at least one example JSON under `examples/`.

---

<a name="中文"></a>
## 中文

**RoboWeaver** 是一个基于 MuJoCo 的多机器人仿真框架。项目通过 JSON 场景配置在运行时动态合并不同机器人的 XML 模型、资产和控制器，把机械臂、移动底盘、无人机以及传送带放进同一个共享场景中。

### 核心特点

- 动态场景拼装：运行时自动合并多个机器人 XML 模板。
- 异构机器人协作：支持机械臂、移动底盘、无人机、传送带同时存在。
- 并行执行：每个机器人控制器在独立线程中执行动作序列。
- 配置驱动：任务、场景布局和动作流程都由 JSON 描述。
- 世界坐标控制：用户直接使用全局坐标，控制器负责基座变换。

### 当前支持的机器人类型

| 类型 (`type`) | 模型 | 类别 | 主要能力 |
| :--- | :--- | :--- | :--- |
| `piper` | AgileX PiPER | 六轴机械臂 | `move_cartesian`, `move_joints`, `gripper` |
| `stretch` | Hello Robot Stretch 3 | 移动操作机器人 | 底盘与机械臂控制 |
| `tracer` | AgileX Tracer 2 | 差速移动底盘 | 底盘运动 |
| `rbtheron` | Robotnik RB-Theron | 全向移动底盘 | 底盘运动 |
| `skydio` | Skydio X2 | 四旋翼无人机 | `takeoff`, `land`, `move_distance` |
| `conveyor` | Parametric Conveyor | 传送带机器人 | `run`, `idle` |

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

### 配置说明

#### `scene`

- `friction`：默认接触摩擦参数 `[滑动, 扭转, 滚动]`
- `solimp` / `solref`：MuJoCo 接触求解器参数
- `timestep`：可选的全局仿真步长
- `objects`：场景内物体，如 `box`、`sphere`、`cylinder`、`capsule`

示例：

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

对 `type: "conveyor"`，额外支持以下尺寸参数：

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

### 传送带 Cargo 约定

如果一个物体需要被传送带运送，请在配置文件中使用语义化字段 `role: "cargo"`，不要直接暴露 `contype` / `conaffinity` 这类底层 MuJoCo 碰撞掩码。

```json
{
  "name": "cargo_red",
  "type": "box",
  "movable": true,
  "role": "cargo"
}
```

`role: "cargo"` 会自动映射到与传送带兼容的碰撞分组。传送带机架仍正常参与碰撞，而隐藏驱动层只会与 cargo 交互，这样更适合机械臂与传送带协作抓取、放置和转运场景。

### 开发说明

若要接入新机器人：

1. 在 `robots/<robot>_control/` 下新增控制器，并继承 `BaseRobotController`。
2. 在 `run_robots.py` 中注册控制器、XML 模板和可选 URDF。
3. 在 `examples/` 中补一个最小可运行示例。
