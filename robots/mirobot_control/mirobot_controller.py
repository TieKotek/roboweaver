import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional

import ikpy.chain
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from common.robot_api import BaseRobotController, RobotState


@dataclass
class MirobotState(RobotState):
    joint_positions: Optional[np.ndarray] = None
    end_effector_pose: Optional[np.ndarray] = None
    end_effector_euler: Optional[np.ndarray] = None


class MirobotController(BaseRobotController):
    """Controller for the 6-DOF WLKATA Mirobot arm."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_name: str = "mirobot",
        urdf_path: str = "robots/mirobot_control/mirobot.urdf",
        base_pos: np.ndarray = None,
        base_quat: np.ndarray = None,
        log_dir: str = None,
    ):
        super().__init__(model, data, robot_name, base_pos, base_quat, log_dir)
        self.prefix = robot_name
        self.r_tool_link6 = R.identity()
        self.r_link6_tool = self.r_tool_link6.inv()
        self.kinematics = MirobotKinematics(model, urdf_path, prefix=self.prefix)
        self.home_joints = np.zeros(6)
        self._reset_to_home()

    def get_robot_state(self) -> MirobotState:
        ee_pose_local = self.kinematics.forward_kinematics(self._get_joint_positions())
        p_local = ee_pose_local[:3]
        q_local = ee_pose_local[3:]

        r_base = R.from_quat([self.base_quat[1], self.base_quat[2], self.base_quat[3], self.base_quat[0]])
        p_world = self.base_pos + r_base.apply(p_local)
        r_world_link6 = r_base * R.from_quat(q_local)
        r_world_tool = r_world_link6 * self.r_link6_tool

        return MirobotState(
            timestamp=self.data.time,
            joint_positions=self._get_joint_positions(),
            end_effector_pose=np.concatenate([p_world, r_world_tool.as_quat()]),
            end_effector_euler=r_world_tool.as_euler("xyz", degrees=True),
        )

    def format_state(self) -> str:
        state = self.get_robot_state()
        lines = [f"[{self.robot_name}] State:"]
        lines.append(f"  Timestamp: {state.timestamp:.3f}")
        if state.joint_positions is not None:
            lines.append(f"  Joints: {np.array2string(state.joint_positions, precision=3, suppress_small=True)}")
        if state.end_effector_pose is not None:
            lines.append(
                f"  EE Pos: {np.array2string(state.end_effector_pose[:3], precision=3, suppress_small=True)}"
            )
            lines.append(f"  EE Quat (xyzw): {np.array2string(state.end_effector_pose[3:], precision=3)}")
        if state.end_effector_euler is not None:
            lines.append(f"  EE Euler (xyz deg): {np.array2string(state.end_effector_euler, precision=3)}")
        return "\n".join(lines)

    def action_move_joints(self, joints: List[float], duration: Optional[float] = None):
        self._move_to_joints(np.array(joints, dtype=float), duration)

    def action_move_cartesian(
        self,
        pose: List[float],
        quat: Optional[List[float]] = None,
        euler: Optional[List[float]] = None,
        eular: Optional[List[float]] = None,
        duration: Optional[float] = None,
        **kwargs,
    ):
        if euler is None and eular is not None:
            euler = eular
        if quat is not None and euler is not None:
            print(f"[{self.robot_name}] Error: Cannot provide both 'quat' and 'euler'.")
            return

        target_pos_world = np.array(pose[:3], dtype=float)
        orientation_needed = False
        r_world_tool = R.identity()

        if quat is not None:
            orientation_needed = True
            r_world_tool = R.from_quat(quat)
        elif euler is not None:
            orientation_needed = True
            r_world_tool = R.from_euler("xyz", euler, degrees=True)

        r_world_link6 = r_world_tool * self.r_tool_link6
        r_base = R.from_quat([self.base_quat[1], self.base_quat[2], self.base_quat[3], self.base_quat[0]])
        p_local = r_base.inv().apply(target_pos_world - self.base_pos)
        r_local = r_base.inv() * r_world_link6
        target_pose_local = np.concatenate([p_local, r_local.as_quat()])

        current_joints = self._get_joint_positions()
        target_joints = self.kinematics.inverse_kinematics(
            target_pose_local,
            current_joints,
            orientation_needed=orientation_needed,
        )
        if target_joints is None:
            msg = f"[{self.robot_name}] CRITICAL: Target {target_pos_world} is unreachable. Action aborted."
            print(msg)
            self.log(msg)
            return

        self._move_to_joints(target_joints, duration)

    def action_move_linear(
        self,
        pose: List[float],
        quat: Optional[List[float]] = None,
        euler: Optional[List[float]] = None,
        eular: Optional[List[float]] = None,
        duration: Optional[float] = None,
        **kwargs,
    ):
        if euler is None and eular is not None:
            euler = eular
        if quat is not None and euler is not None:
            print(f"[{self.robot_name}] Error: Cannot provide both 'quat' and 'euler'.")
            return

        r_base = R.from_quat([self.base_quat[1], self.base_quat[2], self.base_quat[3], self.base_quat[0]])
        target_pos_world = np.array(pose[:3], dtype=float)

        if quat is not None:
            r_world_tool_end = R.from_quat(quat)
        elif euler is not None:
            r_world_tool_end = R.from_euler("xyz", euler, degrees=True)
        else:
            current_state = self.get_robot_state()
            r_world_tool_end = R.from_quat(current_state.end_effector_pose[3:])

        r_local_tool_end = r_base.inv() * r_world_tool_end
        target_pos_local = r_base.inv().apply(target_pos_world - self.base_pos)

        current_joints = self._get_joint_positions()
        ee_pose_local_start = self.kinematics.forward_kinematics(current_joints)
        start_pos_local = ee_pose_local_start[:3]
        r_local_link6_start = R.from_quat(ee_pose_local_start[3:])
        r_local_tool_start = r_local_link6_start * self.r_link6_tool

        if duration is None:
            dist = np.linalg.norm(target_pos_local - start_pos_local)
            duration = max(dist / 0.1, 0.5)

        total_steps = max(int(duration / self.control_dt), 1)
        key_rots = R.from_quat([r_local_tool_start.as_quat(), r_local_tool_end.as_quat()])
        slerp_func = Slerp([0, 1], key_rots)

        r_local_link6_target = r_local_tool_end * self.r_tool_link6
        target_pose_local_end = np.concatenate([target_pos_local, r_local_link6_target.as_quat()])
        if self.kinematics.inverse_kinematics(
            target_pose_local_end,
            current_joints,
            orientation_needed=True,
        ) is None:
            msg = f"[{self.robot_name}] CRITICAL: Linear move endpoint unreachable. Action aborted."
            print(msg)
            self.log(msg)
            return

        last_joints = current_joints
        target_joints = current_joints
        ik_step_skip = 10
        sim_start_time = self.data.time

        for step in range(1, total_steps + 1):
            if self.emergency_stop_flag:
                break

            if step == 1 or step % ik_step_skip == 0 or step == total_steps:
                alpha = step / total_steps
                interp_pos_local = start_pos_local + alpha * (target_pos_local - start_pos_local)
                r_local_tool_interp = slerp_func(alpha)
                r_local_link6_interp = r_local_tool_interp * self.r_tool_link6
                target_pose_local = np.concatenate([interp_pos_local, r_local_link6_interp.as_quat()])
                sol = self.kinematics.inverse_kinematics(
                    target_pose_local,
                    last_joints,
                    orientation_needed=True,
                    max_iter=20,
                )
                if sol is None:
                    msg = f"[{self.robot_name}] CRITICAL: Linear path blocked at step {step}/{total_steps}."
                    print(msg)
                    self.log(msg)
                    return
                target_joints = sol
                last_joints = sol

            target_sim_time = sim_start_time + (step * self.control_dt)
            while self.data.time < target_sim_time:
                if self.emergency_stop_flag:
                    break
                time.sleep(0.001)

            self._update_ctrl(target_joints)

        self._wait_settle(target_joints, timeout=1.0)

    def action_home(self, duration: float = 3.0):
        self._move_to_joints(self.home_joints, duration)

    def _reset_to_home(self):
        self._set_joints_direct(self.home_joints)
        self._update_ctrl(self.home_joints)

    def _get_joint_positions(self) -> np.ndarray:
        return np.array([self.data.qpos[self.model.jnt_qposadr[j_id]] for j_id in self.kinematics.joint_ids])

    def _set_joints_direct(self, positions: np.ndarray):
        for i, j_id in enumerate(self.kinematics.joint_ids):
            self.data.qpos[self.model.jnt_qposadr[j_id]] = positions[i]

    def _update_ctrl(self, positions: np.ndarray):
        for i, ctrl_id in enumerate(self.kinematics.actuator_ids):
            if ctrl_id != -1:
                self.data.ctrl[ctrl_id] = positions[i]

    def _move_to_joints(self, target_joints: np.ndarray, duration: Optional[float]):
        if self.emergency_stop_flag:
            return
        if not self._check_limits(target_joints):
            print(f"[{self.robot_name}] Target exceeds joint limits!")
            return

        if duration is None:
            self._update_ctrl(target_joints)
            self._wait_settle(target_joints)
            return

        start_joints = self._get_joint_positions()
        steps = max(int(duration / self.control_dt), 1)
        sim_start_time = self.data.time

        for step in range(1, steps + 1):
            if self.emergency_stop_flag:
                break
            alpha = step / steps
            alpha_smooth = 3 * alpha**2 - 2 * alpha**3
            interp_joints = start_joints + alpha_smooth * (target_joints - start_joints)
            target_sim_time = sim_start_time + (step * self.control_dt)
            while self.data.time < target_sim_time:
                if self.emergency_stop_flag:
                    break
                time.sleep(0.001)
            self._update_ctrl(interp_joints)

        self._wait_settle(target_joints)

    def _check_limits(self, joints: np.ndarray) -> bool:
        return np.all(joints >= self.kinematics.limits[:, 0]) and np.all(joints <= self.kinematics.limits[:, 1])

    def _wait_settle(self, target_joints: np.ndarray, timeout: float = 2.0):
        sim_start_time = self.data.time
        while self.data.time - sim_start_time < timeout:
            if np.allclose(self._get_joint_positions(), target_joints, atol=0.1):
                return
            time.sleep(0.001)


class MirobotKinematics:
    """IK/FK helpers isolated from the motion logic."""

    def __init__(self, model: mujoco.MjModel, urdf_path: str, prefix: str = ""):
        self.model = model
        p = f"{prefix}_" if prefix else ""
        self.joint_names = [f"{p}joint{i}" for i in range(1, 7)]
        self.ee_name = f"{p}Link6"
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        self.actuator_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.joint_names
        ]

        root = ET.parse(urdf_path).getroot()
        joint_count = sum(1 for joint in root.findall("joint") if joint.get("type") != "fixed")
        active_links_mask = [False] + [True] * joint_count
        self.chain = ikpy.chain.Chain.from_urdf_file(urdf_path, active_links_mask=active_links_mask)
        self.link_count = len(self.chain.links)
        self.limits = np.array([self.chain.links[i].bounds for i in range(1, self.link_count)], dtype=float)

        self.seeds = []
        self._init_seeds()

    def _full_joints(self, joints: np.ndarray) -> np.ndarray:
        full = np.zeros(self.link_count)
        full[1 : 1 + len(joints)] = joints
        return full

    def _init_seeds(self):
        seed_points = [
            [0.15, 0.0, 0.18],
            [0.10, 0.10, 0.15],
            [0.10, -0.10, 0.15],
        ]
        dummy_rot = np.eye(3)
        initial_guess = np.zeros(self.link_count)

        for pt in seed_points:
            target = np.eye(4)
            target[:3, 3] = pt
            target[:3, :3] = dummy_rot
            try:
                self.seeds.append(self.chain.inverse_kinematics_frame(target, initial_position=initial_guess))
            except Exception:
                continue

    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        mat = self.chain.forward_kinematics(self._full_joints(joints))
        pos = mat[:3, 3]
        quat = R.from_matrix(mat[:3, :3]).as_quat()
        return np.concatenate([pos, quat])

    def inverse_kinematics(
        self,
        target_pose: np.ndarray,
        current_joints: np.ndarray,
        orientation_needed: bool,
        max_iter: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        if len(target_pose) == 3:
            target_pose = np.concatenate([target_pose, [0, 0, 0, 1]])

        pos = target_pose[:3]
        rot = R.from_quat(target_pose[3:]).as_matrix()
        target_mat = np.eye(4)
        target_mat[:3, :3] = rot
        target_mat[:3, 3] = pos

        candidates_to_try = [self._full_joints(current_joints)]
        candidates_to_try.extend(self.seeds)
        candidates_to_try.append(np.zeros(self.link_count))

        best_sol = None
        best_error = float("inf")
        acceptance_tolerance = 0.01

        for initial_guess in candidates_to_try:
            try:
                ik_params = {
                    "target": target_mat,
                    "initial_position": initial_guess,
                    "orientation_mode": "all" if orientation_needed else None,
                }
                if max_iter is not None:
                    ik_params["max_iter"] = max_iter
                sol = self.chain.inverse_kinematics_frame(**ik_params)
                fk_mat = self.chain.forward_kinematics(sol)
                pos_error = np.linalg.norm(fk_mat[:3, 3] - pos)
                if pos_error < best_error:
                    best_error = pos_error
                    best_sol = sol
                if best_error < acceptance_tolerance:
                    break
            except Exception:
                continue

        if best_sol is None:
            return None
        return best_sol[1 : 1 + len(self.joint_names)]
