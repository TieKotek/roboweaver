import time
from dataclasses import dataclass
import numpy as np
import mujoco

from common.robot_api import BaseRobotController, RobotState


@dataclass
class ConveyorState(RobotState):
    roller_speed: float = 0.0
    running: bool = False


class ConveyorController(BaseRobotController):
    """Controller for a belt conveyor driven by rollers."""

    DEFAULT_LENGTH = 1.04
    BELT_LINEAR_SCALE = 0.0048

    def __init__(self, model, data, robot_name, base_pos=None, base_quat=None, log_dir=None, **kwargs):
        super().__init__(model, data, robot_name, base_pos, base_quat, log_dir)
        self.base_body_name = f"{robot_name}_base_link"
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.base_body_name)
        self.roller_actuator_ids = []
        self.segment_joint_ids = []
        self.segment_qpos_adrs = []
        self.segment_qvel_adrs = []
        self.segment_offsets = []
        self.belt_phase = 0.0
        self.current_speed = 0.0
        self.control_dt = 0.01
        self.length = float(kwargs.get("length", self.DEFAULT_LENGTH))
        self.belt_travel = max(self.length - 0.08, 0.12)

        for idx in range(256):
            actuator_name = f"{robot_name}_roller_{idx:02d}_drive"
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id < 0:
                break
            self.roller_actuator_ids.append(actuator_id)

        for idx in range(512):
            joint_name = f"{robot_name}_belt_segment_{idx:02d}_joint"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                break
            self.segment_joint_ids.append(joint_id)
            self.segment_qpos_adrs.append(model.jnt_qposadr[joint_id])
            self.segment_qvel_adrs.append(model.jnt_dofadr[joint_id])

        if self.segment_qpos_adrs:
            self.segment_offsets = [float(self.data.qpos[adr]) for adr in self.segment_qpos_adrs]

        if not self.roller_actuator_ids:
            print(f"[{self.robot_name}] Warning: no conveyor actuators found.")
        if not self.segment_joint_ids:
            print(f"[{self.robot_name}] Warning: no belt segments found.")

    def get_robot_state(self) -> ConveyorState:
        return ConveyorState(
            timestamp=self.data.time,
            roller_speed=self.current_speed,
            running=abs(self.current_speed) > 1e-6,
        )

    def _set_roller_speed(self, speed: float):
        self.current_speed = float(speed)
        for actuator_id in self.roller_actuator_ids:
            self.data.ctrl[actuator_id] = self.current_speed

    def _wrap_position(self, value: float) -> float:
        half = self.belt_travel / 2.0
        while value > half:
            value -= self.belt_travel
        while value < -half:
            value += self.belt_travel
        return value

    def _update_belt_segments(self):
        if not self.segment_qpos_adrs:
            return

        displacement = self.belt_phase
        segment_velocity = self.current_speed * self.BELT_LINEAR_SCALE

        for offset, qpos_adr, qvel_adr in zip(self.segment_offsets, self.segment_qpos_adrs, self.segment_qvel_adrs):
            target = self._wrap_position(offset + displacement)
            self.data.qpos[qpos_adr] = target
            self.data.qvel[qvel_adr] = segment_velocity

    def _advance_phase(self):
        self.belt_phase = self._wrap_position(
            self.belt_phase + self.current_speed * self.BELT_LINEAR_SCALE * self.control_dt
        )

    def _hold_speed(self, duration: float):
        start_sim_time = self.data.time
        step_count = 0
        while self.data.time - start_sim_time < duration:
            if self.emergency_stop_flag:
                self._set_roller_speed(0.0)
                return
            self._set_roller_speed(self.current_speed)
            self._advance_phase()
            self._update_belt_segments()
            step_count += 1
            target_time = start_sim_time + step_count * self.control_dt
            while self.data.time < target_time:
                if self.emergency_stop_flag:
                    self._set_roller_speed(0.0)
                    return
                time.sleep(0.001)

    def _hold_position(self, duration: float):
        start_sim_time = self.data.time
        step_count = 0
        while self.data.time - start_sim_time < duration:
            if self.emergency_stop_flag:
                self._set_roller_speed(0.0)
                return
            self._set_roller_speed(0.0)
            self._update_belt_segments()
            step_count += 1
            target_time = start_sim_time + step_count * self.control_dt
            while self.data.time < target_time:
                if self.emergency_stop_flag:
                    self._set_roller_speed(0.0)
                    return
                time.sleep(0.001)

    def action_run(self, speed: float = 8.0, duration: float = 1.0):
        if self.emergency_stop_flag:
            return
        self._set_roller_speed(speed)
        self._hold_speed(duration)
        if not self.emergency_stop_flag:
            # A timed run should end in a true stop state instead of leaving the
            # belt segments with residual commanded velocity until another action
            # arrives. This keeps cargo from gliding after the conveyor stops.
            self._set_roller_speed(0.0)
            self._update_belt_segments()
            self._hold_position(duration=0.3)

    def action_idle(self, duration: float = 1.0):
        self._set_roller_speed(0.0)
        self._update_belt_segments()
        self._hold_position(duration=duration)

    def action_emergency_stop(self):
        self._set_roller_speed(0.0)
        self._update_belt_segments()
        super().action_emergency_stop()
