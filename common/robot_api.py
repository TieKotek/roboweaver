import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
import mujoco

@dataclass
class RobotState:
    """Generic Robot state. Specific robots should subclass this."""
    timestamp: float

class BaseRobotController(ABC):
    """
    The Base Contract for ALL robots (Piper, Wheel, Drone, etc.).
    
    It handles:
    1. Common properties (model, data, name)
    2. Action Dispatching (converting JSON "action": "name" -> method calls)
    3. Universal safety features (E-Stop)
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, robot_name: str, base_pos: np.ndarray = None, base_quat: np.ndarray = None, log_dir: str = None):
        self.model = model
        self.data = data
        self.robot_name = robot_name
        self.base_pos = base_pos if base_pos is not None else np.zeros(3)
        self.base_quat = base_quat if base_quat is not None else np.array([1, 0, 0, 0]) # w, x, y, z
        self.emergency_stop_flag = False
        self.control_dt = 0.01
        
        self.log_file = None
        if log_dir:
            import os
            os.makedirs(log_dir, exist_ok=True)
            self.log_file_path = os.path.join(log_dir, f"{robot_name}.log")
            try:
                self.log_file = open(self.log_file_path, "a", encoding='utf-8')
            except Exception as e:
                print(f"[{self.robot_name}] Error opening log file: {e}")

    def log(self, message: str):
        """Write message to log file if enabled."""
        if self.log_file:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self.log_file.write(f"[{timestamp}] {message}\n")
            self.log_file.flush()

    def execute_action(self, action_config: Dict[str, Any], print_state: bool = False):
        """
        Automatic Action Dispatcher.
        Input: {"action": "move_joints", "parameters": {...}}
        Output: Calls self.action_move_joints(**parameters)
        """
        action_name = action_config.get("action")
        if not action_name:
            err_msg = f"[{self.robot_name}] Error: Action config missing 'action' key"
            print(err_msg)
            self.log(err_msg)
            return

        # Dynamic method lookup: action_xxx
        handler_name = f"action_{action_name}"
        
        if hasattr(self, handler_name):
            description = action_config.get("description", "")
            desc_str = f" ({description})" if description else ""
            
            start_msg = f"[{self.robot_name}] Executing {action_name}{desc_str}..."
            print(start_msg)
            self.log(start_msg)
            
            handler = getattr(self, handler_name)
            params = action_config.get("parameters", {})
            try:
                handler(**params)
                if print_state:
                    # Log to file ONLY
                    self.log(f"[{self.robot_name}] {action_name} completed.")
                    self.log(self.format_state())
                
            except Exception as e:
                 err_msg = f"[{self.robot_name}] Error executing {action_name}: {e}"
                 print(err_msg)
                 self.log(err_msg)
        else:
            err_msg = f"[{self.robot_name}] Error: Action '{action_name}' not implemented for this robot type."
            print(err_msg)
            self.log(err_msg)

    # --- Universal Safety & Utilities ---

    def action_idle(self, duration: float = 1.0):
        """Universal idle."""
        if self.emergency_stop_flag: return
        # print(f"[{self.robot_name}] Idling for {duration}s...")
        time.sleep(duration)

    def action_emergency_stop(self):
        """Universal E-Stop."""
        self.emergency_stop_flag = True
        msg = f"[{self.robot_name}] EMERGENCY STOP ACTIVATED"
        print(msg)
        self.log(msg)
        
    def action_reset_emergency_stop(self):
        self.emergency_stop_flag = False
        msg = f"[{self.robot_name}] E-Stop Reset"
        print(msg)
        self.log(msg)

    def action_print_state(self):
        self.print_state()

    def format_state(self) -> str:
        """
        Return a formatted string of the robot state.
        Default implementation with 2-decimal precision.
        """
        state = self.get_robot_state()
        lines = [f"[{self.robot_name}] State:"]
        fmt = {"float_kind": lambda x: f"{x:.2f}"}
        
        for field, value in state.__dict__.items():
            if value is None:
                continue
            if isinstance(value, np.ndarray):
                lines.append(f"  {field}: {np.array2string(value, formatter=fmt)}")
            else:
                lines.append(f"  {field}: {value}")
        return "\n".join(lines)

    def print_state(self):
        print(self.format_state())

    @abstractmethod
    def get_robot_state(self) -> RobotState:
        pass

    def __del__(self):
        if self.log_file:
            self.log_file.close()
