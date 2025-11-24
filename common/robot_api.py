import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
import mujoco

@dataclass
class RobotState:
    """Generic Robot state."""
    timestamp: float
    # Subclasses can add more fields or use a dynamic dict
    joint_positions: Optional[np.ndarray] = None
    end_effector_pose: Optional[np.ndarray] = None

class BaseRobotController(ABC):
    """
    The Base Contract for ALL robots (Piper, Wheel, Drone, etc.).
    
    It handles:
    1. Common properties (model, data, name)
    2. Action Dispatching (converting JSON "action": "name" -> method calls)
    3. Universal safety features (E-Stop)
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, robot_name: str, base_pos: np.ndarray = None, base_quat: np.ndarray = None):
        self.model = model
        self.data = data
        self.robot_name = robot_name
        self.base_pos = base_pos if base_pos is not None else np.zeros(3)
        self.base_quat = base_quat if base_quat is not None else np.array([1, 0, 0, 0]) # w, x, y, z
        self.emergency_stop_flag = False
        self.control_dt = 0.01

    def execute_action(self, action_config: Dict[str, Any]):
        """
        Automatic Action Dispatcher.
        Input: {"action": "move_joints", "parameters": {...}}
        Output: Calls self.action_move_joints(**parameters)
        """
        action_name = action_config.get("action")
        if not action_name:
            print(f"[{self.robot_name}] Error: Action config missing 'action' key")
            return

        # Dynamic method lookup: action_xxx
        handler_name = f"action_{action_name}"
        
        if hasattr(self, handler_name):
            print(f"[{self.robot_name}] Executing {action_name}...")
            handler = getattr(self, handler_name)
            params = action_config.get("parameters", {})
            try:
                handler(**params)
            except Exception as e:
                 print(f"[{self.robot_name}] Error executing {action_name}: {e}")
        else:
            print(f"[{self.robot_name}] Error: Action '{action_name}' not implemented for this robot type.")

    # --- Universal Safety & Utilities ---

    def action_idle(self, duration: float = 1.0):
        """Universal idle."""
        if self.emergency_stop_flag: return
        # print(f"[{self.robot_name}] Idling for {duration}s...")
        time.sleep(duration)

    def action_emergency_stop(self):
        """Universal E-Stop."""
        self.emergency_stop_flag = True
        print(f"[{self.robot_name}] EMERGENCY STOP ACTIVATED")
        
    def action_reset_emergency_stop(self):
        self.emergency_stop_flag = False
        print(f"[{self.robot_name}] E-Stop Reset")

    def action_print_state(self):
        state = self.get_robot_state()
        print(f"[{self.robot_name}] State: {state}")

    @abstractmethod
    def get_robot_state(self) -> RobotState:
        pass
