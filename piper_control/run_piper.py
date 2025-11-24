#!/usr/bin/env python3
"""
PiPER Robot Arm Command Line Tool

A comprehensive command line interface for controlling the AgileX PiPER robot arm
through JSON configuration files. Supports executing sequences of movements,
gripper operations, and other robot control actions.

Usage:
    python run_piper.py <config_file.json> [options]
"""

import argparse
import json
import sys
import time
import numpy as np
import os
import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from piper_controller import PiperController


class SceneBuilder:
    """
    Builds a custom MuJoCo scene XML based on configuration.
    Handles robot positioning and adding objects to the scene.
    """
    def __init__(self, base_dir: str = "agilex_piper"):
        self.base_dir = Path(base_dir)
        self.scene_xml = self.base_dir / "scene.xml"
        self.piper_xml = self.base_dir / "piper.xml"
        self.temp_files = []

    def build(self, scene_config: Dict[str, Any]) -> str:
        """
        Build scene and return path to temporary XML file.
        
        Args:
            scene_config: Dictionary containing scene configuration
                {
                    "robot_base": [x, y, z],
                    "friction": [sliding, torsional, rolling], # Optional global default
                    "solimp": [dmin, dmax, ...], # Optional global default
                    "solref": [timeconst, dampratio], # Optional global default
                    "objects": [...]
                }
        """
        run_id = str(uuid.uuid4())[:8]
        
        # Extract global defaults from scene config
        self.default_friction = scene_config.get("friction", "2.0 0.005 0.0001")
        if isinstance(self.default_friction, list):
            self.default_friction = " ".join(map(str, self.default_friction))
            
        self.default_solimp = scene_config.get("solimp", "0.9 0.95 0.001")
        if isinstance(self.default_solimp, list):
            self.default_solimp = " ".join(map(str, self.default_solimp))
            
        self.default_solref = scene_config.get("solref", "0.02 1")
        if isinstance(self.default_solref, list):
            self.default_solref = " ".join(map(str, self.default_solref))
        
        # 1. Modify piper.xml (Robot Base Position ONLY)
        piper_tree = ET.parse(self.piper_xml)
        piper_root = piper_tree.getroot()
        
        # Find base_link and set position
        robot_base_pos = scene_config.get("robot_base", [0, 0, 0])
        worldbody = piper_root.find("worldbody")
        if worldbody is not None:
            base_link = worldbody.find("./body[@name='base_link']")
            if base_link is not None:
                base_link.set("pos", f"{robot_base_pos[0]} {robot_base_pos[1]} {robot_base_pos[2]}")
        
        # Save temp piper xml
        temp_piper_name = f"temp_piper_{run_id}.xml"
        temp_piper_path = self.base_dir / temp_piper_name
        piper_tree.write(temp_piper_path)
        self.temp_files.append(temp_piper_path)
        
        # 2. Modify scene.xml (Add Objects & Link to temp piper)
        scene_tree = ET.parse(self.scene_xml)
        scene_root = scene_tree.getroot()
        
        # Update include to point to temp piper
        for include in scene_root.findall("include"):
            if include.get("file") == "piper.xml":
                include.set("file", temp_piper_name)
                break
        
        # Add objects to worldbody
        scene_worldbody = scene_root.find("worldbody")
        if scene_worldbody is None:
            scene_worldbody = ET.SubElement(scene_root, "worldbody")
            
        for obj in scene_config.get("objects", []):
            self._add_object(scene_worldbody, obj)
            
        # Save temp scene xml
        temp_scene_name = f"temp_scene_{run_id}.xml"
        temp_scene_path = self.base_dir / temp_scene_name
        scene_tree.write(temp_scene_path)
        self.temp_files.append(temp_scene_path)
        
        return str(temp_scene_path)

    def _add_object(self, worldbody: ET.Element, obj_config: Dict[str, Any]):
        """Add an object body to the worldbody"""
        name = obj_config.get("name", f"obj_{uuid.uuid4().hex[:6]}")
        pos = obj_config.get("pos", [0.5, 0, 0.1])
        
        body = ET.SubElement(worldbody, "body", name=name)
        body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
        
        # Optional orientation
        if "quat" in obj_config:
            quat = obj_config["quat"]
            body.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")
        elif "euler" in obj_config:
             euler = obj_config["euler"]
             body.set("euler", f"{euler[0]} {euler[1]} {euler[2]}")

        # Free joint if specified (movable object)
        if obj_config.get("movable", False):
            ET.SubElement(body, "freejoint")

        # Geometry
        geom = ET.SubElement(body, "geom")
        geom.set("type", obj_config.get("type", "box"))
        
        size = obj_config.get("size", [0.05, 0.05, 0.05])
        if isinstance(size, list):
            geom.set("size", " ".join(map(str, size)))
        else:
            geom.set("size", str(size))
            
        rgba = obj_config.get("rgba", [1, 0, 0, 1])
        geom.set("rgba", " ".join(map(str, rgba)))
        
        # Mass/Density
        if "mass" in obj_config:
             geom.set("mass", str(obj_config["mass"]))
        
        # Friction
        if "friction" in obj_config:
             geom.set("friction", " ".join(map(str, obj_config["friction"])))
        else:
             geom.set("friction", self.default_friction)
        
        # Solver parameters
        if "solimp" in obj_config:
             geom.set("solimp", " ".join(map(str, obj_config["solimp"])))
        else:
             geom.set("solimp", self.default_solimp)
             
        if "solref" in obj_config:
             geom.set("solref", " ".join(map(str, obj_config["solref"])))
        else:
             geom.set("solref", self.default_solref)


    def cleanup(self):
        """Remove temporary files"""
        for file_path in self.temp_files:
            try:
                if file_path.exists():
                    os.remove(file_path)
            except Exception as e:
                print(f"Warning: Failed to remove temp file {file_path}: {e}")


class PiperCommandLineController:
    """Command line interface for PiPER robot arm control"""
    
    def __init__(self, model_path: str = "agilex_piper/scene.xml", 
                 urdf_path: str = "agilex_piper/piper_description.urdf", 
                 headless: bool = False):
        """
        Initialize the command line controller
        
        Args:
            model_path: Path to MuJoCo model file
            urdf_path: Path to URDF file for kinematics
            headless: Run without visual viewer
        """
        self.controller = PiperController(model_path, urdf_path)
        self.headless = headless
        self.viewer = None
        
        # Action mapping
        self.action_map = {
            "move_joints": self._action_move_joints,
            "move_cartesian": self._action_move_cartesian,
            "open_gripper": self._action_open_gripper,
            "close_gripper": self._action_close_gripper,
            "home": self._action_home,
            "idle": self._action_idle,
            "print_state": self._action_print_state,
            "emergency_stop": self._action_emergency_stop,
            "reset_emergency_stop": self._action_reset_emergency_stop
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Parsed configuration dictionary
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print(f"Error: Configuration file '{config_path}' not found")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in configuration file: {e}")
            sys.exit(1)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check sequence
        if "sequence" not in config:
            print("Error: Configuration must contain 'sequence' key")
            return False
        
        if not isinstance(config["sequence"], list):
            print("Error: 'sequence' must be a list of actions")
            return False
        
        for i, action in enumerate(config["sequence"]):
            if not isinstance(action, dict):
                print(f"Error: Action {i} must be a dictionary")
                return False
            
            if "action" not in action:
                print(f"Error: Action {i} must have 'action' key")
                return False
            
            action_name = action["action"]
            if action_name not in self.action_map:
                print(f"Error: Unknown action '{action_name}' in action {i}")
                return False
        
        # Check scene config (optional but if present must be dict)
        if "scene" in config:
            if not isinstance(config["scene"], dict):
                print("Error: 'scene' must be a dictionary")
                return False
                
        return True
    
    def execute_config(self, config: Dict[str, Any]):
        """
        Execute the configuration sequence
        
        Args:
            config: Configuration dictionary
        """
        # Initialize viewer if not headless
        if not self.headless:
            print("Launching MuJoCo viewer...")
            self.viewer = mujoco.viewer.launch_passive(
                self.controller.model, 
                self.controller.data
            ).__enter__()
        
        try:
            print(f"\n--- Executing sequence with {len(config['sequence'])} actions ---")
            
            # Execute main sequence
            for i, action in enumerate(config["sequence"]):
                if self.controller.emergency_stop_flag:
                    print("Emergency stop active - stopping sequence execution")
                    break
                    
                self._execute_single_action(action, i, "action")
            
            print("\n--- Sequence execution completed ---")
            
            # Keep viewer open if not headless
            if not self.headless and self.viewer:
                print("Viewer will remain open. Close the window to exit.")
                while self.viewer.is_running():
                    time.sleep(0.1)
        
        finally:
            if self.viewer:
                self.viewer.__exit__(None, None, None)
    
    def _execute_single_action(self, action: Dict[str, Any], index: int, action_type: str = "action"):
        """Execute a single action"""
        action_name = action["action"]
        print(f"\n{action_type.capitalize()} {index + 1}: {action_name}")
        
        # Add description if provided
        if "description" in action:
            print(f"  Description: {action['description']}")
        
        # Execute the action
        try:
            success = self.action_map[action_name](action)
            if not success:
                print(f"  Warning: {action_type.capitalize()} {index + 1} did not complete successfully")
        except Exception as e:
            print(f"  Error executing {action_type} {index + 1}: {e}")
    
    # Action implementations
    def _action_move_joints(self, action: Dict[str, Any]) -> bool:
        """Execute move_joints action"""
        params = action.get("parameters", {})
        target_joints = np.array(params.get("joints", None))
        if target_joints is None:
            raise ValueError("Target joint positions must be specified")
        
        duration = params.get("duration", None)
        
        print(f"  Moving to joint positions: {target_joints}")
        if duration:
            print(f"  Duration: {duration}s")
        
        return self.controller.move_to_joint_positions(
            target_joints, duration, self.viewer
        )
    
    def _action_move_cartesian(self, action: Dict[str, Any]) -> bool:
        """Execute move_cartesian action"""
        params = action.get("parameters", {})
        target_pose = np.array(params.get("pose", None))
        if target_pose is None:
            raise ValueError("Target Cartesian pose must be specified")
        
        duration = params.get("duration", None)
        orientation_needed = params.get("orientation_needed", False)
        
        print(f"  Moving to Cartesian pose: {target_pose}")
        if duration:
            print(f"  Duration: {duration}s")
        print(f"  Orientation control: {orientation_needed}")
        
        return self.controller.move_to_cartesian_pose(
            target_pose, duration, orientation_needed, self.viewer
        )

    def _action_open_gripper(self, action: Dict[str, Any]) -> bool:
        """Execute open_gripper action"""
        print("  Opening gripper")
        self.controller.open_gripper(self.viewer)
        return True
    
    def _action_close_gripper(self, action: Dict[str, Any]) -> bool:
        """Execute close_gripper action"""
        print("  Closing gripper")
        self.controller.close_gripper(self.viewer)
        return True
    
    def _action_home(self, action: Dict[str, Any]) -> bool:
        """Execute home action"""
        params = action.get("parameters", {})
        duration = params.get("duration", 3.0)
        
        print(f"  Moving to home position (duration: {duration}s)")
        return self.controller.move_to_joint_positions(
            self.controller.home_joints, duration, self.viewer
        )
    
    def _action_idle(self, action: Dict[str, Any]) -> bool:
        """Execute idle action"""
        params = action.get("parameters", {})
        duration = params.get("duration", 1.0)
        
        print(f"  Idling for {duration}s")
        self.controller.idle(duration, self.viewer)
        return True
    
    def _action_print_state(self, action: Dict[str, Any]) -> bool:
        """Execute print_state action"""
        state = self.controller.get_robot_state()
        print("  Current robot state:")
        print(f"    Joint positions: {state.joint_positions}")
        print(f"    End-effector pose: {state.end_effector_pose}")
        print(f"    Gripper position: {state.gripper_position}")
        return True
    
    def _action_emergency_stop(self, action: Dict[str, Any]) -> bool:
        """Execute emergency_stop action"""
        print("  Activating emergency stop")
        self.controller.emergency_stop()
        return True
    
    def _action_reset_emergency_stop(self, action: Dict[str, Any]) -> bool:
        """Execute reset_emergency_stop action"""
        print("  Resetting emergency stop")
        self.controller.reset_emergency_stop()
        return True



def main():
    """Main entry point for the command line tool"""
    parser = argparse.ArgumentParser(
        description="PiPER Robot Arm Command Line Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_piper.py config.json                    # Run with viewer
  python run_piper.py config.json --headless         # Run without viewer
        """
    )
    
    parser.add_argument(
        "config_file",
        nargs='?',
        help="Path to JSON configuration file"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without visual viewer"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the configuration file without executing"
    )
    
    args = parser.parse_args()
    
    # Require config file
    if not args.config_file:
        parser.error("Configuration file is required")
    
    # Check if config file exists
    if not Path(args.config_file).exists():
        print(f"Error: Configuration file '{args.config_file}' does not exist")
        sys.exit(1)
    
    scene_builder = None
    
    try:
        # Load config first (we need it to build the scene)
        # We use a temporary controller instance just to load config
        # or we can just use json.load directly here.
        # Let's just use json.load to avoid instantiation overhead
        print(f"\nLoading configuration from: {args.config_file}")
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Print config info
        if "description" in config:
            print(f"Description: {config['description']}")

        # Build scene if configured
        model_path = "agilex_piper/scene.xml"
        if "scene" in config:
            print("Building custom scene from configuration...")
            scene_builder = SceneBuilder()
            model_path = scene_builder.build(config["scene"])
            print(f"Temporary scene generated at: {model_path}")

        # Initialize controller with the (possibly custom) scene
        print(f"Initializing PiPER controller...")
        print(f"  Headless mode: {args.headless}")
        
        controller = PiperCommandLineController(
            model_path=model_path,
            headless=args.headless
        )
        
        # Validate configuration
        print("Validating configuration...")
        if not controller.validate_config(config):
            sys.exit(1)
        
        total_actions = len(config.get("sequence", []))
        print(f"Configuration valid: {total_actions} actions")
        
        # If validation only, exit here
        if args.validate_only:
            print("Configuration validation completed successfully")
            return
        
        # Execute configuration
        controller.execute_config(config)
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        # import traceback
        # traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup temp files
        if scene_builder:
            print("\nCleaning up temporary files...")
            scene_builder.cleanup()


if __name__ == "__main__":
    main()
