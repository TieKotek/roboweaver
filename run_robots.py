#!/usr/bin/env python3
"""
RoboWeaver: Heterogeneous Multi-Robot Simulation Framework.
Dynamically weaves multiple robots (Piper, Stretch, Drones, etc.) into a 
unified MuJoCo environment via JSON configuration.
"""

import argparse
import json
import time
import threading
import mujoco
import mujoco.viewer
import os
import xml.etree.ElementTree as ET
import uuid
import numpy as np
import math
from typing import Dict, Any, List, Set

# --- Robot Registry ---
from common.robot_api import BaseRobotController
from robots.piper_control.piper_controller import PiperController
from robots.rbtheron_control.rbtheron_controller import RbtheronController
from robots.tracer_control.tracer_controller import TracerController
from robots.stretch_control.stretch_controller import StretchController
from robots.skydio_control.skydio_controller import SkydioController
from robots.conveyor_control.conveyor_controller import ConveyorController
from robots.mirobot_control.mirobot_controller import MirobotController
from robots.franka_control.franka_controller import FrankaController

ROBOT_CLASSES = {
    "piper": PiperController,
    "rbtheron": RbtheronController,
    "tracer": TracerController,
    "stretch": StretchController,
    "skydio": SkydioController,
    "conveyor": ConveyorController,
    "mirobot": MirobotController,
    "franka": FrankaController,
}

# Optional: Only when controller needs URDF files
ROBOT_URDFS = {
    "piper": "robots/piper_control/agilex_piper/piper_description.urdf",
    "mirobot": "robots/mirobot_control/mirobot.urdf",
    "franka": "robots/franka_control/franka_panda.urdf",
}

ROBOT_XML_TEMPLATES = {
    "piper": "robots/piper_control/agilex_piper/piper.xml",
    "rbtheron": "robots/rbtheron_control/rbtheron/rbtheron.xml",
    "tracer": "robots/tracer_control/agilex_tracer2/tracer2.xml",
    "stretch": "robots/stretch_control/hello_robot_stretch_3/stretch.xml",
    "skydio": "robots/skydio_control/skydio_x2/x2.xml",
    "conveyor": "robots/conveyor_control/conveyor/conveyor.xml",
    "mirobot": "robots/mirobot_control/mirobot.xml",
    "franka": "robots/franka_control/franka_emika_panda/panda.xml",
}

DEFAULT_CONVEYOR_LENGTH = 1.04
DEFAULT_CONVEYOR_WIDTH = 0.32
DEFAULT_CONVEYOR_HEIGHT = 0.144
DEFAULT_MOVABLE_OBJECT_CONTYPE = 2
DEFAULT_MOVABLE_OBJECT_CONAFFINITY = 7

# --- Helper Functions ---

def yaw_to_quat(yaw_deg: float) -> List[float]:
    """Convert yaw angle (degrees) to quaternion [w, x, y, z]."""
    yaw_rad = np.deg2rad(yaw_deg)
    w = np.cos(yaw_rad / 2.0)
    z = np.sin(yaw_rad / 2.0)
    return [w, 0.0, 0.0, z]

# --- Scene Builder ---

class SceneBuilder:
    """Merges robot XMLs into a single scene."""
    def __init__(self):
        self.base_scene_path = "robots/piper_control/agilex_piper/scene.xml"
        self.temp_files = []
        self.included_assets: Set[str] = set() # Track asset names to avoid dups
        self.included_defaults: Set[str] = set() # Track default classes
        self.included_default_base_tags: Set[str] = set() # Track base default children like geom/joint

    def build(self, config: Dict[str, Any]) -> str:
        run_id = str(uuid.uuid4())[:8]
        
        # Extract global defaults
        scene_conf = config.get("scene", {})
        self.default_friction = scene_conf.get("friction", "2.0 0.005 0.0001")
        if isinstance(self.default_friction, list):
            self.default_friction = " ".join(map(str, self.default_friction))

        self.default_solimp = scene_conf.get("solimp", "0.9 0.95 0.001")
        if isinstance(self.default_solimp, list):
            self.default_solimp = " ".join(map(str, self.default_solimp))
            
        self.default_solref = scene_conf.get("solref", "0.02 1")
        if isinstance(self.default_solref, list):
            self.default_solref = " ".join(map(str, self.default_solref))

        # 1. Load Base Scene
        scene_tree = ET.parse(self.base_scene_path)
        scene_root = scene_tree.getroot()
        
        # Ensure compiler exists and set angle to radian
        compiler = scene_root.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(scene_root, "compiler")
        compiler.set("angle", "radian")
        
        # Set physics options for stability
        option = scene_root.find("option")
        if option is None:
            option = ET.SubElement(scene_root, "option")
        option.set("integrator", "implicitfast")
        option.set("cone", "elliptic")
        option.set("impratio", "100") # Higher impratio for stable grasping
        
        # New: Allow override timestep from JSON
        ts_val = scene_conf.get("timestep", 0.002)
        option.set("timestep", str(ts_val))

        # Ensure sections exist
        for sec in ["worldbody", "actuator", "sensor", "contact", "equality", "asset", "default"]:
            if scene_root.find(sec) is None:
                ET.SubElement(scene_root, sec)

        # Clear old includes to avoid conflicts
        for include in scene_root.findall("include"):
            scene_root.remove(include)

        # 2. Add Robots
        robots_conf = config.get("robots", [])
        if not robots_conf and "sequence" in config:
            robots_conf = [
                {
                    "name": "piper", 
                    "type": "piper", 
                    "base_pos": config.get("scene", {}).get("robot_base", [0,0,0]),
                    "base_yaw": config.get("scene", {}).get("robot_yaw", 0.0),
                    "sequence": config["sequence"]
                }
            ]

        for robot in robots_conf:
            r_type = robot.get("type", "piper")
            r_name = robot.get("name", "robot")
            base_pos = robot.get("base_pos", [0,0,0])
            
            if "base_quat" in robot:
                base_quat = robot["base_quat"]
            else:
                base_yaw = robot.get("base_yaw", 0.0)
                # Calculate quaternion from yaw
                base_quat = yaw_to_quat(base_yaw)

            if r_type not in ROBOT_XML_TEMPLATES:
                print(f"Warning: Unknown robot type '{r_type}', skipping.")
                continue

            robot_xml_path = ROBOT_XML_TEMPLATES[r_type]
            if r_type == "conveyor":
                robot_xml_path = self._build_conveyor_xml(robot)

            self._merge_robot_xml(
                robot_xml_path,
                r_name, 
                base_pos,
                base_quat,
                scene_root
            )

        # 3. Add Custom Objects
        worldbody = scene_root.find("worldbody")
        for obj in config.get("scene", {}).get("objects", []):
            self._add_object(worldbody, obj)

        # Save
        temp_path = f"temp_scene_{run_id}.xml"
        scene_tree.write(temp_path)
        self.temp_files.append(temp_path)
        return temp_path

    def _merge_robot_xml(self, xml_path, name_prefix, pos, quat, scene_root):
        """Deep merge of robot XML components."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Resolve asset directory relative to CWD
        xml_dir = os.path.dirname(xml_path)
        compiler = root.find("compiler")
        mesh_dir = ""
        if compiler is not None:
            # Support both old 'meshdir' and new 'assetdir'
            mesh_dir = compiler.get("assetdir") or compiler.get("meshdir") or ""
        
        # The base path for assets from this XML
        asset_base_path = os.path.join(xml_dir, mesh_dir)

        def prefix_str(s): return f"{name_prefix}_{s}" if s else s

        # Helper to recursively rename attributes
        def rename_attrs(elem, ignore_class=False):
            # Don't rename class if it refers to a default class (unless we are inside a default definition)
            if 'name' in elem.attrib:
                elem.set('name', prefix_str(elem.get('name')))
            
            # References to named entities must be prefixed
            for ref in ['joint', 'joint1', 'joint2', 'body1', 'body2', 'actuator', 'tendon', 'site']:
                if ref in elem.attrib:
                    elem.set(ref, prefix_str(elem.get(ref)))
            
            # For children
            for child in elem:
                rename_attrs(child, ignore_class)

        # 1. Assets (Mesh/Material/Texture) - Deduplicate by name
        target_asset = scene_root.find("asset")
        src_asset = root.find("asset")
        if src_asset is not None:
            for item in src_asset:
                # Handle Paths for Meshes and Textures
                attr_to_fix = "file" if item.tag in ["mesh", "texture"] else None
                if attr_to_fix:
                    file_path = item.get(attr_to_fix)
                    if file_path and not os.path.isabs(file_path):
                        # Update to full relative path
                        new_path = os.path.join(asset_base_path, file_path)
                        # Normalize path separators
                        new_path = new_path.replace('\\', '/')
                        item.set(attr_to_fix, new_path)

                name = item.get("name") or item.get("file") 
                if name and name not in self.included_assets:
                    self.included_assets.add(name)
                    target_asset.append(self._copy_elem(item))

        # 2. Defaults - Deduplicate by class name
        target_default = scene_root.find("default")
        src_default = root.find("default")
        if src_default is not None:
            for item in src_default:
                # If it's a top-level default class (e.g. <default class="piper">)
                cls_name = item.get("class")
                if cls_name and cls_name not in self.included_defaults:
                    self.included_defaults.add(cls_name)
                    target_default.append(self._copy_elem(item))
                elif not cls_name: 
                    tag_key = item.tag
                    if tag_key not in self.included_default_base_tags:
                        self.included_default_base_tags.add(tag_key)
                        target_default.append(self._copy_elem(item))

        # 3. Worldbody
        target_wb = scene_root.find("worldbody")
        src_wb = root.find("worldbody")
        if src_wb is not None:
            for body in src_wb.findall("body"):
                new_body = self._copy_elem(body)
                rename_attrs(new_body) # Apply prefix to body/joint names
                new_body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
                
                # Set quaternion (orientation)
                new_body.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")
                
                target_wb.append(new_body)

        # 4. Actuators
        target_act = scene_root.find("actuator")
        src_act = root.find("actuator")
        if src_act is not None:
            for item in src_act:
                new_item = self._copy_elem(item)
                rename_attrs(new_item)
                target_act.append(new_item)

        # 5. Contact (Exclusions)
        target_contact = scene_root.find("contact")
        src_contact = root.find("contact")
        if src_contact is not None:
            for item in src_contact:
                new_item = self._copy_elem(item)
                rename_attrs(new_item)
                target_contact.append(new_item)

        # 6. Equality (Constraints)
        target_eq = scene_root.find("equality")
        src_eq = root.find("equality")
        if src_eq is not None:
            for item in src_eq:
                new_item = self._copy_elem(item)
                rename_attrs(new_item)
                target_eq.append(new_item)

        # 7. Tendons
        target_tendon = scene_root.find("tendon")
        if target_tendon is None and root.find("tendon") is not None:
            target_tendon = ET.SubElement(scene_root, "tendon")
        
        src_tendon = root.find("tendon")
        if src_tendon is not None:
            for item in src_tendon:
                new_item = self._copy_elem(item)
                rename_attrs(new_item)
                target_tendon.append(new_item)

        # 8. Keyframes
        target_key = scene_root.find("keyframe")
        if target_key is None and root.find("keyframe") is not None:
            target_key = ET.SubElement(scene_root, "keyframe")
        
        src_key = root.find("keyframe")
        if src_key is not None:
            for item in src_key:
                new_item = self._copy_elem(item)
                # Apply prefix to keyframe names to avoid conflicts
                rename_attrs(new_item)
                target_key.append(new_item)

    def _copy_elem(self, elem):
        import copy
        return copy.deepcopy(elem)

    def _build_conveyor_xml(self, robot_config: Dict[str, Any]) -> str:
        """Create a conveyor XML variant with configurable overall dimensions."""
        xml_path = ROBOT_XML_TEMPLATES["conveyor"]
        tree = ET.parse(xml_path)
        root = tree.getroot()

        length = float(robot_config.get("length", DEFAULT_CONVEYOR_LENGTH))
        width = float(robot_config.get("width", DEFAULT_CONVEYOR_WIDTH))
        height = float(robot_config.get("height", DEFAULT_CONVEYOR_HEIGHT))
        show_driver_layer = bool(robot_config.get("show_driver_layer", False))
        length = max(length, 0.40)
        width = max(width, 0.12)
        height = max(height, 0.06)

        frame_half = length / 2.0
        belt_half = max(frame_half - 0.03, 0.12)
        belt_surface_length = belt_half * 2.0
        width_scale = width / DEFAULT_CONVEYOR_WIDTH
        height_scale = height / DEFAULT_CONVEYOR_HEIGHT

        worldbody = root.find("worldbody")
        base_link = worldbody.find("body")

        geoms_by_name = {geom.get("name"): geom for geom in base_link.findall("geom")}
        geoms_by_name["frame_base"].set("pos", f"0 0 {0.03 * height_scale:.6f}")
        geoms_by_name["frame_base"].set("size", f"{frame_half:.6f} {0.16 * width_scale:.6f} {0.03 * height_scale:.6f}")
        geoms_by_name["frame_left"].set("pos", f"0 {0.13 * width_scale:.6f} {0.07 * height_scale:.6f}")
        geoms_by_name["frame_left"].set("size", f"{frame_half:.6f} {0.02 * width_scale:.6f} {0.04 * height_scale:.6f}")
        geoms_by_name["frame_right"].set("pos", f"0 {-0.13 * width_scale:.6f} {0.07 * height_scale:.6f}")
        geoms_by_name["frame_right"].set("size", f"{frame_half:.6f} {0.02 * width_scale:.6f} {0.04 * height_scale:.6f}")
        geoms_by_name["frame_front"].set("pos", f"{frame_half - 0.01:.6f} 0 {0.055 * height_scale:.6f}")
        geoms_by_name["frame_front"].set("size", f"0.015 {0.12 * width_scale:.6f} {0.015 * height_scale:.6f}")
        geoms_by_name["frame_rear"].set("pos", f"{-(frame_half - 0.01):.6f} 0 {0.055 * height_scale:.6f}")
        geoms_by_name["frame_rear"].set("size", f"0.015 {0.12 * width_scale:.6f} {0.015 * height_scale:.6f}")
        geoms_by_name["belt_base"].set("pos", f"0 0 {0.138 * height_scale:.6f}")
        geoms_by_name["belt_base"].set("size", f"{belt_half:.6f} {0.102 * width_scale:.6f} {0.006 * height_scale:.6f}")
        if show_driver_layer:
            geoms_by_name["belt_base"].set("rgba", "0.20 0.22 0.25 0.35")
        else:
            geoms_by_name["belt_base"].set("rgba", "0.14 0.15 0.17 1")

        roller_default = root.find("./default/default[@class='roller']/geom")
        belt_segment_default = root.find("./default/default[@class='belt_segment']/geom")
        roller_default.set("size", f"{0.028 * height_scale:.6f} {0.095 * width_scale:.6f}")
        segment_count = max(8, int(round(belt_surface_length / 0.03)))
        segment_spacing = belt_surface_length / segment_count
        segment_half_x = max(segment_spacing * 0.6, 0.01)
        belt_segment_default.set("size", f"{segment_half_x:.6f} {0.096 * width_scale:.6f} {0.006 * height_scale:.6f}")
        if show_driver_layer:
            belt_segment_default.set("rgba", "0.92 0.46 0.14 0.95")
            belt_segment_default.set("group", "0")
        else:
            belt_segment_default.set("rgba", "0.18 0.19 0.2 1")
            belt_segment_default.set("group", "3")

        for child in list(base_link):
            name = child.get("name", "")
            if name.startswith("belt_segment_") or name.startswith("roller_"):
                base_link.remove(child)

        actuator = root.find("actuator")
        for child in list(actuator):
            if child.get("name", "").startswith("roller_"):
                actuator.remove(child)

        first_segment_x = -belt_half + segment_spacing / 2.0
        for idx in range(segment_count):
            x = first_segment_x + idx * segment_spacing
            body = ET.SubElement(base_link, "body", name=f"belt_segment_{idx:02d}", pos=f"{x:.6f} 0 {0.142 * height_scale:.6f}")
            ET.SubElement(body, "joint", name=f"belt_segment_{idx:02d}_joint", **{"class": "belt_segment"})
            ET.SubElement(body, "geom", **{"class": "belt_segment"})

        roller_margin = 0.08
        roller_span = max(length - 2 * roller_margin, 0.0)
        roller_count = max(4, int(math.floor(roller_span / 0.08)) + 1)
        if roller_count == 1:
            roller_positions = [0.0]
        else:
            start = -roller_span / 2.0
            roller_positions = [start + i * (roller_span / (roller_count - 1)) for i in range(roller_count)]

        for idx, x in enumerate(roller_positions):
            body = ET.SubElement(base_link, "body", name=f"roller_{idx:02d}", pos=f"{x:.6f} 0 {0.105 * height_scale:.6f}")
            ET.SubElement(body, "joint", name=f"roller_{idx:02d}_joint", **{"class": "roller"})
            ET.SubElement(body, "geom", **{"class": "roller"})
            ET.SubElement(actuator, "velocity", name=f"roller_{idx:02d}_drive", joint=f"roller_{idx:02d}_joint", kv="4", ctrlrange="-12 12")

        run_id = str(uuid.uuid4())[:8]
        temp_path = f"temp_conveyor_{run_id}.xml"
        tree.write(temp_path)
        self.temp_files.append(temp_path)
        return temp_path

    def _add_object(self, worldbody, obj_config):
        name = obj_config.get("name", f"obj_{uuid.uuid4().hex[:6]}")
        body = ET.SubElement(worldbody, "body", name=name)
        body.set("pos", " ".join(map(str, obj_config.get("pos", [0,0,0]))))
        
        if "quat" in obj_config:
            body.set("quat", " ".join(map(str, obj_config["quat"])))

        if obj_config.get("movable", False):
            ET.SubElement(body, "freejoint")
            
        geom = ET.SubElement(body, "geom")
        geom.set("type", obj_config.get("type", "box"))
        geom.set("size", " ".join(map(str, obj_config.get("size", [0.05]*3))))
        geom.set("rgba", " ".join(map(str, obj_config.get("rgba", [1,0,0,1]))))
        if "mass" in obj_config: geom.set("mass", str(obj_config["mass"]))

        contype = obj_config.get("contype")
        conaffinity = obj_config.get("conaffinity")
        if obj_config.get("movable", False):
            if contype is None:
                contype = DEFAULT_MOVABLE_OBJECT_CONTYPE
            if conaffinity is None:
                conaffinity = DEFAULT_MOVABLE_OBJECT_CONAFFINITY
        if contype is not None: geom.set("contype", str(contype))
        if conaffinity is not None: geom.set("conaffinity", str(conaffinity))
        
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
        for f in self.temp_files:
            try: os.remove(f)
            except: pass

# --- Execution Engine ---

class RobotThread(threading.Thread):
    def __init__(self, controller: BaseRobotController, sequence: List[Dict]):
        super().__init__(daemon=True)
        self.controller = controller
        self.sequence = sequence
        self.running = False
        self.print_state = True

    def run(self):
        self.running = True
        print(f"[{self.controller.robot_name}] Started.")
        for step in self.sequence:
            if not self.running: break
            self.controller.execute_action(step, print_state=self.print_state)
        print(f"[{self.controller.robot_name}] Finished.")
        self.running = False

def main():
    parser = argparse.ArgumentParser(description="RoboWeaver: Heterogeneous Multi-Robot Runner")
    parser.add_argument("config", help="Path to JSON config")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    builder = SceneBuilder()
    scene_path = builder.build(config)
    print(f"Generated Scene: {scene_path}")

    # Create Log Directory
    log_timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"run_{log_timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")

    try:
        # 1. Init Physics
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        
        # 2. Init Controllers
        threads = []
        robots_conf = config.get("robots", [])
        
        if not robots_conf and "sequence" in config:
            raise ValueError("No robots defined in config.")

        for r_conf in robots_conf:
            r_type = r_conf.get("type", None)
            r_name = r_conf.get("name", "robot")
            
            if not r_type:
                print(f"Skipping robot with no type: {r_name}")
                continue
            
            if r_type not in ROBOT_CLASSES:
                print(f"Skipping unknown robot type: {r_type}")
                continue
                
            ControllerClass = ROBOT_CLASSES[r_type]
            urdf_path = ROBOT_URDFS.get(r_type, None)
            
            # Pass base_pos and base_quat to controller
            base_pos = np.array(r_conf.get("base_pos", [0,0,0]))
            
            if "base_quat" in r_conf:
                base_quat = np.array(r_conf["base_quat"])
            else:
                base_yaw = r_conf.get("base_yaw", 0.0)
                base_quat = np.array(yaw_to_quat(base_yaw))
            
            if urdf_path:
                ctrl = ControllerClass(
                    model, 
                    data, 
                    robot_name=r_name, 
                    urdf_path=urdf_path,
                    base_pos=base_pos,
                    base_quat=base_quat,
                    log_dir=log_dir
                )
            else:
                extra_kwargs = {}
                if r_type == "conveyor" and "length" in r_conf:
                    extra_kwargs["length"] = r_conf["length"]
                ctrl = ControllerClass(
                    model,
                    data,
                    robot_name=r_name,
                    base_pos=base_pos,
                    base_quat=base_quat,
                    log_dir=log_dir,
                    **extra_kwargs
                )
            
            t = RobotThread(ctrl, r_conf.get("sequence", []))
            threads.append(t)

        # 3. Run
        # Apply initial positions to kinematics
        mujoco.mj_forward(model, data)

        if not args.headless:
            viewer = mujoco.viewer.launch_passive(model, data)
        else:
            viewer = None

        for t in threads: t.start()

        print("Simulation running...")
        while True:
            start = time.time()
            
            mujoco.mj_step(model, data)
            
            if viewer:
                viewer.sync()
                if not viewer.is_running(): break
            
            if args.headless and not any(t.running for t in threads):
                break

            rem = model.opt.timestep - (time.time() - start)
            if rem > 0: time.sleep(rem)

    except KeyboardInterrupt:
        pass
    finally:
        builder.cleanup()
        if 'viewer' in locals() and viewer: viewer.close()

if __name__ == "__main__":
    main()
