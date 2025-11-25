#!/usr/bin/env python3
"""
Universal Multi-Robot Simulation Runner.
Controls various robots (Piper, Wheel, etc.) in a shared MuJoCo environment.
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
from pathlib import Path
from typing import Dict, Any, List, Set
import numpy as np

# --- Robot Registry ---
from common.robot_api import BaseRobotController
from piper_control.piper_controller import PiperController
from wheel_control.wheel_controller import WheelController

ROBOT_CLASSES = {
    "piper": PiperController,
    "rbtheron": WheelController,
}

# Optional: Only when controller needs URDF files
ROBOT_URDFS = {
    "piper": "piper_control/agilex_piper/piper_description.urdf",
}

ROBOT_XML_TEMPLATES = {
    "piper": "piper_control/agilex_piper/piper.xml",
    "rbtheron": "wheel_control/rbtheron/rbtheron.xml",
}

# --- Scene Builder ---

class SceneBuilder:
    """Merges robot XMLs into a single scene."""
    def __init__(self):
        self.base_scene_path = "piper_control/agilex_piper/scene.xml"
        self.temp_files = []
        self.included_assets: Set[str] = set() # Track asset names to avoid dups
        self.included_defaults: Set[str] = set() # Track default classes

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
            robots_conf = [{
                "name": "piper", 
                "type": "piper", 
                "base_pos": config.get("scene", {}).get("robot_base", [0,0,0]),
                "sequence": config["sequence"]
            }]

        for robot in robots_conf:
            r_type = robot.get("type", "piper")
            r_name = robot.get("name", "robot")
            base_pos = robot.get("base_pos", [0,0,0])

            if r_type not in ROBOT_XML_TEMPLATES:
                print(f"Warning: Unknown robot type '{r_type}', skipping.")
                continue

            self._merge_robot_xml(
                ROBOT_XML_TEMPLATES[r_type], 
                r_name, 
                base_pos, 
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

    def _merge_robot_xml(self, xml_path, name_prefix, pos, scene_root):
        """Deep merge of robot XML components."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Resolve asset directory relative to CWD
        xml_dir = os.path.dirname(xml_path)
        compiler = root.find("compiler")
        mesh_dir = ""
        if compiler is not None and "meshdir" in compiler.attrib:
            mesh_dir = compiler.get("meshdir")
        
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

        # 1. Assets (Mesh/Material) - Deduplicate by name
        target_asset = scene_root.find("asset")
        src_asset = root.find("asset")
        if src_asset is not None:
            for item in src_asset:
                # Handle Mesh Paths
                if item.tag == "mesh":
                    file_path = item.get("file")
                    if file_path and not os.path.isabs(file_path):
                        # Update to full relative path
                        new_path = os.path.join(asset_base_path, file_path)
                        # Normalize path separators
                        new_path = new_path.replace('\\', '/')
                        item.set("file", new_path)

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
                    # Base default without class, maybe merge children? 
                    # For now, append if simple
                    target_default.append(self._copy_elem(item))

        # 3. Worldbody
        target_wb = scene_root.find("worldbody")
        src_wb = root.find("worldbody")
        if src_wb is not None:
            for body in src_wb.findall("body"):
                new_body = self._copy_elem(body)
                rename_attrs(new_body) # Apply prefix to body/joint names
                new_body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
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

    def _copy_elem(self, elem):
        import copy
        return copy.deepcopy(elem)

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

    def run(self):
        self.running = True
        print(f"[{self.controller.robot_name}] Started.")
        for step in self.sequence:
            if not self.running: break
            self.controller.execute_action(step)
        print(f"[{self.controller.robot_name}] Finished.")
        self.running = False

def main():
    parser = argparse.ArgumentParser(description="Universal Robot Runner")
    parser.add_argument("config", help="Path to JSON config")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    builder = SceneBuilder()
    scene_path = builder.build(config)
    print(f"Generated Scene: {scene_path}")

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
            
            # Pass base_pos to controller for coordinate transform
            base_pos = np.array(r_conf.get("base_pos", [0,0,0]))
            
            if urdf_path:
                ctrl = ControllerClass(
                    model, 
                    data, 
                    robot_name=r_name, 
                    urdf_path=urdf_path,
                    base_pos=base_pos
                )
            else:
                ctrl = ControllerClass(
                    model,
                    data,
                    robot_name=r_name,
                    base_pos=base_pos
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
