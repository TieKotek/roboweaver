import mujoco

from common.differential_drive_controller import DifferentialDriveController


class RbtheronController(DifferentialDriveController):
    """Controller for the RB-Theron mobile base."""

    WHEEL_RADIUS = 0.0762
    WHEEL_TRACK = 0.5032
    SUPPORT_GEOM_NAMES = (
        "front_left_support",
        "front_right_support",
        "rear_left_support",
        "rear_right_support",
    )
    # User-facing caps stay conservative; runtime safe limits may clamp further
    # based on actuator capability * SAFETY_SPEED_FACTOR in the shared base class.
    DEFAULT_LINEAR_SPEED = 0.35
    MAX_LINEAR_SPEED = 0.45
    DEFAULT_ANGULAR_SPEED_DEG = 30.0
    MAX_ANGULAR_SPEED_DEG = 45.0
    LINEAR_ACCEL = 0.25
    ANGULAR_ACCEL_DEG = 60.0
    HEADING_KP = 5.0
    TURN_KP = 10.0
    LINEAR_DISTANCE_TOLERANCE = 0.025
    HEADING_TOLERANCE_DEG = 2.5
    MAX_COMPLETION_OVERRUN = 4.0
    WHEEL_RADIUS_TOLERANCE = 0.001
    WHEEL_TRACK_TOLERANCE = 0.005
    SUPPORT_CLEARANCE_MIN = -0.0002
    CHASSIS_SUPPORT_CLEARANCE_MIN = 0.01

    def __init__(
        self,
        model,
        data,
        robot_name,
        base_pos=None,
        base_quat=None,
        log_dir=None,
        **kwargs,
    ):
        self.left_actuator_name = f"{robot_name}_left_wheel_vel"
        self.right_actuator_name = f"{robot_name}_right_wheel_vel"
        self.base_body_name = f"{robot_name}_base_link"
        self.left_sign = 1.0
        self.right_sign = 1.0
        super().__init__(model, data, robot_name, base_pos, base_quat, log_dir, **kwargs)
        self._validate_geometry_assumptions()

    def _warn(self, message: str):
        print(message)
        self.log(message)

    def _require_named_id(self, obj_type, name: str, label: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id == -1:
            raise RuntimeError(f"[{self.robot_name}] Missing required {label} '{name}' in rbtheron MJCF.")
        return obj_id

    def _validate_geometry_assumptions(self):
        """Fail fast if rbtheron geometry no longer matches controller assumptions."""
        if self.model is None or self.body_id == -1:
            return

        left_body_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_BODY,
            f"{self.robot_name}_left_wheel_link",
            "wheel body",
        )
        right_body_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_BODY,
            f"{self.robot_name}_right_wheel_link",
            "wheel body",
        )
        left_geom_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{self.robot_name}_left_wheel_geom",
            "wheel geom",
        )
        right_geom_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{self.robot_name}_right_wheel_geom",
            "wheel geom",
        )

        modeled_track = abs(
            float(self.model.body_pos[left_body_id][1]) - float(self.model.body_pos[right_body_id][1])
        )
        if abs(modeled_track - self.WHEEL_TRACK) > self.WHEEL_TRACK_TOLERANCE:
            raise RuntimeError(
                f"[{self.robot_name}] rbtheron wheel track {modeled_track:.4f}m differs from controller "
                f"constant {self.WHEEL_TRACK:.4f}m."
            )

        for geom_id in (left_geom_id, right_geom_id):
            modeled_radius = float(self.model.geom_size[geom_id][0])
            if abs(modeled_radius - self.WHEEL_RADIUS) > self.WHEEL_RADIUS_TOLERANCE:
                raise RuntimeError(
                    f"[{self.robot_name}] rbtheron wheel radius {modeled_radius:.4f}m differs from controller "
                    f"constant {self.WHEEL_RADIUS:.4f}m."
                )

        wheel_bottom = (
            float(self.model.body_pos[left_body_id][2])
            + float(self.model.geom_pos[left_geom_id][2])
            - float(self.model.geom_size[left_geom_id][0])
        )
        support_bottoms = []
        for support_name in self.SUPPORT_GEOM_NAMES:
            support_geom_id = self._require_named_id(
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{self.robot_name}_{support_name}",
                "support geom",
            )
            support_body_id = int(self.model.geom_bodyid[support_geom_id])
            support_bottom = (
                float(self.model.body_pos[support_body_id][2])
                + float(self.model.geom_pos[support_geom_id][2])
                - self._support_half_height(support_geom_id)
            )
            support_bottoms.append(support_bottom)
            if support_bottom <= (wheel_bottom + self.SUPPORT_CLEARANCE_MIN):
                raise RuntimeError(
                    f"[{self.robot_name}] support geom '{support_name}' is too close to or below the drive-wheel "
                    f"contact plane ({support_bottom:.4f}m vs {wheel_bottom:.4f}m)."
                )

        chassis_geom_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{self.robot_name}_chassis_collision",
            "chassis collision geom",
        )
        chassis_bottom = float(self.model.geom_pos[chassis_geom_id][2]) - float(self.model.geom_size[chassis_geom_id][2])
        highest_support_bottom = max(support_bottoms)
        if chassis_bottom <= (highest_support_bottom + self.CHASSIS_SUPPORT_CLEARANCE_MIN):
            raise RuntimeError(
                f"[{self.robot_name}] chassis collision geom is too close to the support contact plane "
                f"({chassis_bottom:.4f}m vs {highest_support_bottom:.4f}m)."
            )

    def _support_half_height(self, geom_id: int) -> float:
        geom_type = int(self.model.geom_type[geom_id])
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(self.model.geom_size[geom_id][0])
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return float(self.model.geom_size[geom_id][2])
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return float(self.model.geom_size[geom_id][0])
        raise RuntimeError(
            f"[{self.robot_name}] Unsupported rbtheron support geom type "
            f"{mujoco.mjtGeom(geom_type).name.lower()}."
        )
