import mujoco

from common.differential_drive_controller import DifferentialDriveController


class TracerController(DifferentialDriveController):
    """Controller for the Agilex Tracer robot."""

    # Keep these in sync with robots/tracer_control/agilex_tracer2/tracer2.xml.
    WHEEL_RADIUS = 0.085
    WHEEL_TRACK = 0.5074
    DEFAULT_LINEAR_SPEED = 0.45
    MAX_LINEAR_SPEED = 0.55
    DEFAULT_ANGULAR_SPEED_DEG = 35.0
    MAX_ANGULAR_SPEED_DEG = 60.0
    LINEAR_ACCEL = 0.22
    ANGULAR_ACCEL_DEG = 45.0
    HEADING_KP = 2.0
    TURN_KP = 8.0
    LINEAR_DISTANCE_TOLERANCE = 0.03
    HEADING_TOLERANCE_DEG = 3.0
    MAX_COMPLETION_OVERRUN = 5.0

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
        self.left_actuator_name = f"{robot_name}_left_drive"
        self.right_actuator_name = f"{robot_name}_right_drive"
        self.base_body_name = f"{robot_name}_base_link"
        self.left_sign = -1.0
        self.right_sign = 1.0
        super().__init__(model, data, robot_name, base_pos, base_quat, log_dir, **kwargs)
        self._validate_geometry_assumptions()

    def _validate_geometry_assumptions(self):
        """Warn early if the traced MJCF no longer matches the controller constants."""
        if self.body_id == -1:
            return
        left_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{self.robot_name}_left_link")
        right_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{self.robot_name}_right_link")
        if left_body_id == -1 or right_body_id == -1:
            return

        modeled_track = abs(float(self.model.body_pos[left_body_id][1]) - float(self.model.body_pos[right_body_id][1]))
        if abs(modeled_track - self.WHEEL_TRACK) > 0.01:
            msg = (
                f"[{self.robot_name}] Warning: tracer wheel track {modeled_track:.4f}m "
                f"differs from controller constant {self.WHEEL_TRACK:.4f}m."
            )
            print(msg)
            self.log(msg)

        for body_name in (f"{self.robot_name}_left_link", f"{self.robot_name}_right_link"):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                continue
            geom_start = int(self.model.body_geomadr[body_id])
            geom_count = int(self.model.body_geomnum[body_id])
            wheel_geom_id = -1
            for geom_id in range(geom_start, geom_start + geom_count):
                if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    wheel_geom_id = geom_id
                    break
            if wheel_geom_id == -1:
                continue

            modeled_radius = float(self.model.geom_size[wheel_geom_id][0])
            if abs(modeled_radius - self.WHEEL_RADIUS) > 0.005:
                msg = (
                    f"[{self.robot_name}] Warning: tracer wheel radius {modeled_radius:.4f}m "
                    f"differs from controller constant {self.WHEEL_RADIUS:.4f}m."
                )
                print(msg)
                self.log(msg)
                break
