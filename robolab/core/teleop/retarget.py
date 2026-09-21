"""SO-101 pose retargeting, independent of Isaac Sim and serial hardware."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ORIENTATION_MODE_CODES = {"fixed": 0, "pose": 1, "wrist-roll": 2}


class TrackingError(RuntimeError):
    """Tracking must pause and be re-anchored explicitly."""


def rotation(quat):
    """Convert a finite unit wxyz quaternion to a SciPy rotation."""
    quat = vector(quat, 4, "quaternion")
    if abs(np.linalg.norm(quat) - 1) > 0.01:
        raise ValueError("quaternion must have unit norm")
    return Rotation.from_quat(quat[[1, 2, 3, 0]])


def quaternion(rot):
    return rot.as_quat()[[3, 0, 1, 2]]


def vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result.copy()


@dataclass
class RetargetConfig:
    schema_version: int = 1
    alignment_wxyz: tuple = (1.0, 0.0, 0.0, 0.0)
    translation_gain: float = 1.0
    base_pan_gain: float = 1.0
    orientation_mode: str = "fixed"
    filter_alpha: float = 0.5
    position_deadband_m: float = 0.0005
    rotation_deadband_rad: float = 0.003
    max_translation_step_m: float = 0.01
    max_rotation_step_rad: float = 0.05
    # Relative to the measured start/resume pose; valid with differently placed robots.
    workspace_min_offset_m: tuple = (-0.3, -0.3, -0.3)
    workspace_max_offset_m: tuple = (0.3, 0.3, 0.3)
    max_position_error_m: float = 0.12
    max_rotation_error_rad: float = 0.8
    tracking_error_steps: int = 15
    gripper_open: float = 100.0
    gripper_closed: float = 0.0
    gripper_close_threshold: float = 0.65
    gripper_open_threshold: float = 0.35
    sample_timeout_s: float = 0.2

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported teleop config schema_version")
        rotation(self.alignment_wxyz)
        if self.orientation_mode not in ORIENTATION_MODE_CODES:
            raise ValueError("orientation_mode must be fixed, pose, or wrist-roll")
        for name in (
            "translation_gain",
            "base_pan_gain",
            "max_translation_step_m",
            "max_rotation_step_rad",
            "max_position_error_m",
            "max_rotation_error_rad",
            "sample_timeout_s",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("position_deadband_m", "rotation_deadband_rad"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 < self.filter_alpha <= 1:
            raise ValueError("filter_alpha must be in (0, 1]")
        if type(self.tracking_error_steps) is not int or self.tracking_error_steps < 1:
            raise ValueError("tracking_error_steps must be a positive integer")
        lo = vector(self.workspace_min_offset_m, 3, "workspace_min_offset_m")
        hi = vector(self.workspace_max_offset_m, 3, "workspace_max_offset_m")
        if np.any(lo >= hi) or np.any(lo > 0) or np.any(hi < 0):
            raise ValueError("workspace offsets must contain the reference pose and have min < max")
        if not (0 <= self.gripper_open_threshold < self.gripper_close_threshold <= 1):
            raise ValueError("gripper thresholds must satisfy 0 <= open < close <= 1")
        if not np.isfinite([self.gripper_open, self.gripper_closed]).all() or self.gripper_open == self.gripper_closed:
            raise ValueError("gripper endpoints must be finite and different")

    @classmethod
    def load(cls, path=None, orientation_mode=None):
        data = yaml.safe_load(Path(path).read_text()) if path else {}
        if not isinstance(data, dict):
            raise ValueError("teleop config must be a mapping")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown teleop config keys: {sorted(unknown)}")
        if orientation_mode is not None:
            data["orientation_mode"] = orientation_mode
        return cls(**data)

    def to_dict(self):
        return asdict(self)


def validate_action_config(actions):
    """Accept only the supported DROID base_link relative-pose action contract."""
    arm = actions.arm_action
    controller = arm.controller
    if (
        arm.asset_name != "robot"
        or arm.body_name != "base_link"
        or controller.command_type != "pose"
        or not controller.use_relative_mode
    ):
        raise ValueError("SO-101 requires robot/base_link relative pose IK")
    if getattr(arm, "clip", None) is not None:
        raise ValueError("Configure limits in teleop, not an additional action clip")
    offset = getattr(arm, "body_offset", None)
    if offset is not None:
        if not np.allclose(offset.pos, 0) or rotation(offset.rot).magnitude() > 1e-7:
            raise ValueError("SO-101 requires an identity base_link action offset")
    scale = np.asarray(arm.scale, dtype=float)
    if scale.ndim != 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("SO-101 requires a finite positive scalar action scale")
    finger = actions.finger_joint
    if finger.class_type.__name__ != "BinaryJointPositionZeroToOneAction" or finger.asset_name != "robot":
        raise ValueError("SO-101 requires the DROID 0/1 binary gripper action")
    return float(scale)


def bounded(value, limit, deadband):
    norm = np.linalg.norm(value)
    if norm <= deadband:
        return np.zeros_like(value)
    return value * min(1.0, limit / norm)


class PoseRetargeter:
    def __init__(self, config, action_scale=0.5, leader_fk=None):
        if not np.isfinite(action_scale) or action_scale <= 0:
            raise ValueError("action_scale must be positive")
        self.leader_fk = leader_fk
        self.cfg = config
        self.scale = action_scale
        self.alignment = rotation(config.alignment_wxyz)
        self.generation = 0
        self.reset()

    def reset(self):
        self.anchored = False
        self.closed = False
        self.error_steps = 0
        self.last_sequence = None

    def closure(self, value):
        if not np.isfinite(value):
            raise ValueError("Invalid gripper reading")
        return float(np.clip((value - self.cfg.gripper_open) / (self.cfg.gripper_closed - self.cfg.gripper_open), 0, 1))

    def gripper_matches(self, value):
        closure = self.closure(value)
        return (
            closure >= self.cfg.gripper_close_threshold if self.closed else closure <= self.cfg.gripper_open_threshold
        )

    def anchor(self, sample, robot_pos, robot_quat):
        if not self.gripper_matches(sample.gripper):
            raise ValueError(
                f"Move the physical gripper to {'closed' if self.closed else 'open'} before starting/resuming"
            )
        self.leader_pos0 = vector(sample.position, 3, "leader position")
        self.base_pan0 = float(sample.joints_deg[0])
        self.leader_rot0 = rotation(sample.quaternion)
        if self.cfg.orientation_mode == "wrist-roll":
            self.wrist_roll0 = float(sample.joints_deg[4])
        self.robot_pos0 = vector(robot_pos, 3, "robot position")
        self.robot_rot0 = rotation(robot_quat)
        self.target_pos = self.robot_pos0.copy()
        self.target_rot = self.robot_rot0
        self.last_sequence = sample.sequence
        self.error_steps = 0
        self.generation += 1
        self.anchored = True

    def command(self, sample, robot_pos, robot_quat):
        if not self.anchored:
            raise RuntimeError("Capture references before issuing an action")
        pos = vector(robot_pos, 3, "robot position")
        rot = rotation(robot_quat)
        if sample.sequence != self.last_sequence:
            displacement = sample.position - self.leader_pos0
            if self.cfg.base_pan_gain != 1 and float(sample.joints_deg[0]) != self.base_pan0:
                if self.leader_fk is None:
                    raise ValueError("Base-pan gain requires leader forward kinematics")
                # Isolate pan with the other joints held at their current angles.
                neutral = np.array(sample.joints_deg, dtype=float, copy=True)
                neutral[0] = self.base_pan0
                pan_displacement = np.asarray(self.leader_fk(sample.joints_deg)[0]) - np.asarray(
                    self.leader_fk(neutral)[0]
                )
                displacement = displacement + (self.cfg.base_pan_gain - 1) * pan_displacement
            target = self.robot_pos0 + self.cfg.translation_gain * self.alignment.apply(displacement)
            target = np.clip(
                target,
                self.robot_pos0 + self.cfg.workspace_min_offset_m,
                self.robot_pos0 + self.cfg.workspace_max_offset_m,
            )
            target_rot = self.robot_rot0
            if self.cfg.orientation_mode == "pose":
                delta = self.alignment * rotation(sample.quaternion) * self.leader_rot0.inv() * self.alignment.inv()
                target_rot = delta * self.robot_rot0
            elif self.cfg.orientation_mode == "wrist-roll":
                # Protocol joint 4 is calibrated wrist_roll in degrees. DROID
                # base_link +X points along the fingers: postmultiply to spin
                # around that local tool axis, independent of leader tilt.
                angle = np.deg2rad(float(sample.joints_deg[4]) - self.wrist_roll0)
                target_rot = self.robot_rot0 * Rotation.from_rotvec([angle, 0, 0])
            self.target_pos += self.cfg.filter_alpha * (target - self.target_pos)
            delta_rot = (target_rot * self.target_rot.inv()).as_rotvec()
            self.target_rot = Rotation.from_rotvec(self.cfg.filter_alpha * delta_rot) * self.target_rot
            self.last_sequence = sample.sequence
        position_error = self.target_pos - pos
        rotation_error = (self.target_rot * rot.inv()).as_rotvec()
        position_error_m = float(np.linalg.norm(position_error))
        rotation_error_rad = float(np.linalg.norm(rotation_error))
        excessive = (
            position_error_m > self.cfg.max_position_error_m or rotation_error_rad > self.cfg.max_rotation_error_rad
        )
        self.error_steps = self.error_steps + 1 if excessive else 0
        if self.error_steps >= self.cfg.tracking_error_steps:
            raise TrackingError(
                "Sustained target tracking error: "
                f"position {position_error_m * 100:.1f} cm (limit {self.cfg.max_position_error_m * 100:.1f} cm), "
                f"orientation {np.rad2deg(rotation_error_rad):.1f} deg "
                f"(limit {np.rad2deg(self.cfg.max_rotation_error_rad):.1f} deg), "
                f"for {self.error_steps} consecutive steps; reposition the leader and resume with N"
            )
        closure = self.closure(sample.gripper)
        if closure >= self.cfg.gripper_close_threshold:
            self.closed = True
        elif closure <= self.cfg.gripper_open_threshold:
            self.closed = False
        delta = (
            np.concatenate(
                (
                    bounded(position_error, self.cfg.max_translation_step_m, self.cfg.position_deadband_m),
                    bounded(rotation_error, self.cfg.max_rotation_step_rad, self.cfg.rotation_deadband_rad),
                )
            )
            / self.scale
        )
        return np.concatenate((delta, [float(self.closed)])).astype(np.float32)
