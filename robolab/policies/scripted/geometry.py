"""Small, simulator-independent rigid-transform and OBB helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

ArrayLike = Iterable[float] | np.ndarray


def normalize(vector: ArrayLike, *, eps: float = 1e-9) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector")
    return value / norm


def quat_normalize_wxyz(quat: ArrayLike) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quat.shape}")
    return normalize(quat)


def quat_conjugate_wxyz(quat: ArrayLike) -> np.ndarray:
    w, x, y, z = quat_normalize_wxyz(quat)
    return np.array([w, -x, -y, -z])


def quat_multiply_wxyz(left: ArrayLike, right: ArrayLike) -> np.ndarray:
    """Hamilton product in RoboLab/IsaacLab ``(w, x, y, z)`` ordering."""
    w1, x1, y1, z1 = quat_normalize_wxyz(left)
    w2, x2, y2, z2 = quat_normalize_wxyz(right)
    return quat_normalize_wxyz([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_matrix_wxyz(quat: ArrayLike) -> np.ndarray:
    w, x, y, z = quat_normalize_wxyz(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a normalized wxyz quaternion."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected rotation shape (3, 3), got {matrix.shape}")
    trace = float(np.trace(matrix))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        quat = [0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(matrix)))
        if i == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            quat = [(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s,
                    (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s]
        elif i == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            quat = [(matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s,
                    0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            quat = [(matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s,
                    (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s]
    return quat_normalize_wxyz(quat)


def quat_angle_wxyz(left: ArrayLike, right: ArrayLike) -> float:
    dot = abs(float(np.dot(quat_normalize_wxyz(left), quat_normalize_wxyz(right))))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def quat_slerp_wxyz(start: ArrayLike, end: ArrayLike, fraction: float) -> np.ndarray:
    q0, q1 = quat_normalize_wxyz(start), quat_normalize_wxyz(end)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if dot > 0.9995:
        return quat_normalize_wxyz(q0 + fraction * (q1 - q0))
    theta = math.acos(float(np.clip(dot, -1.0, 1.0)))
    return (math.sin((1 - fraction) * theta) * q0 + math.sin(fraction * theta) * q1) / math.sin(theta)


@dataclass(frozen=True)
class Pose:
    """Rigid pose whose quaternion always uses wxyz ordering."""

    position: np.ndarray
    quaternion: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(f"Expected position shape (3,), got {position.shape}")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion", quat_normalize_wxyz(self.quaternion))

    @property
    def rotation(self) -> np.ndarray:
        return quat_to_matrix_wxyz(self.quaternion)

    def matrix(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.position
        return transform

    def inverse(self) -> "Pose":
        rotation_t = self.rotation.T
        return Pose(rotation_t @ -self.position, quat_conjugate_wxyz(self.quaternion))

    def compose(self, child: "Pose") -> "Pose":
        return Pose(
            self.position + self.rotation @ child.position,
            quat_multiply_wxyz(self.quaternion, child.quaternion),
        )

    def transform_point(self, point: ArrayLike) -> np.ndarray:
        return self.position + self.rotation @ np.asarray(point, dtype=np.float64)

    def relative_to(self, parent: "Pose") -> "Pose":
        return parent.inverse().compose(self)


def pose_from_world_state(world, body: str, env_id: int, *, is_relative: bool = True) -> Pose:
    position, quaternion = world.get_pose(body, is_relative=is_relative, env_id=env_id)
    return Pose(_numpy(position), _numpy(quaternion))


def robot_root_pose_env(world, env_id: int) -> Pose:
    """Robot-root pose in env-local coordinates, including root rotation."""
    robot = world.get_articulation("robot")
    pos_w = _numpy(robot.data.root_pos_w[env_id])
    quat_w = _numpy(robot.data.root_quat_w[env_id])
    origin = _numpy(world.env.scene.env_origins[env_id])
    return Pose(pos_w - origin, quat_w)


def env_pose_to_robot(pose_env: Pose, robot_pose_env: Pose) -> Pose:
    """Full SE(3) conversion; unlike origin subtraction this handles root rotation."""
    return pose_env.relative_to(robot_pose_env)


def robot_pose_to_env(pose_robot: Pose, robot_pose_env: Pose) -> Pose:
    return robot_pose_env.compose(pose_robot)


def make_tool_orientation(closing_axis: ArrayLike, approach_axis: ArrayLike) -> np.ndarray:
    """Build a tool pose with +X closing and +Z approach axes."""
    z_axis = normalize(approach_axis)
    x_raw = np.asarray(closing_axis, dtype=np.float64)
    x_axis = normalize(x_raw - np.dot(x_raw, z_axis) * z_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    return matrix_to_quat_wxyz(np.column_stack((x_axis, y_axis, z_axis)))


def projected_half_extents(rotation: np.ndarray, dimensions: ArrayLike) -> np.ndarray:
    """AABB half extents after rotating an OBB with the given relative rotation."""
    return np.abs(rotation) @ (np.asarray(dimensions, dtype=np.float64) / 2.0)


def bbox_min_max(corners) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray([[float(v) for v in corner] for corner in corners], dtype=np.float64)
    return array.min(axis=0), array.max(axis=0)


@dataclass(frozen=True)
class OrientedBox:
    """Eight-corner box expressed in the environment-local frame.

    RoboLab's live ``get_bbox`` result preserves the ordering used by
    ``Gf.Range3d.GetCorner``: corners 1, 2, and 4 are the three adjacent
    corners to corner 0. Unlike ``WorldState.get_dimensions()``, these edge
    lengths include authored geometry transforms below a rigid-object root.
    """

    center_env: np.ndarray
    axes_env: np.ndarray
    dimensions: np.ndarray
    corners_env: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center_env, dtype=np.float64)
        axes = np.asarray(self.axes_env, dtype=np.float64)
        dimensions = np.asarray(self.dimensions, dtype=np.float64)
        corners = np.asarray(self.corners_env, dtype=np.float64)
        if center.shape != (3,) or axes.shape != (3, 3) or dimensions.shape != (3,):
            raise ValueError("Invalid oriented-box center, axes, or dimensions")
        if corners.shape != (8, 3) or np.any(dimensions <= 0):
            raise ValueError("An oriented box requires eight corners and positive dimensions")
        object.__setattr__(self, "center_env", center)
        object.__setattr__(self, "axes_env", axes)
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "corners_env", corners)

    @property
    def pose_env(self) -> Pose:
        return Pose(self.center_env, matrix_to_quat_wxyz(self.axes_env))


def oriented_box_from_corners(corners) -> OrientedBox:
    """Recover an OBB from RoboLab's ordered eight-corner representation."""
    array = np.asarray([[float(v) for v in corner] for corner in corners], dtype=np.float64)
    if array.shape != (8, 3):
        raise ValueError(f"Expected bounding-box corners with shape (8, 3), got {array.shape}")
    edges = np.column_stack((array[1] - array[0], array[2] - array[0], array[4] - array[0]))
    dimensions = np.linalg.norm(edges, axis=0)
    if np.any(dimensions < 1e-8):
        raise ValueError("Bounding box contains a degenerate edge")
    axes = edges / dimensions
    if not np.allclose(axes.T @ axes, np.eye(3), atol=2e-3):
        raise ValueError("Bounding-box edges are not mutually orthogonal")
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0
    return OrientedBox(
        center_env=array.mean(axis=0),
        axes_env=axes,
        dimensions=dimensions,
        corners_env=array,
    )


def oriented_box_from_world(world, body: str, env_id: int) -> OrientedBox:
    """Return live physical OBB geometry for one WorldState entity."""
    corners, _ = world.get_bbox(body, env_id=env_id)
    return oriented_box_from_corners(corners)


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)

