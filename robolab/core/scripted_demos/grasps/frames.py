"""Rigid transforms (numpy, quaternions w-x-y-z) and the GraspDataGen → DROID gripper frame map (D5)."""

from __future__ import annotations

import json
import pathlib

import numpy as np

# Columns are DROID base_link axes expressed in GraspDataGen base_link axes:
# DROID x (approach, fingers) = GDG z; DROID y = -GDG y; DROID z = GDG x.
R_GDG_DROID = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])

APPROACH_AXIS = 0  # DROID base_link +X points from the palm towards the fingertips
CLOSING_AXIS = 1  # the jaws close along DROID base_link Y
TCP = np.array([0.1216, 0.0, 0.0])  # centre of the inner fingers in DROID base_link


def quat_to_matrix(q) -> np.ndarray:
    """Unit quaternion(s) (..., 4) w-x-y-z → rotation matrices (..., 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def matrix_to_quat(R) -> np.ndarray:
    """Rotation matrix to a unit quaternion (w, x, y, z) with w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def pose(position, quat_wxyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(quat_wxyz)
    T[:3, 3] = position
    return T


def rot(axis: int, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def homogeneous(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def invert(T) -> np.ndarray:
    T = np.asarray(T)
    out = np.zeros_like(T)
    R = T[..., :3, :3]
    out[..., :3, :3] = np.swapaxes(R, -1, -2)
    out[..., :3, 3] = -np.einsum("...ji,...j->...i", R, T[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def transform_points(T, points) -> np.ndarray:
    return np.asarray(points) @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]


def rotation_angle(Ra, Rb) -> float:
    """Angle (rad) of the relative rotation between two rotation matrices."""
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1) / 2
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def gdg_to_droid_transform(translation_gdg_from_droid) -> np.ndarray:
    """``T_gdg_droid``: the DROID base_link pose in the GraspDataGen base_link frame."""
    return homogeneous(R_GDG_DROID, translation_gdg_from_droid)


def load_frame_map(path: pathlib.Path) -> np.ndarray:
    data = json.loads(pathlib.Path(path).read_text())
    if not np.allclose(np.asarray(data["rotation_gdg_from_droid"]), R_GDG_DROID):
        raise ValueError(f"{path} was measured with a different rotation")
    return gdg_to_droid_transform(data["translation_gdg_from_droid_m"])


def twin(T_obj_grasp) -> np.ndarray:
    """The same grasp with the gripper turned 180° about its approach axis (the 2F-85 is symmetric)."""
    flip = homogeneous(rot(APPROACH_AXIS, np.pi))
    return np.asarray(T_obj_grasp) @ flip
