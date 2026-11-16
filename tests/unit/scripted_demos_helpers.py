"""Scene and grasp builders shared by several test modules."""

import itertools

import numpy as np

from robolab.core.scripted_demos.grasps import frames
from robolab.core.scripted_demos import kinematics, select


def random_pose(rng):
    q = rng.normal(size=4)
    return frames.pose(rng.uniform(-0.3, 0.3, 3), q / np.linalg.norm(q))


def box_obb(center, half):
    return np.array([np.asarray(center) + np.asarray(half) * s for s in itertools.product((-1, 1), repeat=3)])


def top_down_grasp(point, yaw=0.0):
    """DROID base_link pose whose pad centre is at ``point`` with the fingers pointing down."""
    R = frames.rot(2, yaw) @ frames.rot(1, np.pi / 2)  # base_link x → world -z
    T = frames.homogeneous(R)
    T[:3, 3] = np.asarray(point) - R @ select.TCP
    return T


def side_grasp(point, from_dir):
    """Fingers horizontal, approaching along ``from_dir`` (unit xy), jaws closing horizontally."""
    x = -np.r_[from_dir, 0.0]
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    R = np.stack([x, y, z], 1)
    T = frames.homogeneous(R)
    T[:3, 3] = np.asarray(point) - R @ select.TCP
    return T


def toy_scene(obj_center=(0.5, -0.15, 0.10)):
    # A 8 x 6 x 16 cm box: narrow enough along y for the open Robotiq's linkage to clear it in a side grasp.
    pts = np.array(list(itertools.product((-0.04, 0.04), (-0.03, 0.03), (-0.08, 0.08))))
    pts = np.vstack([pts, np.random.default_rng(0).uniform([-0.04, -0.03, -0.08], [0.04, 0.03, 0.08], (400, 3))])
    T_obj = frames.homogeneous(t=obj_center)
    return select.Scene(T_world_obj=T_obj, object_points=pts, bin_corners=box_obb([0.55, 0.2, 0.065], [0.12, 0.16, 0.065]),
                        table_z=0.0, q_current=kinematics.HOME.copy(), T_world_ee=top_down_grasp([0.45, 0.0, 0.35]))


def toy_library(grasps_in_object_frame):
    T = np.stack(grasps_in_object_frame)
    return {"T_obj_grasp": T, "T_obj_held": T.copy(), "confidence": np.ones(len(T)),
            "passed_1g": np.ones(len(T), dtype=bool)}
