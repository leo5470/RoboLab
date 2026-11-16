"""Grasp frames (GraspDataGen -> DROID) and the grasp library."""

import numpy as np
import yaml

from robolab.core.scripted_demos.grasps import frames, library
from scripted_demos_helpers import random_pose


# Finger boxes in each gripper's own base_link, as measured from the two USDs (research §6,
# <grasp_data>/gripper_frame_map.json): GraspDataGen fingers along +Z, left finger at -Y; DROID along +X, left at +Y.
GDG_LEFT = (np.array([-0.0135, -0.07481, 0.09144]), np.array([0.0135, -0.04355, 0.14842]))
GDG_RIGHT = (np.array([-0.0135, 0.04355, 0.09144]), np.array([0.0135, 0.07481, 0.14842]))
DROID_LEFT = (np.array([0.09286, 0.04167, -0.0135]), np.array([0.15004, 0.07282, 0.0135]))
DROID_RIGHT = (np.array([0.0931, -0.07252, -0.0135]), np.array([0.15029, -0.04138, 0.0135]))
MEASURED_T = np.array([0.0, 0.000145, -0.001643])


def box_center(box):
    return (box[0] + box[1]) / 2


def test_quaternion_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[0] >= 0 else -q
        assert np.allclose(frames.matrix_to_quat(frames.quat_to_matrix(q)), q, atol=1e-9)


def test_invert_and_twin():
    rng = np.random.default_rng(1)
    T = random_pose(rng)
    assert np.allclose(frames.invert(T) @ T, np.eye(4), atol=1e-12)
    twin = frames.twin(T)
    assert np.allclose(twin[:3, 0], T[:3, 0])  # same approach axis
    assert np.allclose(twin[:3, 1], -T[:3, 1])  # jaws swapped
    assert np.allclose(frames.twin(twin), T, atol=1e-12)


def test_frame_map_is_a_rotation_with_the_documented_axes():
    R = frames.R_GDG_DROID
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R[:, 0], [0, 0, 1])  # DROID x (approach) = GDG z
    assert np.allclose(R[:, 1], [0, -1, 0])  # DROID y = -GDG y
    assert np.allclose(R[:, 2], [1, 0, 0])  # DROID z = GDG x


def test_frame_map_puts_droid_fingers_on_gdg_fingers():
    """Round trip with synthetic finger points: a grasp's GDG finger centres, seen through T_gdg_droid, land on the
    DROID finger centres of the same grasp (within the 2 mm per-side open-gap difference)."""
    T_gdg_droid = frames.gdg_to_droid_transform(MEASURED_T)
    rng = np.random.default_rng(2)
    T_obj_gdg = random_pose(rng)
    T_obj_droid = T_obj_gdg @ T_gdg_droid
    gdg_world = [frames.transform_points(T_obj_gdg, box_center(b)) for b in (GDG_LEFT, GDG_RIGHT)]
    droid_world = [frames.transform_points(T_obj_droid, box_center(b)) for b in (DROID_LEFT, DROID_RIGHT)]
    # Each DROID finger sits on one GDG finger (names differ because y is mirrored).
    for p in droid_world:
        assert min(np.linalg.norm(p - q) for q in gdg_world) < 0.0025
    # The pad midpoints coincide closely.
    assert np.linalg.norm(np.mean(gdg_world, 0) - np.mean(droid_world, 0)) < 0.0005


def write_yaml(path, grasps):
    data = {"format": "isaac_grasp", "format_version": "1.0", "grasps": {}}
    for name, (conf, guess, held, finger) in grasps.items():
        qg, qh = frames.matrix_to_quat(guess[:3, :3]), frames.matrix_to_quat(held[:3, :3])
        data["grasps"][name] = {
            "confidence": conf, "position": held[:3, 3].tolist(),
            "orientation": {"w": float(qh[0]), "xyz": qh[1:].tolist()},
            "pregrasp_position": guess[:3, 3].tolist(),
            "pregrasp_orientation": {"w": float(qg[0]), "xyz": qg[1:].tolist()},
            "cspace_position": {"finger_joint": finger},
        }
    path.write_text(yaml.safe_dump(data))


def test_library_parses_filters_maps_and_adds_twins(tmp_path):
    rng = np.random.default_rng(3)
    poses = {f"g{i}": (random_pose(rng), random_pose(rng)) for i in range(5)}
    write_yaml(tmp_path / "x.yaml", {n: (float(i % 2 == 0), g, h, 0.3 + i) for i, (n, (g, h)) in enumerate(poses.items())})
    parsed = library.parse_isaac_grasp_yaml(tmp_path / "x.yaml")
    assert np.allclose(parsed["T_obj_gdg_guess"][1], poses["g1"][0], atol=1e-6)
    assert np.allclose(parsed["T_obj_gdg_held"][1], poses["g1"][1], atol=1e-6)
    T_gdg_droid = frames.gdg_to_droid_transform(MEASURED_T)
    passed_1g = np.array([True, False, False, False, True])
    lib = library.build(parsed, T_gdg_droid, also_passed={"1g": passed_1g})
    kept = [0, 2, 4]  # confidence > 0
    assert len(lib["T_obj_grasp"]) == 6
    assert lib["source_index"].tolist() == kept * 2
    assert lib["twin"].tolist() == [0, 0, 0, 1, 1, 1]
    assert lib["passed_1g"].tolist() == [True, False, True] * 2
    for j, i in enumerate(kept):
        expected = poses[f"g{i}"][0] @ T_gdg_droid
        assert np.allclose(lib["T_obj_grasp"][j], expected, atol=1e-6)
        assert np.allclose(lib["T_obj_grasp"][j + 3], frames.twin(expected), atol=1e-6)
    path = tmp_path / "lib.npz"
    library.save(path, lib)
    loaded = library.load(path)
    assert set(loaded) == set(lib) and np.allclose(loaded["T_obj_grasp"], lib["T_obj_grasp"])
