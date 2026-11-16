"""Six-joint path screen of explicit targets, on a synthetic arm and recording (no USD)."""

import math

import h5py
import numpy as np
import pytest

from robolab.core.omy_teleop.path_screen import (LOCKED, POSITION_M, Arm, load_source, pose_matrix, rotation_vector,
                                                 screen, screen_pair, transformed_path)

OBJECT, CONTAINER = "obj", "bin"
HOME = np.array([0.0, -0.6, 0.0, -2.5, 0.0, 1.9, 0.0])


def _frame(xyz, rpy=(0, 0, 0)):
    out = np.eye(4)
    cr, sr, cp, sp, cy, sy = (f(a) for a in rpy for f in (math.cos, math.sin))
    out[:3, :3] = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                   @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
    out[:3, 3] = xyz
    return out


def _arm():
    """Seven revolute joints about local X with alternating axes, then a fixed tool offset."""
    half = math.pi / 2
    frames = [_frame([0, 0, 0.33], (0, -half, 0)), _frame([0, 0, 0], (0, 0, half)), _frame([0.32, 0, 0], (0, 0, -half)),
              _frame([0, 0.08, 0], (0, 0, half)), _frame([0.38, -0.08, 0], (0, 0, -half)), _frame([0, 0, 0], (0, 0, half)),
              _frame([0, 0.09, 0], (0, 0, half))]
    chain = [(frame, np.eye(4), True) for frame in frames] + [(_frame([0.2, 0, 0]), np.eye(4), False)]
    return Arm(chain, limits=[(-2.9, 2.9)] * 7)


def _quat(yaw=0.0):
    return [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def _record(path, arm, steps=40, grasp=20):
    """A source whose hand moves smoothly; the object is raised with the gripper closed from step ``grasp``."""
    s = np.linspace(0, 1, steps)[:, None]
    joints = HOME + s * np.array([0.5, 0.3, 0, 0.4, 0.3, -0.3, 0.4])
    actions = np.c_[joints, np.where(np.arange(steps) < grasp - 2, 1.0, -1.0)]
    obj = np.tile([0.5, 0.1, 0.0, *_quat()], (steps, 1))
    obj[grasp:, 2] = 0.05
    container = np.tile([0.4, -0.3, 0.0, *_quat()], (steps, 1))
    with h5py.File(path, "w") as stream:
        ep = stream.create_group("data/demo_0")
        ep["actions"] = actions
        ep["initial_state/articulation/robot/joint_position"] = joints[:1]
        ep["states/articulation/robot/joint_position"] = np.r_[joints[1:], joints[-1:]]
        for name, poses in ((OBJECT, obj), (CONTAINER, container)):
            ep[f"initial_state/rigid_object/{name}/root_pose"] = poses[:1]
            ep[f"states/rigid_object/{name}/root_pose"] = np.r_[poses[1:], poses[-1:]]
    return {OBJECT: obj[0].tolist(), CONTAINER: container[0].tolist()}


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    arm, path = _arm(), tmp_path_factory.mktemp("screen") / "omy_demos.hdf5"
    own = _record(path, arm)
    return arm, path, own, load_source(path, "demo_0", OBJECT, CONTAINER, arm)


def test_jacobian_matches_finite_differences():
    arm, joints = _arm(), HOME + 0.1
    pose, jac = arm.fk(joints, jacobian=True)
    for k in range(7):
        moved = joints.copy()
        moved[k] += 1e-4
        other = arm.fk(moved)
        numeric = np.r_[other[:3, 3] - pose[:3, 3], rotation_vector(other[:3, :3] @ pose[:3, :3].T)] / 1e-4
        assert np.allclose(jac[:, k], numeric, atol=1e-3)


def test_ik_reaches_a_nearby_pose_without_moving_the_locked_joint():
    arm = _arm()
    goal = HOME + np.array([0.3, -0.2, 0, 0.25, 0.2, -0.3, 0.2])
    joints, (position, rotation) = arm.solve(arm.fk(goal), HOME)
    assert position < POSITION_M / 10 and rotation < 1e-3 and joints[LOCKED] == HOME[LOCKED]
    assert np.allclose(arm.fk(joints), arm.fk(goal), atol=1e-4)
    far = arm.fk(goal).copy()
    far[:3, 3] += [2.0, 0, 0]
    _, (position, _) = arm.solve(far, HOME)
    assert position > 1.0


def test_segments_move_with_their_reference(case):
    arm, _, own, source = case
    assert source["grasp"] == 21  # the step after the object is first seen raised
    assert np.allclose(transformed_path(source, own, OBJECT, CONTAINER), source["hand"], atol=1e-9)
    target = {OBJECT: [0.45, 0.2, 0.0, *_quat(0.5)], CONTAINER: [0.42, -0.25, 0.0, *_quat()]}
    path = transformed_path(source, target, OBJECT, CONTAINER)
    # The hand keeps its pose relative to the object in the grasp segment and to the container afterwards.
    for name, steps in ((OBJECT, range(0, 21)), (CONTAINER, range(21, 40))):
        for t in steps:
            before = np.linalg.inv(pose_matrix(own[name])) @ source["hand"][t]
            assert np.allclose(np.linalg.inv(pose_matrix(target[name])) @ path[t], before, atol=1e-9)


def test_own_layout_is_reachable_and_far_targets_are_not(case):
    arm, _, own, source = case
    result = screen_pair(arm, source, own, OBJECT, CONTAINER)
    assert result["reachable"] and result["unreached_steps"] == 0 and result["worst_position_m"] < POSITION_M
    assert result["placement_bridge_m"] < 0.05 and result["max_joint_change_per_step_rad"] < 0.1
    far_bin = {**own, CONTAINER: [1.6, -0.3, 0.0, *_quat()]}
    result = screen_pair(arm, source, far_bin, OBJECT, CONTAINER)
    assert not result["reachable"] and result["worst_segment"] == 1 and result["worst_step"] >= source["grasp"]
    assert result["placement_bridge_m"] == pytest.approx(1.2, abs=0.05)  # the container moved 1.2 m
    far_object = {**own, OBJECT: [1.7, 0.1, 0.0, *_quat()]}
    result = screen_pair(arm, source, far_object, OBJECT, CONTAINER)
    assert not result["reachable"] and result["worst_segment"] == 0
    assert result["start_bridge_m"] == pytest.approx(1.2, abs=0.05)


def test_manifest_screen_lists_reachable_sources(case):
    arm, path, own, _ = case
    manifest = {"sampler": {"profile": {"object": OBJECT, "container": CONTAINER}},
                "targets": [{"id": "near", "poses": own},
                            {"id": "far", "poses": {**own, CONTAINER: [1.6, -0.3, 0.0, *_quat()]}}]}
    rows, counts = screen(manifest, {"a": (path, "demo_0")}, arm)
    assert [r["reachable_sources"] for r in rows] == [["a"], []]
    assert counts == {"targets": 2, "targets_with_a_reachable_source": 1}
    assert screen(manifest, {"a": (path, "demo_0")}, arm, workers=2)[0] == rows


def test_controller_emulation_and_repaired_screen(case):
    from robolab.core.omy_teleop.path_screen import emulate_controller, repaired_screen
    from robolab.core.omy_teleop.repairs import RepairSettings
    arm, _, own, source = case
    path = [source["hand"][t] for t in range(source["steps"])]
    position, rotation, excess, _ = emulate_controller(arm, path, source["joints"][0])
    assert position < 0.01 and excess <= 0.0
    lagged = emulate_controller(arm, path, source["joints"][0], response=0.28)[0]
    assert lagged >= position  # a lagging arm tracks no better
    far = [pose.copy() for pose in path]
    for pose in far:
        pose[:3, 3] += [2.0, 0, 0]
    assert emulate_controller(arm, far, source["joints"][0])[0] > 1.0
    geometry = {"object_radius_m": 0.05, "rim_height_m": 0.01, "opening_center_m": [0, 0], "opening_half_m": [5, 5]}
    settings = RepairSettings(connection_speed_m_s=0.25)
    result = repaired_screen(arm, source, own, OBJECT, CONTAINER, geometry, settings)
    (only,) = result["candidates"]
    assert only["offset_deg"] is None and only["feasible"] and result["best"] is only
    assert only["approach_m"] < 0.01 and only["carry_m"] < 0.05  # own layout: connections nearly vanish
    turned = repaired_screen(arm, source, own, OBJECT, CONTAINER, geometry,
                             RepairSettings(connection_speed_m_s=0.25, symmetric_yaw=True), [0.0, 90.0])
    assert [c["offset_deg"] for c in turned["candidates"]] == [0.0, 90.0]
    assert turned["candidates"][1]["approach_rad"] > turned["candidates"][0]["approach_rad"]
