"""Switchable motion repairs: cuts, boundary edits, connection timing and yaw symmetry (no Isaac Sim)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from robolab.core.omy_teleop.repairs import (RepairSettings, connection_steps, edit_boundaries, placement_geometry,
                                             remove_yaw, retime, source_cuts)

GEOMETRY = placement_geometry({"lower": [-0.06, -0.06, 0], "upper": [0.06, 0.06, 0.12], "scale": [1, 1, 1]},
                              {"lower": [-30, -20, 0], "upper": [30, 20, 15], "scale": [0.007] * 3})


def _quat(yaw):
    return [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def _episode(steps=60, close=20, release=50):
    """Hand descends to the object, closes, lifts it to 0.25 m, carries it over the bin, opens."""
    t = np.arange(steps)
    hand = np.zeros((steps, 3))
    hand[:, 0] = np.where(t < close, 0.3 + 0.01 * (close - t), 0.3)  # approaches along x: 1 cm per step
    obj = np.tile([0.3, 0.2, 0.0, *_quat(0.4)], (steps, 1))
    lift = np.clip((t - close - 2) * 0.02, 0, 0.25)  # 2 cm per step from 2 steps after closing
    obj[:, 2] = lift
    carry = np.clip((t - 35) / 10, 0, 1)  # over the bin from step 45
    obj[:, 0] = 0.3 + carry * 0.25
    obj[:, 1] = 0.2 - carry * 0.2
    container = np.tile([0.55, 0.0, 0.0, *_quat(0.0)], (steps, 1))
    actions = np.zeros((steps, 8))
    actions[:, -1] = np.where((t >= close) & (t < release), -1.0, 1.0)
    grasp_end = int(np.argmax((actions[:, -1] < 0) & (lift > 0.02))) + 1
    return actions, hand, obj, container, grasp_end


def test_geometry_from_bounds():
    assert GEOMETRY["object_radius_m"] == pytest.approx(0.06)
    assert GEOMETRY["rim_height_m"] == pytest.approx(0.105)
    assert np.allclose(GEOMETRY["opening_half_m"], [0.21, 0.14]) and np.allclose(GEOMETRY["opening_center_m"], 0)


def test_cuts_follow_their_definitions():
    actions, hand, obj, container, grasp_end = _episode()
    off = source_cuts(actions, hand, obj, container, grasp_end, GEOMETRY, RepairSettings())
    assert off == {"approach": None, "lift": None, "over_bin": None, "release": 50}
    cuts = source_cuts(actions, hand, obj, container, grasp_end, GEOMETRY,
                       RepairSettings(approach_m=0.1, lift_clearance_m=0.03, over_bin=True))
    assert cuts["approach"] == 10  # last step 10 cm from the closing pose
    lift = obj[:, 2]
    assert lift[cuts["lift"]] >= 0.135 > lift[cuts["lift"] - 1]
    # Over the opening: |offset| + radius within the walls plus 1 cm, and only from the lift cut on.
    d = np.abs(obj[cuts["over_bin"], :2] - container[0, :2]) + 0.06
    assert np.all(d <= np.array([0.21, 0.14]) + 0.01) and cuts["lift"] <= cuts["over_bin"] < cuts["release"]
    d = np.abs(obj[cuts["over_bin"] - 1, :2] - container[0, :2]) + 0.06
    assert np.any(d > np.array([0.21, 0.14]) + 0.01)


def test_cuts_respect_container_yaw():
    actions, hand, obj, container, grasp_end = _episode()
    settings = RepairSettings(lift_clearance_m=0.03, over_bin=True)
    plain = source_cuts(actions, hand, obj, container, grasp_end, GEOMETRY, settings)["over_bin"]
    turned = container.copy()
    turned[:, 3:] = _quat(math.pi / 2)  # the long side now runs along y, so the carry from +y enters later
    assert source_cuts(actions, hand, obj, turned, grasp_end, GEOMETRY, settings)["over_bin"] != plain


def test_missing_cuts_are_refused():
    actions, hand, obj, container, grasp_end = _episode()
    with pytest.raises(ValueError, match="never raised"):
        source_cuts(actions, hand, obj, container, grasp_end, GEOMETRY, RepairSettings(lift_clearance_m=0.5))
    far = container.copy()
    far[:, 0] = 2.0
    with pytest.raises(ValueError, match="never over the container"):
        source_cuts(actions, hand, obj, far, grasp_end, GEOMETRY, RepairSettings(lift_clearance_m=0.03, over_bin=True))
    with pytest.raises(ValueError, match="requires lift"):
        RepairSettings(over_bin=True)


def test_boundary_edits_keep_the_randomized_offset_and_order():
    base = np.array([[[0, 26], [26, 60]], [[0, 30], [30, 70]]])
    randomized = base.copy()
    randomized[:, 0, 1] += [2, 0]
    randomized[:, 1, 0] = randomized[:, 0, 1]
    cuts = [{"approach": 10, "lift": 31, "over_bin": 45, "release": 50},
            {"approach": None, "lift": 33, "over_bin": None, "release": 60}]
    out = edit_boundaries(randomized, base, cuts, RepairSettings(lift_clearance_m=0.03))
    assert out.tolist() == [[[10, 33], [45, 60]], [[0, 33], [33, 70]]]
    assert randomized[0, 0, 1] == 28  # input untouched
    nothing = [{"approach": None, "lift": None, "over_bin": None, "release": 50}] * 2
    assert np.array_equal(edit_boundaries(randomized, base, nothing, RepairSettings()), randomized)
    with pytest.raises(ValueError, match="after the release"):
        edit_boundaries(randomized, base, [{**cuts[0], "over_bin": 52}, cuts[1]], RepairSettings())
    with pytest.raises(ValueError, match="not ordered"):
        edit_boundaries(randomized, base, [{**cuts[0], "approach": 40}, cuts[1]], RepairSettings())


def _pose(xyz, yaw=0.0):
    out = np.eye(4)
    out[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    out[:3, 3] = xyz
    return out


def test_connection_steps_bound_speed_and_rotation_never_below_the_fixed_count():
    assert connection_steps(_pose([0, 0, 0]), _pose([0.05, 0, 0]), 15, 0.25, 0.5, 15) == 15
    assert connection_steps(_pose([0, 0, 0]), _pose([0.9, 0, 0]), 15, 0.25, 0.5, 15) == 54  # 3.6 s
    assert connection_steps(_pose([0, 0, 0]), _pose([0, 0, 0], 2.0), 15, 0.25, 0.5, 15) == 60  # 4 s


def test_retime_replaces_only_the_connection():
    def interpolate(start, end, n):
        return [start + (end - start) * k / (n + 1) for k in range(n + 2)]

    waypoint = lambda pose, g=1.0, noise=0.0: SimpleNamespace(pose=pose, gripper_action=g, noise=noise)
    start, first = np.zeros(1), np.ones(1)
    fixed = 3
    sequence = [waypoint(first * k / (fixed + 1), noise=0.5) for k in range(1, fixed + 2)] + \
               [waypoint(first + k, g=-1.0) for k in (1, 2)]
    assert np.allclose(sequence[fixed].pose, first)
    out = retime(sequence, start, fixed, 7, interpolate)
    assert len(out) == 8 + 2 and out[-2:] == sequence[-2:]
    assert np.allclose([w.pose[0] for w in out[:8]], np.arange(1, 9) / 8)
    assert all(w.noise == 0.5 and w.gripper_action == sequence[fixed].gripper_action for w in out[:7])
    assert retime(sequence, start, fixed, fixed, interpolate) == sequence


def test_remove_yaw_keeps_tilt():
    assert np.allclose(remove_yaw(_quat(1.2)), [1, 0, 0, 0])
    tilt = [math.cos(0.1), math.sin(0.1), 0, 0]  # 11.5 degrees about x
    w, x, y, z = _quat(2.0)
    a0, a1, a2, a3 = tilt
    turned = [w * a0 - z * a3, w * a1 - z * a2, w * a2 + z * a1, w * a3 + z * a0]  # yaw * tilt
    out = remove_yaw(np.array([turned, tilt]))
    assert np.allclose(out[0], tilt, atol=1e-9) and np.allclose(out[1], tilt)


def test_bearing_reference_sets_heading_to_the_bearing_and_keeps_tilt():
    from robolab.core.omy_teleop.repairs import bearing_reference
    tilt = [math.cos(0.1), math.sin(0.1), 0, 0]
    w, x, y, z = _quat(2.0)
    a0, a1, a2, a3 = tilt
    pose = [0.0, 0.5, 0.1, w * a0 - z * a3, w * a1 - z * a2, w * a2 + z * a1, w * a3 + z * a0]  # yaw 2.0, bearing 90 deg
    out = bearing_reference(np.array([pose, [0.5, -0.5, 0, *_quat(0.3)]]))
    w, x, y, z = _quat(math.pi / 2)
    assert np.allclose(out[0, 3:], [w * a0 - z * a3, w * a1 - z * a2, w * a2 + z * a1, w * a3 + z * a0])
    assert np.allclose(out[1, 3:], _quat(-math.pi / 4)) and np.allclose(out[:, :3], [pose[:3], [0.5, -0.5, 0]])
