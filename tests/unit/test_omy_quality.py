"""Demonstration quality metrics and filtering, independent of Isaac Sim."""

import json

import h5py
import numpy as np
import pytest

from robolab.core.omy_teleop.quality import (THRESHOLDS, assess, demo_metrics, difference_peaks, generation_records,
                                             main, write_kept)

HZ = 15.0
HOME = np.array([0.0, -0.6, 0.0, -2.5, 0.0, 1.9, 0.0])


def _episode(steps=20, chatter=0.0, mustard_shift=0.0):
    ramp = np.linspace(0.0, 0.2, steps)[:, None]
    actions = np.concatenate((HOME + ramp, np.ones((steps, 1))), axis=1)
    actions[::2, 3] += chatter
    joints = np.zeros((steps, 13))
    joints[:, :7] = HOME + ramp
    cheez_it, mustard = np.zeros((steps, 7)), np.zeros((steps, 7))
    cheez_it[:, 0] = np.linspace(0.0, 0.5, steps)
    mustard[-1, 1] = mustard_shift
    return {
        "actions": actions,
        "initial_state": {"articulation": {"robot": {"joint_position": joints[:1] * 0 + np.r_[HOME, np.zeros(6)]}},
                          "rigid_object": {name: {"root_pose": np.zeros((1, 7))} for name in ("cheez_it", "mustard")}},
        "states": {"articulation": {"robot": {"joint_position": joints, "joint_velocity": np.full((steps, 13), 0.5)}},
                   "rigid_object": {"cheez_it": {"root_pose": cheez_it}, "mustard": {"root_pose": mustard}}},
    }


def _write(group, tree):
    for key, value in tree.items():
        if isinstance(value, dict):
            _write(group.create_group(key), value)
        else:
            group[key] = value


def _recording(path, episodes, generated=True):
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs["robolab_omy_teleop"] = json.dumps({"control_hz": HZ})
        if generated:
            data.attrs["robolab_omy_mimicgen"] = json.dumps({"object": "cheez_it"})
        for index, (tree, success) in enumerate(episodes):
            group = data.create_group(f"demo_{index}")
            _write(group, tree)
            group.attrs["num_samples"], group.attrs["success"] = len(tree["actions"]), success
        data.attrs["total"] = sum(len(tree["actions"]) for tree, _ in episodes)
    return path


def _limits(**changes):
    return {**{name: default for name, (default, _, _) in THRESHOLDS.items()}, **changes}


def test_difference_peaks_scale_with_the_control_rate():
    velocity, acceleration, jerk = difference_peaks(np.array([[0.0], [1.0], [0.0], [1.0]]), 10.0)
    assert (velocity, acceleration, jerk) == (10.0, 200.0, 4000.0)
    assert difference_peaks(np.zeros((2, 7)), HZ) == (0.0, 0.0, 0.0)  # too short for higher differences


def test_metrics_follow_commands_limits_and_other_objects():
    smooth = demo_metrics(_episode(), HZ, grasped=["cheez_it"])
    assert smooth["command_velocity"] == pytest.approx(0.2 / 19 * HZ)
    assert smooth["joint_velocity"] == 0.5 and smooth["limit_excess"] < 0
    assert smooth["displacement"] == 0 and smooth["displaced_object"] == "mustard"
    rough = demo_metrics(_episode(chatter=2.6, mustard_shift=0.1), HZ, grasped=["cheez_it"])
    assert rough["command_velocity"] > 30 and rough["command_jerk"] > rough["command_acceleration"] * HZ
    # Joint 4 at its last chattering step, 18 of 19 along the ramp, against its -0.0698 limit.
    assert rough["limit_excess"] == pytest.approx(-2.5 + 0.2 * 18 / 19 + 2.6 + 0.0698)
    assert rough["displacement"] == pytest.approx(0.1)
    # The grasped object's own transport is not a disturbance, and is unknown without its name.
    assert "displacement" not in demo_metrics(_episode(), HZ)
    assert demo_metrics(_episode(), HZ, grasped=[])["displaced_object"] == "cheez_it"


def test_assessment_combines_recording_and_generation_records(tmp_path):
    path = _recording(tmp_path / "omy_demos.hdf5", [
        (_episode(), True), (_episode(chatter=2.6), True), (_episode(), False), (_episode(), True), (_episode(), True)])
    summary = {"attempts": [
        {"demo": "demo_0", "ik_position_error_m": 0.001, "ik_orientation_error_rad": 0.01, "events": []},
        {"demo": None, "ik_position_error_m": 9.0, "ik_orientation_error_rad": 9.0, "events": []},
        {"demo": "demo_3", "ik_position_error_m": 0.5, "ik_orientation_error_rad": 0.01,
         "events": [{"step": 3, "name": "OBJECT_GRABBED_SUCCESS", "info": ""}]},
        {"demo": "demo_4", "ik_position_error_m": 0.001, "ik_orientation_error_rad": 0.01,
         "events": [{"step": 3, "name": "GRIPPER_HIT_TABLE", "info": ""}]},
    ]}
    assert set(generation_records(summary)) == {"demo_0", "demo_3", "demo_4"}
    results = assess(path, _limits(), summary)
    assert results["demo_0"]["failed"] == []
    assert {"command_velocity", "command_acceleration", "command_jerk", "limit_excess"} <= set(results["demo_1"]["failed"])
    assert "ik_position_error" not in results["demo_1"]["metrics"]  # not recorded for this episode
    assert results["demo_2"]["failed"] == ["unsuccessful"]
    assert results["demo_3"]["failed"] == ["ik_position_error"] and results["demo_3"]["metrics"]["events"] == []
    assert results["demo_4"]["failed"] == ["GRIPPER_HIT_TABLE"]
    assert assess(path, _limits(), summary, events=())["demo_4"]["failed"] == []
    assert assess(path, _limits(command_velocity=np.inf, command_acceleration=np.inf, command_jerk=np.inf,
                                limit_excess=np.inf))["demo_1"]["failed"] == []


def test_displacement_needs_the_grasped_object(tmp_path):
    path = _recording(tmp_path / "omy_demos.hdf5", [(_episode(mustard_shift=0.1), True)], generated=False)
    assert assess(path, _limits())["demo_0"]["failed"] == []
    assert assess(path, _limits(), grasped=["cheez_it"])["demo_0"]["failed"] == ["displacement"]


def test_filtered_recording_keeps_names_and_counts_samples(tmp_path, capsys):
    path = _recording(tmp_path / "omy_demos.hdf5", [
        (_episode(), True), (_episode(chatter=2.6), True), (_episode(steps=12), True)])
    output, report = tmp_path / "kept.hdf5", tmp_path / "report.json"
    assert main([str(path), "--output", str(output), "--report", str(report)]) == 0
    assert "Accepted 2 of 3 episodes." in capsys.readouterr().out
    with h5py.File(output, "r") as file:
        data = file["data"]
        assert sorted(data) == ["demo_0", "demo_2"] and data.attrs["total"] == 32
        assert json.loads(data.attrs["robolab_omy_quality"])["accepted"] == 2
        assert np.array_equal(data["demo_2/actions"][:], _episode(steps=12)["actions"])
        assert bool(data["demo_2"].attrs["success"]) and "robolab_omy_mimicgen" in data.attrs
    saved = json.loads(report.read_text())
    assert saved["rejected"]["command_velocity"] == 1 and saved["episodes"]["demo_1"]["failed"]
    with pytest.raises(FileExistsError):
        write_kept(path, output, ["demo_0"], {})
