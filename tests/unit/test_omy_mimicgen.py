"""MimicGen source preparation, independent of Isaac Sim."""

import numpy as np
import pytest

from robolab.core.omy_teleop.mimicgen import (build_parser, grasp_index, pre_action, source_arrays, teleop_args,
                                              validate_args)


def test_grasp_requires_closed_gripper_and_lift():
    gripper = np.array([1, 1, -1, -1, -1, 1])
    height = np.array([0.02, 0.02, 0.02, 0.03, 0.06, 0.06])
    assert grasp_index(gripper, height) == 4
    with pytest.raises(ValueError):
        grasp_index(gripper, np.full(6, 0.02))
    with pytest.raises(ValueError):
        grasp_index(np.ones(6), height)


def _episode(steps=5):
    def pose(offset):
        rows = np.zeros((steps, 7), dtype=np.float32)
        rows[:, 2] = offset + 0.03 * np.arange(1, steps + 1)
        return rows

    actions = np.zeros((steps, 8), dtype=np.float32)
    actions[:, 7] = (1, -1, -1, -1, 1)
    joints = np.arange(1, steps + 1, dtype=np.float32)[:, None].repeat(13, 1)
    return {
        "actions": actions,
        "initial_state": {"articulation": {"robot": {"joint_position": np.zeros((1, 13), dtype=np.float32)}},
                          "rigid_object": {name: {"root_pose": np.zeros((1, 7), dtype=np.float32)}
                                           for name in ("banana", "bowl")}},
        "states": {"articulation": {"robot": {"joint_position": joints}},
                   "rigid_object": {"banana": {"root_pose": pose(0.0)}, "bowl": {"root_pose": pose(1.0)}}},
    }


def test_source_rows_are_pre_action():
    arrays = source_arrays(_episode(), "banana", "bowl")
    # Row t is the scene before action t: the initial state, then post-step state t - 1.
    assert arrays["joint_position"][:, 0].tolist() == [0, 1, 2, 3, 4]
    assert np.allclose(arrays["object_pose"]["banana"][:, 2], [0, 0.03, 0.06, 0.09, 0.12])
    assert arrays["object_pose"]["bowl"][1, 2] == pytest.approx(1.03)
    assert arrays["grasp"] == 1
    assert pre_action(np.zeros((1, 2)), np.ones((3, 2))).tolist() == [[0, 0], [1, 1], [1, 1]]


def test_misaligned_episode_is_refused():
    episode = _episode()
    episode["actions"] = episode["actions"][:-1]
    with pytest.raises(ValueError):
        source_arrays(episode, "banana", "bowl")


def test_teleop_options_follow_the_recording():
    recorded = {"decimation": 8, "episode_length_s": 120.0, "omy_stepping_profile": "held-target",
                "omy_recording_profile": "compact", "omy_contact_sensor_profile": "required",
                "sim": {"render_interval": 4, "physx": {"max_position_iteration_count": 16}}}
    args = teleop_args("BananaInBowlTask", recorded)
    assert (args.control_hz, args.stepping_profile, args.recording_profile) == (15, "held-target", "compact")
    assert (args.contact_sensors, args.render_interval, args.solver_position_iterations) == ("required", 4, 16)
    recorded["sim"]["physx"] = {}
    assert teleop_args("BananaInBowlTask", recorded).solver_position_iterations is None


def test_arguments_require_complete_source_runs(tmp_path):
    parser = build_parser()
    with pytest.raises(SystemExit):
        validate_args(parser, parser.parse_args(["--input", str(tmp_path)]))
    (tmp_path / "omy_demos.hdf5").touch()
    (tmp_path / "env_cfg.json").touch()
    validate_args(parser, parser.parse_args(["--input", str(tmp_path)]))
    with pytest.raises(SystemExit):
        validate_args(parser, parser.parse_args(["--input", str(tmp_path), "--num-demos", "5", "--max-attempts", "2"]))
    assert parser.parse_args(["--input", str(tmp_path), "--episodes", "1", "4"]).episodes == [1, 4]
    with pytest.raises(SystemExit):
        validate_args(parser, parser.parse_args(["--input", str(tmp_path), str(tmp_path), "--episodes", "1"]))


def test_object_ranges_default_to_the_shared_ranges(tmp_path):
    (tmp_path / "omy_demos.hdf5").touch()
    (tmp_path / "env_cfg.json").touch()
    parser = build_parser()
    args = parser.parse_args(["--input", str(tmp_path), "--pose-range", "0.05", "--object-yaw-range", "1.5"])
    validate_args(parser, args)
    assert (args.object_pose_range, args.object_yaw_range, args.yaw_range) == (0.05, 1.5, 0.0)
    with pytest.raises(SystemExit):
        validate_args(parser, parser.parse_args(["--input", str(tmp_path), "--object-pose-range", "-0.1"]))
