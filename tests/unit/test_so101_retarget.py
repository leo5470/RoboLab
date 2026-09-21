"""Pure retargeting checks; run with --confcutdir=tests/unit."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robolab.core.teleop.retarget import (
    PoseRetargeter,
    RetargetConfig,
    TrackingError,
    quaternion,
    validate_action_config,
)
from robolab.core.teleop.so101_source import LeaderSample


def sample(seq=0, position=(0, 0, 0), quat=(1, 0, 0, 0), gripper=100):
    return LeaderSample(
        "test", seq, 1.0, 1.0, np.zeros(5), gripper, np.array(position, dtype=float), np.array(quat, dtype=float)
    )


def test_anchor_has_no_jump_and_scale_is_applied_once():
    mapper = PoseRetargeter(RetargetConfig(filter_alpha=1))
    robot = np.array([0.4, 0.1, 0.3])
    mapper.anchor(sample(position=(1, 2, 3)), robot, [1, 0, 0, 0])
    np.testing.assert_array_equal(mapper.command(sample(position=(1, 2, 3)), robot, [1, 0, 0, 0]), 0)
    moved = sample(1, (1.005, 2, 3))
    action = mapper.command(moved, robot, [1, 0, 0, 0])
    assert action[0] == pytest.approx(0.01)
    # Same absolute sample must not integrate additional leader motion.
    np.testing.assert_allclose(mapper.command(moved, robot, [1, 0, 0, 0]), action)
    np.testing.assert_allclose(mapper.command(moved, robot + [0.005, 0, 0], [1, 0, 0, 0]), 0, atol=1e-8)


def test_alignment_rotates_translation_and_rotational_error_on_root_axes():
    alignment = Rotation.from_euler("z", 90, degrees=True)
    cfg = RetargetConfig(filter_alpha=1, orientation_mode="pose", alignment_wxyz=quaternion(alignment))
    mapper = PoseRetargeter(cfg)
    robot_rotation = Rotation.from_euler("y", 0.5)
    mapper.anchor(sample(), [0, 0, 0], quaternion(robot_rotation))
    leader_rotation = Rotation.from_rotvec([0.02, 0, 0])
    action = mapper.command(
        sample(1, (0.002, 0, 0), quaternion(leader_rotation)), [0, 0, 0], quaternion(robot_rotation)
    )
    np.testing.assert_allclose(action[:3], [0, 0.004, 0], atol=1e-8)
    np.testing.assert_allclose(action[3:6], [0, 0.04, 0], atol=1e-8)


def test_quaternion_sign_and_wrap_use_shortest_rotation():
    mapper = PoseRetargeter(RetargetConfig(filter_alpha=1, orientation_mode="pose"))
    a = quaternion(Rotation.from_euler("z", 179, degrees=True))
    b = quaternion(Rotation.from_euler("z", -179, degrees=True))
    mapper.anchor(sample(quat=a), [0, 0, 0], [1, 0, 0, 0])
    result = mapper.command(sample(1, quat=-b), [0, 0, 0], [1, 0, 0, 0])
    assert result[5] * 0.5 == pytest.approx(np.deg2rad(2), abs=1e-7)


def test_fixed_orientation_and_bounds():
    mapper = PoseRetargeter(RetargetConfig(filter_alpha=1))
    mapper.anchor(sample(), [0, 0, 0], [1, 0, 0, 0])
    action = mapper.command(sample(1, (10, 0, 0), quaternion(Rotation.from_euler("x", 1))), [0, 0, 0], [1, 0, 0, 0])
    assert mapper.target_pos[0] == pytest.approx(0.3)
    assert np.linalg.norm(action[:3]) * mapper.scale == pytest.approx(0.01)
    np.testing.assert_array_equal(action[3:6], 0)


def test_obstructed_target_pauses_after_configured_error_duration():
    mapper = PoseRetargeter(RetargetConfig(filter_alpha=1, tracking_error_steps=2))
    mapper.anchor(sample(), [0, 0, 0], [1, 0, 0, 0])
    mapper.command(sample(1, (0.3, 0, 0)), [0, 0, 0], [1, 0, 0, 0])
    with pytest.raises(TrackingError):
        mapper.command(sample(2, (0.3, 0, 0)), [0, 0, 0], [1, 0, 0, 0])


def test_gripper_hysteresis_and_resume_matching():
    mapper = PoseRetargeter(RetargetConfig())
    mapper.anchor(sample(), [0, 0, 0], [1, 0, 0, 0])
    for seq, (reading, closed) in enumerate([(20, True), (50, True), (80, False), (50, False)], 1):
        result = mapper.command(sample(seq, gripper=reading), [0, 0, 0], [1, 0, 0, 0])
        assert result[6] == float(closed)
    with pytest.raises(ValueError, match="physical gripper"):
        mapper.anchor(sample(10, gripper=0), [0, 0, 0], [1, 0, 0, 0])
    mapper.reset()
    assert not mapper.anchored and not mapper.closed


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_timeout_s": float("nan")},
        {"filter_alpha": 0},
        {"translation_gain": -1},
        {"gripper_open": 0, "gripper_closed": 0},
        {"tracking_error_steps": 0},
        {"workspace_min_offset_m": [1, 1, 1]},
        {"alignment_wxyz": [0, 0, 0, 0]},
        {"orientation_mode": "euler"},
        {"gripper_close_threshold": 0.1},
    ],
)
def test_bad_config_fails_early(changes):
    with pytest.raises(ValueError):
        RetargetConfig(**changes)


def test_config_rejects_misspelled_settings(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("translation_gian: 2")
    with pytest.raises(ValueError, match="Unknown"):
        RetargetConfig.load(path)


def action_cfg():
    return SimpleNamespace(
        arm_action=SimpleNamespace(
            asset_name="robot",
            body_name="base_link",
            scale=0.5,
            clip=None,
            controller=SimpleNamespace(command_type="pose", use_relative_mode=True),
            body_offset=SimpleNamespace(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        ),
        finger_joint=SimpleNamespace(asset_name="robot", class_type=type("BinaryJointPositionZeroToOneAction", (), {})),
    )


def test_runtime_contract_rejects_rotated_control_frame_or_extra_scaling():
    cfg = action_cfg()
    assert validate_action_config(cfg) == 0.5
    cfg.arm_action.body_offset.rot = (0.5, -0.5, 0.5, -0.5)
    with pytest.raises(ValueError, match="identity"):
        validate_action_config(cfg)
    cfg = action_cfg()
    cfg.arm_action.scale = [0.5] * 6
    with pytest.raises(ValueError, match="scalar"):
        validate_action_config(cfg)


def test_wrist_roll_uses_local_tool_axis_and_ignores_leader_tilt():
    from dataclasses import replace

    mapper = PoseRetargeter(RetargetConfig(orientation_mode="wrist-roll", filter_alpha=1, translation_gain=2))
    robot_rot = Rotation.from_euler("zy", [70, 40], degrees=True)
    initial = replace(sample(), joints_deg=np.array([0, 0, 0, 0, 80.0]))
    mapper.anchor(initial, [0, 0, 0], quaternion(robot_rot))
    np.testing.assert_allclose(mapper.command(initial, [0, 0, 0], quaternion(robot_rot)), 0, atol=1e-8)
    moved = replace(
        sample(1, (0.002, 0, 0), quaternion(Rotation.from_euler("y", 1.0)), gripper=0),
        joints_deg=np.array([30, 40, 50, 60, 82.0]),
    )
    action = mapper.command(moved, [0, 0, 0], quaternion(robot_rot))
    np.testing.assert_allclose(action[:3] * mapper.scale, [0.004, 0, 0], atol=1e-8)
    np.testing.assert_allclose(
        action[3:6] * mapper.scale, robot_rot.apply([np.deg2rad(2), 0, 0]), atol=1e-8
    )
    np.testing.assert_allclose(mapper.target_rot.apply([1, 0, 0]), robot_rot.apply([1, 0, 0]), atol=1e-8)
    assert action[6] == 1
    # Resume must discard the old wrist angle and capture the robot's new pose.
    resumed_rot = Rotation.from_euler("yz", [20, -50], degrees=True)
    mapper.anchor(moved, [0.1, 0, 0], quaternion(resumed_rot))
    resumed = mapper.command(moved, [0.1, 0, 0], quaternion(resumed_rot))
    np.testing.assert_allclose(resumed[:6], 0, atol=1e-8)
    assert resumed[6] == 1


def test_wrist_roll_wrap_filter_and_repeated_sample():
    from dataclasses import replace

    mapper = PoseRetargeter(RetargetConfig(orientation_mode="wrist-roll", filter_alpha=0.5))
    initial = replace(sample(), joints_deg=np.array([0, 0, 0, 0, 179.0]))
    mapper.anchor(initial, [0, 0, 0], [1, 0, 0, 0])
    moved = replace(sample(1), joints_deg=np.array([0, 0, 0, 0, -179.0]))
    action = mapper.command(moved, [0, 0, 0], [1, 0, 0, 0])
    np.testing.assert_allclose(action[3:6] * mapper.scale, [np.deg2rad(1), 0, 0], atol=1e-8)
    np.testing.assert_allclose(mapper.command(moved, [0, 0, 0], [1, 0, 0, 0]), action)
