"""Action-aligned L100 input snapshots; no simulator or hardware dependencies."""

import math

import torch


LEADER_JOINT_NAMES = tuple(f"joint_{i}" for i in range(1, 7))


def leader_state_metadata():
    return {
        "schema_version": 1,
        "dataset_group": "leader",
        "alignment": "one row per action, from the exact sample consumed by device.advance",
        "joint_names": list(LEADER_JOINT_NAMES),
        "joint_position_units": "rad",
        "joint_position_reference": "source reading before wrist turn correction, calibration, mapping or limiting",
        "gripper_position_units": "source-native continuous reading; serial uses legacy half-angle radians",
        "serial_joint_conversion": "(present_ticks - 2048) * 2*pi / 4096",
        "serial_gripper_conversion": "(present_ticks - goal_ticks) * pi / 4095",
        "serial_gripper_goal_ticks": "int(device.gripper_open_pos / 100 * 4095)",
        "wrist_turn_offset_rad": "subtract from raw joint_6 before applying the saved calibration",
        "valid": "false for absent, stale, incomplete or nonfinite samples; numeric sentinels must be ignored",
        "invalid_position_fill": 0.0,
        "invalid_time_fill": -1,
        "source_wall_ns": "source buffer timestamp, Unix wall clock; not a hardware encoder timestamp",
        "read_monotonic_ns": "host monotonic time when device.advance consumed the source sample",
        "source_age_s": "sample age at consumption, seconds",
        "sampling": "control rate; intermediate reader samples are not recorded",
    }


def leader_state_tensors(sample):
    """Own CPU float64 snapshots with a stable shape, including when input is absent."""
    sample = sample or {}
    joints = sample.get("raw_joint_position", ())
    gripper = sample.get("raw_gripper_position")
    offset = sample.get("wrist_turn_offset_rad", 0.0)
    valid = (
        not sample.get("stale", True)
        and len(joints) == 6
        and gripper is not None
        and all(math.isfinite(value) for value in (*joints, gripper, offset))
    )
    return {
        "joint_position": torch.tensor([joints if valid else (0.0,) * 6], dtype=torch.float64),
        "gripper_position": torch.tensor([[gripper if valid else 0.0]], dtype=torch.float64),
        "wrist_turn_offset_rad": torch.tensor([offset if valid else 0.0], dtype=torch.float64),
        "valid": torch.tensor([valid], dtype=torch.bool),
        "source_wall_ns": torch.tensor([(sample.get("source_wall_ns") or -1) if valid else -1],
                                       dtype=torch.int64),
        "read_monotonic_ns": torch.tensor([sample.get("read_monotonic_ns", -1) if valid else -1],
                                          dtype=torch.int64),
        "source_age_s": torch.tensor([sample.get("age_s", -1.0) if valid else -1.0], dtype=torch.float64),
    }
