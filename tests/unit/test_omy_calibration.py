"""READY anchoring through the upstream pure-numpy retargeter; no hardware."""

import math
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("omy_leader_isaaclab.franka_config")
from omy_leader_isaaclab.franka_config import DEFAULT_OMY_TO_FRANKA_CONFIG, load_calib
from omy_leader_isaaclab.franka_retarget import OmyToFrankaRetarget

from robolab.core.omy_teleop.calibration import recenter_direct, validate_home


HOME = tuple(math.radians(q) for q in (0, -36, 0, -144, 0, 108, 0))
READY = {**{f"joint_{i}.pos": q for i, q in enumerate((-.025, 1.548, 2.395, -.606, 1.600, -.014), 1)},
         "gripper.pos": 0.0}


@pytest.mark.parametrize("j6_reference", [math.pi / 2, 3 * math.pi / 5, math.pi])
def test_new_ready_maps_exactly_to_droid_home_without_changing_slopes(j6_reference):
    base = replace(DEFAULT_OMY_TO_FRANKA_CONFIG, j6_reset_rad=j6_reference,
                   omy_sign={f"joint_{i}": (-1.0 if i in (2, 4) else 1.0) for i in range(1, 7)},
                   gripper_open_rad=0.0, gripper_closed_rad=0.6)
    updated = recenter_direct(base, READY, HOME)
    previous = OmyToFrankaRetarget(base)
    mapper = OmyToFrankaRetarget(updated, initial_q=HOME)
    np.testing.assert_allclose(mapper.raw_target(READY), HOME, atol=1e-12)
    np.testing.assert_allclose(mapper.step(READY), [*HOME, 1.0], atol=1e-12)
    assert mapper.gripper_command(0.6) == -1.0
    for key, value in asdict(base).items():
        if key != "omy_zero_rad":
            assert asdict(updated)[key] == value
    for i in range(1, 7):
        moved = {**READY, f"joint_{i}.pos": READY[f"joint_{i}.pos"] + 0.01}
        np.testing.assert_allclose(mapper.raw_target(moved) - mapper.raw_target(READY),
                                   previous.raw_target(moved) - previous.raw_target(READY), atol=1e-12)


@pytest.mark.parametrize("target", [HOME[:6], (*HOME[:6], float("nan")), (9.0, *HOME[1:]),
                                    (*HOME[:2], 0.1, *HOME[3:])])
def test_invalid_home_rejected(target):
    with pytest.raises(ValueError):
        validate_home(DEFAULT_OMY_TO_FRANKA_CONFIG, target)


def test_recenter_rejects_unsupported_mode_or_bad_capture():
    with pytest.raises(ValueError, match="direct"):
        recenter_direct(replace(DEFAULT_OMY_TO_FRANKA_CONFIG, wrist_mode="zyz"), READY, HOME)
    with pytest.raises(ValueError, match="joint_4"):
        recenter_direct(DEFAULT_OMY_TO_FRANKA_CONFIG, {**READY, "joint_4.pos": float("inf")}, HOME)
    with pytest.raises(ValueError, match="joint_2"):
        recenter_direct(replace(DEFAULT_OMY_TO_FRANKA_CONFIG, omy_sign={"joint_2": 0.0}), READY, HOME)


def test_calibrated_wrist_rotation_direction_preserves_ready():
    calibration_path = Path(__file__).resolve().parents[2] / "omy_calib_robolab.json"
    calibration = json.loads(calibration_path.read_text())
    config = load_calib(str(calibration_path))
    home = tuple(calibration["ready_franka_joint_pos_rad"])
    ready = {f"{name}.pos": value for name, value in calibration["ready_reading_rad"].items()
             if name.startswith("joint_")}
    ready["gripper.pos"] = calibration["ready_reading_rad"]["gripper"]
    mapper = OmyToFrankaRetarget(config, initial_q=home)

    np.testing.assert_allclose(mapper.step(ready), [*home, 1.0], atol=1e-9)
    rotated = {**ready, "joint_6.pos": ready["joint_6.pos"] + 0.1}
    np.testing.assert_allclose(mapper.raw_target(rotated)[:6], home[:6], atol=1e-9)
    assert mapper.raw_target(rotated)[6] == pytest.approx(home[6] + 0.1)
