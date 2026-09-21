"""Standalone hardware adapter and URDF FK tests with a fake LeRobot bus."""

import sys
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.so101_leader_reader import JOINT_NAMES, UrdfKinematics, connect_leader


@pytest.fixture
def urdf(tmp_path):
    parts = ['<robot name="test"><link name="base_link"/>']
    parent = "base_link"
    for index, name in enumerate(JOINT_NAMES):
        child = f"link_{index}"
        parts.append(
            f'<link name="{child}"/><joint name="{name}" type="revolute">'
            f'<parent link="{parent}"/><child link="{child}"/>'
            '<origin xyz="0.1 0 0"/><axis xyz="0 0 1"/></joint>'
        )
        parent = child
    parts.append(
        f'<link name="gripper_frame_link"/><joint name="tool" type="fixed">'
        f'<parent link="{parent}"/><child link="gripper_frame_link"/>'
        '<origin xyz="0.1 0 0"/></joint></robot>'
    )
    path = tmp_path / "arm.urdf"
    path.write_text("".join(parts))
    return path


def test_fk_joint_order_degrees_signs_and_offsets(urdf):
    fk = UrdfKinematics(urdf)
    position, quat = fk.pose([0] * 5)
    np.testing.assert_allclose(position, [0.6, 0, 0], atol=1e-9)
    np.testing.assert_allclose(quat, [1, 0, 0, 0])
    position, quat = fk.pose([90, 0, 0, 0, 0])
    np.testing.assert_allclose(position, [0.1, 0.5, 0], atol=1e-9)
    np.testing.assert_allclose(quat, [2**-0.5, 0, 0, 2**-0.5], atol=1e-9)
    corrected = UrdfKinematics(urdf, signs=[-1, 1, 1, 1, 1], offsets=[180, 0, 0, 0, 0])
    np.testing.assert_allclose(corrected.pose([90, 0, 0, 0, 0])[0], position)
    assert len(fk.checksum) == 64


def test_fk_rejects_wrong_chain_or_gripper_in_chain(urdf):
    with pytest.raises(ValueError, match="chain"):
        UrdfKinematics(urdf, "nonexistent")
    urdf.write_text(urdf.read_text().replace('name="wrist_roll"', 'name="gripper"'))
    with pytest.raises(ValueError, match="five arm joints"):
        UrdfKinematics(urdf)


@pytest.mark.parametrize(
    "calibrated,has_calibration,raises", [(True, True, False), (False, True, True), (False, False, True)]
)
def test_connect_never_calibrates_or_enables_torque(monkeypatch, calibrated, has_calibration, raises):
    events = []

    class Leader:
        def __init__(self, cfg):
            assert cfg.use_degrees
            self.calibration = {"test": 1} if has_calibration else {}
            self.is_connected = False
            self.is_calibrated = calibrated

        def connect(self, calibrate=True):
            assert calibrate is False
            events.append("connect")
            self.is_connected = True

        def disconnect(self):
            events.append("disconnect")
            self.is_connected = False

    module = SimpleNamespace(SO101Leader=Leader, SO101LeaderConfig=lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setitem(sys.modules, "lerobot.teleoperators.so_leader", module)
    args = Namespace(port="usb", id="test", calibration_dir="/calibration")
    if raises:
        with pytest.raises(ValueError, match="calibration"):
            connect_leader(args)
        assert events == (["connect", "disconnect"] if has_calibration else [])
    else:
        leader = connect_leader(args)
        assert leader.is_connected and events == ["connect"]


@pytest.mark.parametrize("pan_sign", [-1, 1])
def test_base_pan_gain_isolates_joint_motion_and_reanchors(urdf, pan_sign):
    from dataclasses import replace

    from robolab.core.teleop.retarget import PoseRetargeter, RetargetConfig
    from robolab.core.teleop.so101_source import LeaderSample

    fk = UrdfKinematics(urdf, signs=[pan_sign, 1, 1, 1, 1], offsets=[30, 0, 0, 0, 0])

    def sample(seq, joints):
        position, quat = fk.pose(joints)
        return LeaderSample("test", seq, 0, 0, np.array(joints, dtype=float), 100, np.asarray(position), np.asarray(quat))

    cfg = RetargetConfig(translation_gain=2, base_pan_gain=4, filter_alpha=1, orientation_mode="wrist-roll")
    mapper = PoseRetargeter(cfg, leader_fk=fk.pose)
    baseline = PoseRetargeter(replace(cfg, base_pan_gain=1))
    initial = sample(0, [20, 10, 0, 0, 0])
    for m in (mapper, baseline):
        m.anchor(initial, [0, 0, 0], [1, 0, 0, 0])
    pan = sample(1, [21, 10, 0, 0, 0])
    for m in (mapper, baseline):
        m.command(pan, [0, 0, 0], [1, 0, 0, 0])
    np.testing.assert_allclose(mapper.target_pos, 4 * baseline.target_pos, atol=1e-9)
    # Other-joint movement with unchanged pan keeps the original response.
    other = sample(2, [20, 11, 0, 0, 2])
    for m in (mapper, baseline):
        m.command(other, [0, 0, 0], [1, 0, 0, 0])
    np.testing.assert_allclose(mapper.target_pos, baseline.target_pos, atol=1e-9)
    np.testing.assert_allclose(mapper.target_rot.as_matrix(), baseline.target_rot.as_matrix(), atol=1e-9)
    # Combined motion adds only the additional base-joint contribution.
    combined = sample(3, [21, 11, 0, 0, 2])
    mapper.command(combined, [0, 0, 0], [1, 0, 0, 0])
    expected = 2 * ((other.position - initial.position) + 4 * (combined.position - other.position))
    np.testing.assert_allclose(mapper.target_pos, expected, atol=1e-9)
    mapper.anchor(combined, [0.1, 0.2, 0.3], [1, 0, 0, 0])
    np.testing.assert_allclose(mapper.command(combined, [0.1, 0.2, 0.3], [1, 0, 0, 0]), 0, atol=1e-9)


def test_runtime_fk_loads_without_scripts_package(urdf, tmp_path):
    import subprocess

    # Match script launch: no repository root on sys.path, and an unrelated
    # scripts module must not affect the FK import either.
    code = """
import sys
import types
sys.modules['scripts'] = types.ModuleType('scripts')
from robolab.core.teleop.so101_kinematics import leader_kinematics
fk = leader_kinematics(sys.argv[1])
position, quaternion = fk.pose([0] * 5)
assert abs(position[0] - 0.6) < 1e-9
assert quaternion == [1.0, 0.0, 0.0, 0.0]
assert 'lerobot' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-c", code, str(urdf)], cwd=tmp_path, check=True)
