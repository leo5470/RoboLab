"""Upstream OMY contract and DROID simulator integration; no physical leader required."""

import importlib.metadata
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

try:
    importlib.metadata.version("omy-leader-isaaclab")
except importlib.metadata.PackageNotFoundError:
    pytest.skip("Install requirements-omy.txt for OMY integration tests", allow_module_level=True)

import isaaclab.envs.mdp as mdp
from isaaclab.app import AppLauncher
from isaaclab.devices.teleop_device_factory import create_teleop_device
from omy_leader_isaaclab.device import OmyLeaderCfg, OmyLeaderDevice
from omy_leader_isaaclab.franka_config import DEFAULT_OMY_TO_FRANKA_CONFIG, FRANKA_HOME, FRANKA_VEL_MAX
from omy_leader_isaaclab.franka_retarget import OmyToFrankaRetarget

import robolab.constants
from robolab.core.environments.config import parse_env_cfg
from robolab.core.environments.runtime import create_env
from robolab.core.omy_teleop.config import (
    OmyDroidCfg, configure_device, register_omy_envs, validate_environment,
)
from robolab.core.omy_teleop.options import ARM_JOINTS, build_parser, validate_args
from robolab.core.omy_teleop.runner import (
    close_recording, configure_recording, finish_interrupted, operator_frame, prepare_output, run,
)
from robolab.robots.droid import DroidCfg


def _args(*argv):
    parser = build_parser()
    args = parser.parse_args(list(argv))
    validate_args(parser, args)
    return args


def _raw_home():
    # Input for the pinned default DIRECT wrist map, not the optional ZYZ map.
    values = (0.0, -math.pi / 4, 3 * math.pi / 4, math.pi / 2, 0.0, 0.0)
    return {**{f"joint_{i}.pos": value for i, value in enumerate(values, 1)}, "gripper.pos": 0.0}


def test_launcher_arguments_do_not_conflict():
    parser = build_parser(AppLauncher)
    args = parser.parse_args(["--headless", "--source", "tcp"])
    assert args.headless and args.rendering_mode == "performance"


def test_dedicated_config_and_optional_cameras():
    original = DroidCfg().robot.init_state.joint_pos.copy()
    registered = register_omy_envs("BananaInBowlTask", env_postfix="OMYConfigTest", stream=True, images=True)
    assert len(registered) == 1
    cfg = parse_env_cfg("BananaInBowlTaskOMYConfigTest", device="cpu", num_envs=1)
    assert cfg.sim.dt == pytest.approx(1 / 120)
    assert cfg.decimation == 2 and cfg.sim.render_interval == 2
    assert cfg.scene.view_cam.width == 640 and cfg.scene.view_cam.height == 480
    assert cfg.scene.omy_wrist_cam.width == 240 and cfg.scene.omy_wrist_cam.height == 180
    fields = list(vars(cfg.scene))
    assert fields.index("robot") < fields.index("omy_wrist_cam")
    assert cfg.scene.wrist_cam is not None
    assert cfg.observations.image_obs is not None
    assert cfg.actions.body.use_default_offset is False
    assert cfg.actions.body.joint_names == list(ARM_JOINTS)
    assert cfg.actions.finger_joint.class_type is mdp.BinaryJointPositionAction
    assert tuple(OmyDroidCfg().robot.init_state.joint_pos[name] for name in ARM_JOINTS) == FRANKA_HOME
    assert DroidCfg().robot.init_state.joint_pos == original
    device_cfg = configure_device(cfg, _args("--source", "tcp", "--auto-zero"))
    assert device_cfg.target == "franka" and device_cfg.dt == pytest.approx(1 / 60)
    assert device_cfg.auto_zero


def test_upstream_device_factory_mapping_hold_recovery_and_reset(monkeypatch):
    sample = [_raw_home(), 0.0]
    monkeypatch.setattr(OmyLeaderDevice, "_open_source", lambda device: setattr(device, "_latest", lambda: sample))
    cfg = OmyLeaderCfg(target="franka", source="tcp", sim_device="cpu", dt=1 / 60)
    device = create_teleop_device("omy_leader", {"omy_leader": cfg}, {})
    expected = OmyToFrankaRetarget(replace(DEFAULT_OMY_TO_FRANKA_CONFIG, dt=1 / 60))
    np.testing.assert_allclose(device.advance().numpy(), expected.step(sample[0]), atol=1e-6)
    sample[0] = {**sample[0], "joint_1.pos": 1.0, "gripper.pos": -0.8}
    last = device.advance()
    np.testing.assert_allclose(last.numpy(), expected.step(sample[0]), atol=1e-6)
    assert last[7] == -1 and last[2] == 0
    assert last[0] == pytest.approx(FRANKA_VEL_MAX[0] / 60)
    sample[1] = 10.0
    torch.testing.assert_close(device.advance(), last)
    device.reset()  # Upstream resets the mapper but retains the last command.
    torch.testing.assert_close(device.advance(), last)
    sample[1] = 0.0
    expected.reset()
    np.testing.assert_allclose(device.advance().numpy(), expected.step(sample[0]), atol=1e-6)
    sample[0] = {**sample[0], "joint_1.pos": 0.0, "gripper.pos": 0.0}
    assert device.advance()[7] == 1


def test_upstream_auto_zero_is_not_recaptured_on_reset(monkeypatch):
    sample = _raw_home()
    monkeypatch.setattr(OmyLeaderDevice, "_open_source", lambda device: setattr(device, "_latest", lambda: (sample, 0.0)))
    cfg = OmyLeaderCfg(target="franka", sim_device="cpu", auto_zero=True, auto_zero_settle_s=3.0)
    device = OmyLeaderDevice(cfg)
    np.testing.assert_allclose(device.advance().numpy(), [*FRANKA_HOME, 1.0], atol=1e-6)
    device._t_up -= 4
    device.advance()
    zeros = device._mapper.config.omy_zero_rad.copy()
    device.reset()
    assert device._zeroed and device._mapper.config.omy_zero_rad == zeros


def test_operator_image_layout():
    view = torch.zeros((1, 480, 640, 4), dtype=torch.uint8)
    wrist = torch.full((1, 180, 240, 4), 127, dtype=torch.uint8)
    env = SimpleNamespace(scene={name: SimpleNamespace(data=SimpleNamespace(output={"rgb": value}))
                                for name, value in (("view_cam", view), ("omy_wrist_cam", wrist))})
    image = operator_frame(env)
    assert image.shape == (480, 640, 3)
    assert np.all(image[8:188, 8:248] == 127)
    assert np.all(image[:8] == 0)
    assert torch.all(view == 0)


def test_output_refuses_existing_data(tmp_path):
    assert prepare_output(tmp_path) == tmp_path.resolve()
    (tmp_path / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="new or empty"):
        prepare_output(tmp_path)
    assert (tmp_path / "keep.txt").read_text() == "keep"


@pytest.mark.parametrize("save_failures", [False, True])
def test_simulator_actions_reset_and_recording(tmp_path, monkeypatch, save_failures):
    monkeypatch.setattr(robolab.constants, "_output_dir", str(tmp_path))
    monkeypatch.setattr(robolab.constants, "RECORD_IMAGE_DATA", True)
    postfix = f"OMYPhysicsTest{save_failures}"
    register_omy_envs("BananaInBowlTask", env_postfix=postfix)
    cfg = parse_env_cfg(f"BananaInBowlTask{postfix}", device="cpu", num_envs=1)
    args = _args("--source", "tcp", "--record", *(["--save-failures"] if save_failures else []))
    device_cfg = configure_device(cfg, args)
    env, _ = create_env(cfg, policy="omy_joint_position", rendering_mode="performance")
    actions = []
    try:
        ids = validate_environment(env)
        configure_recording(env, args, device_cfg)
        # Exercise periodic flushing without allocating full-sized camera images.
        env.recorder_manager.set_flush_interval(2)
        env.reset(seed=7)
        env.recorder_manager.set_episode_index(0, env_ids=[0])
        env.recorder_manager.set_episode_seed(7, env_ids=[0])
        np.testing.assert_allclose(env.scene["robot"].data.joint_pos[0, ids].cpu(), FRANKA_HOME, atol=1e-4)
        for gripper in (1.0, -1.0, 1.0):
            action = torch.tensor([[*FRANKA_HOME, gripper]], device=env.device)
            env.step(action)
            torch.testing.assert_close(env.action_manager.action, action)
            torch.testing.assert_close(env.action_manager.get_term("body").processed_actions, action[:, :7])
            expected_gripper = 0.0 if gripper == 1.0 else math.pi / 4
            assert env.action_manager.get_term("finger_joint").processed_actions.item() == pytest.approx(expected_gripper)
            actions.append(action.cpu().numpy()[0])
        finish_interrupted(env, True)
        close_recording(env)
    finally:
        close_recording(env)
        env.close()
    with h5py.File(tmp_path / "omy_demos.hdf5", "r") as dataset:
        if save_failures:
            episode = dataset["data/demo_0"]
            np.testing.assert_allclose(episode["actions"][:], actions)
            assert not episode.attrs["success"]
            assert "states" in episode and "initial_state" in episode and "obs" in episode
        else:
            assert list(dataset["data"].keys()) == []
        metadata = json.loads(dataset["data"].attrs["robolab_omy_teleop"])
        assert metadata["gripper_encoding"] == "+1=open,-1=closed"
        assert metadata["control_hz"] == pytest.approx(60)
        assert metadata["upstream"]["installed_commit"]
    assert (tmp_path / "env_cfg.json").is_file()


def test_runner_records_two_attempts_and_closes_device(tmp_path, monkeypatch):
    monkeypatch.setattr(robolab.constants, "_output_dir", str(tmp_path))
    monkeypatch.setattr(robolab.constants, "RECORD_IMAGE_DATA", False)
    monkeypatch.setattr(OmyLeaderDevice, "_open_source", lambda device: setattr(device, "_latest", lambda: (_raw_home(), 0.0)))
    closed = []
    monkeypatch.setattr(OmyLeaderDevice, "close", lambda device: closed.append(True))
    args = _args("--source", "tcp", "--record", "--save-failures", "--num-episodes", "2",
                 "--episode-length-s", "0.06", "--steps", "12", "--no-realtime",
                 "--output-dir", str(tmp_path))
    args.device = "cpu"
    assert run(args, SimpleNamespace(is_running=lambda: True)) == 0
    assert closed == [True]
    with h5py.File(tmp_path / "omy_demos.hdf5", "r") as dataset:
        assert list(dataset["data"]) == ["demo_0", "demo_1"]
        for name in dataset["data"]:
            episode = dataset[f"data/{name}"]
            assert episode["actions"].shape == (4, 8)
            assert not episode.attrs["success"]
            np.testing.assert_allclose(episode["actions"][:, :7], [FRANKA_HOME] * 4, atol=1e-6)
