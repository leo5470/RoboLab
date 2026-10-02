"""Upstream OMY contract and DROID simulator integration; no physical leader required."""

import importlib.metadata
import json
import math
from copy import deepcopy
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
    OmyDroidCfg, configure_device, configure_performance, register_omy_envs, validate_environment,
)
from robolab.core.omy_teleop.device import RoboLabOmyLeaderCfg, RoboLabOmyLeaderDevice
from robolab.core.omy_teleop.options import ARM_JOINTS, build_parser, validate_args
from robolab.core.omy_teleop.performance import prepare_launcher
from robolab.core.omy_teleop.runner import (
    close_recording, configure_recording, finish_interrupted, operator_frame, prepare_output, run,
)
from robolab.robots.droid import DroidCfg

DROID_HOME = tuple(DroidCfg().robot.init_state.joint_pos[name] for name in ARM_JOINTS)


def _args(*argv):
    parser = build_parser()
    args = parser.parse_args(list(argv))
    validate_args(parser, args)
    return args


def _raw_home():
    # Input for the pinned default DIRECT wrist map, not the optional ZYZ map.
    values = (0.0, -math.pi / 4, 3 * math.pi / 4, math.pi / 2, 0.0, 0.0)
    return {**{f"joint_{i}.pos": value for i, value in enumerate(values, 1)}, "gripper.pos": 0.0}


def _raw_droid_home():
    values = (DROID_HOME[0], DROID_HOME[1], -DROID_HOME[3], math.pi - DROID_HOME[5],
              DROID_HOME[4], math.pi / 4 - DROID_HOME[6])
    return {**{f"joint_{i}.pos": value for i, value in enumerate(values, 1)}, "gripper.pos": 0.0}


def test_launcher_arguments_do_not_conflict():
    parser = build_parser(AppLauncher)
    args = parser.parse_args(["--headless", "--source", "tcp"])
    assert args.headless and args.rendering_mode == "performance"
    args = parser.parse_args(["--performance-profile", "cpu"])
    prepare_launcher(args)
    assert args.device == "cpu"
    args = parser.parse_args(["--performance-profile", "cpu", "--device", "cuda:1"])
    prepare_launcher(args)
    assert args.device == "cuda:1" and not args.kit_args


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
    assert tuple(OmyDroidCfg().robot.init_state.joint_pos[name] for name in ARM_JOINTS) == DROID_HOME
    np.testing.assert_allclose(np.degrees(DROID_HOME), [0, -36, 0, -144, 0, 108, 0], atol=1e-9)
    assert DroidCfg().robot.init_state.joint_pos == original
    device_cfg = configure_device(cfg, _args("--source", "tcp", "--auto-zero"))
    assert device_cfg.target == "franka" and device_cfg.dt == pytest.approx(1 / 60)
    assert device_cfg.auto_zero
    assert isinstance(device_cfg, RoboLabOmyLeaderCfg)
    assert device_cfg.franka_initial_q == DROID_HOME
    cfg.sim.device = "cuda:0"
    assert configure_device(cfg, _args("--source", "tcp")).sim_device == "cpu"


def test_contact_profile_changes_only_audited_omy_sensor_configs():
    from isaaclab.sensors import ContactSensorCfg

    register_omy_envs("BananaInBowlTask", env_postfix="OMYContactProfile")
    full = parse_env_cfg("BananaInBowlTaskOMYContactProfile", device="cpu", num_envs=1)
    configure_performance(full, _args("--contact-sensors", "full"))
    required = parse_env_cfg("BananaInBowlTaskOMYContactProfile", device="cpu", num_envs=1)
    configure_performance(required, _args("--contact-sensors", "required"))
    names = lambda cfg: {name for name, value in vars(cfg.scene).items()
                         if isinstance(value, ContactSensorCfg) and "__" in name}
    assert len(names(full)) == 7
    assert names(required) == names(full) - {"bowl__table"}
    assert full.sim.dt == required.sim.dt
    assert full.sim.gravity == required.sim.gravity


def test_omy_batch_contacts_match_individual_queries_and_cache_per_step(monkeypatch):
    from robolab.core.world import world_state

    forces = torch.zeros((1, 1, 2, 3))
    forces[0, 0, 0, 2] = 1.0
    sensor = SimpleNamespace(cfg=SimpleNamespace(filter_prim_paths_expr=["/banana", "/bowl"]),
                             data=SimpleNamespace(force_matrix_w=forces))
    monkeypatch.setattr(world_state, "get_batch_contact_sensor", lambda scene, body: sensor)
    pair = SimpleNamespace(data=SimpleNamespace(force_matrix_w=forces[:, :, :1, :]))
    monkeypatch.setattr(world_state, "get_contact_sensor", lambda scene, a, b: pair)
    scene_cfg = SimpleNamespace(banana=SimpleNamespace(prim_path="/banana"),
                                bowl=SimpleNamespace(prim_path="/bowl"))
    env = SimpleNamespace(cfg=SimpleNamespace(omy_contact_sensor_profile="required",
                                               contact_object_list=["banana", "bowl", "table"],
                                               contact_gripper={"gripper": "/gripper"}, scene=scene_cfg),
                          scene=SimpleNamespace(), num_envs=1)
    world = world_state.WorldState.__new__(world_state.WorldState)
    world.env = env
    world._omy_batch_cache = {}
    world._omy_batch_stats = {"computed": 0, "reused": 0, "fallback": 0}
    assert world._in_contact_pair("banana", "gripper", 0.1, 0)
    assert world.get_objects_in_contact_with("gripper", ["bowl", "banana"], 0.1, 0) == ["banana"]
    forces.zero_()
    assert world._in_contact_pair("gripper", "banana", 0.1, 0)  # same-step result
    world.clear_contact_cache()
    assert not world._in_contact_pair("gripper", "banana", 0.1, 0)
    assert world._omy_batch_stats == {"computed": 2, "reused": 2, "fallback": 0}


@torch.inference_mode()
def test_optimized_observations_match_frame_sensor_after_motion_and_reset(tmp_path, monkeypatch):
    from isaaclab.managers import RecorderManagerBaseCfg
    from robolab.core.omy_teleop.observations import ObservationPassCache
    from robolab.robots import droid

    monkeypatch.setattr(robolab.constants, "_output_dir", str(tmp_path))
    monkeypatch.setattr(robolab.constants, "RECORD_IMAGE_DATA", False)
    register_omy_envs("BananaInBowlTask", env_postfix="OMYPoseEquivalence")
    cfg = parse_env_cfg("BananaInBowlTaskOMYPoseEquivalence", device="cuda:0", num_envs=1)
    cfg.recorders = RecorderManagerBaseCfg()
    env, _ = create_env(cfg, policy="omy_joint_position", rendering_mode="performance")
    cache = ObservationPassCache(env)
    try:
        for seed in (0, 1):
            env.reset(seed=seed)
            robot = env.scene["robot"]
            # Test a rotated/translated root too: observations must remain
            # robot-root relative, not world/env relative.
            if seed:
                pose = robot.data.root_pose_w.clone()
                pose[:, :3] += pose.new_tensor([0.05, -0.1, 0.03])
                pose[:, 3:] = pose.new_tensor([math.cos(0.2), 0, 0, math.sin(0.2)])
                robot.write_root_pose_to_sim(pose)
                env.sim.forward()
                env.scene.update(0.0)
            for i in range(4):
                values = env.observation_manager.compute()["proprio_obs"]
                assert set(values) == {"arm_joint_pos", "gripper_pos", "ee_pos", "ee_quat", "eef_pos", "eef_quat"}
                for name in ("ee_pos", "ee_quat", "eef_pos", "eef_quat"):
                    reference = getattr(droid, name)(env)
                    torch.testing.assert_close(values[name], reference, atol=2e-5, rtol=2e-5)
                action = torch.tensor([[*DROID_HOME, 1.0]], device=env.device)
                action[:, 0] += 0.02 * i
                env.step(action)
    finally:
        cache.close()
        env.close()


def test_performance_config_is_local_and_preserves_control_and_recording():
    from isaaclab.sensors import ContactSensorCfg

    register_omy_envs("BananaInBowlTask", env_postfix="OMYPerfConfigTest", stream=True, images=True)
    original = parse_env_cfg("BananaInBowlTaskOMYPerfConfigTest", device="cpu", num_envs=1)
    # Isaac Lab's copy() reconstructs this generated config and runs its
    # __post_init__; deepcopy isolates the baseline without reinitialization.
    cfg = deepcopy(original)
    configure_performance(cfg, _args("--stream"))
    assert cfg.sim.render_interval == 4 and original.sim.render_interval == 2
    for name, sensor in vars(cfg.scene).items():
        if isinstance(sensor, ContactSensorCfg):
            assert sensor.history_length == 0
            assert getattr(original.scene, name).history_length == 6
            assert sensor.filter_prim_paths_expr == getattr(original.scene, name).filter_prim_paths_expr
    assert cfg.sim.dt == original.sim.dt == pytest.approx(1 / 120)
    assert cfg.decimation == original.decimation == 2
    assert cfg.scene.robot.to_dict() == original.scene.robot.to_dict()
    assert cfg.observations.to_dict() == original.observations.to_dict()
    assert cfg.terminations.to_dict() == original.terminations.to_dict()
    configure_performance(cfg, _args("--stream", "--record", "--record-images"))
    assert cfg.sim.render_interval == 2
    configure_performance(cfg, _args("--contact-history-length", "6"))
    assert all(sensor.history_length == 6 for sensor in vars(cfg.scene).values()
               if isinstance(sensor, ContactSensorCfg))


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


def test_droid_home_initial_command_rate_limit_and_stale_reset(monkeypatch):
    sample = [None, float("inf")]
    monkeypatch.setattr(OmyLeaderDevice, "_open_source", lambda device: setattr(device, "_latest", lambda: sample))
    cfg = RoboLabOmyLeaderCfg(target="franka", source="tcp", sim_device="cpu", dt=1 / 60,
                             franka_initial_q=DROID_HOME)
    device = create_teleop_device("omy_leader", {"omy_leader": cfg}, {})
    assert isinstance(device, RoboLabOmyLeaderDevice)
    np.testing.assert_allclose(device.advance(), [*DROID_HOME, 1.0], atol=1e-6)
    sample[:] = [_raw_droid_home(), 0.0]
    np.testing.assert_allclose(device.advance(), [*DROID_HOME, 1.0], atol=1e-6)
    sample[0] = {**sample[0], "joint_1.pos": 1.0, "gripper.pos": -0.8}
    moved = device.advance()
    assert moved[0] == pytest.approx(FRANKA_VEL_MAX[0] / 60)
    assert moved[7] == -1
    sample[1] = 10.0
    torch.testing.assert_close(device.advance(), moved)
    device.reset()
    np.testing.assert_allclose(device.advance(), [*DROID_HOME, 1.0], atol=1e-6)
    sample[:] = [_raw_droid_home(), 0.0]
    np.testing.assert_allclose(device.advance(), [*DROID_HOME, 1.0], atol=1e-6)


def test_droid_auto_zero_targets_environment_home_and_survives_reset(monkeypatch):
    sample = [_raw_home(), 0.0]  # deliberately not the DROID-ready raw reading
    monkeypatch.setattr(OmyLeaderDevice, "_open_source", lambda device: setattr(device, "_latest", lambda: sample))
    cfg = RoboLabOmyLeaderCfg(target="franka", source="tcp", sim_device="cpu", auto_zero=True,
                             franka_initial_q=DROID_HOME)
    device = RoboLabOmyLeaderDevice(cfg)
    np.testing.assert_allclose(device.advance(), [*DROID_HOME, 1.0], atol=1e-6)
    device._t_up -= 4
    np.testing.assert_allclose(device.advance(), [*DROID_HOME, 1.0], atol=1e-6)
    np.testing.assert_allclose(device._mapper.raw_target(sample[0]), DROID_HOME, atol=1e-9)
    zeros = device._mapper.config.omy_zero_rad.copy()
    device.reset()
    assert device._zeroed and device._mapper.config.omy_zero_rad == zeros
    np.testing.assert_allclose(device.advance(), [*DROID_HOME, 1.0], atol=1e-6)


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
    configure_performance(cfg, args)
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
        np.testing.assert_allclose(env.scene["robot"].data.joint_pos[0, ids].cpu(), DROID_HOME, atol=1e-4)
        for gripper in (1.0, -1.0, 1.0):
            action = torch.tensor([[*DROID_HOME, gripper]], device=env.device)
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
    monkeypatch.setattr(OmyLeaderDevice, "_open_source", lambda device: setattr(device, "_latest", lambda: (_raw_droid_home(), 0.0)))
    closed = []
    monkeypatch.setattr(OmyLeaderDevice, "close", lambda device: closed.append(True))
    args = _args("--source", "tcp", "--record", "--save-failures", "--num-episodes", "2",
                 "--episode-length-s", "0.06", "--steps", "12", "--no-realtime",
                 "--output-dir", str(tmp_path))
    args.device = "cpu"
    assert run(args, SimpleNamespace(is_running=lambda: True)) == 0
    assert closed == [True]
    performance = json.loads((tmp_path / "omy_performance.json").read_text())
    assert performance["total_steps"] == 8
    assert set(performance["settings"]["contact_history_lengths"].values()) == {0}
    with h5py.File(tmp_path / "omy_demos.hdf5", "r") as dataset:
        assert list(dataset["data"]) == ["demo_0", "demo_1"]
        for name in dataset["data"]:
            episode = dataset[f"data/{name}"]
            assert episode["actions"].shape == (4, 8)
            assert not episode.attrs["success"]
            np.testing.assert_allclose(episode["actions"][:, :7], [DROID_HOME] * 4, atol=1e-6)


@pytest.mark.parametrize("history", [0, 6])
@torch.inference_mode()
def test_current_contact_and_place_success_with_or_without_history(tmp_path, monkeypatch, history):
    from isaaclab.managers import RecorderManagerBaseCfg

    from robolab.core.world.world_state import get_world

    monkeypatch.setattr(robolab.constants, "_output_dir", str(tmp_path))
    postfix = f"OMYContactHistoryTest{history}"
    register_omy_envs("BananaInBowlTask", env_postfix=postfix)
    cfg = parse_env_cfg(f"BananaInBowlTask{postfix}", device="cpu", num_envs=1)
    configure_performance(cfg, _args("--contact-history-length", str(history)))
    cfg.recorders = RecorderManagerBaseCfg()
    env, _ = create_env(cfg, policy="omy_joint_position", rendering_mode="performance")
    try:
        env.reset(seed=0)
        action = torch.tensor([[*DROID_HOME, 1.0]], device=env.device)
        world = get_world(env)
        banana = env.scene["banana"]
        pose = banana.data.root_pose_w.clone()
        pose[:, 2] += 0.10
        banana.write_root_pose_to_sim(pose)
        banana.write_root_velocity_to_sim(torch.zeros_like(banana.data.root_vel_w))
        # Measure an actual landing, not an already sleeping resting body.
        touched_table = False
        for _ in range(60):
            env.step(action)
            if world.in_contact("banana", "table", env_id=0):
                touched_table = True
                break
        assert touched_table, (banana.data.root_pos_w, banana.data.root_vel_w,
                               env.scene["banana__table"].data.force_matrix_w)
        assert not env.all_terminated
        pose = banana.data.root_pose_w.clone()
        pose[:, 2] += 0.3
        banana.write_root_pose_to_sim(pose)
        banana.write_root_velocity_to_sim(torch.zeros_like(banana.data.root_vel_w))
        env.step(action)
        assert not world.in_contact("banana", "table", env_id=0)
        assert torch.isfinite(env.scene["banana__table"].data.force_matrix_w).all()
        # Controlled placement checks the unchanged contact-based task predicate;
        # this is not a claim to have validated a teleoperated grasp.
        pose[:, :3] = env.scene["bowl"].data.root_pos_w
        pose[:, 2] += 0.15
        banana.write_root_pose_to_sim(pose)
        banana.write_root_velocity_to_sim(torch.zeros_like(banana.data.root_vel_w))
        for _ in range(90):
            env.step(action)
            if env.all_terminated:
                break
        assert env.all_terminated
        assert env.get_env_results()[0]["success"]
        assert world.in_contact("banana", "bowl", env_id=0)
        assert not world.in_contact("gripper", "banana", env_id=0)
    finally:
        env.close()


@pytest.mark.parametrize("record_images", [False, True])
def test_stream_capture_phase_and_image_recording_across_odd_resets(tmp_path, monkeypatch, record_images):
    import itertools

    import omy_leader_isaaclab.video as video_module

    import robolab.core.omy_teleop.runner as runner

    counter = itertools.count()
    captured_steps, jpegs = [], []
    monkeypatch.setattr(robolab.constants, "_output_dir", str(tmp_path))
    monkeypatch.setattr(robolab.constants, "RECORD_IMAGE_DATA", False)
    monkeypatch.setattr(OmyLeaderDevice, "_open_source", lambda device: setattr(
        device, "_latest", lambda: ({**_raw_droid_home(), "joint_1.pos": 0.03 * next(counter)}, 0.0)))
    monkeypatch.setattr(OmyLeaderDevice, "close", lambda device: None)
    monkeypatch.setattr(video_module, "VideoServer", lambda **kwargs: SimpleNamespace(push=jpegs.append))
    original_frame = runner.operator_frame

    def frame(env):
        captured_steps.append(env._sim_step_counter)
        return original_frame(env)

    monkeypatch.setattr(runner, "operator_frame", frame)
    original_configure = runner.configure_performance

    def small_test_cameras(cfg, args):
        original_configure(cfg, args)
        # Test recording cadence/schema cheaply; production camera sizes are untouched.
        if args.record_images:
            for camera in (cfg.scene.wrist_cam, cfg.scene.over_shoulder_left_camera):
                camera.width, camera.height = 64, 48

    monkeypatch.setattr(runner, "configure_performance", small_test_cameras)
    args = _args("--source", "tcp", "--stream", "--record", "--save-failures",
                 "--num-episodes", "2", "--episode-length-s", "0.075", "--steps", "10",
                 "--no-realtime", "--output-dir", str(tmp_path),
                 *(["--record-images"] if record_images else []))
    args.device = "cpu"
    assert run(args, SimpleNamespace(is_running=lambda: True)) == 0
    assert len(captured_steps) == len(jpegs) == 5
    assert captured_steps == [4, 8, 12, 16, 20]
    decoded = video_module.decode_jpeg(jpegs[-1])
    assert decoded.shape == (480, 640, 3) and decoded.max() > 0
    result = json.loads((tmp_path / "omy_performance.json").read_text())
    assert result["total_steps"] == 10
    assert result["total_video_frames"] == 5
    assert result["settings"]["render_interval"] == (2 if record_images else 4)
    # Terminal/reset re-renders can add calls, so don't assert an exact render total.
    assert result["total_renders"] >= (10 if record_images else 5)
    with h5py.File(tmp_path / "omy_demos.hdf5", "r") as dataset:
        for episode in dataset["data"].values():
            assert episode["actions"].shape == (5, 8)
            images = []
            episode["obs"].visititems(lambda name, value: images.append(value)
                                     if isinstance(value, h5py.Dataset) and value.ndim == 4 else None)
            assert len(images) == (2 if record_images else 0)
            for image in images:
                assert image.shape == (5, 48, 64, 3)
                assert image[:].max() > 0
                assert not np.array_equal(image[0], image[-1])
