"""GPU manager/recorder integration for OMY; generated commands, no leader access."""

import json
from copy import deepcopy

import h5py
import numpy as np
import pytest
import torch

import robolab.constants
from robolab.core.environments.config import parse_env_cfg
from robolab.core.environments.runtime import create_env
from robolab.core.omy_teleop.config import configure_performance, register_omy_envs
from robolab.core.omy_teleop.environment import OmyEnv
from robolab.core.omy_teleop.options import ARM_JOINTS, build_parser
from robolab.core.omy_teleop.runner import close_recording, configure_recording, finish_interrupted
from robolab.core.omy_teleop.stepping import eligibility_reasons


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(robolab.constants, "_output_dir", str(tmp_path))
    monkeypatch.setattr(robolab.constants, "RECORD_IMAGE_DATA", True)
    monkeypatch.setattr(robolab.constants, "ENABLE_SUBTASK_PROGRESS_CHECKING", True)
    args = build_parser().parse_args([
        "--source", "tcp", "--control-hz", "15", "--solver-position-iterations", "16",
        "--stepping-profile", "held-target", "--recording-profile", "compact", "--record",
        "--save-failures", "--render-interval", "4", "--episode-length-s", "20",
    ])
    register_omy_envs("BananaInBowlTask", env_postfix="OMYFastIntegration")
    original = parse_env_cfg("BananaInBowlTaskOMYFastIntegration", device="cuda:0", num_envs=1)
    cfg = deepcopy(original)
    cfg.episode_length_s = 20
    configure_performance(cfg, args)
    assert original.sim.physx.max_position_iteration_count == 32
    assert cfg.sim.physx.max_position_iteration_count == 16
    assert cfg.sim.physx.max_velocity_iteration_count == original.sim.physx.max_velocity_iteration_count
    with torch.inference_mode():
        env, _ = create_env(cfg, policy="omy_joint_position", env_class=OmyEnv, rendering_mode="performance")
        configure_recording(env, args, type("DeviceCfg", (), {"dt": 1 / 15})())
        env.reset(seed=0)
    try:
        assert cfg.omy_stepping_effective == "held-target", cfg.omy_stepping_fallback_reasons
        assert cfg.omy_episode_max_steps == 300
        yield env
    finally:
        close_recording(env)
        env.close()


@torch.inference_mode()
def test_manager_contract_and_hdf5_alignment(candidate, monkeypatch, tmp_path):
    env = candidate
    counts = {"process": 0, "apply": 0, "write": 0, "physics": 0, "update": [], "obs": 0}
    def count(obj, method, key):
        original = getattr(obj, method)
        def wrapped(*args, **kwargs):
            counts[key] += 1
            return original(*args, **kwargs)
        monkeypatch.setattr(obj, method, wrapped)
    for obj, method, key in ((env.action_manager, "process_action", "process"),
                             (env.action_manager, "apply_action", "apply"),
                             (env.scene, "write_data_to_sim", "write"), (env.sim, "step", "physics"),
                             (env.observation_manager, "compute", "obs")):
        count(obj, method, key)
    update = env.scene.update
    monkeypatch.setattr(env.scene, "update", lambda dt: (counts["update"].append(dt), update(dt))[1])
    env.recorder_manager.set_episode_index(0, [0])
    env.recorder_manager.set_episode_seed(0, [0])
    home = [env.cfg.scene.robot.init_state.joint_pos[name] for name in ARM_JOINTS]
    action = torch.tensor([[*home, 1.0]], device=env.device)
    before = env._sim_step_counter
    for i in range(4):
        action[0, 0] += 0.01
        env.step(action)
    assert counts == {"process": 4, "apply": 4, "write": 4, "physics": 32, "update": [1 / 15] * 4, "obs": 4}
    assert env._sim_step_counter - before == 32
    finish_interrupted(env, True)
    finish_interrupted(env, True)  # no duplicate export
    close_recording(env)
    with h5py.File(tmp_path / "omy_demos.hdf5") as file:
        assert list(file["data"]) == ["demo_0"]
        demo = file["data/demo_0"]
        assert demo["actions"].shape == (4, 8)
        assert demo["states/articulation/robot/joint_position"].shape == (4, 13)
        names = env.scene["robot"].joint_names
        arm_ids = [names.index(name) for name in ARM_JOINTS]
        pre = demo["obs/proprio_obs/arm_joint_pos"][:]
        post = demo["states/articulation/robot/joint_position"][:]
        np.testing.assert_array_equal(pre[0], demo["initial_state/articulation/robot/joint_position"][0, arm_ids])
        np.testing.assert_array_equal(pre[1:], post[:-1, arm_ids])
        np.testing.assert_allclose(demo["timing/sim_after_s"][:] - demo["timing/sim_before_s"][:], 1 / 15)
        metadata = json.loads(file["data"].attrs["robolab_omy_teleop"])
        assert metadata["performance"]["solver"]["scene_max_position_iterations"] == 16
        assert metadata["gripper_observation"]["units"] == "normalized_closure"
        assert not demo.attrs["success"]


@torch.inference_mode()
def test_early_terminal_is_recorded_once_and_reset_isolates_next_episode(candidate, monkeypatch, tmp_path):
    env = candidate
    term = env.termination_manager
    # Inject a boundary success without changing physical stepping.
    success = term._term_names.index("success")
    original = term._term_cfgs[success].func
    term._term_cfgs[success].func = lambda env, **kwargs: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    env.recorder_manager.set_episode_index(0, [0])
    home = [env.cfg.scene.robot.init_state.joint_pos[name] for name in ARM_JOINTS]
    action = torch.tensor([[*home, 1.0]], device=env.device)
    env.step(action)
    assert env.all_terminated
    assert env.get_env_results()[0] == {"env_id": 0, "success": True, "step": 1}
    with pytest.raises(RuntimeError, match="finalized"):
        env.step(action)
    term._term_cfgs[success].func = original
    env.reset(seed=1)
    assert not env.all_terminated and int(env.episode_length_buf[0]) == 0
    env.recorder_manager.set_episode_index(1, [0])
    env.step(action)
    finish_interrupted(env, True)
    close_recording(env)
    with h5py.File(tmp_path / "omy_demos.hdf5") as file:
        assert list(file["data"]) == ["demo_0", "demo_1"]
        for name in file["data"]:
            assert file[f"data/{name}/actions"].shape == (1, 8)
            assert file[f"data/{name}/states/articulation/robot/joint_position"].shape == (1, 13)
        assert file["data/demo_0"].attrs["success"]
        assert not file["data/demo_1"].attrs["success"]


@torch.inference_mode()
def test_runtime_wrench_falls_back_before_processing(candidate):
    env = candidate
    env.scene["robot"].has_external_wrench = True
    assert eligibility_reasons(env)
    action = torch.tensor([[*[env.cfg.scene.robot.init_state.joint_pos[name] for name in ARM_JOINTS], 1.0]], device=env.device)
    env.step(action)
    assert env.cfg.omy_stepping_effective == "ordinary"
    assert env.cfg.omy_stepping_fallback_reasons == ["external wrench requires submission on every physics substep"]


@torch.inference_mode()
def test_timeout_export_and_same_counter_restore(candidate, tmp_path):
    from robolab.core.replay.scene_state import restore_scene_state
    from robolab.core.world.world_state import get_world

    env = candidate
    state = env.scene.get_state(is_relative=True)
    # State restore must refresh world queries without a physics step.
    recorded = {kind: {name: {key: leaf.cpu().numpy() for key, leaf in values.items()}
                        for name, values in assets.items()} for kind, assets in state.items()}
    counter = env._sim_step_counter
    banana = env.scene["banana"]
    shifted = banana.data.root_pose_w.clone()
    shifted[:, 0] += 0.2
    banana.write_root_pose_to_sim(shifted)
    world = get_world(env)
    world.get_pose("banana")
    restore_scene_state(env, recorded)
    assert env._sim_step_counter == counter
    torch.testing.assert_close(world.get_pose("banana")[0], state["rigid_object"]["banana"]["root_pose"][:, :3])
    env.recorder_manager.set_episode_index(0, [0])
    env.episode_length_buf[:] = env.max_episode_length - 1
    action = torch.tensor([[*[env.cfg.scene.robot.init_state.joint_pos[name] for name in ARM_JOINTS], 1.0]], device=env.device)
    env.step(action)
    assert env.all_terminated and not env.get_env_results()[0]["success"]
    close_recording(env)
    with h5py.File(tmp_path / "omy_demos.hdf5") as file:
        assert file["data/demo_0/actions"].shape == (1, 8)
        assert not file["data/demo_0"].attrs["success"]


@torch.inference_mode()
def test_reset_finalizes_pending_attempt(candidate, tmp_path):
    env = candidate
    env.recorder_manager.set_episode_index(0, [0])
    action = torch.tensor([[*[env.cfg.scene.robot.init_state.joint_pos[name] for name in ARM_JOINTS], 1.0]], device=env.device)
    env.step(action)
    env.reset(seed=2)
    env.reset(seed=2)  # no extra finalized episode
    close_recording(env)
    with h5py.File(tmp_path / "omy_demos.hdf5") as file:
        assert list(file["data"]) == ["demo_0"]
        assert file["data/demo_0/actions"].shape == (1, 8)


def test_eligibility_rejections(candidate, monkeypatch):
    env = candidate
    assert eligibility_reasons(env) == []
    cases = [
        (env.cfg, "decimation", 4, "eight fixed 120 Hz"),
        (env.cfg.sim, "dt", 1 / 60, "eight fixed 120 Hz"),
        (env.cfg.sim, "render_interval", 8, "render interval"),
        (env.cfg, "_task_name", "OtherTask", "task predicates"),
        (env.cfg, "omy_substep_diagnostics", True, "substep diagnostics"),
        (env.cfg.sim, "use_fabric", False, "Fabric"),
        (env.action_manager.get_term("body").cfg, "use_default_offset", True, "action mapping"),
        (env.scene.cfg, "num_envs", 2, "exactly one environment"),
        (env.action_manager, "_term_names", ["body", "finger_joint", "unknown"], "absolute Franka targets"),
        (env.observation_manager._group_obs_term_cfgs["proprio_obs"][0], "history_length", 1, "history/modifiers"),
        (env.reward_manager, "_term_names", ["custom"], "reward_manager consumers"),
        (env.curriculum_manager, "_term_names", ["custom"], "curriculum_manager consumers"),
        (env.observation_manager, "_group_obs_term_names", {"image_obs": ["camera"]}, "observation consumers"),
    ]
    contact = next(sensor for sensor in env.scene.sensors.values() if hasattr(sensor.cfg, "track_air_time"))
    cases.extend(((contact.cfg, "track_air_time", True, "history/air-time"),
                  (contact.cfg, "history_length", 2, "history/air-time")))
    for obj, name, value, expected in cases:
        with monkeypatch.context() as patch:
            patch.setattr(obj, name, value)
            assert any(expected in reason for reason in eligibility_reasons(env))
    with monkeypatch.context() as patch:
        patch.setattr("importlib.metadata.version", lambda name: "2.3.2")
        assert any("Isaac Lab 2.2.0" in reason for reason in eligibility_reasons(env))
    with monkeypatch.context() as patch:
        from types import SimpleNamespace

        patch.setattr(env, "sim", SimpleNamespace(device="cpu"))
        assert any("CUDA physics" in reason for reason in eligibility_reasons(env))
    with monkeypatch.context() as patch:
        patch.setitem(env.scene["robot"].actuators, "custom", object())
        assert any("implicit actuators" in reason for reason in eligibility_reasons(env))
    with monkeypatch.context() as patch:
        patch.setitem(env.scene.sensors, "unknown", object())
        assert any("sensor unknown" in reason for reason in eligibility_reasons(env))
    with monkeypatch.context() as patch:
        patch.setitem(env.recorder_manager._terms, "custom", object())
        assert any("recorder consumers" in reason for reason in eligibility_reasons(env))
    with monkeypatch.context() as patch:
        patch.setitem(env.command_manager._terms, "custom", object())
        assert any("command_manager consumers" in reason for reason in eligibility_reasons(env))
    with monkeypatch.context() as patch:
        patch.setitem(env.event_manager._mode_term_names, "interval", ["custom"])
        assert any("interval events" in reason for reason in eligibility_reasons(env))
    with monkeypatch.context() as patch:
        patch.setitem(env.scene.deformable_objects, "custom", object())
        assert any("topology" in reason for reason in eligibility_reasons(env))
    with monkeypatch.context() as patch:
        patch.setitem(env.scene.articulations, "extra_robot", object())
        assert any("topology" in reason for reason in eligibility_reasons(env))
    with monkeypatch.context() as patch:
        patch.setattr(env.termination_manager._term_cfgs[0], "func", lambda env: None)
        assert any("termination" in reason for reason in eligibility_reasons(env))
    assert eligibility_reasons(env) == []


def test_base_artifact_reset_occurs_once(monkeypatch):
    from types import SimpleNamespace

    from isaaclab.envs import ManagerBasedRLEnv

    from robolab.core.environments import env as env_module

    calls = []
    monkeypatch.setattr(ManagerBasedRLEnv, "_reset_idx", lambda self, ids: calls.append(ids.tolist()))
    monkeypatch.setattr(env_module, "get_world", lambda env: SimpleNamespace(reset_predicate_state=lambda ids: None))
    bare = env_module.RobolabEnv.__new__(env_module.RobolabEnv)
    bare.cfg = SimpleNamespace()
    bare._has_stepped = True
    bare._frozen_envs = torch.zeros(1, dtype=torch.bool)
    bare.episode_length_buf = torch.ones(1, dtype=torch.long)
    bare._reset_idx(torch.tensor([0]))
    assert calls == [[0]]


@torch.inference_mode()
def test_reset_to_records_restored_initial_state(candidate):
    env = candidate
    state = env.scene.get_state(is_relative=True)
    state["articulation"]["robot"]["joint_position"][:, 0] += 0.1
    counter = env._sim_step_counter
    env.reset_to(state, env_ids=None, is_relative=True)
    assert env._sim_step_counter == counter
    initial = env.recorder_manager._episodes[0].data["initial_state"]["articulation"]["robot"]["joint_position"]
    torch.testing.assert_close(initial, state["articulation"]["robot"]["joint_position"])
    torch.testing.assert_close(env.obs_buf["proprio_obs"]["arm_joint_pos"][:, 0], initial[:, 0])


@torch.inference_mode()
def test_failure_is_exported_as_unsuccessful(candidate, monkeypatch, tmp_path):
    env = candidate
    term = env.termination_manager
    success = term._term_names.index("success")
    # Ordinary manager termination with a failure consumer is outside eligibility.
    term._term_names[success] = "failure"
    term._term_dones["failure"] = term._term_dones.pop("success")
    term._term_cfgs[success].func = lambda env, **kw: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    env._omy_held_target = env._omy_compact_step = False
    env.recorder_manager.set_episode_index(0, [0])
    action = torch.tensor([[*[env.cfg.scene.robot.init_state.joint_pos[name] for name in ARM_JOINTS], 1.0]], device=env.device)
    env.step(action)
    assert env.all_terminated and not env.get_env_results()[0]["success"]
    close_recording(env)
    with h5py.File(tmp_path / "omy_demos.hdf5") as file:
        assert not file["data/demo_0"].attrs["success"]
        assert file["data/demo_0/actions"].shape == (1, 8)


def test_device_records_the_sample_used_by_advance(monkeypatch):
    import time
    from types import SimpleNamespace

    from omy_leader_isaaclab.device import OmyLeaderDevice

    from robolab.core.omy_teleop.device import RoboLabOmyLeaderDevice

    device = RoboLabOmyLeaderDevice.__new__(RoboLabOmyLeaderDevice)
    reading = {"sample": 1}
    monkeypatch.setattr(OmyLeaderDevice, "_open_source", lambda self: setattr(self, "_latest", lambda: (reading, 0.01)))
    monkeypatch.setattr(time, "time", lambda: 100.3)
    monkeypatch.setattr(time, "monotonic_ns", lambda: 777)
    device.cfg = SimpleNamespace(source="serial", stale_s=0.5, sim_device="cpu")
    device._buf = (reading, 100.2)
    device._open_source()
    device._latest()  # READY observes a different sample before command admission.
    device._buf = ({"sample": 2}, 100.25)
    seen = []
    device._mapper = SimpleNamespace(step=lambda sample: seen.append(sample) or np.zeros(8))
    device._zeroed = True
    device._warned = False
    action = device.advance()
    assert action.shape == (8,) and seen == [{"sample": 2}]
    assert device.last_input_sample["source_wall_ns"] == 100_250_000_000
    assert device.last_input_sample["age_s"] == pytest.approx(0.05)
    assert device.last_input_sample["read_monotonic_ns"] == 777
