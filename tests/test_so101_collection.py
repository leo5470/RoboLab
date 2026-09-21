"""Real CPU PhysX/IK recording with aligned leader diagnostics."""

from dataclasses import replace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from isaaclab.managers import DatasetExportMode
from isaaclab.sensors import ContactSensorCfg

from robolab.core.environments.config import parse_env_cfg
from robolab.core.environments.factory import get_envs
from robolab.core.environments.runtime import create_env
from robolab.core.replay.materialize import RecordedEpisode, copy_episode
from robolab.core.teleop.lifecycle import LeaderControls
from robolab.core.teleop.recorders import LeaderDiagnosticsRecorderCfg
from robolab.core.teleop.retarget import PoseRetargeter, RetargetConfig, validate_action_config
from robolab.core.teleop.so101_controller import LeaderController
from robolab.core.teleop.so101_source import LeaderSample
from robolab.registrations.droid.auto_env_registrations_rel_ik import auto_register_droid_rel_ik_envs
from robolab.robots.droid import ee_pos, ee_quat


@pytest.mark.parametrize("flush_interval", [0, 2])
def test_so101_records_exact_actions_and_aligned_diagnostics(tmp_path, monkeypatch, flush_interval):
    monkeypatch.setattr("robolab.constants.RECORD_IMAGE_DATA", True)
    monkeypatch.setattr("robolab.core.environments.runtime.get_output_dir", lambda: str(tmp_path))
    suffix = f"SO101RecordTest{flush_interval}"
    auto_register_droid_rel_ik_envs(
        task="BananaInBowlTask", env_postfix=suffix, include_policy_cameras=False, include_viewport_camera=False
    )
    name = next(n for n in get_envs(task="BananaInBowlTask") if suffix in n)
    cfg = parse_env_cfg(name, device="cpu", num_envs=1, seed=0, use_fabric=True)
    cfg.recorders.dataset_export_dir_path = str(tmp_path)
    cfg.recorders.record_so101 = LeaderDiagnosticsRecorderCfg()
    for sensor in vars(cfg.scene).values():
        if isinstance(sensor, ContactSensorCfg):
            sensor.history_length = 0
    env, _ = create_env(cfg, policy="so101_test", rendering_mode="performance")
    try:
        recorder = env.recorder_manager
        recorder.set_buffered_recording()
        recorder.cfg.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        recorder.set_hdf5_compression(None)
        recorder.set_hdf5_file("leader_test.hdf5")
        recorder.set_flush_interval(flush_interval)
        recorder.set_flush_memory_cleanup(False)
        env.reset(seed=17)
        recorder.set_episode_seed(17, [0])
        recorder.set_episode_index(0, [0])
        initial_joints = env.scene["robot"].data.joint_pos.clone()
        initial_position = ee_pos(env)[0].cpu().numpy().copy()
        controls = LeaderControls()
        source = SimpleNamespace(
            sample=LeaderSample("test", 0, 0.0, 0.0, np.zeros(5), 100.0, np.zeros(3), np.array([1.0, 0, 0, 0]))
        )
        source.current = lambda: source.sample
        mapper = PoseRetargeter(RetargetConfig(filter_alpha=1), validate_action_config(env.cfg.actions))
        controller = LeaderController(source, mapper, controls, clock=lambda: 0.1)
        controls.request_start()
        controller.begin(ee_pos(env)[0].cpu().numpy(), ee_quat(env)[0].cpu().numpy())
        actions = []
        for index in range(8):
            source.sample = replace(source.sample, sequence=index + 1, position=np.array([0.002 * (index + 1), 0, 0]))
            action = controller.action(ee_pos(env)[0].cpu().numpy(), ee_quat(env)[0].cpu().numpy())
            actions.append(action.copy())
            env.so101_diagnostics = {
                key: torch.as_tensor(value.copy()).unsqueeze(0) for key, value in controller.diagnostics.items()
            }
            env.step(torch.tensor(action).unsqueeze(0))
            assert not env.all_terminated
        assert not torch.allclose(initial_joints, env.scene["robot"].data.joint_pos)
        controls.request_clutch()
        before = env.episode_length_buf.clone()
        assert controller.action(ee_pos(env)[0].cpu().numpy(), ee_quat(env)[0].cpu().numpy()) is None
        torch.testing.assert_close(env.episode_length_buf, before)
        # The test fixture explicitly labels its temporary test episode successful.
        recorder.set_success_to_episodes([0], torch.ones(1, dtype=torch.bool))
        recorder.export_episodes([0])
        with h5py.File(tmp_path / "leader_test.hdf5") as f:
            demo = f["data/demo_0"]
            recorded = RecordedEpisode.load(demo)
            assert recorded.num_actions == 8
            assert recorded.seed == 17
            np.testing.assert_array_equal(demo["actions"][:], actions)
            np.testing.assert_array_equal(demo["teleop/so101/sequence"][:, 0], np.arange(1, 9))
            assert demo["teleop/so101/joints_deg"].shape == (8, 5)
            np.testing.assert_allclose(demo["obs/proprio_obs/ee_pos"][0], initial_position)
            assert demo["states/articulation/robot/joint_position"].shape[0] == 8
            assert recorded.state_for(0)[0] is recorded.initial_state
            assert recorded.state_for(1)[0] is recorded.states
            with h5py.File(tmp_path / "copied.hdf5", "w") as dest:
                copy_episode(demo, dest.create_group("data"), "demo_0")
                np.testing.assert_array_equal(
                    dest["data/demo_0/teleop/so101/sequence"][:], demo["teleop/so101/sequence"][:]
                )
    finally:
        env.close()
