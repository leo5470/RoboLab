"""Remote input regressions and recording-buffer compatibility."""

from types import SimpleNamespace

import carb.input
import h5py
import numpy as np
import pytest
import torch
from isaaclab.devices import Se3KeyboardCfg

from robolab.core.logging.buffered_episode import BufferedEpisodeData
from robolab.core.logging.streaming_hdf5_handler import StreamingHDF5DatasetFileHandler
from robolab.core.teleop.keyboard import HeldKeyState, ResponsiveSe3Keyboard
from robolab.core.teleop.performance import RealtimePacer


def test_clear_and_late_release_never_reverse_motion_or_open_gripper():
    keys = HeldKeyState()
    keys.event("W", "press")
    keys.event("K", "press")
    keys.event("L", "press")
    keys.event("W", "release")
    assert not (keys.held & keys.MOTION_KEYS)
    assert keys.closed
    keys.reset()
    keys.event("S", "release")
    assert not keys.held


def test_duplicate_presses_and_autorepeat_toggle_only_once():
    keys = HeldKeyState()
    assert keys.event("K", "press")
    assert not keys.event("K", "press")
    assert not keys.event("K", "repeat")
    assert keys.closed
    keys.event("K", "release")
    keys.event("K", "press")
    assert not keys.closed


def test_buffered_carb_input_reaches_action_without_rendering():
    keyboard = ResponsiveSe3Keyboard(Se3KeyboardCfg(sim_device="cpu", pos_sensitivity=0.02))
    interface = carb.input.acquire_input_interface()
    provider = interface.get_input_provider()
    # An isolated input device avoids feeding test events to the Kit UI.
    device = provider.create_keyboard("robolab-teleop-test")
    subscription = interface.subscribe_to_keyboard_events(device, keyboard._on_keyboard_event)
    starts = []
    keyboard.add_callback("N", lambda: starts.append(True))

    def event(key, kind):
        provider.buffer_keyboard_key_event(device, kind, getattr(carb.input.KeyboardInput, key), 0)

    press = carb.input.KeyboardEventType.KEY_PRESS
    release = carb.input.KeyboardEventType.KEY_RELEASE
    try:
        event("W", press)
        event("W", press)
        event("N", press)
        event("N", press)
        command = keyboard.advance()
        assert command[0].item() == pytest.approx(0.02)
        assert len(starts) == 1
        event("S", press)
        assert keyboard.advance()[0].item() == pytest.approx(0)
        event("W", release)
        assert keyboard.advance()[0].item() == pytest.approx(-0.02)
        event("K", press)
        event("L", press)
        event("S", release)
        command = keyboard.advance()
        assert torch.count_nonzero(command[:6]).item() == 0
        assert command[6].item() == -1  # closed, same convention as Isaac
    finally:
        interface.unsubscribe_to_keyboard_events(device, subscription)
        provider.destroy_keyboard(device)
        keyboard.close()


def test_pacer_accounts_for_work_and_never_catches_up_after_a_stall():
    now = [0.0]
    pumps = []
    def sleep(seconds):
        now[0] += seconds
    pacer = RealtimePacer(0.1, clock=lambda: now[0], sleep=sleep)
    pacer.wait()
    now[0] = 0.03  # work already consumed 30 ms of the period
    pacer.wait(lambda: pumps.append(now[0]))
    assert now[0] == pytest.approx(0.1)
    assert pumps
    now[0] = 1.0  # stalled for several periods
    pacer.wait()
    pacer.wait()
    assert now[0] == pytest.approx(1.1)


def test_growing_episode_preserves_snapshots_nested_keys_and_hdf5(tmp_path):
    episode = BufferedEpisodeData()
    episode.seed = 42
    episode.success = True
    image = torch.zeros((4, 4, 3), dtype=torch.uint8)
    for step in range(33):
        image.fill_(step)
        episode.add("obs", {"image_obs/camera": image})
        episode.add("actions", torch.full((7,), float(step)))
    image.fill_(255)  # modifying the source cannot change recorded frames
    assert episode.data["obs"]["image_obs"]["camera"].shape == (33, 4, 4, 3)
    assert episode.get_action(32)[0].item() == 32
    assert episode.data["actions"].is_contiguous()
    handler = StreamingHDF5DatasetFileHandler(compression=None)
    path = tmp_path / "buffered.hdf5"
    handler.create(str(path), env_name="test")
    handler.write_episode(episode)
    handler.close()
    with h5py.File(path) as f:
        np.testing.assert_array_equal(f["data/demo_0/obs/image_obs/camera"][:, 0, 0, 0], np.arange(33))
        assert f["data/demo_0"].attrs["seed"] == 42
        assert f["data/demo_0"].attrs["success"]
        assert f["data/demo_0"].attrs["num_samples"] == 33


def test_growing_episode_import_clear_and_shape_validation():
    episode = BufferedEpisodeData()
    episode.data = {"actions": torch.ones((3, 7))}
    episode.add("actions", torch.zeros(7))
    assert len(episode.data["actions"]) == 4
    with pytest.raises(ValueError, match="changed shape"):
        episode.add("actions", torch.zeros(6))
    episode.data = {}
    assert episode.is_empty()
    episode.add("actions", torch.full((7,), 2.0))
    assert episode.data["actions"].shape == (1, 7)
    assert (episode.data["actions"] == 2).all()


def test_recorder_keeps_buffering_and_metadata_after_flush(tmp_path):
    from isaaclab.managers import DatasetExportMode
    from robolab.core.logging.recorder_manager import RobolabRecorderManager

    recorder = RobolabRecorderManager.__new__(RobolabRecorderManager)
    recorder._env = SimpleNamespace(num_envs=1)
    recorder._term_names = ["actions"]
    recorder._terms = {}
    recorder._episodes = {0: BufferedEpisodeData()}
    recorder._episodes[0].seed = 13
    recorder._episodes[0].success = True
    recorder._episodes[0].add("actions", torch.zeros(7))
    recorder._episode_type = BufferedEpisodeData
    recorder._streaming_active = {0: False}
    recorder._current_episode_index = {0: 0}
    recorder._flush_memory_cleanup = False
    recorder._hdf5_initialized = True
    recorder._dataset_file_handler = StreamingHDF5DatasetFileHandler(compression=None)
    recorder._dataset_file_handler.create(str(tmp_path / "stream.hdf5"), env_name="test")
    recorder.cfg = SimpleNamespace(dataset_export_mode=DatasetExportMode.EXPORT_SUCCEEDED_ONLY)
    try:
        recorder.flush_buffer([0], verbose=False)
        assert isinstance(recorder._episodes[0], BufferedEpisodeData)
        assert recorder._episodes[0].seed == 13
        assert recorder._episodes[0].success
        assert recorder._episodes[0].is_empty()
    finally:
        recorder._dataset_file_handler.close()


def test_cpu_collection_moves_arm_and_exports_aligned_states(tmp_path, monkeypatch):
    """Exercise real PhysX/IK plus lazy contacts and the buffered recorder."""
    import robolab.constants
    from isaaclab.managers import DatasetExportMode
    from isaaclab.sensors import ContactSensorCfg
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env
    from robolab.core.replay.materialize import RecordedEpisode
    from robolab.registrations.droid.auto_env_registrations_rel_ik import auto_register_droid_rel_ik_envs

    monkeypatch.setattr("robolab.core.environments.runtime.get_output_dir", lambda: str(tmp_path))

    auto_register_droid_rel_ik_envs(task="BananaInBowlTask", env_postfix="KeyboardSmoke",
                                  include_policy_cameras=False, include_viewport_camera=False)
    name = next(name for name in get_envs(task="BananaInBowlTask") if "KeyboardSmoke" in name)
    cfg = parse_env_cfg(name, device="cpu", num_envs=1, seed=0, use_fabric=True)
    cfg.recorders.dataset_export_dir_path = str(tmp_path)
    for sensor in vars(cfg.scene).values():
        if isinstance(sensor, ContactSensorCfg):
            sensor.history_length = 0
    env, _ = create_env(cfg, policy="keyboard_test", rendering_mode="performance")
    try:
        recorder = env.recorder_manager
        recorder.set_buffered_recording()
        recorder.cfg.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        recorder.set_hdf5_compression(None)
        recorder.set_hdf5_file("cpu_smoke.hdf5")
        recorder.set_flush_interval(0)
        env.reset(seed=17)
        recorder.set_episode_seed(17, [0])
        recorder.set_episode_index(0, [0])
        initial_joints = env.scene["robot"].data.joint_pos.clone()
        action = torch.zeros((1, 7))
        action[:, 0] = 0.003
        for _ in range(8):
            env.step(action)
        assert not torch.allclose(initial_joints, env.scene["robot"].data.joint_pos)
        # Lazy reads report current-step forces using physics dt, not elapsed
        # control dt. They must agree with a direct read of the contact view.
        for sensor in env.scene.sensors.values():
            if not isinstance(sensor.cfg, ContactSensorCfg):
                continue
            actual = sensor.data.net_forces_w
            expected = sensor.contact_physx_view.get_net_contact_forces(dt=env.physics_dt).reshape_as(actual)
            torch.testing.assert_close(actual, expected)
        recorder.set_success_to_episodes([0], torch.ones(1, dtype=torch.bool))
        recorder.export_episodes([0])
        with h5py.File(tmp_path / "cpu_smoke.hdf5") as f:
            demo = f["data/demo_0"]
            episode = RecordedEpisode.load(demo)
            assert episode.num_actions == 8
            assert demo.attrs["seed"] == 17
            np.testing.assert_allclose(demo["actions"][:, 0], 0.003)
            assert demo["states/articulation/robot/joint_position"].shape[0] == 8
            assert demo["initial_state/articulation/robot/joint_position"].shape[0] == 1
            assert "image_obs" not in demo.get("obs", {})
    finally:
        env.close()
