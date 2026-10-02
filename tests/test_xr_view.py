"""Isaac Sim integration checks; no SteamVR, headset, or leader required."""

import torch
from isaaclab.sensors import Camera, CameraCfg

import robolab.constants
from robolab.core.environments.runtime import create_env
from robolab.core.xr_view.options import build_parser
from robolab.core.xr_view.scene import build_scene_cfg
from robolab.robots.droid import DroidCfg


def test_xr_view_scene_has_no_control_or_camera_sensors_and_stays_frozen(tmp_path):
    args = build_parser().parse_args(["--desktop"])
    cfg = build_scene_cfg(args)
    assert not any(isinstance(getattr(cfg.scene, name), CameraCfg) for name in dir(cfg.scene))
    assert cfg.scene.robot.init_state.joint_pos == DroidCfg().robot.init_state.joint_pos
    # Scene construction must not mutate standard DROID's camera or home.
    assert isinstance(DroidCfg().wrist_cam, CameraCfg)
    previous_output = robolab.constants.get_output_dir()
    robolab.constants.set_output_dir(str(tmp_path))
    env = None
    try:
        with torch.inference_mode():
            env, _ = create_env(cfg, policy="view_only", rendering_mode="performance")
            assert env.action_manager.total_action_dim == 0
            assert not env.observation_manager.active_terms
            assert not any(isinstance(sensor, Camera) for sensor in env.scene.sensors.values())
            env.reset(seed=0)
            q = env.scene["robot"].data.joint_pos.clone()
            banana = env.scene["banana"].data.root_state_w.clone()
            bowl = env.scene["bowl"].data.root_state_w.clone()
            physics_index = env.sim.current_time_step_index
            for _ in range(20):
                env.sim.render()
            assert env.sim.current_time_step_index == physics_index
            # Refresh caches to avoid merely comparing cached pre-render state.
            env.scene.update(0.0)
            torch.testing.assert_close(env.scene["robot"].data.joint_pos, q, rtol=0, atol=0)
            torch.testing.assert_close(env.scene["banana"].data.root_state_w, banana, rtol=0, atol=0)
            torch.testing.assert_close(env.scene["bowl"].data.root_state_w, bowl, rtol=0, atol=0)
    finally:
        if env is not None:
            env.close()
        robolab.constants.set_output_dir(str(previous_output))
