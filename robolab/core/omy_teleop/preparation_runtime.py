"""Unrecorded scene preparation, imported only after Isaac starts."""

from pathlib import Path

import torch

from .options import ARM_JOINTS
from .preparation import PreparationSettings, StabilityGate, supported_root_height
from robolab.core.scenes.placement import bounds_corners


def configure_preparation(cfg, args):
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sensors import ContactSensorCfg

    settings = PreparationSettings.from_args(args)
    cfg.omy_preparation = settings.metadata()
    for name, asset in list(vars(cfg.scene).items()):
        if isinstance(asset, RigidObjectCfg):
            setattr(cfg.scene, "omy_prepare_" + name, ContactSensorCfg(
                prim_path=asset.prim_path, update_period=0.0, history_length=0,
                track_air_time=False, filter_prim_paths_expr=[]))


class ScenePreparation:
    def __init__(self, env, args):
        from robolab.core.world.world_state import get_world

        self.env = env
        self.settings = PreparationSettings.from_args(args)
        self.gate = StabilityGate(self.settings)
        self.elapsed_s = 0.0
        self.world = get_world(env)
        self.placement = []
        self.action = torch.tensor([[*[env.cfg.scene.robot.init_state.joint_pos[n] for n in ARM_JOINTS], 1.0]],
                                   device=env.device)
        self.names = list(env.scene.rigid_objects)
        path = getattr(getattr(getattr(env.cfg.scene, "scene", None), "spawn", None), "usd_path", "")
        self.flat_table = Path(str(path)).parent.name == "simple_pnp" and "table" in self.names
        self.setup_error = None
        try:
            self._correct_placement()
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            self.setup_error = f"{type(error).__name__}: {error}"
        self.started = env._sim_step_counter

    def _bounds(self, name):
        from pxr import Gf, Usd, UsdGeom

        # ComputeUntransformedBound excludes this prim's pose and scale while
        # retaining descendant transforms. ComputeLocalBound includes the root
        # transform and must not be combined with a second PhysX root pose.
        prim = self.world._get_prim(name, env_id=0)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        bounds = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        if bounds.IsEmpty():
            raise ValueError(f"Cannot prepare {name}: missing geometry bounds")
        transform = UsdGeom.Xformable(prim).GetLocalTransformation()
        scale = Gf.Transform(transform).GetScale()
        pose = self.env.scene.rigid_objects[name].data.root_pose_w[0].detach().cpu().numpy()
        return bounds_corners(bounds.GetMin(), bounds.GetMax(), pose, scale)

    def _correct_placement(self):
        env = self.env
        # Only flat simple-PnP layouts declare all targets on the same table.
        # Other tasks can intentionally start stacked or inside a container.
        if not self.flat_table:
            return
        self.world.clear_contact_cache()
        table = self._bounds("table")
        surface_z = float(table[:, 2].max())
        lo, hi = table[:, :2].min(axis=0), table[:, :2].max(axis=0)
        for name, asset in env.scene.rigid_objects.items():
            if name == "table":
                continue
            corners = self._bounds(name)
            center = corners.mean(axis=0)
            if not bool(((center[:2] >= lo) & (center[:2] <= hi)).all()):
                raise ValueError(f"Cannot prepare {name}: authored placement is outside the table")
            pose = asset.data.root_pose_w.clone()
            before = pose[0, 2].item()
            pose[0, 2] = supported_root_height(before, float(corners[:, 2].min()),
                                                surface_z,
                                                self.settings.clearance_m)
            asset.write_root_pose_to_sim(pose)
            asset.write_root_velocity_to_sim(torch.zeros_like(asset.data.root_vel_w))
            self.placement.append({"body": name, "root_z_before": before, "root_z_after": pose[0, 2].item()})
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=0.0)
        self.world.clear_contact_cache()

    def advance(self):
        env = self.env
        if self.setup_error is not None:
            return {"status": "failed", "elapsed_s": 0.0, "stable_s": 0.0,
                    "reasons": [self.setup_error], "bodies": {}, "settings": self.settings.metadata(),
                    "placement": self.placement}
        env.action_manager.process_action(self.action)
        for _ in range(env.cfg.decimation):
            env._sim_step_counter += 1
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            if env._sim_step_counter % env.cfg.sim.render_interval == 0:
                env.sim.render()
            env.scene.update(dt=env.physics_dt)
        self.elapsed_s = (env._sim_step_counter - self.started) * env.physics_dt
        self.world.clear_contact_cache()
        bodies = {}
        for name in self.names:
            asset = env.scene.rigid_objects[name]
            force = env.scene.sensors["omy_prepare_" + name].data.net_forces_w[0].sum(dim=0)
            supported = force[2].item() > self.settings.support_force_n
            if self.flat_table and name != "table":
                supported = supported and bool(self.world.is_supported_on_surface(
                    name, "table", force_threshold=self.settings.support_force_n, env_id=0))
            bodies[name] = {"pose": asset.data.root_pose_w[0].detach().cpu().numpy(),
                            "velocity": asset.data.root_vel_w[0].detach().cpu().numpy(),
                            "supported": supported}
        report = self.gate.observe(self.elapsed_s, bodies)
        report.update(settings=self.settings.metadata(), placement=self.placement)
        if report["status"] == "ready":
            # Clear caches/timers and capture the prepared boundary, without a
            # scene reset that would restore the authored airborne poses.
            env.reset_eval_state()
            env.episode_length_buf.zero_()
            env.termination_manager.reset()
            env.reward_manager.reset()
            env.command_manager.reset()
            env.observation_manager.reset()
            env.invalidate_omy_state()
            env.obs_buf = env.observation_manager.compute(update_history=False)
            env.recorder_manager.clear()
            env.recorder_manager.reset()
            env.recorder_manager.record_post_reset(None)
            env.omy_preparation_report = report
        return report
