"""Rotate layout recordings while retaining one simulator and environment."""

import json
from pathlib import Path


def apply_layout(env, cfg, args, layout):
    """Install the authored pose as the default used by every subsequent retry."""
    import torch

    layout.verify_current()
    for name, values in layout.poses.items():
        asset = env.scene.rigid_objects[name]
        pose = torch.as_tensor(values, dtype=asset.data.default_root_state.dtype, device=env.device)
        asset.data.default_root_state[:, :7] = pose
        asset.data.default_root_state[:, 7:] = 0
        for initial in (asset.cfg.init_state, getattr(cfg.scene, name).init_state):
            initial.pos = tuple(values[:3])
            initial.rot = tuple(values[3:7])
    cfg.scene.scene.spawn.usd_path = str(layout.scene_file)
    cfg._task_name = layout.task
    cfg.env_name = layout.task + "OMYL100"
    args.task = layout.task
    args.task_resolution = layout.resolution
    args.task_dirs = list(layout.resolution.task_dirs)
    env.recorder_manager._env_name = cfg.env_name


class LayoutRecording:
    def __init__(self, env, cfg, args, device_cfg, xr_session, session, collection_metadata):
        self.env, self.cfg, self.args = env, cfg, args
        self.device_cfg, self.xr_session, self.session = device_cfg, xr_session, session
        self.collection_metadata = collection_metadata
        self.part = None

    def open(self, part, *, initialize=False):
        from robolab.constants import set_output_dir
        from .runner import configure_recording

        self.part = Path(part)
        layout = self.session.current
        apply_layout(self.env, self.cfg, self.args, layout)
        self.env.output_dir = str(self.part)
        set_output_dir(str(self.part))
        recorder = self.env.recorder_manager
        recorder.clear()
        recorder.cfg.dataset_export_dir_path = str(self.part)
        self.cfg.recorders.dataset_export_dir_path = str(self.part)
        attributes = {
            "collection_task": layout.task,
            "robolab_omy_collection": json.dumps({
                **self.collection_metadata,
                "object": self.session.plan.object_name,
                "catalog_index": self.session.plan.catalog_index,
                "layout": layout.metadata(),
                "session": str(self.session.root),
                "part": self.part.relative_to(self.session.root).as_posix(),
            }),
        }
        configure_recording(self.env, self.args, self.device_cfg, self.xr_session,
                            initialize=initialize, dataset_attrs=attributes)
        recorder.set_flush_interval(0)
        (self.part / "env_cfg.json").write_text(json.dumps(self.cfg.to_dict(), default=str) + "\n")

    def finalize(self, episode_index, success, reason):
        recorder = self.env.recorder_manager
        recorder._dataset_file_handler.set_episode_attrs(episode_index, {
            "collection_reason": reason,
            "collection_task": self.session.current.task,
            "collection_finalized": True,
        })
        self.session.record_attempt(success=success, reason=reason, part=self.part, episode_index=episode_index)
