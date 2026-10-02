"""Build a camera-free, zero-action RoboLab scene after SimulationApp starts."""


def build_scene_cfg(args):
    from isaaclab.devices.openxr.xr_cfg import remove_camera_configs
    from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass

    from robolab.constants import TASK_DIR
    from robolab.core.environments.config import auto_generate_task_env
    from robolab.core.task.task_utils import resolve_task_path
    from robolab.robots.droid import DroidCfg, contact_gripper
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.lighting import SphereLightCfg

    @configclass
    class EmptyCfg:
        pass

    @configclass
    class ViewingDroidCfg(DroidCfg):
        wrist_cam = None

    task_path, _ = resolve_task_path(args.task, TASK_DIR)
    cfg_type = auto_generate_task_env(
        str(task_path), robot_cfg=ViewingDroidCfg, camera_cfg=[],
        lighting_cfg=SphereLightCfg, background_cfg=HomeOfficeBackgroundCfg,
        observations_cfg=EmptyCfg(), actions_cfg=EmptyCfg(), contact_gripper=contact_gripper,
        dt=1 / 120, decimation=2, render_interval=2, seed=args.seed,
        eye=tuple(args.eye), lookat=tuple(args.lookat),
    )
    cfg = cfg_type()
    # Follow the installed Isaac Lab XR API, and verify its postcondition. No
    # image observation terms are constructed in this dedicated view config.
    remove_camera_configs(cfg)
    # Configclass can expose inherited defaults after delattr; shadow any such
    # CameraCfg explicitly, as InteractiveScene treats None as absent.
    for name in dir(cfg.scene):
        if isinstance(getattr(cfg.scene, name), CameraCfg):
            setattr(cfg.scene, name, None)
    cfg.sim.device = args.device
    cfg.recorders = RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
    return cfg
