"""RoboLab task configuration for upstream OMY joint-space control.

Import after Isaac Sim starts. Control math remains upstream, with a local home adapter.
"""

import math
import time

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.devices import DevicesCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from omy_leader_isaaclab.device import OmyLeaderCfg
from scipy.spatial.transform import Rotation

from robolab.constants import TASK_DIR
from robolab.core.environments.factory import create_env_cfg
from robolab.core.task.task_index import normalize_task_dirs, resolve_selected_task
from robolab.core.task.task_utils import load_task_from_file
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
from robolab.robots.droid import DroidCfg, ProprioceptionObservationCfg, WristCameraCfg, contact_gripper
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
from robolab.variations.camera import OverShoulderLeftCameraCfg
from robolab.variations.lighting import SphereLightCfg

from .device import RoboLabOmyLeaderCfg
from .options import ARM_JOINTS, control_decimation
from . import observations as omy_observations
from .performance import render_interval
from .contact_profile import selected_contact_sensors
from .requirements import build_task_requirements

ENV_POSTFIX = "OMYL100"


@configclass
class OmyDroidActionsCfg:
    # Keep upstream action values and order, including the +1/-1 gripper.
    body = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=list(ARM_JOINTS), preserve_order=True,
        scale=1.0, use_default_offset=False,
    )
    finger_joint = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0}, close_command_expr={"finger_joint": math.pi / 4},
    )


@configclass
class OmyDroidCfg(DroidCfg):
    # Inherit DROID's reset pose; don't substitute upstream's cube-demo home.
    wrist_cam = None


def _overview_rotation():
    # Upstream operator viewpoint; Isaac's world camera convention is +X forward, +Z up.
    forward = np.array((0.55, 0.0, 0.05)) - np.array((-1.25, 0.0, 1.55))
    forward /= np.linalg.norm(forward)
    left = np.cross((0.0, 0.0, 1.0), forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    xyzw = Rotation.from_matrix(np.stack((forward, left, up), axis=1)).as_quat()
    return tuple(xyzw[[3, 0, 1, 2]])


@configclass
class OmyOperatorCamerasCfg:
    view_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/omy_view_cam", update_period=0.0,
        height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5.0)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-1.25, 0.0, 1.55), rot=_overview_rotation(), convention="world"
        ),
    )


def _operator_wrist_camera():
    # DROID has a Robotiq mount instead of panda_hand. Use its calibrated mount/lens;
    # retain upstream's 240x180 operator inset and separate it from policy observations.
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/omy_operator_wrist",
        update_period=0.0, height=180, width=240, data_types=["rgb"],
        spawn=DroidCfg().wrist_cam.spawn.copy(),
        offset=DroidCfg().wrist_cam.offset.copy(),
    )


def register_omy_envs(task, *, images=False, stream=False, env_postfix=ENV_POSTFIX,
                      optimized_observations=True, task_dirs=None, resolution=None, expected_source=None):
    """Register dedicated task variants; never alter the standard DROID registrations."""
    if resolution is None:
        resolution = resolve_selected_task(task, task_dirs, task_root=TASK_DIR, expected_source=expected_source)
    elif (resolution.class_name != task and "/" not in task and "\\" not in task and not task.endswith(".py")):
        raise ValueError("Task name conflicts with its preflight resolution")
    if task_dirs is not None and normalize_task_dirs(task_dirs) != resolution.task_dirs:
        raise ValueError("Task directories conflict with their preflight resolution")
    resolution.verify_expected_source(expected_source)
    if resolution.class_name != task:
        # Filename/path callers are accepted; consumers should use the returned
        # canonical key rather than recording a machine-specific file path.
        task = resolution.class_name
    resolution.verify_current()
    started = time.perf_counter()
    task_class = load_task_from_file(str(resolution.task_file))
    if task_class.__name__ != resolution.class_name:
        raise ValueError(f"Resolved task {resolution.class_name!r} imported as {task_class.__name__!r}")
    import_seconds = time.perf_counter() - started
    started = time.perf_counter()
    @configclass
    class SelectedRobotCfg(OmyDroidCfg):
        wrist_cam = DroidCfg().wrist_cam if images else None
        # Robot-mounted sensors must follow the robot in scene field order.
        # Putting this in the scene-camera mixin spawns it before its parent prim.
        omy_wrist_cam = _operator_wrist_camera() if stream else None

    proprio = ProprioceptionObservationCfg()
    if optimized_observations:
        for name in ("ee_pos", "ee_quat", "eef_pos", "eef_quat"):
            getattr(proprio, name).func = getattr(omy_observations, name)
    groups = {"proprio_obs": proprio}
    cameras = []
    if images:
        cameras.append(OverShoulderLeftCameraCfg)
        groups["image_obs"] = generate_image_obs_from_cameras([OverShoulderLeftCameraCfg, WristCameraCfg])()
    if stream:
        cameras.append(OmyOperatorCamerasCfg)
    cfg = create_env_cfg(
        str(resolution.task_file), task_dir=TASK_DIR,
        env_postfix=env_postfix, observations_cfg=generate_obs_cfg(groups)(),
        actions_cfg=OmyDroidActionsCfg(), robot_cfg=SelectedRobotCfg, camera_cfg=cameras,
        lighting_cfg=SphereLightCfg, background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper, dt=1 / 120, decimation=2, render_interval=2, seed=1,
        eye=(-1.25, 0.0, 1.55), lookat=(0.55, 0.0, 0.05),
    )
    print(f"OMY task registration: {task}; lookup={resolution.lookup_seconds * 1000:.1f} ms "
          f"({resolution.method}), import={import_seconds * 1000:.1f} ms, "
          f"config/register={(time.perf_counter() - started) * 1000:.1f} ms; "
          "selected task modules=1", flush=True)
    return {resolution.class_name: cfg}


def configure_performance(env_cfg, args):
    """Modify only this OMY run's config, never shared DROID or task defaults."""
    env_cfg.decimation = control_decimation(args)
    env_cfg.sim.render_interval = render_interval(args, env_cfg.decimation)
    env_cfg.omy_stepping_profile = getattr(args, "stepping_profile", "ordinary")
    env_cfg.omy_substep_diagnostics = args.diagnostics
    env_cfg.omy_recording_profile = getattr(args, "recording_profile", "full")
    env_cfg.omy_episode_buffer_bytes = int(getattr(args, "episode_buffer_mb", 64) * 1024**2)
    cap = getattr(args, "solver_position_iterations", None)
    if cap is not None:
        # Change the scene cap before PhysX is created, not the USD actor request.
        physx = env_cfg.sim.physx
        if not hasattr(physx, "max_position_iteration_count"):
            raise ValueError("This OMY solver option requires scene iteration-cap support")
        if physx.min_position_iteration_count > cap:
            raise ValueError("The requested solver cap is below the scene minimum")
        physx.max_position_iteration_count = cap
    if env_cfg.omy_recording_profile == "compact":
        from .recording import configure_compact_recording

        configure_compact_recording(env_cfg, args)
    contact_names = [name for name, sensor in vars(env_cfg.scene).items()
                     if isinstance(sensor, ContactSensorCfg) and "__" in name]
    requirements = build_task_requirements(env_cfg)
    grippers = tuple(name for name, target in (env_cfg.contact_gripper or {}).items() if isinstance(target, str))
    effective, retained = selected_contact_sensors(requirements, contact_names, args.contact_sensors,
                                                 grippers=grippers)
    env_cfg.omy_dependency_plan = requirements.report()
    env_cfg.omy_contact_sensor_profile = effective
    env_cfg.omy_contact_sensor_count_full = len(contact_names)
    env_cfg.omy_contact_sensor_count = len(retained)
    if effective == "required":
        for name in contact_names:
            if name not in retained:
                setattr(env_cfg.scene, name, None)
    elif args.contact_sensors == "required":
        print("OMY contact sensor fallback: " + "; ".join(requirements.unknown), flush=True)
    print(f"OMY contact sensors: {effective}, {len(retained)}/{len(contact_names)} retained.", flush=True)
    for name, sensor in vars(env_cfg.scene).items():
        if isinstance(sensor, ContactSensorCfg) and not sensor.track_air_time:
            sensor = sensor.copy()
            sensor.history_length = args.contact_history_length
            setattr(env_cfg.scene, name, sensor)


def configure_device(env_cfg, args):
    if "gripper_open_pos" not in OmyLeaderCfg.__dataclass_fields__:
        raise RuntimeError("Install the L100 trigger correction: python scripts/patch_omy.py")
    cfg = RoboLabOmyLeaderCfg(
        target="franka", source=args.source, port=args.port, baudrate=args.baudrate,
        franka_initial_q=tuple(env_cfg.scene.robot.init_state.joint_pos[name] for name in ARM_JOINTS),
        gripper_open_pos=args.gripper_open,
        tcp_host=args.tcp_host, tcp_port=args.tcp_port, calib_path=args.calib,
        auto_zero=args.auto_zero, auto_zero_settle_s=args.auto_zero_settle_s,
        stale_s=args.stale_s, dt=env_cfg.sim.dt * env_cfg.decimation,
        # Retargeting is NumPy/CPU work. Validate on CPU before the runner's
        # single transfer, avoiding a GPU reduction/readback just to check 8 values.
        vel_scale=args.vel_scale, hz=args.read_hz, sim_device="cpu",
    )
    env_cfg.teleop_devices = DevicesCfg(devices={"omy_leader": cfg})
    return cfg


def validate_environment(env):
    if env.num_envs != 1 or env.action_manager.total_action_dim != 8:
        raise ValueError("OMY requires one environment with seven arm joints and a binary gripper")
    arm = env.action_manager.get_term("body")
    ids, names = env.scene["robot"].find_joints(list(ARM_JOINTS), preserve_order=True)
    if tuple(names) != ARM_JOINTS or tuple(arm._joint_names) != ARM_JOINTS:
        raise ValueError("OMY arm action ordering does not match panda_joint1..7")
    cfg = arm.cfg
    if cfg.use_default_offset or cfg.scale != 1.0 or cfg.offset != 0.0:
        raise ValueError("OMY needs absolute, unscaled joint targets")
    finger = env.action_manager.get_term("finger_joint").cfg
    if finger.class_type is not mdp.BinaryJointPositionAction:
        raise ValueError("OMY needs Isaac Lab's +1 open / -1 closed binary gripper")
    return ids
