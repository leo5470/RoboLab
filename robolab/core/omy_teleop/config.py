"""RoboLab task configuration for upstream OMY joint-space control.

Import after Isaac Sim starts. Control math and the device remain upstream.
"""

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.devices import DevicesCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from omy_leader_isaaclab.device import OmyLeaderCfg
from omy_leader_isaaclab.franka_config import FRANKA_HOME
from scipy.spatial.transform import Rotation

from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR
from robolab.core.environments.factory import auto_discover_and_create_cfgs
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
from robolab.robots.droid import DroidCfg, ProprioceptionObservationCfg, WristCameraCfg, contact_gripper
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
from robolab.variations.camera import OverShoulderLeftCameraCfg
from robolab.variations.lighting import SphereLightCfg

from .options import ARM_JOINTS

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


_robot = DroidCfg().robot.copy()
_robot.init_state.joint_pos.update(dict(zip(ARM_JOINTS, FRANKA_HOME)))


@configclass
class OmyDroidCfg(DroidCfg):
    robot = _robot
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


def register_omy_envs(task, *, images=False, stream=False, env_postfix=ENV_POSTFIX):
    """Register dedicated task variants; never alter the standard DROID registrations."""
    @configclass
    class SelectedRobotCfg(OmyDroidCfg):
        wrist_cam = DroidCfg().wrist_cam if images else None
        # Robot-mounted sensors must follow the robot in scene field order.
        # Putting this in the scene-camera mixin spawns it before its parent prim.
        omy_wrist_cam = _operator_wrist_camera() if stream else None

    groups = {"proprio_obs": ProprioceptionObservationCfg()}
    cameras = []
    if images:
        cameras.append(OverShoulderLeftCameraCfg)
        groups["image_obs"] = generate_image_obs_from_cameras([OverShoulderLeftCameraCfg, WristCameraCfg])()
    if stream:
        cameras.append(OmyOperatorCamerasCfg)
    return auto_discover_and_create_cfgs(
        task_dir=TASK_DIR, task_subdirs=DEFAULT_TASK_SUBFOLDERS, tasks=task, pattern="*.py",
        env_postfix=env_postfix, observations_cfg=generate_obs_cfg(groups)(),
        actions_cfg=OmyDroidActionsCfg(), robot_cfg=SelectedRobotCfg, camera_cfg=cameras,
        lighting_cfg=SphereLightCfg, background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper, dt=1 / 120, decimation=2, render_interval=2, seed=1,
        eye=(-1.25, 0.0, 1.55), lookat=(0.55, 0.0, 0.05),
    )


def configure_device(env_cfg, args):
    if "gripper_open_pos" not in OmyLeaderCfg.__dataclass_fields__:
        raise RuntimeError("Install the L100 trigger correction: python scripts/patch_omy.py")
    cfg = OmyLeaderCfg(
        target="franka", source=args.source, port=args.port, baudrate=args.baudrate,
        gripper_open_pos=args.gripper_open,
        tcp_host=args.tcp_host, tcp_port=args.tcp_port, calib_path=args.calib,
        auto_zero=args.auto_zero, auto_zero_settle_s=args.auto_zero_settle_s,
        stale_s=args.stale_s, dt=env_cfg.sim.dt * env_cfg.decimation,
        vel_scale=args.vel_scale, hz=args.read_hz, sim_device=env_cfg.sim.device,
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
