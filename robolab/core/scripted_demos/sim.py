"""Everything that touches the simulator: Isaac app launch, stock DROID joint-position env creation (requirement R2
in docs/scripted_demos.md), the arm's IK controller, and reading the scene state (env-local frame, env 0).

Import order matters: ``launch_app`` must run before anything imports isaaclab or robolab modules.
"""

from __future__ import annotations

import pathlib

import numpy as np

from robolab.core.scripted_demos.grasps import frames, objects
from robolab.core.scripted_demos.kinematics import Chain
from robolab.core.scripted_demos.select import Scene

KIT_ARGS = (
    "--/app/settings/persistent=false --/renderer/multiGpu/enabled=false "
    "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8 "
    "--/app/fastShutdown=false"
)

_app = None
_registered: set[str] = set()


def kit_portable_root() -> pathlib.Path:
    """Kit's data/cache/logs root. Without it Kit writes into the venv (possibly another user's tree, R6), and its
    shader cache there is locked by other Kit processes, so every start recompiles shaders (~4 min)."""
    from robolab.core.scripted_demos import paths

    return paths.kit_portable_root()


def launch_app(headless: bool = True):
    """Start Isaac Sim once per process (cameras on: the PPS rows need both images)."""
    global _app
    if _app is None:
        import cv2  # noqa: F401  RoboLab imports cv2 before Isaac; keep that order.
        from isaaclab.app import AppLauncher

        root = kit_portable_root()
        root.mkdir(parents=True, exist_ok=True)
        _app = AppLauncher(headless=headless, device="cuda:0", enable_cameras=True,
                           kit_args=f"{KIT_ARGS} --portable-root {root}").app
    return _app


def make_env(task: str, output_dir: pathlib.Path, *, seed: int = 0, render: bool = True, events: dict | None = None):
    """Create one stock-registration env (``auto_register_droid_envs``, WRIST_LEFT cameras, 15 Hz).

    ``render=False`` keeps the same registration and physics but never calls the renderer (the camera observations
    go stale); it is for trials that record nothing, where rendering is most of the step time.
    Returns ``(env, env_cfg)``. RoboLab's own recorder writes into ``output_dir``.
    """
    import robolab.constants
    from robolab.constants import set_output_dir
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

    robolab.constants.RECORD_IMAGE_DATA = False
    robolab.constants.VERBOSE = False
    if task not in _registered:
        auto_register_droid_envs(task_dirs=["single_pnp"], task=[task])
        _registered.add(task)
    names = get_envs(task=[task])
    if len(names) != 1:
        raise RuntimeError(f"expected one registered env for {task}, got {names}")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_output_dir(str(output_dir))
    # ``events`` are merged after the task's own (``merge_events_cfg``), so a reset event runs after
    # ``reset_scene_to_default`` (sampled MimicGen scenes place their objects this way).
    env, env_cfg = create_env(names[0], device="cuda:0", num_envs=1, seed=seed, events=events)
    if not render:
        # ManagerBasedRLEnv.step/reset render only if the sim reports a GUI or RTX sensors.
        env.sim.has_rtx_sensors = lambda: False
    return env, env_cfg


def close_app(exit_status: int = 0) -> None:
    """Close Kit, bounded: Kit can hang in teardown, which would also hold its locks against the next process.

    Same watchdog as ``scripts/qualify_scene_trial.py``; all outputs are written before this.
    """
    import os
    import sys
    import threading

    sys.stdout.flush()
    sys.stderr.flush()
    watchdog = threading.Timer(20, lambda: os._exit(exit_status))
    watchdog.daemon = True
    watchdog.start()
    if _app is not None:
        _app.close()
    os._exit(exit_status)


# ---- arm control ---------------------------------------------------------------------------------------------------

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
MAX_JOINT_STEP = 0.04  # rad per control step


class ArmIK:
    """Absolute-pose IK step on the DROID ``base_link`` that produces joint targets for the stock env (D1).

    Derived from ``scripts/qualify_scene_trial.py:335-378``: IsaacLab's
    ``DifferentialIKController(pose, absolute, dls)`` over the seven arm joints (less the chain's ``locked`` ones, which
    keep their reset value), one step per control step, the joint change clamped to ±0.04 rad and then to the soft
    joint limits.

    One difference: the qualify driver evaluates the controller at the *measured* joints and clamps around them. With
    the DROID arm's joint PD (stiffness 400, damping 80) the joints then move only about K/D x 0.04 = 0.2 rad/s, so a
    reach to pregrasp takes ~60 control steps. Here the controller is evaluated at the previous *command* (pose and
    Jacobian from the numpy chain, which matches the simulator to 0.01 mm, see ``kinematics``) and the clamp is around
    that command, so commands move at up to 0.6 rad/s and the PD follows a few steps behind. Commands are then a
    function of the plan alone, which also makes recorded actions replay exactly.
    """

    def __init__(self, env, chain: Chain, max_joint_step: float = MAX_JOINT_STEP):
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        self.env = env
        self.chain = chain
        self.robot = env.scene["robot"]
        self.max_joint_step = max_joint_step
        self.controller = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=1, device=env.device)
        self.arm_ids, _ = self.robot.find_joints(ARM_JOINTS, preserve_order=True)
        self.ee = self.robot.find_bodies("base_link")[0][0]
        self.finger = self.robot.find_joints(["finger_joint"])[0][0]
        self.q_cmd: np.ndarray | None = None

    # -- state, in the env-local frame (the robot root sits at the env origin with identity rotation) ---------------
    def _origin(self):
        return self.env.scene.env_origins[0]

    def joints(self) -> np.ndarray:
        return self.robot.data.joint_pos[0, self.arm_ids].cpu().numpy().astype(np.float64)

    def finger_position(self) -> float:
        """Raw ``finger_joint`` angle (rad)."""
        return float(self.robot.data.joint_pos[0, self.finger])

    def ee_pose(self) -> np.ndarray:
        pos = (self.robot.data.body_pos_w[0, self.ee] - self._origin()).cpu().numpy()
        quat = self.robot.data.body_quat_w[0, self.ee].cpu().numpy()
        return frames.pose(pos, quat)

    def root_pose(self) -> np.ndarray:
        pos = (self.robot.data.root_pos_w[0] - self._origin()).cpu().numpy()
        return frames.pose(pos, self.robot.data.root_quat_w[0].cpu().numpy())

    # -- control ----------------------------------------------------------------------------------------------------
    def reset_command(self) -> None:
        """Start integrating from the measured joints (call after every env reset)."""
        self.q_cmd = self.joints().astype(np.float32).astype(np.float64)

    def joint_target(self, T_target: np.ndarray) -> np.ndarray:
        """One DLS step from the last command towards ``T_target`` (env-local base_link pose) → joint target (7,)."""
        import torch

        if self.q_cmd is None:
            self.reset_command()
        device = self.env.device
        f32 = dict(device=device, dtype=torch.float32)
        q = self.q_cmd
        J, T = self.chain.jacobian(q)
        self.controller.set_command(torch.cat((torch.tensor(T_target[:3, 3], **f32),
                                               torch.tensor(frames.matrix_to_quat(T_target[:3, :3]), **f32)))[None])
        joints = torch.tensor(q, **f32)[None]
        wanted = self.controller.compute(torch.tensor(T[:3, 3], **f32)[None],
                                         torch.tensor(frames.matrix_to_quat(T[:3, :3]), **f32)[None],
                                         torch.tensor(J, **f32)[None], joints)
        command = joints + (wanted - joints).clamp(-self.max_joint_step, self.max_joint_step)
        limits = self.robot.data.soft_joint_pos_limits[:, self.arm_ids]
        command = torch.maximum(torch.minimum(command, limits[..., 1]), limits[..., 0])
        self.q_cmd = command[0].cpu().numpy().astype(np.float64)
        return self.q_cmd.copy()

    def pose_error(self, T_target: np.ndarray) -> tuple[float, float]:
        """(position error m, rotation error rad) of the measured base_link against ``T_target``."""
        T = self.ee_pose()
        return float(np.linalg.norm(T[:3, 3] - T_target[:3, 3])), frames.rotation_angle(T[:3, :3], T_target[:3, :3])


# ---- scene state ---------------------------------------------------------------------------------------------------

BIN = "grey_bin"
TABLE = "table"


def object_pose(world, name: str) -> np.ndarray:
    pos, quat = world.get_pose(name, env_id=0)
    return frames.pose(pos.cpu().numpy(), quat.cpu().numpy())


def pre_action_state(env, names) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """The robot's joints and the named objects' root poses now, as Isaac Lab's recorder stores scene state (root pose
    ``x y z qw qx qy qz`` relative to the env origin), and the robot's joint order."""
    robot = env.scene["robot"]
    state = {"articulation/robot/joint_position": robot.data.joint_pos[0].cpu().numpy()}
    origin = env.scene.env_origins[0]
    for name in names:
        pose = env.scene[name].data.root_state_w[0, :7].clone()
        pose[:3] -= origin
        state[f"rigid_object/{name}/root_pose"] = pose.cpu().numpy()
    return state, tuple(robot.joint_names)


def write_pose(env, name: str, T_env: np.ndarray) -> None:
    """Teleport a rigid object to an env-local pose, at rest (as RoboLab's ``reset_pose_uniform`` does)."""
    import torch

    asset = env.scene[name]
    ids = torch.tensor([0], device=env.device)
    pos = torch.tensor(np.asarray(T_env, dtype=np.float64)[:3, 3], device=env.device, dtype=torch.float32)
    quat = torch.tensor(frames.matrix_to_quat(np.asarray(T_env)[:3, :3]), device=env.device, dtype=torch.float32)
    pose = torch.cat([pos + env.scene.env_origins[0], quat])[None]
    asset.write_root_pose_to_sim(pose, env_ids=ids)
    asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=ids)


def bbox_corners(world, name: str) -> np.ndarray:
    """``WorldState.get_bbox`` corners. Exact for the unrotated bin and table; inflated for yawed YCB objects."""
    corners, _ = world.get_bbox(name, env_id=0)
    return np.array([[c[0], c[1], c[2]] for c in corners], dtype=np.float64)


def table_top(world) -> float:
    return float(bbox_corners(world, TABLE)[:, 2].max())


def read_scene(world, ik: ArmIK, obj: objects.BatchObject) -> Scene:
    """What the selector needs about the current reset."""
    return Scene(T_world_obj=object_pose(world, obj.name), object_points=objects.object_points(obj),
                 bin_corners=bbox_corners(world, BIN), table_z=table_top(world), q_current=ik.joints(),
                 T_world_ee=ik.ee_pose(), mass=objects.simulated_mass(obj.key))


def object_speed(world, name: str) -> float:
    return float(world.get_velocity(name, env_id=0)[:3].norm())


def closedness(ik: ArmIK) -> float:
    """Gripper observation convention: finger_joint / (pi/4), 0 open … 1 closed (unclipped)."""
    return ik.finger_position() / (np.pi / 4)


def finger_contact(world, obj: objects.BatchObject) -> bool | None:
    """Gripper-object contact from the task's contact sensors; None if the scene cannot report it."""
    try:
        return bool(world.in_contact(obj.name, "gripper", env_id=0))
    except Exception:
        return None
