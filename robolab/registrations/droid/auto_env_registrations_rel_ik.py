# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import robolab.constants
from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR


def robot_cfg_without_cameras(robot_cfg):
    """Return a copy of a robot config with its robot-mounted cameras removed.

    Some cameras are attached to the robot rather than the scene (the DROID
    wrist camera lives under the gripper prim), so dropping the scene camera
    mixins is not enough to build a camera-free environment. ``InteractiveScene``
    skips config members that are ``None``, so the sensors are shadowed rather
    than deleted, leaving the rest of the robot config untouched.

    The camera fields are found on an instance rather than the class:
    ``@configclass`` turns mutable defaults into dataclass default factories, so
    they are not readable as class attributes.
    """
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass

    camera_names = [name for name, value in vars(robot_cfg()).items() if isinstance(value, CameraCfg)]
    if not camera_names:
        return robot_cfg
    members = {name: None for name in camera_names}
    members["__annotations__"] = {name: "CameraCfg | None" for name in camera_names}
    return configclass(type(f"{robot_cfg.__name__}NoCameras", (robot_cfg,), members))


def auto_register_droid_rel_ik_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, task=None, cameras=None,
                                    randomize_background=False, background_seed=None,
                                    env_postfix="", include_viewport_camera=True,
                                    include_policy_cameras=True):
    """Register tasks against ``DroidRelIKActionCfg`` (relative end-effector pose IK).

    Mirrors :func:`robolab.registrations.droid.auto_env_registrations_jointpos.auto_register_droid_envs`
    but swaps the action config from joint-position to relative differential-IK. Used by
    policies that emit (dx, dy, dz, droll, dpitch, dyaw, gripper) deltas — e.g. VAMs
    trained on LIBERO-format OSC_POSE data.

    Args:
        include_viewport_camera: Add the third-person ``viewport_cam`` observation
            group and its scene sensor. Independent of the Kit viewport used for
            local/WebRTC viewing, which needs no sensor at all.
        include_policy_cameras: Register the policy cameras (the ``image_obs``
            group, its scene sensors, and the robot-mounted wrist camera). Set
            False for state-only collection: no RTX sensor is created, so nothing
            renders per control step, while the Kit/WebRTC viewport keeps working
            because it draws the stage directly. The images are reconstructed
            afterwards from the recorded states — see
            ``scripts/materialize_demo_images.py`` and
            :mod:`robolab.core.replay.materialize`.
    """
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
    from robolab.registrations.droid.camera_presets import WRIST_LEFT
    from robolab.robots.droid import (
        DroidCfg,
        DroidRelIKActionCfg,
        ProprioceptionObservationCfg,
        WristCameraCfg,
        contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    if cameras is None:
        cameras = WRIST_LEFT
    if not include_policy_cameras:
        cameras = []

    # Proprioception is computed from articulation state, so it stays cheap and
    # available with every camera removed.
    observation_groups = {"proprio_obs": ProprioceptionObservationCfg()}
    if include_policy_cameras:
        ImageObsCfg = generate_image_obs_from_cameras(cameras)
        observation_groups = {"image_obs": ImageObsCfg(), **observation_groups}
    viewport_scene_cameras = []
    if include_viewport_camera:
        ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
        observation_groups["viewport_cam"] = ViewportCameraCfg()
        viewport_scene_cameras = [EgocentricMirroredCameraCfg]
    ObservationCfg = generate_obs_cfg(observation_groups)

    # WristCameraCfg is robot-mounted (wrist_cam is already attached via DroidCfg).
    # Including it as a scene mixin puts wrist_cam before robot in dataclass field
    # order, causing the camera to spawn before its parent prim exists.
    scene_cameras = [c for c in cameras if c is not WristCameraCfg]
    robot_cfg = DroidCfg if include_policy_cameras else robot_cfg_without_cameras(DroidCfg)

    if randomize_background:
        from robolab.variations.backgrounds import find_background_files, generate_background_config

        rng = random.Random(background_seed)
        all_bgs = find_background_files()
        default_bg_path = HomeOfficeBackgroundCfg.dome_light.spawn.texture_file
        all_bgs = [p for p in all_bgs if p != default_bg_path]
        if not all_bgs:
            raise FileNotFoundError(
                "No backgrounds available for randomization after excluding the default."
            )

        def _bg_factory():
            return generate_background_config(rng.choice(all_bgs))

        background_cfg = _bg_factory
    else:
        background_cfg = HomeOfficeBackgroundCfg

    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=task_dirs,
        tasks=task,
        pattern="*.py",
        env_prefix="",
        env_postfix=env_postfix,
        observations_cfg=ObservationCfg(),
        actions_cfg=DroidRelIKActionCfg(),
        robot_cfg=robot_cfg,
        camera_cfg=[*scene_cameras, *viewport_scene_cameras],
        lighting_cfg=SphereLightCfg,
        background_cfg=background_cfg,
        contact_gripper=contact_gripper,
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        seed=1,
    )

    if robolab.constants.VERBOSE:
        from robolab.core.environments.factory import print_env_table
        print_env_table()
