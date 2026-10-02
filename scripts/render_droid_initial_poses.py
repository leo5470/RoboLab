"""Capture the reset pose of DROID benchmark tasks from five fixed cameras.

Example:
    .venv/bin/python scripts/render_droid_initial_poses.py --headless
"""

import argparse
import copy
import json
from pathlib import Path

import cv2  # import before Isaac Lab
import numpy as np
import torch
from isaaclab.app import AppLauncher


DEFAULT_TASKS = (
    "BananaInBowlTask",
    "RubiksCubeTask",
    "BowlInBinTask",
    "StackYellowOnRedTask",
)
CAMERAS = (
    ("left", "image_obs", "over_shoulder_left_camera"),
    ("right", "image_obs", "over_shoulder_right_camera"),
    ("overhead", "image_obs", "head_camera"),
    ("front_arm", "viewport_cam", "egocentric_mirrored_camera"),
    ("rear_arm", "image_obs", "rear_arm_camera"),
)
SHEET_CAMERAS = ("left", "right", "overhead", "rear_arm")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
parser.add_argument("--output-dir", type=Path, default=Path("output/droid_initial_poses"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--settle-renders", type=int, default=30,
                    help="Extra render passes after reset for materials to finish loading")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

if not torch.cuda.is_available():
    parser.error("Isaac Sim camera rendering needs a visible NVIDIA GPU")

simulation_app = AppLauncher(args).app

from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)
from robolab.variations.camera import (  # noqa: E402
    EgocentricMirroredCameraCfg,
    HeadCameraCfg,
    OverShoulderLeftCameraCfg,
    OverShoulderRightCameraCfg,
)
from isaaclab.utils import configclass  # noqa: E402


@configclass
class FrontArmCameraCfg:
    """Front view aimed at DROID's arm, with room for the tabletop."""

    egocentric_mirrored_camera = copy.deepcopy(
        EgocentricMirroredCameraCfg().egocentric_mirrored_camera
    )
    egocentric_mirrored_camera.offset.pos = (2.2, 0.0, 0.6)
    egocentric_mirrored_camera.offset.rot = (0.535, 0.463, 0.463, 0.535)
    egocentric_mirrored_camera.spawn.focal_length = 18.0


@configclass
class RearArmCameraCfg:
    """Rear view of DROID with the arm and task table in frame."""

    rear_arm_camera = copy.deepcopy(
        EgocentricMirroredCameraCfg().egocentric_mirrored_camera
    )
    rear_arm_camera.prim_path = "{ENV_REGEX_NS}/rear_arm_camera"
    rear_arm_camera.offset.pos = (-1.5, 0.0, 0.65)
    rear_arm_camera.offset.rot = (0.538, 0.459, -0.459, -0.538)
    rear_arm_camera.spawn.focal_length = 18.0


def rgb_array(image):
    array = image[0].detach().cpu().numpy()
    if array.dtype != np.uint8:
        if array.max() <= 1:
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.ndim != 3 or array.shape[-1] != 3 or not array.any():
        raise RuntimeError(f"Camera returned an empty or invalid image: {array.shape}")
    return array


def main():
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    auto_register_droid_envs(
        task=args.tasks,
        cameras=[OverShoulderLeftCameraCfg, OverShoulderRightCameraCfg,
                 HeadCameraCfg, RearArmCameraCfg],
        viewport_camera=FrontArmCameraCfg,
    )

    manifest = []
    try:
        for task in args.tasks:
            matches = get_envs(task=task)
            if len(matches) != 1:
                raise RuntimeError(f"Expected one registered environment for {task}: {matches}")
            task_dir = output_dir / task
            set_output_dir(str(task_dir))
            env, cfg = create_env(matches[0], device=args.device, seed=args.seed, num_envs=1)
            try:
                env.reset()
                # USD materials can appear black on the first RTX frame. Render
                # again without advancing physics, then refresh sensor buffers.
                env.sim.forward()
                for _ in range(args.settle_renders):
                    env.sim.render()
                env.scene.update(dt=env.step_dt)
                obs = env.observation_manager.compute()
                images = {}
                camera_files = {}
                for label, group, name in CAMERAS:
                    rgb = rgb_array(obs[group][name])
                    path = task_dir / f"{label}.png"
                    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    camera_files[label] = str(path.relative_to(output_dir))
                    thumb = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
                    cv2.rectangle(thumb, (0, 0), (190, 49), (0, 0, 0), -1)
                    cv2.putText(thumb, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (255, 255, 255), 2, cv2.LINE_AA)
                    images[label] = thumb
                sheet_images = [images[label] for label in SHEET_CAMERAS]
                grid = np.concatenate((np.concatenate(sheet_images[:2], axis=1),
                                       np.concatenate(sheet_images[2:], axis=1)), axis=0)
                sheet = task_dir / "contact_sheet.png"
                cv2.imwrite(str(sheet), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                manifest.append({"task": task, "instruction": cfg.instruction,
                                 "seed": args.seed, "cameras": camera_files,
                                 "contact_sheet": str(sheet.relative_to(output_dir))})
                print(f"Saved {task}: {sheet}", flush=True)
            finally:
                env.close()
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
