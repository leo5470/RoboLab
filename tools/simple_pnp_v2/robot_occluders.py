#!/usr/bin/env python3
"""Record the robot's link boxes at its frozen start pose (Isaac + GPU).

    .venv/bin/python tools/simple_pnp_v2/run_pinned.py cuda:0 tools/simple_pnp_v2/robot_occluders.py \
        --task SimplePnpV2230MilkjugA0100Task \
        --output output/simple_pnp_v2_validation/inputs/robot_occluders.json

Creates the OMY environment (OmyDroidCfg at the DROID reset pose), resets it,
and for every robot rigid body records an oriented box: exact local geometry
bounds (geometry points, the body's own transform excluded) placed at the
simulated body pose. The generator ray-tests handle points against these boxes
so the robot's start pose (e.g. the gripper) is part of the occlusion screen.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2  # noqa: F401,E402
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True)
parser.add_argument("--task-dirs", nargs="+", default=["single_pnp_v2_drafts"])
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app


def main():
    import numpy as np
    import torch
    from pxr import Usd, UsdPhysics
    import robolab.constants
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.runtime import create_env
    from robolab.core.omy_teleop.collection import build_parser
    from robolab.core.omy_teleop.config import configure_device, configure_performance, register_omy_envs
    from robolab.core.omy_teleop.environment import OmyEnv
    from robolab.core.scenes.placement import prim_local_bounds
    from robolab.core.task.task_index import resolve_selected_task
    import omni.usd

    robolab.constants.VERBOSE = False
    options = build_parser().parse_args([
        "--desktop", "--source", "tcp", "--device", args.device, "--control-hz", "15",
        "--recording-profile", "compact", "--stepping-profile", "held-target",
        "--contact-sensors", "required", "--contact-history-length", "0",
        "--solver-position-iterations", "16", "--scene-settle-clearance-m", "0"])
    options.task, options.task_dirs = args.task, args.task_dirs
    options.task_resolution = resolve_selected_task(args.task, options.task_dirs)
    robolab.constants.set_output_dir(str((args.output.parent / "isaac_robot_occluders").resolve()))
    register_omy_envs(args.task, task_dirs=options.task_dirs, resolution=options.task_resolution)
    cfg = parse_env_cfg(args.task + "OMYL100", device=args.device, num_envs=1)
    configure_performance(cfg, options)
    configure_device(cfg, options)
    env, cfg = create_env(cfg, policy="simple_pnp_v2_robot_occluders", rendering_mode="performance", env_class=OmyEnv)
    with torch.inference_mode():
        env.reset(seed=0)
        robot = env.scene["robot"]
        pos = robot.data.body_pos_w[0].cpu().numpy(); quat = robot.data.body_quat_w[0].cpu().numpy()
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(robot.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0"))
    if not root:
        raise RuntimeError(f"robot prim not found: {robot.cfg.prim_path}")
    print("robot root", root.GetPath(), flush=True)
    bodies = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        # Names repeat in this asset; prefer a rigid body, else the first prim of that name.
        if prim.GetName() in robot.body_names and (prim.GetName() not in bodies or prim.HasAPI(UsdPhysics.RigidBodyAPI)
                                                   and not bodies[prim.GetName()].HasAPI(UsdPhysics.RigidBodyAPI)):
            bodies[prim.GetName()] = prim
    boxes, missing = [], []
    for i, name in enumerate(robot.body_names):
        prim = bodies.get(name)
        if prim is None:
            missing.append(name)
            continue
        try:
            lower, upper = prim_local_bounds(prim)
        except ValueError:
            continue  # bodies without geometry (frames)
        boxes.append({"body": name, "prim": str(prim.GetPath()), "position": pos[i].tolist(), "quat_wxyz": quat[i].tolist(),
                      "local_lower": lower.tolist(), "local_upper": upper.tolist()})
    out = {"task": args.task, "joint_pos": robot.data.joint_pos[0].cpu().tolist(), "joint_names": robot.joint_names,
           "robot_usd": str(robot.cfg.spawn.usd_path), "boxes": boxes, "bodies_without_prim": missing,
           "note": "body poses from PhysX after reset (DROID reset pose); local bounds from geometry points"}
    args.output.write_text(json.dumps(out, indent=1))
    print("wrote", args.output, len(boxes), "boxes", hashlib.sha256(args.output.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
