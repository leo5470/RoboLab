#!/usr/bin/env python3
"""Diagnostic only: observe a Simple-PnP scene's start-up physics in the OMY environment.

Run on an isolated GPU from the repository root, for example:
    .venv/bin/python tools/simple_pnp_v2/run_isolated.py cuda:0 tools/simple_pnp_v2/start_probe.py \
        --task SimplePnp230MilkjugA0100Task --seeds 0 1 2 0 --clearances 0.002 \
        --output output/simple_pnp_v2_validation/probes/reset_determinism

    ... --clearances 0.002 0.00001 --ground-z -0.700 --output .../ground_fix

The probe uses the OMY collection profile (OmyDroidCfg, 15 Hz, held-target
stepping, required contact sensors, 16 solver position iterations). For each
task it creates one environment. Then, for every (clearance, seed) case in
order, it resets, runs ScenePreparation with that clearance, and holds the
robot at home for --hold-s. Every rigid body (including the table) is recorded
per control step. analyze_start_probe.py summarises and compares the results.

--ground-z writes a candidate copy of the task's scene with GroundPlane moved,
inside <output>/candidates/simple_pnp/. ScenePreparation only runs its flat-table
checks for parent directories named exactly "simple_pnp". Asset paths are
absolutised, and the original scene is not modified. PreparationSettings
rejects a clearance of 0; use a tiny positive value (e.g. 1e-5) for contact.
This writes diagnostics, not demonstrations or qualification evidence.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2  # noqa: F401,E402 -- must precede Isaac
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", nargs="+", required=True)
parser.add_argument("--task-dirs", nargs="+", default=["single_pnp"])
parser.add_argument("--seeds", type=int, nargs="+", default=[0])
parser.add_argument("--clearances", type=float, nargs="+", default=[0.002])
parser.add_argument("--hold-s", type=float, default=5.5)
parser.add_argument("--ground-z", type=float, help="Run a candidate scene copy with GroundPlane at this Z")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.kit_args = (args.kit_args or "") + " --/app/fastShutdown=false"
app = AppLauncher(args).app


def candidate_scene(original: Path, ground_z: float) -> Path:
    from pxr import Sdf, UsdUtils
    folder = args.output / "candidates" / "simple_pnp"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / original.name.replace(".usda", f"_ground{abs(ground_z) * 1000:.0f}mm.usda")
    layer = Sdf.Layer.FindOrOpen(str(original))
    new = Sdf.Layer.CreateNew(str(path)); new.TransferContent(layer)

    def absolute(p):
        if not p or os.path.isabs(p) or "://" in p or p.startswith("omniverse:"):
            return p
        return os.path.normpath(os.path.join(original.parent, p))
    UsdUtils.ModifyAssetPaths(new, absolute)
    attr = new.GetAttributeAtPath("/world/GroundPlane.xformOp:translate")
    if attr is None:
        raise ValueError(f"{original}: /world/GroundPlane.xformOp:translate not found")
    x, y, _ = attr.default
    attr.default = (x, y, ground_z)
    new.Save()
    return path


def main():
    import numpy as np
    import torch
    import robolab.constants
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.runtime import create_env
    from robolab.core.omy_teleop.collection import build_parser
    from robolab.core.omy_teleop.config import configure_device, configure_performance, register_omy_envs
    from robolab.core.omy_teleop.environment import OmyEnv
    from robolab.core.omy_teleop.options import ARM_JOINTS
    from robolab.core.omy_teleop.preparation_runtime import ScenePreparation
    from robolab.core.scenes.placement import tilt_change_deg
    from robolab.core.task.task_index import resolve_selected_task

    robolab.constants.VERBOSE = False
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {"argv": sys.argv, "device": args.device, "seeds": args.seeds, "clearances": args.clearances,
            "hold_s": args.hold_s, "ground_z": args.ground_z,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (args.output / "meta.json").write_text(json.dumps(meta, indent=1))
    for task in args.task:
        options = build_parser().parse_args([
            "--desktop", "--source", "tcp", "--device", args.device, "--control-hz", "15",
            "--recording-profile", "compact", "--stepping-profile", "held-target",
            "--contact-sensors", "required", "--contact-history-length", "0",
            "--solver-position-iterations", "16", "--episode-length-s", "120",
            "--max-attempt-steps", "1800", "--max-attempt-seconds", "120"])
        options.task = task
        options.task_dirs = args.task_dirs
        options.task_resolution = resolve_selected_task(task, options.task_dirs)
        robolab.constants.set_output_dir(str((args.output / task).resolve()))
        register_omy_envs(task, task_dirs=options.task_dirs, resolution=options.task_resolution)
        cfg = parse_env_cfg(task + "OMYL100", device=args.device, num_envs=1)
        cfg.episode_length_s = 120
        row = {"task": task, "scene": str(cfg.scene.scene.spawn.usd_path), "cases": []}
        if args.ground_z is not None:
            cfg.scene.scene.spawn.usd_path = str(candidate_scene(Path(cfg.scene.scene.spawn.usd_path), args.ground_z))
            row["candidate_scene"] = cfg.scene.scene.spawn.usd_path
        configure_performance(cfg, options)
        configure_device(cfg, options)
        env = None
        try:
            env, cfg = create_env(cfg, policy="start_probe_diagnostic", rendering_mode="performance", env_class=OmyEnv)
            names = list(env.scene.rigid_objects)
            action = torch.tensor([[*[cfg.scene.robot.init_state.joint_pos[n] for n in ARM_JOINTS], 1.0]], device=env.device)
            with torch.inference_mode():
                for clearance in args.clearances:
                    for seed in args.seeds:
                        started = time.time()
                        options.scene_settle_clearance_m = clearance
                        env.reset(seed=seed)
                        prepare = ScenePreparation(env, options)
                        table_top = float(prepare._bounds("table")[:, 2].max()) if "table" in names else 0.0

                        def sample(phase, start=None):
                            s = {"phase": phase}
                            for n in names:
                                a = env.scene.rigid_objects[n]
                                pose = a.data.root_pose_w[0].cpu().numpy()
                                rec = {"bottom_gap_mm": (float(prepare._bounds(n)[:, 2].min()) - table_top) * 1000,
                                       "lin": float(torch.linalg.vector_norm(a.data.root_lin_vel_w[0])),
                                       "ang": float(torch.linalg.vector_norm(a.data.root_ang_vel_w[0])),
                                       "pose": pose.tolist()}
                                if start is not None:
                                    rec["disp_mm"] = float(np.linalg.norm(pose[:3] - np.array(start[n][:3]))) * 1000
                                    rec["tilt_deg"] = float(tilt_change_deg(np.array(start[n][3:]), pose[3:]))
                                s[n] = rec
                            return s

                        trace = [sample("after_placement")]
                        start = {n: trace[0][n]["pose"] for n in names}
                        while app.is_running():
                            result = prepare.advance()
                            trace.append(sample("prep", start))
                            if result["status"] != "settling":
                                break
                        for _ in range(math.ceil(args.hold_s / env.step_dt)):
                            env.step(action)
                            trace.append(sample("hold", start))
                        row["cases"].append({"clearance_m": clearance, "seed": seed, "status": result["status"],
                                             "elapsed_s": result["elapsed_s"], "reasons": result.get("reasons"),
                                             "wall_s": time.time() - started, "trace": trace})
                        print(f"{task} clearance={clearance} seed={seed} prep={result['status']} {result['elapsed_s']:.3f}s", flush=True)
        except Exception:
            row["error"] = traceback.format_exc()
            traceback.print_exc()
        finally:
            # Persist before teardown: Isaac can crash at shutdown (exit 139).
            (args.output / f"{task}.json").write_text(json.dumps(row))
            if env is not None:
                env.close()
    (args.output / "done").write_text("ok\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
