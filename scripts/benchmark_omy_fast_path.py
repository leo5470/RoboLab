#!/usr/bin/env python3
"""Headless regression of production OMY profiles; never opens a leader device."""

import argparse
import hashlib
import json
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--manifest", type=Path, default=Path("output/omy_pipeline_benchmark/20261002_solver_transfer/sources.json"))
parser.add_argument("--source-name", default="fast16_native15")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--settle-commands", type=int, default=30)
args = parser.parse_args()
if args.repeats < 1 or args.settle_commands < 0:
    parser.error("repeats must be positive and settling nonnegative")
if args.output_dir.exists() and any(args.output_dir.iterdir()):
    parser.error("output directory must be new or empty")
manifest_bytes = args.manifest.read_bytes()
source = next(item for item in json.loads(manifest_bytes) if item["name"] == args.source_name)
if source["task"] != "BananaInBowlTask" or source["control_hz"] != 15 or source["gripper_encoding"] != "omy_signed":
    parser.error("the audited replay requires native-15 BananaInBowl OMY commands")
args.output_dir.mkdir(parents=True, exist_ok=True)

import cv2  # noqa: E402,F401 -- precede Isaac Lab
from isaaclab.app import AppLauncher  # noqa: E402

launcher = AppLauncher(
    headless=True, device="cuda:0", enable_cameras=False,
    kit_args="--/app/settings/persistent=false --/renderer/multiGpu/enabled=false "
             "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8 "
             "--/persistent/physics/numThreads=2",
)
app = launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robolab.constants  # noqa: E402
from robolab.core.environments.config import parse_env_cfg  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.omy_teleop.config import configure_performance, register_omy_envs  # noqa: E402
from robolab.core.omy_teleop.environment import OmyEnv  # noqa: E402
from robolab.core.omy_teleop.options import build_parser  # noqa: E402
from robolab.core.omy_teleop.runner import close_recording, configure_recording, finish_interrupted  # noqa: E402


@torch.inference_mode()
def main():
    robolab.constants.RECORD_IMAGE_DATA = True
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True
    profiles = ((32, "ordinary"), (16, "ordinary"), (16, "held-target"))
    runs, trajectories = [], {}
    for cap, stepping in profiles:
        label = f"cap{cap}_{stepping}"
        directory = args.output_dir / label
        directory.mkdir()
        robolab.constants.set_output_dir(str(directory))
        postfix = f"OMYBench{cap}{stepping.replace('-', '')}"
        register_omy_envs("BananaInBowlTask", env_postfix=postfix)
        run_args = build_parser().parse_args([
            "--source", "tcp", "--control-hz", "15", "--record", "--save-failures",
            "--recording-profile", "compact", "--solver-position-iterations", str(cap),
            "--stepping-profile", stepping, "--render-interval", "4", "--episode-length-s", "20",
        ])
        cfg = parse_env_cfg(f"BananaInBowlTask{postfix}", device="cuda:0", num_envs=1)
        cfg.episode_length_s = 20
        configure_performance(cfg, run_args)
        env, _ = create_env(cfg, env_class=OmyEnv, policy="omy_joint_position", rendering_mode="performance")
        try:
            configure_recording(env, run_args, type("DeviceCfg", (), {"dt": 1 / 15})())
            env.recorder_manager.set_dataset_attrs({"robolab_omy_collection": json.dumps({
                "schema_version": 1, "start": "saved_command_replay", "images": "deferred",
                "recording": "state_only", "training_ready": False,
            })})
            actions = torch.tensor(source["actions"], device=env.device)
            for repeat in range(args.repeats):
                env.reset(seed=0)
                current = env.scene.get_state(is_relative=True)
                for kind, assets in source["initial_state"].items():
                    for name, values in assets.items():
                        for key, value in values.items():
                            current[kind][name][key] = torch.tensor(value, device=env.device, dtype=current[kind][name][key].dtype)
                env.reset_to(current, env_ids=None, is_relative=True)
                env.recorder_manager.set_episode_index(repeat, [0])
                env.recorder_manager.set_episode_seed(0, [0])
                work, banana, terminated = [], [], False
                for step in range(len(actions) + args.settle_commands):
                    action = actions[min(step, len(actions) - 1)].unsqueeze(0)
                    start = time.perf_counter()
                    env.step(action)
                    torch.cuda.synchronize()
                    work.append(1000 * (time.perf_counter() - start))
                    state = env.scene.get_state(is_relative=True)
                    assert all(torch.isfinite(value).all() for assets in state.values()
                               for values in assets.values() for value in values.values())
                    banana.append(env.scene["banana"].data.root_pos_w[0].cpu().numpy())
                    if env.all_terminated:
                        terminated = True
                        break
                success = terminated and env.get_env_results()[0]["success"]
                finish_interrupted(env, True)
                trajectory = np.array(banana)
                trajectories[(label, repeat)] = trajectory
                np.save(directory / f"banana_{repeat}.npy", trajectory)
                result = {
                    "profile": label, "repeat": repeat, "source_commands": len(actions),
                    "allowed_settle_commands": args.settle_commands, "executed_commands": len(work),
                    "success_within_source": bool(success and len(work) <= len(actions)),
                    "success_during_settling": bool(success and len(work) > len(actions)),
                    "nonterminal_at_limit": not terminated,
                    "effective_stepping": cfg.omy_stepping_effective,
                    "work_ms_mean": float(np.mean(work)), "work_ms_p95": float(np.percentile(work, 95)),
                    "peak_banana_z_m": float(trajectory[:, 2].max()),
                }
                runs.append(result)
                print(json.dumps(result), flush=True)
        finally:
            close_recording(env)
            env.close()
    differences = []
    # Same-cap stepping comparison; solver transfer remains a separate comparison.
    for first in range(args.repeats):
        for second in range(args.repeats):
            a, b = trajectories[("cap16_ordinary", first)], trajectories[("cap16_held-target", second)]
            common = min(len(a), len(b), len(source["actions"]))
            differences.append(float(np.linalg.norm(a[:common] - b[:common], axis=1).max() * 1000))
    summary = {
        "serial_port_opened": False, "xr_enabled": False, "training_ready": False,
        "source_name": source["name"], "source_file_sha256": source["sha256"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "runs": runs, "same_cap_cross_profile_max_banana_difference_mm": differences,
        "notes": "One saved grasp regression with identical settling horizons; headless work timings include "
                 "synchronization and export. This does not validate operator collection, XR, or training transfer.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


status = 0
try:
    main()
except BaseException:
    status = 1
    raise
finally:
    app.app.post_quit(status)
    app.close()
