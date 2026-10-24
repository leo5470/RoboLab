#!/usr/bin/env python3
"""Exercise prepared starts and generate fresh diagnostic fixtures, without a leader.

These synthetic hold trajectories are explicitly marked diagnostic-only. Their
assigned outcomes exercise postprocessing selection; they are not demonstrations.
Run through isolated_gpu_launch when reserving another GPU for collection.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
bootstrap = argparse.ArgumentParser(add_help=False)
bootstrap.add_argument("--host-gpu", type=int)
host, remaining = bootstrap.parse_known_args()
if host.host_gpu is not None and "--help" not in remaining and "-h" not in remaining:
    import subprocess
    from robolab.core.omy_postprocess.gpu_isolation import isolated_gpu_launch
    from robolab.core.omy_postprocess.render import _gpu_slot
    from robolab.core.omy_teleop.collection_handoff import CollectionHandoff

    if host.host_gpu < 0:
        bootstrap.error("--host-gpu must be nonnegative")
    selection = argparse.ArgumentParser(add_help=False)
    selection.add_argument("--output-dir", type=Path, required=True)
    selected, _ = selection.parse_known_args(remaining)
    gpu = isolated_gpu_launch(f"cuda:{host.host_gpu}")
    with _gpu_slot(f"cuda:{host.host_gpu}"):
        status = subprocess.call([*gpu.prefix, sys.executable, "-B", str(Path(__file__).resolve()),
                                  *remaining, "--device", "cuda:0"], env=gpu.env)
    if status:
        raise SystemExit(status)
    root = selected.output_dir.resolve()
    results = json.loads((root / "preparation_check.json").read_text())
    if not results["all_passed"]:
        raise SystemExit(1)
    handoff = CollectionHandoff(root / "diagnostic_handoff.json")
    handoff.set_session(root)
    for task in results["tasks"]:
        handoff.add_recording(root / task["task"] / "part_000")
    handoff.publish(status=0, cleanup_complete=True)
    print(f"Fresh diagnostic handoff: {root / 'collection_runs' / (handoff.run_id + '.json')}", flush=True)
    raise SystemExit(0)

import cv2  # noqa: F401 -- must precede Isaac
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", nargs="+", default=["SimplePnp082WoodenBowl00Task", "SimplePnp034CannedMushrooms00Task"])
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--frames", type=int, default=15)
parser.add_argument("--settle-angular-rad-s", type=float, default=0.5)
parser.add_argument("--host-gpu", type=int, help="Isolate this physical GPU before starting Isaac; use this instead of --device")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1 or not 1 <= args.frames <= 1800:
    parser.error("--episodes must be positive and --frames must be in 1..1800")
args.headless = True
args.kit_args = (args.kit_args or "") + " --/app/fastShutdown=false"
app = AppLauncher(args).app


def main():
    import numpy as np
    import torch
    import h5py
    import robolab.constants
    from types import SimpleNamespace
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.runtime import create_env
    from robolab.core.omy_teleop.collection import build_parser
    from robolab.core.omy_teleop.config import configure_device, configure_performance, register_omy_envs
    from robolab.core.omy_teleop.environment import OmyEnv
    from robolab.core.omy_teleop.options import ARM_JOINTS
    from robolab.core.omy_teleop.preparation_runtime import ScenePreparation
    from robolab.core.omy_teleop.runner import configure_recording
    from robolab.core.omy_postprocess.validation import validate_recording
    from robolab.core.task.task_index import resolve_selected_task

    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"diagnostic_only": True, "tasks": [], "all_passed": False}
    (args.output_dir / "DIAGNOSTIC_ONLY.txt").write_text("Synthetic validation fixtures with assigned success labels; not human demonstrations.\n")
    try:
        for task in args.task:
            output = args.output_dir / task / "part_000"
            output.mkdir(parents=True)
            options = build_parser().parse_args(["--desktop", "--source", "tcp", "--device", args.device,
                "--control-hz", "15", "--recording-profile", "compact", "--stepping-profile", "held-target",
                "--contact-sensors", "required", "--contact-history-length", "0", "--solver-position-iterations", "16",
                "--episode-length-s", "120", "--max-attempt-steps", "1800", "--max-attempt-seconds", "120"])
            options.scene_settle_angular_rad_s = args.settle_angular_rad_s
            options.task = task
            options.task_dirs = ["single_pnp"]
            options.task_resolution = resolve_selected_task(task, options.task_dirs)
            robolab.constants.set_output_dir(str(output.resolve()))
            robolab.constants.RECORD_IMAGE_DATA = True
            register_omy_envs(task, task_dirs=options.task_dirs, resolution=options.task_resolution)
            cfg = parse_env_cfg(task + "OMYL100", device=args.device, num_envs=1)
            cfg.episode_length_s = 120
            configure_performance(cfg, options)
            device_cfg = configure_device(cfg, options)
            env = None
            row = {"task": task, "preparations": []}
            report["tasks"].append(row)
            try:
                env, cfg = create_env(cfg, policy="omy_preparation_diagnostic", rendering_mode="performance", env_class=OmyEnv)
                configure_recording(env, options, device_cfg, SimpleNamespace(metadata={}))
                recorder = env.recorder_manager
                recorder.set_flush_interval(0)
                recorder.set_dataset_attrs({"diagnostic_only": True,
                    "diagnostic_note": "Synthetic fixed-home trajectories; success labels assigned for pipeline tests."})
                action = torch.tensor([[*[cfg.scene.robot.init_state.joint_pos[n] for n in ARM_JOINTS], 1.0]], device=env.device)
                with torch.inference_mode():
                    for episode in range(args.episodes):
                        env.reset(seed=episode)
                        prepare = ScenePreparation(env, options)
                        while app.is_running():
                            result = prepare.advance()
                            if result["status"] != "settling":
                                break
                        row["preparations"].append(result)
                        assert result["status"] == "ready", result
                        initial = env.scene.get_state(is_relative=True)
                        before = {n: a.data.root_pos_w.clone() for n,a in env.scene.rigid_objects.items()}
                        rotations = {n: a.data.root_quat_w.clone() for n,a in env.scene.rigid_objects.items()}
                        recorder.set_episode_index(episode, env_ids=[0])
                        recorder.set_episode_seed(episode, env_ids=[0])
                        recorder.set_episode_preparation(result, env_ids=[0])
                        for _ in range(args.frames):
                            env.step(action)
                        after = {n: a.data.root_pos_w.clone() for n,a in env.scene.rigid_objects.items()}
                        drift = {n: float(torch.linalg.vector_norm(after[n] - before[n]).max()) for n in before}
                        assert max(drift.values(), default=0) < 0.002, drift
                        angles = {n: float(torch.rad2deg(2 * torch.acos(torch.abs(torch.sum(
                            rotations[n] * a.data.root_quat_w, dim=-1)).clamp(max=1))).max())
                                  for n,a in env.scene.rigid_objects.items()}
                        assert max(angles.values(), default=0) < 0.5, angles
                        recorder.set_success_to_episodes([0], torch.tensor([True], device=env.device))
                        recorder.export_episodes([0])
                        with h5py.File(output / "omy_demos.hdf5", "r") as file:
                            group = file[f"data/demo_{episode}"]
                            assert group["actions"].shape[0] == args.frames
                            assert json.loads(group.attrs["omy_preparation"])["status"] == "ready"
                            for name in before:
                                np.testing.assert_allclose(group[f"initial_state/rigid_object/{name}/root_pose"][0],
                                    initial["rigid_object"][name]["root_pose"][0].cpu().numpy(), rtol=0, atol=1e-7)
                        print(f"Prepared {task} episode {episode}: {result['elapsed_s']:.3f}s; drift {drift}", flush=True)
                recorder._dataset_file_handler.close()
                validation = validate_recording(output / "omy_demos.hdf5")
                assert not validation["errors"] and all(e["structural_valid"] and e["native_15_hz"] for e in validation["episodes"])
                row["passed"] = True
            finally:
                if env is not None:
                    env.close()
        report["all_passed"] = True
    finally:
        (args.output_dir / "preparation_check.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()
