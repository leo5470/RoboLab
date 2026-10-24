#!/usr/bin/env python3
"""Replay saved OMY commands to compare gripper speed and solver iterations.

Never opens a leader or XR session. Each invocation uses a fresh simulator.
Results are diagnostics, not training demonstrations or an XR latency benchmark.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np


def load_source(path, episodes):
    with h5py.File(path, "r") as file:
        metadata = json.loads(file["data"].attrs["robolab_omy_teleop"])
        if metadata["control_hz"] != 15 or metadata["gripper_encoding"] != "+1=open,-1=closed":
            raise ValueError("Replay requires native 15 Hz signed OMY commands")
        sources = {}
        for name in episodes:
            group = file["data"][name]
            initial = {}
            for kind, assets in group["initial_state"].items():
                initial[kind] = {
                    asset: {key: dataset[:] for key, dataset in values.items()}
                    for asset, values in assets.items()
                }
            sources[name] = {"initial": initial, "actions": group["actions"][:]}
    return metadata, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", default=["demo_1", "demo_2"])
    parser.add_argument("--iterations", type=int, choices=(16, 32, 64), required=True)
    parser.add_argument("--legacy-speed", action="store_true",
                        help="Reproduce the old ignored velocity_limit field for a baseline.")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    metadata, sources = load_source(args.source, args.episodes)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()

    import cv2  # noqa: F401 -- import before Isaac Lab
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        headless=True, device="cuda:0", enable_cameras=False,
        kit_args="--/app/settings/persistent=false --/renderer/multiGpu/enabled=false "
                 "--/plugins/carb.tasking.plugin/threadCount=8 "
                 "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8 "
                 "--/persistent/physics/numThreads=2",
    )
    app = launcher.app
    env = None
    status = 1
    try:
        import torch
        from isaaclab.managers import DatasetExportMode
        import robolab.constants
        from robolab.core.environments.config import parse_env_cfg
        from robolab.core.environments.runtime import create_env
        from robolab.core.omy_teleop.config import configure_performance, register_omy_envs
        from robolab.core.omy_teleop.environment import OmyEnv
        from robolab.core.omy_teleop.options import build_parser
        from robolab.core.omy_teleop.runner import close_recording, finish_interrupted
        from robolab.robots.droid import DroidCfg

        robolab.constants.RECORD_IMAGE_DATA = True
        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True
        robolab.constants.set_output_dir(str(args.output_dir.resolve()))
        task = metadata["task"]
        task_dirs = metadata["task_dirs"]
        postfix = "OMYGraspReplay"
        register_omy_envs(task, env_postfix=postfix, task_dirs=task_dirs,
                         expected_source=metadata.get("task_source"))
        run_args = build_parser().parse_args([
            "--source", "tcp", "--control-hz", "15", "--record", "--save-failures",
            "--recording-profile", "compact", "--solver-position-iterations", str(args.iterations),
            "--stepping-profile", "held-target", "--render-interval", "4",
            "--contact-sensors", "required", "--contact-history-length", "0",
            "--episode-length-s", "120",
        ])
        cfg = parse_env_cfg(f"{task}{postfix}", device="cuda:0", num_envs=1)
        cfg.episode_length_s = 120
        configure_performance(cfg, run_args)
        gripper = cfg.scene.robot.actuators["gripper"]
        assert gripper.velocity_limit_sim == 5.0
        assert DroidCfg().robot.actuators["gripper"].velocity_limit_sim is None
        if args.legacy_speed:
            gripper.velocity_limit = 5.0
            gripper.velocity_limit_sim = None
        env, _ = create_env(cfg, env_class=OmyEnv, policy="omy_joint_position", rendering_mode="performance")
        recorder = env.recorder_manager
        recorder.set_buffered_recording()
        recorder.cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        recorder.set_hdf5_compression(None)
        recorder.set_flush_interval(0)
        recorder.set_flush_memory_cleanup(False)
        recorder.set_hdf5_file("replay_diagnostics")
        recorder.set_dataset_attrs({"training_ready": False, "source_sha256": source_hash})
        robot = env.scene["robot"]
        names = list(robot.joint_names)
        finger = names.index("finger_joint")
        coupled = [names.index(name) for name in [
            "right_outer_knuckle_joint", "left_inner_finger_joint", "right_inner_finger_joint",
            "left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint",
        ]]
        signs = np.array([1, -1, 1, -1, -1])
        limit = float(robot.data.joint_vel_limits[0, finger].item())
        if not args.legacy_speed:
            assert abs(limit - 5.0) < 1e-6, limit
        runs = []
        summary = {
            "source": str(args.source.resolve()), "source_sha256": source_hash,
            "task": task, "iterations": args.iterations, "legacy_speed": args.legacy_speed,
            "effective_finger_velocity_limit_rad_s": limit, "joint_names": names,
            "serial_port_opened": False, "xr_enabled": False,
            "stepping": cfg.omy_stepping_effective, "runs": runs,
            "notes": "Identical saved initial states and commands. Stop at task termination or source end. "
                     "Substep diagnostic reads add overhead; reported step work subtracts those reads. "
                     "Headless timings exclude XR/rendering and are not live collection latency. "
                     "Linked-joint error is not a measurement of collider penetration.",
        }
        original_step = env.sim.step
        capture = False
        sub_q, sub_v, read_seconds = [], [], 0.0

        def sampled_step(*step_args, **step_kwargs):
            nonlocal read_seconds
            result = original_step(*step_args, **step_kwargs)
            if capture:
                started = time.perf_counter()
                # Direct PhysX reads avoid modifying scene/sensor cache cadence.
                sub_q.append(robot.root_physx_view.get_dof_positions()[0].cpu().numpy().copy())
                sub_v.append(robot.root_physx_view.get_dof_velocities()[0].cpu().numpy().copy())
                read_seconds += time.perf_counter() - started
            return result

        env.sim.step = sampled_step
        with torch.inference_mode():
            for repeat in range(args.repeats):
                for episode, source in sources.items():
                    env.reset(seed=0)
                    current = env.scene.get_state(is_relative=True)
                    for kind, assets in source["initial"].items():
                        for name, values in assets.items():
                            for key, value in values.items():
                                current[kind][name][key] = torch.as_tensor(
                                    value, device=env.device, dtype=current[kind][name][key].dtype)
                    env.reset_to(current, env_ids=None, is_relative=True)
                    recorder.set_episode_index(len(runs), [0])
                    recorder.set_episode_seed(0, [0])
                    sub_q, sub_v, read_seconds = [], [], 0.0
                    work, objects, commands = [], [], []
                    actions = torch.as_tensor(source["actions"], device=env.device)
                    capture = True
                    for action in actions:
                        overhead = read_seconds
                        started = time.perf_counter()
                        env.step(action.unsqueeze(0))
                        torch.cuda.synchronize()
                        work.append(1000 * (time.perf_counter() - started - (read_seconds - overhead)))
                        commands.append(action.cpu().numpy().copy())
                        objects.append(env.scene["milkjug_a02"].data.root_pose_w[0].cpu().numpy().copy())
                        if env.all_terminated:
                            break
                    capture = False
                    succeeded = bool(env.all_terminated and env.get_env_results()[0]["success"])
                    finish_interrupted(env, True)
                    q, v = np.array(sub_q), np.array(sub_v)
                    assert np.isfinite(q).all() and np.isfinite(v).all()
                    assert len(q) == len(commands) * cfg.decimation
                    closed = np.repeat(np.array(commands)[:, -1] < 0, cfg.decimation)
                    errors = np.rad2deg(abs(q[:, coupled] - q[:, finger, None] * signs)).max(1)
                    output = f"{episode}_repeat{repeat}"
                    np.savez_compressed(
                        args.output_dir / f"{output}.npz", joint_position=q, joint_velocity=v,
                        actions=commands, object_pose=objects, step_work_ms=work,
                    )
                    result = {
                        "episode": episode, "repeat": repeat, "success": succeeded,
                        "executed_commands": len(commands), "source_commands": len(actions),
                        "max_finger_speed_rad_s": float(abs(v[:, finger]).max()),
                        "max_closed_mimic_error_deg": float(errors[closed].max()) if closed.any() else None,
                        "p95_closed_mimic_error_deg": float(np.percentile(errors[closed], 95)) if closed.any() else None,
                        "closed_fraction_error_above_3deg": float(np.mean(errors[closed] > 3)) if closed.any() else None,
                        "mean_step_work_ms": float(np.mean(work)),
                        "p95_step_work_ms": float(np.percentile(work, 95)),
                        "diagnostic_read_seconds": read_seconds,
                        "trajectory": f"{output}.npz",
                    }
                    runs.append(result)
                    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                    print("GRASP_RESULT " + json.dumps(result), flush=True)
        env.sim.step = original_step
        close_recording(env)
        status = 0
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.app.post_quit(status)
        app.close()


if __name__ == "__main__":
    main()
