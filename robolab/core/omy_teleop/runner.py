"""Dedicated OMY runner using the upstream device and RoboLab recording primitives."""

import json
import time
from datetime import datetime
from pathlib import Path

import torch
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg

import robolab.constants
from robolab.core.environments.config import parse_env_cfg
from robolab.core.environments.factory import get_envs
from robolab.core.environments.runtime import create_env

from .config import ENV_POSTFIX, configure_device, register_omy_envs, validate_environment
from .options import ARM_JOINTS, calibration_metadata, upstream_metadata


def prepare_output(path=None):
    path = Path(path) if path else Path("output/omy_teleop") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output directory must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def configure_recording(env, args, device_cfg):
    recorder = env.recorder_manager
    recorder.set_buffered_recording()
    recorder.cfg.dataset_export_mode = (
        DatasetExportMode.EXPORT_ALL
        if args.save_failures else DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    )
    recorder.set_hdf5_compression("lzf" if args.record_images else None)
    recorder.set_flush_interval(60 if args.record_images else 0)
    recorder.set_flush_memory_cleanup(False)
    # Keep failures in the same file: periodic streaming starts before the outcome
    # is known, and switching handlers on failure would lose already-flushed data.
    recorder.set_hdf5_file("omy_demos")
    metadata = {
        "schema_version": 1,
        "upstream": upstream_metadata(),
        "task": args.task,
        "action_type": "absolute_joint_position",
        "action_names": [*ARM_JOINTS, "gripper"],
        "joint_units": "rad",
        "gripper_encoding": "+1=open,-1=closed",
        "control_hz": 1 / device_cfg.dt,
        "calibration": calibration_metadata(args.calib),
        "device": {key: value for key, value in vars(device_cfg).items() if key not in ("class_type", "retargeters", "callbacks")},
        "images": "live" if args.record_images else "none",
        "save_failures": args.save_failures,
        "stale_behavior": "hold_last_command_and_automatically_recover",
    }
    recorder.set_dataset_attrs({"robolab_omy_teleop": json.dumps(metadata)})


def finish_interrupted(env, record):
    if record and not env.all_terminated:
        recorder = env.recorder_manager
        recorder.set_success_to_episodes([0], torch.zeros(1, dtype=torch.bool, device=env.device))
        recorder.export_episodes([0])


def close_recording(env):
    # Isaac Lab normally closes these in the manager destructor. Close explicitly
    # so the files are usable immediately, even if recorder/environment cycles survive.
    for name in ("_dataset_file_handler", "_failed_episode_dataset_file_handler"):
        handler = getattr(env.recorder_manager, name, None)
        if handler is not None:
            handler.close()


def operator_frame(env):
    def rgb(name):
        return env.scene[name].data.output["rgb"][0, ..., :3].to(torch.uint8).cpu().numpy()

    image = rgb("view_cam").copy()
    wrist = rgb("omy_wrist_cam")
    image[8:8 + wrist.shape[0], 8:8 + wrist.shape[1]] = wrist
    return image


@torch.inference_mode()
def run(args, app):
    # Resets and stepping must share inference mode: RoboLab stores tensors made
    # during step() and updates them in-place when beginning the next attempt.
    output = prepare_output(args.output_dir)
    robolab.constants.set_output_dir(str(output))
    # This flag also enables inexpensive proprioception recording without images.
    robolab.constants.RECORD_IMAGE_DATA = args.record
    register_omy_envs(args.task, images=args.record_images, stream=args.stream)
    names = [name for name in get_envs(task=args.task) if name.endswith(ENV_POSTFIX)]
    if len(names) != 1:
        raise ValueError(f"Expected one OMY task environment, found {names}")
    cfg = parse_env_cfg(names[0], device=args.device, num_envs=1, seed=args.seed, use_fabric=True)
    cfg.episode_length_s = args.episode_length_s
    device_cfg = configure_device(cfg, args)
    if not args.record:
        cfg.recorders = RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
    env = device = None
    active = False
    total_steps = 0
    try:
        env, cfg = create_env(cfg, policy="omy_joint_position", rendering_mode=args.rendering_mode)
        joint_ids = validate_environment(env)
        if args.record:
            configure_recording(env, args, device_cfg)
        video = None
        if args.stream:
            from omy_leader_isaaclab.video import VideoServer, encode_jpeg

            video = VideoServer(port=args.video_port)
            print(f"OMY JPEG viewer: 127.0.0.1:{args.video_port} (use upstream omy-leader-viewer)", flush=True)
        # Native device: preserve calibration, auto-zero, limiting, stale holding, and reset.
        device = create_teleop_device("omy_leader", cfg.teleop_devices.devices, {})
        print(device, flush=True)
        print(f"Task: {cfg.instruction}\nOutput: {output}\nControl target: {1 / env.step_dt:.1f} Hz", flush=True)
        print("Teleoperation starts immediately; Ctrl+C stops. Stale input holds the last target.", flush=True)
        for episode in range(args.num_episodes):
            if episode:
                env.reset_eval_state()
            seed = args.seed + episode
            env.reset(seed=seed)
            device.reset()
            if args.record:
                env.recorder_manager.set_episode_index(episode, env_ids=[0])
                env.recorder_manager.set_episode_seed(seed, env_ids=[0])
            active = True
            next_step = report_start = time.perf_counter()
            report_steps = 0
            while app.is_running() and not env.all_terminated:
                action = device.advance().to(env.device)
                if action.shape != (8,) or not torch.isfinite(action).all():
                    raise ValueError("OMY device returned an invalid eight-dimensional action")
                env.step(action.unsqueeze(0))
                total_steps += 1
                report_steps += 1
                if video is not None and total_steps % args.video_every == 0:
                    video.push(encode_jpeg(operator_frame(env)))
                now = time.perf_counter()
                if now - report_start >= 5:
                    q = env.scene["robot"].data.joint_pos[0, joint_ids]
                    error = (q - action[:7]).abs().max().item()
                    print(f"OMY: {report_steps / (now - report_start):.1f}/60 Hz; "
                          f"max joint error {error:.4f} rad; step {total_steps}", flush=True)
                    report_start, report_steps = now, 0
                if args.steps and total_steps >= args.steps:
                    finish_interrupted(env, args.record)
                    active = False
                    return 0
                if not args.no_realtime:
                    next_step += env.step_dt
                    delay = next_step - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_step = time.perf_counter()
            if env.all_terminated:
                print(f"Attempt {episode + 1}: {env.get_env_results()}", flush=True)
                active = False  # RoboLab exports on termination.
            else:
                break
        return 0
    except KeyboardInterrupt:
        print("Stopping OMY teleoperation.", flush=True)
        return 130
    finally:
        try:
            if env is not None and active:
                finish_interrupted(env, args.record)
        finally:
            try:
                if device is not None:
                    device.close()
            finally:
                if env is not None:
                    try:
                        close_recording(env)
                    finally:
                        env.close()
