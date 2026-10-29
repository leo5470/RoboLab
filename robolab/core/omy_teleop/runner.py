"""Dedicated OMY runner using the upstream device and RoboLab recording primitives."""

import json
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg

import robolab.constants
from robolab.core.environments.config import parse_env_cfg
from robolab.core.environments.factory import get_envs
from robolab.core.environments.runtime import create_env

from .config import ENV_POSTFIX, configure_device, configure_performance, register_omy_envs, validate_environment
from .diagnostics import DiagnosticFailure, RuntimeDiagnostics
from .options import ARM_JOINTS, calibration_metadata, upstream_metadata
from .observations import ObservationPassCache
from .leader_state import leader_state_metadata
from .performance import PerformanceMonitor, RenderPacer, capture_due


def prepare_output(path=None, *, task=None):
    if not path:
        root = Path("output/omy_teleop")
        root.mkdir(parents=True, exist_ok=True)
        if task:
            task_name = "".join(char if char.isalnum() or char in "_-" else "_" for char in task)
            name = f"{datetime.now():%Y%m%d_%H%M%S}_{task_name}"
        else:
            name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        candidate = root / name
        suffix = 1
        while True:
            try:
                candidate.mkdir()
            except FileExistsError:
                suffix += 1
                candidate = root / f"{name}_{suffix}"
            else:
                return candidate.resolve()
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output directory must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def configure_recording(env, args, device_cfg, xr_session=None, *, initialize=True, dataset_attrs=None):
    recorder = env.recorder_manager
    if initialize:
        recorder.set_buffered_recording()
        recorder.cfg.dataset_export_mode = (
            DatasetExportMode.EXPORT_ALL
            if args.save_failures else DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        )
        recorder.set_hdf5_compression("lzf" if args.record_images else None)
        recorder.set_flush_interval(60 if args.record_images else 0)
        recorder.set_flush_memory_cleanup(False)
    metadata = {
        "schema_version": 1,
        "upstream": upstream_metadata(),
        "task": args.task,
        "task_dirs": list(args.task_dirs),
        "task_source": args.task_resolution.metadata(),
        "action_type": "absolute_joint_position",
        "action_names": [*ARM_JOINTS, "gripper"],
        "joint_units": "rad",
        "gripper_encoding": "+1=open,-1=closed",
        "gripper_observation": {"joint": "finger_joint", "units": "normalized_closure",
                                "range": [0, 1], "raw_state_units": "rad"},
        "leader_proprioception": leader_state_metadata(),
        "control_hz": 1 / device_cfg.dt,
        "calibration": calibration_metadata(args.calib),
        "device": {key: value for key, value in vars(device_cfg).items() if key not in ("class_type", "retargeters", "callbacks")},
        "images": "live" if args.record_images else "none",
        "save_failures": args.save_failures,
        "stale_behavior": "hold_last_command_and_automatically_recover",
        "reset_behavior": "reset_mapper_and_cached_command_to_environment_home_open_gripper",
        "performance": performance_settings(env, args),
    }
    if xr_session is not None:
        metadata["viewing"] = {key: value for key, value in xr_session.metadata.items()
                               if key not in ("completed", "display_waits", "xr_display_seen")}
    # All identity fields are prepared before rotating the file. Each finalized
    # recording retains its metadata, including if a transition is interrupted.
    recorder.set_hdf5_file("omy_demos", attrs={
        "robolab_omy_teleop": json.dumps(metadata), **(dataset_attrs or {}),
    })


def performance_settings(env, args):
    import carb
    from isaaclab.sensors import ContactSensorCfg

    settings = carb.settings.get_settings()
    return {
        "profile": args.performance_profile,
        "physics_device": str(env.device),
        "cuda_connections_requested": getattr(args, "cuda_connections", None),
        "cuda_environment": getattr(args, "omy_cuda_environment", None),
        "task_checks_requested": getattr(env.cfg, "omy_task_checks", "native"),
        "task_checks_effective": getattr(env.cfg, "omy_task_checks_effective", "native"),
        "task_checks_fallback_reasons": getattr(env.cfg, "omy_task_checks_fallback_reasons", []),
        "gpu_max_num_partitions": getattr(env.cfg.sim.physx, "gpu_max_num_partitions", 8),
        "placement_checks_requested": getattr(args, "placement_checks", "independent"),
        "placement_checks_effective": "shared" if (
            getattr(env.cfg, "omy_shared_placement_checks", False)
            and getattr(env, "_omy_query_cache", False)) else "independent",
        "finger_colliders": getattr(env.cfg, "omy_finger_collider_metadata", {"profile": "original"}),
        "initial_joint_pos_rad": [env.cfg.scene.robot.init_state.joint_pos[name] for name in ARM_JOINTS],
        "physics_dt": env.cfg.sim.dt,
        "control_hz": 1 / env.step_dt,
        "decimation": env.cfg.decimation,
        "render_interval": env.cfg.sim.render_interval,
        "nominal_render_hz": 1 / (env.cfg.sim.dt * env.cfg.sim.render_interval),
        "video_every": args.video_every,
        "record_images": args.record_images,
        "optimized_observations": not args.reference_observations,
        "solver": {
            "requested_position_cap": getattr(args, "solver_position_iterations", None),
            "scene_min_position_iterations": env.cfg.sim.physx.min_position_iteration_count,
            "scene_max_position_iterations": env.cfg.sim.physx.max_position_iteration_count,
            "scene_min_velocity_iterations": env.cfg.sim.physx.min_velocity_iteration_count,
            "scene_max_velocity_iterations": env.cfg.sim.physx.max_velocity_iteration_count,
            "articulation_position_request": env.cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count,
            "articulation_velocity_request": env.cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count,
        },
        "stepping_requested": getattr(args, "stepping_profile", "ordinary"),
        "stepping_effective": getattr(env.cfg, "omy_stepping_effective", "ordinary"),
        "stepping_fallback_reasons": getattr(env.cfg, "omy_stepping_fallback_reasons", []),
        "capabilities": getattr(env.cfg, "omy_capabilities", {}),
        "dependency_plan": getattr(env.cfg, "omy_dependency_plan", {}),
        "recording_profile": getattr(args, "recording_profile", "full"),
        "episode_buffer_bytes_limit": getattr(env.cfg, "omy_episode_buffer_bytes", None),
        "episode_max_steps": getattr(env.cfg, "omy_episode_max_steps", None),
        "joint_order": list(env.scene["robot"].joint_names),
        "contact_history_lengths": {
            name: sensor.history_length for name, sensor in vars(env.cfg.scene).items()
            if isinstance(sensor, ContactSensorCfg)
        },
        "contact_sensor_profile": getattr(env.cfg, "omy_contact_sensor_profile", "full"),
        "contact_sensor_count": getattr(env.cfg, "omy_contact_sensor_count", None),
        "contact_sensor_count_full": getattr(env.cfg, "omy_contact_sensor_count_full", None),
        "kit": {key: settings.get(key) for key in (
            "/renderer/activeGpu", "/renderer/multiGpu/enabled",
            "/plugins/carb.tasking.plugin/threadCount", "/plugins/omni.tbb.globalcontrol/maxThreadCount",
            "/persistent/physics/numThreads", "/app/settings/persistent",
            "/app/runLoops/main/rateLimitEnabled", "/app/runLoops/main/rateLimitFrequency",
        )},
    }


def contact_cache_stats(env):
    if not getattr(env.cfg, "omy_batch_contacts_enabled",
                   getattr(env.cfg, "omy_contact_sensor_profile", "full") == "required"):
        return None
    from robolab.core.world.world_state import get_world

    return get_world(env)._omy_batch_stats.copy()


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
def run(args, app, *, xr_session=None):
    # Resets and stepping must share inference mode: RoboLab stores tensors made
    # during step() and updates them in-place when beginning the next attempt.
    output = prepare_output(args.output_dir, task=args.task)
    robolab.constants.set_output_dir(str(output))
    # This flag also enables inexpensive proprioception recording without images.
    robolab.constants.RECORD_IMAGE_DATA = args.record
    register_omy_envs(args.task, images=args.record_images, stream=args.stream,
                      optimized_observations=not args.reference_observations,
                      task_dirs=args.task_dirs, resolution=args.task_resolution)
    names = [name for name in get_envs(task=args.task) if name.endswith(ENV_POSTFIX)]
    if len(names) != 1:
        raise ValueError(f"Expected one OMY task environment, found {names}")
    cfg = parse_env_cfg(names[0], device=args.device, num_envs=1, seed=args.seed, use_fabric=True)
    cfg.episode_length_s = args.episode_length_s
    configure_performance(cfg, args)
    device_cfg = configure_device(cfg, args)
    if xr_session is not None:
        xr_session.configure(cfg)
        if args.desktop and args.headless:
            import carb

            carb.settings.get_settings().set_bool("/isaaclab/render/offscreen", True)
    if not args.record:
        cfg.recorders = RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
    env = device = diagnostics = observation_cache = pacer = None
    active = False
    total_steps = 0
    timing = PerformanceMonitor()
    settings = {}
    try:
        extra = {}
        if args.stepping_profile != "ordinary" or args.recording_profile != "full":
            from .environment import OmyEnv

            extra["env_class"] = OmyEnv
        env_started = time.perf_counter()
        env, cfg = create_env(cfg, policy="omy_joint_position", rendering_mode=args.rendering_mode, **extra)
        print(f"OMY environment startup: {time.perf_counter() - env_started:.3f} s", flush=True)
        joint_ids = validate_environment(env)
        if xr_session is not None:
            import carb

            carb.settings.get_settings().set_bool("/app/runLoops/main/rateLimitEnabled", False)
        settings = performance_settings(env, args)
        print(f"OMY performance configuration: {json.dumps(settings, sort_keys=True)}", flush=True)
        if not args.reference_observations:
            observation_cache = ObservationPassCache(env)
        timing.install(env)
        if xr_session is not None and args.control_hz < 60 and not args.no_realtime:
            pacer = RenderPacer(env.sim, cfg.sim.dt * cfg.sim.render_interval, monitor=timing)
        if args.diagnostics:
            diagnostics = RuntimeDiagnostics(env, output, timing,
                                             joint_tolerance=args.diagnostic_joint_tolerance)
            diagnostics.install()
            print(f"Physics-substep diagnostics: {output / 'omy_diagnostics.jsonl'}", flush=True)
        if args.record:
            configure_recording(env, args, device_cfg, xr_session)
        video = None
        if args.stream:
            from omy_leader_isaaclab.video import VideoServer, encode_jpeg

            video = VideoServer(port=args.video_port)
            print(f"OMY JPEG viewer: 127.0.0.1:{args.video_port} (use upstream omy-leader-viewer)", flush=True)
        # Upstream mapping/input/limiting with environment-aligned home/reset state.
        if xr_session is None:
            device = create_teleop_device("omy_leader", cfg.teleop_devices.devices, {})
            print(device, flush=True)
        print(f"Task: {cfg.instruction}\nOutput: {output}\nControl target: {1 / env.step_dt:.1f} Hz", flush=True)
        print("Teleoperation starts when the XR display is active; hold the calibrated READY pose."
              if xr_session is not None and not args.desktop else "Teleoperation starts immediately.", flush=True)
        print("Ctrl+C stops. Stale leader input holds the last target.", flush=True)
        for episode in range(args.num_episodes):
            if episode:
                env.reset_eval_state()
            seed = args.seed + episode
            env.reset(seed=seed)
            if xr_session is not None:
                if episode == 0:
                    xr_session.start()
                if not xr_session.wait_for_display(app, env.sim.render):
                    return 0
                if device is None:
                    device = create_teleop_device("omy_leader", cfg.teleop_devices.devices, {})
                    print(device, flush=True)
            device.reset()
            if diagnostics is not None:
                diagnostics.start_episode(episode, device)
            if args.record:
                env.recorder_manager.set_episode_index(episode, env_ids=[0])
                env.recorder_manager.set_episode_seed(seed, env_ids=[0])
            active = True
            next_step = report_start = time.perf_counter()
            timing.reset_window()
            if pacer is not None:
                pacer.reset()
            while app.is_running() and not env.all_terminated:
                if xr_session is not None and not xr_session.ready():
                    if not xr_session.wait_for_display(app, env.sim.render):
                        break
                    # Exclude the pause from throughput and pacing; do not reset
                    # the mapper (that would jump its limiter back to home).
                    next_step = report_start = time.perf_counter()
                    timing.reset_window()
                    if pacer is not None:
                        pacer.reset()
                timing.begin_step()
                renders_before = timing.calls["render"]
                with timing.measure("input"):
                    action = device.advance()
                    env.omy_input_sample = getattr(device, "last_input_sample", None)
                    if action.shape != (8,) or not torch.isfinite(action).all():
                        raise ValueError("OMY device returned an invalid eight-dimensional action")
                    action = action.to(env.device)
                if diagnostics is not None:
                    diagnostics.begin_control(action, total_steps + 1)
                try:
                    with pacer.step() if pacer is not None else nullcontext():
                        env.step(action.unsqueeze(0))
                except DiagnosticFailure:
                    diagnostics.end_control(incomplete=True)
                    timing.end_step(work_end=time.perf_counter())
                    raise
                total_steps += 1
                produced = 0
                if video is not None and capture_due(
                    env._sim_step_counter, decimation=cfg.decimation, video_every=args.video_every,
                    renders_before=renders_before, renders_after=timing.calls["render"],
                ):
                    with timing.measure("capture"):
                        image = operator_frame(env)
                    with timing.measure("jpeg"):
                        jpeg = encode_jpeg(image)
                    with timing.measure("video_enqueue"):
                        video.push(jpeg)
                    produced = 1
                if diagnostics is not None:
                    diagnostics.end_control()
                work_end = time.perf_counter()
                stop = args.steps > 0 and total_steps >= args.steps
                if not args.no_realtime and not stop and pacer is None:
                    next_step += env.step_dt
                    delay = next_step - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_step = time.perf_counter()
                timing.end_step(work_end=work_end, video_frames=produced)
                now = time.perf_counter()
                if args.stats_interval > 0 and now - report_start >= args.stats_interval:
                    q = env.scene["robot"].data.joint_pos[0, joint_ids]
                    error = (q - action[:7]).abs().max().item()
                    print(timing.report(env.step_dt), flush=True)
                    print(f"OMY: max joint error {error:.4f} rad; step {total_steps}", flush=True)
                    report_start = now
                if stop:
                    finish_interrupted(env, args.record)
                    active = False
                    if xr_session is not None:
                        xr_session.metadata["completed"] = True
                    return 0
            if env.all_terminated:
                print(f"Attempt {episode + 1}: {env.get_env_results()}", flush=True)
                active = False  # RoboLab exports on termination.
            else:
                break
        if xr_session is not None:
            xr_session.metadata["completed"] = True
        return 0
    except DiagnosticFailure as exc:
        print(f"OMY diagnostic stop: {exc}\nFirst failure saved in {output / 'omy_diagnostic_summary.json'}", flush=True)
        if env is not None:
            env.sim.pause()
        return 2
    except KeyboardInterrupt:
        print("Stopping OMY teleoperation.", flush=True)
        return 130
    finally:
        if pacer is not None:
            pacer.close()
        if diagnostics is not None:
            try:
                diagnostics.close()
            except Exception as exc:
                print(f"Unable to finalize OMY diagnostics: {exc}", flush=True)
        timing.close()
        if observation_cache is not None:
            observation_cache.close()
        try:
            if env is not None:
                print(timing.report(env.step_dt), flush=True)
                result = {
                    "settings": settings,
                    "last_window": timing.summary(env.step_dt),
                    "total_steps": timing.total_steps,
                    "total_renders": timing.total_renders,
                    "total_video_frames": timing.total_video_frames,
                    "contact_cache": contact_cache_stats(env),
                    "notes": "Host elapsed timings include synchronization; rates exclude episode resets. "
                             "Video FPS is produced JPEGs, not network reception or X11 presentation.",
                }
                try:
                    (output / "omy_performance.json").write_text(json.dumps(result, indent=2) + "\n")
                except OSError as exc:
                    print(f"Unable to save OMY performance summary: {exc}", flush=True)
            if env is not None and active:
                finish_interrupted(env, args.record)
        finally:
            try:
                if device is not None:
                    device.close()
            finally:
                try:
                    if xr_session is not None:
                        try:
                            xr_session.close(env.sim.render if env is not None else lambda: None)
                        finally:
                            (output / "xr_status.json").write_text(json.dumps(xr_session.metadata, indent=2) + "\n")
                finally:
                    if env is not None:
                        try:
                            close_recording(env)
                        finally:
                            env.close()
