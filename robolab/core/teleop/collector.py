"""Shared relative-IK demo collector. Importing this module does not launch Isaac Sim."""

import argparse
import cProfile
import json
import os
import sys
import time
import traceback
from datetime import datetime

import torch

from robolab.constants import PACKAGE_DIR
from robolab.core.teleop.performance import LoopTiming, RealtimePacer
from robolab.core.teleop.views import inset_resolution


def build_parser(backend="keyboard", app_launcher=None):
    if app_launcher is None:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher
    parser = argparse.ArgumentParser(
        description=f"Collect {backend}-controlled RoboLab demonstrations.", allow_abbrev=False
    )
    parser.add_argument("--task", default="BananaInBowlTask", help="RoboLab task class name.")
    parser.add_argument(
        "--teleop-profile",
        choices=("wifi", "standard"),
        default="wifi",
        help="wifi (default): smaller fixed stream, deferred images for remote sessions, "
        "and responsive input/pacing. standard: original stream/image defaults.",
    )
    parser.add_argument("--stream-width", type=int, default=None, help="WebRTC viewport width (wifi: 960).")
    parser.add_argument("--stream-height", type=int, default=None, help="WebRTC viewport height (wifi: 540).")
    parser.add_argument(
        "--view-layout",
        choices=("single", "inset"),
        default="single",
        help="inset: overview plus a small wrist view inside the same stream. "
        "Press B to hide/show it. single (default): lowest rendering cost.",
    )
    parser.add_argument(
        "--inset-camera",
        choices=("wrist", "top"),
        default="wrist",
        help="Camera for --view-layout inset (default: wrist, follows the gripper).",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cap interactive control at the task's real-time rate; never catch up with stale actions.",
    )
    parser.add_argument(
        "--contact-history",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep unused contact-force history buffers (wifi: disabled). Current contact forces remain.",
    )
    parser.add_argument(
        "--torch-threads", type=int, default=1, help="CPU tensor threads for one teleoperated environment (default: 1)."
    )
    parser.add_argument(
        "--kit-threads",
        type=int,
        default=None,
        help="Kit/TBB worker limit (wifi + CPU: up to 16; otherwise unchanged). 0 keeps Kit's existing setting.",
    )
    parser.add_argument(
        "--physics-threads",
        type=int,
        default=None,
        help="CPU PhysX workers (wifi + CPU: up to 2; otherwise unchanged). "
        "0 keeps Kit's existing setting. Does not change solver iterations.",
    )
    parser.add_argument(
        "--num-demos",
        "--num-episodes",
        dest="num_demos",
        type=int,
        default=1,
        help="Number of successful demonstrations to collect.",
    )
    parser.add_argument(
        "--output-dir",
        help=f"Output directory. Defaults to a timestamped directory under output/{backend}_demos/.",
    )
    parser.add_argument("--filename", default="keyboard_demos.hdf5", help="HDF5 filename.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the first attempt.")
    parser.add_argument("--pos-sensitivity", type=float, default=0.02, help="Translation command magnitude in meters.")
    parser.add_argument("--rot-sensitivity", type=float, default=0.05, help="Rotation command magnitude in radians.")
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=None,
        help="Flush recorder tensors to HDF5 every N control steps; 0 disables streaming and writes "
        "each attempt once, at the end (default: 25 with images, 0 with --defer-images).",
    )
    parser.add_argument(
        "--hdf5-compression",
        choices=("lzf", "gzip", "none"),
        default="none",
        help="HDF5 compression codec. Raw/uncompressed is the low-latency default.",
    )
    parser.add_argument(
        "--camera-scale",
        type=float,
        default=0.5,
        help="Scale policy-camera width and height (default: 0.5, producing 640x360).",
    )
    parser.add_argument(
        "--render-interval",
        type=int,
        default=None,
        help="Render every N physics sub-steps (default: 8 = once per action). Larger values "
        "reduce visual feedback and may delay Kit input delivery; avoid them for live-image recording.",
    )
    parser.add_argument(
        "--record-viewport-camera",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record a third-person sensor image in addition to the WebRTC viewport.",
    )
    parser.add_argument(
        "--flush-memory-cleanup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force garbage collection and CUDA cache clearing after every HDF5 flush.",
    )
    parser.add_argument(
        "--record-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record the environment observation dictionary, including camera images (default: enabled).",
    )
    parser.add_argument(
        "--defer-images",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Collect state-only demonstrations without policy-camera rendering. Enabled by default "
        "for the wifi livestream profile. Actions, per-step scene "
        "states, seeds, and the env config are recorded; reconstruct the camera observations "
        "afterwards with scripts/materialize_demo_images.py.",
    )
    parser.add_argument(
        "--save-failures",
        action="store_true",
        help="Save failed, timed-out, and manually retried attempts in a separate HDF5 file.",
    )
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable demo recording during hardware commissioning with --no-record.",
    )
    if backend == "so101":
        from robolab.core.teleop.so101_options import add_arguments

        add_arguments(parser)
    app_launcher.add_app_launcher_args(parser)
    parser.set_defaults(rendering_mode="performance")
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=0,
        help="Run N neutral-action steps without keyboard input, report timing, then exit. "
        "Use a separate output directory; benchmark attempts are discarded.",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=20,
        help="Exclude the first N benchmark steps from timing (default: 20).",
    )
    parser.add_argument(
        "--benchmark-realtime",
        action="store_true",
        help="Pace the benchmark at the task control rate and count missed step budgets; "
        "the default benchmark runs unpaced to measure capacity.",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Save a cProfile trace of benchmark steps to the output directory."
    )
    parser.set_defaults(backend=backend, filename=f"{backend}_demos.hdf5")
    return parser


def configure_args(parser, args_cli):
    if args_cli.num_demos < 1:
        parser.error("--num-demos must be positive")
    if args_cli.pos_sensitivity <= 0 or args_cli.rot_sensitivity <= 0:
        parser.error("--pos-sensitivity and --rot-sensitivity must be positive")
    if args_cli.flush_interval is not None and args_cli.flush_interval < 0:
        parser.error("--flush-interval must be zero (write once per attempt) or positive")
    if not 0 < args_cli.camera_scale <= 1:
        parser.error("--camera-scale must be greater than 0 and at most 1")
    if args_cli.render_interval is not None and args_cli.render_interval < 1:
        parser.error("--render-interval must be positive")
    if args_cli.benchmark_steps < 0 or args_cli.benchmark_warmup < 0:
        parser.error("benchmark step counts must be non-negative")
    if args_cli.benchmark_steps and args_cli.save_failures:
        parser.error("benchmarks discard attempts; do not combine with --save-failures")
    if args_cli.profile and not args_cli.benchmark_steps:
        parser.error("--profile requires --benchmark-steps")
    if args_cli.benchmark_realtime and not args_cli.benchmark_steps:
        parser.error("--benchmark-realtime requires --benchmark-steps")
    if args_cli.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    if any(count is not None and count < 0 for count in (args_cli.kit_threads, args_cli.physics_threads)):
        parser.error("--kit-threads and --physics-threads must be non-negative")
    for dimension in (args_cli.stream_width, args_cli.stream_height):
        if dimension is not None and (dimension < 64 or dimension % 2):
            parser.error("stream dimensions must be even integers of at least 64 pixels")

    livestream = args_cli.livestream if args_cli.livestream >= 0 else int(os.environ.get("LIVESTREAM", 0))
    wifi = args_cli.teleop_profile == "wifi"
    from robolab.core.teleop.performance import worker_settings

    worker_overrides = worker_settings(
        args_cli.teleop_profile, args_cli.device, args_cli.kit_threads, args_cli.physics_threads
    )
    worker_args = " ".join(f"--{key}={value}" for key, value in worker_overrides.items())
    args_cli.kit_args = f"{worker_args} {args_cli.kit_args or ''}".strip()
    if args_cli.defer_images is None:
        args_cli.defer_images = wifi and livestream > 0
    if args_cli.contact_history is None:
        args_cli.contact_history = not wifi
    if args_cli.stream_width is None:
        args_cli.stream_width = 960 if wifi and livestream else 1280
    if args_cli.stream_height is None:
        args_cli.stream_height = 540 if wifi and livestream else 720
    if args_cli.view_layout == "inset":
        from robolab.core.teleop.views import inset_resolution

        try:
            inset_resolution(args_cli.stream_width, args_cli.stream_height)
        except ValueError as exc:
            parser.error(str(exc))
        if args_cli.headless and not livestream:
            parser.error("--view-layout inset requires --livestream or a local GUI (omit --headless)")

    if livestream:
        # Configure capture and (below) env_cfg.viewer.resolution. The actual Kit
        # viewport texture is also pinned after environment creation.
        stream_settings = {
            "/app/window/width": args_cli.stream_width,
            "/app/window/height": args_cli.stream_height,
            "/app/renderer/resolution/width": args_cli.stream_width,
            "/app/renderer/resolution/height": args_cli.stream_height,
        }
        if wifi or args_cli.view_layout == "inset":
            # Spend the small stream on the operator views rather than editor
            # panels. Mouse camera navigation and teleop input remain available.
            stream_settings["/app/window/hideUi"] = "true"
        if wifi:
            stream_settings.update(
                {
                    "/app/livestream/allowResize": "false",
                    "/app/livestream/allowDynamicResize": "false",
                    "/app/livestream/sendFrameBlocking": "false",
                    "/app/livestream/asyncCopy": "true",
                    "/app/livestream/maxPushStreamDataAttempts": 1,
                }
            )
        stream_args = " ".join(f"--{key}={value}" for key, value in stream_settings.items())
        # Explicit Kit overrides remain last, for advanced users.
        args_cli.kit_args = f"{stream_args} {args_cli.kit_args or ''}".strip()

    # Isaac Kit may otherwise use every visible GPU for rendering. A busy second
    # GPU can then stall a collector whose simulation device is explicitly cuda:0.
    single_gpu_kit_arg = "--/renderer/multiGpu/enabled=false"
    if single_gpu_kit_arg not in (getattr(args_cli, "kit_args", "") or ""):
        args_cli.kit_args = f"{getattr(args_cli, 'kit_args', '') or ''} {single_gpu_kit_arg}".strip()

    if not args_cli.record and args_cli.save_failures:
        parser.error("--save-failures requires recording")
    if args_cli.backend == "so101":
        from robolab.core.teleop.so101_options import validate_arguments

        validate_arguments(parser, args_cli)
    return livestream, wifi


def _output_dir(args_cli) -> str:
    if args_cli.output_dir:
        return os.path.abspath(args_cli.output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(PACKAGE_DIR, "output", f"{args_cli.backend}_demos", args_cli.task, timestamp)


def _available_filename(output_dir: str, requested: str) -> str:
    """Return a filename that will not overwrite an existing recording."""
    candidate = requested
    stem, suffix = os.path.splitext(requested)
    index = 1
    while os.path.exists(os.path.join(output_dir, candidate)):
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    return candidate


def _mark_attempt_failed(env) -> None:
    """Finalize a manually retried attempt as failed (or discard it in success-only mode)."""
    env.recorder_manager.set_success_to_episodes([0], torch.zeros(1, dtype=torch.bool, device=env.device))
    env.recorder_manager.export_episodes(env_ids=[0])


def _deferred_metadata(env_cfg, env_name: str, args_cli) -> dict:
    """Describe a state-only recording so its images can be rebuilt later.

    Stamped on the HDF5 ``data`` group next to the recording's ``env_cfg.json``.
    It carries what the sidecar cannot: which environment registration produced
    the recording (the deferred config has no camera entries to restore from),
    and the control timing the actions were issued at.
    """
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    return {
        "schema_version": 1,
        "collector": f"examples/collect_{args_cli.backend}_demos.py",
        "task": args_cli.task,
        "env_name": env_name,
        "env_postfix": "RelIK",
        "policy_cameras": False,
        "viewport_camera": bool(args_cli.record_viewport_camera),
        "observations_recorded": bool(args_cli.record_images),
        "instruction": env_cfg.instruction,
        "base_seed": args_cli.seed,
        "sim_dt": env_cfg.sim.dt,
        "decimation": env_cfg.decimation,
        "render_interval": env_cfg.sim.render_interval,
        "control_dt": control_dt,
        "control_hz": 1.0 / control_dt,
        "episode_length_s": env_cfg.episode_length_s,
    }


def _apply_render_interval(env_cfg, args_cli) -> str:
    """Set the viewport render cadence and describe it.

    ``sim.render_interval`` counts physics sub-steps, so it is compared against
    ``decimation`` (sub-steps per control step) to say how often the operator's
    view refreshes. Changing this trades visual feedback against render work.
    With policy
    cameras present, a render that does not happen leaves the sensor holding its
    previous frame, and the recorded observation repeats it.
    """
    if args_cli.render_interval is not None:
        env_cfg.sim.render_interval = args_cli.render_interval

    interval = env_cfg.sim.render_interval
    decimation = env_cfg.decimation
    control_hz = 1.0 / (env_cfg.sim.dt * decimation)
    per_control_step = interval / decimation
    return (
        f"every {interval} sim steps = every {per_control_step:g} control step(s), "
        f"{control_hz / per_control_step:.1f} Hz"
    )


def _benchmark(env, output_dir, args_cli, simulation_app):
    """Exercise the real recording loop without saving a synthetic demonstration."""
    recorder = env.recorder_manager
    original_export = recorder.export_episodes

    def discard_export(env_ids=None):
        ids = [0] if env_ids is None else env_ids
        recorder.set_success_to_episodes(ids, torch.zeros(len(ids), dtype=torch.bool, device=env.device))
        return original_export(env_ids=ids)

    # Even a task that succeeds with neutral actions must not leave synthetic
    # demonstrations in a success dataset. Keep real streaming writes enabled
    # during the benchmark so their latency is measured.
    recorder.export_episodes = discard_export
    action = torch.zeros((1, 7), device=env.device)
    profiler = cProfile.Profile() if args_cli.profile else None
    timing = LoopTiming(capacity=max(1, args_cli.benchmark_steps))
    measured_start = None
    measured_seconds = 0.0
    total = args_cli.benchmark_steps + args_cli.benchmark_warmup
    try:
        env.reset(seed=args_cli.seed)
        timing.install(env.sim)
        pacer = RealtimePacer(env.step_dt if args_cli.benchmark_realtime else 0)
        with torch.no_grad():
            for step in range(total):
                if env.all_terminated or not simulation_app.is_running():
                    break
                if step == args_cli.benchmark_warmup:
                    timing.clear()
                    measured_start = time.perf_counter()
                    if profiler is not None:
                        profiler.enable()
                pacer.wait()
                start = time.perf_counter()
                env.step(action)
                if "cuda" in env.device:
                    torch.cuda.synchronize(env.device)
                elapsed = time.perf_counter() - start
                if step >= args_cli.benchmark_warmup:
                    timing.record_step(elapsed)
                    measured_seconds = time.perf_counter() - measured_start
                if (step + 1) % 50 == 0:
                    print(f"Benchmark progress: {step + 1}/{total} steps", flush=True)
    finally:
        timing.close()
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(os.path.join(output_dir, "benchmark.prof"))
        try:
            _mark_attempt_failed(env)
        finally:
            recorder.export_episodes = original_export
    if not timing.steps:
        raise RuntimeError("Benchmark ended before any measured steps")
    result = {
        "steps": len(timing.steps),
        "steps_per_second": len(timing.steps) / sum(timing.steps),
        "wall_steps_per_second": len(timing.steps) / measured_seconds,
        "benchmark_realtime": args_cli.benchmark_realtime,
        **timing.summary(env.step_dt),
        "defer_images": args_cli.defer_images,
        "render_interval": env.cfg.sim.render_interval,
        "control_dt": env.step_dt,
        "device": env.device,
        "teleop_profile": args_cli.teleop_profile,
        "viewport_resolution": list(env.cfg.viewer.resolution),
        "view_layout": args_cli.view_layout,
        "inset_camera": args_cli.inset_camera if args_cli.view_layout == "inset" else None,
        "inset_resolution": list(inset_resolution(*env.cfg.viewer.resolution))
        if args_cli.view_layout == "inset"
        else None,
    }
    with open(os.path.join(output_dir, "benchmark.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("Benchmark: " + json.dumps(result), flush=True)
    return 0


def collect(args_cli, simulation_app) -> int:
    import torch
    from isaaclab.devices import Se3KeyboardCfg
    from isaaclab.managers import DatasetExportMode

    import robolab.constants
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env, end_episode
    from robolab.core.replay.materialize import scale_camera_resolutions
    from robolab.core.teleop.keyboard import ResponsiveSe3Keyboard
    from robolab.core.teleop.lifecycle import AttemptControls
    from robolab.core.teleop.performance import LoopTiming, RealtimePacer, StreamQuality
    from robolab.core.teleop.views import OperatorInset, pin_viewport_resolution
    from robolab.registrations.droid.auto_env_registrations_rel_ik import (
        auto_register_droid_rel_ik_envs,
    )

    livestream = args_cli.livestream if args_cli.livestream >= 0 else int(os.environ.get("LIVESTREAM", 0))
    wifi = args_cli.teleop_profile == "wifi"
    torch.set_num_threads(args_cli.torch_threads)
    output_dir = _output_dir(args_cli)
    robolab.constants.set_output_dir(output_dir)
    robolab.constants.RECORD_IMAGE_DATA = args_cli.record_images
    robolab.constants.VERBOSE = False

    auto_register_droid_rel_ik_envs(
        task=args_cli.task,
        env_postfix="RelIK",
        include_viewport_camera=args_cli.record_viewport_camera,
        include_policy_cameras=not args_cli.defer_images,
    )
    env_names = get_envs(task=args_cli.task)
    if not env_names:
        raise ValueError(f"No relative-IK environment found for task '{args_cli.task}'")

    env = None
    keyboard = None
    timing = None
    stream_quality = None
    operator_inset = None
    leader = None
    active_attempt = False
    try:
        env_cfg = parse_env_cfg(
            env_names[0],
            device=args_cli.device,
            seed=args_cli.seed,
            num_envs=1,
            use_fabric=True,
        )
        camera_resolutions = scale_camera_resolutions(env_cfg, args_cli.camera_scale)
        if livestream:
            env_cfg.viewer.resolution = (args_cli.stream_width, args_cli.stream_height)
        if not args_cli.contact_history:
            from isaaclab.sensors import ContactSensorCfg

            # RoboLab predicates read current force_matrix_w, never force history.
            # history_length > 0 forces all 11 sensors to update at every physics
            # substep even with lazy_sensor_update=True. Keep current forces lazy.
            for sensor in vars(env_cfg.scene).values():
                if isinstance(sensor, ContactSensorCfg) and not sensor.track_air_time:
                    sensor.history_length = 0
        if not args_cli.record:
            from isaaclab.managers import RecorderManagerBaseCfg

            env_cfg.recorders = RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
        elif args_cli.backend == "so101" and not args_cli.benchmark_steps:
            from robolab.core.teleop.recorders import LeaderDiagnosticsRecorderCfg

            env_cfg.recorders.record_so101 = LeaderDiagnosticsRecorderCfg()
        render_summary = _apply_render_interval(env_cfg, args_cli)
        env, env_cfg = create_env(
            env_cfg,
            policy=f"{args_cli.backend}_rel_ik",
            rendering_mode=args_cli.rendering_mode,
        )
        import carb.settings

        settings = carb.settings.get_settings()
        actual_workers = {
            "kit": settings.get("/plugins/carb.tasking.plugin/threadCount"),
            "tbb": settings.get("/plugins/omni.tbb.globalcontrol/maxThreadCount"),
            "physics": settings.get("/persistent/physics/numThreads"),
            "torch": torch.get_num_threads(),
        }
        print(f"CPU workers: {actual_workers}; save Kit preferences={settings.get('/app/settings/persistent')}")
        print(
            "Kit frame limiter: "
            f"enabled={settings.get('/app/runLoops/main/rateLimitEnabled')}, "
            f"frequency={settings.get('/app/runLoops/main/rateLimitFrequency')}"
        )
        if wifi:
            # SimulationContext may enable this during creation. Control-loop
            # pacing below accounts for physics AND render time, with input
            # draining during slack, rather than sleeping inside Kit updates.
            settings.set_bool("/app/runLoops/main/rateLimitEnabled", False)

        if livestream:
            actual_resolution = pin_viewport_resolution(env_cfg.viewer.resolution)
            print(f"Operator viewport render texture: {actual_resolution}", flush=True)
        if args_cli.view_layout == "inset":
            from robolab.robots.droid import WristCameraCfg

            # Deferred collection strips the scene sensor; keep its calibration
            # for a private operator camera without re-enabling image recording.
            wrist_cfg = getattr(env_cfg.scene, "wrist_cam", None) or WristCameraCfg().wrist_cam
            operator_inset = OperatorInset(
                *env_cfg.viewer.resolution,
                target=env_cfg.viewer.lookat,
                camera=args_cli.inset_camera,
                wrist_camera_cfg=wrist_cfg,
                env_prim_path=env.scene.env_prim_paths[0],
            )
            print(
                f"Operator views: overview + {args_cli.inset_camera} {operator_inset.resolution}; "
                f"one {tuple(env_cfg.viewer.resolution)} stream. B hides/shows the inset.",
                flush=True,
            )

        recorder = env.recorder_manager
        filename = _available_filename(output_dir, args_cli.filename)
        flush_interval = 0
        if args_cli.record:
            recorder.set_buffered_recording()
            recorder.cfg.dataset_export_mode = (
                DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
                if args_cli.save_failures
                else DatasetExportMode.EXPORT_SUCCEEDED_ONLY
            )
            recorder.set_hdf5_compression(args_cli.hdf5_compression)
            recorder.set_hdf5_file(filename)
            # Image batches must remain bounded even with amortized tensor appends.
            # State-only attempts fit in memory and avoid disk stalls until export.
            default_flush_interval = 0 if args_cli.defer_images else 25
            flush_interval = default_flush_interval if args_cli.flush_interval is None else args_cli.flush_interval
            recorder.set_flush_interval(flush_interval)
            recorder.set_flush_memory_cleanup(args_cli.flush_memory_cleanup)
            recorder.set_dataset_attrs(
                {
                    f"robolab_{args_cli.backend}_teleop": json.dumps(
                        {
                            "profile": args_cli.teleop_profile,
                            "device": env.device,
                            "viewport_resolution": list(env_cfg.viewer.resolution),
                            "view_layout": args_cli.view_layout,
                            "inset_camera": args_cli.inset_camera if operator_inset else None,
                            "inset_resolution": list(operator_inset.resolution) if operator_inset else None,
                            "contact_history": args_cli.contact_history,
                            "realtime": args_cli.realtime,
                            "pos_sensitivity": args_cli.pos_sensitivity,
                            "rot_sensitivity": args_cli.rot_sensitivity,
                            "workers": actual_workers,
                        }
                    )
                }
            )
            if args_cli.defer_images:
                recorder.set_dataset_attrs(
                    {"robolab_deferred_images": json.dumps(_deferred_metadata(env_cfg, env_names[0], args_cli))}
                )

        if livestream:
            stream_quality = StreamQuality()
        if args_cli.benchmark_steps:
            return _benchmark(env, output_dir, args_cli, simulation_app)

        timing = LoopTiming()
        timing.install(env.sim)
        if args_cli.backend == "so101":
            from robolab.core.teleop.so101_runtime import SO101Teleop

            leader = SO101Teleop(env, args_cli, operator_inset)
            controls = leader.controls
            print("SO-101 local controls: N start/resume; H clutch; R reject; B inset; Esc exit.")
            print(f"Control input: {args_cli.control_source}; arm input: local USB/IPC.")
            poll_input = leader.poll
        else:
            keyboard = ResponsiveSe3Keyboard(
                Se3KeyboardCfg(
                    sim_device=env.device,
                    pos_sensitivity=args_cli.pos_sensitivity,
                    rot_sensitivity=args_cli.rot_sensitivity,
                )
            )
            controls = AttemptControls()
            keyboard.add_callback("N", controls.request_start)
            keyboard.add_callback("R", controls.request_retry)
            if operator_inset is not None:
                keyboard.add_callback("B", operator_inset.toggle)
            poll_input = keyboard.poll
            print(keyboard)
            print("N: start; R: reject/reset; L: clear held motion; B: toggle inset.")
        print(f"Teleop profile: {args_cli.teleop_profile}; stream {args_cli.stream_width}x{args_cli.stream_height}")
        print(f"Physics device: {env.device}; interactive target {1 / env.step_dt:.1f} steps/s")
        if wifi and "cuda" in env.device:
            print("For one environment, also benchmark --device cpu; the viewport still uses the GPU.")
        print(f"\nTask: {env_cfg.instruction}")
        print(
            f"Recording: {os.path.join(output_dir, filename)}"
            if args_cli.record
            else "Commissioning only: no demonstration files will be written."
        )
        if args_cli.record and args_cli.defer_images:
            print(
                "Recording mode: state-only (--defer-images). Actions, per-step scene states, "
                "seeds, and proprioception are recorded; policy-camera rendering is deferred. "
                "The live viewport still renders."
            )
            print(
                "Materialize the images afterwards with:\n"
                f"  python scripts/materialize_demo_images.py --input {os.path.join(output_dir, filename)} "
                f"--output <final>/{filename}"
            )
        elif args_cli.record:
            print("Recording mode: images recorded live during teleoperation")
        print(f"Observation recording: {'enabled' if args_cli.record and args_cli.record_images else 'disabled'}")
        print(f"HDF5 compression: {args_cli.hdf5_compression}")
        print(f"HDF5 flush: {f'every {flush_interval} steps' if flush_interval else 'once per attempt'}")
        camera_summary = (
            ", ".join(f"{name}={width}x{height}" for name, (width, height) in camera_resolutions.items())
            or "none (state-only collection)"
        )
        print(f"Camera sensors: {camera_summary}")
        print(f"Viewport render: {render_summary}")
        print(f"Third-person observation camera: {'enabled' if args_cli.record_viewport_camera else 'disabled'}")
        if camera_resolutions and env_cfg.sim.render_interval > env_cfg.decimation:
            print(
                "\033[93mNOTE: a camera sensor is configured but the scene renders less often "
                "than once per control step, so recorded images repeat the previous render "
                "instead of showing the step they are stored against.\033[0m"
            )
        if args_cli.defer_images and args_cli.record_viewport_camera:
            print(
                "\033[93mNOTE: --record-viewport-camera creates a sensor that renders every "
                "control step, so it gives back part of the latency --defer-images buys.\033[0m"
            )

        attempts = 0
        successes = 0
        while simulation_app.is_running() and (
            successes < args_cli.num_demos or (leader is not None and args_cli.smoke_steps)
        ):
            if attempts:
                env.reset_eval_state()
            attempt_seed = args_cli.seed + attempts
            env.reset(seed=attempt_seed)
            if leader is not None:
                leader.reset()
            else:
                keyboard.reset()
            controls.arm()
            location = "server control device" if leader is not None else "streamed viewport"
            print(
                f"\nAttempt {attempts + 1}; collected {successes}/{args_cli.num_demos}. "
                f"READY — physics is paused. Press N on the {location} to start.",
                flush=True,
            )
            idle_pacer = RealtimePacer(1 / 30)
            next_ready_notice = time.perf_counter() + 10
            while simulation_app.is_running() and not controls.running:
                idle_pacer.wait(poll_input)
                poll_input()  # Also drain controls when rendering leaves no pacing slack.
                env.sim.render()
                if getattr(controls, "stop_requested", False):
                    return 1
                if leader is not None and args_cli.smoke_steps:
                    controls.request_start()
                if controls.start_requested:
                    if leader is not None:
                        leader.try_begin()
                    else:
                        keyboard.reset()
                        controls.begin()
                if not controls.running and time.perf_counter() >= next_ready_notice:
                    print(f"Attempt {attempts + 1}: READY, waiting for N on the {location}.", flush=True)
                    next_ready_notice = time.perf_counter() + 10
            if not simulation_app.is_running():
                break
            if args_cli.record:
                recorder.set_episode_index(attempts if args_cli.save_failures else successes, env_ids=[0])
                recorder.set_episode_seed(attempt_seed, env_ids=[0])
            active_attempt = True
            print(
                f"Attempt {attempts + 1}: {'RECORDING' if args_cli.record else 'COMMISSIONING'}. R rejects this attempt.",
                flush=True,
            )
            manually_retried = False
            rate_window_start = time.perf_counter()
            rate_window_steps = 0
            attempt_start = rate_window_start
            attempt_steps = 0
            timing.clear()
            pacer = RealtimePacer(env.step_dt if args_cli.realtime else 0)
            while simulation_app.is_running() and not env.all_terminated:
                pacer.wait(poll_input)
                poll_input()  # Serial/render overruns must not starve local keys.
                if getattr(controls, "stop_requested", False):
                    return 1
                if controls.retry_requested:
                    manually_retried = True
                    if args_cli.record:
                        _mark_attempt_failed(env)
                    active_attempt = False
                    print("Attempt rejected; resetting the scene.")
                    break
                if leader is not None:
                    action = leader.advance()
                    if action is None:
                        env.sim.render()
                        continue
                    if leader.controller.resumed:
                        pacer = RealtimePacer(env.step_dt)
                        rate_window_start = time.perf_counter()
                        rate_window_steps = 0
                        timing.clear()
                else:
                    command = keyboard.advance()
                    # A callback may have arrived in advance()'s extra input poll.
                    if controls.retry_requested:
                        continue
                    action = command.clone()
                    action[6] = (1.0 - command[6]) * 0.5
                step_start = time.perf_counter()
                with torch.no_grad():
                    env.step(action.unsqueeze(0))
                timing.record_step(time.perf_counter() - step_start)
                rate_window_steps += 1
                attempt_steps += 1
                if leader is not None:
                    leader.step_completed()
                    if args_cli.smoke_steps and leader.total_steps >= args_cli.smoke_steps:
                        print(
                            f"SO-101 smoke complete: {leader.total_steps} actions; no demonstrations saved.", flush=True
                        )
                        return 0
                if time.perf_counter() - rate_window_start >= 5:
                    elapsed = time.perf_counter() - rate_window_start
                    stats = timing.summary(env.step_dt)
                    print(
                        f"Control rate: {rate_window_steps / elapsed:.1f}/{1 / env.step_dt:.1f} steps/s; "
                        f"step p50/p95 {stats['step_ms_p50']:.0f}/{stats['step_ms_p95']:.0f} ms; "
                        f"physics {stats['physics_ms_mean']:.0f} ms; "
                        f"render {stats['render_ms_mean']:.0f} ms; "
                        f"IK/record/other {stats['other_ms_mean']:.0f} ms; "
                        f"over budget {stats['over_budget_steps']}/{len(timing.steps)}"
                        + (f"; {stream_quality.summary()}" if stream_quality else ""),
                        flush=True,
                    )
                    rate_window_start = time.perf_counter()
                    rate_window_steps = 0
                    timing.clear()
            if not simulation_app.is_running():
                break
            if attempt_steps:
                print(
                    f"Attempt {attempts + 1}: {attempt_steps} actions over "
                    f"{time.perf_counter() - attempt_start:.1f}s wall time (including pauses)."
                )
            if not manually_retried and env.all_terminated:
                active_attempt = False
                result = env.get_env_results()[0]
                succeeded = bool(result["success"])
                label = "SUCCESS" if succeeded else "FAILED/TIMED OUT"
                print(f"Attempt {attempts + 1}: {label} at step {result['step']}")
                if succeeded:
                    successes += 1
            attempts += 1
        print(f"\nFinished: {successes} successful demos from {attempts} attempts.")
        print(f"Output directory: {output_dir}")
        return 0 if successes == args_cli.num_demos else 1
    except BaseException as exc:
        if leader is not None:
            try:
                leader.event("interrupted", reason=str(exc), paused=leader.controls.paused)
            except Exception:
                pass  # Keep the original failure when event logging itself fails.
        raise
    finally:
        from contextlib import ExitStack

        # Run every cleanup even if another cleanup raises.
        with ExitStack() as cleanup:
            if env is not None:
                cleanup.callback(env.close)
                if args_cli.record:
                    cleanup.callback(end_episode, env)
                    if active_attempt:
                        cleanup.callback(_mark_attempt_failed, env)
            for resource in (operator_inset, timing, stream_quality, keyboard, leader):
                if resource is not None:
                    cleanup.callback(resource.close)


def launch(backend="keyboard", argv=None):
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
    import cv2  # noqa: F401  # Must precede Isaac Lab imports.
    from isaaclab.app import AppLauncher

    parser = build_parser(backend, AppLauncher)
    args_cli = parser.parse_args(argv)
    configure_args(parser, args_cli)
    args_cli.enable_cameras = True
    # AppLauncher consumes/pops entries from the supplied Namespace.
    app = AppLauncher(argparse.Namespace(**vars(args_cli))).app
    try:
        return collect(args_cli, app)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"{backend} demo collection failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        app.close()
