"""Space-key episode control for the dedicated L100/XR collector."""

import math
import time
from contextlib import nullcontext

import numpy as np

from .options import ARM_JOINTS


class SpaceReset:
    """Kit-window Space presses, consumed once by the simulation thread."""

    def __init__(self):
        import carb.input
        import omni.appwindow

        self._input = carb.input.acquire_input_interface()
        window = omni.appwindow.get_default_app_window()
        if window is None or window.get_keyboard() is None:
            raise RuntimeError("Space reset requires a Kit window with keyboard input")
        self._keyboard = window.get_keyboard()
        self._pressed = False
        self._pending = 0
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def _on_event(self, event, *unused):
        import carb.input

        if event.input.name in ("SPACE", "SPACEBAR"):
            if event.type == carb.input.KeyboardEventType.KEY_PRESS and not self._pressed:
                self._pending += 1
                self._pressed = True
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self._pressed = False
        return True

    def take(self):
        if self._pending:
            self._pending -= 1
            return True
        return False

    def clear(self):
        self._pending = 0

    def close(self):
        if self._sub is not None:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._sub)
            self._sub = None


class ReadyGate:
    """Compare fresh raw L100 samples against the fixed calibrated home."""

    def __init__(self, device, initial_q, *, tolerance_rad=0.12, movement_rad=0.035,
                 stable_s=0.3, clock=time.monotonic):
        self.device = device
        self.initial_q = np.asarray(initial_q, dtype=float)
        self.tolerance_rad = tolerance_rad
        self.movement_rad = movement_rad
        self.stable_s = stable_s
        self.clock = clock
        self._near_since = None
        self._near_vector = None
        self._anchor = None
        self.last_age_s = None

    def sample(self):
        reading, age = self.device._latest()
        self.last_age_s = age
        if reading is None or not math.isfinite(age) or age > self.device.cfg.stale_s:
            self._near_since = None
            return None
        return reading

    def _near_home(self, reading):
        target = self.device._mapper.raw_target(reading)
        arm_ok = bool(np.max(np.abs(target - self.initial_q)) <= self.tolerance_rad)
        grip_ok = self.device._mapper.gripper_command(reading["gripper.pos"]) > 0
        return arm_ok and grip_ok

    def observe_waiting(self):
        reading = self.sample()
        if reading is None or not self._near_home(reading):
            self._near_since = None
            self._near_vector = None
            return False
        now = self.clock()
        vector = self._vector(reading)
        if self._near_since is None or self._near_vector is None or bool(
            np.max(np.abs(vector - self._near_vector)) >= self.movement_rad
        ):
            self._near_since = now
            self._near_vector = vector
        return now - self._near_since >= self.stable_s

    def confirm(self):
        reading = self.sample()
        if reading is None or not self._near_home(reading) or self._near_since is None:
            return False
        if self.clock() - self._near_since < self.stable_s:
            return False
        self._anchor = self._vector(reading)
        return True

    @staticmethod
    def _vector(reading):
        return np.asarray([reading[f"joint_{i}.pos"] for i in range(1, 7)]
                          + [reading["gripper.pos"]], dtype=float)

    def moved(self):
        reading = self.sample()
        if reading is None or self._anchor is None:
            return False
        # Physical leader movement, including the trigger, starts the attempt.
        return bool(np.max(np.abs(self._vector(reading) - self._anchor)) >= self.movement_rad)


def build_parser():
    from .xr import build_parser as xr_parser

    parser = xr_parser()
    parser.description = "Collect state-only L100/XR demonstrations; Space resets or confirms READY."
    parser.set_defaults(record=True, save_failures=True, num_episodes=1000)
    parser.add_argument("--ready-tolerance-rad", type=float, default=0.12)
    parser.add_argument("--start-motion-rad", type=float, default=0.035)
    parser.add_argument("--ready-stable-s", type=float, default=0.3)
    parser.add_argument("--max-attempt-steps", type=int, default=1200,
                        help="Maximum recorded control steps per attempt; bounds live recorder memory.")
    parser.add_argument("--max-attempt-seconds", type=float, default=20.0,
                        help="Maximum simulated seconds per attempt (default: 20, independent of control rate).")
    return parser


def validate_args(parser, args):
    from .xr import validate_args as validate_xr

    validate_xr(parser, args)
    if args.auto_zero or not args.calib:
        parser.error("collection needs the fixed READY calibration; --auto-zero is unsupported")
    if args.stream or args.record_images or not args.record or not args.save_failures:
        parser.error("collection requires state-only recording with failed attempts retained")
    if args.headless:
        parser.error("Space reset requires the Isaac Sim window; omit --headless")
    if args.max_attempt_steps < 1:
        parser.error("--max-attempt-steps must be positive")
    if not math.isfinite(args.max_attempt_seconds) or args.max_attempt_seconds <= 0:
        parser.error("--max-attempt-seconds must be finite and positive")
    for key in ("ready_tolerance_rad", "start_motion_rad", "ready_stable_s"):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")


def run(args, app, *, xr_session):
    """Collect attempts with deliberate Space reset/confirm transitions."""
    import json
    from pathlib import Path

    import torch
    from isaaclab.devices.teleop_device_factory import create_teleop_device

    import robolab.constants
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env

    from .config import (ENV_POSTFIX, configure_device, configure_performance,
                         register_omy_envs, validate_environment)
    from .observations import ObservationPassCache
    from .environment import OmyEnv
    from .performance import PerformanceMonitor, RenderPacer
    from .runner import (close_recording, configure_recording, contact_cache_stats, finish_interrupted,
                         performance_settings, prepare_output)

    @torch.inference_mode()
    def collect():
        output = prepare_output(args.output_dir)
        robolab.constants.set_output_dir(str(output))
        # RoboLab's historical flag enables policy-observation recorder terms;
        # the registered OMY variant has no image observations or cameras.
        robolab.constants.RECORD_IMAGE_DATA = True
        register_omy_envs(args.task, images=False, stream=False,
                          optimized_observations=not args.reference_observations,
                          task_dirs=args.task_dirs, resolution=args.task_resolution)
        names = [name for name in get_envs(task=args.task) if name.endswith(ENV_POSTFIX)]
        if len(names) != 1:
            raise ValueError(f"Expected one OMY task environment, found {names}")
        cfg = parse_env_cfg(names[0], device=args.device, num_envs=1, seed=args.seed, use_fabric=True)
        cfg.episode_length_s = args.episode_length_s
        configure_performance(cfg, args)
        device_cfg = configure_device(cfg, args)
        xr_session.configure(cfg)
        env = device = keyboard = cache = pacer = None
        timing = PerformanceMonitor(capacity=min(args.max_attempt_steps,
                                                 math.ceil(args.max_attempt_seconds * args.control_hz)))
        active = False
        total_steps = 0
        attempt_steps = 0
        attempt_timing = []
        attempt = 0
        settings = {}
        try:
            env_started = time.perf_counter()
            env, cfg = create_env(cfg, policy="omy_joint_position", rendering_mode=args.rendering_mode,
                                  env_class=OmyEnv)
            print(f"OMY environment startup: {time.perf_counter() - env_started:.3f} s", flush=True)
            joint_ids = validate_environment(env)
            import carb

            carb.settings.get_settings().set_bool("/app/runLoops/main/rateLimitEnabled", False)
            settings = performance_settings(env, args)
            print(f"OMY performance configuration: {json.dumps(settings, sort_keys=True)}", flush=True)
            if not args.reference_observations:
                cache = ObservationPassCache(env)
            timing.install(env)
            if args.control_hz < 60 and not args.no_realtime:
                pacer = RenderPacer(env.sim, cfg.sim.dt * cfg.sim.render_interval, monitor=timing)
            configure_recording(env, args, device_cfg, xr_session)
            recorder = env.recorder_manager
            # Keep HDF5 writes at episode boundaries, outside the live loop.
            # max_attempt_steps bounds the in-memory state-only buffer.
            recorder.set_flush_interval(0)
            recorder.set_dataset_attrs({"robolab_omy_collection": json.dumps({
                "schema_version": 1, "start": "move_after_space_confirmed_ready",
                "reset_key": "SPACE", "ready_tolerance_rad": args.ready_tolerance_rad,
                "start_motion_rad": args.start_motion_rad, "ready_stable_s": args.ready_stable_s,
                "images": "deferred", "recording": "state_only",
                "max_attempt_steps": args.max_attempt_steps,
                "max_attempt_seconds": args.max_attempt_seconds,
            })})
            keyboard = SpaceReset()
            xr_session.start()
            env.reset(seed=args.seed)
            if not xr_session.wait_for_display(app, env.sim.render):
                return 0
            device = create_teleop_device("omy_leader", cfg.teleop_devices.devices, {})
            device.reset()
            home = np.asarray([cfg.scene.robot.init_state.joint_pos[name] for name in ARM_JOINTS])
            gate = ReadyGate(device, home, tolerance_rad=args.ready_tolerance_rad,
                             movement_rad=args.start_motion_rad, stable_s=args.ready_stable_s)
            state = "waiting"
            print(f"Task: {cfg.instruction}\nOutput: {output}", flush=True)
            print("Return L100 to READY, then press Space in the Isaac Sim window. "
                  "Move L100 to start; press Space while recording to reset. Ctrl+C exits.", flush=True)
            events_path = Path(output) / "collection_events.jsonl"

            def event(kind, **fields):
                with events_path.open("a") as stream:
                    stream.write(json.dumps({"event": kind, "attempt": attempt,
                                             "wall_time_s": time.time(), "total_steps": total_steps,
                                             **fields}) + "\n")

            def save_timing():
                if not attempt_timing:
                    return None
                path = output / f"attempt_{attempt:04d}_timing.npz"
                columns = tuple(zip(*attempt_timing))
                np.savez(path, action_wall_ns=np.asarray(columns[0], dtype=np.int64),
                         step_done_wall_ns=np.asarray(columns[1], dtype=np.int64),
                         sim_before_s=np.asarray(columns[2], dtype=np.float64),
                         sim_after_s=np.asarray(columns[3], dtype=np.float64),
                         leader_age_s=np.asarray(columns[4], dtype=np.float64),
                         source_wall_ns=np.asarray(columns[5], dtype=np.int64),
                         source_read_monotonic_ns=np.asarray(columns[6], dtype=np.int64))
                (output / f"attempt_{attempt:04d}_performance.json").write_text(json.dumps({
                    "active_control": timing.summary(env.step_dt),
                    "export": getattr(recorder, "export_timings", []),
                    "stepping_effective": getattr(cfg, "omy_stepping_effective", "ordinary"),
                    "fallback_reasons": getattr(cfg, "omy_stepping_fallback_reasons", []),
                }, indent=2) + "\n")
                attempt_timing.clear()
                return path.name

            def reset_scene(reason):
                nonlocal active, state, attempt, attempt_steps
                if active:
                    finish_interrupted(env, True)
                    active = False
                    event("attempt_end", result=reason, timing_file=save_timing())
                    attempt += 1
                if attempt >= args.num_episodes:
                    return False
                env.reset_eval_state()
                env.reset(seed=args.seed)  # repeat the same object arrangement
                device.reset()
                attempt_steps = 0
                attempt_timing.clear()
                keyboard.clear()
                if pacer is not None:
                    pacer.reset()
                state = "waiting"
                gate._near_since = None
                gate._near_vector = None
                print("Scene reset. Return L100 to READY, then press Space.", flush=True)
                return True

            report_start = time.perf_counter()
            next_step = report_start
            while app.is_running() and attempt < args.num_episodes:
                if not xr_session.ready():
                    if active:
                        if not reset_scene("xr_lost"):
                            break
                    elif state == "armed":
                        state = "waiting"
                        gate._near_since = None
                        gate._near_vector = None
                    if not xr_session.wait_for_display(app, env.sim.render):
                        break
                    keyboard.clear()
                    continue

                # Pump Kit's keyboard and XR display in the idle states; physics
                # and recorder terms remain untouched until movement starts.
                if state != "recording":
                    env.sim.render()
                    if state == "waiting":
                        aligned = gate.observe_waiting()
                        if keyboard.take():
                            if aligned and gate.confirm():
                                device.reset()
                                state = "armed"
                                event("armed")
                                print("READY confirmed. Move L100 to start recording.", flush=True)
                            else:
                                print("L100 is not steady at calibrated READY; adjust and press Space again.", flush=True)
                    elif keyboard.take():
                        state = "waiting"
                        gate._near_since = None
                        gate._near_vector = None
                        print("Arming cancelled. Press Space at READY.", flush=True)
                    elif gate.moved():
                        recorder.set_episode_index(attempt, env_ids=[0])
                        recorder.set_episode_seed(args.seed, env_ids=[0])
                        active = True
                        attempt_steps = 0
                        attempt_timing.clear()
                        state = "recording"
                        timing.reset_window()
                        if pacer is not None:
                            pacer.reset()
                        next_step = report_start = time.perf_counter()
                        event("attempt_start")
                        print(f"Attempt {attempt + 1} recording.", flush=True)
                    else:
                        time.sleep(0.005)
                    continue

                if keyboard.take():
                    if not reset_scene("manual_reset"):
                        break
                    continue
                reading = gate.sample()
                if reading is None:
                    if not reset_scene("leader_stale"):
                        break
                    continue
                timing.begin_step()
                action_wall_ns = time.monotonic_ns()
                sim_before_s = env._sim_step_counter * cfg.sim.dt
                with timing.measure("input"):
                    action = device.advance()
                    sample = getattr(device, "last_input_sample", None)
                    if sample is None:
                        raise RuntimeError("OMY source did not capture the sample consumed by advance()")
                    if sample["stale"]:
                        timing.end_step(work_end=time.perf_counter(), admitted=False)
                        if not reset_scene("leader_stale"):
                            break
                        continue
                    env.omy_input_sample = sample
                    if action.shape != (8,) or not torch.isfinite(action).all():
                        raise ValueError("OMY device returned an invalid eight-dimensional action")
                    action = action.to(env.device)
                with pacer.step() if pacer is not None else nullcontext():
                    env.step(action.unsqueeze(0))
                attempt_timing.append((action_wall_ns, time.monotonic_ns(), sim_before_s,
                                       env._sim_step_counter * cfg.sim.dt,
                                       sample["age_s"], sample["source_wall_ns"], sample["read_monotonic_ns"]))
                total_steps += 1
                attempt_steps += 1
                work_end = time.perf_counter()
                if not args.no_realtime and pacer is None:
                    next_step += env.step_dt
                    delay = next_step - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_step = time.perf_counter()
                timing.end_step(work_end=work_end)
                if time.perf_counter() - report_start >= args.stats_interval:
                    q = env.scene["robot"].data.joint_pos[0, joint_ids]
                    error = (q - action[:7]).abs().max().item()
                    print(timing.report(env.step_dt), flush=True)
                    print(f"OMY: max joint error {error:.4f} rad; step {total_steps}", flush=True)
                    report_start = time.perf_counter()
                if env.all_terminated:
                    results = env.get_env_results()
                    active = False  # environment already exported on termination
                    event("attempt_end", result=results, timing_file=save_timing())
                    attempt += 1
                    print(f"Attempt {attempt}: {results}", flush=True)
                    if not reset_scene("terminal"):
                        break
                elif attempt_steps >= args.max_attempt_steps:
                    if not reset_scene("attempt_step_limit"):
                        break
                elif attempt_steps * env.step_dt >= args.max_attempt_seconds:
                    if not reset_scene("attempt_time_limit"):
                        break
                if args.steps and total_steps >= args.steps:
                    break
            xr_session.metadata["completed"] = True
            return 0
        except KeyboardInterrupt:
            print("Stopping OMY collection.", flush=True)
            return 130
        finally:
            if pacer is not None:
                pacer.close()
            timing.close()
            if cache is not None:
                cache.close()
            if env is not None:
                print(timing.report(env.step_dt), flush=True)
                try:
                    (output / "omy_performance.json").write_text(json.dumps({
                        "settings": settings, "last_window": timing.summary(env.step_dt),
                        "total_steps": timing.total_steps,
                        "contact_cache": contact_cache_stats(env),
                    }, indent=2) + "\n")
                    if active:
                        finish_interrupted(env, True)
                        event("attempt_end", result="interrupted", timing_file=save_timing())
                finally:
                    if device is not None:
                        device.close()
                    if keyboard is not None:
                        keyboard.close()
                    try:
                        xr_session.close(env.sim.render)
                    finally:
                        (output / "xr_status.json").write_text(json.dumps(xr_session.metadata, indent=2) + "\n")
                        close_recording(env)
                        env.close()

    return collect()
