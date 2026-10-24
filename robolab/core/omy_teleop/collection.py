"""F9-key episode control for the dedicated L100/XR collector."""

import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np

from .options import ARM_JOINTS


RESET_KEY = "F9"


@dataclass
class CollectionProgress:
    """Keep saved attempt IDs separate from the successful-episode target."""

    target_successes: int
    attempts: int = 0
    successes: int = 0

    @property
    def complete(self):
        return self.successes >= self.target_successes

    def finish(self, results=()):
        self.attempts += 1
        self.successes += int(bool(results) and all(result.get("success", False) for result in results))

    def summary(self):
        return {"episode_count_mode": "successes", "completed_attempts": self.attempts,
                "successful_episodes": self.successes, "target_successful_episodes": self.target_successes}


class EpisodeReset:
    """Kit-window F9 presses, consumed once by the simulation thread."""

    def __init__(self):
        import carb.input
        import omni.appwindow

        self._input = carb.input.acquire_input_interface()
        window = omni.appwindow.get_default_app_window()
        if window is None or window.get_keyboard() is None:
            raise RuntimeError(f"{RESET_KEY} reset requires a Kit window with keyboard input")
        self._keyboard = window.get_keyboard()
        self._pressed = False
        self._pending = 0
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def _on_event(self, event, *unused):
        import carb.input

        # Kit can supply either a KeyboardInput enum or a string key.
        key = getattr(event.input, "name", event.input)
        if isinstance(key, str):
            key = key.upper()
        if key == RESET_KEY:
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
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None


class ReadyGate:
    """Compare fresh raw L100 samples against the fixed calibrated home."""

    def __init__(self, device, initial_q, *, tolerance_rad=0.12, movement_rad=0.035,
                 stable_s=0.3, clock=time.monotonic, reset_key=RESET_KEY):
        self.device = device
        self.initial_q = np.asarray(initial_q, dtype=float)
        self.tolerance_rad = tolerance_rad
        self.movement_rad = movement_rad
        self.stable_s = stable_s
        self.clock = clock
        self.reset_key = reset_key
        self._near_since = None
        self._near_vector = None
        self._anchor = None
        self.last_age_s = None
        self.last_reading = None

    def sample(self):
        reading, age = self.device._latest()
        self.last_age_s = age
        self.last_reading = reading
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

    def rejection_reason(self):
        """Explain the last sampled READY check without another source read."""
        if (self.last_reading is None or self.last_age_s is None
                or not math.isfinite(self.last_age_s) or self.last_age_s > self.device.cfg.stale_s):
            return "no fresh leader sample"
        target = self.device._mapper.raw_target(self.last_reading)
        errors = target - self.initial_q
        if not np.isfinite(errors).all():
            return "invalid mapped joint angles"
        reasons = []
        outside = np.flatnonzero(np.abs(errors) > self.tolerance_rad)
        if len(outside):
            joints = ", ".join(f"J{i + 1}={math.degrees(errors[i]):+.1f} deg" for i in outside)
            reasons.append(f"mapped offsets from home: {joints} (tolerance {math.degrees(self.tolerance_rad):.1f} deg)")
        if self.device._mapper.gripper_command(self.last_reading["gripper.pos"]) <= 0:
            reasons.append("trigger reads closed; release it fully")
        return "; ".join(reasons) if reasons else f"hold READY still for {self.stable_s:g} s before pressing {self.reset_key}"

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
    parser.description = f"Collect state-only L100/XR demonstrations; {RESET_KEY} resets or confirms READY."
    parser.set_defaults(record=True, save_failures=True, num_episodes=1000)
    parser._option_string_actions["--num-episodes"].help = (
        "Total successful episodes to collect across the selected task/layouts; "
        "object mode defaults to all layout quotas. Failed/reset attempts do not count."
    )
    parser.add_argument("--ready-tolerance-rad", type=float, default=0.12)
    parser.add_argument("--start-motion-rad", type=float, default=0.035)
    parser.add_argument("--ready-stable-s", type=float, default=0.3)
    from .preparation import add_arguments as preparation_arguments

    preparation_arguments(parser)
    parser.add_argument("--max-attempt-steps", type=int, default=1200,
                        help="Maximum recorded control steps per attempt; bounds live recorder memory.")
    parser.add_argument("--max-attempt-seconds", type=float, default=20.0,
                        help="Maximum simulated seconds per attempt (default: 20, independent of control rate).")
    parser.add_argument("--collection-handoff", help=argparse.SUPPRESS)
    parser.add_argument("--collection-run-id", help=argparse.SUPPRESS)
    from .collection_options import add_arguments

    add_arguments(parser)
    return parser


def validate_args(parser, args):
    from .xr import validate_args as validate_xr
    from .collection_options import configure_selection

    try:
        configure_selection(parser, args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    validate_xr(parser, args)
    from .preparation import PreparationSettings

    try:
        PreparationSettings.from_args(args)
    except ValueError as error:
        parser.error(str(error))
    if args.auto_zero or not args.calib:
        parser.error("collection needs the fixed READY calibration; --auto-zero is unsupported")
    if args.stream or args.record_images or not args.record or not args.save_failures:
        parser.error("collection requires state-only recording with failed attempts retained")
    if args.headless:
        parser.error(f"{RESET_KEY} reset requires the Isaac Sim window; omit --headless")
    if args.max_attempt_steps < 1:
        parser.error("--max-attempt-steps must be positive")
    if not math.isfinite(args.max_attempt_seconds) or args.max_attempt_seconds <= 0:
        parser.error("--max-attempt-seconds must be finite and positive")
    for key in ("ready_tolerance_rad", "start_motion_rad", "ready_stable_s"):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")


def run(args, app, *, xr_session):
    """Collect attempts with deliberate F9 reset/confirm transitions."""
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
    from .preparation import PreparationSettings
    from .preparation_runtime import ScenePreparation
    from .performance import PerformanceMonitor, RenderPacer
    from .runner import (close_recording, configure_recording, contact_cache_stats, finish_interrupted,
                         performance_settings, prepare_output)
    session = None

    @torch.inference_mode()
    def collect():
        nonlocal session
        plan = getattr(args, "layout_plan", None)
        if plan is not None:
            from .collection_options import run_contract, session_metadata
            from .layout_session import LayoutSession

            if args.resume:
                session = LayoutSession.resume(args.resume, plan=plan, run_metadata=run_contract(args))
            else:
                root = prepare_output(args.output_dir, task=plan.object_name)
                session = LayoutSession.create(root, plan, args.num_episodes, session_metadata(args),
                                               target_total_successes=getattr(args, "target_total_successes", None))
            handoff = getattr(args, "_collection_handoff", None)
            if handoff is not None:
                handoff.set_session(session.root)
            if session.finished:
                print(f"No pending layouts: {json.dumps(session.summary())}", flush=True)
                return 0
            args.task = session.current.task
            args.task_resolution = session.current.resolution
            args.task_dirs = list(session.current.resolution.task_dirs)
            output = session.start_part()
        else:
            output = prepare_output(args.output_dir, task=args.task)
        root_output = session.root if session else output
        handoff = getattr(args, "_collection_handoff", None)
        if handoff is not None:
            if not session:
                handoff.set_session(root_output)
            handoff.add_recording(output)
        reset_key = getattr(args, "reset_key", RESET_KEY)
        next_key = getattr(args, "next_layout_key", "F12")
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
        env = device = keyboard = cache = pacer = status = layout_recording = None
        timing = PerformanceMonitor(capacity=min(args.max_attempt_steps,
                                                 math.ceil(args.max_attempt_seconds * args.control_hz)))
        active = False
        total_steps = 0
        attempt_steps = 0
        attempt_timing = []
        progress = CollectionProgress(args.num_episodes)
        if session:
            progress.attempts, progress.successes = session.attempts, session.successes
        settings = {}
        finish_attempt = save_part_summary = event = None
        end_event_written = False
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
            recorder = env.recorder_manager
            # Keep HDF5 writes at episode boundaries, outside the live loop.
            # max_attempt_steps bounds the in-memory state-only buffer.
            collection_metadata = {
                "schema_version": 1, "start": "move_after_ready_confirmation",
                "reset_key": reset_key, "ready_tolerance_rad": args.ready_tolerance_rad,
                "start_motion_rad": args.start_motion_rad, "ready_stable_s": args.ready_stable_s,
                "images": "deferred", "recording": "state_only",
                "max_attempt_steps": args.max_attempt_steps,
                "max_attempt_seconds": args.max_attempt_seconds,
                "episode_count_mode": "successes", "target_successful_episodes": args.num_episodes,
                "preparation": PreparationSettings.from_args(args).metadata(),
            }
            if session:
                from .collection_controls import CollectionControls
                from .collection_status import CollectionStatus
                from .layout_runtime import LayoutRecording

                collection_metadata["next_layout_key"] = next_key
                collection_metadata["episodes_per_layout"] = session.target_successes
                collection_metadata["target_total_successful_episodes"] = session.target_total_successes
                layout_recording = LayoutRecording(env, cfg, args, device_cfg, xr_session, session,
                                                   collection_metadata)
                layout_recording.open(output, initialize=True)
                keyboard = CollectionControls(reset_key, next_key)
                status = CollectionStatus()
            else:
                configure_recording(env, args, device_cfg, xr_session)
                recorder.set_flush_interval(0)
                recorder.set_dataset_attrs({"robolab_omy_collection": json.dumps(collection_metadata)})
                if reset_key == RESET_KEY:
                    keyboard = EpisodeReset()
                else:
                    from .collection_controls import CollectionControls

                    keyboard = CollectionControls(reset_key, None)
            if getattr(keyboard, "binding_check", None):
                print(keyboard.binding_check, flush=True)
            xr_session.start()
            env.reset(seed=args.seed)
            if not xr_session.wait_for_display(app, env.sim.render):
                return 0
            device = create_teleop_device("omy_leader", cfg.teleop_devices.devices, {})
            device.reset()
            keyboard.clear()
            home = np.asarray([cfg.scene.robot.init_state.joint_pos[name] for name in ARM_JOINTS])
            gate = ReadyGate(device, home, tolerance_rad=args.ready_tolerance_rad,
                             movement_rad=args.start_motion_rad, stable_s=args.ready_stable_s, reset_key=reset_key)
            preparation = ScenePreparation(env, args)
            state = "settling"
            print(f"Task: {cfg.instruction}\nOutput: {output}", flush=True)
            print(f"Collection target: {args.num_episodes} successful episodes "
                  f"{'per layout' if session else 'for this task'}; unsuccessful attempts are retained.",
                  flush=True)
            if session and session.target_total_successes is not None:
                print(f"Total success target across layouts: {session.total_successes}/"
                      f"{session.target_total_successes}.", flush=True)
            print(f"Return L100 to READY, then press {reset_key} in the Isaac Sim window. "
                  f"Move L100 to start; press {reset_key} while recording to retry. "
                  + (f"{next_key} skips to the next layout. " if session else "") + "Ctrl+C exits.", flush=True)
            events_path = Path(root_output) / "collection_events.jsonl"

            def event(kind, **fields):
                line = json.dumps({"event": kind, "attempt": progress.attempts,
                                   "wall_time_s": time.time(), "total_steps": total_steps,
                                   **({"task": args.task, "part": str(output.relative_to(root_output))}
                                      if session else {}), **fields}) + "\n"
                paths = [events_path]
                if session:
                    # Offline archival reads events beside each HDF5 file. Keep
                    # that identity/time evidence as well as the session journal.
                    paths.append(output / "collection_events.jsonl")
                for path in paths:
                    with path.open("a") as stream:
                        stream.write(line)

            def show_status():
                if status is not None and session.current is not None:
                    summary = session.summary()
                    position = next(i for i, layout in enumerate(plan.layouts) if layout.task == args.task)
                    status.update(object_name=plan.object_name, layout_index=position, layout_count=len(plan.layouts),
                                  task=args.task, successes=progress.successes, target_successes=args.num_episodes,
                                  attempts=progress.attempts, completed=summary["completed"], skipped=summary["skipped"],
                                  state=state, reset_key=reset_key, next_key=next_key,
                                  total_successes=session.total_successes,
                                  target_total_successes=session.target_total_successes)
                    if session.target_total_successes is not None:
                        print(f"Total successes: {session.total_successes}/{session.target_total_successes}.", flush=True)
                    print(f"{plan.object_name}: layout {position + 1}/{len(plan.layouts)} ({args.task}); "
                          f"successes {progress.successes}/{args.num_episodes}; attempts {progress.attempts}; "
                          f"completed {summary['completed']}, skipped {summary['skipped']}; {state.upper()}", flush=True)

            show_status()

            def save_timing():
                if not attempt_timing:
                    return None
                path = output / f"attempt_{progress.attempts:04d}_timing.npz"
                columns = tuple(zip(*attempt_timing))
                np.savez(path, action_wall_ns=np.asarray(columns[0], dtype=np.int64),
                         step_done_wall_ns=np.asarray(columns[1], dtype=np.int64),
                         sim_before_s=np.asarray(columns[2], dtype=np.float64),
                         sim_after_s=np.asarray(columns[3], dtype=np.float64),
                         leader_age_s=np.asarray(columns[4], dtype=np.float64),
                         source_wall_ns=np.asarray(columns[5], dtype=np.int64),
                         source_read_monotonic_ns=np.asarray(columns[6], dtype=np.int64))
                (output / f"attempt_{progress.attempts:04d}_performance.json").write_text(json.dumps({
                    "active_control": timing.summary(env.step_dt),
                    "export": getattr(recorder, "export_timings", []),
                    "stepping_effective": getattr(cfg, "omy_stepping_effective", "ordinary"),
                    "fallback_reasons": getattr(cfg, "omy_stepping_fallback_reasons", []),
                }, indent=2) + "\n")
                attempt_timing.clear()
                return path.name

            def finish_attempt(reason, results=(), *, already_exported=False):
                nonlocal active
                if not active:
                    return
                if session and attempt_steps == 0:
                    recorder.clear()
                    active = False
                    event("attempt_cancelled", reason=reason)
                    return
                if not already_exported:
                    finish_interrupted(env, True)
                active = False
                event("attempt_end", result=results or reason, timing_file=save_timing())
                if session:
                    succeeded = bool(results) and all(result.get("success", False) for result in results)
                    layout_recording.finalize(progress.attempts, succeeded, reason)
                    progress.attempts, progress.successes = session.attempts, session.successes
                else:
                    progress.finish(results)
                print(f"Attempt {progress.attempts} ended: {reason}; "
                      f"successes {progress.successes}/{progress.target_successes}.", flush=True)

            def save_part_summary():
                (output / "omy_performance.json").write_text(json.dumps({
                    "settings": settings, "last_window": timing.summary(env.step_dt),
                    "total_steps": timing.total_steps, "contact_cache": contact_cache_stats(env),
                    "collection": progress.summary(),
                }, indent=2) + "\n")

            def reset_scene(reason, *, skip_layout=False):
                nonlocal state, attempt_steps, output, progress, settings, preparation
                if active:
                    finish_attempt(reason)
                if session and session.total_target_reached:
                    save_part_summary()
                    event("layout_end", reason="total_success_target_reached", **progress.summary())
                    return False
                if session and (progress.complete or skip_layout):
                    save_part_summary()
                    event("layout_end", reason="success_target_reached" if progress.complete else "manual_skip",
                          **progress.summary())
                    session.advance(skipped=skip_layout and not progress.complete)
                    if session.finished:
                        return False
                    progress = CollectionProgress(args.num_episodes, session.attempts, session.successes)
                    output = session.start_part()
                    if handoff is not None:
                        handoff.add_recording(output)
                    layout_recording.open(output)
                    settings = performance_settings(env, args)
                    event("layout_start", layout=session.current.metadata())
                elif progress.complete:
                    return False
                env.reset_eval_state()
                env.reset(seed=args.seed)
                device.reset()
                attempt_steps = 0
                attempt_timing.clear()
                keyboard.clear()
                if pacer is not None:
                    pacer.reset()
                preparation = ScenePreparation(env, args)
                state = "settling"
                gate._near_since = None
                gate._near_vector = None
                gate._anchor = None
                print(f"Scene reset; settling before READY. {reset_key} retries; {next_key} skips a layout.", flush=True)
                show_status()
                return True

            report_start = time.perf_counter()
            next_step = report_start
            while app.is_running() and not (session.finished if session else progress.complete):
                if not xr_session.ready():
                    if active:
                        if not reset_scene("xr_lost"):
                            break
                    elif state == "armed":
                        state = "waiting"
                        gate._near_since = None
                        gate._near_vector = None
                        gate._anchor = None
                        show_status()
                    if not xr_session.wait_for_display(app, env.sim.render):
                        break
                    keyboard.clear()
                    continue

                # Preparation advances physics without admitting a recorded
                # command. READY and ARMED hold the prepared scene still.
                if state != "recording":
                    env.sim.render()
                    if session and keyboard.take_next():
                        if not reset_scene("layout_skip", skip_layout=True):
                            break
                        continue
                    if state in ("settling", "preparation_failed"):
                        if keyboard.take():
                            if not reset_scene("preparation_retry"):
                                break
                        elif state == "settling":
                            report = preparation.advance()
                            if report["status"] != "settling":
                                from robolab.core.omy_teleop.collection_handoff import replace_report

                                replace_report(output / "preparation_latest.json", report)
                                event("scene_preparation", **report)
                                state = "waiting" if report["status"] == "ready" else "preparation_failed"
                                keyboard.clear()
                                gate._near_since = gate._near_vector = gate._anchor = None
                                print("Scene settled; return L100 to READY and confirm." if state == "waiting"
                                      else "Scene preparation failed: " + "; ".join(report["reasons"]), flush=True)
                                show_status()
                        continue
                    if state == "waiting":
                        aligned = gate.observe_waiting()
                        if keyboard.take():
                            if aligned and gate.confirm():
                                device.reset()
                                state = "armed"
                                event("armed", wrist_turn_offset_rad=(
                                    getattr(device, "last_input_sample", None) or {}
                                ).get("wrist_turn_offset_rad", 0.0))
                                print("READY confirmed. Move L100 to start recording.", flush=True)
                                show_status()
                            else:
                                print(f"L100 is not steady at calibrated READY: {gate.rejection_reason()}; "
                                      f"press {reset_key} again.", flush=True)
                    elif keyboard.take():
                        state = "waiting"
                        gate._near_since = None
                        gate._near_vector = None
                        gate._anchor = None
                        print(f"Arming cancelled. Press {reset_key} at READY.", flush=True)
                        show_status()
                    elif gate.moved():
                        recorder.set_episode_index(progress.attempts, env_ids=[0])
                        recorder.set_episode_seed(args.seed, env_ids=[0])
                        recorder.set_episode_preparation(getattr(env, "omy_preparation_report", {}), env_ids=[0])
                        active = True
                        attempt_steps = 0
                        attempt_timing.clear()
                        state = "recording"
                        timing.reset_window()
                        if pacer is not None:
                            pacer.reset()
                        next_step = report_start = time.perf_counter()
                        event("attempt_start")
                        show_status()
                        print(f"Attempt {progress.attempts + 1} recording; "
                              f"successes {progress.successes}/{progress.target_successes}.", flush=True)
                    else:
                        time.sleep(0.005)
                    continue

                if session and keyboard.take_next():
                    if not reset_scene("layout_skip", skip_layout=True):
                        break
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
                if args.stats_interval > 0 and time.perf_counter() - report_start >= args.stats_interval:
                    q = env.scene["robot"].data.joint_pos[0, joint_ids]
                    error = (q - action[:7]).abs().max().item()
                    print(timing.report(env.step_dt), flush=True)
                    print(f"OMY: max joint error {error:.4f} rad; step {total_steps}", flush=True)
                    from robolab.core.task.conditionals import object_resting_in_container, resting_in_container_diagnostics

                    terms = env.termination_manager
                    success = terms.get_term_cfg("success") if "success" in terms.active_terms else None
                    if success is not None and success.func is object_resting_in_container:
                        diagnostic = resting_in_container_diagnostics(env, success.params)
                        event("placement_check", **diagnostic)
                        if diagnostic["checks"]["inside"]:
                            print("Placement check: " + json.dumps(diagnostic, sort_keys=True), flush=True)
                    report_start = time.perf_counter()
                skip_after_step = session is not None and keyboard.take_next()
                if env.all_terminated:
                    results = env.get_env_results()
                    finish_attempt("success" if all(result.get("success", False) for result in results)
                                   else "terminal_failure", results, already_exported=True)
                    print(f"Attempt {progress.attempts}: {results}; "
                          f"successes {progress.successes}/{progress.target_successes}.", flush=True)
                    if not reset_scene("terminal", skip_layout=skip_after_step):
                        break
                elif skip_after_step:
                    if not reset_scene("layout_skip", skip_layout=True):
                        break
                elif attempt_steps >= args.max_attempt_steps:
                    if not reset_scene("attempt_step_limit"):
                        break
                elif attempt_steps * env.step_dt >= args.max_attempt_seconds:
                    if not reset_scene("attempt_time_limit"):
                        break
                if args.steps and total_steps >= args.steps:
                    break
            if active:
                finish_attempt("interrupted")
            if session:
                event("collection_end", reason="pass_finished" if session.finished else "interrupted",
                      **session.summary())
                end_event_written = True
                print(f"Collection session: {json.dumps(session.summary())}\nSession directory: {session.root}", flush=True)
            elif progress.complete:
                event("collection_end", reason="success_target_reached", **progress.summary())
                print(f"Collection complete: {progress.successes}/{progress.target_successes} successful episodes "
                      f"from {progress.attempts} attempts.", flush=True)
            xr_session.metadata["completed"] = True
            return 0
        except KeyboardInterrupt:
            print("Stopping OMY collection.", flush=True)
            return 130
        finally:
            from contextlib import ExitStack

            # Every resource must close even if finalization or another cleanup fails.
            with ExitStack() as cleanup:
                if env is not None:
                    cleanup.callback(env.close)
                    cleanup.callback(close_recording, env)

                    def save_xr_status():
                        (output / "xr_status.json").write_text(json.dumps(xr_session.metadata, indent=2) + "\n")

                    def close_xr():
                        if session:
                            xr_session.metadata["layout_collection"] = session.summary()
                        else:
                            xr_session.metadata.update(progress.summary())
                        xr_session.close(env.sim.render)

                    cleanup.callback(save_xr_status)
                    cleanup.callback(close_xr)
                for resource in (status, keyboard, device, cache, timing, pacer):
                    if resource is not None:
                        cleanup.callback(resource.close)
                if env is not None:
                    print(timing.report(env.step_dt), flush=True)
                    if active:
                        finish_attempt("interrupted")
                    if session and event is not None and not end_event_written:
                        event("collection_end", reason="interrupted", **session.summary())
                    if save_part_summary is not None:
                        save_part_summary()

    try:
        return collect()
    finally:
        if session is not None:
            session.close()
