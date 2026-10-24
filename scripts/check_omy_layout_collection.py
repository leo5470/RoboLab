#!/usr/bin/env python3
"""Exercise real nonstop collection on CUDA without connecting XR or a leader.

This diagnostic replaces task success with a three-command counter. It verifies
collector transitions, simulator resets, recording rotation and source metadata;
it does not validate grasping or physical keyboard/headset delivery. Every HDF5
file and episode is marked training_ready=False and diagnostic_success=True.
"""

from __future__ import annotations

import time

SCRIPT_STARTED = time.perf_counter()

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COMMANDS_PER_LAYOUT = 3


def diagnostic_success(env):
    """Synthetic success only; exercise the real termination and export path."""
    return env.episode_length_buf >= COMMANDS_PER_LAYOUT


def _write_report(output, report):
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


def _smoke_real_controls_and_status(app, layout):
    """Inject provider events through real Carbonite subscription when possible."""
    import carb.input
    import omni.appwindow

    from robolab.core.omy_teleop.collection_controls import CollectionControls
    from robolab.core.omy_teleop.collection_status import CollectionStatus

    report = {"physical_keyboard_delivery_validated": False}
    window = omni.appwindow.get_default_app_window()
    if window is None or window.get_keyboard() is None:
        report.update(skipped=True, reason="Hidden Kit experience has no default window keyboard")
        return report
    controls = status = None
    try:
        controls = CollectionControls()
        report["binding_check"] = controls.binding_check
        provider = carb.input.acquire_input_provider()
        input_interface = carb.input.acquire_input_interface()

        def send(name, kind, modifiers=0):
            provider.buffer_keyboard_key_event(
                window.get_keyboard(), getattr(carb.input.KeyboardEventType, kind),
                getattr(carb.input.KeyboardInput, name), modifiers,
            )
            input_interface.distribute_buffered_events()

        send("F9", "KEY_PRESS")
        send("F9", "KEY_REPEAT")
        assert controls.take() and not controls.take(), "Real F9 subscription did not suppress repeats"
        send("F9", "KEY_RELEASE")
        send("F12", "KEY_PRESS")
        send("F12", "KEY_RELEASE")
        assert controls.take_next() and not controls.take_next(), "Real F12 subscription failed"
        send("F9", "KEY_PRESS", carb.input.KEYBOARD_MODIFIER_FLAG_SHIFT)
        send("F9", "KEY_RELEASE", carb.input.KEYBOARD_MODIFIER_FLAG_SHIFT)
        assert not controls.take(), "Modified F9 incorrectly triggered collection"
        status = CollectionStatus()
        status.update(object_name="DIAGNOSTIC", layout_index=0, layout_count=1,
                      task=layout.task, successes=0, target_successes=1, attempts=0,
                      completed=0, skipped=0, state="READY", reset_key="F9", next_key="F12")
        app.update()
        report.update(skipped=False, controls_passed=True, status_passed=True,
                      event_delivery="Carbonite provider buffer and real subscription")
    finally:
        if controls is not None:
            controls.close()
        if status is not None:
            status.close()
    return report


def run(args):
    from robolab.core.omy_teleop import collection

    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory must be new or empty: {output}")
    parser = collection.build_parser()
    run_args = parser.parse_args([
        "--object", args.object, "--episodes-per-layout", "1", "--desktop", "--source", "tcp",
        "--device", "cuda:0", "--control-hz", "15", "--stepping-profile", "held-target",
        "--recording-profile", "compact", "--solver-position-iterations", "16",
        "--render-interval", "4", "--no-realtime", "--episode-length-s", "120",
        "--max-attempt-seconds", "120", "--max-attempt-steps", "1800",
        "--output-dir", str(output),
    ])
    # The production collector requires a visible-window configuration. The
    # diagnostic supplies synthetic controls and independently hides its window.
    collection.validate_args(parser, run_args)
    plan = run_args.layout_plan
    layouts = {layout.task: layout for layout in plan.layouts}
    report = {
        "schema_version": 1, "object": plan.object_name, "catalog_index": plan.catalog_index,
        "expected_layouts": len(plan.layouts), "commands_per_layout": COMMANDS_PER_LAYOUT,
        "training_ready": False, "diagnostic_success": True,
        "success_condition": "episode_length_buf >= 3; no grasp validation",
        "serial_port_opened": False, "tcp_connection_opened": False, "xr_enabled": False,
        "headless": True, "render_camera": args.render_camera,
        "notes": "Real collection.run, environment, terminations, recorder, sessions and automatic advancement. "
                 "Synthetic controls, READY gate, fixed-home leader actions and XR lifecycle. "
                 "Provider-event checks do not validate physical keyboard or headset delivery.",
        "stages": {}, "reset_checks": [], "step_times_s": [], "recording_opens": [],
    }
    counts = Counter()
    app = None
    status = 1
    stage_start = time.perf_counter()
    try:
        import cv2  # noqa: F401 -- must precede Isaac Lab imports
        from isaaclab.app import AppLauncher

        report["stages"]["launcher_import_s"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        launcher = AppLauncher(
            headless=True, device="cuda:0", enable_cameras=args.render_camera,
            # Keep desktop input/UI extensions for the provider-event smoke,
            # while selecting the camera-enabled experience when needed.
            experience=("isaaclab.python.rendering.kit" if args.render_camera
                        else "isaaclab.python.kit"), rendering_mode="performance",
            kit_args="--/app/settings/persistent=false --/renderer/multiGpu/enabled=false "
                     "--/plugins/carb.tasking.plugin/threadCount=8 "
                     "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8 "
                     "--/persistent/physics/numThreads=2",
        )
        app = launcher.app
        report["stages"]["app_launch_s"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        import h5py
        import torch
        from isaaclab.devices import teleop_device_factory

        from robolab.core.environments import runtime
        from robolab.core.omy_teleop import collection_controls, layout_runtime
        from robolab.core.omy_teleop.config import OmyOperatorCamerasCfg
        from robolab.core.omy_postprocess.validation import validate_recording

        report["stages"]["runtime_import_s"] = time.perf_counter() - stage_start
        report["gpu"] = torch.cuda.get_device_name(0)
        report["real_controls_status_smoke"] = _smoke_real_controls_and_status(app, plan.layouts[0])
        real_create_env = runtime.create_env
        real_open = layout_runtime.LayoutRecording.open
        transition_started = {}

        def verify_pose(pose, velocity, values, label):
            expected = torch.as_tensor(values, device=pose.device, dtype=pose.dtype)
            position_error = (pose[:3] - expected[:3]).abs().max().item()
            quaternion_error = min((pose[3:7] - expected[3:7]).abs().max().item(),
                                   (pose[3:7] + expected[3:7]).abs().max().item())
            velocity_error = velocity.abs().max().item()
            assert max(position_error, quaternion_error, velocity_error) < 1e-5, (
                f"Wrong reset state for {label}: {pose}, {velocity}"
            )
            return max(position_error, quaternion_error, velocity_error)

        def create_environment(cfg, **kwargs):
            counts["env_create"] += 1
            cfg.terminations.success.func = diagnostic_success
            cfg.terminations.success.params = {}
            cfg.omy_collection_diagnostic = {"training_ready": False, "diagnostic_success": True}
            if args.render_camera:
                cfg.scene.view_cam = OmyOperatorCamerasCfg().view_cam
            begin = time.perf_counter()
            env, resolved = real_create_env(cfg, **kwargs)
            torch.cuda.synchronize()
            report["stages"]["environment_creation_s"] = time.perf_counter() - begin
            report["effective_stepping"] = cfg.omy_stepping_effective
            report["stepping_fallback_reasons"] = cfg.omy_stepping_fallback_reasons
            reset = env.reset
            step = env.step

            def reset_verified(*reset_args, **reset_kwargs):
                counts["env_reset"] += 1
                reset_begin = time.perf_counter()
                result = reset(*reset_args, **reset_kwargs)
                env.sim.render()
                pixels = (env.scene["view_cam"].data.output["rgb"][0, ..., :3].cpu().numpy()
                          if args.render_camera else None)
                torch.cuda.synchronize()
                ready = time.perf_counter()
                layout = layouts[run_args.task]
                maximum = 0.0
                for name, values in layout.poses.items():
                    asset = env.scene.rigid_objects[name]
                    pose = asset.data.root_pose_w[0].clone()
                    pose[:3] -= env.scene.env_origins[0]
                    maximum = max(maximum, verify_pose(pose, asset.data.root_vel_w[0], values,
                                                      f"{layout.task}/{name}"))
                check = {
                    "task": layout.task, "reset_s": ready - reset_begin,
                    "recording_open_to_ready_s": ready - transition_started[layout.task],
                    "reset_max_error": maximum,
                    "initial_ready_s": ready - SCRIPT_STARTED if len(report["reset_checks"]) == 0 else None,
                }
                if pixels is not None:
                    assert pixels.shape == (480, 640, 3) and pixels.std() > 1, "Blank or invalid rendered camera"
                    check["rgb_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
                    if len(report["reset_checks"]) in (0, len(plan.layouts) // 2, len(plan.layouts) - 1):
                        from PIL import Image

                        Image.fromarray(pixels).save(Path(env.output_dir) / "diagnostic_reset.png")
                report["reset_checks"].append(check)
                print("DIAGNOSTIC_LAYOUT_READY " + json.dumps(check), flush=True)
                return result

            def step_measured(action):
                begin = time.perf_counter()
                result = step(action)
                torch.cuda.synchronize()
                report["step_times_s"].append(time.perf_counter() - begin)
                counts["env_step"] += 1
                return result

            env.reset = reset_verified
            env.step = step_measured
            return env, resolved

        def open_recording(recording, part, **kwargs):
            task = recording.session.current.task
            transition_started[task] = time.perf_counter()
            recording.collection_metadata.update(training_ready=False, diagnostic_success=True,
                                                 success_condition="synthetic three-command counter")
            result = real_open(recording, part, **kwargs)
            recording.env.recorder_manager.set_dataset_attrs({
                "training_ready": False, "diagnostic_success": True,
                "diagnostic_warning": "Synthetic success; not a grasp demonstration",
            })
            report["recording_opens"].append({"task": task, "part": str(part)})
            counts["recording_open"] += 1
            return result

        class SyntheticControls:
            def __init__(self, *unused):
                counts["controls_create"] += 1
                self.binding_check = "Diagnostic synthetic controls; real controls smoke is reported separately."
                self.pending = True

            def take(self):
                pending, self.pending = self.pending, False
                return pending

            def take_next(self):
                return False

            def clear(self):
                self.pending = True

            def close(self):
                counts["controls_close"] += 1

        class SyntheticGate:
            def __init__(self, *unused, **kwargs):
                self._near_since = self._near_vector = self._anchor = None

            def observe_waiting(self):
                return True

            def confirm(self):
                return True

            def moved(self):
                return True

            def sample(self):
                return {}

        class SyntheticDevice:
            def __init__(self, cfg):
                counts["device_create"] += 1
                self.command = torch.tensor([*cfg.franka_initial_q, 1.0])
                self.last_input_sample = None

            def reset(self):
                counts["device_reset"] += 1

            def advance(self):
                counts["device_advance"] += 1
                self.last_input_sample = {
                    "stale": False, "age_s": 0.0, "source_wall_ns": time.time_ns(),
                    "read_monotonic_ns": time.monotonic_ns(), "raw_joint_position": [0.0] * 6,
                    "raw_gripper_position": 0.0, "wrist_turn_offset_rad": 0.0,
                }
                return self.command.clone()

            def close(self):
                counts["device_close"] += 1

        def create_device(name, device_configs, unused):
            assert name == "omy_leader"
            return SyntheticDevice(device_configs[name])

        class SyntheticXR:
            def __init__(self):
                self.metadata = {"mode": "diagnostic", "training_ready": False,
                                 "diagnostic_success": True, "completed": False}

            def configure(self, cfg):
                counts["xr_configure"] += 1

            def start(self):
                counts["xr_start"] += 1

            def ready(self):
                return True

            def wait_for_display(self, unused_app, unused_render):
                return True

            def close(self, unused_render):
                counts["xr_close"] += 1

        # Restore every replacement before closing Kit, even on failed assertions.
        with patch.object(runtime, "create_env", create_environment), \
                patch.object(teleop_device_factory, "create_teleop_device", create_device), \
                patch.object(collection, "ReadyGate", SyntheticGate), \
                patch.object(collection_controls, "CollectionControls", SyntheticControls), \
                patch.object(layout_runtime.LayoutRecording, "open", open_recording):
            stage_start = time.perf_counter()
            exit_code = collection.run(run_args, app, xr_session=SyntheticXR())
            report["stages"]["collection_run_s"] = time.perf_counter() - stage_start
            assert exit_code == 0, f"Collector exited with {exit_code}"

        session = json.loads((output / "session.json").read_text())
        assert session["position"] == len(plan.layouts), "Session did not finish every layout"
        assert len(report["reset_checks"]) == len(plan.layouts)
        assert [check["task"] for check in report["reset_checks"]] == list(layouts)
        expected_counts = {
            "env_create": 1, "env_reset": len(plan.layouts), "env_step": len(plan.layouts) * COMMANDS_PER_LAYOUT,
            "recording_open": len(plan.layouts), "device_create": 1, "device_close": 1,
            "device_advance": len(plan.layouts) * COMMANDS_PER_LAYOUT,
            "controls_create": 1, "controls_close": 1, "xr_configure": 1, "xr_start": 1, "xr_close": 1,
        }
        for name, expected in expected_counts.items():
            assert counts[name] == expected, f"Expected {name}={expected}, got {counts[name]}"

        report["recordings_verified"] = []
        for layout in plan.layouts:
            row = session["layouts"][layout.task]
            assert len(row["attempts"]) == 1 and not row["skipped"]
            attempt = next(iter(row["attempts"].values()))
            assert attempt["success"] and attempt["reason"] == "success"
            part = output / attempt["part"]
            source = part / "omy_demos.hdf5"
            configuration = json.loads((part / "env_cfg.json").read_text())
            assert configuration["scene"]["scene"]["spawn"]["usd_path"] == str(layout.scene_file)
            with h5py.File(source, "r+") as stream:
                data = stream["data"]
                assert not bool(data.attrs["training_ready"]) and bool(data.attrs["diagnostic_success"])
                assert data.attrs["collection_task"] == layout.task
                assert json.loads(data.attrs["robolab_omy_teleop"])["task"] == layout.task
                collection_metadata = json.loads(data.attrs["robolab_omy_collection"])
                assert not collection_metadata["training_ready"] and collection_metadata["diagnostic_success"]
                assert collection_metadata["layout"]["task"] == layout.task
                assert list(data) == ["demo_0"]
                episode = data["demo_0"]
                # Carry the diagnostic markers on episodes as well as datasets.
                episode.attrs.update(training_ready=False, diagnostic_success=True)
                assert bool(episode.attrs["collection_finalized"]) and bool(episode.attrs["success"])
                assert episode.attrs["collection_task"] == layout.task
                assert episode["actions"].shape == (COMMANDS_PER_LAYOUT, 8)
                assert int(episode.attrs["num_samples"]) == COMMANDS_PER_LAYOUT
                for name, values in layout.poses.items():
                    initial = episode[f"initial_state/rigid_object/{name}"]
                    verify_pose(torch.as_tensor(initial["root_pose"][0]),
                                torch.as_tensor(initial["root_velocity"][0]), values,
                                f"saved {layout.task}/{name}")
            report["recordings_verified"].append({"task": layout.task, "source": str(source),
                                                  "actions": COMMANDS_PER_LAYOUT, "training_ready": False})

        report["postprocess_validation"] = []
        indices = sorted({0, (len(plan.layouts) - 1) // 2, len(plan.layouts) - 1})
        for index in indices:
            source = Path(report["recordings_verified"][index]["source"])
            validation = validate_recording(source)
            (source.parent / "diagnostic_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
            assert not validation["errors"], f"Postprocessing rejected source {source}: {validation['errors']}"
            assert len(validation["episodes"]) == 1
            episode = validation["episodes"][0]
            assert episode["structural_valid"], f"Invalid episode {source}: {episode['errors']}"
            assert episode["native_15_hz"], f"Non-native timeline {source}: {episode['errors']}"
            assert not episode["training_eligible"], "Synthetic successes must remain excluded from training"
            assert episode["training_excluded"], "Missing explicit diagnostic exclusion"
            report["postprocess_validation"].append({
                "task": plan.layouts[index].task, "structural_valid": episode["structural_valid"],
                "native_15_hz": episode["native_15_hz"],
                "training_ready": False,
                "note": "Structural/timing validation does not turn synthetic success into grasp evidence",
            })
        transitions = [item["recording_open_to_ready_s"] for item in report["reset_checks"]][1:]
        report["transition_time"] = {
            "measurement": "LayoutRecording.open entry through env.reset, render/readback and CUDA synchronization",
            "count": len(transitions), "median_s": statistics.median(transitions) if transitions else None,
            "max_s": max(transitions) if transitions else None, "sum_s": sum(transitions),
            "excludes": "Attempt export, session manifest advancement, hardware/XR connection and operator waits",
        }
        if args.render_camera:
            assert len({check["rgb_sha256"] for check in report["reset_checks"]}) == len(plan.layouts)
        report["counts"] = dict(counts)
        report["session_complete"] = True
        report["elapsed_s"] = time.perf_counter() - SCRIPT_STARTED
        report["passed"] = True
        status = 0
    except BaseException as exc:
        report.update(passed=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                      counts=dict(counts), elapsed_s=time.perf_counter() - SCRIPT_STARTED)
        traceback.print_exc()
        raise
    finally:
        _write_report(output, report)
        print("DIAGNOSTIC_SUMMARY " + json.dumps({
            "passed": report.get("passed", False), "summary": str(output / "summary.json"),
            "elapsed_s": report.get("elapsed_s"), "transition_time": report.get("transition_time"),
            "training_ready": False,
        }), flush=True)
        if app is not None:
            app.app.post_quit(status)
            app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object", default="canned_mushrooms")
    parser.add_argument("--render-camera", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
