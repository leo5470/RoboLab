#!/usr/bin/env python3
"""Compare fresh simulator launches with pose-only simple-PnP layout switches.

Diagnostic only: headless CUDA physics, no XR or leader connection. Uses the
collection launcher's physics/recording profile and writes synthetic episodes
under a new output directory. Does not modify the production collector.
"""

import time

SCRIPT_STARTED = time.perf_counter()

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def select_layouts(object_name):
    manifest = ROOT / "robolab/tasks/single_pnp/manifest.json"
    rows = [row for row in json.loads(manifest.read_text())["tasks"] if row["target"] == object_name]
    rows.sort(key=lambda row: row["class_name"])
    if not rows:
        raise ValueError(f"No retained layouts for {object_name!r}")
    prefixes = {row["task"].rsplit("_", 2)[0] for row in rows}
    if len(prefixes) != 1:
        raise ValueError(f"Object name selects multiple catalog entries: {sorted(prefixes)}")
    # Reusing the termination managers is valid only for identical task logic.
    signatures = set()
    for row in rows:
        source = (ROOT / "robolab/tasks/single_pnp" / row["task"]).read_text()
        source = source.replace(row["class_name"].removesuffix("Task"), "Layout")
        signatures.add(source.replace(Path(row["scene"]).name, "layout.usda"))
    if len(signatures) != 1:
        raise ValueError("Selected layouts have different task definitions")
    return rows


def load_poses(rows):
    """Read exact authored XYZ/WXYZ, including height; do not reconstruct yaw."""
    from pxr import Sdf

    signatures = set()
    layouts = []
    for row in rows:
        path = ROOT / "assets/scenes" / row["scene"]
        source = path.read_text()
        layer = Sdf.Layer.FindOrOpen(str(path))
        poses = {}
        for name in (row["target"], row["destination"]):
            position = layer.GetAttributeAtPath(f"/world/{name}.xformOp:translate").default
            rotation = layer.GetAttributeAtPath(f"/world/{name}.xformOp:orient").default
            poses[name] = [*position, rotation.GetReal(), *rotation.GetImaginary()]
            pattern = (
                r'(    def(?: Xform)? "' + re.escape(name)
                + r'"[^\n]*\([^}]*?\n    \{)(.*?)(\n    \})'
            )

            def strip_pose(match):
                body = re.sub(
                    r"((?:quat[fd] xformOp:orient|double3 xformOp:translate) = )\([^\n]+\)",
                    r"\1<pose>", match.group(2),
                )
                return match.group(1) + body + match.group(3)

            source, count = re.subn(pattern, strip_pose, source, flags=re.S)
            if count != 1:
                raise ValueError(f"Unsupported scene structure: {path}: {name}")
        signatures.add(source)
        layouts.append({**row, "poses": poses, "scene_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if len(signatures) != 1:
        raise ValueError("Scenes differ beyond target/bin translation and orientation")
    return layouts


def worker(args):
    started = args.launched_at or SCRIPT_STARTED
    stages = {}
    rows = select_layouts(args.object)
    if args.worker == "fresh":
        rows = [rows[args.layout_index]]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stage_start = time.perf_counter()
    import cv2  # noqa: F401 -- must precede Isaac Lab imports
    from isaaclab.app import AppLauncher

    stages["launcher_import_s"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    launcher = AppLauncher(
        headless=True, device="cuda:0", enable_cameras=args.render_camera,
        kit_args="--/app/settings/persistent=false --/renderer/multiGpu/enabled=false "
                 "--/plugins/carb.tasking.plugin/threadCount=8 "
                 "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8 "
                 "--/persistent/physics/numThreads=2",
    )
    app = launcher.app
    stages["app_launch_s"] = time.perf_counter() - stage_start
    env = None
    status = 1
    try:
        stage_start = time.perf_counter()
        import h5py
        import torch
        from isaaclab.managers import DatasetExportMode

        import robolab.constants
        from robolab.core.environments.config import parse_env_cfg
        from robolab.core.environments.runtime import create_env
        from robolab.core.omy_teleop.config import OmyOperatorCamerasCfg, configure_performance, register_omy_envs
        from robolab.core.omy_teleop.environment import OmyEnv
        from robolab.core.omy_teleop.options import ARM_JOINTS, build_parser
        from robolab.core.omy_teleop.runner import close_recording, finish_interrupted

        stages["runtime_import_s"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        layouts = load_poses(rows)
        stages["layout_preload_s"] = time.perf_counter() - stage_start
        robolab.constants.RECORD_IMAGE_DATA = True
        robolab.constants.set_output_dir(str(output / "bootstrap"))
        task = rows[0]["class_name"]
        postfix = "OMYLoadBenchmark"
        stage_start = time.perf_counter()
        register_omy_envs(task, env_postfix=postfix, task_dirs=["single_pnp"])
        run_args = build_parser().parse_args([
            "--control-hz", "15", "--record", "--save-failures",
            "--recording-profile", "compact", "--solver-position-iterations", "16",
            "--stepping-profile", "held-target", "--render-interval", "4",
            "--contact-sensors", "required", "--contact-history-length", "0",
            "--episode-length-s", "120",
        ])
        cfg = parse_env_cfg(f"{task}{postfix}", device="cuda:0", num_envs=1, seed=0)
        cfg.episode_length_s = 120
        if args.render_camera:
            cfg.scene.view_cam = OmyOperatorCamerasCfg().view_cam
        configure_performance(cfg, run_args)
        stages["task_configuration_s"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        env, cfg = create_env(cfg, env_class=OmyEnv, policy="omy_joint_position", rendering_mode="performance")
        torch.cuda.synchronize()
        stages["environment_creation_s"] = time.perf_counter() - stage_start
        recorder = env.recorder_manager
        recorder.set_buffered_recording()
        recorder.cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        recorder.set_hdf5_compression(None)
        recorder.set_flush_interval(0)
        recorder.set_flush_memory_cleanup(False)
        command = torch.tensor(
            [[cfg.scene.robot.init_state.joint_pos[name] for name in ARM_JOINTS] + [1.0]],
            device=env.device,
        )
        # Diagnostics retain the production recorder terms; absent leader samples
        # are marked invalid by LeaderStateRecorder.
        report = {
            "mode": args.worker, "object": args.object, "device": torch.cuda.get_device_name(0),
            "stages": stages, "runs": [], "serial_port_opened": False, "xr_enabled": False,
            "headless": True, "training_ready": False, "commands_per_layout": args.commands,
            "render_camera": args.render_camera,
            "effective_stepping": cfg.omy_stepping_effective,
            "notes": "Readiness includes output rotation, config writing, reset, render call, and CUDA sync. "
                     "Optional 640x480 operator camera includes RGB readback. "
                     "No XR, operator wait, policy cameras, or cold disk/shader-cache guarantee.",
        }

        def validate(layout, *, recorded=False):
            maximum = 0.0
            for name, values in layout["poses"].items():
                expected = torch.tensor(values, device=env.device)
                if recorded:
                    state = recorder._episodes[0].data["initial_state"]["rigid_object"][name]
                    pose = state["root_pose"][0]
                    velocity = state["root_velocity"][0]
                else:
                    asset = env.scene.rigid_objects[name]
                    pose = asset.data.root_pose_w[0].clone()
                    pose[:3] -= env.scene.env_origins[0]
                    velocity = asset.data.root_vel_w[0]
                position_error = (pose[:3] - expected[:3]).abs().max().item()
                quaternion_error = min(
                    (pose[3:7] - expected[3:7]).abs().max().item(),
                    (pose[3:7] + expected[3:7]).abs().max().item(),
                )
                maximum = max(maximum, position_error, quaternion_error)
                if maximum > 1e-5 or velocity.abs().max().item() > 1e-5:
                    raise AssertionError(f"Wrong reset state for {layout['class_name']}/{name}: {pose}, {velocity}")
            return maximum

        with torch.inference_mode():
            for index, layout in enumerate(layouts):
                torch.cuda.synchronize()
                transition_start = time.perf_counter()
                directory = output / layout["class_name"]
                directory.mkdir()
                # The existing scene remains alive; change the reset buffers,
                # not USD references or rigid-body handles.
                for name, pose in layout["poses"].items():
                    asset = env.scene.rigid_objects[name]
                    if index == 0:
                        expected = torch.tensor(pose, device=env.device)
                        actual = asset.data.default_root_state[0, :7]
                        # USD decomposition may choose the opposite quaternion
                        # sign, which represents exactly the same orientation.
                        if torch.dot(actual[3:7], expected[3:7]) < 0:
                            expected[3:7] *= -1
                        torch.testing.assert_close(
                            actual, expected, atol=1e-5, rtol=1e-5,
                        )
                    asset.data.default_root_state[:, :7] = torch.tensor(pose, device=env.device)
                    asset.data.default_root_state[:, 7:] = 0
                    getattr(cfg.scene, name).init_state.pos = tuple(pose[:3])
                    getattr(cfg.scene, name).init_state.rot = tuple(pose[3:7])
                cfg.scene.scene.spawn.usd_path = str(ROOT / "assets/scenes" / layout["scene"])
                cfg._task_name = layout["class_name"]
                cfg.env_name = layout["class_name"] + postfix
                env.output_dir = str(directory)
                robolab.constants.set_output_dir(str(directory))
                recorder.cfg.dataset_export_dir_path = str(directory)
                cfg.recorders.dataset_export_dir_path = str(directory)
                recorder._env_name = cfg.env_name
                recorder.clear()
                recorder.set_hdf5_file("loading_diagnostic")
                recorder.set_dataset_attrs({
                    "training_ready": False, "benchmark_task": layout["class_name"],
                    "scene_sha256": layout["scene_sha256"],
                })
                (directory / "env_cfg.json").write_text(json.dumps(cfg.to_dict(), default=str))
                reset_start = time.perf_counter()
                env.reset_eval_state()
                env.reset(seed=0)
                env.sim.render()
                pixels = env.scene["view_cam"].data.output["rgb"][0, ..., :3].cpu().numpy() if args.render_camera else None
                torch.cuda.synchronize()
                ready = time.perf_counter()
                result = {
                    "task": layout["class_name"], "poses": layout["poses"],
                    "transition_s": ready - transition_start,
                    "reset_s": ready - reset_start,
                    "initial_ready_s": ready - started if index == 0 else None,
                    "reset_max_error": validate(layout),
                    "recorded_reset_max_error": validate(layout, recorded=True),
                }
                if pixels is not None:
                    if pixels.shape != (480, 640, 3) or pixels.std() < 1:
                        raise AssertionError("Rendered camera returned an invalid or blank frame")
                    result["rgb_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
                    if index in (0, (len(layouts) - 1) // 2, len(layouts) - 1):
                        from PIL import Image

                        Image.fromarray(pixels).save(directory / "reset.png")
                recorder.set_episode_index(0, [0])
                recorder.set_episode_seed(0, [0])
                step_start = time.perf_counter()
                for _ in range(args.commands):
                    env.step(command)
                    for asset in env.scene.rigid_objects.values():
                        if not torch.isfinite(asset.data.root_state_w).all():
                            raise AssertionError("Nonfinite rigid-body state after a layout switch")
                    if env.all_terminated:
                        break
                torch.cuda.synchronize()
                result["synthetic_episode_s"] = time.perf_counter() - step_start
                export_start = time.perf_counter()
                finish_interrupted(env, True)
                torch.cuda.synchronize()
                result["export_s"] = time.perf_counter() - export_start
                report["runs"].append(result)
                (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
                print("LAYOUT_READY " + json.dumps(result), flush=True)
            close_recording(env)
            for layout in layouts:
                path = output / layout["class_name"] / "loading_diagnostic.hdf5"
                with h5py.File(path, "r") as file:
                    assert file["data"].attrs["benchmark_task"] == layout["class_name"]
                    assert list(file["data"].keys()) == ["demo_0"]
                    episode = file["data/demo_0"]
                    assert len(episode["actions"]) == args.commands
                    for name, pose in layout["poses"].items():
                        actual = episode[f"initial_state/rigid_object/{name}/root_pose"][0]
                        torch.testing.assert_close(torch.tensor(actual), torch.tensor(pose), atol=1e-5, rtol=1e-5)
            report["saved_initial_states_verified"] = len(layouts)
            if args.render_camera and len({run["rgb_sha256"] for run in report["runs"]}) != len(layouts):
                raise AssertionError("At least two layout switches produced the same rendered frame")
            (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
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


def orchestrate(args):
    rows = select_layouts(args.object)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    indices = [0, (len(rows) - 1) // 2, len(rows) - 1]
    jobs = [("fresh_0", "fresh", indices[0]), ("reuse_all", "reuse", 0),
            ("fresh_1", "fresh", indices[1]), ("fresh_2", "fresh", indices[2])]
    results = {}
    for label, mode, index in jobs:
        log_path = output / f"{label}.log"
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker", mode,
            "--object", args.object, "--layout-index", str(index),
            "--commands", str(args.commands), "--output-dir", str(output / label),
            "--launched-at", str(time.perf_counter()),
        ]
        if args.render_camera:
            command.append("--render-camera")
        print(f"Starting {label}; log: {log_path}", flush=True)
        process_start = time.perf_counter()
        with log_path.open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=600, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "Y", "XR": "0"})
        results[label] = json.loads((output / label / "result.json").read_text())
        results[label]["process_total_s"] = time.perf_counter() - process_start
        print(f"{label}: first layout ready in {results[label]['runs'][0]['initial_ready_s']:.3f} s", flush=True)
    baseline = [results[f"fresh_{i}"]["runs"][0]["initial_ready_s"] for i in range(3)]
    reused = results["reuse_all"]["runs"]
    transitions = [run["transition_s"] for run in reused[1:]]
    summary = {
        "object": args.object, "layout_count": len(rows), "fresh_launch_samples_s": baseline,
        "fresh_launch_median_s": statistics.median(baseline),
        "reuse_initial_ready_s": reused[0]["initial_ready_s"],
        "reuse_transition_samples_s": transitions,
        "reuse_transition_median_s": statistics.median(transitions),
        "reuse_transition_max_s": max(transitions),
        "reuse_total_loading_s": reused[0]["initial_ready_s"] + sum(transitions),
        "estimated_fresh_total_loading_s": statistics.median(baseline) * len(rows),
        "saved_initial_states_verified": results["reuse_all"]["saved_initial_states_verified"],
        "headless": True, "xr_enabled": False, "training_ready": False,
        "render_camera": args.render_camera,
        "notes": "Fresh total is extrapolated from three launches. Reuse loading is measured across every layout. "
                 "Excludes demonstration duration, prior-episode export, shutdown, XR, and operator readiness.",
    }
    summary["estimated_loading_reduction_percent"] = 100 * (
        1 - summary["reuse_total_loading_s"] / summary["estimated_fresh_total_loading_s"]
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", default="canned_mushrooms")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commands", type=int, default=3)
    parser.add_argument("--render-camera", action="store_true",
                        help="Include a 640x480 operator RGB camera and wait for its frame at each reset.")
    parser.add_argument("--worker", choices=("fresh", "reuse"), help=argparse.SUPPRESS)
    parser.add_argument("--layout-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--launched-at", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.commands < 1:
        parser.error("--commands must be positive")
    worker(args) if args.worker else orchestrate(args)


if __name__ == "__main__":
    main()
