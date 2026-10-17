#!/usr/bin/env python3
"""Diagnostic-only continuation of a closed OMY terminal snapshot.

Reuses the materializer's simulator bootstrap/config builder, not its export
loop. Never writes training episodes or opens a leader. The added trajectory
is new physics, not recovered recording/contact-solver history.
"""

import argparse
import json
import math
from pathlib import Path
import runpy
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def continuation_schedule(physics_dt, seconds, fps):
    if not math.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("Physics timestep must be finite and positive")
    if not math.isfinite(seconds) or not 1 <= seconds <= 2:
        raise ValueError("Diagnostic continuation must be 1–2 simulated seconds")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
        raise ValueError("Diagnostic FPS must be a positive integer")
    ticks_per_frame = round(1 / (fps * physics_dt))
    ticks = round(seconds / physics_dt)
    if (ticks_per_frame < 1 or ticks % ticks_per_frame
            or not math.isclose(ticks_per_frame * physics_dt, 1 / fps, abs_tol=1e-10)
            or not math.isclose(ticks * physics_dt, seconds, abs_tol=1e-10)):
        raise ValueError("Duration/FPS must align exactly with recorded physics ticks")
    return ticks, ticks_per_frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--recorded-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--tail-seconds", type=float, default=2)
    parser.add_argument("--tail-fps", type=int, default=60)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    import h5py
    import numpy as np
    from robolab.core.omy_postprocess.cameras import preflight_omy_materialization
    from robolab.core.omy_postprocess.io import (
        fsync_directory, publish_directory, sha256_file, sync_file, write_json,
    )
    from robolab.core.omy_postprocess.render import compare_raw_channels, verify_render_cache

    source = args.input.resolve()
    images = args.recorded_images.resolve()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing diagnostic output: {output}")
    metadata, validation = preflight_omy_materialization(
        source, episodes=[args.episode], require_native15=True)
    episode = next(e for e in validation["episodes"] if e["name"] == f"demo_{args.episode}")
    group = episode["group"]
    config_path = source.parent / "env_cfg.json"
    config = json.loads(config_path.read_text())
    config_sha = sha256_file(config_path)
    physics_dt = float(config["sim"]["dt"])
    ticks, ticks_per_frame = continuation_schedule(physics_dt, args.tail_seconds, args.tail_fps)
    if args.tail_fps % 15:
        raise ValueError("Combined preview FPS must be an integer multiple of native 15")
    image_sha = sha256_file(images)
    cache_manifest = json.loads((images.parent / "manifest.json").read_text())
    if cache_manifest["profile"]["source_sha256"] != validation["source"]["sha256"]:
        raise ValueError("Camera cache does not come from this source snapshot")
    verify_render_cache(images.parent, cache_manifest["profile"])
    preservation = compare_raw_channels(source, images, group)
    with h5py.File(images, "r") as file:
        camera_names = ("over_shoulder_left_camera", "wrist_cam")
        shapes = [file[f"{group}/obs/image_obs/{name}"].shape for name in camera_names]
        if shapes[0] != shapes[1] or len(shapes[0]) != 4 or shapes[0][-1] != 3:
            raise ValueError("Diagnostic preview requires matching RGB policy cameras")
        height, width = shapes[0][1:3]

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    runtime = None
    env = None
    writers = []
    try:
        # The existing CLI is also the authoritative OMY config reconstruction.
        # Import under a non-main name: it boots Kit but never runs its HDF5
        # writer. Scratch output and final diagnostic output are separate.
        saved_argv = sys.argv
        sys.argv = ["materialize_demo_images.py", "--input", str(source),
                    "--output", str(partial / "unused_materializer_output.hdf5"),
                    "--episode", str(args.episode), "--require-native15", "--headless",
                    "--device", args.device, "--camera-width", str(width),
                    "--camera-height", str(height), "--kit_args",
                    "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false "
                    "--/plugins/carb.tasking.plugin/threadCount=8 "
                    "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8 "
                    "--/persistent/physics/numThreads=2"]
        try:
            runtime = runpy.run_path(str(Path(__file__).with_name("materialize_demo_images.py")),
                                     run_name="omy_diagnostic_materializer_bootstrap")
        finally:
            sys.argv = saved_argv
        import cv2
        import torch
        import robolab.constants
        from robolab.core.replay.materialize import RecordedEpisode
        from robolab.core.replay.scene_state import restore_scene_state
        from robolab.core.utils.video_utils import VideoWriter

        robolab.constants.set_output_dir(str(partial / "runtime"))
        robolab.constants.RECORD_IMAGE_DATA = False
        robolab.constants.VERBOSE = False
        env, resolutions, assets = runtime["_build_env"](metadata)
        if not math.isclose(env.physics_dt, physics_dt, abs_tol=1e-12):
            raise ValueError("Rebuilt physics timestep differs from recorded configuration")
        with h5py.File(source, "r") as file:
            recorded = RecordedEpisode.load(file[group])
            final_action = file[f"{group}/actions"][-1].copy()
            source_terminal_time = float(file[f"{group}/timing/sim_after_s"][-1])
        env.reset(seed=recorded.seed)
        skipped = restore_scene_state(env, recorded.states, recorded.num_actions - 1)
        if skipped:
            raise ValueError(f"Terminal scene state was not completely restored: {skipped}")
        final_action[-1] = 1.0  # Explicit diagnostic release, not a training command.
        command = torch.as_tensor(final_action[None], device=env.device)
        env.action_manager.process_action(command)
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.forward()

        combined = VideoWriter(str(partial / "recorded_plus_settling.mp4"), args.tail_fps)
        tail = VideoWriter(str(partial / "settling_only.mp4"), args.tail_fps)
        writers = [combined, tail]

        def panel(frames, label):
            strip = np.zeros((32, width * 2, 3), dtype=np.uint8)
            cv2.putText(strip, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (255, 255, 255), 1)
            return np.concatenate([np.concatenate(frames, axis=1), strip], axis=0)

        with h5py.File(images, "r") as file:
            channels = file[f"{group}/obs/image_obs"]
            for step in range(recorded.num_actions):
                frame = panel([channels[name][step] for name in camera_names],
                              f"RECORDED 15 Hz | frame {step} | t={step / 15:.3f}s")
                # Zero-order display repetition, never extra dataset samples.
                for _ in range(args.tail_fps // 15):
                    combined.write(frame)

        times, callback_times, poses, velocities, joints = [], [], [], [], []
        start_clock = float(env.sim.current_time)
        start_tick = int(env.sim.current_time_step_index)

        def capture(expected_tick):
            elapsed_ticks = int(env.sim.current_time_step_index) - start_tick
            if elapsed_ticks != expected_tick:
                raise ValueError(f"Unexpected physics advance: {elapsed_ticks} ticks, expected {expected_tick}")
            # Isaac's timer callback accumulates float32 step_size. Use the
            # independently checked actual tick counter and recorded float64
            # timestep, just as native OMY collection does. Retain both clocks.
            elapsed = elapsed_ticks * physics_dt
            callback_elapsed = float(env.sim.current_time) - start_clock
            env.sim.forward()
            env.sim.render()
            env.scene.update(dt=0.0)
            if int(env.sim.current_time_step_index) - start_tick != expected_tick:
                raise ValueError("Rendering unexpectedly advanced diagnostic physics")
            frames = []
            for name in camera_names:
                sensor = env.scene.sensors[name]
                sensor.update(dt=0.0, force_recompute=True)
                frames.append(sensor.data.output["rgb"][0, ..., :3].cpu().numpy().copy())
            frame = panel(frames, f"DIAGNOSTIC RESIMULATION | +{elapsed:.3f}s | gripper OPEN")
            combined.write(frame)
            tail.write(frame)
            banana = env.scene.rigid_objects["banana"]
            times.append(elapsed)
            callback_times.append(callback_elapsed)
            poses.append(banana.data.root_pose_w[0].cpu().numpy().copy())
            velocities.append(banana.data.root_vel_w[0].cpu().numpy().copy())
            joints.append(env.scene.articulations["robot"].data.joint_pos[0].cpu().numpy().copy())
            if len(times) % 30 == 1:
                print(f"Diagnostic +{elapsed:.3f}s: banana z={poses[-1][2]:.5f}, "
                      f"vz={velocities[-1][2]:.5f}", flush=True)

        capture(0)  # Extra terminal frame, from the recorded final post-state.
        # Bypass env.step(): no success freezing, reset, or recorder callbacks.
        # Submit held targets each physics tick using ordinary scene updates.
        for tick in range(1, ticks + 1):
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=physics_dt)
            if tick % ticks_per_frame == 0:
                capture(tick)
        for writer in writers:
            writer.release()
        if not math.isclose(times[-1], args.tail_seconds, abs_tol=1e-8):
            raise ValueError("Actual simulated continuation duration does not match the request")
        with h5py.File(partial / "diagnostic_states.hdf5", "w") as file:
            file.attrs.update(artifact_kind="diagnostic_resimulation", training_ready=False,
                              source_sha256=validation["source"]["sha256"], source_group=group)
            for name, values in (("elapsed_sim_time_s", times), ("banana_root_pose_w", poses),
                                 ("callback_elapsed_time_s", callback_times),
                                 ("banana_root_velocity_w", velocities), ("robot_joint_position", joints),
                                 ("held_action", final_action)):
                file.create_dataset(name, data=np.asarray(values))
        for path, expected in ((source, validation["source"]["sha256"]),
                               (config_path, config_sha), (images, image_sha)):
            if sha256_file(path) != expected:
                raise RuntimeError(f"Immutable input changed during diagnostic rendering: {path}")
        report = {"schema_version": 1, "artifact_kind": "diagnostic_resimulation",
                  "training_ready": False, "source": str(source), "source_group": group,
                  "source_sha256": validation["source"]["sha256"], "source_unchanged": True,
                  "recorded_images_sha256": image_sha, "raw_preservation": preservation,
                  "source_samples": recorded.num_actions, "source_fps": 15,
                  "source_terminal_sim_time_s": source_terminal_time,
                  "tail_simulated_seconds": times[-1], "tail_physics_ticks": ticks,
                  "callback_elapsed_seconds": callback_times[-1],
                  "time_basis": "checked actual physics tick index times recorded float64 dt",
                  "physics_dt": physics_dt, "gravity": config["sim"]["gravity"],
                  "tail_fps": args.tail_fps, "tail_frames_including_terminal": len(times),
                  "held_action": final_action.tolist(), "camera_resolutions": resolutions,
                  "asset_dependencies": assets, "start_banana_pose_w": poses[0].tolist(),
                  "end_banana_pose_w": poses[-1].tolist(),
                  "end_banana_velocity_w": velocities[-1].tolist(),
                  "limitations": ["Contact/solver history is not restored: tail is newly simulated.",
                                  "Recorded frames are repeated only for 60 FPS display, not resampled data.",
                                  "Success/freeze/reset/recorder managers are not run.",
                                  "The final video image is held for one frame interval."]}
        report["artifact_sha256"] = {p.name: sha256_file(p) for p in partial.iterdir() if p.is_file()}
        write_json(partial / "report.json", report)
        for path in partial.rglob("*"):
            if path.is_file():
                sync_file(path)
        fsync_directory(partial)
        publish_directory(partial, output)
        print(f"Diagnostic videos and report: {output}", flush=True)
        return 0
    except BaseException as error:
        if partial.exists() and not (partial / "failure.json").exists():
            write_json(partial / "failure.json", {"type": type(error).__name__, "message": str(error)})
        print(f"Diagnostic failed; retained {partial}: {error}", file=sys.stderr, flush=True)
        raise
    finally:
        for writer in writers:
            writer.release()
        if env is not None:
            env.close()
        if runtime is not None:
            runtime["simulation_app"].close()


if __name__ == "__main__":
    raise SystemExit(main())
