"""Replay a source demo through the DROID Mimic adapter and write a review video.

This stops at source review. It does not annotate or generate demonstrations.
Run from the repository root with the collection virtual environment.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import cv2  # Must precede Isaac Sim imports.
import h5py
import numpy as np

from robolab.datagen.audit import audit_dataset, fingerprint, installed_versions


def parser():
    from isaaclab.app import AppLauncher

    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are rejected")
    result.add_argument("--episode", default="demo_0")
    result.add_argument("--physics-device", default="cpu", help="Match the collection physics device")
    result.add_argument("--trials", type=int, default=2, help="Sequential resets/replays to check reproducibility")
    result.add_argument("--reference-video", action="store_true", help="Also render recorded states for comparison")
    AppLauncher.add_app_launcher_args(result)
    return result


def read_tree(group):
    return {name: read_tree(item) if isinstance(item, h5py.Group) else item[:]
            for name, item in group.items()}


def adapter_checks(env):
    import torch
    from isaaclab.utils import math as pose

    errors = []
    for values in ([0, 0, 0, 0, 0, 0, 0], [0.02, -0.01, 0.03, 0.02, 0.03, -0.01, 1],
                   [-0.01, 0.02, 0.01, 1.2, -0.8, 0.2, 0]):
        action = torch.tensor([values], dtype=torch.float32, device=env.device)
        target = env.action_to_target_eef_pose(action)[env.eef_name]
        env.action_manager.process_action(action)
        controller = env._arm._ik_controller
        actual = pose.make_pose(controller.ee_pos_des, pose.matrix_from_quat(controller.ee_quat_des))
        torch.testing.assert_close(target, actual, atol=2e-6, rtol=2e-6)
        reconstructed = env.target_eef_pose_to_action(
            {env.eef_name: target[0]}, {env.eef_name: action[0, -1:]})
        torch.testing.assert_close(reconstructed, action[0], atol=3e-6, rtol=3e-6)
        errors.append(float((reconstructed - action[0]).abs().max()))
    for shape in ((4, 7), (2, 4, 7)):
        actions = torch.zeros(shape, device=env.device)
        actions[..., -1] = 1
        extracted = env.actions_to_gripper_actions(actions)[env.eef_name]
        assert extracted.shape == (*shape[:-1], 1) and (extracted == 1).all()
    # Check object coordinates against an independent homogeneous transform.
    root = pose.make_pose(env._robot.data.root_pos_w, pose.matrix_from_quat(env._robot.data.root_quat_w))
    for name, converted in env.get_object_poses().items():
        obj = env.scene[name]
        world = pose.make_pose(obj.data.root_pos_w, pose.matrix_from_quat(obj.data.root_quat_w))
        torch.testing.assert_close(root @ converted, world, atol=3e-6, rtol=3e-6)
    return {"controller_target_matches": True, "roundtrip_max_error": max(errors),
            "batched_gripper_shapes_pass": True, "object_frame_check_pass": True}


def quaternion_error_degrees(left, right):
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    return float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(left, right)), 0, 1))))


def label(image, title, step, count, success=None):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Unexpected RGB frame: {image.shape} {image.dtype}")
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Kit may override its render-product size. OpenCV silently drops frames
    # whose dimensions differ from VideoWriter's configured size.
    if frame.shape[:2] != (540, 960):
        frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 63), (20, 20, 20), -1)
    cv2.putText(frame, title, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    suffix = "" if success is None else f" | success={success}"
    cv2.putText(frame, f"Post-action {step + 1}/{count}{suffix}", (14, 49),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def writer(path, fps, width=960, height=540):
    stream = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not stream.isOpened():
        raise RuntimeError(f"Cannot create video {path}")
    return stream


def validate_video(path, expected_frames):
    video = cv2.VideoCapture(str(path))
    try:
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        available, frame = video.read()
        if not available or count != expected_frames:
            raise ValueError(f"Invalid video {path}: decoded={available}, frames={count}, expected={expected_frames}")
        return {"frames": count, "width": frame.shape[1], "height": frame.shape[0]}
    finally:
        video.release()


def run(args, report):
    import gymnasium as gym
    import torch
    from isaaclab.utils import math as pose

    import robolab.constants
    from robolab.core.replay.scene_state import overlay_state_row, restore_scene_state
    from robolab.datagen.envs.registration import source_review_config
    from robolab.policies.scripted.task_spec_extractors import extract_task_spec

    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
    robolab.constants.RECORD_IMAGE_DATA = False
    robolab.constants.VERBOSE = False
    source_audit = report["source_audit"]
    name, cfg, skipped = source_review_config(args.input, source_audit["collection"], args.physics_device)
    report["config_overlay_skipped"] = skipped
    report["task_intent"] = extract_task_spec(cfg, task_name=source_audit["collection"]["task"]).to_dict()
    report["physics_device"] = args.physics_device
    report["render_device"] = args.device
    report["coordinate_frame"] = "robot_root"
    report["control_dt"] = cfg.sim.dt * cfg.decimation
    with h5py.File(args.input, "r") as file:
        episode = file["data"][args.episode]
        actions = torch.as_tensor(episode["actions"][:], device=args.physics_device, dtype=torch.float32)
        initial = read_tree(episode["initial_state"])
        states = read_tree(episode["states"])
        recorded_ee_position = episode["ee_pose/position"][:]
        recorded_ee_quaternion = episode["ee_pose/orientation"][:]
        seed = int(episode.attrs.get("seed", 0))
    env = gym.make(name, cfg=cfg, render_mode="rgb_array").unwrapped
    try:
        def reset_source():
            state = env.scene.get_state(is_relative=True)
            missing = overlay_state_row(state, initial, num_envs=1, device=env.device)
            if missing:
                raise ValueError(f"Initial-state fields not restored: {missing}")
            env.reset_to(state, env_ids=None, seed=seed, is_relative=True)

        reset_source()
        report["adapter_checks"] = adapter_checks(env)
        report["trials"] = []
        for trial in range(args.trials):
            reset_source()
            for _ in range(12):
                env.render()
            video_path = args.output_dir / f"action_replay_{trial}.mp4"
            video = writer(video_path, 1 / report["control_dt"])
            metrics = []
            try:
                for step, action in enumerate(actions):
                    # Target reconstruction deliberately happens BEFORE action t.
                    target = env.action_to_target_eef_pose(action[None])[env.eef_name]
                    if not torch.isfinite(target).all():
                        raise ValueError(f"Nonfinite target at step {step}")
                    env.step(action[None])
                    success = bool(env.check_success()[0])
                    current = env.get_robot_eef_pose(env.eef_name)[0]
                    position = current[:3, 3].cpu().numpy()
                    quaternion = pose.quat_from_matrix(current[:3, :3]).cpu().numpy()
                    objects = {}
                    for object_name, trajectory in states["rigid_object"].items():
                        actual = env.scene[object_name].data.root_pos_w[0] - env.scene.env_origins[0]
                        objects[object_name] = float(np.linalg.norm(actual.cpu().numpy() - trajectory["root_pose"][step, :3]))
                    metrics.append({"step": step, "success": success,
                                    "eef_position_error_m": float(np.linalg.norm(position - recorded_ee_position[step])),
                                    "eef_rotation_error_deg": quaternion_error_degrees(quaternion, recorded_ee_quaternion[step]),
                                    "object_position_errors_m": objects})
                    # A headless scene with no camera sensors does not render
                    # in env.step(). Explicitly refresh the viewport annotator.
                    env.sim.render()
                    rgb = env.render(recompute=True)
                    if step == 0:
                        report["native_render_shape"] = list(rgb.shape)
                    frame = label(rgb, f"Action replay {trial + 1}: physics", step, len(actions), success)
                    video.write(frame)
                    if step in (0, 250, 400, len(actions) - 1):
                        cv2.imwrite(str(args.output_dir / f"action_{trial}_step{step:04d}.jpg"), frame)
                    if step % 100 == 0:
                        print(f"Replay {trial + 1}: {step + 1}/{len(actions)} success={success}", flush=True)
            finally:
                video.release()
            summary = {"trial": trial, "steps": len(metrics), "video": video_path.name,
                       "success_at_end": metrics[-1]["success"],
                       "first_success_step": next((row["step"] for row in metrics if row["success"]), None),
                       "max_eef_position_error_m": max(row["eef_position_error_m"] for row in metrics),
                       "max_eef_rotation_error_deg": max(row["eef_rotation_error_deg"] for row in metrics),
                       "max_object_position_errors_m": {
                           key: max(row["object_position_errors_m"][key] for row in metrics)
                           for key in metrics[0]["object_position_errors_m"]}}
            summary["video_validation"] = validate_video(video_path, len(metrics))
            report["trials"].append(summary)
            (args.output_dir / f"replay_metrics_{trial}.json").write_text(json.dumps(metrics, indent=2))
            print(json.dumps(summary, indent=2), flush=True)
        if args.reference_video:
            reset_source()
            comparison = writer(args.output_dir / "source_vs_replay.mp4", 1 / report["control_dt"], width=1920)
            replay = cv2.VideoCapture(str(args.output_dir / "action_replay_0.mp4"))
            try:
                for step in range(len(actions)):
                    missing = restore_scene_state(env, states, step)
                    if missing:
                        raise ValueError(f"Reference states not restored: {missing}")
                    env.sim.forward()
                    env.scene.update(dt=env.step_dt)
                    reference = label(env.render(), "Recorded states: visual reconstruction", step, len(actions))
                    available, action_frame = replay.read()
                    if not available:
                        raise ValueError(f"Replay video missing frame {step}")
                    comparison.write(np.concatenate((reference, action_frame), axis=1))
                    if step in (0, len(actions) // 2, len(actions) - 1):
                        cv2.imwrite(str(args.output_dir / f"comparison_{step:04d}.jpg"),
                                    np.concatenate((reference, action_frame), axis=1))
                report["comparison_video"] = "source_vs_replay.mp4"
            finally:
                replay.release()
                comparison.release()
            report["comparison_video_validation"] = validate_video(args.output_dir / "source_vs_replay.mp4", len(actions))
        report["status"] = "awaiting_source_review"
        report["next_gate"] = "User review of replay, drift, and source motion before annotation"
    finally:
        env.close()


def main():
    args = parser().parse_args()
    if args.trials < 1:
        raise ValueError("trials must be positive")
    source = audit_dataset(args.input)
    if not source.get("ready_for_replay"):
        raise ValueError(f"Source audit failed: {source}")
    if not any(ep["name"] == args.episode and ep["eligible_for_replay"] for ep in source["episodes"]):
        raise ValueError("Requested episode is not eligible for replay")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "initializing", "source_audit": source,
              "installed_versions": installed_versions(), "episode": args.episode,
              "code_sha256": {str(path): fingerprint(path) for path in [Path(__file__), *Path("robolab/datagen").rglob("*.py")]}}
    app = None
    try:
        from isaaclab.app import AppLauncher

        args.enable_cameras = True
        app = AppLauncher(args, multi_gpu=False).app
        import torch
        with torch.inference_mode():
            run(args, report)
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
        raise
    finally:
        (args.output_dir / "review.json").write_text(json.dumps(report, indent=2))
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
