#!/usr/bin/env python3
"""Validate observed settling with real drops, table placements, and airborne targets.

Runs selected single_pnp definitions with three parallel environments by default,
or one environment with --omy-profile, without a policy server or teleoperation
device. Writes diagnostics rather than training demonstrations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PILOT_TASKS = [
    "SimplePnp000BlueBlock00Task", "SimplePnp010Orange0200Task", "SimplePnp045MilkCarton00Task",
    "SimplePnp258Spatula0100Task", "SimplePnp082WoodenBowl00Task", "SimplePnp221ForkSmall00Task",
    "SimplePnp247PlateSmall00Task", "SimplePnp064ComputerMouse02Task",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", nargs="+", default=PILOT_TASKS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--keep-going", action="store_true", help="Collect all pilot traces even when a case fails")
    parser.add_argument("--omy-profile", action="store_true",
                        help="Use the production single-env OMY stepping, sensor and solver profile")
    parser.add_argument("--target-roll-deg", type=float, default=0,
                        help="Rotate the target before the diagnostic drop, to test side placements")
    parser.add_argument("--offset-xy", type=float, nargs=2, default=(0., 0.),
                        help="Drop offset from the bin center in meters")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= .5:
        parser.error("--seconds must be finite and greater than the settling interval")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import cv2  # noqa: F401 -- import before Isaac Lab
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        headless=True, device=args.device, enable_cameras=False,
        kit_args="--/app/settings/persistent=false --/renderer/multiGpu/enabled=false "
                 "--/plugins/carb.tasking.plugin/threadCount=8 "
                 "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8 "
                 "--/persistent/physics/numThreads=2",
    )
    report = {"passed": False, "training_ready": False, "tasks": []}
    try:
        import torch
        from isaaclab.utils.math import quat_mul
        import robolab.constants
        from robolab.core.environments.config import parse_env_cfg
        from robolab.core.environments.runtime import create_env
        from robolab.core.events.subtask_recorder import SubtaskCompletionRecorderTerm
        from robolab.core.omy_teleop.config import register_omy_envs
        from robolab.core.omy_teleop.config import configure_performance
        from robolab.core.omy_teleop.environment import OmyEnv
        from robolab.core.omy_teleop.options import build_parser as omy_parser
        from robolab.core.omy_teleop.options import ARM_JOINTS
        from robolab.core.task.conditionals import object_in_container, object_resting_in_container, resting_in_container_diagnostics
        from robolab.core.task.predicate_logic import gripper_detached, in_contact, in_opentop_container, stationary, upright
        from robolab.core.world.world_state import get_world

        robolab.constants.VERBOSE = False
        robolab.constants.DEBUG = False
        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True
        robolab.constants.RECORD_IMAGE_DATA = args.omy_profile

        for task in args.task:
            env = None
            row = {"task": task, "passed": False}
            report["tasks"].append(row)
            try:
                directory = args.output_dir / task
                directory.mkdir()
                robolab.constants.set_output_dir(str(directory.resolve()))
                register_omy_envs(task, task_dirs=["single_pnp"], env_postfix="RestingPilot")
                count = 1 if args.omy_profile else 3
                cfg = parse_env_cfg(task + "RestingPilot", device=args.device, num_envs=count)
                cfg.decimation = 8
                cfg.episode_length_s = args.seconds + 1
                if args.omy_profile:
                    options = omy_parser().parse_args([
                        "--source", "tcp", "--control-hz", "15", "--record", "--save-failures",
                        "--recording-profile", "compact", "--solver-position-iterations", "16",
                        "--stepping-profile", "held-target", "--render-interval", "4",
                        "--contact-sensors", "required", "--contact-history-length", "0",
                        "--episode-length-s", str(args.seconds + 1),
                    ])
                    configure_performance(cfg, options)
                term = cfg.terminations.success
                assert term.func is object_resting_in_container, "Select a migrated pilot definition"
                env, _ = create_env(cfg, policy="resting_pnp_diagnostic", rendering_mode="performance",
                                    env_class=OmyEnv if args.omy_profile else None)
                with torch.inference_mode():
                    env.reset(seed=0)
                    world = get_world(env)
                    target = term.params["object"]
                    body = env.scene[target]
                    if args.target_roll_deg:
                        rotated = body.data.root_pose_w.clone()
                        half = math.radians(args.target_roll_deg) / 2
                        rotation = torch.tensor([[math.cos(half), math.sin(half), 0., 0.]], device=env.device)
                        rotated[:, 3:] = quat_mul(rotation.expand(count, -1), rotated[:, 3:])
                        body.write_root_pose_to_sim(rotated)
                        env.scene.write_data_to_sim()
                        env.sim.forward()
                        env.scene.update(env.physics_dt)
                        world.clear_contact_cache()
                    bin_corners, bin_center = world.get_bbox("grey_bin")
                    target_corners, target_center = world.get_bbox(target)
                    bin_top = bin_corners[:, :, 2].amax(dim=1)
                    lower_offset = target_center[:, 2] - target_corners[:, :, 2].amin(dim=1)
                    desired = bin_center.clone()
                    desired[:, 2] = bin_top + lower_offset + .04
                    desired[:, :2] += torch.tensor(args.offset_xy, device=env.device)
                    if count > 1:
                        desired[1, 0] = .55
                        desired[1, 1] = .27 if bin_center[1, 1] < 0 else -.27
                        desired[1, 2] = target_center[1, 2]
                        desired[2, 2] += .4
                    pose = body.data.root_pose_w.clone()
                    pose[:, :3] += desired - target_center
                    body.write_root_pose_to_sim(pose)
                    body.write_root_velocity_to_sim(torch.zeros_like(body.data.root_vel_w))
                    env.scene.write_data_to_sim()
                    env.sim.forward()
                    env.scene.update(env.physics_dt)
                    world.clear_contact_cache()
                    world.reset_predicate_state(list(range(count)))
                    recorder = next(value for value in env.recorder_manager._terms.values()
                                    if isinstance(value, SubtaskCompletionRecorderTerm))
                    action = torch.tensor([[*[cfg.scene.robot.init_state.joint_pos[name] for name in ARM_JOINTS], 1.0]],
                                          device=env.device).repeat(count, 1)
                    first_legacy = first_success = None
                    trace = []
                    for _ in range(math.ceil(args.seconds / env.step_dt)):
                        # Keep env 2 airborne: it cannot acquire bin support or
                        # accumulate success merely by having a center above it.
                        if count > 1:
                            body.write_root_pose_to_sim(pose[2:3], env_ids=torch.tensor([2], device=env.device))
                            body.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device),
                                                            env_ids=torch.tensor([2], device=env.device))
                        env.step(action)
                        world.clear_contact_cache()
                        step = env.episode_length_buf.clone()
                        result = term.func(env, **term.params)
                        legacy = object_in_container(env, object=target, container="grey_bin",
                                                     require_contact_with=True, require_gripper_detached=True)
                        velocity = world.get_velocity(target)
                        checks = {
                            "inside": in_opentop_container(world, target, "grey_bin"),
                            "upright": upright(world, "grey_bin", tolerance=math.radians(15)),
                            "bin_stationary": stationary(world, "grey_bin", term.params["linear_threshold"], term.params["angular_threshold"]),
                            "target_stationary": stationary(world, target, term.params["linear_threshold"], term.params["angular_threshold"]),
                            "supported": world.is_supported_on_surface(target, "grey_bin", force_threshold=.01),
                            "detached": gripper_detached(world, target, "gripper"),
                            "table_contact": in_contact(world, target, "table", force_threshold=.01),
                        }
                        diagnostic = resting_in_container_diagnostics(env, term.params)
                        states = [v for k, v in world._predicate_state.items() if k.startswith("resting_in_container::")]
                        frame = {"step": step.tolist(), "time_s": (step * env.step_dt).tolist(),
                                 "legacy": legacy.tolist(), "resting": result.tolist(),
                                 "linear_speed": torch.linalg.vector_norm(velocity[:, :3], dim=1).tolist(),
                                 "angular_speed": torch.linalg.vector_norm(velocity[:, 3:], dim=1).tolist(),
                                 "bin_force": world.get_contact_force(target, "grey_bin").tolist(),
                                 "table_force": world.get_contact_force(target, "table").tolist(),
                                 "position": world.get_pose(target)[0].tolist(),
                                 "quaternion": world.get_pose(target)[1].tolist(),
                                 "bin_position": world.get_pose("grey_bin")[0].tolist(),
                                 "bin_quaternion": world.get_pose("grey_bin")[1].tolist(),
                                 "bin_velocity": world.get_velocity("grey_bin").tolist(),
                                 "checks": {key: value.tolist() for key, value in checks.items()},
                                 "timer_since": [state["since"].tolist() for state in states],
                                 "placement_diagnostic": diagnostic,
                                 "score": [info["score"] for info in recorder.infos]}
                        trace.append(frame)
                        assert not result[1:].any(), "Table/airborne control incorrectly succeeded"
                        assert all(info["score"] == 0 for info in recorder.infos[1:])
                        assert bool(result[0]) == (recorder.infos[0]["score"] == 1), "Score and termination disagree"
                        if legacy[0] and first_legacy is None:
                            first_legacy = int(step[0])
                        if result[0]:
                            first_success = int(step[0])
                            assert frame["linear_speed"][0] < term.params["linear_threshold"]
                            assert frame["angular_speed"][0] < term.params["angular_threshold"]
                            required = math.ceil(term.params["settle_time_s"] / env.step_dt - 1e-9)
                            tail = trace[-(required + 1):]
                            assert len(tail) == required + 1
                            assert all(f["linear_speed"][0] < term.params["linear_threshold"]
                                       and f["angular_speed"][0] < term.params["angular_threshold"] for f in tail)
                            for pos_key, quat_key in (("position", "quaternion"),
                                                      ("bin_position", "bin_quaternion")):
                                positions = torch.tensor([f[pos_key][0] for f in tail], dtype=torch.float64)
                                quats = torch.tensor([f[quat_key][0] for f in tail], dtype=torch.float64)
                                assert torch.linalg.vector_norm(positions - positions[0], dim=-1).max() <= term.params["position_tolerance"]
                                dots = (quats * quats[0]).sum(dim=-1).abs()
                                dots /= torch.linalg.vector_norm(quats, dim=-1) * torch.linalg.vector_norm(quats[0])
                                assert dots.min() >= math.cos(math.radians(term.params["rotation_tolerance_deg"]) / 2)
                            assert env.get_env_results()[0]["success"] is True
                            break
                    (directory / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
                    row.update(target=target, first_legacy_step=first_legacy, success_step=first_success,
                               step_dt=env.step_dt, elapsed_s=None if first_success is None else first_success * env.step_dt,
                               negative_controls_passed=True if count > 1 else None, final=trace[-1],
                               omy_profile=args.omy_profile)
                    assert first_success is not None, f"{target} never settled successfully; inspect trace.json"
                    row["passed"] = True
                    print("PILOT_RESULT " + json.dumps(row), flush=True)
            except Exception as error:
                row["error"] = str(error)
                print("PILOT_FAILURE " + json.dumps(row), flush=True)
                if not args.keep_going:
                    raise
            finally:
                if env is not None:
                    env.close()
                (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        report["passed"] = all(row["passed"] for row in report["tasks"])
    finally:
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        launcher.app.close()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
