#!/usr/bin/env python3
"""Collect demonstrations with the generic WorldState scripted expert.

Examples:
    python examples/collect_scripted_demos.py --task BananaInBowlTask --num-episodes 20 --headless
    python examples/collect_scripted_demos.py --task KeyboardOutOfBinTask --dry-run-spec --headless

The HDF5 ``actions`` channel contains the submitted absolute-IK commands
``[x, y, z, qw, qx, qy, qz, gripper]``. It does not contain resulting joint targets.
"""

# isort: skip_file
import argparse
from collections import Counter
import json
import math
import os
import sys
import traceback
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

import cv2  # Must be imported before isaaclab. Do not remove.  # noqa: F401
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect config-driven scripted absolute-IK demonstrations.")
parser.add_argument("--task", required=True, help="Registered RoboLab task class name.")
parser.add_argument("--task-spec", help="YAML/JSON normalized task-spec override.")
parser.add_argument("--grasp-profiles", help="YAML/JSON optional semantic grasp profiles.")
parser.add_argument("--num-episodes", type=int, default=20, help="Total attempted episodes.")
parser.add_argument("--num-envs", type=int, default=1, help="Parallel environments per batch.")
parser.add_argument("--output-dir", required=True, help="Recorder output directory.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-grasp-attempts", type=int, default=4)
parser.add_argument("--save-failures", action="store_true", help="Keep failed/truncated HDF5 demos too.")
parser.add_argument("--visualize-world-state", action="store_true", help="Draw required entity OBBs (GUI only).")
parser.add_argument("--dry-run-spec", action="store_true", help="Print normalized spec without creating an env.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if args_cli.num_episodes < 1 or args_cli.num_envs < 1:
    parser.error("--num-episodes and --num-envs must be positive")
args_cli.enable_cameras = not args_cli.dry_run_spec
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import robolab.constants  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from robolab.core.environments.config import parse_env_cfg  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env, end_episode  # noqa: E402
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.policies.scripted.policy import ScriptedExpertPolicy, ScriptedPolicyConfig  # noqa: E402
from robolab.policies.scripted.profiles import GraspProfileLibrary  # noqa: E402
from robolab.policies.scripted.state_machine import StateMachineConfig  # noqa: E402
from robolab.policies.scripted.task_spec_extractors import load_or_extract_task_spec  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs  # noqa: E402


def _resolve() -> tuple[str, object, object, GraspProfileLibrary]:
    robolab.constants.set_output_dir(os.path.abspath(args_cli.output_dir))
    robolab.constants.RECORD_IMAGE_DATA = not args_cli.dry_run_spec
    robolab.constants.VERBOSE = False
    auto_register_droid_abs_ik_envs(task=args_cli.task)
    env_names = get_envs(task=args_cli.task)
    if not env_names:
        raise ValueError(f"No registered absolute-IK environment found for task '{args_cli.task}'")
    env_name = env_names[0]
    cfg = parse_env_cfg(env_name, device=args_cli.device, seed=args_cli.seed, num_envs=args_cli.num_envs)
    spec = load_or_extract_task_spec(cfg, args_cli.task_spec, args_cli.task)
    contact_entities = set(getattr(cfg, "contact_object_list", []) or []) | {"robot"}
    spec.validate_entities(contact_entities)
    profiles = GraspProfileLibrary.load(args_cli.grasp_profiles)
    return env_name, cfg, spec, profiles


def _print_dry_run(env_name, cfg, spec, profiles) -> None:
    selected = {}
    for target in spec.targets:
        profile = profiles.profiles.get(target.object_name)
        selected[target.object_name] = profile.to_dict() if profile else {
            "selection": "geometry-derived at runtime",
            "reason": "dimensions require live WorldState geometry",
        }
    print(json.dumps({
        "requested_task": args_cli.task,
        "registered_environment": env_name,
        "spec": spec.to_dict(),
        "required_entities": list(spec.required_entities),
        "available_task_entities": list(getattr(cfg, "contact_object_list", []) or []),
        "selected_profiles": selected,
    }, indent=2))


def _configure_recorder(env) -> None:
    mode = DatasetExportMode.EXPORT_ALL if args_cli.save_failures else DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env.recorder_manager.cfg.dataset_export_mode = mode
    env.recorder_manager.set_hdf5_file("scripted_demos.hdf5")


def _finalize_policy_failures(env, policy, active_ids: list[int]) -> None:
    """Freeze exhausted policies and finalize their recordings as failures."""
    failed = set(policy.failed_env_ids())
    for env_id in active_ids:
        if env_id not in failed or bool(env._frozen_envs[env_id]):
            continue
        env.recorder_manager.set_success_to_episodes(
            [env_id], torch.zeros(1, dtype=torch.bool, device=env.device)
        )
        env.recorder_manager.export_episodes(env_ids=[env_id])
        env._env_results[env_id] = False
        env._env_term_step[env_id] = int(env.episode_length_buf[env_id].item())
        env._frozen_envs[env_id] = True


def _episode_result(env, policy, env_id: int, attempt: int) -> dict:
    runtime = env.get_env_results()[env_id]
    policy_result = policy.result(env_id)
    success = bool(runtime["success"])
    reason = None if success else (policy_result["failure_reason"] or "TASK_NOT_TERMINATED")
    return {
        "attempt": attempt,
        "env_id": env_id,
        "success": success,
        "state": policy_result["state"],
        "steps": runtime["step"],
        "grasp_attempts": policy_result["grasp_attempts"],
        "failure_reason": reason,
        "failure_detail": policy_result["failure_detail"],
        "grasp_candidates": policy_result["grasp_candidates"],
    }


def main() -> int:
    env_name, cfg, spec, profiles = _resolve()
    if args_cli.dry_run_spec:
        _print_dry_run(env_name, cfg, spec, profiles)
        return 0

    env = None
    results: list[dict] = []
    try:
        # create_env parses its own fresh cfg; extraction above deliberately happened before simulator creation.
        env, env_cfg = create_env(
            env_name, device=args_cli.device, seed=args_cli.seed,
            num_envs=args_cli.num_envs, use_fabric=True, policy="scripted_abs_ik",
        )
        _configure_recorder(env)
        state_cfg = StateMachineConfig(max_grasp_attempts=args_cli.max_grasp_attempts)
        policy = ScriptedExpertPolicy(env, spec, profiles=profiles, cfg=ScriptedPolicyConfig(state_machine=state_cfg))
        batches = math.ceil(args_cli.num_episodes / args_cli.num_envs)
        attempt = 0
        for batch in range(batches):
            if batch:
                env.reset_eval_state()
            env.reset()
            policy.reset()
            active_count = min(args_cli.num_envs, args_cli.num_episodes - attempt)
            active_ids = list(range(active_count))
            inactive_ids = list(range(active_count, args_cli.num_envs))
            if inactive_ids:
                env._frozen_envs[inactive_ids] = True
            for env_id in active_ids:
                env.recorder_manager.set_episode_index(attempt + env_id, env_ids=[env_id])

            if args_cli.visualize_world_state and not args_cli.headless:
                get_world(env).visualize(list(spec.required_entities), env_id=0)

            max_steps = int(getattr(env, "max_episode_length", 1000)) + 2
            for _ in range(max_steps):
                actions = policy.act()
                _, _, terminated, truncated, info = env.step(actions)
                policy.observe_step(terminated, truncated, info)
                _finalize_policy_failures(env, policy, active_ids)
                if env.all_terminated:
                    break

            for env_id in active_ids:
                result = _episode_result(env, policy, env_id, attempt + env_id)
                results.append(result)
                print(json.dumps(result, sort_keys=True))
            attempt += active_count

        successes = sum(result["success"] for result in results)
        failures = Counter(result["failure_reason"] for result in results if not result["success"])
        mean_steps = sum((result["steps"] or 0) for result in results) / len(results)
        mean_grasps = sum(result["grasp_attempts"] for result in results) / len(results)
        summary = {
            "task": args_cli.task,
            "successes": successes,
            "attempts": len(results),
            "success_rate": successes / len(results),
            "mean_steps": mean_steps,
            "mean_grasp_attempts": mean_grasps,
            "failure_counts": dict(failures),
            "recording": os.path.join(os.path.abspath(args_cli.output_dir), "scripted_demos.hdf5"),
            "recorded_action_representation": "absolute IK command in robot-root frame, not joint targets",
        }
        results_path = os.path.join(os.path.abspath(args_cli.output_dir), "scripted_results.json")
        with open(results_path, "w", encoding="utf-8") as stream:
            json.dump({"summary": summary, "episodes": results}, stream, indent=2)
        print(json.dumps(summary, indent=2))
        return 0 if successes else 2
    finally:
        if env is not None:
            end_episode(env)
            env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception as exc:
        print(f"Scripted collection failed: {exc}", file=sys.stderr)
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
    raise SystemExit(exit_code)

