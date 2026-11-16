"""Isolated five-environment action replay; does not contact a policy server."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SimplePnp045MilkCarton17Task")
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--modes", nargs="+", default=[
    "baseline", "baseline_repeat", "optimized", "no_viewport", "no_allocator_cleanup",
])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args)
application = launcher.app

import h5py
import numpy as np
import torch

import robolab.constants
from policies.pi0_family.client import Pi0DroidJointposClient
from robolab.core.environments import runtime
from robolab.core.logging.results import init_experiment
from robolab.eval.base_client import InferenceClient
from robolab.eval.episode import run_episode
from robolab.eval.summarize import summarize_run
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
from tools.pnp_perf_candidate import headless_infer


def memory():
    status = Path("/proc/self/status").read_text().splitlines()
    rss_kib = int(next(line for line in status if line.startswith("VmRSS:")).split()[1])
    return {
        "rss_bytes": rss_kib * 1024,
        "cuda_allocated": torch.cuda.memory_allocated(),
        "cuda_reserved": torch.cuda.memory_reserved(),
    }


class ReplayClient(Pi0DroidJointposClient):
    def __init__(self, actions, optimized):
        InferenceClient.__init__(self)
        self.open_loop_horizon = 15
        self.actions = actions
        self.optimized = optimized
        self.steps = {env_id: 0 for env_id in actions}
        self.request_trace = []
        self.action_trace = []
        self.images = {}
        self.memory_trace = []
        self.context = None

    def infer(self, obs, instruction, *, env_id=0):
        step = self.steps[env_id]
        self.context = (env_id, step)
        implementation = headless_infer if self.optimized else InferenceClient.infer
        result = implementation(self, obs, instruction, env_id=env_id)
        self.action_trace.append([env_id, step, result["action"].tolist()])
        if env_id == 0 and step % 100 == 0:
            self.memory_trace.append(dict(memory(), step=step))
        self.steps[env_id] += 1
        return result

    def _query_server(self, request):
        env_id, step = self.context
        hashes = {}
        for name, value in request.items():
            if isinstance(value, np.ndarray):
                hashes[name] = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            else:
                hashes[name] = value
        self.request_trace.append({"env_id": env_id, "step": step, "hashes": hashes})
        if step in (0, 150, 450, 735):
            for name in ("observation/exterior_image_1_left", "observation/wrist_image_left"):
                self.images[f"{env_id}_{step}_{name.split('/')[-1]}"] = request[name].copy()
        indices = np.minimum(np.arange(step, step + 15), len(self.actions[env_id]) - 1)
        return {"actions": self.actions[env_id][indices].copy()}

    def close(self):
        self.reset()


def main():
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True
    robolab.constants.RECORD_IMAGE_DATA = False
    auto_register_droid_envs(task_dirs=["single_pnp"], task=[args.task])
    with h5py.File(args.source) as source:
        actions = {env_id: source[f"data/demo_{env_id}/actions"][:]
                   for env_id in range(5)}
    original_parser = runtime.parse_env_cfg
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for mode in args.modes:
        if mode not in ("baseline", "baseline_repeat", "optimized", "no_viewport", "no_allocator_cleanup"):
            raise ValueError(mode)
        output = args.output / mode
        if output.exists():
            raise RuntimeError(f"Refusing to overwrite {output}")
        output.mkdir()
        robolab.constants.set_output_dir(str(output))
        result_file, results = init_experiment(str(output))
        print(f"VALIDATION_START {mode}", flush=True)
        started = time.perf_counter()

        def parse_config(*config_args, **config_kwargs):
            config = original_parser(*config_args, **config_kwargs)
            if mode == "no_viewport":
                config.scene.egocentric_mirrored_camera = None
                config.observations.viewport_cam = None
            return config

        runtime.parse_env_cfg = parse_config
        environment, config = runtime.create_env(
            args.task, device=args.device, num_envs=5, seed=0,
            instruction_type="default", policy="pi05",
        )
        runtime.parse_env_cfg = original_parser
        created = time.perf_counter()
        optimized = mode in ("optimized", "no_viewport", "no_allocator_cleanup")
        client = ReplayClient(actions, optimized)
        if mode == "no_allocator_cleanup":
            environment.recorder_manager.set_flush_memory_cleanup(False)
        try:
            torch.cuda.synchronize()
            episode_started = time.perf_counter()
            env_results, messages, timing = run_episode(
                environment, config, 0, client,
                headless=True, save_videos=False, video_mode="none",
                collect_subtask_history=not optimized,
            )
            torch.cuda.synchronize()
            episode_finished = time.perf_counter()
            results = summarize_run(
                env_results=env_results, msgs=messages, env=environment,
                env_cfg=config, num_envs=5, run_idx=0, run_name=f"{args.task}_0",
                task_env=args.task, scene_output_dir=str(output), policy="pi05",
                episode_results=results, episode_results_file=result_file,
                enable_subtask_progress=True, timing=timing, instruction_type="default",
            )
            report = {
                "mode": mode, "task": args.task, "source": str(args.source),
                "create_s": created - started,
                "episode_elapsed_s": episode_finished - episode_started,
                "timing": timing, "results": results, "history_entries": len(messages),
                "memory": memory(), "memory_trace": client.memory_trace,
                "request_count": len(client.request_trace),
            }
        finally:
            client.close()
            environment.close()
        report["task_elapsed_s"] = time.perf_counter() - started
        (output / "report.json").write_text(json.dumps(report, indent=2))
        (output / "requests.json").write_text(json.dumps(client.request_trace))
        (output / "actions.json").write_text(json.dumps(client.action_trace))
        np.savez_compressed(output / "policy_images.npz", **client.images)
        reports.append(report)
        (args.output / "reports.json").write_text(json.dumps(reports, indent=2))
        print("VALIDATION_DONE " + json.dumps({key: report[key] for key in (
            "mode", "episode_elapsed_s", "task_elapsed_s", "memory", "request_count",
        )}), flush=True)
        del client, environment, config, messages, results


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    application.close()
    sys.exit(1)
else:
    application.close()
