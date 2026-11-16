#!/usr/bin/env python3
"""Expand L100 teleoperation demonstrations with MimicGen; see docs/omy_mimicgen.md."""

# isort: skip_file
import sys
from pathlib import Path
import argparse
import subprocess
import signal
import os
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
bootstrap = argparse.ArgumentParser(add_help=False)
bootstrap.add_argument("--host-gpu", type=int)
host, remaining = bootstrap.parse_known_args()
if host.host_gpu is not None and "--help" not in remaining and "-h" not in remaining:
    from robolab.core.omy_postprocess.gpu_isolation import isolated_gpu_launch
    from robolab.core.omy_postprocess.render import _gpu_slot

    gpu = isolated_gpu_launch(f"cuda:{host.host_gpu}")
    with _gpu_slot(f"cuda:{host.host_gpu}"):
        raise SystemExit(subprocess.call(
            [*gpu.prefix, sys.executable, "-B", str(Path(__file__).resolve()), *remaining,
             "--device", "cuda:0"], env=gpu.env))

import cv2  # noqa: F401 -- must precede Isaac Lab imports
from isaaclab.app import AppLauncher

from robolab.core.omy_teleop.mimicgen import build_parser, validate_args

# Multisource pooling is deprecated; OMY datasets use one human source per generation
# (scripts/omy_dataset_pipeline.py). Marked here so mimicgen.py, whose hash stamps source
# qualifications, stays unchanged.
# --selection-seed is kept: it also seeds single-source segment boundaries and trajectories.
DEPRECATED = ("source_selection", "selection_strategy", "nn_k", "fixed_source_index", "trial_manifest",
              "follow_source_schedule")


def mark_deprecated(parser):
    for action in parser._actions:
        if action.dest in DEPRECATED:
            action.help = f"DEPRECATED (multisource). {action.help or ''}".strip()
    return parser


def warn_deprecated(parser, args):
    used = [f"--{name.replace('_', '-')}" for name in DEPRECATED if getattr(args, name) != parser.get_default(name)]
    if len(args.input) > 1 or (args.episodes and len(args.episodes) > 1):
        used.append("several source demonstrations")
    if used:
        print(f"DEPRECATED: multisource generation ({', '.join(used)}); generate from one source "
              "demonstration with a single --input and --episodes N.", file=sys.stderr, flush=True)
    elif args.episodes is None:
        print("Note: without --episodes every successful episode is pooled, which is deprecated "
              "multisource generation; pass --episodes N for single-source generation.", file=sys.stderr, flush=True)


def main():
    parser = mark_deprecated(build_parser())
    args = parser.parse_args()
    validate_args(parser, args)
    warn_deprecated(parser, args)
    app = AppLauncher(headless=True, device=args.device, fast_shutdown=False, kit_args=(
        "--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0 "
        "--/plugins/carb.tasking.plugin/threadCount=4 "
        "--/plugins/omni.tbb.globalcontrol/maxThreadCount=4 "
        "--/persistent/physics/numThreads=2 --/app/settings/persistent=false "
        "--/app/runLoops/main/rateLimitEnabled=false --/app/fastShutdown=false")).app
    signal.signal(signal.SIGINT, signal.default_int_handler)
    status = 1
    try:
        from robolab.core.omy_teleop.mimicgen import run

        status = run(args)
    except KeyboardInterrupt:
        status = 130
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if args.worker_exit:
            # run() explicitly closes its recording and environment in finally.
            # The parent waits for process exit and independently validates artifacts.
            # Isaac Sim 5.0's subsequent plugin teardown segfaults on this stack.
            os._exit(status)
        print("MimicGen closing simulation application.", flush=True)
        app.close(wait_for_replicator=False)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
