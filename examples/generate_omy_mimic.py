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


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
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
