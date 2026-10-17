#!/usr/bin/env python3
"""Run all-task OMY validation in bounded headless batches, without leader/XR.

The default GPU is 1. Results/HDF5 are test-only under omy_pipeline_benchmark.
No production simulator, calibration, SteamVR or system settings are modified.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.seeds) < 1 or args.gpu < 0:
        parser.error("batch-size, steps and seeds must be positive; GPU must be nonnegative")
    return args


def worker(args):
    # Start Kit before importing the test's simulator modules. pytest must not
    # import conftest and boot a second, GPU-0 SimulationApp.
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ["OMY_TEST_DEVICE"] = f"cuda:{args.gpu}"
    os.environ["OMY_BENCHMARK_OUTPUT"] = str(args.output_dir)
    os.environ["OMY_BENCHMARK_STEPS"] = str(args.steps)
    os.environ["OMY_BENCHMARK_SEEDS"] = str(args.seeds)
    os.environ["OMY_TEST_TASKS"] = json.dumps(args.tasks)
    import cv2  # noqa: F401 -- precedes AppLauncher
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        headless=True, device=f"cuda:{args.gpu}", enable_cameras=False,
        kit_args=f"--/app/settings/persistent=false --/renderer/activeGpu={args.gpu} "
                 "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false "
                 "--/plugins/carb.tasking.plugin/threadCount=4 "
                 "--/plugins/omni.tbb.globalcontrol/maxThreadCount=4 --/persistent/physics/numThreads=2",
    )
    import pytest

    try:
        status = pytest.main(["--noconftest", "-q", "-s", "tests/test_omy_benchmark_all.py"])
        (args.output_dir / "pytest_status.json").write_text(json.dumps({"exit_code": int(status)}) + "\n")
    finally:
        launcher.app.close()


def main(args):
    from robolab.core.task.task_index import _task_declarations

    output = (args.output_dir or ROOT / "output/omy_pipeline_benchmark" /
              datetime.now().strftime("%Y%m%d_%H%M%S_all_benchmark")).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for path in sorted((ROOT / "robolab/tasks/benchmark").glob("*.py")):
        if path.name == "__init__.py":
            continue
        contents = path.read_bytes()
        names = _task_declarations(contents, path)
        if len(names) != 1:
            raise ValueError(f"Expected one Task declaration in {path}: {names}")
        if args.tasks is None or names[0] in args.tasks:
            cases.append({"task": names[0], "source": str(path.relative_to(ROOT)),
                          "sha256": hashlib.sha256(contents).hexdigest()})
    if args.tasks is not None and {row["task"] for row in cases} != set(args.tasks):
        raise ValueError("Some --tasks are not benchmark tasks")
    (output / "inventory.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Output: {output}\nTasks: {len(cases)}; GPU: {args.gpu}; seeds: {args.seeds}; commands/seed: {args.steps}", flush=True)
    started = time.perf_counter()
    summaries = []
    for batch_index, offset in enumerate(range(0, len(cases), args.batch_size)):
        batch = cases[offset:offset + args.batch_size]
        directory = output / f"batch_{batch_index:03d}"
        directory.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--gpu", str(args.gpu),
                   "--output-dir", str(directory), "--steps", str(args.steps), "--seeds", str(args.seeds),
                   "--tasks", *[row["task"] for row in batch]]
        print(f"Batch {batch_index + 1}: {', '.join(row['task'] for row in batch)}", flush=True)
        timed_out = False
        with (directory / "worker.log").open("w") as log:
            try:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                exit_code, timed_out = -1, True
        for row in batch:
            result_path = directory / "tasks" / row["task"] / "validation.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {
                "task": row["task"], "passed": False,
                "error": f"Worker produced no task report (exit={exit_code}, timed_out={timed_out})",
            }
            result["report_path"] = str(result_path.relative_to(output))
            result["worker_log"] = str((directory / "worker.log").relative_to(output))
            summaries.append(result)
            print(f"  {row['task']}: {'PASS' if result['passed'] else 'FAIL'}; "
                  f"sensors={result.get('sensors_retained')}/{result.get('sensors_full')}; "
                  f"profile={result.get('sensor_profile')}", flush=True)
        summary = {"training_ready": False, "serial_port_opened": False, "xr_enabled": False,
                   "gpu": args.gpu, "seeds": args.seeds, "commands_per_seed": args.steps,
                   "task_count": len(cases), "completed_count": len(summaries),
                   "passed_count": sum(bool(row["passed"]) for row in summaries),
                   "elapsed_s": time.perf_counter() - started, "results": summaries}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Finished: {summary['passed_count']}/{len(cases)} passed; {output / 'summary.json'}", flush=True)
    return 0 if summary["passed_count"] == len(cases) else 1


if __name__ == "__main__":
    options = arguments()
    if options.worker:
        worker(options)
    else:
        sys.exit(main(options))
