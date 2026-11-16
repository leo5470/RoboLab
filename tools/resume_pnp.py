"""Resume the original PnP shards in isolated, memory-limited user services."""

import argparse
import collections
import fnmatch
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "output/pi05_fresh_5eps_logs"
OPENPI = ROOT.parent / "openpi"


def pending_tasks(gpu, shard_gpu=None, workers=1, worker_slot=0):
    """Pending source-shard tasks assigned to this destination worker.

    ``gpu`` selects the evaluator/output shard. ``shard_gpu`` can move work
    from a finished shard to another GPU; fixed manifest-index partitioning
    keeps concurrently launched workers disjoint as results accumulate.
    """
    if workers < 1 or not 0 <= worker_slot < workers:
        raise ValueError("worker_slot must be in [0, workers)")
    shard_gpu = gpu if shard_gpu is None else shard_gpu
    directory = ROOT / "robolab/tasks/single_pnp"
    manifest = json.loads((directory / "manifest.json").read_text())["tasks"]
    patterns = json.loads((directory / "exclusions.json").read_text())["target_patterns"]
    assigned = manifest[:3600] if shard_gpu == 0 else manifest[3600:]
    counts = collections.defaultdict(set)
    # A worker redistributing source tasks writes to its destination result
    # file. Read both files so a restarted worker will not repeat completed work.
    result_gpus = {0, 1}
    for result_gpu in result_gpus:
        results = ROOT / f"output/pi05_simple_pnp_5eps_gpu{result_gpu}/episode_results.jsonl"
        if results.exists():
            with results.open() as stream:
                for line in stream:
                    record = json.loads(line)
                    counts[record["env_name"]].add(record["episode"])
    return [
        task["class_name"] for index, task in enumerate(assigned)
        if index % workers == worker_slot
        if not any(fnmatch.fnmatchcase(task["target"], pattern) for pattern in patterns)
        and len(counts[task["class_name"]]) < 5
    ]


def memory_sample():
    relative = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
    directory = Path("/sys/fs/cgroup") / relative.lstrip("/")
    sample = {"time": time.time()}
    for filename in ("memory.current", "memory.peak", "memory.swap.current", "memory.events"):
        path = directory / filename
        if path.exists():
            sample[filename] = path.read_text().strip()
    return sample


def run_worker(gpu, batch_size, shard_gpu=None, workers=1, worker_slot=0):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        pending = pending_tasks(gpu, shard_gpu, workers, worker_slot)
        if not pending:
            return
        batch = pending[:batch_size]
        command = [
            str(ROOT / ".venv/bin/python"), str(ROOT / "policies/pi0_family/run.py"),
            "--policy", "pi05", "--headless", "--device", "cuda:0",
            "--remote-port", str(8000 + gpu), "--task-dirs", "single_pnp",
            "--num-envs", "5", "--num-runs", "1", "--video-mode", "none",
            "--output-folder-name", f"pi05_simple_pnp_5eps_gpu{gpu}", "--task", *batch,
        ]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMNI_KIT_ACCEPT_EULA="YES")
        with (LOG_DIR / f"eval_gpu{gpu}.log").open("ab") as output:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=output,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            with (LOG_DIR / f"memory_gpu{gpu}.jsonl").open("a", buffering=1) as telemetry:
                while True:
                    telemetry.write(json.dumps(dict(memory_sample(), evaluator_pid=process.pid)) + "\n")
                    try:
                        returncode = process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        pass
        if returncode:
            raise RuntimeError(f"GPU {gpu} evaluator exited with {returncode}; see eval_gpu{gpu}.log")
        unfinished = set(batch) & set(pending_tasks(gpu, shard_gpu, workers, worker_slot))
        if unfinished:
            raise RuntimeError(f"GPU {gpu} batch exited without completing {len(unfinished)} tasks")


def service_command(gpu, kind, batch_size, shard_gpu=None, workers=1, worker_slot=0):
    unit = f"robolab-pnp-{kind}-gpu{gpu}"
    command = [
        "systemd-run", "--user", f"--unit={unit}", "--collect",
        "--property=MemoryAccounting=yes", "--property=OOMPolicy=stop",
        "--property=KillMode=control-group", "--property=TimeoutStopSec=30",
        f"--setenv=CUDA_VISIBLE_DEVICES={gpu}",
    ]
    if kind == "server":
        command += [
            "--property=MemoryHigh=56G", "--property=MemoryMax=64G",
            "--property=MemorySwapMax=256M", f"--working-directory={OPENPI}",
            "--setenv=XLA_PYTHON_CLIENT_MEM_FRACTION=0.5",
            str(OPENPI / ".venv/bin/python"), str(OPENPI / "scripts/serve_policy.py"),
            "--port", str(8000 + gpu), "policy:checkpoint",
            "--policy.config", "pi05_droid_jointpos",
            "--policy.dir", "gs://openpi-assets-simeval/pi05_droid_jointpos",
        ]
    else:
        command += [
            "--property=MemoryHigh=56G", "--property=MemoryMax=64G",
            f"--property=BindsTo=robolab-pnp-server-gpu{gpu}.service",
            f"--property=After=robolab-pnp-server-gpu{gpu}.service",
            "--property=MemorySwapMax=256M", f"--working-directory={ROOT}",
            sys.executable, str(Path(__file__).resolve()),
            "--worker", "--gpu", str(gpu), "--batch-size", str(batch_size),
        ]
        if shard_gpu is not None:
            command += ["--shard-gpu", str(shard_gpu), "--workers", str(workers),
                        "--worker-slot", str(worker_slot)]
    return command


def check_no_existing_jobs():
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            arguments = directory.joinpath("cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for argument in arguments:
            if argument in (
                str(ROOT / "policies/pi0_family/run.py").encode(),
            ):
                raise RuntimeError(f"Existing evaluator PID {directory.name}; refusing duplicate launch")


def service_is_active(unit):
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit], check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--gpu", choices=(0, 1), type=int)
    parser.add_argument("--shard-gpu", choices=(0, 1), type=int,
                        help="Read tasks/results from this original shard; defaults to --gpu.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Stable modulo partition count when redistributing a shard.")
    parser.add_argument("--worker-slot", type=int, default=0,
                        help="This worker's stable partition index in [0, workers).")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.workers < 1 or not 0 <= args.worker_slot < args.workers:
        parser.error("--worker-slot must be in [0, workers)")
    if args.worker:
        if args.gpu is None or args.dry_run:
            parser.error("--worker requires --gpu and cannot use --dry-run")
        run_worker(args.gpu, args.batch_size, args.shard_gpu, args.workers, args.worker_slot)
        return
    selected = (0, 1) if args.gpu is None else (args.gpu,)
    assignments = {gpu: pending_tasks(gpu, args.shard_gpu, args.workers, args.worker_slot)
                   for gpu in selected}
    if not args.dry_run:
        check_no_existing_jobs()
    for gpu, tasks in assignments.items():
        print(f"GPU {gpu}: {len(tasks)} unfinished tasks, {args.batch_size} tasks per Isaac process", flush=True)
        if not tasks:
            continue
        for kind in ("server", "eval"):
            if (not args.dry_run and kind == "server"
                    and service_is_active(f"robolab-pnp-server-gpu{gpu}.service")):
                print(f"GPU {gpu}: reusing active policy server", flush=True)
                continue
            command = service_command(gpu, kind, args.batch_size, args.shard_gpu,
                                      args.workers, args.worker_slot)
            if args.dry_run:
                print(shlex.join(command))
            else:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
