#!/usr/bin/env python3
"""Evaluate trained PPS proxy pairs on an object's Simple-PnP v2 layouts, paired with the base policy.

One PPS server (default mode none, one --arm per pair) answers every run, so the base policy and
every steered arm go through the same code path, client flags and noise keys, and layout i of one
run can be compared with layout i of another. The runs are

    base            gamma 0 (taken from --base-run when that names a finished run of these layouts)
    PAIR x GAMMA    --mode pps, for every --pair and every gamma

each with one episode per layout, no video, written to output/<prefix>_<run>_<tag>/. A run that is
already complete is skipped and an interrupted one resumes, so the same command can be repeated.
The paired report (analyse_proxy_pairs.py) is written to output/<prefix>_eval_<tag>/ at the end.

    scripts/pps_single_stage/eval_proxy_pairs.py --object utilityjug_a01 \
        --pair T50=/path/SimplePnp283UtilityjugA01_T50 --pair T200=/path/SimplePnp283UtilityjugA01_T200 \
        --gammas 0.4 0.8

A pair folder holds v_ref/ and v_task/, each with final.pt, manifest.json and tracking.json, as the
training job writes them. See docs/pps_proxy_evaluation_howto.md.
"""

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "robolab/tasks/single_pnp_v2"
SERVER = REPO / "scripts/pps_single_stage/start_pps_server.sh"
ANALYSE = REPO / "scripts/pps_single_stage/analyse_proxy_pairs.py"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def object_tasks(name, folder=TASKS):
    """(task number, [v2 task class per layout, in layout order]) of an object such as ``utilityjug_a01``."""
    pattern = re.compile(rf"^pnp_(\d+)_{re.escape(name)}_(\d+)_task\.py$")
    found = sorted((int(m.group(2)), m.group(1), path) for path in Path(folder).iterdir()
                   if (m := pattern.match(path.name)))
    if not found:
        raise SystemExit(f"No v2 tasks for object {name!r} in {folder}")
    classes = []
    for _, _, path in found:
        match = re.search(r"^class (\w+)\(Task\)", path.read_text(), re.M)
        if not match:
            raise SystemExit(f"No task class in {path}")
        classes.append(match.group(1))
    return found[0][1], classes


def parse_layouts(text, count):
    """Indices from ``0-49`` or ``0,3,10-12``."""
    picked = []
    for part in text.split(","):
        low, _, high = part.partition("-")
        picked += range(int(low), int(high or low) + 1)
    if not picked or min(picked) < 0 or max(picked) >= count or len(set(picked)) != len(picked):
        raise SystemExit(f"--layouts {text!r} must name distinct layouts between 0 and {count - 1}")
    return picked


def pair_info(name, folder, verify=True):
    """Paths, hashes and training contract of one pair; checks each file against what training recorded."""
    folder, info = Path(folder).resolve(), {"name": name}
    for part in ("v_ref", "v_task"):
        checkpoint = folder / part / "final.pt"
        if not checkpoint.is_file():
            raise SystemExit(f"{name}: missing {checkpoint}")
        info[part] = str(checkpoint)
        tracking = folder / part / "tracking.json"
        recorded = json.loads(tracking.read_text()).get("final_sha256") if tracking.is_file() else None
        if verify:
            info[f"{part}_sha256"] = actual = sha256(checkpoint)
            if recorded and recorded != actual:
                raise SystemExit(f"{name}: {checkpoint} does not match the hash recorded at training time")
            if not recorded:
                print(f"{name}: no recorded hash for {part} (no tracking.json); not verified", flush=True)
    manifest = folder / "v_task" / "manifest.json"
    if manifest.is_file():
        meta = json.loads(manifest.read_text()).get("meta", {})
        contract = meta.get("contract", {})
        info.update(task=contract.get("task"), prompt=contract.get("prompt"),
                    dataset_digest=contract.get("dataset_digest"), steps=meta.get("recipe", {}).get("steps"),
                    dataset_rows=meta.get("recipe", {}).get("dataset_size"))
        parent = meta.get("reference_parent_sha256")
        if verify and parent and parent != info["v_ref_sha256"]:
            raise SystemExit(f"{name}: v_task was not trained from this v_ref")
    return info


def gpu_free_mib(index):
    out = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=memory.total,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    total, used = (int(v) for v in out.strip().split(","))
    return total - used


def wait_for_memory(index, need_mib, what, wait=True):
    told = False
    while gpu_free_mib(index) < need_mib:
        if not wait:
            raise SystemExit(f"GPU {index} has under {need_mib} MiB free for {what}")
        if not told:
            print(f"waiting: GPU {index} has under {need_mib} MiB free for {what}", flush=True)
            told = True
        time.sleep(60)


def gamma_tag(gamma):
    return "g" + f"{gamma:g}".replace(".", "")


def finished(folder, tasks):
    """True when the run folder holds an episode for every layout."""
    path = Path(folder) / "episode_results.jsonl"
    if not path.is_file():
        return False
    done = {json.loads(line)["task_name"] for line in path.read_text().splitlines() if line.strip()}
    return set(tasks) <= done


def client_command(args, tasks, label, folder, arm=None, gamma=0.0):
    steer = ["--mode", "pps", "--gamma", f"{gamma:g}", "--arm", arm] if arm else ["--mode", "none", "--gamma", "0.0"]
    return [str(REPO / ".venv/bin/python"), "policies/pps/run.py", "--remote-host", "localhost",
            "--remote-port", str(args.port), *steer, "--headless", "--device", f"cuda:{args.client_gpu}",
            "--num-envs", "1", "--num-runs", "1", "--video-mode", "none", "--task-dirs", "single_pnp_v2",
            "--pps-task", label, "--output-folder-name", folder, "--task", *tasks]


def port_open(port):
    with socket.socket() as probe:
        probe.settimeout(2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", required=True, help="Object name as in the v2 task files, e.g. utilityjug_a01")
    parser.add_argument("--pair", action="append", default=[], metavar="NAME=DIR", required=True,
                        help="A trained pair: a short arm name and its folder (v_ref/, v_task/); repeat")
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.4], help="Steering strengths (default 0.4)")
    parser.add_argument("--layouts", default=None, help="Layout indices, e.g. 0-49 (default: all) or 0,3,7")
    parser.add_argument("--tag", default=date.today().strftime("%Y%m%d"), help="Suffix of the output folders")
    parser.add_argument("--prefix", default=None, help="Output folder prefix (default pps_v2_<object>)")
    parser.add_argument("--base-run", type=Path, default=None,
                        help="Finished base-policy run of the same layouts to pair with, made with the same "
                             "--pps-task label and flags under the current success rule; skips the base run")
    parser.add_argument("--pps-task", default=None, help="Noise-key label (default: the pairs' training task)")
    parser.add_argument("--port", type=int, default=8018)
    parser.add_argument("--server-gpu", type=int, default=1)
    parser.add_argument("--client-gpu", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=3, help="Simulations at once on the client GPU (default 3)")
    parser.add_argument("--stagger-s", type=float, default=90.0, help="Delay between simulation starts")
    parser.add_argument("--server-mib", type=int, default=9500, help="Free memory the server needs on its GPU")
    parser.add_argument("--client-mib", type=int, default=7000, help="Free memory one simulation needs on its GPU")
    parser.add_argument("--no-wait", action="store_true", help="Fail rather than wait when a GPU lacks free memory")
    parser.add_argument("--no-verify", action="store_true", help="Skip hashing the checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Print the server and client commands and stop")
    args = parser.parse_args(argv)

    number, classes = object_tasks(args.object)
    picked = parse_layouts(args.layouts, len(classes)) if args.layouts else list(range(len(classes)))
    tasks = [classes[i] for i in picked]
    pairs = []
    for item in args.pair:
        name, _, folder = item.partition("=")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not folder or name in {p["name"] for p in pairs} | {"base"}:
            parser.error(f"--pair needs a unique NAME=DIR with a plain name (got {item})")
        pairs.append(pair_info(name, folder, verify=not (args.no_verify or args.dry_run)))
    expected = "SimplePnp" + number + "".join(part.capitalize() for part in args.object.split("_"))
    trained = {p.get("task") for p in pairs} - {None}
    if len(trained) > 1:
        parser.error(f"The pairs were trained for different tasks: {sorted(trained)}")
    label = args.pps_task or next(iter(trained), expected)
    if label != expected:
        print(f"note: noise-key label {label!r} differs from this object's task {expected!r}", flush=True)
    for pair in pairs:
        print(f"pair {pair['name']}: task {pair.get('task')}, {pair.get('steps')} steps on {pair.get('dataset_rows')} "
              f"rows, data {str(pair.get('dataset_digest'))[:12]}, prompt {pair.get('prompt')!r}", flush=True)

    prefix = args.prefix or f"pps_v2_{args.object}"
    report_dir = REPO / "output" / f"{prefix}_eval_{args.tag}"
    base_dir = args.base_run.resolve() if args.base_run else REPO / "output" / f"{prefix}_base_{args.tag}"
    if args.base_run and not finished(base_dir, tasks):
        parser.error(f"--base-run {base_dir} does not hold every requested layout")
    runs = [] if args.base_run else [{"name": "base", "folder": base_dir.name,
                                      "command": client_command(args, tasks, label, base_dir.name)}]
    for gamma in args.gammas:
        for pair in pairs:
            folder = f"{prefix}_{pair['name']}_{gamma_tag(gamma)}_{args.tag}"
            runs.append({"name": f"{pair['name']}@{gamma:g}", "folder": folder, "gamma": gamma,
                         "command": client_command(args, tasks, label, folder, pair["name"], gamma)})
    server_env = {"PORT": str(args.port), "SERVER_GPU": str(args.server_gpu), "MODE": "none", "GAMMA": "0.0"}
    server_command = [str(SERVER), *(x for p in pairs for x in ("--arm", f"{p['name']}={p['v_ref']},{p['v_task']}"))]
    todo = [run for run in runs if not finished(REPO / "output" / run["folder"], tasks)]
    print(f"{len(tasks)} layouts of {args.object}; label {label}; {len(runs)} runs, {len(todo)} to do", flush=True)
    if args.dry_run:
        print(" ".join(f"{k}={v}" for k, v in server_env.items()), " ".join(server_command))
        for run in todo:
            print("OMNI_KIT_ACCEPT_EULA=Y", " ".join(run["command"]))
        return 0

    report_dir.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    (report_dir / "evaluation.json").write_text(json.dumps({
        "object": args.object, "layouts": picked, "tasks": tasks, "pps_task": label, "gammas": args.gammas,
        "pairs": pairs, "base_run": str(base_dir), "server": {"env": server_env, "command": server_command},
        "runs": runs, "robolab_revision": revision, "started_unix_s": time.time()}, indent=1) + "\n")
    failed = []
    if todo:
        if port_open(args.port):
            raise SystemExit(f"Port {args.port} is already in use; pass another --port")
        wait_for_memory(args.server_gpu, args.server_mib, "the PPS server", not args.no_wait)
        with (report_dir / "server.log").open("a") as log:
            server = subprocess.Popen(server_command, env={**os.environ, **server_env}, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        active = []
        try:
            for _ in range(240):  # the 3B base model takes a minute or two to load
                if port_open(args.port):
                    break
                if server.poll() is not None:
                    raise SystemExit(f"The PPS server exited ({server.returncode}); see {report_dir / 'server.log'}")
                time.sleep(5)
            else:
                raise SystemExit(f"The PPS server did not open port {args.port}; see {report_dir / 'server.log'}")
            print(f"server up on port {args.port} (GPU {args.server_gpu})", flush=True)
            queue, last_start = list(todo), 0.0
            while queue or active:
                for run, process in list(active):
                    if process.poll() is not None:
                        active.remove((run, process))
                        complete = finished(REPO / "output" / run["folder"], tasks)
                        print(f"{run['name']}: client exit {process.returncode}, "
                              f"{'complete' if complete else 'INCOMPLETE'}", flush=True)
                        if not complete:
                            failed.append(run["name"])
                if server.poll() is not None:
                    raise SystemExit(f"The PPS server died ({server.returncode}); see {report_dir / 'server.log'}")
                if queue and len(active) < args.parallel and time.time() - last_start >= args.stagger_s:
                    wait_for_memory(args.client_gpu, args.client_mib, f"run {queue[0]['name']}", not args.no_wait)
                    run = queue.pop(0)
                    with (report_dir / f"client_{run['folder']}.log").open("a") as log:
                        process = subprocess.Popen(run["command"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                                   env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "Y"})
                    active.append((run, process))
                    last_start = time.time()
                    print(f"{run['name']}: started -> output/{run['folder']}", flush=True)
                time.sleep(5)
        finally:
            for _, process in active:
                process.terminate()
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    done = [run for run in runs if finished(REPO / "output" / run["folder"], tasks)]
    if not args.base_run and not finished(base_dir, tasks):
        print("The base run is incomplete; no report written", flush=True)
        return 1
    arms = [x for run in done if run["name"] != "base" for x in ("--arm", f"{run['name']}={REPO / 'output' / run['folder']}")]
    if arms:
        result = subprocess.run([sys.executable, str(ANALYSE), "--base", str(base_dir), *arms,
                                 "--csv", str(report_dir / "per_layout.csv")], capture_output=True, text=True)
        (report_dir / "report.txt").write_text(result.stdout + result.stderr)
        print(result.stdout + result.stderr, end="", flush=True)
        print(f"report: {report_dir / 'report.txt'}", flush=True)
    if failed:
        print(f"incomplete runs: {failed}; rerun the same command to resume them", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
