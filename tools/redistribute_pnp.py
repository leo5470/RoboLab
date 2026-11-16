"""Rebalance the existing five-episode run when a GPU completes its shard.

Only untouched tasks move between result directories. Completed and partially
recorded tasks remain with their original directory. Run with --check to inspect
the initial assignment without signalling or launching any processes.
"""
import argparse
import fnmatch
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "output/pi05_fresh_5eps_logs"


def eligible_tasks():
    directory = ROOT / 'robolab/tasks/single_pnp'
    tasks = json.loads((directory / 'manifest.json').read_text())['tasks']
    patterns = json.loads((directory / 'exclusions.json').read_text())['target_patterns']
    return {t['class_name'] for t in tasks
            if not any(fnmatch.fnmatchcase(t['target'], p) for p in patterns)}


def results(gpu):
    path = ROOT / f"output/pi05_simple_pnp_5eps_gpu{gpu}/episode_results.jsonl"
    counts = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A writer may be finishing the last line.
            counts.setdefault(row['env_name'], set()).add(row['episode'])
    return counts


def alive(pid):
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
        return b'/RoboLab/policies/pi0_family/run.py' in cmd
    except FileNotFoundError:
        return False


def stop(pid):
    if alive(pid):
        os.kill(pid, signal.SIGTERM)
        for _ in range(60):
            if not alive(pid):
                return
            time.sleep(1)
        raise RuntimeError(f'Worker {pid} did not stop; refusing duplicate launch')


def launch(gpu, tasks):
    eligible = eligible_tasks()
    tasks = [t for t in tasks if t in eligible]
    if not tasks:
        raise ValueError('No eligible tasks to launch')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMNI_KIT_ACCEPT_EULA='YES')
    cmd = [str(ROOT / '.venv/bin/python'), str(ROOT / 'policies/pi0_family/run.py'),
           '--policy', 'pi05', '--headless', '--device', 'cuda:0',
           '--remote-port', str(8000 + gpu), '--task-dirs', 'single_pnp',
           '--num-envs', '5', '--num-runs', '1', '--video-mode', 'none',
           '--output-folder-name', f'pi05_simple_pnp_5eps_gpu{gpu}', '--task', *tasks]
    with (LOG / f'eval_gpu{gpu}.log').open('ab') as log:
        return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True).pid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--pids', nargs=2, type=int, required=True,
                        help='Current evaluator PIDs in GPU 0, GPU 1 order')
    args = parser.parse_args()
    tasks = [r['class_name'] for r in json.loads(
        (ROOT / 'robolab/tasks/single_pnp/manifest.json').read_text())['tasks']]
    assert len(tasks) == len(set(tasks)) == 7200
    eligible = eligible_tasks()
    assigned = [[t for t in shard if t in eligible]
                for shard in (tasks[:3600], tasks[3600:])]
    pids = args.pids
    if args.check:
        for gpu in (0, 1):
            counts = results(gpu)
            print(gpu, 'alive', alive(pids[gpu]), 'remaining',
                  sum(len(counts.get(t, set())) < 5 for t in assigned[gpu]))
        return
    while True:
        counts = [results(g) for g in (0, 1)]
        remaining = [[t for t in assigned[g] if len(counts[g].get(t, set())) < 5]
                     for g in (0, 1)]
        if not any(remaining):
            print(f'All {len(eligible)} eligible tasks have five recorded episodes.', flush=True)
            return
        for gpu in (0, 1):
            if remaining[gpu] and not alive(pids[gpu]):
                raise RuntimeError(f'GPU {gpu} exited with unfinished tasks; inspect its log')
        for idle in (0, 1):
            donor = 1 - idle
            if remaining[idle] or len(remaining[donor]) < 2:
                continue
            # Stop before reading the authoritative task counts for reassignment.
            # Any interrupted unrecorded batch is rerun, not counted twice.
            stop(pids[idle])
            stop(pids[donor])
            current = results(donor)
            pending = [t for t in assigned[donor] if len(current.get(t, set())) < 5]
            untouched = [t for t in pending if not current.get(t)]
            moved = untouched[len(untouched) // 2:]
            kept = [t for t in pending if t not in set(moved)]
            assigned[idle], assigned[donor] = moved, kept
            assert not set(moved) & set(kept)
            for gpu in (0, 1):
                if assigned[gpu]:
                    pids[gpu] = launch(gpu, assigned[gpu])
            snapshot = {'time': time.time(), 'pids': pids, 'assignments': assigned}
            with (LOG / 'redistribution.jsonl').open('a') as stream:
                stream.write(json.dumps(snapshot) + '\n')
            print(f'Rebalanced: GPU0={len(assigned[0])}, GPU1={len(assigned[1])}', flush=True)
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
