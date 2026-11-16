#!/usr/bin/env python3
"""Paired report for PPS runs over the same layouts: totals, each run against the base, and run against run.

Every RUN is a folder written by policies/pps/run.py (it holds episode_results.jsonl). Runs are
compared layout by layout on the layouts they all contain, one episode per layout; with more than
one episode per layout a layout counts as a success when most of its episodes succeed, and the
report says so. The p-value is the exact two-sided McNemar (sign) test on the discordant layouts.

    analyse_proxy_pairs.py --base output/pps_v2_r3_base_pail_20261111 \
        --arm T=output/pps_v2_pail_T_g04_20261113 --arm L=output/pps_v2_pail_L_g04_20261113

See docs/pps_proxy_evaluation.md.
"""

import argparse
import csv
import json
from collections import Counter
from itertools import combinations
from math import comb, sqrt
from pathlib import Path


def load(directory):
    """{layout task name: [episode rows]} of one run folder."""
    path = Path(directory) / "episode_results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No episode_results.jsonl in {directory}")
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows.setdefault(row["task_name"], []).append(row)
    return rows


def succeeded(episodes):
    """A layout's outcome: its one episode, or the majority of several (ties count as failures)."""
    return 2 * sum(bool(e["success"]) for e in episodes) > len(episodes)


def mcnemar(gained, lost):
    """Exact two-sided p-value for ``gained`` against ``lost`` discordant layouts."""
    n = gained + lost
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(gained, lost) + 1)) / 2 ** n)


def wilson(successes, n, z=1.96):
    """95% Wilson interval of a success rate, as fractions."""
    if n == 0:
        return 0.0, 1.0
    p, denominator = successes / n, 1 + z * z / n
    centre, half = p + z * z / (2 * n), z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / denominator), min(1.0, (centre + half) / denominator)


def compare(first, second, layouts):
    """Layouts only ``first`` solves, only ``second`` solves, and the exact p-value."""
    gained = [t for t in layouts if first[t] and not second[t]]
    lost = [t for t in layouts if second[t] and not first[t]]
    return gained, lost, mcnemar(len(gained), len(lost))


def short(task):
    """Layout index of a task name (SimplePnpV2242PlasticpailA0107Task -> 07), else the name."""
    return task[-6:-4] if task.endswith("Task") and task[-6:-4].isdigit() else task


def report(runs, base=None):
    """Text report for ``runs`` ({name: {task: [rows]}}); ``base`` names the reference run."""
    layouts = sorted(set.intersection(*(set(r) for r in runs.values())))
    if not layouts:
        raise ValueError("The runs share no layout")
    outcome = {name: {t: succeeded(r[t]) for t in layouts} for name, r in runs.items()}
    lines = [f"layouts in common: {len(layouts)}"]
    repeats = {name: Counter(len(r[t]) for t in layouts) for name, r in runs.items()}
    if any(set(c) != {1} for c in repeats.values()):
        lines.append("episodes per layout differ from 1 (majority outcome used): "
                     + ", ".join(f"{n} {dict(c)}" for n, c in repeats.items()))
    for name, r in runs.items():
        k = sum(outcome[name].values())
        low, high = wilson(k, len(layouts))
        extra = "" if len(r) == len(layouts) else f"; {len(r)} layouts in this run"
        lines.append(f"{name}: {k}/{len(layouts)} ({k / len(layouts):.0%}, 95% CI {low:.0%}-{high:.0%}{extra})")
    names = list(runs)
    pairs = ([(n, base) for n in names if n != base] if base else []) + \
            [(b, a) for a, b in combinations([n for n in names if n != base], 2)]
    for first, second in pairs:
        gained, lost, p = compare(outcome[first], outcome[second], layouts)
        lines.append(f"{first} vs {second}: {first}-only {len(gained)}, {second}-only {len(lost)}, "
                     f"difference {(len(gained) - len(lost)) / len(layouts):+.0%}, exact McNemar "
                     + ("p<0.001" if p < 0.001 else f"p={p:.3f}"))
    for name, r in runs.items():
        failed = [t for t in layouts if not outcome[name][t]]
        reasons = Counter(str(e.get("reason", ""))[:60] for t in failed for e in r[t] if not e["success"])
        lines.append(f"{name} failed layouts: {' '.join(short(t) for t in failed) or '-'}")
        if reasons:
            lines.append(f"    {dict(reasons)}")
    return "\n".join(lines), layouts, outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, help="Base-policy run folder (gamma 0); each arm is compared with it")
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=DIR", help="A steered run; repeat")
    parser.add_argument("--csv", type=Path, help="Also write one row per layout with every run's outcome")
    args = parser.parse_args(argv)
    runs = {}
    if args.base:
        runs["base"] = load(args.base)
    for item in args.arm:
        name, _, directory = item.partition("=")
        if not name or not directory or name in runs:
            parser.error(f"--arm needs a unique NAME=DIR (got {item})")
        runs[name] = load(directory)
    if len(runs) < 2 and not (runs and not args.base):
        parser.error("Give --base and at least one --arm, or at least two --arm")
    text, layouts, outcome = report(runs, "base" if args.base else None)
    print(text)
    if args.csv:
        with args.csv.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["layout", *runs])
            for task in layouts:
                writer.writerow([task, *(int(outcome[name][task]) for name in runs)])


if __name__ == "__main__":
    main()
