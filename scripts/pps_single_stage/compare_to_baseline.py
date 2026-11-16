#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare a PPS-server run against the JAX-server baseline, task by task.

The two runs share their scenes but not their noise. RoboLab's benchmark tasks reset to their
authored state (`reset_scene_to_default`), so episode *i* of task *T* starts from the same place in
both — but the JAX server drew epsilon from its own rng while the PPS server derives it from the
trial indices. So per-episode outcomes are not expected to line up, and this reports the paired
test that *is* valid — McNemar over tasks is not it either, since a task is not a trial.

What it reports:

  * per-task rates, and the difference,
  * pooled rates with Wilson intervals,
  * a two-proportion z-test on the pooled counts, which is the right test for two independent
    samples of the same 22 tasks,
  * the failure-stage breakdown RoboLab gives for free in each row's `reason` field — whether a
    failure was a failed grasp or a failed place. That is where a backend difference would show
    up first: the gripper is a thresholded channel, so a small numerical difference either does
    nothing or flips a grasp.

Usage:
    python scripts/pps_single_stage/compare_to_baseline.py \\
        --baseline output/pi05_single_stage_5eps \\
        --arm output/pps_jointpos_container_placement_5eps \\
              output/pps_jointpos_surface_placement_5eps
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re


def load(paths: list[pathlib.Path]) -> dict[str, list[dict]]:
    """Read one or more RoboLab output folders into {task: [row, ...]} in episode order."""
    rows: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    for path in paths:
        results = path / "episode_results.jsonl"
        if not results.exists():
            raise SystemExit(f"no episode_results.jsonl under {path}")
        for line in results.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["env_name"]][int(row["episode"])] = row
    return {task: [r for _, r in sorted(eps.items())] for task, eps in rows.items()}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — the one that stays inside [0, 1] at k=0, which happens a lot here."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z-test. Independent samples: the runs share scenes, not noise."""
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (k1 / n1 - k2 / n2) / se
    # two-sided normal tail
    return z, math.erfc(abs(z) / math.sqrt(2))


_STAGE = re.compile(r"Condition not satisfied: (\w+)\(")


def stage(row: dict) -> str:
    """`grasp`, `place`, `other`, or `success`, from RoboLab's own failed-predicate string."""
    if row.get("success"):
        return "success"
    reason = row.get("reason") or ""
    match = _STAGE.search(reason)
    if not match:
        return "other"
    name = match.group(1)
    if "grab" in name or "grasp" in name:
        return "grasp"
    if "in_container" in name or "on_surface" in name or "place" in name:
        return "place"
    return "other"


def rate(rows: list[dict]) -> tuple[int, int]:
    return sum(1 for r in rows if r.get("success")), len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=pathlib.Path, nargs="+", required=True,
                    help="Output folder(s) of the JAX-server run.")
    ap.add_argument("--arm", type=pathlib.Path, nargs="+", required=True,
                    help="Output folder(s) of the PPS-server run.")
    ap.add_argument("--markdown", type=pathlib.Path, default=None,
                    help="Also write the table to this file.")
    args = ap.parse_args()

    base, arm = load(args.baseline), load(args.arm)
    tasks = [t for t in base if t in arm]
    missing = sorted(set(base) - set(arm))

    lines = []
    w = max((len(t) for t in tasks), default=10)
    lines.append(f"| {'Task':<{w}} | baseline | this run | Δ |")
    lines.append(f"|{'-' * (w + 2)}|---:|---:|---:|")

    bk = bn = ak = an = 0
    moved = []
    for task in tasks:
        k1, n1 = rate(base[task])
        k2, n2 = rate(arm[task])
        bk, bn, ak, an = bk + k1, bn + n1, ak + k2, an + n2
        d = (k2 / n2 - k1 / n1) * 100 if n1 and n2 else 0.0
        lines.append(f"| {task:<{w}} | {k1}/{n1} ({100*k1/n1:.0f}%) | {k2}/{n2} ({100*k2/n2:.0f}%) "
                     f"| {d:+.0f} pts |")
        if k1 != k2:
            moved.append((task, k1, n1, k2, n2))

    lo_b, hi_b = wilson(bk, bn)
    lo_a, hi_a = wilson(ak, an)
    z, p = two_proportion_z(ak, an, bk, bn)

    lines.append(f"| **pooled** | **{bk}/{bn} ({100*bk/bn:.1f}%)** | **{ak}/{an} "
                 f"({100*ak/an:.1f}%)** | **{100*(ak/an - bk/bn):+.1f} pts** |")

    print("\n".join(lines))
    print()
    print(f"baseline 95% CI: {100*lo_b:.1f}% – {100*hi_b:.1f}%")
    print(f"this run 95% CI: {100*lo_a:.1f}% – {100*hi_a:.1f}%")
    print(f"two-proportion z = {z:+.2f}, p = {p:.3f} "
          f"({'no detectable difference' if p >= 0.05 else 'DIFFERENT'})")
    if missing:
        print(f"\ntasks in the baseline but not in this run: {', '.join(missing)}")

    print(f"\ntasks whose count moved ({len(moved)} of {len(tasks)}):")
    for task, k1, n1, k2, n2 in moved:
        print(f"  {task:<{w}} {k1}/{n1} -> {k2}/{n2}")

    print("\nfailure stage, pooled (RoboLab's own failed-predicate name):")
    for name, rows in (("baseline", base), ("this run", arm)):
        counts = collections.Counter(stage(r) for t in tasks for r in rows[t])
        total = sum(counts.values())
        parts = " ".join(f"{k}={v} ({100*v/total:.0f}%)" for k, v in sorted(counts.items()))
        print(f"  {name:<9} {parts}")

    if args.markdown:
        args.markdown.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
