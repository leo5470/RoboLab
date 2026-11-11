#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Group an eval run's videos by what actually happened in the episode.

RoboLab names videos `{instruction}_{run}_env{N}.mp4`, which says nothing about the outcome — so
finding "the episodes where it grasped the object and then failed to place it" means cross-
referencing `episode_results.jsonl` by hand every time. This does it once and leaves symlinked
views beside the originals:

    <task>/successes/ep07.mp4          -> the sensor video (exterior + wrist, what the policy saw)
    <task>/successes/ep07_viewport.mp4 -> third-person
    <task>/grasp_failures/...
    <task>/place_failures/...

The buckets come from RoboLab's own failed-predicate string in each row's `reason`, not from a
guess about the trajectory: `object_grabbed` never satisfied is a grasp failure, `object_in_container`
or `object_on_top` unsatisfied is a place failure — it had the object and did not get it there.
That distinction is the interesting one for steering, because the two call for different residuals.

Symlinks rather than copies: the originals stay put and the buckets cost nothing.

Usage:
    python scripts/pps_single_stage/sort_videos_by_outcome.py --run output/pps_mustard_videos
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib

BUCKETS = ("successes", "grasp_failures", "place_failures", "other_failures")


def bucket_for(row: dict) -> str:
    if row.get("success"):
        return "successes"
    reason = row.get("reason") or ""
    if "object_grabbed" in reason:
        return "grasp_failures"
    if "object_in_container" in reason or "object_on_top" in reason:
        return "place_failures"
    return "other_failures"


def sort_run(run_dir: pathlib.Path, *, clean: bool = False) -> None:
    results = run_dir / "episode_results.jsonl"
    if not results.exists():
        raise SystemExit(f"no episode_results.jsonl under {run_dir}")
    rows = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]

    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_task[r["env_name"]].append(r)

    for task, task_rows in sorted(by_task.items()):
        root = run_dir / task
        if not root.is_dir():
            continue
        for b in BUCKETS:
            d = root / b
            if clean and d.is_dir():
                for link in d.glob("*.mp4"):
                    link.unlink()
            d.mkdir(exist_ok=True)

        counts: collections.Counter = collections.Counter()
        linked = 0
        for r in task_rows:
            b = bucket_for(r)
            counts[b] += 1
            for suffix in ("", "_viewport"):
                pattern = f"*_{r['run']}_env{r['env_id']}{suffix}.mp4"
                for video in root.glob(pattern):
                    link = root / b / f"ep{r['episode']:02d}{suffix}.mp4"
                    if not link.exists():
                        os.symlink(os.path.relpath(video, root / b), link)
                        linked += 1

        total = sum(counts.values())
        summary = "  ".join(f"{b.replace('_', ' ')} {counts[b]}" for b in BUCKETS if counts[b])
        print(f"{task}: {total} episodes ({summary}); {linked} videos linked")
        # Empty buckets are left in place on purpose: their absence would be ambiguous with
        # "this outcome never happened", which is itself worth seeing.


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=pathlib.Path, required=True,
                    help="An eval output folder containing episode_results.jsonl.")
    ap.add_argument("--clean", action="store_true",
                    help="Remove existing links first (use after re-rendering a run).")
    args = ap.parse_args()
    sort_run(args.run, clean=args.clean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
