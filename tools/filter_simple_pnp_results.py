#!/usr/bin/env python3
"""Save an audited copy of combined PnP results, retaining all original files.

Exclude all episodes of tasks that fell in the launch audit, together with
placements whose asset scale was replaced (the 50 oversized apple_01 tasks),
and explicitly excluded targets (wood_block, birdhouse, coffee_can, dumbbell, cubebox_a02, metalfstylecan_a01, and brick by default).
The destination must be a new directory. Summary grouping and success counting
match the original object_success_summary.json; null outcomes remain in the
episode denominator, as they do in the original summary.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def save_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(episodes, original, source):
    grouped = collections.defaultdict(list)
    for episode in episodes:
        grouped[episode["task_name"]].append(episode)
    objects = []
    known_tasks = set()
    for obj in original["objects"]:
        tasks = []
        for task in obj["tasks"]:
            name = task["task"]
            known_tasks.add(name)
            rows = grouped.get(name, [])
            if rows:
                successes = sum(row.get("success") is True for row in rows)
                tasks.append({"task": name, "episodes": len(rows), "successes": successes,
                              "success_rate": successes / len(rows)})
        if tasks:
            count = sum(t["episodes"] for t in tasks)
            successes = sum(t["successes"] for t in tasks)
            objects.append({"catalog_index": obj["catalog_index"], "name": obj["name"],
                            "usd_path": obj["usd_path"], "task_count": len(tasks),
                            "episode_count": count, "successes": successes,
                            "success_rate": successes / count, "tasks": tasks})
    if set(grouped) - known_tasks:
        raise ValueError("Episode tasks are missing from original object summary")
    count = len(episodes)
    successes = sum(e.get("success") is True for e in episodes)
    return {"source": source, "grouping": original["grouping"], "object_count": len(objects),
            "task_count": len(grouped), "episode_count": count, "successes": successes,
            "success_rate": successes / count, "objects": objects}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "output/pi05_simple_pnp_5eps_combined")
    parser.add_argument("--audit", type=Path, default=ROOT / "output/generated_launch_audit/report.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/pi05_simple_pnp_5eps_combined_launch_filtered")
    parser.add_argument("--exclude-target", action="append", default=["wood_block", "birdhouse", "coffee_can", "dumbbell", "cubebox_a02", "metalfstylecan_a01", "brick"],
                        help="Also exclude every task for this target; repeatable. wood_block, birdhouse, coffee_can, dumbbell, cubebox_a02, metalfstylecan_a01, and brick are excluded by default.")
    args = parser.parse_args()
    excluded_targets = set(args.exclude_target)
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() or output == source:
        raise FileExistsError(f"Destination must be a new directory: {output}")
    originals = [source / "episode_results.jsonl", source / "object_success_summary.json"]
    hashes = {str(path): digest(path) for path in originals}
    original = json.loads(originals[1].read_text())
    audit = json.loads(args.audit.read_text())
    exclusions = {}
    for task in audit["tasks"]:
        reasons = []
        if task.get("fell_below_table"):
            reasons.append("fell_below_table_in_launch_audit")
        if "asset_scale_replaced" in task["flags"]:
            reasons.append("oversized_asset_scale_replaced")
        if task["target"] in excluded_targets:
            reasons.append("user_excluded_target")
        if reasons:
            exclusions[task["class_name"]] = {
                "task": task["class_name"], "catalog_index": task["catalog_index"],
                "object": task["target"], "reasons": "; ".join(reasons),
            }
    kept, removed, kept_lines, removed_lines, all_rows = [], [], [], [], []
    for line in originals[0].read_bytes().splitlines(keepends=True):
        if not line.strip():
            raise ValueError("Unexpected blank line in source results")
        row = json.loads(line)
        if row["env_name"] != row["task_name"]:
            raise ValueError("Task/environment identity mismatch")
        all_rows.append(row)
        if row["task_name"] in exclusions:
            removed.append(row)
            removed_lines.append(line)
        else:
            kept.append(row)
            kept_lines.append(line)
    if set(exclusions) - {r["task_name"] for r in all_rows}:
        raise ValueError("Excluded tasks missing from source results")
    reconstructed = summarize(all_rows, original, original["source"])
    if reconstructed != original:
        raise ValueError("Original summary does not match original episode data")
    summary = summarize(kept, original, str(output / "episode_results.jsonl"))
    removed_counts = collections.Counter(r["task_name"] for r in removed)
    removed_successes = collections.Counter(r["task_name"] for r in removed if r.get("success") is True)
    for name, record in exclusions.items():
        record.update(episodes_removed=removed_counts[name], successes_removed=removed_successes[name])
    output.mkdir(parents=True, exist_ok=False)
    (output / "episode_results.jsonl").write_bytes(b"".join(kept_lines))
    (output / "excluded_episode_results.jsonl").write_bytes(b"".join(removed_lines))
    save_json(output / "object_success_summary.json", summary)
    save_csv(output / "object_success_summary.csv", summary["objects"],
             ["catalog_index", "name", "usd_path", "task_count", "episode_count", "successes", "success_rate"])
    save_csv(output / "excluded_tasks.csv", sorted(exclusions.values(), key=lambda x: x["task"]),
             ["task", "catalog_index", "object", "reasons", "episodes_removed", "successes_removed"])
    remaining_by_index = {o["catalog_index"]: o for o in summary["objects"]}
    comparison = []
    for obj in original["objects"]:
        new = remaining_by_index.get(obj["catalog_index"])
        comparison.append({"catalog_index": obj["catalog_index"], "name": obj["name"],
                           "original_tasks": obj["task_count"], "remaining_tasks": new["task_count"] if new else 0,
                           "original_episodes": obj["episode_count"], "remaining_episodes": new["episode_count"] if new else 0,
                           "original_success_rate": obj["success_rate"], "filtered_success_rate": new["success_rate"] if new else None})
    save_csv(output / "object_comparison.csv", comparison, list(comparison[0]))
    provenance = {
        "source_directory": str(source), "audit": str(args.audit.resolve()),
        "exclusion_rule": "All episodes of tasks that fell or had asset scale replaced, plus user-excluded targets "
                          f"({', '.join(sorted(excluded_targets))}); geometry warnings alone do not exclude a task.",
        "excluded_targets": sorted(excluded_targets),
        "excluded_task_count": len(exclusions), "excluded_episode_count": len(removed),
        "excluded_successes": sum(r.get("success") is True for r in removed),
        "original": {k: original[k] for k in ["object_count", "task_count", "episode_count", "successes", "success_rate"]},
        "filtered": {k: summary[k] for k in ["object_count", "task_count", "episode_count", "successes", "success_rate"]},
        "source_sha256": hashes, "excluded_tasks": sorted(exclusions),
    }
    # Verify the kept records are byte-for-byte copies and original files were retained.
    assert digest(output / "episode_results.jsonl") == hashlib.sha256(b"".join(kept_lines)).hexdigest()
    assert all(digest(Path(path)) == value for path, value in hashes.items())
    assert original["episode_count"] == summary["episode_count"] + len(removed)
    assert original["successes"] == summary["successes"] + provenance["excluded_successes"]
    provenance["original_files_verified_unchanged"] = True
    save_json(output / "filter_manifest.json", provenance)
    text = [f"Filtered combined pi05 results: {summary['successes']:,} / {summary['episode_count']:,} successful episodes "
            f"({100 * summary['success_rate']:.2f}%).", "",
            f"Excluded {len(exclusions)} tasks for launch-audit falls, replaced asset scales, or user-excluded targets "
            f"({', '.join(sorted(excluded_targets))}). "
            f"Removed all {len(removed)} recorded episodes for those tasks, including {provenance['excluded_successes']} successes. "
            "Retained tasks with geometry warnings alone. Original files verified unchanged by SHA-256.", "",
            "| Metric | Original | Filtered |", "| --- | ---: | ---: |",
            f"| Catalog entries | {original['object_count']:,} | {summary['object_count']:,} |",
            f"| Tasks | {original['task_count']:,} | {summary['task_count']:,} |",
            f"| Episodes | {original['episode_count']:,} | {summary['episode_count']:,} |",
            f"| Successes | {original['successes']:,} | {summary['successes']:,} |",
            f"| Success rate | {100 * original['success_rate']:.2f}% | {100 * summary['success_rate']:.2f}% |", "",
            "Per-object rates use all remaining recorded episodes as the denominator, including null outcomes, "
            "matching the original summary. Catalog grouping and original USD asset labels are retained.", "",
            "| Index | Object | Tasks | Episodes | Successes | Success rate |",
            "| --- | --- | ---: | ---: | ---: | ---: |"]
    for obj in sorted(summary["objects"], key=lambda o: o["catalog_index"]):
        text.append(f"| {obj['catalog_index']:03d} | {obj['name']} | {obj['task_count']} | "
                    f"{obj['episode_count']} | {obj['successes']} | {100 * obj['success_rate']:.2f}% |")
    (output / "summary.md").write_text("\n".join(text) + "\n")
    print(json.dumps({k: v for k, v in provenance.items() if k not in ("source_sha256", "excluded_tasks")}))


if __name__ == "__main__":
    main()
