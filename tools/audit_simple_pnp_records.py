#!/usr/bin/env python3
"""Read retained PnP recordings and report integrity issues without changing them.

Run with the project's Python environment (h5py and numpy are required). The
combined JSON is authoritative for episode identity. A complete source recording
is preferred when an interrupted recording also exists on the other GPU.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "output/pi05_simple_pnp_5eps_combined_launch_filtered")
    parser.add_argument("--static", type=Path, default=ROOT / "output/generated_launch_audit/static.json")
    parser.add_argument("--recordings", type=Path, default=ROOT / "output")
    parser.add_argument("--output", type=Path, default=ROOT / "output/generated_additional_failure_audit")
    args = parser.parse_args()
    originals = [args.results / "episode_results.jsonl", args.results / "object_success_summary.json"]
    hashes = {str(p.resolve()): digest(p) for p in originals}
    by_task = collections.defaultdict(list)
    episode_keys = set()
    for line in originals[0].open():
        episode = json.loads(line)
        key = (episode["task_name"], episode["run"], episode["env_id"])
        if key in episode_keys:
            raise ValueError(f"Duplicate episode: {key}")
        episode_keys.add(key)
        by_task[episode["task_name"]].append(episode)
    static = {t["class_name"]: t for t in json.loads(args.static.read_text())["tasks"]}
    summary = json.loads(originals[1].read_text())
    assert summary["episode_count"] == len(episode_keys)
    assert summary["successes"] == sum(e.get("success") is True for es in by_task.values() for e in es)
    args.output.mkdir(parents=True, exist_ok=True)
    issues, early_successes, duplicate_sources = [], [], []
    counts = collections.Counter()
    task_counts = collections.defaultdict(set)
    started = time.monotonic()
    with (args.output / "record_episode_diagnostics.jsonl").open("w") as diagnostics:
        for index, (name, episodes) in enumerate(sorted(by_task.items())):
            target = static[name]["target"]
            candidates = [args.recordings / f"pi05_simple_pnp_5eps_gpu{gpu}" / name / "run_0.hdf5" for gpu in (0, 1)]
            sources = []
            for path in candidates:
                if path.exists():
                    with h5py.File(path, "r") as handle:
                        sources.append((path, len(handle["data"])))
            if not sources:
                for episode in episodes:
                    issues.append({"task": name, "env_id": episode["env_id"], "flags": ["missing_recording"]})
                    counts["missing_recording"] += 1
                    task_counts["missing_recording"].add(name)
                continue
            sources.sort(key=lambda p: (-p[1], str(p[0])))
            path = sources[0][0]
            if len(sources) > 1:
                duplicate_sources.append({"task": name, "selected": str(path),
                                          "sources": [{"path": str(p), "demo_count": n} for p, n in sources]})
            with h5py.File(path, "r") as handle:
                for episode in episodes:
                    eid = episode["env_id"]
                    row = {"task": name, "target": target, "env_id": eid, "source": str(path),
                           "json_success": episode.get("success"), "json_step": episode.get("episode_step"), "flags": []}
                    key = f"data/demo_{eid}"
                    if key not in handle:
                        row["flags"].append("missing_demonstration")
                    else:
                        demo = handle[key]
                        n = int(demo.attrs.get("num_samples", -1))
                        hdf_success = bool(demo.attrs["success"]) if "success" in demo.attrs else None
                        row.update(num_samples=n, hdf_success=hdf_success)
                        if hdf_success is not episode.get("success"):
                            row["flags"].append("hdf_json_success_mismatch")
                        if episode.get("success") is None:
                            row["flags"].append("null_evaluation_outcome")
                        if episode.get("episode_step") is not None and n != episode["episode_step"]:
                            row["flags"].append("sample_count_step_mismatch")
                        initial_key = f"initial_state/rigid_object/{target}/root_pose"
                        if initial_key in demo:
                            row["initial_state_count"] = demo[initial_key].shape[0]
                            if row["initial_state_count"] > 2:
                                row["flags"].append("repeated_initial_states")
                        else:
                            row["flags"].append("missing_initial_state")
                        fields = ["actions", f"states/rigid_object/{target}/root_pose",
                                  f"states/rigid_object/{target}/root_velocity",
                                  "states/articulation/robot/joint_position", "states/articulation/robot/joint_velocity",
                                  "ee_pose/position", "ee_pose/orientation"]
                        nonfinite_fields, wrong_length_fields, missing_fields = [], [], []
                        poses = None
                        for field in fields:
                            if field not in demo:
                                missing_fields.append(field)
                                continue
                            data = demo[field][:]
                            if len(data) != n:
                                wrong_length_fields.append(field)
                            if not np.isfinite(data).all():
                                nonfinite_fields.append(field)
                            if field.endswith(f"{target}/root_pose"):
                                poses = data
                        if missing_fields:
                            row["flags"].append("missing_state_or_action_data")
                            row["missing_fields"] = missing_fields
                        if wrong_length_fields:
                            row["flags"].append("dataset_length_mismatch")
                            row["wrong_length_fields"] = wrong_length_fields
                        if nonfinite_fields:
                            row["flags"].append("nonfinite_state_or_action_data")
                            row["nonfinite_fields"] = nonfinite_fields
                        if poses is not None and len(poses):
                            row["target_max_displacement_from_first_sample_m"] = float(np.linalg.norm(poses[:, :3] - poses[0, :3], axis=1).max())
                            row["target_net_displacement_m"] = float(np.linalg.norm(poses[-1, :3] - poses[0, :3]))
                        if "subtask/score" in demo and len(demo["subtask/score"]):
                            row["first_subtask_score"] = float(demo["subtask/score"][0])
                            row["final_subtask_score"] = float(demo["subtask/score"][-1])
                        if episode.get("success") is True and episode.get("episode_step", 750) < 15:
                            early_successes.append(dict(row, events=episode["events"]))
                    diagnostics.write(json.dumps(row) + "\n")
                    if row["flags"]:
                        issues.append(row)
                    for flag in row["flags"]:
                        counts[flag] += 1
                        task_counts[flag].add(name)
            if (index + 1) % 250 == 0 or index + 1 == len(by_task):
                print(json.dumps({"tasks_checked": index + 1, "total_tasks": len(by_task),
                                  "elapsed_s": round(time.monotonic() - started, 1), "issue_counts": dict(counts)}), flush=True)
    unchanged = all(digest(Path(p)) == value for p, value in hashes.items())
    assert unchanged
    report = {"task_count": len(by_task), "episode_count": len(episode_keys),
              "checked_datasets": ["actions", "target root pose/velocity", "robot joint position/velocity", "end effector pose"],
              "issue_episode_counts": dict(counts), "issue_task_counts": {k: len(v) for k, v in task_counts.items()},
              "issues": issues, "early_successes": early_successes, "duplicate_recording_sources": duplicate_sources,
              "source_sha256": hashes, "source_files_verified_unchanged": unchanged,
              "elapsed_s": time.monotonic() - started,
              "interpretation": "Early success and repeated initial states need scene/physics review; neither alone proves a bad task. HDF success is not a replacement for null JSON outcomes."}
    (args.output / "record_integrity.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ["task", "target", "env_id", "source", "json_success", "json_step", "hdf_success", "num_samples", "initial_state_count", "flags"]
    with (args.output / "record_issues.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, "flags": "; ".join(row["flags"])} for row in issues)
    print(json.dumps({k: v for k, v in report.items() if k not in ("issues", "early_successes", "source_sha256")}))


if __name__ == "__main__":
    main()
