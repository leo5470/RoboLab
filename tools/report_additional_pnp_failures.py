#!/usr/bin/env python3
"""Combine additional scene and recording audits, leaving prior results intact."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def save_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def final_bounds(task, asset):
    bounds = asset["local_collider_bounds"] or asset["local_visual_bounds"]
    corners = np.array([[x, y, z] for x in (bounds["min"][0], bounds["max"][0])
                        for y in (bounds["min"][1], bounds["max"][1])
                        for z in (bounds["min"][2], bounds["max"][2])])
    sample = task["samples"][-1]
    w, x, y, z = sample["rotation_wxyz"]
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    world = corners @ rotation.T + np.array(sample["position"])
    return world.min(0), world.max(0)


def episode_detail(row):
    """Locate the onset of bad data in a flagged episode, without replaying it."""
    with h5py.File(row["source"], "r") as handle:
        demo = handle[f"data/demo_{row['env_id']}"]
        details = {}
        keys = ["actions", f"states/rigid_object/{row['target']}/root_pose",
                f"states/rigid_object/{row['target']}/root_velocity",
                "states/articulation/robot/joint_position", "states/articulation/robot/joint_velocity"]
        for key in keys:
            data = demo[key][:]
            nonfinite = np.flatnonzero(~np.isfinite(data).all(axis=1))
            extreme = np.flatnonzero(np.abs(data).max(axis=1) > 100)
            details[key] = {"first_nonfinite_sample": int(nonfinite[0]) + 1 if len(nonfinite) else None,
                            "first_absolute_value_over_100_sample": int(extreme[0]) + 1 if len(extreme) else None}
        return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/generated_additional_failure_audit")
    args = parser.parse_args()
    output = args.output
    launch = json.loads((ROOT / "output/generated_launch_audit/report.json").read_text())
    filtered_dir = ROOT / "output/pi05_simple_pnp_5eps_combined_launch_filtered"
    summary = json.loads((filtered_dir / "object_success_summary.json").read_text())
    kept = {t["task"] for o in summary["objects"] for t in o["tasks"]}
    tasks = {t["class_name"]: t for t in launch["tasks"] if t["class_name"] in kept}
    assets = {a["catalog_index"]: a for a in launch["assets"]}
    selections = [json.loads((output / name).read_text()) for name in ("selection.json", "supplemental_selection.json")]
    expected = {t["class_name"] for selection in selections for t in selection}
    physics, errors, sources = {}, [], []
    for part in ("priority", "extended", "supplemental"):
        for path in sorted((output / part).glob("physics_*.jsonl")):
            sources.append(str(path.resolve()))
            for line in path.open():
                group = json.loads(line)
                if "error" in group:
                    errors.append(group)
                for task in group.get("tasks", []):
                    name = task["class_name"]
                    if name in physics:
                        raise ValueError(f"Duplicate physics task: {name}")
                    physics[name] = dict(task, seconds=group["seconds"], device=group["device"])
    if errors or set(physics) != expected:
        raise ValueError(f"Incomplete physics audit: errors={len(errors)}, missing={len(expected-set(physics))}, extra={len(set(physics)-expected)}")
    assert all(t["seconds"] == 15 for t in physics.values())
    integrity = json.loads((output / "record_integrity.json").read_text())
    episodes = [json.loads(line) for line in (output / "record_episode_diagnostics.jsonl").open()]
    assert len(episodes) == summary["episode_count"]
    assert {e["task"] for e in episodes} == kept
    by_task = collections.defaultdict(list)
    for e in episodes:
        by_task[e["task"]].append(e)
    scene_issues = collections.defaultdict(list)
    for name, task in physics.items():
        if task["nonfinite_state"]:
            scene_issues[name].append({"reason": "nonfinite_state_without_robot_motion", "source": "15_second_followup"})
        if task["fell_below_table"]:
            scene_issues[name].append({"reason": "additional_no_action_fall", "source": "15_second_followup",
                                       "first_time_s": task["first_below_table_time_s"]})
        if task["success_without_motion"]:
            scene_issues[name].append({"reason": "success_without_robot_motion", "source": "15_second_followup",
                                       "first_time_s": task["first_success_without_motion_s"],
                                       "first_bin_contact_s": task["first_bin_contact_s"],
                                       "first_gripper_contact_s": task["first_gripper_contact_s"]})
    for name, task in tasks.items():
        mins, maxs = final_bounds(task, assets[task["catalog_index"]])
        table = task["table_bounds"]
        if any(maxs[i] < table["min"][i] or mins[i] > table["max"][i] for i in (0, 1)):
            scene_issues[name].append({"reason": "entire_collision_bounds_off_table_at_5s", "source": "original_5_second_audit",
                                       "final_position": task["final_position"], "final_bounds_min": mins.tolist(), "final_bounds_max": maxs.tolist(),
                                       "repeated_off_table": bool(physics.get(name, {}).get("left_table_bounds"))})
        if task["final_position"][2] > 1 and task["displacement_m"] > 5:
            scene_issues[name].append({"reason": "extreme_launch_flyoff", "source": "original_5_second_audit",
                                       "final_root_height_m": task["final_position"][2], "displacement_m": task["displacement_m"]})
    numerical, extreme = [], []
    for e in episodes:
        if "nonfinite_state_or_action_data" in e["flags"]:
            numerical.append(dict(e, onset=episode_detail(e)))
        if e.get("target_max_displacement_from_first_sample_m", 0) > 5:
            extreme.append(dict(e, onset=episode_detail(e)))
    confirmed = []
    for name, evidence in sorted(scene_issues.items()):
        task = tasks[name]
        recorded = by_task[name]
        confirmed.append({"task": name, "catalog_index": task["catalog_index"], "target": task["target"],
                          "reasons": [e["reason"] for e in evidence], "evidence": evidence,
                          "recorded_episodes": len(recorded), "recorded_successes": sum(e["json_success"] is True for e in recorded),
                          "recorded_null_outcomes": sum(e["json_success"] is None for e in recorded),
                          "static_flags": task["flags"]})
    null_names = sorted({e["task"] for e in episodes if e["json_success"] is None})
    spontaneous = sorted(n for n, t in physics.items() if t["success_without_motion"])
    falls = sorted(n for n, t in physics.items() if t["fell_below_table"])
    late_falls = [n for n in falls if physics[n]["first_below_table_time_s"] > 5]
    latest_fall = max((physics[n]["first_below_table_time_s"] for n in falls), default=0)
    entirely_off = sorted(n for n, es in scene_issues.items() if any(e["reason"] == "entire_collision_bounds_off_table_at_5s" for e in es))
    # Prior originals and the filtered dataset are preserved. Check both against
    # their saved provenance hashes rather than just checking file existence.
    manifest = json.loads((filtered_dir / "filter_manifest.json").read_text())
    hashes = {**manifest["source_sha256"], **integrity["source_sha256"]}
    unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in hashes.items())
    assert unchanged
    report = {
        "scope": {"retained_tasks": len(kept), "retained_episodes": len(episodes), "all_generated_tasks_in_initial_audit": len(launch["tasks"]),
                  "additional_physics_tasks": len(physics), "additional_physics_catalog_entries": len({t["catalog_index"] for t in physics.values()}),
                  "physics_duration_s": 15, "policy": "none; robot joints hold their reset positions", "physics_sources": sources},
        "counts": {"confirmed_scene_issue_tasks": len(confirmed), "additional_fall_tasks": len(falls),
                   "additional_falls_after_5s": len(late_falls),
                   "success_without_robot_motion_tasks": len(spontaneous), "entirely_off_table_in_original_audit_tasks": len(entirely_off),
                   "null_outcome_tasks": len(null_names), "null_outcome_episodes": sum(e["json_success"] is None for e in episodes),
                   "nonfinite_recording_tasks": len({e["task"] for e in numerical}), "nonfinite_recording_episodes": len(numerical),
                   "over_5m_recorded_displacement_tasks": len({e["task"] for e in extreme}), "over_5m_recorded_displacement_episodes": len(extreme)},
        "confirmed_scene_issues": confirmed, "additional_falls": falls, "success_without_robot_motion": spontaneous,
        "entirely_off_table_in_original_audit": entirely_off, "null_outcome_tasks": null_names,
        "nonfinite_recording_episodes": numerical, "extreme_recorded_motion_candidates": extreme,
        "remaining_static_warning_counts": dict(collections.Counter(f for t in tasks.values() for f in t["flags"])),
        "record_integrity_counts": integrity["issue_episode_counts"],
        "original_and_filtered_results_verified_unchanged": unchanged, "source_sha256": hashes,
        "interpretation": [
            "Confirmed scene issues were observed in at least one no-action audit; physics variations mean one stable rerun does not clear an earlier failure.",
            f"Latest additional fall began at {latest_fall:.3f}s; {len(late_falls)} began after the original five-second observation window.",
            "Current no-action success does not imply every existing policy episode of that task was a false positive. Butter13's original five episodes failed.",
            "Four null-outcome tasks repeatedly record initial states (501 each) and HDF success despite null JSON outcomes; do not turn these into valid policy successes.",
            "NaNs and extreme motion during policy execution are episode-level evidence. They do not by themselves establish that all placements of an asset or every episode of a task are invalid.",
            "AABB overlap, initial penetration, floating placement, transient boundary crossing, and residual velocity are review warnings rather than automatic exclusions.",
            "All retained recordings were checked; additional 15-second physics tests covered selected risk cases, not every retained task. No combined results were refiltered.",
        ],
    }
    save_json(output / "report.json", report)
    save_csv(output / "confirmed_scene_failures.csv", confirmed,
             ["task", "catalog_index", "target", "reasons", "recorded_episodes", "recorded_successes", "recorded_null_outcomes", "static_flags"])
    save_csv(output / "numerical_episode_failures.csv", numerical,
             ["task", "target", "env_id", "source", "json_success", "json_step", "nonfinite_fields", "target_max_displacement_from_first_sample_m", "onset"])
    save_csv(output / "extreme_motion_episodes.csv", extreme,
             ["task", "target", "env_id", "source", "json_success", "json_step", "target_max_displacement_from_first_sample_m", "flags", "onset"])
    task_rows = []
    for name, task in sorted(tasks.items()):
        dynamic = physics.get(name, {})
        task_rows.append({"task": name, "catalog_index": task["catalog_index"], "target": task["target"],
                          "confirmed_scene_reasons": [e["reason"] for e in scene_issues.get(name, [])],
                          "static_warnings": task["flags"], "tested_15s": name in physics,
                          "success_without_robot_motion": dynamic.get("success_without_motion"),
                          "first_success_time_s": dynamic.get("first_success_without_motion_s"),
                          "additional_fall": dynamic.get("fell_below_table"),
                          "first_additional_fall_time_s": dynamic.get("first_below_table_time_s"),
                          "left_table_bounds_15s": dynamic.get("left_table_bounds"),
                          "moving_at_15s": dynamic.get("moving_at_end"),
                          "first_gripper_contact_s": dynamic.get("first_gripper_contact_s"),
                          "null_recorded_episodes": sum(e["json_success"] is None for e in by_task[name]),
                          "nonfinite_recorded_episodes": sum("nonfinite_state_or_action_data" in e["flags"] for e in by_task[name])})
    save_csv(output / "tasks.csv", task_rows, list(task_rows[0]))
    lines = [f"Additional failure audit: {len(confirmed)} retained tasks have a confirmed scene issue in at least one no-action run.", "",
             f"Checked all {len(kept):,} retained tasks' {len(episodes):,} saved episodes, and ran {len(physics)} selected placements for 15 seconds "
             "with the robot held still and the generated success predicate/contact sensors enabled.", "",
             "| Finding | Count | Evidence |", "| --- | ---: | --- |",
             f"| Additional falls | {len(falls)} tasks | Latest began at {latest_fall:.3f}s; {len(late_falls)} began after 5s. |",
             f"| Success without robot motion | {len(spontaneous)} tasks | Exact task predicate became true with no gripper contact. |",
             f"| Entire object off tabletop, missed by floor-fall rule | {len(entirely_off)} tasks | Conservative collision bounds at 5 seconds were wholly outside tabletop XY bounds. |",
             f"| Null outcomes with repeated resets | {len(null_names)} tasks / {report['counts']['null_outcome_episodes']} episodes | HDF success=true, 501 saved initial states, JSON success=null. |",
             f"| Nonfinite recorded states/actions | {len(numerical)} episodes / {report['counts']['nonfinite_recording_tasks']} tasks | Robot states contain NaNs; some subsequent actions do too. All were recorded as failures. |",
             f"| Extreme recorded object travel | {len(extreme)} episodes / {report['counts']['over_5m_recorded_displacement_tasks']} tasks | More than 5 m from the first recorded sample; review candidates, overlapping the NaN cases. |", "",
             "Confirmed scene cases (categories overlap):", "",
             "| Task | Evidence | Recorded successes / episodes |", "| --- | --- | ---: |"]
    for row in confirmed:
        lines.append(f"| {row['task']} | {'; '.join(row['reasons'])} | {row['recorded_successes']} / {row['recorded_episodes']} |")
    lines += ["", "The original launch-fall exclusion did not catch Bagel16 on the robot-side fixture or Lunchbag45's upward flyoff. "
              "Lunchbag45 reached a root height of approximately 81.9 m at 5 seconds. Both were stable in this follow-up, so these are observed intermittent failures.", "",
              "The four null-outcome cases are WhitepackerbottleA0107, WhitepackerbottleA0130, WhitepackerbottleA0248, and SoftScrub30. "
              "`robolab/core/environments/env.py` resets terminations at episode length <= 2 without saving a result; these already-successful starts repeatedly take that path. "
              "The HDF success flags do not represent valid policy completions.", "",
              "The generated success termination checks current containment/contact and gripper detachment; it does not require a previous pickup or stationary placement. "
              "This permits spontaneous success from overlapping spawns. Butter13 is a new no-action observation; all five of its original recorded policy episodes failed.", "",
              "Numerical failure recordings concern orange_01/orange_02. In representative episodes, extreme target/robot states appeared before NaN actions, "
              "which is consistent with simulation instability feeding invalid observations back to the controller. The original failure labels remain valid, "
              "but these are poor episodes for judging normal manipulation performance. No replay or root-cause proof is claimed.", "",
              "Remaining geometry warnings: " + "; ".join(f"{k}: {v}" for k, v in report["remaining_static_warning_counts"].items()) + ". "
              "These warnings and ordinary residual movement alone are insufficient reasons to exclude a task.", "",
              "Original and previously filtered combined results were verified unchanged by SHA-256. This audit does not change the 57.49% filtered success summary. "
              "The 15-second follow-up covers selected risk cases; it is not a guarantee that every other placement is stable.", "",
              "Files: `confirmed_scene_failures.csv`, `numerical_episode_failures.csv`, `extreme_motion_episodes.csv`, `tasks.csv`, "
              "`record_integrity.json`, and `report.json` contain the full evidence and coverage."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"scope": report["scope"], "counts": report["counts"], "preserved_results": unchanged}))


if __name__ == "__main__":
    main()
