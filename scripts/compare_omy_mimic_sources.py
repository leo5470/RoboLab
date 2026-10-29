#!/usr/bin/env python3
"""Qualify human sources and compare MimicGen strategies on paired scene trials."""

import argparse
from collections import Counter
import importlib.util
import json
import fcntl
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robolab.core.omy_postprocess.automatic_dataset import discover_recordings
from robolab.core.omy_teleop.final_dataset import gather_candidates
from robolab.core.omy_teleop.mimic_sources import checksum, contract, STRATEGIES
from robolab.core.omy_teleop.quality import THRESHOLDS

spec = importlib.util.spec_from_file_location("omy_batch", Path(__file__).with_name("run_omy_mimic_collection.py"))
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)
LIMITS = {k: v[0] for k, v in THRESHOLDS.items()}
POSITION_TOLERANCE_M = 1e-4
ANGLE_TOLERANCE_RAD = float(np.deg2rad(.05))


def validate_options(previous, requested):
    core = lambda row: {k: v for k, v in row.items() if k != "confirm_attempts"}
    if core(previous) != core(requested):
        raise ValueError("Experiment options changed; use a fresh output")
    old, new = previous.get("confirm_attempts", 0), requested.get("confirm_attempts", 0)
    if old != new and not (old == 0 and new > 0):
        raise ValueError("A started confirmation budget cannot change; use a fresh output")


def job_for(source, session, family, name, attempts):
    return {"id": name, "family": family, "task": source["task"], "input": str(Path(source["file"]).parent),
            "episode": source["episode"], "num_demos": attempts, "max_attempts": attempts,
            "seed": 12340, "selection_seed": 98760, "stage": "generate", "tier": name, "keep_failed": True,
            "output_dir": str(Path(source["task"]) / name / Path(source["file"]).parent.name),
            "ranges": {"object_pose_range": .025, "object_yaw_range": .0872664626,
                       "pose_range": .01, "yaw_range": 0.}}


def summarize(root, result):
    path = batch.job_recording(root, result)
    summary_path = path.parent / "mimicgen_summary.json"
    attempts = json.loads(summary_path.read_text())["attempts"] if summary_path.exists() else []
    errors = [a["ik_position_error_m"] for a in attempts if a.get("ik_position_error_m") is not None]
    angles = [a["ik_orientation_error_rad"] for a in attempts if a.get("ik_orientation_error_rad") is not None]
    successes = sum(a.get("success", False) for a in attempts)
    sources = Counter(f"{a['source_reference']['file']}:{a['source_reference']['episode']}"
                      for a in attempts if a.get("source_reference"))
    metrics = {"attempts": len(attempts), "successes": successes, "accepted": len(result["accepted"]),
               "timing_scope": "job wall time includes startup and GPU queue waits",
               "setup_rejections": sum(a.get("preparation", {}).get("status") != "ready" for a in attempts),
               "success_rate": successes / len(attempts) if attempts else 0,
               "acceptance_rate": len(result["accepted"]) / len(attempts) if attempts else 0,
               "accepted_per_minute": len(result["accepted"]) * 60 / max(result["wall_s"], .001),
               "ik_position_max_m": max(errors, default=None), "ik_orientation_max_rad": max(angles, default=None),
               "ik_position_median_m": float(np.median(errors)) if errors else None,
               "rejection_counts": dict(Counter(reason for reasons in result["rejections"].values() for reason in reasons)),
               "source_counts": dict(sources), "errors": result["errors"], "wall_s": result["wall_s"]}
    return metrics, attempts


def paired_poses(arms):
    baseline = arms["baseline"]
    differences = []
    for strategy, trials in arms.items():
        if len(trials) != len(baseline):
            differences.append({"strategy": strategy, "reason": "trial count mismatch"})
            continue
        for ref, trial in zip(baseline, trials):
            a, b = ref.get("prepared_initial_poses", {}), trial.get("prepared_initial_poses", {})
            if (trial["trial_id"] != ref["trial_id"] or not a or a.keys() != b.keys()
                    or ref["preparation"]["status"] != trial["preparation"]["status"]):
                differences.append({"strategy": strategy, "trial_id": trial["trial_id"], "reason": "prepared scene mismatch"})
                continue
            for name in a:
                pa, pb = np.asarray(a[name], dtype=float), np.asarray(b[name], dtype=float)
                qa, qb = pa[3:], pb[3:]
                position = float(np.linalg.norm(pa[:3] - pb[:3]))
                norms = np.linalg.norm(qa) * np.linalg.norm(qb)
                angle = float(2 * np.arccos(np.clip(abs(np.dot(qa, qb)) / norms, 0, 1))) if norms > 0 else float("inf")
                if not np.isfinite(position + angle) or position > POSITION_TOLERANCE_M or angle > ANGLE_TOLERANCE_RAD:
                    differences.append({"strategy": strategy, "trial_id": trial["trial_id"], "object": name,
                                        "position_error_m": position, "angle_error_rad": angle})
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", action="append")
    parser.add_argument("--host-gpu", type=int, default=1)
    parser.add_argument("--limit-sources", type=int)
    parser.add_argument("--qualify-attempts", type=int, default=2)
    parser.add_argument("--pilot-attempts", type=int, default=10)
    parser.add_argument("--confirm-attempts", type=int, default=0,
                        help="Additional attempts for baseline and pilot winner; use 50 for confirmation")
    args = parser.parse_args()
    if args.qualify_attempts < 1 or args.pilot_attempts < 1 or args.confirm_attempts < 0 or (args.limit_sources is not None and args.limit_sources < 1):
        parser.error("Attempt counts and source limits must be positive; confirmation may be zero")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "experiment.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    options_path = root / "options.json"
    if options_path.exists():
        validate_options(json.loads(options_path.read_text()), options)
    batch.atomic_json(options_path, options)
    sessions = args.session or [p.name for p in sorted(Path("output/omy_teleop").glob("20261028*"))]
    report = {}
    for session in sessions:
        folder = root / "generated" / session
        folder.mkdir(parents=True, exist_ok=True)
        snapshot = folder / "source_manifest.json"
        if snapshot.exists():
            saved = json.loads(snapshot.read_text())
            sources, family = saved["sources"], saved["family"]
        else:
            recordings, family, _ = discover_recordings(Path("output/omy_teleop"), Path("output/omy_mimicgen"), session)
            recordings = {key: origin for key, origin in recordings.items() if origin["kind"] == "collected"}
            rows, rejected = gather_candidates(recordings, family, LIMITS)
            sources = [{"file": r["file"], "episode": r["episode"], "sha256": r["source_sha256"],
                        "task": r["task"], "family": family, "input": str(Path(r["file"]).parent)} for r in rows]
            if args.limit_sources:
                sources = sources[:args.limit_sources]
            batch.atomic_json(snapshot, {"sources": sources, "family": family, "rejected": rejected})
        target_contract = contract(Path(sources[0]["file"]).parent)
        if any(contract(Path(s["file"]).parent) != target_contract for s in sources):
            raise ValueError(f"Incompatible source contracts in {session}")
        qualified = []
        entry = report[session] = {"qualification": {}, "pilot": {}}
        for i, source in enumerate(sources):
            if checksum(source["file"]) != source["sha256"]:
                raise ValueError("Source recording changed")
            job = job_for(source, session, family, f"qualify_{i:02}", args.qualify_attempts)
            job.update(stage="validate", sources=[source], fixed_source_index=0)
            result = batch.process_job(folder, job, args.host_gpu, LIMITS)
            entry["qualification"][str(i)] = summarize(folder, result)[0]
            if result["accepted"] and not result["errors"]:
                qualified.append(source)
            batch.atomic_json(root / "report.json", report)
            if result["errors"]:
                raise RuntimeError(f"Setup/validation failed; inspect {folder / 'jobs' / job['id']}")
        entry["qualified_sources"] = len(qualified)
        if len(qualified) < 2:
            entry["status"] = "insufficient_qualified_sources"
            batch.atomic_json(root / "report.json", report)
            continue
        arms = {}
        target = qualified[0]
        for phase, attempts in (("pilot", args.pilot_attempts), ("confirmation", args.confirm_attempts)):
            if not attempts:
                continue
            strategies = ["baseline", *STRATEGIES] if phase == "pilot" else ["baseline", entry["pilot_winner"]]
            trials = {"trials": [{"id": f"{phase}_{i:03}", "seed": 27180 + (10000 if phase == "confirmation" else 0) + i,
                                  "source_index": i % len(qualified)} for i in range(attempts)]}
            trial_path = folder / f"{phase}_trials.json"
            batch.atomic_json(trial_path, trials)
            arms = {}
            entry[phase] = {}
            for strategy in strategies:
                job = job_for(target, session, family, f"{phase}_{strategy}", attempts)
                job.update(sources=qualified, trial_manifest=str(trial_path), follow_source_schedule=strategy == "baseline",
                           selection_strategy="random" if strategy == "baseline" else strategy, nn_k=3)
                result = batch.process_job(folder, job, args.host_gpu, LIMITS)
                metrics, arms[strategy] = summarize(folder, result)
                entry[phase][strategy] = metrics
                batch.atomic_json(root / "report.json", report)
                if result["errors"]:
                    raise RuntimeError(f"Comparison worker failed: {job['id']}")
            differences = paired_poses(arms)
            entry["pose_tolerances"] = {"position_m": POSITION_TOLERANCE_M, "angle_rad": ANGLE_TOLERANCE_RAD}
            entry[phase + "_pose_mismatches"] = differences
            batch.atomic_json(root / "report.json", report)
            if differences:
                raise RuntimeError("Target scenes were not matched; do not interpret comparison results")
            if phase == "pilot":
                entry["pilot_winner"] = max(STRATEGIES, key=lambda name: (
                    entry[phase][name]["accepted"], entry[phase][name]["successes"],
                    -(entry[phase][name]["ik_position_max_m"] if entry[phase][name]["ik_position_max_m"] is not None
                      else float("inf"))))
        entry["status"] = "complete"
        entry["sources_unchanged"] = all(checksum(s["file"]) == s["sha256"] for s in sources)
        batch.atomic_json(root / "report.json", report)
    print(f"Comparison report: {root / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
