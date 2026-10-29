#!/usr/bin/env python3
"""Resume a bounded two-GPU MimicGen collection and curate its diverse subset."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robolab.core.omy_postprocess.validation import validate_recording
from robolab.core.omy_teleop.diversity import curate, descriptor, yaw_features
from robolab.core.omy_teleop.quality import assess
from robolab.core.omy_teleop.dataset_paths import job_recording, named_output, source_names


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_key(row):
    return (str((Path(row["input"]) / "omy_demos.hdf5").resolve()), row["episode"])


def process_job(root, job, gpu, thresholds):
    """One isolated child; output is independently checked after the child exits."""
    import h5py

    folder = root / "jobs" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        payload = job_recording(root, saved)
        artifacts = saved.get("artifact_sha256", {})
        if not artifacts and saved.get("sha256"):
            artifacts = {"omy_demos.hdf5": saved["sha256"]}
        for name, expected in artifacts.items():
            path = payload.parent / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Completed job artifact changed: {path}")
        return saved
    output = job_recording(root, job).parent
    if output.exists():
        raise RuntimeError(f"Unfinalized job output requires explicit recovery: {output}")
    command = [sys.executable, "-B", "-u", "examples/generate_omy_mimic.py", "--worker-exit", "--host-gpu", str(gpu),
               "--input", job["input"], "--target-input", job["input"], "--target-episode", job["episode"].split("_")[-1],
               "--output-dir", str(output), "--num-demos", str(job["num_demos"]),
               "--max-attempts", str(job["max_attempts"]), "--seed", str(job["seed"])]
    if job.get("sources"):
        source_selection = folder / "sources.json"
        atomic_json(source_selection, {"sources": job["sources"]})
        command += ["--source-selection", str(source_selection)]
    else:
        command += ["--episodes", job["episode"].split("_")[-1]]
    command += ["--selection-strategy", job.get("selection_strategy", "random"),
                "--nn-k", str(job.get("nn_k", 3)), "--selection-seed", str(job.get("selection_seed", 0))]
    if job.get("trial_manifest"):
        command += ["--trial-manifest", job["trial_manifest"]]
    if job.get("follow_source_schedule"):
        command += ["--follow-source-schedule"]
    if job.get("fixed_source_index") is not None:
        command += ["--fixed-source-index", str(job["fixed_source_index"])]
    if job["stage"] == "validate":
        command += ["--no-randomize", "--keep-failed"]
    else:
        if job.get("no_randomize"):
            command += ["--no-randomize"]
        if job.get("keep_failed"):
            command += ["--keep-failed"]
        for key, value in job["ranges"].items():
            command += ["--" + key.replace("_", "-"), str(value)]
    implementation = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                      for name in ("robolab/core/omy_teleop/mimicgen.py", "robolab/core/omy_teleop/mimic_sources.py")}
    atomic_json(folder / "job.json", {**job, "gpu": gpu, "command": command, "implementation": implementation})
    started = time.time()
    with (folder / "run.log").open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                 env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "Y", "PYTHONUNBUFFERED": "1"})
        atomic_json(folder / "process.json", {"pid": child.pid, "gpu": gpu, "started": started})
        try:
            code = child.wait(timeout=3600)
        except KeyboardInterrupt:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGINT)
            try:
                code = child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
    summary_path = output / "mimicgen_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"attempts": []}
    result = {**job, "gpu": gpu, "implementation": implementation, "returncode": code, "wall_s": time.time() - started,
              "attempts": len(summary["attempts"]), "accepted": [], "rejections": {}, "errors": [], "warnings": []}
    finalized_path = output / "generation_finalized.json"
    if finalized_path.exists():
        finalized = json.loads(finalized_path.read_text())
        result["finalized"] = finalized
        if finalized.get("generation_exit_status") not in (0,1):
            result["errors"].append("Generation was interrupted before completing its trial loop")
        elif code not in (0,1):
            result["warnings"].append(f"Worker exited {code} after recording finalization; outputs independently verified")
    else:
        result["errors"].append("Missing generation finalization record")
    path = output / "omy_demos.hdf5"
    if path.exists():
        try:
            quality = assess(path, thresholds, summary)
            validation = validate_recording(path)
            atomic_json(output / "quality.json", quality)
            atomic_json(output / "validation.json", validation)
            valid = {e["name"]:e for e in validation["episodes"]}
            with h5py.File(path, "r") as stream:
                meta = json.loads(stream["data"].attrs["robolab_omy_mimicgen"])
                for name, item in quality.items():
                    reasons = list(item["failed"])
                    if validation["errors"]:
                        reasons.append("recording_structure")
                    if not valid[name]["structural_valid"] or not valid[name]["native_15_hz"]:
                        reasons += [e["code"] for e in valid[name]["errors"]] or ["invalid_structure_or_cadence"]
                    generated = next((a for a in summary["attempts"] if a.get("demo") == name), {})
                    if generated.get("ik_position_error_m") is None or generated.get("events") is None:
                        reasons.append("missing_generation_checks")
                    try:
                        descriptor(stream["data"][name], meta["object"], meta["container"])
                    except (ValueError, KeyError) as error:
                        reasons.append("diversity_features:" + str(error))
                    if reasons:
                        result["rejections"][name] = sorted(set(reasons))
                    else:
                        result["accepted"].append(name)
            result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            result["errors"].append(traceback.format_exc())
    elif not summary["attempts"]:
        result["errors"].append("Worker produced no finalized recording or attempt summary")
    result["artifact_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                for name in ("omy_demos.hdf5", "env_cfg.json", "mimicgen_summary.json", "generation_finalized.json")
                                if (output / name).is_file()}
    atomic_json(result_path, result)
    print(f"{job['id']}: {len(result['accepted'])} accepted / {result['attempts']} attempts; "
          f"exit={code}, {result['wall_s']:.1f}s", flush=True)
    return result


def novel_cells(root, results):
    """Coarse scene bins measure spatial saturation without rewarding numerical noise."""
    import h5py
    cells = set()
    additions = []
    import numpy as np
    for result in results:
        path = job_recording(root, result)
        if not result["accepted"]:
            continue
        with h5py.File(path, "r") as stream:
            meta = json.loads(stream["data"].attrs["robolab_omy_mimicgen"])
            for name in result["accepted"]:
                initial = stream["data"][name]["initial_state/rigid_object"]
                obj = initial[meta["object"]]["root_pose"][0]
                container = initial[meta["container"]]["root_pose"][0]
                angle = np.arctan2(*yaw_features(obj[3:]))
                cell = (*np.floor(np.r_[obj[:2], container[:2]] / .05).astype(int),
                        int(np.floor((angle + np.pi) / np.deg2rad(15))))
                additions.append(cell not in cells)
                cells.add(cell)
    return {"occupied_scene_bins": len(cells), "new_bins_last_50": int(sum(additions[-50:])),
            "saturated": len(additions) >= 300 and sum(additions[-50:]) < 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    lock = (root / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((root / "run_config.json").read_text())
    sources = json.loads((root / "source_manifest.json").read_text())["sources"]
    for row in sources:
        path = Path(row["input"]) / "omy_demos.hdf5"
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError(f"Source changed: {path}")
    results = [json.loads(p.read_text()) for p in sorted((root / "jobs").glob("*/result.json"))]
    next_index = max([int(r["id"].split("_")[0]) for r in results] + [-1]) + 1
    active = {}
    started = time.time()

    def snapshot(status):
        generated = Counter()
        for r in results:
            if r["stage"] == "generate":
                generated[r["family"]] += len(r["accepted"])
        atomic_json(root / "status.json", {"status": status, "updated_unix_s": time.time(),
            "supervisor_pid": os.getpid(), "elapsed_s": time.time() - started,
            "completed_jobs": len(results), "generation_candidates": dict(generated),
            "validated_sources": len({source_key(r) for r in results if r["stage"] == "validate" and r["accepted"]}),
            "failed_sources": [r["task"] for r in results if r["stage"] == "validate" and not r["accepted"]],
            "active": [{"gpu":gpu,"job":job} for gpu,job in active.values()]})

    def next_job():
        nonlocal next_index
        active_tasks = {job["task"] for _,job in active.values()}
        validated = {source_key(r) for r in results if r["stage"] == "validate"}
        pending = [s for s in sources if source_key(s) not in validated and s["task"] not in active_tasks]
        if pending:
            # Alternate object families as early as possible in qualification.
            family_seen = Counter(r["family"] for r in results if r["stage"] == "validate")
            family_seen.update(job["family"] for _,job in active.values() if job["stage"] == "validate")
            source = min(pending, key=lambda s: (family_seen[s["family"]], s["task"]))
            stage, tier, target, budget = "validate", "baseline", 3, 10
        else:
            eligible = {source_key(r) for r in results if r["stage"] == "validate" and r["accepted"]}
            candidates = []
            for source in sources:
                task = source["task"]
                if source_key(source) not in eligible or task in active_tasks:
                    continue
                task_results = [r for r in results if r["task"] == task]
                attempts = sum(r["attempts"] for r in task_results)
                remaining = config["max_attempts_per_layout"] - attempts
                if remaining < 1 or any(r["errors"] and not r.get("resolved_by") for r in task_results):
                    continue
                family_results = [r for r in results if r["family"] == source["family"] and r["stage"] == "generate"]
                total = sum(len(r["accepted"]) for r in family_results)
                reserved = sum(job["num_demos"] for _,job in active.values()
                               if job["family"] == source["family"] and job["stage"] == "generate")
                if total + reserved >= config["max_candidates_per_family"]:
                    continue
                if total >= config["initial_candidates_per_family"] and novel_cells(root, family_results)["saturated"]:
                    continue
                tiers = []
                for priority, (t, weight) in enumerate([("near",.5),("moderate",.25),("wide",.25)]):
                    previous = [r for r in task_results if r["tier"] == t]
                    if sum(r["attempts"] for r in previous) >= 20 and not any(r["accepted"] for r in previous):
                        continue
                    accepted = sum(len(r["accepted"]) for r in previous)
                    tiers.append((accepted / weight, priority, t))
                if not tiers:
                    continue
                local = sum(len(r["accepted"]) for r in task_results if r["stage"] == "generate")
                candidates.append((total, local, source, min(tiers)[2], remaining))
            if not candidates:
                return None
            _, _, source, tier, remaining = min(candidates, key=lambda row: (row[0], row[1], row[2]["task"]))
            stage = "generate"
            current = sum(len(r["accepted"]) for r in results if r["family"] == source["family"] and r["stage"] == stage)
            reserved = sum(job["num_demos"] for _,job in active.values()
                           if job["family"] == source["family"] and job["stage"] == stage)
            target = min(8, remaining, config["max_candidates_per_family"] - current - reserved)
            budget = min(remaining, max(target, 24))
        job = {**{k:source[k] for k in ("family","task","input","episode")},
               "id": f"{next_index:05d}_{source['task']}_{tier}", "stage":stage,"tier":tier,
               "seed": 1000000 + next_index * 1000, "num_demos":target,"max_attempts":budget,
               "ranges":config["tiers"].get(tier,{})}
        relative = named_output(source["input"], source["task"], job["id"])
        job["output_dir"] = str(Path(*relative.parts[1:]) if root.name == source_names(source["input"], source["task"])[0]
                                else Path("recordings") / relative)
        if stage == "generate" and config.get("source_pool", False):
            job["sources"] = [{"file": str(Path(s["input"]) / "omy_demos.hdf5"),
                               "episode": s["episode"], "sha256": s["sha256"]}
                              for s in sources if s["family"] == source["family"] and source_key(s) in eligible]
            job["selection_strategy"] = config.get("source_selection_strategy", "random")
            job["nn_k"] = config.get("source_selection_nn_k", 3)
            job["selection_seed"] = job["seed"] + 500000
        next_index += 1
        return job

    with ThreadPoolExecutor(max_workers=len(config["host_gpus"])) as pool:
        while True:
            busy = {gpu for gpu,_ in active.values()}
            for gpu in config["host_gpus"]:
                if gpu in busy:
                    continue
                job = next_job()
                if job is not None:
                    future = pool.submit(process_job,root,job,gpu,config["quality_thresholds"])
                    active[future] = (gpu,job)
            snapshot("running")
            if not active:
                break
            complete,_ = wait(active,timeout=15,return_when=FIRST_COMPLETED)
            for future in complete:
                active.pop(future)
                results.append(future.result())
                atomic_json(root / "batch_summary.json",results)
            # A broken worker setup stops the batch instead of repeating 63 identical failures.
            if sum(bool(r["errors"]) for r in results[-3:]) >= 3:
                raise RuntimeError("Three consecutive jobs failed setup/validation; see batch_summary.json")
    selected = curate(root,config["selected_per_family"])
    wanted = len({s["family"] for s in sources}) * config["selected_per_family"]
    snapshot("selected" if selected["selected_count"] == wanted else "incomplete")
    print(f"Selected {selected['selected_count']} / {wanted}. See {root / 'selection.json'}",flush=True)
    # Selection is the admission boundary: render only this manifest's episodes.
    # An already-running video service will discover it on its next refresh.
    video_command = [sys.executable, "-B", "-u", "scripts/serve_demo_library.py", "--worker", "--once",
                     "--run", str(root), "--generated-root", str(root.parent), "--state-dir",
                     str(config.get("video_state_dir", Path(__file__).resolve().parents[1] / "output/demo_library")), "--gpus",
                     *[str(gpu) for gpu in config["host_gpus"]]]
    video_code = subprocess.call(video_command, cwd=Path(__file__).resolve().parents[1])
    if video_code:
        print(f"Selected video rendering needs attention (exit {video_code}); see video_status.json", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        if "--run" in sys.argv:
            run_root = Path(sys.argv[sys.argv.index("--run") + 1])
            atomic_json(run_root / "status.json", {"status":"failed", "error":traceback.format_exc(),
                                                   "updated_unix_s":time.time()})
        raise SystemExit(1)
