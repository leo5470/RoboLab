#!/usr/bin/env python3
"""Complete exact-size mixed OMY pools and optionally their camera training files."""

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_omy_mimic_collection as batch
from robolab.core.omy_postprocess.automatic_dataset import discover_sessions, discover_recordings, locked, process_one
from robolab.core.omy_postprocess.io import publish_directory, sha256_file
from robolab.core.omy_postprocess.split_batch import copy_artifact
from robolab.core.omy_teleop.final_dataset import gather_candidates, choose, export
from robolab.core.omy_teleop.quality import THRESHOLDS

LIMITS = {k: v[0] for k, v in THRESHOLDS.items()}


def read(path):
    return json.loads(Path(path).read_text())


def adopt_pilot(source, target):
    if target.exists():
        return
    if not source.is_dir():
        raise ValueError(f"Missing generation sources and qualification evidence: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".adopt_{target.name}_", dir=target.parent))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True, copy_function=copy_artifact)
        batch.atomic_json(temporary / "adopted_from.json", {"source": str(source),
            "source_manifest_sha256": sha256_file(source / "source_manifest.json"),
            "payload_policy": "immutable payload hard links; independent JSON metadata"})
        publish_directory(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_pool(output, count):
    manifest = read(output / "dataset.json")
    rows = manifest["episodes"]
    if manifest["selected_count"] != count or len(rows) != count or len({r["fingerprint"] for r in rows}) != count:
        raise ValueError(f"Pool count or identity mismatch: {output}")
    if manifest["family"] == "wooden_bowl":
        raise ValueError("Wooden bowl is excluded")
    observed = Counter(r["kind"] for r in rows)
    if observed["collected"] > min(10, count) or dict(observed) != manifest["counts_by_kind"]:
        raise ValueError("Pool origin counts are inconsistent")
    checked = set()
    for row in rows:
        file = output / row["file"]
        if file not in checked:
            if sha256_file(file) != row["sha256"]:
                raise ValueError(f"Pool checksum mismatch: {file}")
            if sha256_file(file.parent / "env_cfg.json") != row["config_sha256"]:
                raise ValueError(f"Pool configuration checksum mismatch: {file}")
            checked.add(file)
    return manifest


def training_plan(pool, collected, generated, cache, count):
    options = {"session": pool["session"], "count": count, "collected_count": None,
               "selection_policy": "up_to_10_humans_v1", "collected_root": str(collected),
               "generated_root": str(generated), "format": "omy_training_hdf5_v1", "camera_profile": "native_realtime_2"}
    rows = []
    for item in pool["episodes"]:
        row = {**item, "file": item["source_file"]}
        cfg = read(Path(row["file"]).parent / "env_cfg.json")
        row["task_instruction"] = cfg.get("instruction") or row["task"]
        rows.append(row)
    plan = {"options": options, "family": pool["family"], "episodes": rows,
            "quality_review": pool["quality_review"], "rejected": pool.get("rejected", [])}
    folder = cache / pool["session"]
    with locked(folder / ".processing.lock"):
        path = folder / "selection.json"
        if path.exists():
            previous = read(path)
            if previous["options"] != options or {r["fingerprint"] for r in previous["episodes"]} != {r["fingerprint"] for r in rows}:
                raise ValueError(f"Training cache selection differs from the completed pool: {path}")
        else:
            batch.atomic_json(path, plan)


def complete_pool(session, collected, generated, study, output, *, count=50, gpu=1, max_attempts=200):
    target = output / session
    if target.exists():
        return verify_pool(target, count)
    folder = generated / session
    if not folder.exists():
        adopt_pilot(study / "generated" / session, folder)
    used = 0
    while True:
        with locked(collected / session / ".session.lock"):
            recordings, family, reviews = discover_recordings(collected, generated, session)
            if family == "wooden_bowl":
                raise ValueError("Wooden bowl is excluded")
            rows, rejected = gather_candidates(recordings, family, LIMITS)
        humans = min(10, count, sum(r["kind"] == "collected" for r in rows))
        generated_count = sum(r["kind"] == "generated" for r in rows)
        needed = max(0, count - humans - generated_count)
        print(f"{session}: eligible human={sum(r['kind']=='collected' for r in rows)}, generated={generated_count}; need {needed} generated", flush=True)
        if not needed:
            selected, normalization = choose(rows, count)
            report = {"schema_version": 1, "status": "complete", "session": session, "family": family,
                      "selected_count": count, "counts_by_kind": dict(Counter(r["kind"] for r in selected)),
                      "candidate_count": len(rows), "quality_review": reviews, "rejected": rejected,
                      "normalization": normalization, "thresholds": LIMITS, "training_ready": False,
                      "method": "up to 10 eligible humans; layout-balanced farthest-first generated fill"}
            export(target, selected, report)
            result = verify_pool(target, count)
            print(f"{session}: raw pool complete: {target / 'dataset.json'}", flush=True)
            return result
        sources = read(folder / "source_manifest.json")["sources"]
        results = [read(p) for p in (folder / "jobs").glob("*/result.json")]
        qualified = {batch.source_key(r) for r in results if r["stage"] == "validate" and r["accepted"] and not r["errors"]}
        pool = [s for s in sources if batch.source_key(s) in qualified]
        if not pool:
            raise ValueError(f"No qualified generation sources for {session}")
        fills = [r for r in results if r.get("completion50")]
        used = sum(r["attempts"] for r in fills)
        if used >= max_attempts:
            raise RuntimeError(f"Completion attempt budget {max_attempts} exhausted with {needed} candidates still missing")
        task_attempts = Counter()
        for result in results:
            task_attempts[result["task"]] += result["attempts"]
        source = min(pool, key=lambda s: (task_attempts[s["task"]], s["task"], s["episode"]))
        index = len(fills)
        name = f"{90000 + index:05d}_{source['task']}_pool_near"
        pilot = read(study / "report.json")[session]
        strategy = pilot["pilot_winner"]
        budget = min(24, max(8, needed * 3), max_attempts - used)
        successes = min(8, needed, budget)
        job = {"id": name, "family": family, "task": source["task"], "input": source["input"], "episode": source["episode"],
               "sources": [{"file": str(Path(s["input"]) / "omy_demos.hdf5"), "episode": s["episode"], "sha256": s["sha256"]} for s in pool],
               "stage": "generate", "tier": "near", "completion50": True, "num_demos": successes, "max_attempts": budget,
               "seed": 700000 + index * 1000, "selection_seed": 1700000 + index * 1000,
               "selection_strategy": strategy, "nn_k": 3, "keep_failed": True,
               "output_dir": str(Path(source["task"]) / name / Path(source["input"]).name),
               "ranges": {"object_pose_range": .025, "object_yaw_range": .0872664626, "pose_range": .01, "yaw_range": 0.}}
        result = batch.process_job(folder, job, gpu, LIMITS)
        if result["errors"]:
            raise RuntimeError(f"Generation setup/validation failed: {folder / 'jobs' / name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected-root", type=Path, default=Path("output/omy_teleop"))
    parser.add_argument("--generated-root", type=Path, default=Path("output/omy_mimicgen"))
    parser.add_argument("--study-root", type=Path, default=Path("output/omy_mimic_tests/multisource_pilot_20261028"))
    parser.add_argument("--output-root", type=Path, default=Path("output/omy_final_pool50"))
    parser.add_argument("--training-root", type=Path, default=Path("output/omy_training"))
    parser.add_argument("--cache-root", type=Path, default=Path("output/omy_training_cache"))
    parser.add_argument("--state-dir", type=Path, default=Path("output/omy_pool_completion"))
    parser.add_argument("--host-gpu", type=int, default=1)
    parser.add_argument("--max-additional-attempts", type=int, default=200)
    parser.add_argument("--render", action="store_true", help="Also publish the selected pool as minimal camera training HDF5")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir / "completion.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sessions = [s for s in discover_sessions(args.collected_root, args.generated_root)
                if s[:8].isdigit() and not s.endswith("_wooden_bowl")]
    summary = {"schema_version": 1, "excluded_families": ["wooden_bowl"], "sessions": {}, "status": "processing"}
    def save():
        batch.atomic_json(args.state_dir / "summary.json", summary)
    save()
    for session in sessions:
        item = summary["sessions"][session] = {"status": "completing_pool"}
        save()
        try:
            pool = complete_pool(session, args.collected_root, args.generated_root, args.study_root, args.output_root,
                                 gpu=args.host_gpu, max_attempts=args.max_additional_attempts)
            training_plan(pool, args.collected_root, args.generated_root, args.cache_root, 50)
            item.update(status="pool_complete", family=pool["family"], selected_count=pool["selected_count"],
                        counts_by_kind=pool["counts_by_kind"], pool=str(args.output_root / session / "dataset.json"))
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            item.update(status="failed", error=str(error))
        save()
    if args.render:
        for session, item in summary["sessions"].items():
            if item["status"] != "pool_complete":
                continue
            item["status"] = "processing_cameras"
            save()
            try:
                process_one(session, args.collected_root, args.generated_root, args.training_root, args.cache_root, device=f"cuda:{args.host_gpu}")
                item.update(status="complete", training_file=str(args.training_root / session / "training.hdf5"))
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
                item.update(status="camera_processing_failed", error=str(error))
            save()
    good = ("complete",) if args.render else ("pool_complete",)
    summary["status"] = "complete" if all(s["status"] in good for s in summary["sessions"].values()) else "incomplete"
    save()
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
