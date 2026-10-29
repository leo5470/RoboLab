"""Select an exact-size mixed session dataset without starting Isaac Sim."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np

from robolab.core.omy_postprocess.validation import validate_recording, _file_sha256
from .dataset_paths import job_recording, source_names
from .diversity import feature_matrix, resample, yaw_features
from .quality import THRESHOLDS, assess, generation_records


def read_json(path):
    return json.loads(Path(path).read_text())


def shared_descriptor(episode, obj, container):
    """Use measured joints and object motion available in both recording kinds."""
    initial = episode["initial_state/rigid_object"]
    states = episode["states/rigid_object"]
    start = initial[obj]["root_pose"][0]
    target = initial[container]["root_pose"][0]
    poses = np.concatenate((start[None], states[obj]["root_pose"][:]))
    joints = episode["obs/proprio_obs/arm_joint_pos"][:]
    blocks = [np.r_[start[:3], target[:3], yaw_features(start[3:]), yaw_features(target[3:])],
              resample(poses[:, :3] - target[:3]).ravel(), resample(joints).ravel(),
              np.r_[poses[-1, :3] - target[:3], poses[:, 2].max() - start[2],
                    resample(episode["actions"][:, -1:]).ravel()]]
    if any(not np.isfinite(b).all() for b in blocks):
        raise ValueError("Nonfinite comparison features")
    return blocks


def choose(rows, count, collected_count=None):
    """Keep up to ten eligible humans, then select complementary generated motion."""
    if count < 1 or len(rows) < count:
        raise ValueError(f"Need {count} unique quality-passing demos; have {len(rows)}")
    available = Counter(r["kind"] for r in rows)
    if collected_count is None:
        collected_count = min(available["collected"], 10, count)
    if not 0 <= collected_count <= count:
        raise ValueError("--collected-count must be between zero and --count")
    quotas = {"collected": collected_count, "generated": count - collected_count}
    if any(available[k] < n for k, n in quotas.items()):
        raise ValueError(f"Requested origin counts {quotas}; available {dict(available)}")
    features, normalization = feature_matrix([r["blocks"] for r in rows])
    chosen, task_counts, kind_counts = [], Counter(), Counter()
    distances = np.full(len(rows), np.inf)

    def add(index):
        chosen.append(index)
        task_counts[rows[index]["task"]] += 1
        kind_counts[rows[index]["kind"]] += 1
        np.minimum(distances, np.linalg.norm(features - features[index], axis=1), out=distances)

    for kind in ("collected", "generated"):
        pool = [i for i, r in enumerate(rows) if r["kind"] == kind]
        if quotas[kind] == len(pool):
            for index in pool:
                add(index)
            continue
        while kind_counts[kind] < quotas[kind]:
            eligible = [i for i in pool if i not in chosen]
            if not chosen:
                center = np.median(features[pool], axis=0)
                add(min(eligible, key=lambda i: (float(np.linalg.norm(features[i] - center)), i)))
            else:
                add(min(eligible, key=lambda i: (task_counts[rows[i]["task"]], -float(distances[i]), i)))
    return [rows[i] for i in chosen], normalization


def session_sources(run, session):
    sources = read_json(run / "source_manifest.json")["sources"]
    names = {source_names(s["input"], s["task"])[0] for s in sources}
    selected = [s for s in sources if source_names(s["input"], s["task"])[0] == session]
    if not selected:
        raise ValueError(f"Unknown session {session!r}; choose from {sorted(names)}")
    families = {s["family"] for s in selected}
    if len(families) != 1:
        raise ValueError(f"Session must identify one family; found {sorted(families)}")
    return selected, next(iter(families))


def candidates(run, session, collected_root=None):
    """Discover the session's original parts and only its generated descendants."""
    sources, family = session_sources(run, session)
    found = {}
    for source in sources:
        original = Path(source["input"])
        session, task, part = source_names(original, source["task"])
        if collected_root is not None:
            original = collected_root / session / task / part
        session_dir = original.parent.parent if original.parent.name == task else original
        paths = sorted(session_dir.glob("*/part_*/omy_demos.hdf5"))
        paths += [original / "omy_demos.hdf5"]
        for path in paths:
            if path.is_file():
                found[str(path.resolve())] = {"kind": "collected", "session": session}
    for result_path in sorted((run / "jobs").glob("*/result.json")):
        result = read_json(result_path)
        if result["family"] != family or result["stage"] != "generate":
            continue
        if result.get("errors") and not result.get("resolved_by"):
            continue
        path = job_recording(run, result).resolve()
        source_session, _, _ = source_names(result["input"], result["task"])
        if source_session != session:
            continue
        found[str(path)] = {"kind": "generated", "session": session, "job_id": result["id"],
                            "accepted": result["accepted"], "parent_source": result["input"],
                            "parent_episode": result["episode"]}
    return found


def gather(run, session, thresholds, collected_root=None):
    _, family = session_sources(run, session)
    return gather_candidates(candidates(run, session, collected_root), family, thresholds)


def gather_candidates(recordings, family, thresholds):
    """Assess a discovered recording index using the same mixed-data gates."""
    rows, rejected, seen = [], [], set()
    for filename, origin in recordings.items():
        path = Path(filename)
        try:
            cfg = read_json(path.parent / "env_cfg.json")
            params = cfg["terminations"]["success"]["params"]
            obj, container = params["object"], params["container"]
            if obj != family:
                continue
            config_sha = _file_sha256(path.parent / "env_cfg.json")
            validation = validate_recording(path)
            if validation["errors"]:
                raise ValueError(f"Recording validation failed: {validation['errors']}")
            task = validation["metadata"]["robolab_omy_teleop"]["task"]
            summary_path = path.parent / "mimicgen_summary.json"
            summary = read_json(summary_path) if summary_path.exists() else None
            quality = assess(path, thresholds, summary, grasped=[obj])
            records = generation_records(summary)
            with h5py.File(path, "r") as stream:
                for ep in validation["episodes"]:
                    name = ep["name"]
                    if origin.get("accepted") is not None and name not in origin["accepted"]:
                        continue
                    reasons = list(quality[name]["failed"])
                    if not ep["structural_valid"] or not ep["native_15_hz"] or ep["training_excluded"]:
                        reasons.append("invalid_structure_cadence_or_diagnostic")
                    if origin["kind"] == "generated" and any(
                            records.get(name, {}).get(k) is None
                            for k in ("ik_position_error", "ik_orientation_error", "events")):
                        reasons.append("missing_generation_checks")
                    fingerprint = ep["fingerprint"]
                    if fingerprint in seen:
                        reasons.append("duplicate_trajectory")
                    if reasons:
                        rejected.append({"file": filename, "episode": name, "reasons": reasons})
                        continue
                    try:
                        blocks = shared_descriptor(stream["data"][name], obj, container)
                    except (KeyError, ValueError) as error:
                        rejected.append({"file": filename, "episode": name, "reasons": [str(error)]})
                        continue
                    seen.add(fingerprint)
                    generated_source = next((a.get("source_reference") for a in (summary or {}).get("attempts", [])
                                             if a.get("demo") == name), None)
                    if generated_source is None and origin["kind"] == "generated":
                        stored = json.loads(stream["data"][name].attrs.get("mimicgen_source", "{}"))
                        generated_source = stored.get("source")
                        if generated_source is None:
                            refs = json.loads(stream["data"].attrs.get("robolab_omy_mimicgen", "{}")).get("sources", [])
                            attempt = next((a for a in (summary or {}).get("attempts", []) if a.get("demo") == name), {})
                            index = attempt.get("source", 0 if len(refs) == 1 else None)
                            if index is not None and 0 <= int(index) < len(refs):
                                generated_source = refs[int(index)]
                    rows.append({**{k: v for k, v in origin.items() if k != "accepted"},
                                 "file": filename, "episode": name, "family": family, "task": task,
                                 "fingerprint": fingerprint, "source_sha256": validation["source"]["sha256"],
                                 "config_sha256": config_sha, "num_samples": ep["sample_count"],
                                 "quality": quality[name], "warnings": ep["warnings"], "blocks": blocks,
                                 "generation_source": generated_source,
                                 "unavailable_checks": (["generation_ik", "generation_contacts"]
                                                        if origin["kind"] == "collected" else [])})
        except (OSError, KeyError, ValueError) as error:
            rejected.append({"file": filename, "reasons": [str(error)]})
    return rows, rejected


def export(output, rows, report):
    """Publish self-contained HDF5 shards atomically; never overwrite a dataset."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        shards = defaultdict(list)
        for row in rows:
            shards[row["file"]].append(row)
        exported = []
        for filename, entries in sorted(shards.items()):
            source, first = Path(filename), entries[0]
            if _file_sha256(source) != first["source_sha256"]:
                raise ValueError(f"Source changed after selection: {source}")
            if _file_sha256(source.parent / "env_cfg.json") != first["config_sha256"]:
                raise ValueError(f"Configuration changed after selection: {source}")
            relative = (Path(first["kind"]) / first["session"] / first["task"] /
                        first["source_sha256"][:16] / source.parent.name / "omy_demos.hdf5")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(source, "r") as src, h5py.File(target, "x") as dst:
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
                data = dst.create_group("data")
                for key, value in src["data"].attrs.items():
                    data.attrs[key] = value
                for row in entries:
                    src.copy(src["data"][row["episode"]], data, row["episode"])
                data.attrs["total"] = sum(r["num_samples"] for r in entries)
            for sidecar in ("env_cfg.json", "mimicgen_summary.json"):
                if (source.parent / sidecar).exists():
                    shutil.copy2(source.parent / sidecar, target.parent / sidecar)
            if _file_sha256(target.parent / "env_cfg.json") != first["config_sha256"]:
                raise ValueError(f"Configuration changed during export: {source}")
            checked = validate_recording(target)
            fingerprints = {e["name"]: e["fingerprint"] for e in checked["episodes"]}
            if checked["errors"] or any(fingerprints.get(r["episode"]) != r["fingerprint"] for r in entries):
                raise ValueError(f"Copied trajectory verification failed: {target}")
            if _file_sha256(source) != first["source_sha256"]:
                raise ValueError(f"Source changed during export: {source}")
            checksum = checked["source"]["sha256"]
            for row in entries:
                exported.append({**{k: v for k, v in row.items() if k not in ("blocks", "file")},
                                 "source_file": filename, "file": str(relative), "sha256": checksum})
        manifest = {**report, "episodes": exported}
        (staging / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-run", type=Path, required=True)
    parser.add_argument("--session", required=True, help="Original collection session folder name")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--collected-root", type=Path, help="Override root of original collection sessions")
    parser.add_argument("--collected-count", type=int, help="Optional exact collected quota; remainder generated")
    parser.add_argument("--output", type=Path, help="New dataset directory (required unless --dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Assess and select without writing a dataset")
    for name, (default, _, _) in THRESHOLDS.items():
        parser.add_argument(f"--max-{name.replace('_', '-')}", type=float, default=default)
    args = parser.parse_args(argv)
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run")
    if args.output is not None and args.output.exists() and not args.dry_run:
        parser.error("--output must be a new directory")
    thresholds = {k: getattr(args, f"max_{k}") for k in THRESHOLDS}
    if args.count < 1 or any(np.isnan(v) or v < 0 for v in thresholds.values()):
        parser.error("Count must be positive and thresholds nonnegative (inf disables a check)")
    run = args.mimic_run.resolve()
    _, family = session_sources(run, args.session)
    rows, rejected = gather(run, args.session, thresholds, args.collected_root)
    print(f"Eligible {args.session} ({family}): {dict(Counter(r['kind'] for r in rows))}; rejected {len(rejected)}", flush=True)
    selected, normalization = choose(rows, args.count, args.collected_count)
    prior = read_json(run / "selection.json") if (run / "selection.json").exists() else {}
    review = dict(prior.get("quality_review", {}))
    if review.get("report"):
        evidence = run / review.pop("report")
        review["source_report"] = str(evidence.resolve())
        if evidence.is_file():
            review["evidence"] = read_json(evidence)
    report = {"schema_version": 1, "session": args.session, "family": family, "selected_count": len(selected),
              "counts_by_kind": dict(Counter(r["kind"] for r in selected)),
              "counts_by_task": dict(Counter(r["task"] for r in selected)),
              "candidate_count": len(rows), "thresholds": thresholds, "rejected": rejected,
              "method": "up to 10 eligible humans; layout-balanced farthest-first generated fill",
              "normalization": normalization, "quality_review": review,
              "training_ready": False, "mimic_run": str(run)}
    print(f"Selected {len(selected)}: {report['counts_by_kind']}", flush=True)
    if not args.dry_run:
        export(args.output.resolve(), selected, report)
        print(f"Dataset: {args.output.resolve() / 'dataset.json'}")
    return 0
