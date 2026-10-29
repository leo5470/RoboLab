"""Reversible separation of a completed legacy batch into original sessions."""

from collections import Counter, defaultdict
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil

from .io import sha256_file, publish_directory
from .library import identity, save_json
from robolab.core.omy_teleop.dataset_paths import source_names, job_recording


def read(path):
    return json.loads(Path(path).read_text())


def rewrite(value, mapping):
    if isinstance(value, str):
        for old, new in sorted(mapping.items(), key=lambda pair: -len(pair[0])):
            if value == old or value.startswith(old + "/"):
                return new + value[len(old):]
        return value
    if isinstance(value, list):
        return [rewrite(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: rewrite(v, mapping) for k, v in value.items()}
    return value


def copy_artifact(source, target):
    # Immutable payloads share disk blocks; JSON is independently editable.
    if Path(source).suffix == ".json":
        shutil.copy2(source, target)
    else:
        os.link(source, target)
    return str(target)


def plan_split(root):
    root = Path(root).resolve()
    sources = read(root / "source_manifest.json")
    selection = read(root / "selection.json")
    if selection["selected_count"] != len(selection["episodes"]):
        raise ValueError("Incomplete selection")
    sessions = defaultdict(list)
    families = {}
    for source in sources["sources"]:
        session = source_names(source["input"], source["task"])[0]
        sessions[session].append(source)
        families[session] = source["family"]
    jobs = [read(p) for p in sorted((root / "jobs").glob("*/result.json"))]
    job_sessions, mapping = {}, {}
    for job in jobs:
        session, task, part = source_names(job["input"], job["task"])
        if session not in sessions:
            raise ValueError(f"Job references an unknown session: {job['id']}")
        job_sessions[job["id"]] = session
        target = root.parent / session
        mapping[str(job_recording(root, job).parent)] = str(target / task / job["id"] / part)
        mapping[str(root / "jobs" / job["id"])] = str(target / "jobs" / job["id"])
    episodes = []
    for ep in selection["episodes"]:
        job_id = ep.get("job_id", Path(ep["candidate_file"]).parent.parent.name)
        session = job_sessions[job_id]
        part = source_names(next(j["input"] for j in jobs if j["id"] == job_id), ep["task"])[2]
        mapping[str(Path(ep["file"]).parent)] = str(root.parent / session / "selected" / ep["task"] / job_id / part)
        episodes.append({**ep, "job_id": job_id, "session": session,
                         "catalog_id": ep.get("catalog_id") or identity(ep["candidate_file"], "data/" + ep["episode"])})
    return {"root": root, "sessions": dict(sessions), "families": families, "sources": sources,
            "selection": selection, "jobs": jobs, "job_sessions": job_sessions,
            "episodes": episodes, "mapping": mapping}


def split_batch(root, state, *, apply=False):
    root, state = Path(root).resolve(), Path(state).resolve()
    marker = root / "migration.json"
    if marker.exists():
        return read(marker)
    plan = plan_split(root)
    report = {"schema_version": 1, "source_batch": str(root), "status": "planned",
              "sessions": {s: {"family": plan["families"][s],
                  "jobs": sum(v == s for v in plan["job_sessions"].values()),
                  "selected": sum(e["session"] == s for e in plan["episodes"])} for s in plan["sessions"]},
              "selected_count": len(plan["episodes"]), "catalog_ids": [e["catalog_id"] for e in plan["episodes"]]}
    if not apply:
        return report
    archive = root.parent / ".legacy" / root.name
    staging = root.parent / (".split_" + root.name)
    if archive.exists() or staging.exists() or any((root.parent / s).exists() for s in plan["sessions"]):
        raise FileExistsError("Migration destination/staging exists; inspect it before retrying")
    status = read(root / "status.json")
    if status.get("active") or status.get("status") != "selected":
        raise ValueError("Only finalized, selected batches can be split")
    with ExitStack() as stack:
        for path in (root / "supervisor.lock", state / "worker.lock"):
            lock = stack.enter_context(path.open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        staging.mkdir()
        published, prepared, backups, archived = [], [], {}, False
        try:
            mapping = plan["mapping"]
            hashes = []
            # Preserve every job payload, including qualification and rejected attempts.
            for old, new in mapping.items():
                src, dst = Path(old), staging / Path(new).relative_to(root.parent)
                if src.parent == root / "jobs":
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("data"), copy_function=copy_artifact)
                elif src.exists():
                    shutil.copytree(src, dst, copy_function=copy_artifact)
                    for artifact in src.rglob("*"):
                        if artifact.is_file():
                            hashes.append({"source": str(artifact), "target": str(Path(new) / artifact.relative_to(src)),
                                           "sha256": sha256_file(artifact)})
            for session, sources in plan["sessions"].items():
                folder = staging / session
                family = plan["families"][session]
                jobs = []
                for original in plan["jobs"]:
                    if plan["job_sessions"][original["id"]] != session:
                        continue
                    job = rewrite(original, mapping)
                    job["output_dir"] = str(Path(mapping[str(job_recording(root, original).parent)]).relative_to(root.parent / session))
                    jobs.append(job)
                    meta = folder / "jobs" / job["id"]
                    save_json(meta / "result.json", job)
                    if (meta / "job.json").exists():
                        spec = rewrite(read(meta / "job.json"), mapping)
                        spec["output_dir"] = job["output_dir"]
                        save_json(meta / "job.json", spec)
                selected = [rewrite(e, mapping) for e in plan["episodes"] if e["session"] == session]
                manifest = {**plan["selection"], "episodes": selected, "selected_count": len(selected),
                            "families": {family: plan["selection"].get("families", {}).get(family, {})},
                            "session": session, "migrated_from": str(archive)}
                review = dict(manifest.get("quality_review", {}))
                if review:
                    review["held_families"] = [f for f in review.get("held_families", []) if f == family]
                    if review.get("report"):
                        evidence = rewrite(read(root / review["report"]), mapping)
                        if evidence.get("visual_evidence"):
                            evidence["visual_evidence"] = str(archive / evidence["visual_evidence"])
                        save_json(folder / review["report"], evidence)
                    manifest["quality_review"] = review
                save_json(folder / "selection.json", manifest)
                save_json(folder / "source_manifest.json", {"sources": sources, "skipped": plan["sources"].get("skipped", [])})
                save_json(folder / "batch_summary.json", jobs)
                save_json(folder / "status.json", {"status": "selected", "active": [], "completed_jobs": len(jobs),
                    "generation_candidates": dict(Counter({family: sum(len(j["accepted"]) for j in jobs if j["stage"] == "generate")}))})
                shutil.copy2(root / "run_config.json", folder / "run_config.json")
                reports = root / "reports" / (family + "_diversity.png")
                if reports.exists():
                    (folder / "reports").mkdir()
                    copy_artifact(reports, folder / "reports" / reports.name)
            for item in hashes:
                target = staging / Path(item["target"]).relative_to(root.parent)
                if sha256_file(target) != item["sha256"]:
                    raise ValueError(f"Payload changed during staging: {target}")
            report["verified_artifacts"] = len(hashes)
            report["payloads"] = hashes
            save_json(staging / "verification.json", report)
            for session in plan["sessions"]:
                target = root.parent / session
                sibling = root.parent / (".split_publish_" + session)
                (staging / session).rename(sibling)
                prepared.append(sibling)
                publish_directory(sibling, target)
                published.append(target)
            # Update path-bearing cache metadata only; encoded videos/images stay byte-identical.
            cache_files = list((state / "videos").glob("*/ready.json")) + list((state / "renders").glob("*/*/manifest.json"))
            for path in cache_files:
                value = read(path)
                changed = rewrite(value, mapping)
                if value != changed:
                    backups[str(path)] = value
                    save_json(path, changed)
            save_json(staging / "cache_backups.json", backups)
            archive.parent.mkdir(exist_ok=True)
            root.rename(archive)
            archived = True
            root.mkdir()
            # Historical reports and embedded HDF5 provenance keep resolving.
            for name in ("jobs", "selected", "preview", "reports", "pilots"):
                if (archive / name).exists():
                    (root / name).symlink_to(archive / name, target_is_directory=True)
            report.update(status="complete", archive=str(archive), cache_metadata_updated=len(backups))
            save_json(root / "migration.json", report)
            (root / "README.md").write_text("This batch was split into original session folders. See migration.json.\n"
                                          "Compatibility links retain historical provenance; the original batch is under .legacy/.\n")
            staging.rename(archive / "migration_records")
            return report
        except BaseException:
            for path, value in backups.items():
                save_json(Path(path), value)
            if archived:
                shutil.rmtree(root)
                archive.rename(root)
            for path in published:
                shutil.rmtree(path)
            for path in prepared:
                if path.exists():
                    shutil.rmtree(path)
            if staging.exists():
                shutil.rmtree(staging)
            raise
