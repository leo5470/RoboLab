"""Process the closed recordings from exactly one collection invocation."""

import errno
from contextlib import contextmanager
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from robolab.core.omy_teleop.collection_handoff import load_handoff, new_run_id, replace_report

from .archive import package_recording, verify_archive
from .render import render_recording
from .validation import validate_recording


def _relative(path, root):
    return os.path.relpath(path, root)


@contextmanager
def _invocation_lock(session, run_id):
    # Collection resume never writes to an earlier part. This lock additionally
    # prevents two offline retries from updating the same invocation's reports.
    lock_path = session / "collection_runs" / f"{run_id}.postprocess.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Postprocessing is already running for this invocation") from error
        yield


def process_session(handoff_path, *, device=None, skip_failed=None, preview_videos=None, batch=True, run_id=None):
    handoff = load_handoff(handoff_path)
    session = Path(handoff["session_dir"])
    retry_args = None
    if run_id is not None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("Invalid postprocessing run ID")
        if run_id != handoff["run_id"] and (session / "collection_runs" / f"{run_id}.json").exists():
            raise ValueError("Postprocessing run ID belongs to another collection invocation")
        handoff = {**handoff, "run_id": run_id, "output_dir": str(session / "postprocess" / run_id)}
        retry_args = ["--handoff", handoff["handoff"], "--run-id", run_id]
    output = Path(handoff["output_dir"])
    if (output / "session_input.json").exists():
        raise ValueError("This run snapshots a closed session; retry it with --session")
    saved = handoff.get("postprocessing", {})
    options_path = output / "processing_options.json"
    existing = json.loads(options_path.read_text()) if options_path.exists() else {}
    device = device or existing.get("device") or saved.get("device", "cuda:1")
    if not options_path.exists():
        skip_failed = saved.get("skip_failed") if skip_failed is None else skip_failed
        preview_videos = saved.get("preview_videos") if preview_videos is None else preview_videos
    with _invocation_lock(session, handoff["run_id"]):
        return _process(handoff, output, device=device, skip_failed=skip_failed, preview_videos=preview_videos,
                        batch=batch, retry_args=retry_args)


def _closed_session_input(session_dir, *, run_id=None, device="cuda:1"):
    """Snapshot an older layout session while excluding collection/resume writers.

    A persistent input snapshot makes retries use the same recording parts even
    if the collection is resumed later. No collector exit status is invented.
    """
    session = Path(session_dir).resolve(strict=True)
    run_id = run_id or new_run_id()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid postprocessing run ID")
    if not re.fullmatch(r"cuda:[0-9]+", device):
        raise ValueError("Postprocessing device must be cuda:N")
    output = session / "postprocess" / run_id
    if output.resolve() != output:
        raise ValueError("Postprocessing destination cannot use symbolic links")
    if (session / "collection_runs" / f"{run_id}.json").exists():
        raise ValueError("This run has a collection handoff; use --handoff to retry it")
    lock_path = session / ".session.lock"
    if lock_path.is_symlink():
        raise ValueError("Session lock cannot be a symbolic link")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Collection or postprocessing is already using this session") from error
        snapshot_path = output / "session_input.json"
        if snapshot_path.exists():
            snapshot = json.loads(snapshot_path.read_text())
            if (snapshot.get("schema_version") != 1 or snapshot.get("session_dir") != str(session)
                    or snapshot.get("run_id") != run_id
                    or snapshot.get("input_kind") != "closed_layout_session"):
                raise ValueError("Postprocessing input snapshot does not match this session/run")
        else:
            if (output / "dataset.json").exists() or (output / "summary.json").exists():
                raise ValueError("Existing processing reports have no input snapshot; choose another run ID")
            state = json.loads((session / "session.json").read_text())
            if state.get("schema_version") != 1 or not isinstance(state.get("layouts"), dict):
                raise ValueError("Expected an object-layout collection session.json")
            recordings = []
            for task, layout in state["layouts"].items():
                parts = layout.get("parts") if isinstance(layout, dict) else None
                if not isinstance(parts, list):
                    raise ValueError(f"Invalid recording parts for {task}")
                for relative in parts:
                    if not isinstance(relative, str):
                        raise ValueError(f"Invalid recording part: {relative}")
                    part = Path(relative)
                    if (part.is_absolute() or len(part.parts) != 2 or part.parts[0] != task
                            or not re.fullmatch(r"part_[0-9]+", part.name)):
                        raise ValueError(f"Invalid recording part: {relative}")
                    recordings.append(str((session / part / "omy_demos.hdf5").resolve(strict=True)))
            snapshot = {"schema_version": 1, "input_kind": "closed_layout_session",
                        "session_dir": str(session), "run_id": run_id,
                        "collection_status": None, "recordings": recordings}
        sources = snapshot.get("recordings")
        if (not isinstance(sources, list) or not sources
                or not all(isinstance(source, str) for source in sources)
                or len(set(sources)) != len(sources)):
            raise ValueError("Expected a nonempty, unique list of closed recording parts")
        for source in sources:
            path = Path(source).resolve(strict=True)
            relative = path.relative_to(session)
            if (str(path) != source or len(relative.parts) != 3 or path.name != "omy_demos.hdf5"
                    or relative.parts[0] == "postprocess"
                    or not re.fullmatch(r"part_[0-9]+", relative.parts[1])):
                raise ValueError(f"Invalid closed-session source: {source}")
        output.mkdir(parents=True, exist_ok=True)
        if not snapshot_path.exists():
            replace_report(snapshot_path, snapshot)
        return snapshot, output


def process_closed_session(session_dir, *, run_id=None, device="cuda:1", skip_failed=None, preview_videos=None, batch=True):
    snapshot, output = _closed_session_input(session_dir, run_id=run_id, device=device)
    session = Path(snapshot["session_dir"])
    # Resume always creates new parts, so the frozen old parts can be processed
    # after releasing the collection lock. Only this output invocation is locked.
    with _invocation_lock(session, snapshot["run_id"]):
        return _process(snapshot, output, device=device, skip_failed=skip_failed, preview_videos=preview_videos, batch=batch,
                        retry_args=["--session", str(session), "--run-id", snapshot["run_id"]])


def _process(handoff, output, *, device="cuda:1", retry_args=None, skip_failed=None, preview_videos=None, batch=True):
    processing_started = time.perf_counter()
    if not re.fullmatch(r"cuda:[0-9]+", device):
        raise ValueError("Postprocessing device must be cuda:N")
    session = Path(handoff["session_dir"])
    if output.resolve() != output:
        raise ValueError("Postprocessing destination cannot use symbolic links")
    output.mkdir(parents=True, exist_ok=True)
    options_path = output / "processing_options.json"
    options = {"skip_failed": bool(skip_failed), "preview_videos": bool(preview_videos), "device": device}
    if options_path.exists():
        saved = json.loads(options_path.read_text())
        for name, explicit in (("skip_failed", skip_failed), ("preview_videos", preview_videos)):
            if explicit is not None and bool(explicit) != saved.get(name):
                raise ValueError("Processing selection changed; use a new output run")
        options = saved
    else:
        replace_report(options_path, options)
    skip_failed, preview_videos = options["skip_failed"], options["preview_videos"]
    profile = {"device": device, "renderer": "realtime", "camera_scale": 1.0,
               "renders_per_frame": 2}
    summary = {"schema_version": 1, "run_id": handoff["run_id"],
               "input_kind": handoff.get("input_kind", "collection_handoff"),
               "collection_status": handoff["collection_status"], "status": "processing",
               "render_profile": profile, "recordings": [], "errors": []}
    summary["selection"] = options
    summary["renderer_mode"] = "batch" if batch else "per-episode"
    summary["timings"] = {name: 0.0 for name in ("validation_s", "packaging_s", "render_and_verify_s", "video_s")}
    summary["recordings"] = [{"source": _relative(source, output), "status": "pending", "attempts": []}
                             for source in handoff["recordings"]]
    dataset = {"schema_version": 1, "run_id": handoff["run_id"],
               "status": "processing", "training_ready": False, "episodes": []}
    current = None

    @contextmanager
    def stage(name):
        started = time.perf_counter()
        try:
            yield
        finally:
            summary["timings"][name] += time.perf_counter() - started

    def save():
        dataset["status"] = summary["status"]
        replace_report(output / "dataset.json", dataset)
        replace_report(output / "summary.json", summary)

    def is_shared_failure(error):
        return (isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EDQUOT)
                or shutil.disk_usage(output).free < 64 * 1024**2)

    def record_error(target, error):
        target.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(f"Postprocessing failed: {target['error']}", flush=True)

    print(f"Postprocessing destination: {output}", flush=True)
    print(f"Postprocessing device: {device}", flush=True)
    script = Path(__file__).resolve().parents[3] / "scripts/postprocess_omy_session.py"
    if retry_args is None:
        retry_args = ["--handoff", handoff["handoff"]]
    retry = [sys.executable, str(script), *retry_args, "--device", device]
    if skip_failed:
        retry.append("--skip-failed")
    if preview_videos:
        retry.append("--preview-videos")
    if not batch:
        retry.extend(["--renderer-mode", "per-episode"])
    print(f"Retry postprocessing: {shlex.join(retry)}", flush=True)
    save()
    stop = False
    from .batch import BatchRenderer

    worker = BatchRenderer(output, device) if batch else None
    try:
        for source_name, recording in zip(handoff["recordings"], summary["recordings"]):
            source = Path(source_name)
            key = source.parent.relative_to(session)
            if key == Path("."):
                key = Path("recording")
            recording["status"] = "validating"
            current = recording
            save()
            try:
                with stage("validation_s"):
                    report = validate_recording(source)
                report_path = output / "reports" / key / "index.json"
                replace_report(report_path, report)
                recording["report"] = _relative(report_path, output)
                if not report["episodes"] and all(e["code"] == "no_episodes" for e in report["errors"]):
                    recording["status"] = "empty"
                    save()
                    continue
                if report["errors"]:
                    raise ValueError(f"Recording validation failed: {report['errors']}")
                recording["status"] = "processing"
                for episode in report["episodes"]:
                    attempt = {"group": episode["group"], "status": "pending"}
                    recording["attempts"].append(attempt)
                save()
                for episode, attempt in zip(report["episodes"], recording["attempts"]):
                    current = attempt
                    attempt["outcome"] = episode["outcome"]
                    if skip_failed and episode["outcome"] != "success":
                        attempt.update(status="skipped", reason="not_successful")
                        save()
                        continue
                    try:
                        if not episode["structural_valid"] or not episode["native_15_hz"]:
                            raise ValueError(f"Attempt is not complete native-15: {episode['errors']}")
                        attempt["status"] = "packaging"
                        save()
                        with stage("packaging_s"):
                            packaged = package_recording(source, output / "raw" / key, episodes=[episode["group"]])
                            archive = Path(packaged["attempts"][0])
                            metadata = verify_archive(archive)
                        if metadata["source"]["sha256"] != report["source"]["sha256"]:
                            raise ValueError("Source recording changed after session validation")
                        quality = json.loads((archive / "quality.json").read_text())
                        attempt.update(status="rendering", archive=_relative(archive, output))
                        save()
                        print(f"Rendering {key}/{archive.name}", flush=True)
                        with stage("render_and_verify_s"):
                            rendered = render_recording(archive / "trajectory.hdf5",
                                                        output / "rendered" / key / archive.name,
                                                        group=episode["group"], **profile,
                                                        **({"runner": worker.run} if worker is not None else {}))
                        images = Path(rendered) / "images.hdf5"
                        dataset["episodes"].append({
                            "attempt_id": metadata["attempt_id"], "attempt_uuid": metadata["attempt_uuid"],
                            "source": _relative(source, output), "source_group": episode["group"],
                            "source_sha256": report["source"]["sha256"],
                            "trajectory": _relative(archive / "trajectory.hdf5", output),
                            "images": _relative(images, output),
                            "render_manifest": _relative(Path(rendered) / "manifest.json", output),
                            "quality": _relative(archive / "quality.json", output),
                            "outcome": metadata["outcome"], "sample_count": episode["sample_count"],
                            "warnings": quality["warnings"], "training_ready": False,
                        })
                        attempt.update(status="complete", images=_relative(images, output))
                    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
                        record_error(attempt, error)
                        if is_shared_failure(error):
                            stop = True
                    save()
                    if stop:
                        break
                recording["status"] = ("failed" if any(a["status"] not in ("complete", "skipped") for a in recording["attempts"])
                                       else "complete")
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
                record_error(recording, error)
                stop = is_shared_failure(error)
            save()
            if stop:
                summary["errors"].append("Stopped after a shared storage failure; remaining recordings were not processed")
                break
        summary["status"] = "incomplete" if summary["errors"] or any(
            r["status"] == "failed" for r in summary["recordings"]) else "complete"
    except KeyboardInterrupt:
        if current is not None and current["status"] not in ("complete", "empty", "failed"):
            current["status"] = "interrupted"
        summary["status"] = "interrupted"
    finally:
        if worker is not None:
            try:
                worker.close()
            except KeyboardInterrupt:
                summary["status"] = "interrupted"
            except (OSError, RuntimeError) as error:
                summary["errors"].append(f"Worker shutdown failed: {error}")
                summary["status"] = "incomplete"
            summary["worker"] = worker.metrics
        summary["completed_episodes"] = len(dataset["episodes"])
        summary["skipped_episodes"] = sum(a["status"] == "skipped" for r in summary["recordings"] for a in r["attempts"])
        summary["skipped_by_outcome"] = dict(Counter(a["outcome"] for r in summary["recordings"] for a in r["attempts"]
                                                     if a["status"] == "skipped"))
        summary["collected_episodes"] = sum(len(r["attempts"]) for r in summary["recordings"])
        summary["failed_episodes"] = sum(a["status"] == "failed" for r in summary["recordings"]
                                          for a in r["attempts"])
        save()
        if preview_videos and dataset["episodes"] and summary["status"] != "interrupted":
            try:
                from .videos import export_success_videos

                with stage("video_s"):
                    previews = export_success_videos(output / "dataset.json")
                summary["verification_videos"] = str(previews) if previews else None
            except KeyboardInterrupt:
                summary["status"] = "interrupted"
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                summary["errors"].append(f"Verification video export failed: {error}")
                summary["status"] = "incomplete"
            save()
        summary["elapsed_s"] = time.perf_counter() - processing_started
        save()
        count = len(dataset["episodes"])
        if summary["status"] == "complete" and not count:
            print("No eligible successful episodes; rendering skipped." if summary["skipped_episodes"]
                  else "No recorded episodes; postprocessing skipped.", flush=True)
            print(f"Raw session: {session}", flush=True)
            print(f"Postprocessing output directory: {output}", flush=True)
        else:
            print(f"Postprocessing {summary['status']}: {count} episodes rendered; "
                  f"{summary['skipped_episodes']} skipped; {summary['failed_episodes']} episodes failed.", flush=True)
            print(f"Postprocessed dataset: {output}", flush=True)
        print(f"Dataset manifest: {output / 'dataset.json'}", flush=True)
        print(f"Processing report: {output / 'summary.json'}", flush=True)
    return {"complete": 0, "incomplete": 2, "interrupted": 130}[summary["status"]]
