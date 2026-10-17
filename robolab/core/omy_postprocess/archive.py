"""Immutable one-attempt archives from closed OMY session recordings.

This module never changes the collector, source HDF5, calibration, or physics.
Warnings do not prevent archival, but are retained for training selection.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import tempfile
import uuid

import h5py
import numpy as np

from .io import fsync_directory, publish_directory, sha256_file, sync_file, write_json
from .validation import validate_recording

SCHEMA_VERSION = 1
ATTEMPT_ATTR = "robolab_omy_attempt"
_NAMESPACE = uuid.UUID("1f151d8c-b8e1-49d4-a1a7-0f115fe20a67")


def _read_events(directory, episode_index, *, packaged=False):
    path = directory / "collection_events.jsonl"
    if not path.is_file():
        if not packaged:
            return []
        path = directory / "collection_events.json"
        if not path.is_file():
            return []
        events = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
            raise ValueError("Packaged collection_events.json must be an event list")
        return [event for event in events if event.get("attempt") == episode_index]
    events = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("attempt") == episode_index:
                events.append(event)
    return events


def _start_time(metadata, events):
    for event in events:
        if event.get("event") == "attempt_start" and event.get("wall_time_s") is not None:
            return datetime.fromtimestamp(event["wall_time_s"], timezone.utc), "collection_attempt_start"
    value = metadata.get("recorded_at")
    if not value:
        raise ValueError("No recorded UTC start/session timestamp; cannot fabricate attempt identity")
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Legacy recorded_at must include timezone; local-time identity is ambiguous")
    return stamp.astimezone(timezone.utc), "legacy_session_recorded_at"


def _existing_identity(data):
    raw = data.attrs.get(ATTEMPT_ATTR)
    if raw is None:
        return None
    value = json.loads(raw)
    for key in ("attempt_id", "attempt_uuid", "source", "fingerprint"):
        if key not in value:
            raise ValueError(f"Packaged recording lacks identity field: {key}")
    uuid.UUID(value["attempt_uuid"])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value["attempt_id"]):
        raise ValueError("Unsafe attempt ID in source metadata")
    return value


def verify_archive(directory):
    """Verify all acknowledged artifacts before reusing a completed archive."""
    directory = Path(directory)
    path = directory / "metadata.json"
    if directory.is_symlink() or path.is_symlink():
        raise ValueError(f"Symlink archive destinations/manifests are not trusted: {directory}")
    if not path.is_file():
        raise FileExistsError(f"Existing destination is not a published attempt: {directory}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("publication_status") != "published_raw":
        raise ValueError(f"Attempt is not acknowledged complete: {directory}")
    for name, expected in metadata.get("artifact_sha256", {}).items():
        if Path(name).name != name:
            raise ValueError(f"Unsafe artifact filename: {name}")
        artifact = directory / name
        if artifact.is_symlink() or not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError(f"Archive checksum mismatch: {artifact}")
    required = {"trajectory.hdf5", "env_cfg.json", "calibration.json", "timing.npz",
                "performance.json", "quality.json"}
    if not required.issubset(metadata.get("artifact_sha256", {})):
        raise ValueError("Archive is missing required artifact acknowledgments")
    with h5py.File(directory / "trajectory.hdf5", "r") as file:
        if len(file["data"]) != 1:
            raise ValueError("An attempt archive must contain exactly one episode")
        identity = _existing_identity(file["data"])
        if identity is None or any(metadata.get(key) != value for key, value in identity.items()):
            raise ValueError("Archive HDF5 and sidecar identities disagree")
    return metadata


def package_recording(path, output_root, *, episodes=None, min_free_bytes=64 * 1024 * 1024):
    """Package structurally valid attempts, preserving failed/flagged episodes.

    Outputs are on the explicitly selected filesystem. New partial directories
    survive failure for inspection; final directories never overwrite conflicts.
    Closed sources must keep the same checksum before and after copying.
    """
    path, output_root = Path(path).resolve(), Path(output_root).resolve()
    if min_free_bytes < 0:
        raise ValueError("min_free_bytes must be nonnegative")
    report = validate_recording(path)
    if report.get("errors"):
        raise ValueError(f"Recording validation failed: {report['errors']}")
    source_sha = sha256_file(path)
    if source_sha != report["source"]["sha256"]:
        raise RuntimeError("Source changed after validation")
    wanted = set(episodes) if episodes is not None else None
    selected = [e for e in report["episodes"] if wanted is None or e["name"] in wanted
                or e["group"] in wanted]
    if not selected or (wanted is not None and len(selected) != len(wanted)):
        raise ValueError("No matching episodes or an unknown/duplicate selection")
    invalid = [e["group"] for e in selected if not e["structural_valid"]]
    if invalid:
        raise ValueError(f"Refusing incomplete/malformed episodes: {invalid}; inspect the validation report")
    output_root.mkdir(parents=True, exist_ok=True)
    packaged = []
    for episode in selected:
        directory = _package_episode(path, output_root, report, episode, source_sha, min_free_bytes)
        packaged.append(str(directory))
    if sha256_file(path) != source_sha:
        raise RuntimeError("Source changed while packaging; do not process actively written recordings")
    return {"schema_version": SCHEMA_VERSION, "source": str(path), "source_sha256": source_sha,
            "attempts": packaged}


def _package_episode(path, output_root, report, episode, source_sha, min_free_bytes):
    # Sidecars are part of the input, not merely convenient live configuration.
    # Detect their appearance, disappearance, or mutation during a closed import.
    index = int(episode["name"].removeprefix("demo_"))
    sidecar_names = ("env_cfg.json", "collection_events.jsonl", "collection_events.json", "omy_performance.json",
                     f"attempt_{index:04d}_performance.json", "xr_status.json", "metadata.json", "quality.json",
                     "performance.json", "calibration.json", "timing.npz")
    sidecar_sha = {name: sha256_file(path.parent / name) if (path.parent / name).is_file() else None
                   for name in sidecar_names}

    def check_sidecars():
        for filename, expected in sidecar_sha.items():
            candidate = path.parent / filename
            actual = sha256_file(candidate) if candidate.is_file() else None
            if actual != expected:
                raise RuntimeError(f"Source sidecar changed while packaging: {candidate}")

    with h5py.File(path, "r") as source:
        data = source["data"]
        name = episode["name"]
        group = data[name]
        index = int(name.removeprefix("demo_"))
        identity = _existing_identity(data)
        inherited_metadata = verify_archive(path.parent) if identity is not None else None
        events = _read_events(path.parent, index, packaged=identity is not None)
        omy = json.loads(data.attrs["robolab_omy_teleop"])
        task = omy.get("task")
        if not isinstance(task, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", task):
            raise ValueError("Task name is missing or unsafe for attempt identity")
        if identity is not None:
            if len(data) != 1 or identity["fingerprint"] != episode["fingerprint"]:
                raise ValueError("Packaged source identity does not match its episode content")
            attempt_id, attempt_uuid = identity["attempt_id"], identity["attempt_uuid"]
            original = identity["source"]
            stamp_source = identity.get("start_time_source", "inherited_attempt_identity")
            start_utc = identity.get("start_utc")
        else:
            stamp, stamp_source = _start_time({"recorded_at": data.attrs.get("recorded_at")}, events)
            attempt_id = f"{task}_{stamp.strftime('%Y%m%dT%H%M%S%fZ')}_a{index + 1:04d}"
            original = {"path": str(path), "sha256": source_sha, "group": f"data/{name}"}
            attempt_uuid = str(uuid.uuid5(_NAMESPACE, f"{source_sha}:data/{name}"))
            start_utc = stamp.isoformat(timespec="microseconds")
        destination = output_root / attempt_id
        if destination.exists() or destination.is_symlink():
            existing = verify_archive(destination)
            if existing["attempt_uuid"] != attempt_uuid or existing["fingerprint"] != episode["fingerprint"]:
                raise FileExistsError(f"Conflicting attempt destination: {destination}")
            if sidecar_sha["env_cfg.json"] != existing["artifact_sha256"]["env_cfg.json"]:
                raise FileExistsError(f"Source configuration conflicts with archived attempt: {destination}")
            if identity is None:
                old_sidecars = existing.get("source_sidecar_sha256", {})
                if any(old_sidecars.get(filename) != expected for filename, expected in sidecar_sha.items()):
                    raise FileExistsError(f"Source sidecars conflict with archived attempt: {destination}")
            check_sidecars()
            return destination
        # Conservative input-size reserve, not just a count of queued episodes.
        if shutil.disk_usage(output_root).free < min_free_bytes + path.stat().st_size:
            raise OSError("Insufficient output space; refusing to begin an attempt archive")
        temporary = Path(tempfile.mkdtemp(prefix=f".partial-{attempt_uuid}-", dir=output_root))
        # Preserve the raw success attribute unchanged, but distinguish an
        # interrupted/manual-reset attempt using the actual recorded end event.
        # Bind that classification to HDF5 identity and repeat it in quality and
        # manifest, so reimports cannot lose or silently contradict the reason.
        end_events = [e for e in events if e.get("event") == "attempt_end"]
        quality = dict(episode)
        quality["recorded_episode_outcome"] = episode["outcome"]
        quality["recorded_episode_outcome_evidence"] = episode["outcome_evidence"]
        end_event = end_events[-1] if end_events else None
        if inherited_metadata is not None:
            quality["outcome"] = inherited_metadata["outcome"]
            quality["outcome_evidence"] = inherited_metadata["outcome_evidence"]
            quality["training_eligible"] = bool(episode["training_eligible"]
                                                 and inherited_metadata.get("training_eligible", False))
            end_event = inherited_metadata.get("end_event", end_event)
        elif end_event and isinstance(end_event.get("result"), str):
            quality["outcome"] = "interrupted"
            quality["outcome_evidence"] = {"source": "collection_attempt_end", "event": end_event}
            quality["training_eligible"] = False
        else:
            quality["outcome_evidence"] = {"source": "recorded_episode_attributes", **episode["outcome_evidence"]}
        identity = {"schema_version": SCHEMA_VERSION, "attempt_id": attempt_id,
                    "attempt_uuid": attempt_uuid, "source": original,
                    "fingerprint": episode["fingerprint"], "start_utc": start_utc,
                    "start_time_source": stamp_source, "outcome": quality["outcome"],
                    "outcome_evidence": quality["outcome_evidence"], "end_event": end_event}
        try:
            with h5py.File(temporary / "trajectory.hdf5", "w") as dest:
                for key, value in source.attrs.items():
                    dest.attrs[key] = value
                target = dest.create_group("data")
                for key, value in data.attrs.items():
                    target.attrs[key] = value
                source.copy(group, target, name=name)
                target.attrs["total"] = episode["sample_count"]
                target.attrs[ATTEMPT_ATTR] = json.dumps(identity, sort_keys=True)
            config_path = path.parent / "env_cfg.json"
            if not config_path.is_file():
                raise ValueError("env_cfg.json is required for faithful offline rendering")
            shutil.copyfile(config_path, temporary / "env_cfg.json")
            calibration = omy.get("calibration", {})
            if not isinstance(calibration.get("contents"), dict):
                raise ValueError("Recording lacks its embedded calibration snapshot")
            write_json(temporary / "calibration.json", calibration["contents"])
            timing = {key: value[()] for key, value in group["timing"].items()
                      if isinstance(value, h5py.Dataset)}
            np.savez(temporary / "timing.npz", **timing)
            if inherited_metadata is not None:
                # Already-published evidence keeps its true scope and original
                # source path; it is not a new session measurement.
                shutil.copyfile(path.parent / "performance.json", temporary / "performance.json")
            else:
                performance_path = path.parent / f"attempt_{index:04d}_performance.json"
                performance_scope = "attempt"
                if not performance_path.is_file():
                    performance_path = path.parent / "omy_performance.json"
                    performance_scope = "session"
                performance = json.loads(performance_path.read_text()) if performance_path.is_file() else None
                write_json(temporary / "performance.json", {"scope": performance_scope if performance is not None else "missing",
                           "source": str(performance_path) if performance is not None else None, "measurements": performance})
            if events:
                write_json(temporary / "collection_events.json", events)
            if sidecar_sha["xr_status.json"] is not None:
                shutil.copyfile(path.parent / "xr_status.json", temporary / "xr_status.json")
            write_json(temporary / "quality.json", quality)
            artifacts = {p.name: sha256_file(p) for p in temporary.iterdir() if p.is_file()}
            metadata = {**identity, "publication_status": "published_raw", "task": task,
                        "session_id": original.get("sha256"), "attempt_number": index + 1,
                        "sample_count": episode["sample_count"],
                        "start_time_precision": "exact_event" if stamp_source == "collection_attempt_start"
                                                else "legacy_session_fallback",
                        "control_hz": omy.get("control_hz"), "timing": episode.get("timing"),
                        "training_eligible": quality.get("training_eligible", False),
                        "runtime_versions": {key: str(data.attrs[key]) for key in
                                             ("isaaclab_version", "isaacsim_version") if key in data.attrs},
                        "source_sidecar_sha256": sidecar_sha,
                        "artifact_sha256": artifacts}
            write_json(temporary / "metadata.json", metadata)
            # Check before publication, not merely at the end of a multi-attempt job.
            if sha256_file(path) != source_sha:
                raise RuntimeError("Source changed while packaging")
            check_sidecars()
            for artifact in temporary.iterdir():
                if artifact.is_file():
                    sync_file(artifact)
            fsync_directory(temporary)
            verify_archive(temporary)
            try:
                publish_directory(temporary, destination)
            except FileExistsError:
                # A cooperating concurrent importer may have published the same
                # immutable source. Keep this partial as diagnostics either way.
                existing = verify_archive(destination)
                if existing["attempt_uuid"] != attempt_uuid or existing["fingerprint"] != episode["fingerprint"]:
                    raise
                if existing["artifact_sha256"]["env_cfg.json"] != artifacts["env_cfg.json"]:
                    raise FileExistsError(f"Source configuration conflicts with concurrent archived attempt: {destination}")
                if inherited_metadata is None:
                    old_sidecars = existing.get("source_sidecar_sha256", {})
                    if any(old_sidecars.get(filename) != expected for filename, expected in sidecar_sha.items()):
                        raise FileExistsError(f"Source sidecars conflict with concurrent archived attempt: {destination}")
            return destination
        except BaseException as error:
            if temporary.exists() and not (temporary / "failure.json").exists():
                write_json(temporary / "failure.json", {"type": type(error).__name__, "message": str(error),
                                                      "source": str(path), "attempt_uuid": attempt_uuid})
            raise
