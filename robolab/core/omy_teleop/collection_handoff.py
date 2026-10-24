"""Simulator-free handoff between collection and its offline supervisor."""

import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


def replace_report(path, value):
    """Atomically update an owned progress report, including on a retry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def new_run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:12]


class CollectionHandoff:
    def __init__(self, path, run_id=None):
        self.path = Path(path).resolve()
        self.run_id = run_id or new_run_id()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.run_id):
            raise ValueError("Invalid collection invocation ID")
        self.session = None
        self.recordings = []

    def set_session(self, root):
        self.session = Path(root).resolve()
        self.publish()

    def add_recording(self, directory):
        path = str(Path(directory).resolve() / "omy_demos.hdf5")
        if path not in self.recordings:
            self.recordings.append(path)
        self.publish()

    def publish(self, *, status=None, cleanup_complete=False, app_shutdown_pending=False):
        if self.session is None:
            return
        saved = self.session / "collection_runs" / f"{self.run_id}.json"
        payload = {
            "schema_version": 1, "run_id": self.run_id,
            "session_dir": str(self.session), "recordings": self.recordings,
            "collection_status": status, "cleanup_complete": cleanup_complete,
            "app_shutdown_pending": app_shutdown_pending,
            "output_dir": str(self.session / "postprocess" / self.run_id),
            "handoff": str(saved),
        }
        replace_report(saved, payload)
        replace_report(self.path, payload)


def load_handoff(path, *, allow_pending=False):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported collection handoff")
    run_id = payload.get("run_id", "")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid collection invocation ID")
    closed = payload.get("cleanup_complete") is True
    pending = allow_pending and payload.get("app_shutdown_pending") is True
    if not (closed or pending) or payload.get("collection_status") not in (0, 130):
        raise ValueError("Collection did not finish cleanup successfully")
    session = Path(payload["session_dir"])
    if not session.is_absolute() or not session.is_dir():
        raise ValueError("Collection session must be an existing absolute directory")
    expected = session / "postprocess" / run_id
    if Path(payload["output_dir"]) != expected:
        raise ValueError("Postprocessing destination does not match this invocation")
    if Path(payload["handoff"]) != session / "collection_runs" / f"{run_id}.json":
        raise ValueError("Saved handoff does not match this invocation")
    recordings = payload.get("recordings")
    if not isinstance(recordings, list) or not all(isinstance(item, str) for item in recordings):
        raise ValueError("Collection handoff must explicitly list recording paths")
    if len(recordings) != len(set(recordings)):
        raise ValueError("Duplicate collection recording")
    for item in recordings:
        source = Path(item)
        if not source.is_absolute() or source.name != "omy_demos.hdf5":
            raise ValueError("Unexpected collection recording path")
        source.resolve().relative_to(session.resolve())
        if "postprocess" in source.relative_to(session).parts:
            raise ValueError("Processed artifacts cannot be collection inputs")
    return payload


def confirm_collector_exit(path, *, run_id, status):
    """Called by the supervisor only after it has reaped the collector process.

    Isaac's fast shutdown may exit natively inside app.close(). A pending app
    shutdown is accepted only when recording cleanup already returned and the
    actual process status matches the status requested before app shutdown.
    """
    payload = load_handoff(path, allow_pending=True)
    if payload["run_id"] != run_id or payload["collection_status"] != status:
        raise ValueError("Collector process status or invocation does not match its handoff")
    payload.update(cleanup_complete=True, app_shutdown_pending=False, app_exit_confirmed=True)
    replace_report(payload["handoff"], payload)
    replace_report(path, payload)
    return payload
