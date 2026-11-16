"""Small file helpers shared by the grasp and scripted-demo scripts (standard library only, so any interpreter can
import them, with or without Isaac Sim)."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess


def file_sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def git_revision(path) -> str:
    """HEAD of the repository at ``path``, with ``+dirty`` when it has uncommitted changes."""
    out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True).stdout
    return out + ("+dirty" if dirty.strip() else "")


def read_json(path: pathlib.Path, default):
    """The file's JSON, or ``default`` if it is missing or unreadable (e.g. being rewritten)."""
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def read_jsonl(path: pathlib.Path, skip_bad: bool = False) -> list[dict]:
    """Every row of a JSONL file ([] if it is missing; blank lines ignored).

    A line that does not parse raises, unless ``skip_bad``. Only read-only viewers should skip: a job that resumes by
    appending must stop on a half-written last row, or its next row would be glued onto it."""
    path = pathlib.Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if not skip_bad:
                raise
    return rows


def parse_targets(text: str) -> dict[str, int]:
    """``"1:5 09:3"`` -> ``{"01": 5, "09": 3}`` (layout: quota pairs)."""
    return {f"{int(k):02d}": int(v) for k, v in (item.split(":") for item in text.split())}
