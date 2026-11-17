"""Scripted OMY sessions for the demo viewer (``scripts/scripted_demos/demo_viewer.py``); h5py and numpy only.

A session is a folder written by ``collect_omy.py``: ``<layout task>/part_NNN/omy_demos.hdf5`` (OMY recordings, state
only), ``attempts.jsonl`` (one row per attempt, with the plan and executor report), and, once
``render_omy_review.py`` has run, ``review/<layout task>/part_NNN/images.hdf5`` with the policy cameras rendered
from the recorded states. The viewer lists a session as a family with id ``omy/<path below the sessions root>``;
its episodes, every recorded attempt, are named ``episode_NNNNNN`` in part and recording order.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import subprocess
import tempfile

import h5py
import numpy as np

from robolab.core.scripted_demos.records import read_jsonl

PREFIX = "omy/"
FPS = 15
LAYOUT_RE = re.compile(r"(\d+)Task$")


def _open(path: pathlib.Path) -> h5py.File:
    return h5py.File(path, "r", locking=False)


def sessions(root: pathlib.Path) -> dict[str, pathlib.Path]:
    """``{family id: session folder}`` for every folder below ``root`` holding OMY recordings."""
    found = {}
    if root.is_dir():
        for recording in sorted(root.glob("**/part_*/omy_demos.hdf5")):
            session = recording.parents[2]
            if "review" in recording.relative_to(session).parts:
                continue
            found[PREFIX + session.relative_to(root).as_posix()] = session
    return found


@dataclasses.dataclass(frozen=True)
class Entry:
    name: str  # episode_NNNNNN
    recording: pathlib.Path
    group: str  # demo_N
    part: str  # <layout task>/part_NNN, relative to the session

    @property
    def task(self) -> str:
        return self.part.split("/")[0]


def entries(session: pathlib.Path) -> list[Entry]:
    out = []
    for recording in sorted(session.glob("*/part_*/omy_demos.hdf5")):
        with _open(recording) as f:
            groups = sorted(f["data"], key=lambda n: int(n.rsplit("_", 1)[1]))
        part = recording.parent.relative_to(session).as_posix()
        out += [Entry(f"episode_{len(out) + i:06d}", recording, g, part) for i, g in enumerate(groups)]
    return out


def attempts(session: pathlib.Path) -> dict[tuple[str, str], dict]:
    """``attempts.jsonl`` rows by (part, demo group)."""
    rows = {}
    for row in read_jsonl(session / "attempts.jsonl", skip_bad=True):
        omy = (row.get("extra") or {}).get("omy") or {}
        if omy.get("episode"):
            rows[(row.get("part"), omy["episode"])] = row
    return rows


def _arrays(f: h5py.File, group: str):
    g = f["data"][group]
    q = g["obs/proprio_obs/arm_joint_pos"][:]
    gripper = g["obs/proprio_obs/gripper_pos"][:, 0]
    actions = g["actions"][:].copy()
    actions[:, 7] = (1.0 - actions[:, 7]) / 2.0  # OMY +1 open / -1 closed -> 0 open / 1 closed, as PPS rows
    preparation = json.loads(g.attrs.get("omy_preparation", "{}") or "{}")
    return g, q, gripper, actions, preparation


def row(entry: Entry, attempt: dict | None) -> dict:
    """The viewer's episode row (``demo_viewer.episode_row`` fields) for one OMY episode."""
    with _open(entry.recording) as f:
        g, q, _, actions, preparation = _arrays(f, entry.group)
        success = str(g.attrs.get("success", "False")) in ("True", "true", "1")
        seed = int(g.attrs.get("seed", -1))
    attempt = attempt or {}
    plan, executor = attempt.get("plan") or {}, attempt.get("executor") or {}
    planned, taken = executor.get("segment_steps_planned") or {}, executor.get("segment_steps_taken") or {}
    target = preparation.get("target") or {}
    match = LAYOUT_RE.search(entry.task)
    n = len(actions)
    return {
        "scene": None, "mimic": None, "name": entry.name, "task": entry.task,
        # The page's layout column: the explicit target, else the v1 layout; failed attempts (OMY keeps them) marked.
        "layout": ("" if success else "\u2717 ") + (target.get("id") or (match.group(1) if match else entry.task))
                  + ("" if success else f" ({attempt.get('failure') or 'failed'})"),
        "seed": seed, "steps": n, "success": success,
        "failure": None if success else (attempt.get("failure") or "not successful"),
        "grasp_id": plan.get("grasp_id"), "cost": (plan.get("costs") or {}).get("total"),
        "closedness": executor.get("closedness_after_close"), "contact": executor.get("held_finger_contact_fraction"),
        "extra_steps": sum(int(v) - int(planned.get(k, v)) for k, v in taken.items()),
        "max_speed": float(np.abs(np.diff(q, axis=0)).max() * FPS) if n > 1 else 0.0,
        "max_lead": float(np.abs(actions[:, :7] - q).max()) if n else 0.0,
        "omy": {"part": entry.part, "group": entry.group, "target": target.get("id"),
                "target_error": target.get("achieved")},
    }


def detail(entry: Entry, attempt: dict | None) -> dict:
    """Per-row traces of one OMY episode: measured joints and gripper closure, and the command (gripper 0/1)."""
    with _open(entry.recording) as f:
        _, q, gripper, actions, preparation = _arrays(f, entry.group)
    attempt = attempt or {}
    plan = attempt.get("plan") or {}
    return {
        "plan": {k: plan.get(k) for k in ("grasp_id", "backoff", "lift_extra", "preplace_yaw", "costs",
                                          "held_object_bottom_offset", "release_drop") if k in plan},
        "executor": attempt.get("executor") or {},
        "state": np.round(np.c_[q, gripper], 4).T.tolist(),
        "command": np.round(actions, 4).T.tolist(),
        "preparation": {k: preparation.get(k) for k in ("status", "elapsed_s", "placement", "target")},
    }


def review_images(session: pathlib.Path, entry: Entry) -> pathlib.Path:
    return session / "review" / entry.part / "images.hdf5"


def review_video(session: pathlib.Path, entry: Entry) -> pathlib.Path:
    return session / "review" / entry.part / f"{entry.group}.mp4"


def encode(frames: np.ndarray, out: pathlib.Path, fps: int = FPS) -> None:
    """H.264 MP4 of RGB frames (N, H, W, 3), one frame per recorded row, written atomically."""
    n, h, w, _ = frames.shape
    frames = np.pad(frames, ((0, 0), (0, h % 2), (0, w % 2), (0, 0)))  # yuv420p needs even sides
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".mp4", dir=out.parent)
    os.close(fd)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{frames.shape[2]}x{frames.shape[1]}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", str(fps), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-threads", "2", tmp]
    try:
        proc = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[-500:]}")
        os.chmod(tmp, 0o644)  # mkstemp creates 0600
        os.replace(tmp, out)
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)


def frames(session: pathlib.Path, entry: Entry) -> np.ndarray:
    """The rendered policy cameras of one episode, side by side (N, H, W_total, 3); KeyError until rendered."""
    path = review_images(session, entry)
    if not path.is_file():
        raise KeyError(f"not rendered yet: scripts/scripted_demos/render_omy_review.py {session}")
    with _open(path) as f:
        images = f["data"][entry.group]["obs/image_obs"]
        cameras = [name for name in ("over_shoulder_left_camera", "wrist_cam") if name in images] or sorted(images)
        views = [images[name][:] for name in cameras]
    height = min(v.shape[1] for v in views)
    return np.concatenate([v[:, :height] for v in views], axis=2)
