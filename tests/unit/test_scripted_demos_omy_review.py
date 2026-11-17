"""Scripted OMY sessions in the demo viewer (omy_review): discovery, episode naming across parts, the attempt log
join, explicit targets and failed attempts in the layout column, traces, and review frames."""

import json

import h5py
import numpy as np
import pytest

from robolab.core.scripted_demos import omy_review


def _recording(path, episodes):
    """An OMY recording (state only); ``episodes``: list of (success, steps, target id or None)."""
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as f:
        for i, (success, steps, target) in enumerate(episodes):
            g = f.create_group(f"data/demo_{i}")
            g.attrs.update(success=success, seed=i, num_samples=steps)
            prep = {"status": "ready", "elapsed_s": 0.5}
            if target:
                prep["target"] = {"id": target, "poses": {}, "achieved": {"milkjug_a01": {"xy_error_m": 0.001}}}
            g.attrs["omy_preparation"] = json.dumps(prep)
            q = np.linspace(0, 1, steps)[:, None] * np.ones((1, 7), np.float32)
            g["obs/proprio_obs/arm_joint_pos"] = q.astype(np.float32)
            g["obs/proprio_obs/gripper_pos"] = np.zeros((steps, 1), np.float32)
            actions = np.c_[q, np.where(np.arange(steps) < steps // 2, 1.0, -1.0)].astype(np.float32)
            g["actions"] = actions


@pytest.fixture()
def session(tmp_path):
    root = tmp_path / "sessions"
    s = root / "milkjug_a01_v2" / "20261117_scripted"
    _recording(s / "SimplePnp230MilkjugA0101Task/part_000/omy_demos.hdf5", [(True, 10, "t0_0000"), (False, 4, "t0_0001")])
    _recording(s / "SimplePnp230MilkjugA0103Task/part_000/omy_demos.hdf5", [(True, 8, None)])
    rows = [{"part": "SimplePnp230MilkjugA0101Task/part_000", "success": True, "failure": None,
             "plan": {"grasp_id": 7}, "executor": {"total_steps": 9}, "extra": {"omy": {"episode": "demo_0"}}},
            {"part": "SimplePnp230MilkjugA0101Task/part_000", "success": False, "failure": "tracking_error",
             "extra": {"omy": {"episode": "demo_1"}}}]
    (s / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    review = s / "review/SimplePnp230MilkjugA0101Task/part_000"
    review.mkdir(parents=True)
    with h5py.File(review / "images.hdf5", "w") as f:
        for i, n in enumerate((10, 4)):
            f[f"data/demo_{i}/obs/image_obs/over_shoulder_left_camera"] = np.full((n, 6, 8, 3), 1, np.uint8)
            f[f"data/demo_{i}/obs/image_obs/wrist_cam"] = np.full((n, 6, 8, 3), 2, np.uint8)
    return root, s


def test_sessions_and_episode_names(session):
    root, s = session
    assert omy_review.sessions(root) == {"omy/milkjug_a01_v2/20261117_scripted": s}
    listed = omy_review.entries(s)
    assert [(e.name, e.part.split("/")[0][-8:-4], e.group) for e in listed] == [
        ("episode_000000", "0101", "demo_0"), ("episode_000001", "0101", "demo_1"), ("episode_000002", "0103", "demo_0")]


def test_rows_show_targets_failures_and_the_plan(session):
    _, s = session
    listed, rows = omy_review.entries(s), omy_review.attempts(s)
    first, failed, plain = (omy_review.row(e, rows.get((e.part, e.group))) for e in listed)
    assert first["layout"] == "t0_0000" and first["grasp_id"] == 7 and first["success"]
    assert failed["layout"].startswith("✗ t0_0001") and "tracking_error" in failed["layout"]
    assert plain["layout"] == "0103" and plain["grasp_id"] is None
    detail = omy_review.detail(listed[0], rows[(listed[0].part, "demo_0")])
    assert len(detail["state"]) == 8 and len(detail["state"][0]) == 10
    assert detail["command"][7][0] == 0.0 and detail["command"][7][-1] == 1.0  # +1 open / -1 closed -> 0 / 1


def test_frames_side_by_side_and_unrendered_parts(session):
    _, s = session
    listed = omy_review.entries(s)
    frames = omy_review.frames(s, listed[0])
    assert frames.shape == (10, 6, 16, 3) and frames[0, 0, 0, 0] == 1 and frames[0, 0, -1, 0] == 2
    with pytest.raises(KeyError, match="render_omy_review"):
        omy_review.frames(s, listed[2])


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="needs ffmpeg")
def test_review_videos_are_encoded_beside_the_part(session):
    _, s = session
    entry = omy_review.entries(s)[0]
    out = omy_review.review_video(s, entry)
    omy_review.encode(omy_review.frames(s, entry), out)
    assert out.name == "demo_0.mp4" and out.parent == s / "review" / entry.part
    assert out.stat().st_size > 0 and (out.stat().st_mode & 0o777) == 0o644
