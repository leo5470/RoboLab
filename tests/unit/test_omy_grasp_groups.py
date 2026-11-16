"""Grasp groups of qualified sources, from synthetic reproductions."""

import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from robolab.core.omy_teleop import grasp_groups, pipeline
from robolab.core.omy_teleop.grasp_groups import group_sources, in_hand_position, placement_parents, session_groups

OBJECT = "pail"


def _quat_z(angle):
    return [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]


def _write(path, offset, hand_yaw=0.3, steps=12, grasp=6):
    """A reproduction whose hand holds the object at ``offset`` (hand frame) from step ``grasp`` on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    c, s = math.cos(hand_yaw), math.sin(hand_yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    hand = np.array([[0.4, 0.1, 0.3 + 0.02 * max(0, t - grasp + 2)] for t in range(steps)])
    obj = np.array([hand[t] + rotation @ np.asarray(offset) if t >= grasp else [0.5, 0.2, 0.02] for t in range(steps)])
    # Lift the held object above its start so the grasp rule fires at ``grasp``.
    obj[grasp:, 2] = np.maximum(obj[grasp:, 2], 0.05)
    hand[grasp:] = obj[grasp:] - np.array([rotation @ np.asarray(offset) for _ in range(steps - grasp)])
    actions = np.zeros((steps, 8))
    actions[:, -1] = [1.0 if t < grasp - 1 else -1.0 for t in range(steps)]
    with h5py.File(path, "w") as stream:
        ep = stream.create_group("data/demo_0")
        ep["actions"] = actions
        ep["mimic/eef_pose"] = np.c_[hand, np.tile(_quat_z(hand_yaw), (steps, 1))]
        poses = np.c_[obj, np.tile([1, 0, 0, 0], (steps, 1))]
        ep[f"initial_state/rigid_object/{OBJECT}/root_pose"] = poses[:1]
        ep[f"states/rigid_object/{OBJECT}/root_pose"] = np.r_[poses[1:], poses[-1:]]
    return path


def test_in_hand_position_is_in_the_hand_frame(tmp_path):
    offset = [0.25, 0.05, -0.01]
    for yaw in (0.0, 0.3, 2.0):
        path = _write(tmp_path / f"{yaw}.hdf5", offset, hand_yaw=yaw)
        with h5py.File(path) as stream:
            assert np.allclose(in_hand_position(stream["data/demo_0"], OBJECT), offset, atol=1e-6)


def test_clear_sides_group_and_close_sides_are_refused():
    sides = {"a": [0.25, 0.05, 0], "b": [0.24, 0.04, -0.04], "c": [0.25, -0.06, 0], "d": [0.25, -0.055, 0.01]}
    assert group_sources(sides) == [["a", "b"], ["c", "d"]]
    assert group_sources({"a": [0, 0, 0]}) == [["a"]]
    with pytest.raises(ValueError, match="not clearly separated"):
        group_sources({"a": [0, 0, 0], "b": [0.07, 0, 0]})  # neither linked nor clearly apart


def test_session_record_and_placement_parents(tmp_path):
    recordings = {name: (_write(tmp_path / name / "omy_demos.hdf5", offset), "demo_0")
                  for name, offset in (("02", [0.25, 0.05, 0]), ("04", [0.25, 0.045, -0.01]), ("01", [0.25, -0.06, 0]))}
    record = session_groups(recordings, OBJECT)
    assert record["status"] == "grouped" and record["groups"] == [["01"], ["02", "04"]]
    assert placement_parents(record, "02") == ["02", "04"] and placement_parents(record, "01") == ["01"]
    recordings["05"] = (_write(tmp_path / "05" / "omy_demos.hdf5", [0.25, -0.0, 0]), "demo_0")
    ambiguous = session_groups(recordings, OBJECT)
    assert ambiguous["status"] == "ambiguous" and ambiguous["groups"] is None
    with pytest.raises(ValueError, match="Mixed parents refused"):
        placement_parents(ambiguous, "02")


def test_pipeline_writes_the_record_after_qualification(tmp_path):
    p = pipeline.Pipeline({"state": tmp_path / "state", "summary": tmp_path / "s.json", "pools": tmp_path / "pools"},
                          [0], log=lambda *a, **k: None)
    work = pipeline.SessionWork("session")
    work.family, work.folder = OBJECT, tmp_path / "generated"
    for name, offset in (("A", [0.25, 0.05, 0]), ("B", [0.25, -0.06, 0])):
        key = (str(tmp_path / f"collected/{name}/part_000"), "demo_0")
        result = {"id": f"q_{name}", "stage": "validate", "accepted": ["demo_0"], "output_dir": f"{name}/q/part_000"}
        _write(pipeline.job_recording(work.folder, result), offset)
        work.qualified.add(key)
        work.validations[key] = result
    p.record_grasp_groups(work)
    record = json.loads((tmp_path / "state/session/grasp_groups.json").read_text())
    assert record["status"] == "grouped" and len(record["groups"]) == 2
    assert record["implementation"] == pipeline.implementation()
    # A reproduction without achieved hand poses leaves the record unavailable, not the session failed.
    with h5py.File(pipeline.job_recording(work.folder, work.validations[sorted(work.qualified)[0]]), "r+") as stream:
        del stream["data/demo_0/mimic"]
    p.record_grasp_groups(work)
    assert json.loads((tmp_path / "state/session/grasp_groups.json").read_text())["status"] == "unavailable"
