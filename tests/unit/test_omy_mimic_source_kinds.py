"""MimicGen source episodes are read through one reader per source kind (mimic_sources.read_source)."""

import h5py
import numpy as np
import pytest

from robolab.core.omy_teleop import mimic_sources
from robolab.core.omy_teleop.mimic_sources import SourceEpisode, read_source, source_arrays

OBJECT, CONTAINER = "banana", "bowl"


def _write_teleop(path, steps=6, success=True):
    """An Isaac Lab recording: post-step ``states``, one ``initial_state`` row, and data the reader must skip."""
    rng = np.random.default_rng(0)
    actions = rng.normal(size=(steps, 8)).astype(np.float32)
    actions[:, 7] = (1, 1, -1, -1, -1, 1)
    joints = rng.normal(size=(steps, 13)).astype(np.float32)
    with h5py.File(path, "w") as stream:
        ep = stream.create_group("data/demo_0")
        ep.attrs["success"] = success
        ep["actions"] = actions
        ep["obs/table_cam"] = np.zeros((steps, 4, 4, 3), dtype=np.uint8)
        ep["initial_state/articulation/robot/joint_position"] = joints[:1] - 1
        ep["states/articulation/robot/joint_position"] = joints
        for name, z0 in ((OBJECT, 0.0), (CONTAINER, 1.0)):
            poses = np.zeros((steps, 7), dtype=np.float32)
            poses[:, 3] = 1
            poses[:, 2] = z0 + 0.03 * np.arange(1, steps + 1)
            ep[f"initial_state/rigid_object/{name}/root_pose"] = np.r_[poses[:1] * 0 + [[0, 0, z0, 1, 0, 0, 0]]]
            ep[f"states/rigid_object/{name}/root_pose"] = poses
    return path


def _tree(group):
    return {name: _tree(item) if isinstance(item, h5py.Group) else item[()] for name, item in group.items()}


def test_teleop_reader_matches_the_recorded_tree(tmp_path):
    path = _write_teleop(tmp_path / "omy_demos.hdf5")
    source = read_source({"file": str(path), "episode": "demo_0"}, OBJECT, CONTAINER)  # no kind: teleop
    with h5py.File(path) as stream:
        expected = source_arrays(_tree(stream["data/demo_0"]), OBJECT, CONTAINER)
    assert isinstance(source, SourceEpisode) and source.success and source.joint_names is None
    assert np.array_equal(source.actions, expected["actions"])
    assert np.array_equal(source.joint_position, expected["joint_position"])
    assert source.object_pose.keys() == expected["object_pose"].keys()
    for name in (OBJECT, CONTAINER):
        assert np.array_equal(source.object_pose[name], expected["object_pose"][name])
    assert source.grasp == expected["grasp"] == 2  # first closed row with the object 2 cm up


def test_unsuccessful_source_is_reported_not_hidden(tmp_path):
    path = _write_teleop(tmp_path / "omy_demos.hdf5", success=False)
    assert not read_source({"file": str(path), "episode": "demo_0", "kind": "teleop"}, OBJECT, CONTAINER).success


def test_unknown_kind_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Unknown source kind"):
        read_source({"file": "x.hdf5", "episode": "demo_0", "kind": "puppet"}, OBJECT, CONTAINER)


def test_readers_can_be_registered_by_import_path(tmp_path, monkeypatch):
    path = _write_teleop(tmp_path / "omy_demos.hdf5")
    monkeypatch.setitem(mimic_sources.SOURCE_READERS, "alias", "robolab.core.omy_teleop.mimic_sources:read_teleop_episode")
    source = read_source({"file": str(path), "episode": "demo_0", "kind": "alias"}, OBJECT, CONTAINER)
    assert source.grasp == 2


@pytest.mark.parametrize("attrs, seven", [
    ({}, False),  # L100 teleoperation
    ({"robolab_omy_teleop": '{"task": "T"}'}, False),
    ({"robolab_omy_teleop": '{"operator": {"kind": "scripted", "locked_joints": []}}'}, True),
    ({"robolab_omy_teleop": '{"operator": {"kind": "scripted", "locked_joints": ["panda_joint3"]}}'}, False),
    ({"robolab_omy_mimicgen": '{"seven_axis_ik": true}'}, True),
    ({"robolab_omy_mimicgen": '{"seven_axis_ik": false}'}, False),
])
def test_teleop_reader_takes_seven_axis_from_the_recording(tmp_path, attrs, seven):
    """Scripted recordings in the OMY env (omy_collect) and --seven-axis-ik output move panda_joint3 by design."""
    path = _write_teleop(tmp_path / "omy_demos.hdf5")
    with h5py.File(path, "a") as stream:
        stream["data"].attrs.update(attrs)
    source = read_source({"file": str(path), "episode": "demo_0"}, OBJECT, CONTAINER)
    assert source.seven_axis is seven
