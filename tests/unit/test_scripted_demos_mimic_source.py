"""Scripted episodes as MimicGen sources: the same motion recorded as a teleoperation recording and as a scripted
episode reads back as the same SourceEpisode, for the generator and for the path screen."""

import math

import h5py
import numpy as np
import pytest

from robolab.core.omy_teleop.mimic_sources import read_source
from robolab.core.omy_teleop.path_screen import Arm, load_source
from robolab.core.scripted_demos.mimic_source import generator_actions
from robolab.core.scripted_demos.writer import STATE_GROUP, EpisodeBuffer, Writer

OBJECT, CONTAINER = "pitcher", "grey_bin"
HOME = np.array([0.0, -0.6, 0.0, -2.5, 0.0, 1.9, 0.0])
JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8)) + tuple(f"finger_{i}" for i in range(6))
STEPS, CLOSE = 30, 12


def _motion():
    """Pre-action rows of one pick: arm joints (13 incl. fingers), wire actions, object and bin root poses."""
    s = np.linspace(0, 1, STEPS)[:, None]
    arm = HOME + s * np.array([0.5, 0.3, 0.0, 0.4, 0.3, -0.3, 0.4])
    joints = np.c_[arm, np.zeros((STEPS, 6))].astype(np.float32)
    wire = np.c_[arm, (np.arange(STEPS) >= CLOSE).astype(float)].astype(np.float32)
    obj = np.tile([0.5, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0], (STEPS, 1)).astype(np.float32)
    obj[CLOSE + 3:, 2] = 0.01 * np.arange(1, STEPS - CLOSE - 2)  # lifted once the gripper has closed
    bin_ = np.tile([0.4, -0.3, 0.0, 1.0, 0.0, 0.0, 0.0], (STEPS, 1)).astype(np.float32)
    return joints, wire, {OBJECT: obj, CONTAINER: bin_}


def _scripted(path, with_state=True):
    joints, wire, poses = _motion()
    buf = EpisodeBuffer(source_sim_origin_s=0.0)
    for t in range(STEPS):
        request = {"observation/joint_position": joints[t, :7], "observation/gripper_position": wire[t, 7:],
                   "observation/exterior_image_1_left": np.zeros((8, 8, 3), np.uint8),
                   "observation/wrist_image_left": np.zeros((8, 8, 3), np.uint8)}
        state = {"articulation/robot/joint_position": joints[t],
                 **{f"rigid_object/{name}/root_pose": rows[t] for name, rows in poses.items()}}
        buf.add(request, wire[t], raw_finger=0.0, sim_time=t / 15, state=state if with_state else None,
                joint_names=JOINT_NAMES)
    return Writer(path, task="SimplePnp303Pitcher", prompt="p").append(buf.arrays(), {"layout_task": "x"})


def _teleop(path):
    """The same motion as Isaac Lab records it: one initial_state row, then post-step states."""
    joints, wire, poses = _motion()
    with h5py.File(path, "w") as stream:
        ep = stream.create_group("data/demo_0")
        ep.attrs["success"] = True
        ep["actions"] = generator_actions(wire)
        ep["initial_state/articulation/robot/joint_position"] = joints[:1]
        ep["states/articulation/robot/joint_position"] = np.r_[joints[1:], joints[-1:]]
        for name, rows in poses.items():
            ep[f"initial_state/rigid_object/{name}/root_pose"] = rows[:1]
            ep[f"states/rigid_object/{name}/root_pose"] = np.r_[rows[1:], rows[-1:]]


def _arm():
    def frame(xyz, rpy=(0, 0, 0)):
        out = np.eye(4)
        cr, sr, cp, sp, cy, sy = (f(a) for a in rpy for f in (math.cos, math.sin))
        out[:3, :3] = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                       @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
        out[:3, 3] = xyz
        return out

    half = math.pi / 2
    frames = [frame([0, 0, 0.33], (0, -half, 0)), frame([0, 0, 0], (0, 0, half)), frame([0.32, 0, 0], (0, 0, -half)),
              frame([0, 0.08, 0], (0, 0, half)), frame([0.38, -0.08, 0], (0, 0, -half)), frame([0, 0, 0], (0, 0, half)),
              frame([0, 0.09, 0], (0, 0, half))]
    return Arm([(f, np.eye(4), True) for f in frames] + [(frame([0.2, 0, 0]), np.eye(4), False)], limits=[(-2.9, 2.9)] * 7)


@pytest.fixture()
def files(tmp_path):
    scripted, teleop = tmp_path / "SimplePnp303Pitcher" / "pps.hdf5", tmp_path / "run" / "omy_demos.hdf5"
    teleop.parent.mkdir()
    name = _scripted(scripted)
    _teleop(teleop)
    return {"file": str(scripted), "episode": name, "kind": "scripted"}, {"file": str(teleop), "episode": "demo_0"}


def test_scripted_and_teleop_recordings_of_one_motion_read_alike(files):
    scripted, teleop = (read_source(ref, OBJECT, CONTAINER) for ref in files)
    assert scripted.success and teleop.success
    assert np.array_equal(scripted.actions, teleop.actions)
    assert set(np.unique(scripted.actions[:, 7])) == {-1.0, 1.0}  # +1 open, -1 closed
    assert np.array_equal(scripted.joint_position, teleop.joint_position)
    for name in (OBJECT, CONTAINER):
        assert np.array_equal(scripted.object_pose[name], teleop.object_pose[name])
    assert scripted.grasp == teleop.grasp == CLOSE + 5  # closed, and 2 cm up (> 0.02 m)
    assert scripted.joint_names == JOINT_NAMES and teleop.joint_names is None


def test_path_screen_loads_both_kinds_alike(files):
    arm = _arm()
    scripted = load_source(files[0]["file"], files[0]["episode"], OBJECT, CONTAINER, arm, kind="scripted")
    teleop = load_source(files[1]["file"], files[1]["episode"], OBJECT, CONTAINER, arm)
    assert scripted.keys() == teleop.keys()
    for key in scripted:
        a, b = scripted[key], teleop[key]
        assert all(np.array_equal(x, y) for x, y in zip(a, b)) if isinstance(a, tuple) else np.array_equal(a, b), key


def test_episode_without_state_is_refused(tmp_path):
    path = tmp_path / "pps.hdf5"
    name = _scripted(path, with_state=False)
    with h5py.File(path) as stream:
        assert STATE_GROUP not in stream["data"][name]
    with pytest.raises(ValueError, match="collect it again"):
        read_source({"file": str(path), "episode": name, "kind": "scripted"}, OBJECT, CONTAINER)


def test_state_is_all_or_nothing_per_episode():
    joints, wire, poses = _motion()
    buf = EpisodeBuffer()
    request = {"observation/joint_position": joints[0, :7], "observation/gripper_position": wire[0, 7:],
               "observation/exterior_image_1_left": np.zeros((8, 8, 3), np.uint8),
               "observation/wrist_image_left": np.zeros((8, 8, 3), np.uint8)}
    buf.add(request, wire[0], 0.0, 0.0, state={"articulation/robot/joint_position": joints[0]})
    with pytest.raises(ValueError):
        buf.add(request, wire[1], 0.0, 0.0)
