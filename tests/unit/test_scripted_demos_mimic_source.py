"""Scripted episodes as MimicGen sources: the same motion recorded as a teleoperation recording and as a scripted
episode reads back as the same SourceEpisode, for the generator and for the path screen."""

import json
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


def _motion(joint3=0.0):
    """Pre-action rows of one pick: arm joints (13 incl. fingers), wire actions, object and bin root poses."""
    s = np.linspace(0, 1, STEPS)[:, None]
    arm = HOME + s * np.array([0.5, 0.3, joint3, 0.4, 0.3, -0.3, 0.4])
    joints = np.c_[arm, np.zeros((STEPS, 6))].astype(np.float32)
    wire = np.c_[arm, (np.arange(STEPS) >= CLOSE).astype(float)].astype(np.float32)
    obj = np.tile([0.5, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0], (STEPS, 1)).astype(np.float32)
    obj[CLOSE + 3:, 2] = 0.01 * np.arange(1, STEPS - CLOSE - 2)  # lifted once the gripper has closed
    bin_ = np.tile([0.4, -0.3, 0.0, 1.0, 0.0, 0.0, 0.0], (STEPS, 1)).astype(np.float32)
    return joints, wire, {OBJECT: obj, CONTAINER: bin_}


def _settle(joints, wire, poses, drop):
    """The collector's settle rows before the motion: arm held at its first command while the objects fall by
    ``drop`` metres and come to rest (``collect.Session.settle``)."""
    n = len(drop)
    poses = {name: np.r_[np.repeat(rows[:1], n, 0), rows] for name, rows in poses.items()}
    poses[OBJECT][:n, 2] += np.asarray(drop, dtype=np.float32)
    return np.r_[np.repeat(joints[:1], n, 0), joints], np.r_[np.repeat(wire[:1], n, 0), wire], poses


def _scripted(path, with_state=True, joint3=0.0, locked=(), drop=()):
    joints, wire, poses = _motion(joint3)
    if len(drop):
        joints, wire, poses = _settle(joints, wire, poses, drop)
    buf = EpisodeBuffer(source_sim_origin_s=0.0)
    for t in range(len(wire)):
        request = {"observation/joint_position": joints[t, :7], "observation/gripper_position": wire[t, 7:],
                   "observation/exterior_image_1_left": np.zeros((8, 8, 3), np.uint8),
                   "observation/wrist_image_left": np.zeros((8, 8, 3), np.uint8)}
        state = {"articulation/robot/joint_position": joints[t],
                 **{f"rigid_object/{name}/root_pose": rows[t] for name, rows in poses.items()}}
        buf.add(request, wire[t], raw_finger=0.0, sim_time=t / 15, state=state if with_state else None,
                joint_names=JOINT_NAMES)
    return Writer(path, task="SimplePnp303Pitcher", prompt="p").append(
        buf.arrays(), {"layout_task": "x", **({"locked_joints": json.dumps(list(locked))} if locked else {})})


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
    assert scripted.seven_axis and not teleop.seven_axis  # only teleop sources must hold panda_joint3


def test_path_screen_loads_both_kinds_alike(files):
    arm = _arm()
    scripted = load_source(files[0]["file"], files[0]["episode"], OBJECT, CONTAINER, arm, kind="scripted")
    teleop = load_source(files[1]["file"], files[1]["episode"], OBJECT, CONTAINER, arm)
    assert scripted.keys() == teleop.keys()
    for key in scripted:
        a, b = scripted[key], teleop[key]
        assert all(np.array_equal(x, y) for x, y in zip(a, b)) if isinstance(a, tuple) else np.array_equal(a, b), key


def test_path_screen_holds_joint3_at_home_for_seven_axis_sources(tmp_path):
    """A scripted source that moves panda_joint3 is screened as the generator's six-joint arm follows it: IK seeds and
    the start keep joint 3 at the episode's first (home) value, while achieved hand poses use the joints it had."""
    arm, path = _arm(), tmp_path / "pps.hdf5"
    name = _scripted(path, joint3=0.2)
    source = read_source({"file": str(path), "episode": name, "kind": "scripted"}, OBJECT, CONTAINER)
    assert np.ptp(source.actions[:, 2]) > 0.19 and np.ptp(source.joint_position[:, 2]) > 0.19
    screened = load_source(path, name, OBJECT, CONTAINER, arm, kind="scripted")
    assert np.all(screened["joints"][:, 2] == HOME[2])
    assert np.array_equal(np.delete(screened["joints"], 2, 1), np.delete(source.joint_position[:, :7], 2, 1))
    moved = np.array([arm.fk(row) for row in source.joint_position[:, :7]])
    assert np.array_equal(screened["achieved"], moved)
    assert np.array_equal(screened["hand"], np.array([arm.fk(row) for row in source.actions[:, :7]]))


def test_episode_collected_with_joint3_held_is_a_six_axis_source(tmp_path):
    """--hold-joint3 episodes record locked_joints: the generator then checks joint 3, as for teleoperation."""
    held, free = tmp_path / "held" / "pps.hdf5", tmp_path / "free" / "pps.hdf5"
    held.parent.mkdir(), free.parent.mkdir()
    name = _scripted(held, locked=("panda_joint3",))
    _scripted(free)
    assert not read_source({"file": str(held), "episode": name, "kind": "scripted"}, OBJECT, CONTAINER).seven_axis
    assert read_source({"file": str(free), "episode": name, "kind": "scripted"}, OBJECT, CONTAINER).seven_axis


def test_settle_rows_are_not_part_of_the_source(tmp_path, files):
    """Objects spawn slightly airborne and settle while the arm is held: the source starts at rest, as a teleoperation
    recording does, so the motion is relative to the resting scene and the 2 cm lift is measured from it."""
    path = tmp_path / "settled" / "pps.hdf5"
    path.parent.mkdir()
    name = _scripted(path, drop=[0.085, 0.04, 0.001, 0.0])
    settled = read_source({"file": str(path), "episode": name, "kind": "scripted"}, OBJECT, CONTAINER)
    teleop = read_source(files[1], OBJECT, CONTAINER)
    assert np.array_equal(settled.actions, teleop.actions) and settled.grasp == teleop.grasp
    for key in (OBJECT, CONTAINER):
        assert np.array_equal(settled.object_pose[key], teleop.object_pose[key])


def test_source_whose_objects_still_move_when_the_arm_starts_is_refused(tmp_path):
    path = tmp_path / "pps.hdf5"
    name = _scripted(path, drop=[0.085, 0.04])
    with pytest.raises(ValueError, match="still moving"):
        read_source({"file": str(path), "episode": name, "kind": "scripted"}, OBJECT, CONTAINER)


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
