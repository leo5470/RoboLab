"""Scripted episodes as MimicGen sources: the ``"scripted"`` source kind of ``robolab.core.omy_teleop.mimic_sources``.

A scripted family's ``pps.hdf5`` holds, per successful episode, the PPS rows and the ``pre_action_state`` group the
collector records (``writer.STATE_GROUP``): the robot's full joint vector and the object and bin root poses before
each action, in Isaac Lab's recorder layout. Together they give the generator everything a teleoperation recording
gives it. The PPS ``actions`` are the joint targets the arm was sent, with the gripper in the wire convention
(0 open, 1 closed); the generator's convention is +1 open, -1 closed. The source starts after the collector's settle
phase (``settled_start``), with the objects at rest, as a teleoperation recording does.

By default the scripted arm uses all seven joints, so its episodes are seven-axis sources (``SourceEpisode.seven_axis``):
the generator accepts their ``panda_joint3`` motion and follows their end-effector poses with that joint held at home.
Episodes collected with ``--hold-joint3`` (episode attribute ``locked_joints``) held it at home themselves; they are
read as six-axis sources, so the generator checks that they did, as it does for teleoperation recordings.

Episodes collected before the state group existed cannot be read; collect them again.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from robolab.core.omy_teleop.mimic_sources import SourceEpisode, env_contract, grasp_index
from robolab.core.omy_teleop.options import ARM_JOINTS
from robolab.core.scripted_demos.writer import STATE_GROUP


_NO_STATE = (f"{{path}}:{{episode}} has no {STATE_GROUP} group: it was collected before the scripted collector "
             f"recorded the scene state, so it cannot be a MimicGen source; collect it again")


def generator_actions(wire_actions: np.ndarray) -> np.ndarray:
    """PPS wire actions (7 joint targets, gripper 0 open / 1 closed) in the generator's encoding (+1 open, -1 closed)."""
    actions = np.array(wire_actions, dtype=np.float32, copy=True)
    actions[:, 7] = 1.0 - 2.0 * actions[:, 7]
    return actions


def settled_start(wire_actions: np.ndarray) -> int:
    """First row a MimicGen source starts from: the last row of the collector's settle phase.

    An episode begins with ``collect.Session.settle``: the arm held at its reset pose while the objects drop from
    their authored, slightly airborne poses and come to rest. A teleoperation recording starts from objects already at
    rest, and the generator takes the first row as the scene the motion is relative to (the reference for the grasp
    segment and the 2 cm lift of ``grasp_index``). So the source starts at the last held row: arm still at home,
    objects resting. Rows before it are the same arm command and carry no motion.
    """
    arm = np.asarray(wire_actions)[:, :7]
    moving = np.flatnonzero(np.any(arm != arm[0], axis=1))
    return max(int(moving[0]) - 1, 0) if len(moving) else 0


def source_contract(path, episode) -> tuple[dict, str]:
    """``(contract, task)`` of one scripted episode, for ``mimic_sources.source_references``.

    The contract is read from the ``env_cfg.json`` RoboLab's recorder wrote for the episode's layout task when it was
    collected (``<family>/robolab_output/<layout_task>/env_cfg.json``), in the fields of ``mimic_sources.contract``.
    The joint order is the one recorded with the scene state, and the action description is that of the file's
    ``actions`` (the reader converts the gripper encoding).
    """
    with h5py.File(path, "r") as stream:
        group = stream["data"][episode]
        if STATE_GROUP not in group:
            raise ValueError(_NO_STATE.format(path=path, episode=episode))
        task = group.attrs["layout_task"]
        task = task.decode() if isinstance(task, bytes) else str(task)
        names = group.get(f"{STATE_GROUP}/articulation/robot/joint_names")
        joint_order = [n.decode() for n in names[()]] if names is not None else None
    cfg_path = Path(path).parent / "robolab_output" / task / "env_cfg.json"
    if not cfg_path.is_file():
        raise ValueError(f"{path}:{episode}: no recorded environment configuration at {cfg_path}")
    cfg = json.loads(cfg_path.read_text())
    actions = {"action_type": "absolute_joint_position", "action_names": [*ARM_JOINTS, "gripper"], "joint_units": "rad",
               "gripper_encoding": "0=open,1=closed", "control_hz": 1.0 / (cfg["sim"]["dt"] * cfg["decimation"])}
    return env_contract(cfg, joint_order, actions), task


def read_episode(path, episode, object_name, container_name) -> SourceEpisode:
    with h5py.File(path, "r") as stream:
        group = stream["data"][episode]
        if STATE_GROUP not in group:
            raise ValueError(_NO_STATE.format(path=path, episode=episode))
        state = group[STATE_GROUP]
        missing = [name for name in (object_name, container_name) if f"rigid_object/{name}/root_pose" not in state]
        if missing:
            raise ValueError(f"{path}:{episode} records no pose for {missing}; it has {sorted(state['rigid_object'])}")
        wire = group["actions"][()]
        joints = state["articulation/robot/joint_position"][()]
        poses = {name: state[f"rigid_object/{name}/root_pose"][()] for name in (object_name, container_name)}
        names = state.get("articulation/robot/joint_names")
        joint_names = tuple(n.decode() for n in names[()]) if names is not None else None
        success = bool(group.attrs.get("success", False))
        locked = json.loads(group.attrs.get("locked_joints", "[]"))
    if not len(joints) == len(wire) == len(poses[object_name]) == len(poses[container_name]):
        raise ValueError(f"{path}:{episode}: actions and pre-action state are not aligned")
    start = settled_start(wire)
    z = poses[object_name][:, 2]
    if start and abs(z[start] - z[start - 1]) > 1e-3:
        raise ValueError(f"{path}:{episode}: the object is still moving when the arm starts (row {start})")
    wire, joints = wire[start:], joints[start:]
    poses = {name: rows[start:] for name, rows in poses.items()}
    actions = generator_actions(wire)
    return SourceEpisode(actions=actions, joint_position=joints, object_pose=poses,
                         grasp=grasp_index(actions[:, -1], poses[object_name][:, 2]), success=success,
                         joint_names=joint_names, seven_axis="panda_joint3" not in locked)
