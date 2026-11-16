"""Scripted episodes as MimicGen sources: the ``"scripted"`` source kind of ``robolab.core.omy_teleop.mimic_sources``.

A scripted family's ``pps.hdf5`` holds, per successful episode, the PPS rows and the ``pre_action_state`` group the
collector records (``writer.STATE_GROUP``): the robot's full joint vector and the object and bin root poses before
each action, in Isaac Lab's recorder layout. Together they give the generator everything a teleoperation recording
gives it. The PPS ``actions`` are the joint targets the arm was sent, with the gripper in the wire convention
(0 open, 1 closed); the generator's convention is +1 open, -1 closed.

Episodes collected before the state group existed cannot be read; collect them again.
"""

from __future__ import annotations

import h5py
import numpy as np

from robolab.core.omy_teleop.mimic_sources import SourceEpisode, grasp_index
from robolab.core.scripted_demos.writer import STATE_GROUP


def generator_actions(wire_actions: np.ndarray) -> np.ndarray:
    """PPS wire actions (7 joint targets, gripper 0 open / 1 closed) in the generator's encoding (+1 open, -1 closed)."""
    actions = np.array(wire_actions, dtype=np.float32, copy=True)
    actions[:, 7] = 1.0 - 2.0 * actions[:, 7]
    return actions


def read_episode(path, episode, object_name, container_name) -> SourceEpisode:
    with h5py.File(path, "r") as stream:
        group = stream["data"][episode]
        if STATE_GROUP not in group:
            raise ValueError(f"{path}:{episode} has no {STATE_GROUP} group: it was collected before the scripted "
                             f"collector recorded the scene state, so it cannot be a MimicGen source; collect it again")
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
    if not len(joints) == len(wire) == len(poses[object_name]) == len(poses[container_name]):
        raise ValueError(f"{path}:{episode}: actions and pre-action state are not aligned")
    actions = generator_actions(wire)
    return SourceEpisode(actions=actions, joint_position=joints, object_pose=poses,
                         grasp=grasp_index(actions[:, -1], poses[object_name][:, 2]), success=success,
                         joint_names=joint_names)
