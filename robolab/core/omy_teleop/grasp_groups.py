"""Group qualified human sources by how they hold the object, without Isaac Sim.

A placement segment replays its parent's hand motion and assumes that
parent's in-hand object pose. Mixing a grasp from one human with a placement
from another is only sensible when both hold the object alike, so mixed-parent
generation restricts placement parents to the grasp parent's group.

The in-hand position comes from a qualification reproduction, which records
the achieved hand pose (``mimic/eef_pose``) next to the object pose.
"""

import json
from pathlib import Path

import h5py
import numpy as np

LINK_M = 0.06        # sources closer than this, in the hand frame, are linked into one group
SEPARATION_M = 0.08  # every pair of groups must be at least this far apart, or grouping is ambiguous


def _matrix(quat):
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def in_hand_position(episode, object_name, lift=0.02):
    """Object root position in the hand frame at the first closed-gripper step that has lifted it."""
    actions = np.asarray(episode["actions"])
    initial = np.asarray(episode[f"initial_state/rigid_object/{object_name}/root_pose"])
    states = np.asarray(episode[f"states/rigid_object/{object_name}/root_pose"])
    obj = np.concatenate((initial[:1], states[:-1]))  # pre-action rows, aligned with the hand poses
    eef = np.asarray(episode["mimic/eef_pose"])
    if eef.shape != (len(actions), 7) or len(obj) != len(actions):
        raise ValueError("Episode lacks aligned achieved hand poses")
    held = np.flatnonzero((actions[:, -1] < 0) & (obj[:, 2] - obj[0, 2] > lift))
    if not len(held):
        raise ValueError("The object is never lifted with the gripper closed")
    t = int(held[0])
    return _matrix(eef[t, 3:]).T @ (obj[t, :3] - eef[t, :3])


def group_sources(positions, link_m=LINK_M, separation_m=SEPARATION_M):
    """Single-linkage groups of in-hand positions ``{key: xyz}``; raises if two groups are not clearly apart."""
    keys = sorted(positions)
    points = np.array([positions[k] for k in keys], dtype=float).reshape(len(keys), 3)
    distance = np.linalg.norm(points[:, None] - points[None], axis=-1)
    label = list(range(len(keys)))
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if distance[i, j] < link_m and label[i] != label[j]:
                old, new = label[j], label[i]
                label = [new if value == old else value for value in label]
    groups = [[keys[i] for i in range(len(keys)) if label[i] == value] for value in sorted(set(label))]
    index = {key: n for n, group in enumerate(groups) for key in group}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if index[keys[i]] != index[keys[j]] and distance[i, j] < separation_m:
                raise ValueError(f"Grasp groups are not clearly separated: {keys[i]} and {keys[j]} are "
                                 f"{distance[i, j] * 1000:.0f} mm apart (link {link_m * 1000:.0f} mm, "
                                 f"separation {separation_m * 1000:.0f} mm)")
    return groups


def session_groups(recordings, object_name, link_m=LINK_M, separation_m=SEPARATION_M):
    """Grasp groups from one reproduction per source: ``{key: (omy_demos.hdf5, episode)}``.

    Returns a JSON-ready record. Its ``status`` is ``"ambiguous"`` (with the
    reason) when groups are not clearly separated; mixed-parent generation must
    then refuse rather than guess.
    """
    positions = {}
    for key, (path, episode) in recordings.items():
        with h5py.File(path, "r") as stream:
            positions[key] = in_hand_position(stream["data"][episode], object_name)
    record = {"schema_version": 1, "object": object_name, "link_m": link_m, "separation_m": separation_m,
              "sources": [{"key": list(key) if isinstance(key, tuple) else key, "recording": str(Path(path).resolve()),
                           "episode": episode, "in_hand_position_m": positions[key].round(6).tolist()}
                          for key, (path, episode) in sorted(recordings.items())]}
    try:
        groups = group_sources(positions, link_m, separation_m)
    except ValueError as error:
        return {**record, "status": "ambiguous", "reason": str(error), "groups": None}
    return {**record, "status": "grouped",
            "groups": [[list(k) if isinstance(k, tuple) else k for k in group] for group in groups]}


def placement_parents(record, grasp_key):
    """Sources a placement may come from for this grasp parent (the parent's own group)."""
    if record["status"] != "grouped":
        raise ValueError(f"Mixed parents refused: {record.get('reason', record['status'])}")
    key = list(grasp_key) if isinstance(grasp_key, tuple) else grasp_key
    for group in record["groups"]:
        if key in group:
            return [tuple(k) if isinstance(k, list) else k for k in group]
    raise ValueError(f"{grasp_key} has no grasp group")


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)
