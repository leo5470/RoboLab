"""Simulator-free source-pool contracts, source-episode readers and independent trajectory RNG state."""

import ast
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import random

import h5py
import numpy as np

STRATEGIES = ("random", "nearest_neighbor_object", "nearest_neighbor_robot_distance")
ROOT = Path(__file__).resolve().parents[3]
# Success parameters that may differ between sources and target. Differences are
# recorded on each source reference (contract_differences), never ignored silently.
# speed_source only selects how resting speed is measured (reported vs pose-derived).
ALLOWED_SUCCESS_DIFFERENCES = ("speed_source",)


def source_record(index, sources):
    """Nearest-neighbor strategies can return scalar tensors rather than ints."""
    index = int(index)
    return {"source": index, "source_reference": sources[index]}


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _literal_defaults(func):
    """Literal keyword defaults of a "module:function" reference, read statically.

    Recorded configs store only the explicitly passed parameters; filling in the
    defaults makes a recording that relied on a default compare equal to one that
    wrote the same value explicitly. Unresolvable references return no defaults.
    """
    if not isinstance(func, str) or ":" not in func:
        return {}
    module, name = func.split(":", 1)
    path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
    if not path.is_file():
        return {}
    node = next((n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if node is None:
        return {}
    positional = node.args.posonlyargs + node.args.args
    pairs = list(zip(positional[len(positional) - len(node.args.defaults):], node.args.defaults))
    pairs += [(a, d) for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults) if d is not None]
    defaults = {}
    for arg, default in pairs:
        if arg.arg == "env_id":
            continue
        try:
            defaults[arg.arg] = ast.literal_eval(default)
        except ValueError:
            continue
    return defaults


def normalized_success(success):
    """Recorded success term with the function's literal defaults filled in."""
    defaults = _literal_defaults(success.get("func"))
    return {**success, "params": {**defaults, **(success.get("params") or {})}}


def contract_differences(expected, actual):
    """Return (blocking keys, allowed differences) between two contracts."""
    blocking = [k for k in expected if k != "success" and expected[k] != actual.get(k)]
    allowed = {}
    first, second = expected["success"], actual["success"]
    strip = lambda s: {**s, "params": {k: v for k, v in s["params"].items() if k not in ALLOWED_SUCCESS_DIFFERENCES}}
    if strip(first) != strip(second):
        blocking.append("success")
    else:
        for key in ALLOWED_SUCCESS_DIFFERENCES:
            if first["params"].get(key) != second["params"].get(key):
                allowed[f"success.params.{key}"] = {"target": first["params"].get(key),
                                                    "source": second["params"].get(key)}
    return blocking, allowed


def contract(run):
    run = Path(run)
    cfg = json.loads((run / "env_cfg.json").read_text())
    with h5py.File(run / "omy_demos.hdf5") as stream:
        meta = json.loads(stream["data"].attrs["robolab_omy_teleop"])
    success = cfg["terminations"]["success"]
    objects = success["params"]
    return {"sim": cfg["sim"], "decimation": cfg["decimation"], "actions": cfg["actions"],
            "robot": cfg["scene"]["robot"], "success": normalized_success(success),
            "assets": {name: {k: v for k, v in cfg["scene"][name].items() if k != "init_state"}
                       for name in (objects["object"], objects["container"])},
            "profiles": {k: cfg.get(k) for k in ("omy_stepping_profile", "omy_recording_profile", "omy_contact_sensor_profile")},
            "joint_order": meta.get("performance", {}).get("joint_order"),
            "action_metadata": {k: meta.get(k) for k in ("action_type", "action_names", "joint_units", "gripper_encoding", "control_hz")}}


def source_references(args):
    """Resolve exact source episodes before launching the simulator."""
    if args.source_selection:
        manifest = Path(args.source_selection).resolve()
        rows = json.loads(manifest.read_text())["sources"]
        references = []
        for row in rows:
            path = Path(row["file"])
            if not path.is_absolute():
                path = manifest.parent / path
            references.append({**row, "file": str(path.resolve())})
    else:
        references = []
        for run in args.input:
            path = (Path(run) / "omy_demos.hdf5").resolve()
            with h5py.File(path) as stream:
                for name in sorted(stream["data"], key=lambda n: int(n.split("_")[-1])):
                    if args.episodes is not None and int(name.split("_")[-1]) not in args.episodes:
                        continue
                    if stream["data"][name].attrs.get("success", False):
                        references.append({"file": str(path), "episode": name})
        if args.episodes is not None:
            missing = {f"demo_{i}" for i in args.episodes} - {r["episode"] for r in references}
            if missing:
                raise ValueError(f"Missing/unsuccessful selected sources: {sorted(missing)}")
    if not references:
        raise ValueError("Source pool is empty")
    seen, contracts, hashes = set(), {}, {}
    target = Path(args.target_input or args.input[0]).resolve()
    expected = contract(target)
    for ref in references:
        key = ref["file"], ref["episode"]
        if key in seen:
            raise ValueError(f"Duplicate source reference: {key}")
        seen.add(key)
        run = Path(ref["file"]).parent
        if run not in contracts:
            contracts[run] = contract(run)
        differences, allowed = contract_differences(expected, contracts[run])
        if differences:
            raise ValueError(f"Incompatible source {run}: {differences}")
        # Selection-manifest rows may carry an entry from an earlier comparison.
        ref.pop("contract_differences", None)
        if allowed:
            ref["contract_differences"] = allowed
        path = ref["file"]
        if path not in hashes:
            hashes[path] = checksum(path)
        if ref.get("sha256", hashes[path]) != hashes[path]:
            raise ValueError(f"Source checksum changed: {path}")
        ref["sha256"] = hashes[path]
        with h5py.File(path) as stream:
            if ref["episode"] not in stream["data"] or not stream["data"][ref["episode"]].attrs.get("success", False):
                raise ValueError(f"Missing/unsuccessful source: {key}")
            ref["task"] = json.loads(stream["data"].attrs["robolab_omy_teleop"])["task"]
    return references


@dataclass
class SourceEpisode:
    """What MimicGen reads from one source demonstration, whichever kind of recording it comes from.

    Row ``t`` is the scene before ``actions[t]``. Poses are root poses ``x y z qw qx qy qz`` relative to the env
    origin (the robot root), as Isaac Lab's recorder stores them.
    """

    actions: np.ndarray  # (N, 8): ARM_JOINTS targets, then the gripper command (+1 open, -1 closed)
    joint_position: np.ndarray  # (N, J): the robot's full joint vector, in articulation order
    object_pose: dict  # asset name -> (N, 7)
    grasp: int  # first step at which the closed gripper has raised the object (grasp_index)
    success: bool
    joint_names: tuple | None = None  # articulation order of joint_position, when the recording stores it


def grasp_index(gripper_command, object_z, lift=0.02):
    """First step at which the closed gripper has raised the object by ``lift`` meters."""
    held = (np.asarray(gripper_command) < 0) & (np.asarray(object_z) - object_z[0] > lift)
    if not held.any():
        raise ValueError("the object is never lifted with the gripper closed")
    return int(np.argmax(held))


def pre_action(initial, states):
    """Rows seen before each action: the initial state, then post-step state ``t - 1``."""
    return np.concatenate((np.asarray(initial)[:1], np.asarray(states)[:-1]))


def source_arrays(episode, object_name, container_name):
    """Pre-action joints and reference poses of one recorded episode, as NumPy arrays."""
    actions = np.asarray(episode["actions"])
    initial, states = episode["initial_state"], episode["states"]
    if len(states["articulation"]["robot"]["joint_position"]) != len(actions):
        raise ValueError("recorded actions and states are not aligned")
    poses = {name: pre_action(initial["rigid_object"][name]["root_pose"], states["rigid_object"][name]["root_pose"])
             for name in (object_name, container_name)}
    return {
        "actions": actions,
        "joint_position": pre_action(initial["articulation"]["robot"]["joint_position"],
                                     states["articulation"]["robot"]["joint_position"]),
        "object_pose": poses,
        "grasp": grasp_index(actions[:, -1], poses[object_name][:, 2]),
    }


def read_teleop_episode(path, episode, object_name, container_name):
    """One episode of an Isaac Lab recording (``omy_demos.hdf5``: ``initial_state``, ``states``, ``actions``).

    Only the datasets ``source_arrays`` uses are read (no images), with h5py alone, so CPU tools can call it."""
    with h5py.File(path, "r") as file:
        group = file["data"][episode]
        success = bool(group.attrs.get("success", False))
        tree = {"actions": group["actions"][()]}
        for part in ("initial_state", "states"):
            tree[part] = {
                "articulation": {"robot": {"joint_position": group[f"{part}/articulation/robot/joint_position"][()]}},
                "rigid_object": {name: {"root_pose": group[f"{part}/rigid_object/{name}/root_pose"][()]}
                                 for name in (object_name, container_name)},
            }
    arrays = source_arrays(tree, object_name, container_name)
    return SourceEpisode(actions=arrays["actions"], joint_position=arrays["joint_position"],
                         object_pose=arrays["object_pose"], grasp=arrays["grasp"], success=success)


# Reader per source kind: a function, or "module:function" imported on first use. A reference without a
# "kind" is a teleoperation (or teleoperation-format MimicGen) recording.
SOURCE_READERS = {
    "teleop": read_teleop_episode,
    "scripted": "robolab.core.scripted_demos.mimic_source:read_episode",  # docs/scripted_demos.md
}


def read_source(reference, object_name, container_name):
    """The :class:`SourceEpisode` of a source reference ``{"file", "episode", "kind"?}``."""
    kind = reference.get("kind", "teleop")
    if kind not in SOURCE_READERS:
        raise ValueError(f"Unknown source kind {kind!r} for {reference['file']}:{reference['episode']}; "
                         f"known: {sorted(SOURCE_READERS)}")
    reader = SOURCE_READERS[kind]
    if isinstance(reader, str):
        module, name = reader.split(":")
        reader = getattr(importlib.import_module(module), name)
    return reader(reference["file"], reference["episode"], object_name, container_name)


@contextmanager
def trajectory_rng(seed):
    """Keep source/segment randomness from changing scene/action randomness."""
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
