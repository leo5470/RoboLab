"""PPS episode-layout HDF5 writer for scripted demonstrations (the pps repository's ``docs/omy_export_spec.md`` §2).

One file per object family, one group per successful episode, appended as episodes finish so long collections can
resume. Rows are in the wire convention because they come from ``Pi0DroidJointposClient``'s own conversion
(requirement R1); in the stock DROID registration the robot-native command equals the wire command, so
``original_actions`` equals ``actions``.
"""

from __future__ import annotations

import dataclasses
import pathlib

import h5py
import numpy as np

FPS = 15
IMAGE_KEYS = ("exterior_image_1_left", "wrist_image_left")
# Per-episode group of the scene before each action (not read by PPS); makes the episode a MimicGen source
# (robolab/core/scripted_demos/mimic_source.py).
STATE_GROUP = "pre_action_state"


@dataclasses.dataclass
class EpisodeBuffer:
    """Rows of one attempt; row ``t`` is the observation before ``actions[t]`` (pre-action alignment, spec §3.4)."""

    joint_position: list = dataclasses.field(default_factory=list)
    gripper_position: list = dataclasses.field(default_factory=list)
    actions: list = dataclasses.field(default_factory=list)
    exterior_image_1_left: list = dataclasses.field(default_factory=list)
    wrist_image_left: list = dataclasses.field(default_factory=list)
    raw_finger_position: list = dataclasses.field(default_factory=list)
    normalized_gripper_position: list = dataclasses.field(default_factory=list)
    source_sim_before_s: list = dataclasses.field(default_factory=list)
    source_sim_origin_s: float = 0.0
    # Scene before each action, for use as a MimicGen source: "<kind>/<name>/<field>" -> rows, in the layout of Isaac
    # Lab's recorder state ("articulation/robot/joint_position", "rigid_object/<name>/root_pose"); see STATE_GROUP.
    state: dict = dataclasses.field(default_factory=dict)
    joint_names: tuple | None = None

    def add(self, request: dict, action: np.ndarray, raw_finger: float, sim_time: float,
            state: dict | None = None, joint_names: tuple | None = None) -> None:
        if state is not None:
            if len(self) and set(state) != set(self.state):
                raise ValueError("pre-action state must cover the same entries on every row")
            for key, value in state.items():
                self.state.setdefault(key, []).append(np.asarray(value))
            self.joint_names = joint_names or self.joint_names
        elif self.state:
            raise ValueError("pre-action state must be recorded on every row or none")
        self.joint_position.append(np.asarray(request["observation/joint_position"], dtype=np.float32).reshape(7))
        self.gripper_position.append(np.asarray(request["observation/gripper_position"], dtype=np.float32).reshape(1))
        self.exterior_image_1_left.append(np.asarray(request["observation/exterior_image_1_left"], dtype=np.uint8))
        self.wrist_image_left.append(np.asarray(request["observation/wrist_image_left"], dtype=np.uint8))
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(8))
        self.raw_finger_position.append(np.float32(raw_finger))
        self.normalized_gripper_position.append(np.float32(raw_finger / (np.pi / 4)))
        self.source_sim_before_s.append(float(sim_time))

    def __len__(self) -> int:
        return len(self.actions)

    def arrays(self) -> dict[str, np.ndarray]:
        n = len(self)
        actions = np.stack(self.actions).astype(np.float32)
        return {
            "joint_position": np.stack(self.joint_position).astype(np.float32),
            "gripper_position": np.stack(self.gripper_position).astype(np.float32),
            "actions": actions,
            "exterior_image_1_left": np.stack(self.exterior_image_1_left).astype(np.uint8),
            "wrist_image_left": np.stack(self.wrist_image_left).astype(np.uint8),
            "original_actions": actions.copy(),
            "raw_finger_position": np.asarray(self.raw_finger_position, dtype=np.float32).reshape(n, 1),
            "normalized_gripper_position": np.asarray(self.normalized_gripper_position, dtype=np.float32).reshape(n, 1),
            "source_step": np.arange(n, dtype=np.int64),
            "timestamp": np.arange(n, dtype=np.float64) / FPS,
            "source_sim_before_s": np.asarray(self.source_sim_before_s, dtype=np.float64),
            "source_sim_origin_s": np.full(n, self.source_sim_origin_s, dtype=np.float64),
            **{f"{STATE_GROUP}/{key}": np.stack(rows).astype(np.float32) for key, rows in self.state.items()},
            **({f"{STATE_GROUP}/articulation/robot/joint_names": np.array(self.joint_names, dtype="S")}
               if self.state and self.joint_names else {}),
        }


def check_episode(arrays: dict[str, np.ndarray]) -> None:
    """The PPS reader's per-episode checks (pps ``src/pps/droid/data.py:35-95`` at d9c50fd), raised early."""
    n = len(arrays["actions"])
    if n < 1:
        raise ValueError("empty episode")
    widths = {"joint_position": 7, "gripper_position": 1, "actions": 8, "original_actions": 8,
              "raw_finger_position": 1, "normalized_gripper_position": 1}
    for key, width in widths.items():
        a = arrays[key]
        if a.shape != (n, width) or a.dtype != np.float32:
            raise ValueError(f"{key}: expected float32 ({n}, {width}), got {a.shape}/{a.dtype}")
        if not np.isfinite(a).all():
            raise ValueError(f"nonfinite values in {key}")
    for key, col in (("gripper_position", 0), ("actions", 7)):
        lo, hi = float(arrays[key][:, col].min()), float(arrays[key][:, col].max())
        if lo < -1e-6 or hi > 1 + 1e-6:
            raise ValueError(f"{key}: gripper outside [0, 1] ({lo}, {hi})")
    for key in IMAGE_KEYS:
        a = arrays[key]
        if a.ndim != 4 or a.shape[0] != n or a.shape[-1] != 3 or a.dtype != np.uint8:
            raise ValueError(f"{key}: expected uint8 (N,H,W,3)")
    for key, dtype in (("source_step", np.int64), ("timestamp", np.float64), ("source_sim_before_s", np.float64),
                       ("source_sim_origin_s", np.float64)):
        if arrays[key].shape != (n,) or arrays[key].dtype != dtype or not np.isfinite(arrays[key]).all():
            raise ValueError(f"invalid audit field {key}")
    if not np.array_equal(arrays["source_step"], np.arange(n)):
        raise ValueError("source_step must be 0..N-1")
    if not np.allclose(arrays["timestamp"], np.arange(n) / FPS, atol=1e-8, rtol=0):
        raise ValueError("timestamp must be t/15")


class Writer:
    """Appends successful episodes to ``path``; the file stays valid between episodes."""

    def __init__(self, path: pathlib.Path, *, task: str, prompt: str):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.path, "a") as f:
            for key, value in (("fps", FPS), ("task", task), ("prompt", prompt)):
                if key in f.attrs and f.attrs[key] != value:
                    raise ValueError(f"{self.path}: existing {key}={f.attrs[key]!r} differs from {value!r}")
                f.attrs[key] = value
            f.require_group("data")

    def episodes(self) -> list[dict]:
        with h5py.File(self.path, "r") as f:
            return [dict(name=name, **{k: (v.decode() if isinstance(v, bytes) else v) for k, v in f["data"][name].attrs.items()})
                    for name in sorted(f["data"])]

    def append(self, arrays: dict[str, np.ndarray], attrs: dict) -> str:
        check_episode(arrays)
        with h5py.File(self.path, "a") as f:
            data = f["data"]
            name = f"episode_{len(data):06d}"
            group = data.create_group(name)
            for key, value in arrays.items():
                if value.ndim == 4:
                    group.create_dataset(key, data=value, chunks=(1, *value.shape[1:]), compression="lzf")
                else:
                    group.create_dataset(key, data=value)
            group.attrs["num_samples"] = len(arrays["actions"])
            group.attrs["success"] = True
            for key, value in attrs.items():
                group.attrs[key] = value
        return name
