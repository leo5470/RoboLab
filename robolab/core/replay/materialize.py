# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline materialization of camera observations for state-only recordings.

Keyboard teleoperation collected with ``--defer-images`` records actions and
per-step scene states but no policy-camera observations, so the control loop
never waits on RTX renders (see ``docs/keyboard_teleoperation.md``). This module
turns such a recording into an ordinary RoboLab image demonstration by walking
the recorded states, restoring each one into a rebuilt environment, rendering,
and storing the resulting observations under the standard HDF5 paths.

Two properties make this faithful rather than approximate:

- **States, not open-loop actions.** Every frame is rendered from the scene
  state the recording actually reached. Re-stepping the recorded actions would
  accumulate contact-rich drift, so the images would slowly stop matching the
  trajectory they are stored next to.
- **Pre-action alignment.** RoboLab records observations pre-step, so
  observation ``t`` is the scene *before* action ``t`` is applied: frame 0 is
  the initial state and frame ``t`` is post-step state ``t - 1`` (see
  :func:`observation_state_source`). Materialized frames use the same mapping,
  so an image demonstration produced here indexes exactly like a live-recorded
  one.

Nothing here imports IsaacSim at module level (the few simulator types needed
are imported inside the functions that use them), and the environment is only
touched through the small protocol used by :func:`render_observations` and
:func:`materialize_episode` — so the file and array helpers can be tested
against plain HDF5 files. ``scripts/materialize_demo_images.py`` is the CLI
driver.
"""

import contextlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import h5py
import numpy as np

# Observation groups that hold rendered sensor data. These are the groups a
# deferred collection omits and this module regenerates; every other recorded
# channel is copied through untouched.
IMAGE_OBS_GROUPS = ("image_obs", "viewport_cam")

# ``data``-group attr holding the JSON metadata a deferred collection stamps
# (task, env name, camera plan, control timing) so materialization can rebuild
# the same environment with its policy cameras restored.
DEFERRED_ATTR = "robolab_deferred_images"

# ``data``-group attr stamped by materialization, recording where the images
# came from. Purely provenance; the recording's own attrs are left untouched.
MATERIALIZED_ATTR = "robolab_materialized"


def observation_state_source(step: int) -> tuple[str, int]:
    """Return the recorded state a pre-action observation must be rendered from.

    RoboLab records observations pre-step (``PreStepFlatPolicyObservationsRecorder``
    stores ``env.obs_buf``, which holds the observation computed at the end of
    the previous step). So observation 0 belongs to the episode's
    ``initial_state`` and observation ``t`` belongs to post-step state ``t - 1``.

    Args:
        step: Index of the observation frame.

    Returns:
        ``(group_name, row)`` — the recorded HDF5 group and the row within it.

    Raises:
        ValueError: If ``step`` is negative.
    """
    if step < 0:
        raise ValueError(f"observation index must be non-negative, got {step}")
    if step == 0:
        return "initial_state", 0
    return "states", step - 1


def state_tree_rows(tree: dict) -> int:
    """Number of rows shared by every leaf of a recorded state tree.

    Raises:
        ValueError: If the tree is empty or its leaves disagree on row count.
    """
    rows = set()

    def _walk(node):
        for value in node.values():
            if isinstance(value, dict):
                _walk(value)
            else:
                rows.add(int(np.asarray(value).shape[0]))

    _walk(tree)
    if not rows:
        raise ValueError("recorded state tree has no leaves")
    if len(rows) > 1:
        raise ValueError(f"recorded state leaves disagree on row count: {sorted(rows)}")
    return rows.pop()


def validate_episode_alignment(name: str, num_actions: int, num_states: int,
                               num_initial_rows: int) -> None:
    """Check that a recorded episode's actions and states line up.

    Actions are recorded pre-step and states post-step, once per control step,
    so a complete episode has exactly one state row per action. A mismatch means
    the recording was truncated (e.g. the process died mid-episode) and the
    frame-to-state mapping would silently shift, so materialization refuses it.

    Raises:
        ValueError: If the counts do not line up or the episode is empty.
    """
    if num_actions == 0:
        raise ValueError(f"{name}: no actions recorded; nothing to materialize")
    if num_states != num_actions:
        raise ValueError(
            f"{name}: recorded {num_actions} actions but {num_states} state rows. "
            "Actions and per-step states are recorded once per control step, so a "
            "mismatch means the episode is truncated and frames cannot be aligned."
        )
    if num_initial_rows < 1:
        raise ValueError(f"{name}: initial_state has no rows; frame 0 cannot be rendered")


def read_group_tree(group: h5py.Group) -> dict:
    """Load a nested HDF5 group as a dict of numpy arrays."""
    out = {}
    for name, item in group.items():
        out[name] = read_group_tree(item) if isinstance(item, h5py.Group) else item[()]
    return out


def episode_names(hdf5_path: str) -> list[str]:
    """List the demo group names in a recording, ordered by demo index."""
    with h5py.File(hdf5_path, "r") as f:
        data = f.get("data")
        if data is None:
            raise ValueError(f"{hdf5_path} has no 'data' group; not a RoboLab recording")
        names = list(data.keys())
    return sorted(names, key=_demo_index)


def _demo_index(name: str) -> int:
    """Sort key for ``demo_<n>`` names; non-conforming names sort last by name."""
    prefix = "demo_"
    if name.startswith(prefix) and name[len(prefix):].isdigit():
        return int(name[len(prefix):])
    return 1 << 30


def read_deferred_metadata(hdf5_path: str) -> dict:
    """Read the deferred-collection metadata stamped on a recording.

    Returns an empty dict for recordings made without ``--defer-images`` (or by
    an older collector); callers fall back to CLI arguments in that case.
    """
    with h5py.File(hdf5_path, "r") as f:
        data = f.get("data")
        if data is None or DEFERRED_ATTR not in data.attrs:
            return {}
        raw = data.attrs[DEFERRED_ATTR]
    if isinstance(raw, bytes):
        raw = raw.decode()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


@contextlib.contextmanager
def atomic_hdf5_output(path: str, overwrite: bool = False):
    """Write an HDF5 file through a temporary path, renaming it only on success.

    A materialization run is long and a half-rendered file is worse than none:
    it looks like a demonstration but its images stop partway. So the file is
    built beside the destination and moved into place only after the ``with``
    body returns; an error, a ``KeyboardInterrupt``, or a killed loop leaves the
    destination absent (or, with ``--overwrite``, still holding the old file)
    and removes the partial write.

    Raises:
        FileExistsError: If ``path`` exists and ``overwrite`` is False.
    """
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"output already exists: {path}. Pass --overwrite to replace it.")
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    temp_path = os.path.join(directory, f".{os.path.basename(path)}.partial-{os.getpid()}")
    handle = h5py.File(temp_path, "w")
    try:
        yield handle
        handle.close()
        os.replace(temp_path, path)
    finally:
        if handle:
            handle.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)


def copy_tree(src: h5py.Group, dst: h5py.Group, drop_paths: tuple[str, ...] = (),
              _prefix: str = "") -> None:
    """Recursively copy an HDF5 group, omitting ``drop_paths`` (slash-separated).

    Subtrees that contain nothing to drop are copied wholesale by h5py, which
    preserves values, dtypes, and attrs exactly.
    """
    drop = set(drop_paths)
    for key, item in src.items():
        path = f"{_prefix}/{key}" if _prefix else key
        if path in drop:
            continue
        if isinstance(item, h5py.Group) and any(d.startswith(f"{path}/") for d in drop):
            sub = dst.create_group(key)
            for attr_key, attr_value in item.attrs.items():
                sub.attrs[attr_key] = attr_value
            copy_tree(item, sub, drop_paths, _prefix=path)
        else:
            src.copy(item, dst, name=key)


def copy_episode(src_demo: h5py.Group, dst_parent: h5py.Group, name: str,
                 drop_groups: tuple[str, ...] = IMAGE_OBS_GROUPS) -> h5py.Group:
    """Copy a recorded episode into another file, minus its image observations.

    Every non-image channel (actions, states, initial_state, ee_pose, bbox,
    proprio observations, subtask status, ...) is copied through with its
    recorded values, and the episode attrs (``num_samples``, ``success``,
    ``seed``) come along unchanged. Image groups are dropped so a re-run can
    regenerate them; on a state-only recording there are none to drop.

    Returns:
        The newly created destination group.
    """
    dst = dst_parent.create_group(name)
    for attr_key, attr_value in src_demo.attrs.items():
        dst.attrs[attr_key] = attr_value
    copy_tree(src_demo, dst, drop_paths=tuple(f"obs/{group}" for group in drop_groups))
    return dst


def image_observation_arrays(obs: dict, groups: tuple[str, ...] = IMAGE_OBS_GROUPS,
                             env_id: int = 0) -> dict[str, np.ndarray]:
    """Flatten one env's rendered observation groups to ``"<group>/<term>" -> array``."""
    arrays: dict[str, np.ndarray] = {}
    for group in groups:
        group_obs = obs.get(group)
        if group_obs is None:
            continue
        for term, value in group_obs.items():
            row = value[env_id]
            arrays[f"{group}/{term}"] = row.detach().cpu().numpy() if hasattr(row, "detach") else np.asarray(row)
    return arrays


class EpisodeImageWriter:
    """Writes rendered observation frames into a demo group, one frame at a time.

    Datasets are allocated at their final length on the first frame (the frame
    count is known up front from the recorded actions), so nothing is buffered
    in memory: a 500-step episode of 1280x720 RGB is well over a gigabyte.
    """

    def __init__(self, demo_group: h5py.Group, num_frames: int, compression: str | None = None):
        self._demo = demo_group
        self._num_frames = num_frames
        self._compression = compression
        self._datasets: dict[str, h5py.Dataset] = {}
        self._written = 0

    @property
    def dataset_paths(self) -> list[str]:
        return sorted(self._datasets)

    def _create(self, arrays: dict[str, np.ndarray]) -> None:
        obs_group = self._demo.require_group("obs")
        for path, array in arrays.items():
            group_name, term = path.split("/", 1)
            target = obs_group.require_group(group_name)
            if term in target:
                del target[term]
            shape = (self._num_frames,) + array.shape
            # One frame per chunk: frames are written in order and read back
            # frame-wise by data loaders, and a full RGB frame is already a
            # multi-megabyte chunk.
            chunks = (1,) + array.shape if array.ndim >= 3 else True
            self._datasets[path] = target.create_dataset(
                term, shape=shape, dtype=array.dtype, chunks=chunks, compression=self._compression
            )

    def write(self, step: int, arrays: dict[str, np.ndarray]) -> None:
        """Store one frame's observation arrays at index ``step``."""
        if step >= self._num_frames:
            raise ValueError(f"frame {step} is past the allocated length {self._num_frames}")
        if not self._datasets:
            self._create(arrays)
        missing = set(self._datasets) - set(arrays)
        if missing:
            raise ValueError(f"frame {step} is missing observation terms: {sorted(missing)}")
        for path, dataset in self._datasets.items():
            dataset[step] = arrays[path]
        self._written += 1

    def finalize(self) -> dict[str, tuple[int, ...]]:
        """Check every allocated frame was written and report dataset shapes.

        Raises:
            ValueError: If fewer frames were written than allocated.
        """
        if self._written != self._num_frames:
            raise ValueError(
                f"wrote {self._written} image frames but the episode has {self._num_frames} actions"
            )
        return {path: tuple(dataset.shape) for path, dataset in self._datasets.items()}


def restore_nulled_cameras(env_cfg, recorded_cfg: dict) -> list[str]:
    """Drop nulled camera sensors from a recorded scene config, in place.

    A state-only recording's ``env_cfg.json`` carries ``"<camera>": null`` for
    every camera the collection left out — that is how the sensors were removed.
    Overlaying that config onto the rebuilt environment would strip the cameras
    back out of the very environment we built to render them, and the image
    observation terms would then fail to resolve their scene entities. So those
    entries are removed from the recorded config before the overlay, leaving the
    freshly registered camera in place; every other recorded value still wins.

    Args:
        env_cfg: The freshly built env config (read only, to identify cameras).
        recorded_cfg: The parsed ``env_cfg.json``, mutated in place.

    Returns:
        Names of the cameras kept from the current registration.
    """
    from isaaclab.sensors import CameraCfg

    scene = recorded_cfg.get("scene")
    if not isinstance(scene, dict):
        return []
    restored = []
    for name in list(scene):
        if scene[name] is None and isinstance(getattr(env_cfg.scene, name, None), CameraCfg):
            del scene[name]
            restored.append(name)
    return restored


def scale_camera_resolutions(env_cfg, scale: float = 1.0) -> dict[str, tuple[int, int]]:
    """Scale every scene camera's resolution, keeping dimensions 8-pixel aligned.

    Args:
        env_cfg: Env config whose ``scene`` camera sensors are resized in place.
        scale: Multiplier on width and height; 1.0 leaves the configured
            resolution untouched and only reports it.

    Returns:
        ``{camera_name: (width, height)}`` after scaling.
    """
    from isaaclab.sensors import CameraCfg

    resolutions = {}
    for name, sensor_cfg in vars(env_cfg.scene).items():
        if not isinstance(sensor_cfg, CameraCfg):
            continue
        if scale != 1.0:
            sensor_cfg.width = max(8, int(round(sensor_cfg.width * scale / 8)) * 8)
            sensor_cfg.height = max(8, int(round(sensor_cfg.height * scale / 8)) * 8)
        resolutions[name] = (sensor_cfg.width, sensor_cfg.height)
    return resolutions


def render_observations(env, renders_per_frame: int = 1) -> dict:
    """Sync the simulator to the scene state just written and compute observations.

    Mirrors what ``ManagerBasedEnv.reset_to`` does after writing a state:
    ``sim.forward()`` pushes the new poses through fabric, ``sim.render()``
    makes the RTX sensors produce a frame for them, and ``scene.update()`` marks
    the sensor buffers outdated so the observation terms read the new frame
    rather than a cached one. No physics is stepped, so the state is exactly the
    recorded one.

    Args:
        env: The environment to render.
        renders_per_frame: Number of ``sim.render()`` calls per frame. One is
            the default because it matches what live recording does — the env
            renders once per control step — so materialized frames carry the
            same degree of temporal accumulation as live-recorded ones. More
            passes give a more converged (less noisy) image of the same state;
            they do not correct any frame offset, since the render already
            reflects the state just written.
    """
    env.sim.forward()
    for _ in range(max(1, renders_per_frame)):
        env.sim.render()
    env.scene.update(dt=env.step_dt)
    return env.observation_manager.compute()


@dataclass
class RecordedEpisode:
    """One recorded episode's state-only content, loaded into memory.

    States are small (tens of floats per step), so the whole tree is held while
    the episode is materialized; the images that dominate size are streamed
    straight to disk.
    """

    name: str
    num_actions: int
    initial_state: dict
    states: dict
    seed: int | None = None
    success: bool | None = None
    attrs: dict = field(default_factory=dict)

    @classmethod
    def load(cls, demo_group: h5py.Group) -> "RecordedEpisode":
        name = demo_group.name.rsplit("/", 1)[-1]
        if "actions" not in demo_group:
            raise ValueError(f"{name}: no 'actions' dataset; not a RoboLab episode")
        for group in ("initial_state", "states"):
            if group not in demo_group:
                raise ValueError(
                    f"{name}: no '{group}' group. Materialization renders recorded states, so a "
                    "recording without them cannot be materialized."
                )
        initial_state = read_group_tree(demo_group["initial_state"])
        states = read_group_tree(demo_group["states"])
        num_actions = int(demo_group["actions"].shape[0])
        validate_episode_alignment(name, num_actions, state_tree_rows(states),
                                   state_tree_rows(initial_state))
        attrs = dict(demo_group.attrs)
        return cls(
            name=name,
            num_actions=num_actions,
            initial_state=initial_state,
            states=states,
            seed=int(attrs["seed"]) if "seed" in attrs else None,
            success=bool(attrs["success"]) if "success" in attrs else None,
            attrs=attrs,
        )

    def state_for(self, step: int) -> tuple[dict, int]:
        """Recorded state tree and row to render observation ``step`` from."""
        group, row = observation_state_source(step)
        return (self.initial_state if group == "initial_state" else self.states), row


def materialize_episode(env, episode: RecordedEpisode, demo_group: h5py.Group, *,
                        compression: str | None = None, renders_per_frame: int = 1,
                        groups: tuple[str, ...] = IMAGE_OBS_GROUPS,
                        progress=None) -> dict[str, tuple[int, ...]]:
    """Render every frame of a recorded episode into ``demo_group``.

    For each frame the recorded scene state is written into the simulator (never
    re-stepped from actions), rendered, and stored under
    ``obs/<group>/<term>`` — the paths an ordinary RoboLab image demonstration
    uses.

    Args:
        env: A rebuilt environment whose policy cameras are configured.
        episode: The recorded episode to render.
        demo_group: Destination demo group (already holding the copied channels).
        compression: HDF5 compression for the new image datasets.
        renders_per_frame: Passed to :func:`render_observations`.
        groups: Observation groups to store.
        progress: Optional callable invoked with each completed frame index.

    Returns:
        Mapping of written dataset path to shape.
    """
    from robolab.core.replay.scene_state import restore_scene_state

    writer = EpisodeImageWriter(demo_group, episode.num_actions, compression=compression)
    for step in range(episode.num_actions):
        state_tree, row = episode.state_for(step)
        restore_scene_state(env, state_tree, row)
        obs = render_observations(env, renders_per_frame=renders_per_frame)
        arrays = image_observation_arrays(obs, groups=groups)
        if not arrays:
            raise ValueError(
                "the rebuilt environment produced no image observations "
                f"(looked for groups {list(groups)}). Rebuild it with policy cameras enabled."
            )
        writer.write(step, arrays)
        if progress is not None:
            progress(step)
    return writer.finalize()


def camera_extrinsics_from_env(env) -> dict[str, dict[str, np.ndarray]]:
    """Camera poses in the layout ``InitialStateRecorder`` writes under ``initial_state/cameras``.

    Position is env-local (world minus the env origin) and orientation is the
    world-frame ROS quaternion, matching ``robolab.core.events.basic_recorders``.
    """
    from isaaclab.sensors import Camera

    origins = env.scene.env_origins[:, 0:3]
    poses = {}
    for name, sensor in env.scene.sensors.items():
        if not isinstance(sensor, Camera):
            continue
        poses[name] = {
            "position": (sensor.data.pos_w - origins).detach().cpu().numpy(),
            "orientation": sensor.data.quat_w_ros.detach().cpu().numpy(),
        }
    return poses


def write_camera_extrinsics(demo_group: h5py.Group, poses: dict[str, dict[str, np.ndarray]]) -> None:
    """Add ``initial_state/cameras`` to a materialized demo if the recording had none.

    A state-only recording has no cameras in the scene, so the initial-state
    recorder could not stamp their extrinsics. Filling them in from the rebuilt
    environment keeps materialized files structurally identical to ordinary
    image demonstrations. Existing entries are never overwritten.
    """
    if not poses:
        return
    initial_state = demo_group.require_group("initial_state")
    cameras = initial_state.require_group("cameras")
    for name, pose in poses.items():
        if name in cameras:
            continue
        group = cameras.create_group(name)
        for key, value in pose.items():
            group.create_dataset(key, data=np.asarray(value))


def stamp_materialization_provenance(data_group: h5py.Group, source_path: str,
                                     extra: dict | None = None) -> None:
    """Record where a materialized file's images came from, without touching recording provenance."""
    import importlib.metadata

    info = {
        "source": os.path.abspath(source_path),
        "materialized_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for package in ("isaaclab", "isaacsim"):
        try:
            info[f"{package}_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if extra:
        info.update(extra)
    data_group.attrs[MATERIALIZED_ATTR] = json.dumps(info)
