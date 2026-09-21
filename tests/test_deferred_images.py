# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deferred-image demo collection and offline materialization.

Keyboard collection with ``--defer-images`` records actions and per-step scene
states but no camera observations, and
``scripts/materialize_demo_images.py`` renders the images afterwards from those
states. The two halves have to agree on one thing above all: which recorded
state each frame is rendered from. RoboLab records observations pre-step, so
frame 0 is the initial state and frame ``t`` is post-step state ``t - 1``; get
that wrong and every materialized demonstration is silently off by one frame.
These tests pin that mapping, the "no policy cameras were configured" half of
the contract, and the file-level guarantees (exact copies, atomic output).
"""

import os
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch

from robolab.constants import PACKAGE_DIR
from robolab.core.replay.materialize import (
    EpisodeImageWriter,
    RecordedEpisode,
    atomic_hdf5_output,
    copy_episode,
    episode_names,
    materialize_episode,
    observation_state_source,
    read_deferred_metadata,
    state_tree_rows,
    validate_episode_alignment,
)

NUM_STEPS = 4
CAMERA = "over_shoulder_left_camera"


# ---------------------------------------------------------------------------
# Fixtures: a synthetic state-only recording and a fake env that "renders" it
# ---------------------------------------------------------------------------

def _state_tree(rows: int, offset: float = 0.0) -> dict:
    """A minimal InteractiveScene.get_state()-shaped tree with distinguishable rows."""
    steps = np.arange(rows, dtype=np.float32).reshape(rows, 1) + offset
    return {
        "articulation": {
            "robot": {
                "root_pose": np.tile(steps, (1, 7)),
                "root_velocity": np.tile(steps, (1, 6)),
                "joint_position": np.tile(steps, (1, 3)),
                "joint_velocity": np.tile(steps, (1, 3)),
            }
        },
        "rigid_object": {"banana": {"root_pose": np.tile(steps, (1, 7)),
                                    "root_velocity": np.tile(steps, (1, 6))}},
    }


def _write_tree(group: h5py.Group, tree: dict) -> None:
    for key, value in tree.items():
        if isinstance(value, dict):
            _write_tree(group.create_group(key), value)
        else:
            group.create_dataset(key, data=value)


@pytest.fixture
def state_only_recording(tmp_path):
    """A state-only recording shaped like the keyboard collector's output."""
    path = tmp_path / "keyboard_demos.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["total"] = NUM_STEPS
        data.attrs["env_args"] = '{"env_name": "", "type": 2}'
        data.attrs["robolab_deferred_images"] = (
            '{"schema_version": 1, "task": "BananaInBowlTask", "policy_cameras": false, '
            '"env_name": "BananaInBowlTaskRelIK", "control_hz": 15.0}'
        )
        demo = data.create_group("demo_0")
        demo.attrs["num_samples"] = NUM_STEPS
        demo.attrs["success"] = True
        demo.attrs["seed"] = 7
        demo.create_dataset("actions", data=np.arange(NUM_STEPS * 7, dtype=np.float32).reshape(NUM_STEPS, 7))
        _write_tree(demo.create_group("initial_state"), _state_tree(1, offset=-1.0))
        _write_tree(demo.create_group("states"), _state_tree(NUM_STEPS))
        obs = demo.create_group("obs")
        proprio = obs.create_group("proprio_obs")
        proprio.create_dataset("arm_joint_pos", data=np.full((NUM_STEPS, 7), 0.25, dtype=np.float32))
        ee = demo.create_group("ee_pose")
        ee.create_dataset("position", data=np.ones((NUM_STEPS, 3), dtype=np.float32))
    return str(path)


class _FakeScene:
    """Just enough InteractiveScene to drive materialize_episode without a simulator."""

    def __init__(self):
        self.state = {
            "articulation": {"robot": {
                "root_pose": torch.zeros(1, 7), "root_velocity": torch.zeros(1, 6),
                "joint_position": torch.zeros(1, 3), "joint_velocity": torch.zeros(1, 3)}},
            "rigid_object": {"banana": {"root_pose": torch.zeros(1, 7),
                                        "root_velocity": torch.zeros(1, 6)}},
        }
        self.applied: list[float] = []

    def get_state(self, is_relative: bool = False):
        return {group: {name: dict(fields) for name, fields in entities.items()}
                for group, entities in self.state.items()}

    def reset_to(self, state, env_ids=None, is_relative=False):
        self.state = state
        # Row 0 of each recorded leaf is the step index (see _state_tree), so
        # this records which recorded state row the frame was rendered from.
        self.applied.append(float(state["articulation"]["robot"]["joint_position"][0, 0]))

    def update(self, dt):
        pass


class _FakeEnv:
    """Fake env whose "rendered" image encodes the restored state row."""

    class _Sim:
        def __init__(self, env):
            self._env = env

        def forward(self):
            self._env.renders.append("forward")

        def render(self):
            self._env.renders.append("render")

    class _ObservationManager:
        def __init__(self, env):
            self._env = env

        def compute(self):
            value = self._env.scene.applied[-1]
            return {
                "image_obs": {CAMERA: torch.full((1, 2, 2, 3), int(value + 2), dtype=torch.uint8)},
                "proprio_obs": {"arm_joint_pos": torch.zeros(1, 7)},
            }

    def __init__(self):
        self.num_envs = 1
        self.device = "cpu"
        self.step_dt = 0.0667
        self.scene = _FakeScene()
        self.sim = self._Sim(self)
        self.observation_manager = self._ObservationManager(self)
        self.renders: list[str] = []


# ---------------------------------------------------------------------------
# Temporal alignment
# ---------------------------------------------------------------------------

def test_frame_zero_uses_initial_state_and_later_frames_use_the_previous_post_state():
    # Observations are recorded pre-step, so frame t shows the scene *before*
    # action t was applied: the initial state for t=0, post-step state t-1 after.
    assert observation_state_source(0) == ("initial_state", 0)
    assert observation_state_source(1) == ("states", 0)
    assert observation_state_source(9) == ("states", 8)
    with pytest.raises(ValueError):
        observation_state_source(-1)


def test_episode_state_lookup_follows_the_pre_action_mapping(state_only_recording):
    with h5py.File(state_only_recording, "r") as f:
        episode = RecordedEpisode.load(f["data/demo_0"])

    tree, row = episode.state_for(0)
    assert tree is episode.initial_state and row == 0
    for step in range(1, episode.num_actions):
        tree, row = episode.state_for(step)
        assert tree is episode.states
        assert row == step - 1


def test_materialized_frames_are_rendered_from_the_recorded_states(state_only_recording, tmp_path):
    env = _FakeEnv()
    with h5py.File(state_only_recording, "r") as src, h5py.File(tmp_path / "out.hdf5", "w") as dst:
        episode = RecordedEpisode.load(src["data/demo_0"])
        demo = copy_episode(src["data/demo_0"], dst.create_group("data"), "demo_0")
        shapes = materialize_episode(env, episode, demo)

        # Frame 0 restored the initial state (row value -1), frames 1..T-1
        # restored post-step states 0..T-2 (row values 0, 1, ...).
        assert env.scene.applied == [-1.0, 0.0, 1.0, 2.0]
        # Every frame syncs the sim and renders before observations are read.
        assert env.renders == ["forward", "render"] * NUM_STEPS
        assert shapes == {f"image_obs/{CAMERA}": (NUM_STEPS, 2, 2, 3)}
        # The stored image of each frame is the state it was rendered from.
        np.testing.assert_array_equal(
            dst[f"data/demo_0/obs/image_obs/{CAMERA}"][:, 0, 0, 0],
            np.array([1, 2, 3, 4], dtype=np.uint8),
        )


def test_materialized_image_count_equals_action_count(state_only_recording, tmp_path):
    env = _FakeEnv()
    with h5py.File(state_only_recording, "r") as src, h5py.File(tmp_path / "out.hdf5", "w") as dst:
        episode = RecordedEpisode.load(src["data/demo_0"])
        demo = copy_episode(src["data/demo_0"], dst.create_group("data"), "demo_0")
        materialize_episode(env, episode, demo)

        num_actions = dst["data/demo_0/actions"].shape[0]
        num_states = dst["data/demo_0/states/articulation/robot/root_pose"].shape[0]
        num_images = dst[f"data/demo_0/obs/image_obs/{CAMERA}"].shape[0]
        assert num_actions == num_states == num_images == NUM_STEPS


def test_short_render_pass_is_rejected_rather_than_written(tmp_path):
    # A writer that stops early would leave zero-filled trailing frames that
    # still look like a complete demonstration; finalize() refuses instead.
    with h5py.File(tmp_path / "out.hdf5", "w") as f:
        writer = EpisodeImageWriter(f.create_group("demo_0"), num_frames=3)
        writer.write(0, {f"image_obs/{CAMERA}": np.zeros((2, 2, 3), dtype=np.uint8)})
        with pytest.raises(ValueError, match="wrote 1 image frames"):
            writer.finalize()


# ---------------------------------------------------------------------------
# State-only recordings and exact copying
# ---------------------------------------------------------------------------

def test_state_only_episode_has_actions_and_states_but_no_images(state_only_recording):
    with h5py.File(state_only_recording, "r") as f:
        demo = f["data/demo_0"]
        assert demo["actions"].shape == (NUM_STEPS, 7)
        assert state_tree_rows({"states": {"x": demo["states/articulation/robot/root_pose"][()]}}) == NUM_STEPS
        assert "obs" in demo and "proprio_obs" in demo["obs"]
        assert "image_obs" not in demo["obs"]
        assert "viewport_cam" not in demo["obs"]

    metadata = read_deferred_metadata(state_only_recording)
    assert metadata["policy_cameras"] is False
    assert metadata["task"] == "BananaInBowlTask"
    assert episode_names(state_only_recording) == ["demo_0"]


def test_non_image_channels_are_copied_exactly(state_only_recording, tmp_path):
    with h5py.File(state_only_recording, "r") as src, h5py.File(tmp_path / "out.hdf5", "w") as dst:
        source_demo = src["data/demo_0"]
        demo = copy_episode(source_demo, dst.create_group("data"), "demo_0")
        materialize_episode(_FakeEnv(), RecordedEpisode.load(source_demo), demo)

        def leaves(group, prefix=""):
            found = {}
            for key, item in group.items():
                path = f"{prefix}/{key}" if prefix else key
                if isinstance(item, h5py.Group):
                    found.update(leaves(item, path))
                else:
                    found[path] = item[()]
            return found

        recorded = leaves(source_demo)
        materialized = leaves(demo)
        assert set(recorded) <= set(materialized)
        for path, value in recorded.items():
            np.testing.assert_array_equal(materialized[path], value, err_msg=path)
        # Episode provenance survives the copy.
        assert demo.attrs["success"] == source_demo.attrs["success"]
        assert demo.attrs["seed"] == 7
        # ...and the only additions are the rendered images.
        assert set(materialized) - set(recorded) == {f"obs/image_obs/{CAMERA}"}


def test_truncated_episodes_are_refused():
    validate_episode_alignment("demo_0", num_actions=5, num_states=5, num_initial_rows=1)
    with pytest.raises(ValueError, match="truncated"):
        validate_episode_alignment("demo_0", num_actions=5, num_states=4, num_initial_rows=1)
    with pytest.raises(ValueError, match="no actions"):
        validate_episode_alignment("demo_0", num_actions=0, num_states=0, num_initial_rows=1)


# ---------------------------------------------------------------------------
# Output protection and atomicity
# ---------------------------------------------------------------------------

def test_existing_output_is_protected_unless_overwrite_is_given(tmp_path):
    output = tmp_path / "final.hdf5"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="--overwrite"):
        with atomic_hdf5_output(str(output)):
            pass
    assert output.read_bytes() == b"existing"

    with atomic_hdf5_output(str(output), overwrite=True) as f:
        f.create_group("data")
    with h5py.File(output, "r") as f:
        assert "data" in f


def test_interrupted_conversion_leaves_no_partial_file(tmp_path):
    output = tmp_path / "final.hdf5"

    with pytest.raises(KeyboardInterrupt):
        with atomic_hdf5_output(str(output)) as f:
            f.create_group("data/demo_0")
            raise KeyboardInterrupt
    assert not output.exists()
    assert list(tmp_path.iterdir()) == [], "a partial file was left behind"

    # An existing output is left untouched when a re-run is interrupted.
    output.write_bytes(b"previous")
    with pytest.raises(RuntimeError):
        with atomic_hdf5_output(str(output), overwrite=True) as f:
            f.create_group("data")
            raise RuntimeError("render failed")
    assert output.read_bytes() == b"previous"


def test_cli_refuses_to_clobber_an_existing_output(tmp_path, state_only_recording):
    # Guards the argparse wiring: this must fail before Isaac Sim is launched,
    # so it stays a sub-second check rather than a minute of startup.
    output = tmp_path / "final.hdf5"
    output.write_bytes(b"existing")
    script = os.path.join(PACKAGE_DIR, "scripts", "materialize_demo_images.py")

    result = subprocess.run(
        [sys.executable, script, "--input", state_only_recording, "--output", str(output)],
        capture_output=True, text=True, cwd=PACKAGE_DIR, timeout=300,
        env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "Y"},
    )
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert output.read_bytes() == b"existing"

    missing = subprocess.run(
        [sys.executable, script, "--input", str(tmp_path / "nope.hdf5"), "--output", str(tmp_path / "o.hdf5")],
        capture_output=True, text=True, cwd=PACKAGE_DIR, timeout=300,
        env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "Y"},
    )
    assert missing.returncode != 0
    assert "not found" in missing.stderr


# ---------------------------------------------------------------------------
# Deferred collection configuration
# ---------------------------------------------------------------------------

def test_deferred_registration_configures_no_policy_cameras():
    from isaaclab.sensors import CameraCfg

    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.factory import get_envs
    from robolab.registrations.droid.auto_env_registrations_rel_ik import (
        auto_register_droid_rel_ik_envs,
    )

    auto_register_droid_rel_ik_envs(
        task="BananaInBowlTask", env_postfix="DeferredImagesTest",
        include_viewport_camera=False, include_policy_cameras=False,
    )
    env_names = [n for n in get_envs(task="BananaInBowlTask") if n.endswith("DeferredImagesTest")]
    assert env_names, "deferred registration produced no environment"
    env_cfg = parse_env_cfg(env_names[0], device="cpu", seed=0, num_envs=1, use_fabric=True)

    # No camera sensor anywhere in the scene: not the scene-mounted policy
    # cameras, and not the robot-mounted wrist camera either. Nothing renders
    # per control step, which is the whole point of deferring.
    cameras = [name for name, value in vars(env_cfg.scene).items() if isinstance(value, CameraCfg)]
    assert cameras == []
    obs_groups = [name for name in vars(env_cfg.observations) if not name.startswith("_")]
    assert obs_groups == ["proprio_obs"]


def test_recorded_null_cameras_do_not_strip_the_rebuilt_ones():
    # A state-only recording removed its cameras by setting them to null, and
    # env_cfg.json records those nulls. Overlaying the recorded config verbatim
    # would delete the cameras from the environment we rebuilt to render with,
    # and the image observation terms would then fail to resolve.
    from isaaclab.sensors import CameraCfg

    from robolab.core.replay import apply_recorded_env_cfg
    from robolab.core.replay.materialize import restore_nulled_cameras

    class _Scene:
        def __init__(self):
            self.wrist_cam = CameraCfg(prim_path="/World/wrist_cam", height=8, width=8,
                                       data_types=["rgb"])
            self.episode_length_s = 10.0

    class _Cfg:
        def __init__(self):
            self.scene = _Scene()

    env_cfg = _Cfg()
    recorded = {"scene": {"wrist_cam": None, "episode_length_s": 25.0}}

    assert restore_nulled_cameras(env_cfg, recorded) == ["wrist_cam"]
    apply_recorded_env_cfg(env_cfg, recorded)
    assert isinstance(env_cfg.scene.wrist_cam, CameraCfg)
    # Every other recorded value still wins.
    assert env_cfg.scene.episode_length_s == 25.0


def test_default_registration_still_configures_policy_cameras():
    from isaaclab.sensors import CameraCfg

    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.factory import get_envs
    from robolab.registrations.droid.auto_env_registrations_rel_ik import (
        auto_register_droid_rel_ik_envs,
    )

    auto_register_droid_rel_ik_envs(
        task="BananaInBowlTask", env_postfix="PolicyCamerasTest", include_viewport_camera=False,
    )
    env_names = [n for n in get_envs(task="BananaInBowlTask") if n.endswith("PolicyCamerasTest")]
    assert env_names
    env_cfg = parse_env_cfg(env_names[0], device="cpu", seed=0, num_envs=1, use_fabric=True)

    cameras = {name for name, value in vars(env_cfg.scene).items() if isinstance(value, CameraCfg)}
    assert cameras == {"over_shoulder_left_camera", "wrist_cam"}
    assert "image_obs" in vars(env_cfg.observations)
