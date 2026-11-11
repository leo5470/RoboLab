"""Per-subtask source selection, lineage and explicit targets, without Isaac Sim.

The installed Isaac Lab Mimic ``DataGenerator`` and ``DataGenInfoPool`` run
against a fake environment; Isaac Sim modules they import are stubbed only
while this module's tests run. The config comes from the production
``mimic_cfg`` and trials run through the production ``execute_trial``.
"""

import argparse
import asyncio
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import h5py
import numpy as np
import pytest
import torch

from robolab.core.omy_teleop import mimicgen
from robolab.core.omy_teleop.mimicgen import (EEF, GRASP_SIGNAL, build_parser, execute_trial, mimic_cfg, target_poses,
                                              trial_lineage, validate_args)
from robolab.core.omy_teleop.mimic_sources import trajectory_rng

ISAAC = Path(mimicgen.__file__).resolve().parents[3] / ".venv/lib/python3.11/site-packages/isaaclab/source"
DATAGEN = ISAAC / "isaaclab_mimic/isaaclab_mimic/datagen"
OBJECT, CONTAINER = "pail", "bin"
pytestmark = pytest.mark.skipif(not DATAGEN.is_dir(), reason="installed Isaac Lab Mimic not found")


class _Config:
    """Stand-in for Isaac Lab configclasses: keyword fields over class defaults."""

    def __init__(self, **values):
        self.__dict__.update(values)


class DataGenConfig(_Config):
    generation_select_src_per_subtask = False
    generation_select_src_per_arm = False
    generation_transform_first_robot_pose = False
    generation_interpolate_from_last_target_pose = True


class SubTaskConfig(_Config):
    selection_strategy_kwargs = {}
    first_subtask_start_offset_range = (0, 0)
    subtask_term_offset_range = (0, 0)
    action_noise = 0.03
    num_interpolation_steps = 5
    num_fixed_steps = 0
    apply_noise_during_interpolation = False


class MimicEnvCfg:
    pass


@pytest.fixture(scope="module")
def datagen():
    """Load the installed generator modules behind temporary Isaac Sim stubs."""
    saved = dict(sys.modules)

    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module
        return module

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    stub("omni")
    sys.modules["omni"].log = stub("omni.log", warn=print, error=print, info=lambda *a, **k: None)
    enum = types.SimpleNamespace(COORDINATION="coordination", REPLAY="replay")
    stub("isaaclab")
    stub("isaaclab.utils")
    stub("isaaclab.envs", ManagerBasedRLMimicEnv=object, ManagerBasedEnv=object, MimicEnvCfg=MimicEnvCfg,
         SubTaskConstraintCoordinationScheme=enum, SubTaskConstraintType=enum)
    stub("isaaclab.envs.mimic_env_cfg", DataGenConfig=DataGenConfig, MimicEnvCfg=MimicEnvCfg,
         SubTaskConfig=SubTaskConfig)
    stub("isaaclab.managers", TerminationTermCfg=object, EventTermCfg=object)
    math = load("isaaclab.utils.math", ISAAC / "isaaclab/isaaclab/utils/math.py")
    sys.modules["isaaclab.utils"].math = math
    episodes = load("isaaclab.utils.datasets.episode_data", ISAAC / "isaaclab/isaaclab/utils/datasets/episode_data.py")
    stub("isaaclab.utils.datasets", EpisodeData=episodes.EpisodeData, HDF5DatasetFileHandler=object)
    stub("isaaclab_mimic").__path__ = []
    stub("isaaclab_mimic.datagen").__path__ = [str(DATAGEN)]
    loaded = {name: load(f"isaaclab_mimic.datagen.{name}", DATAGEN / f"{name}.py")
              for name in ("datagen_info", "selection_strategy", "waypoint", "datagen_info_pool", "data_generator")}

    class AlwaysDifferent(loaded["selection_strategy"].SelectionStrategy):
        """Pick the source after the one picked last, so consecutive choices always differ."""

        NAME = "test_always_different"
        last = -1

        def select_source_demo(self, eef_pose, object_pose, src_subtask_datagen_infos, **kwargs):
            AlwaysDifferent.last = (AlwaysDifferent.last + 1) % len(src_subtask_datagen_infos)
            return AlwaysDifferent.last

    yield types.SimpleNamespace(EpisodeData=episodes.EpisodeData, **loaded)
    # Keep the stubs from leaking into other test modules.
    for name in set(sys.modules) - set(saved):
        del sys.modules[name]
    sys.modules.update(saved)


def _pose(xyz, yaw=0.0):
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor(xyz, dtype=torch.float32)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    pose[:2, :2] = torch.tensor([[c, -s], [s, c]])
    return pose


class FakeEnv:
    device = "cpu"

    def __init__(self, cfg):
        self.cfg = cfg
        self.objects = {OBJECT: _pose([0.5, 0.0, 0.02]), CONTAINER: _pose([0.5, 0.3, 0.0])}

    def actions_to_gripper_actions(self, actions):
        return {EEF: actions[:, -1:]}

    def get_object_poses(self, env_ids=None):
        return {name: pose.unsqueeze(0) for name, pose in self.objects.items()}

    def get_robot_eef_pose(self, eef_name, env_ids=None):
        return _pose([0.3, 0.0, 0.5]).unsqueeze(0)


def _args(**overrides):
    values = dict(task="Test", seed=0, num_demos=1, keep_failed=False, selection_strategy="random", nn_k=3,
                  action_noise=0.0, interpolation_s=1.0, select_source_per_subtask=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def _generator(datagen, sources=4, **overrides):
    """Production config over ``sources`` synthetic episodes whose segments differ in length and pose."""
    cfg = type("Cfg", (), {})()
    cfg.sim, cfg.decimation = types.SimpleNamespace(dt=1 / 120), 8
    args = _args(source_pool_size=sources, **overrides)
    mimic_cfg(cfg, OBJECT, CONTAINER, args)
    env = FakeEnv(cfg)
    pool = datagen.datagen_info_pool.DataGenInfoPool(env, cfg, "cpu")
    rng = torch.Generator().manual_seed(7)
    for index in range(sources):
        steps = 40 + 3 * index
        eef = torch.stack([_pose(row.tolist()) for row in torch.rand(steps, 3, generator=rng)])
        episode = datagen.EpisodeData()
        episode.data = {"actions": torch.zeros(steps, 8), "obs": {"datagen_info": {
            "eef_pose": {EEF: eef}, "target_eef_pose": {EEF: eef.clone()},
            "object_pose": {OBJECT: _pose([0.4 + 0.05 * index, 0.0, 0.02], 0.3 * index).repeat(steps, 1, 1),
                            CONTAINER: _pose([0.5, 0.2 + 0.05 * index, 0.0]).repeat(steps, 1, 1)},
            "subtask_term_signals": {GRASP_SIGNAL: torch.arange(steps) >= 15 + 2 * index}}}}
        asyncio.run(pool.add_episode(episode))
    return env, cfg, datagen.data_generator.DataGenerator(env=env, src_demo_datagen_info_pool=pool)


@contextlib.contextmanager
def _counting(datagen):
    """Count calls of every registered selection strategy."""
    registry = datagen.selection_strategy.REGISTERED_SELECTION_STRATEGIES
    calls, originals = [0], {name: cls.select_source_demo for name, cls in registry.items()}

    def wrap(function):
        def counted(self, *args, **kwargs):
            calls[0] += 1
            return function(self, *args, **kwargs)
        return counted

    for name, cls in registry.items():
        cls.select_source_demo = wrap(originals[name])
    try:
        yield calls
    finally:
        for name, cls in registry.items():
            cls.select_source_demo = originals[name]


def _trial(generator, seed=0, **options):
    """One trial through the production execute_trial; returns segments and executed waypoint poses."""
    poses = []

    def step(waypoint):
        poses.append(waypoint.pose.clone())
        return False

    with trajectory_rng(seed):
        boundaries = generator.randomize_subtask_boundaries()
    segments, _, steps, ended = execute_trial(generator, 2, boundaries, seed, step, **options)
    assert steps == len(poses) and not ended
    return segments, torch.stack(poses)


def _legacy_trial(generator, seed=0, fixed=None):
    """The trial loop before per-subtask selection, kept as the switch-off reference."""
    with trajectory_rng(seed):
        boundaries = generator.randomize_subtask_boundaries()
    selected, previous, poses = {EEF: fixed}, {EEF: None}, []
    for subtask in range(2):
        with trajectory_rng(seed + subtask * 100000):
            previous[EEF] = generator.generate_trajectory(0, EEF, subtask, boundaries, {}, selected, previous)
        poses += [waypoint.pose.clone() for waypoint in previous[EEF]]
    return int(selected[EEF]), torch.stack(poses)


def test_config_passes_the_switch_and_never_the_per_arm_flag(datagen):
    for switch in (False, True):
        _, cfg, _ = _generator(datagen, select_source_per_subtask=switch)
        assert cfg.datagen_config.generation_select_src_per_subtask is switch
        assert cfg.datagen_config.generation_select_src_per_arm is False


@pytest.mark.parametrize("strategy", ["random", "nearest_neighbor_object", "nearest_neighbor_robot_distance"])
def test_switch_off_matches_the_previous_loop(datagen, strategy):
    _, _, generator = _generator(datagen, selection_strategy=strategy)
    for seed in range(20):
        with _counting(datagen) as calls:
            segments, poses = _trial(generator, seed)
        parent, reference = _legacy_trial(generator, seed)
        assert calls[0] == 1
        assert [s["source"] for s in segments] == [parent, parent]
        assert torch.equal(poses, reference)


def test_switch_on_selects_once_per_subtask(datagen):
    _, _, generator = _generator(datagen, selection_strategy="test_always_different", select_source_per_subtask=True)
    for seed in range(10):
        with _counting(datagen) as calls:
            segments, _ = _trial(generator, seed, per_subtask=True)
        assert calls[0] == 2
        first, second = (s["source"] for s in segments)
        assert first != second


def test_clearing_without_the_config_flag_is_refused(datagen):
    # Isaac Lab skips selection unless the flag is set, then asserts on the cleared index.
    _, _, generator = _generator(datagen, select_source_per_subtask=False)
    with pytest.raises(AssertionError):
        _trial(generator, per_subtask=True)


def test_forced_source_applies_to_the_grasp_only_when_switched(datagen):
    _, _, off = _generator(datagen, selection_strategy="test_always_different")
    _, _, on = _generator(datagen, selection_strategy="test_always_different", select_source_per_subtask=True)
    for seed in range(10):
        segments, poses = _trial(off, seed, forced=2)
        assert [s["source"] for s in segments] == [2, 2]
        assert torch.equal(poses, _legacy_trial(off, seed, fixed=2)[1])
        with _counting(datagen) as calls:
            segments, _ = _trial(on, seed, forced=2, per_subtask=True)
        assert calls[0] == 1 and segments[0]["source"] == 2
        with _counting(datagen) as calls:
            segments, _ = _trial(on, seed, forced=2, later=0, per_subtask=True)
        assert calls[0] == 0 and [s["source"] for s in segments] == [2, 0]


def test_single_source_pool_is_unchanged_by_the_switch(datagen):
    _, _, off = _generator(datagen, sources=1)
    _, _, on = _generator(datagen, sources=1, select_source_per_subtask=True)
    for seed in range(5):
        assert torch.equal(_trial(off, seed)[1], _trial(on, seed, per_subtask=True)[1])


def test_lineage_names_the_segments_actually_used(datagen):
    env, _, generator = _generator(datagen, select_source_per_subtask=True)
    pool = generator.src_demo_datagen_info_pool
    sources = [{"episode": f"demo_{i}"} for i in range(4)]
    for grasp, placement in ((1, 3), (2, 2), (0, 1)):
        segments, poses = _trial(generator, 5, forced=grasp, later=placement, per_subtask=True)
        # The trial ends with the recorded placement parent's realized segment, transformed about the bin.
        start, end = segments[1]["boundary"]
        info = pool.datagen_infos[placement]
        expected = (env.objects[CONTAINER] @ torch.linalg.inv(info.object_poses[CONTAINER][start])
                    @ info.target_eef_pose[EEF][start:end])
        assert torch.allclose(poses[-len(expected):], expected, atol=1e-5)
        # The grasp segment's realized end is the placement segment's start, from the grasp parent's boundaries.
        assert segments[0]["boundary"][0] == 0 and segments[1]["boundary"][1] == len(info.target_eef_pose[EEF])
        lineage = trial_lineage(segments, sources)
        assert lineage["sources_by_subtask"] == [grasp, placement]
        assert lineage["source_references_by_subtask"] == [sources[grasp], sources[placement]]
        if grasp == placement:
            assert (lineage["source"], lineage["source_reference"]) == (grasp, sources[grasp])
        else:
            assert lineage["source"] is None and lineage["source_reference"] is None


def test_ended_episode_records_only_the_executed_segment(datagen):
    _, _, generator = _generator(datagen, select_source_per_subtask=True)
    with trajectory_rng(0):
        boundaries = generator.randomize_subtask_boundaries()
    calls = []
    segments, last, steps, ended = execute_trial(generator, 2, boundaries, 0, lambda w: calls.append(w) or len(calls) == 3,
                                                 forced=1, later=2, per_subtask=True)
    assert ended and steps == 3 and last is calls[-1]
    assert [s["subtask"] for s in segments] == [0]


def _inputs(tmp_path):
    (tmp_path / "omy_demos.hdf5").touch()
    (tmp_path / "env_cfg.json").touch()
    return ["--input", str(tmp_path)]


def test_target_manifest_arguments(tmp_path):
    parser = build_parser()
    manifest = tmp_path / "targets.json"
    manifest.write_text("{}")
    base = _inputs(tmp_path)
    for extra in (["--target-manifest", str(manifest)], ["--target-id", "t0"],
                  ["--target-manifest", str(manifest), "--target-id", "t0"],
                  ["--target-manifest", str(tmp_path / "missing.json"), "--target-id", "t0", "--no-randomize"]):
        with pytest.raises(SystemExit):
            validate_args(parser, parser.parse_args(base + extra))
    validate_args(parser, parser.parse_args(base + ["--target-manifest", str(manifest), "--target-id", "t0",
                                                    "--no-randomize"]))
    assert parser.parse_args(base + ["--select-source-per-subtask"]).select_source_per_subtask


def test_target_rows_are_exact_and_complete(tmp_path):
    manifest = tmp_path / "targets.json"
    pail = [0.5, 0.1, 0.03, 1, 0, 0, 0]
    rows = [{"id": "t0", "poses": {OBJECT: pail, CONTAINER: [0.6, -0.2, 0.0, 0.7071, 0, 0, 0.7071]}, "stratum": "explicit"},
            {"id": "t1", "poses": {OBJECT: pail}},
            {"id": "t2", "poses": {OBJECT: pail, CONTAINER: [0, 0, 0, 0, 0, 0, 0]}},
            {"id": "dup", "poses": {}}, {"id": "dup", "poses": {}}]
    manifest.write_text(json.dumps({"targets": rows}))
    poses, row = target_poses(manifest, "t0", (OBJECT, CONTAINER))
    assert poses[OBJECT] == pail and row["stratum"] == "explicit"
    for target in ("t1", "t2", "dup", "absent"):
        with pytest.raises(ValueError):
            target_poses(manifest, target, (OBJECT, CONTAINER))


def _generated_recording(path, parents_by_episode, loaded):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as stream:
        data = stream.create_group("data")
        data.attrs["robolab_omy_mimicgen"] = json.dumps({"sources": loaded})
        for name, parents in parents_by_episode.items():
            data.create_group(name).attrs["mimicgen_source"] = json.dumps({"parents": parents})


def test_pipeline_rejects_multisource_jobs_and_mixed_parents(tmp_path):
    from robolab.core.omy_teleop.pipeline import process_job, verify_lineage

    with pytest.raises(ValueError, match="Multisource"):
        process_job(tmp_path, {"id": "x", "sources": [{"file": "a"}, {"file": "b"}]}, 0, None)
    loaded = [{"file": str(tmp_path / "h" / "omy_demos.hdf5"), "episode": f"demo_{i}"} for i in range(2)]
    _generated_recording(tmp_path / "pool" / "a" / "omy_demos.hdf5", {"demo_0": [0, 1]}, loaded)
    row = {"kind": "generated", "file": "a/omy_demos.hdf5", "episode": "demo_0", "generation_source": None}
    with pytest.raises(ValueError, match="2 loaded sources"):
        verify_lineage(tmp_path / "pool", {"episodes": [row]}, set())
    # A row from a one-source file whose recorded parents still differ is refused too.
    _generated_recording(tmp_path / "pool" / "b" / "omy_demos.hdf5", {"demo_0": [0, 1]}, loaded[:1])
    row = {**row, "file": "b/omy_demos.hdf5", "generation_source": {"file": loaded[0]["file"], "episode": "demo_0"}}
    with pytest.raises(ValueError, match="mixed parents"):
        verify_lineage(tmp_path / "pool", {"episodes": [row]}, {(str(tmp_path / "h"), "demo_0")})
