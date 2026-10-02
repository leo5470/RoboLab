"""OMY compact raw contract, using the ordinary recorder lifecycle and HDF5 writer."""

import math
import time

import torch
from isaaclab.managers import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData

from robolab.core.events.basic_recorders import (
    InitialStateRecorderCfg,
    PostStepStatesRecorderCfg,
    PreStepActionsRecorderCfg,
    PreStepFlatPolicyObservationsRecorderCfg,
)
from robolab.core.events.subtask_recorder import SubtaskCompletionRecorderCfg
from robolab.core.logging.recorder_manager import RobolabRecorderManager

from .buffers import EpisodeStorage


def admitted_steps(args, step_dt):
    limits = [math.ceil(args.episode_length_s / step_dt)]
    if hasattr(args, "max_attempt_steps"):
        limits.extend((args.max_attempt_steps, math.ceil(args.max_attempt_seconds / step_dt)))
    if args.steps:
        limits.append(args.steps)
    return min(limits)


class BoundaryTimeRecorder(RecorderTerm):
    def record_pre_step(self):
        sample = getattr(self._env, "omy_input_sample", None) or {}
        valid = not sample.get("stale", True)
        # CPU timing leaves are owned snapshots; missing samples use explicit
        # validity flags and finite sentinels rather than NaN training data.
        return "timing", {
            "sim_before_s": torch.tensor([self._env._sim_step_counter * self._env.physics_dt], dtype=torch.float64),
            "pre_monotonic_ns": torch.tensor([time.monotonic_ns()], dtype=torch.int64),
            "source_valid": torch.tensor([valid]),
            "source_wall_ns": torch.tensor([sample.get("source_wall_ns") or -1], dtype=torch.int64),
            "source_read_monotonic_ns": torch.tensor([sample.get("read_monotonic_ns", -1)], dtype=torch.int64),
            "source_age_s": torch.tensor([sample.get("age_s") if valid else -1.0], dtype=torch.float64),
            "held_target": torch.tensor([getattr(self._env, "_omy_held_target", False)]),
            "position_iteration_cap": torch.tensor([self._env.cfg.sim.physx.max_position_iteration_count]),
        }

    def record_post_step(self):
        return "timing", {
            "sim_after_s": torch.tensor([self._env._sim_step_counter * self._env.physics_dt], dtype=torch.float64),
            "post_monotonic_ns": torch.tensor([time.monotonic_ns()], dtype=torch.int64),
        }


@configclass
class BoundaryTimeRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = BoundaryTimeRecorder


@configclass
class CompactRecorderCfg(RecorderManagerBaseCfg):
    record_initial_state = InitialStateRecorderCfg()
    record_actions = PreStepActionsRecorderCfg()
    record_states = PostStepStatesRecorderCfg()
    record_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()
    record_subtask_completion = None
    record_timing = BoundaryTimeRecorderCfg()
    export_in_record_pre_reset = False


def configure_compact_recording(cfg, args):
    """Replace only this copied OMY config, retaining task/event recording."""
    previous = cfg.recorders
    cfg.recorders = CompactRecorderCfg(
        dataset_export_dir_path=previous.dataset_export_dir_path,
        dataset_filename=previous.dataset_filename,
        dataset_export_mode=previous.dataset_export_mode,
        record_subtask_completion=(SubtaskCompletionRecorderCfg()
                                   if getattr(previous, "record_subtask_completion", None) is not None else None),
    )
    proprio = cfg.observations.proprio_obs
    for name in ("ee_pos", "ee_quat", "eef_pos", "eef_quat"):
        setattr(proprio, name, None)
    cfg.omy_episode_max_steps = admitted_steps(args, cfg.sim.dt * cfg.decimation)


class CompactEpisodeData(EpisodeData):
    def __init__(self, max_steps, max_bytes):
        super().__init__()
        self.buffer = EpisodeStorage(max_steps, max_bytes)
        self._data = self.buffer.data

    def add(self, key, value):
        self.buffer.add(key, value)


class OmyRecorderManager(RobolabRecorderManager):
    """Fixed-capacity episodes when requested; unchanged full recorder otherwise."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.compact = getattr(env.cfg, "omy_recording_profile", "full") == "compact"
        self.export_timings = []
        if self.compact:
            self._episodes = {eid: self._new_episode() for eid in range(env.num_envs)}

    def _new_episode(self):
        if getattr(self, "compact", False):
            return CompactEpisodeData(self._env.cfg.omy_episode_max_steps, self._env.cfg.omy_episode_buffer_bytes)
        return super()._new_episode()

    def set_buffered_recording(self):
        if not self.compact:
            super().set_buffered_recording()

    def set_episode_index(self, episode_index, env_ids=None):
        if self.compact:
            # Reserve all steady-state leaves while still in READY. Reset has
            # already recorded initial_state; no allocation is needed per frame.
            for eid, episode in self._episodes.items():
                if env_ids is not None and eid not in env_ids:
                    continue
                episode.buffer.reserve("actions", self._env.action_manager.action[eid])
                state = self._env.scene.get_state(is_relative=True)
                def reserve_tree(key, tree):
                    for name, leaf in tree.items():
                        if isinstance(leaf, dict):
                            reserve_tree(f"{key}/{name}", leaf)
                        else:
                            episode.buffer.reserve(f"{key}/{name}", leaf[eid])
                reserve_tree("states", state)
                reserve_tree("obs", self._env.obs_buf)
                for term in self._terms.values():
                    if isinstance(term, BoundaryTimeRecorder):
                        for key, tree in (term.record_pre_step(), term.record_post_step()):
                            reserve_tree(key, tree)
                if "record_subtask_completion" in self._terms:
                    for name, dtype in (("status", torch.uint16), ("completed", torch.uint8), ("score", torch.float32)):
                        episode.buffer.reserve(f"subtask/{name}", torch.tensor(0, dtype=dtype, device=self._env.device))
        super().set_episode_index(episode_index, env_ids)

    def export_episodes(self, env_ids=None):
        pending = any(value is not None for value in getattr(self, "_current_episode_index", {}).values())
        if not pending:
            return
        allocated = sum(getattr(getattr(ep, "buffer", None), "allocated_bytes", 0) for ep in self._episodes.values())
        started = time.perf_counter()
        super().export_episodes(env_ids)
        self.export_timings.append({"allocated_bytes": allocated, "export_ms": 1000 * (time.perf_counter() - started)})
        # One record per bounded attempt; the collector persists these at boundaries.
        self.export_timings[:] = self.export_timings[-1:]
