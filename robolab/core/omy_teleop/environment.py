"""OMY environment with an opt-in Isaac Lab 2.2 manager stepping contract."""

import logging

import torch

from robolab.core.environments.env import RobolabEnv
from robolab.core.world.world_state import get_world

from .boundary import boundary_queries
from .recording import OmyRecorderManager
from .stepping import advance_physics, eligibility_reasons, runtime_reasons

logger = logging.getLogger(__name__)


class OmyEnv(RobolabEnv):
    recorder_manager_type = OmyRecorderManager

    def __init__(self, cfg, **kwargs):
        cfg.omy_account_early_terminations = True
        super().__init__(cfg, **kwargs)
        requested = getattr(cfg, "omy_stepping_profile", "ordinary")
        compact = getattr(cfg, "omy_recording_profile", "full") == "compact"
        self._omy_audit = eligibility_reasons(self) if requested == "held-target" or compact else []
        self._omy_compact_step = compact and not self._omy_audit
        self._omy_held_target = requested == "held-target" and not self._omy_audit
        cfg.omy_stepping_effective = "held-target" if self._omy_held_target else "ordinary"
        cfg.omy_stepping_fallback_reasons = self._omy_audit if requested == "held-target" else []
        cfg.omy_skip_duplicate_observations = self._omy_compact_step
        if requested == "held-target" and self._omy_audit:
            logger.warning("OMY held-target fallback: %s", "; ".join(self._omy_audit))

    def step(self, action):
        if self.all_terminated:
            # A caller cannot admit another command to a finalized single-env episode.
            raise RuntimeError("OMY episode is finalized; reset and arm the next attempt before stepping")
        if not self._omy_held_target and not self._omy_compact_step:
            return super().step(action)
        rejected = runtime_reasons(self)
        if rejected:
            # Switch the entire command to ordinary stepping before action processing.
            self._omy_held_target = self._omy_compact_step = False
            self.cfg.omy_stepping_effective = "ordinary"
            self.cfg.omy_stepping_fallback_reasons = rejected
            self.cfg.omy_skip_duplicate_observations = False
            logger.warning("OMY stepping fallback: %s", "; ".join(rejected))
            return super().step(action)
        self._has_stepped = True
        self._pre_step_frozen = self._frozen_envs.clone()
        world = get_world(self)
        world.clear_contact_cache()
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        advance_physics(self, held_target=self._omy_held_target)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        with boundary_queries(world):
            self.reset_buf = self.termination_manager.compute()
            self.reset_terminated = self.termination_manager.terminated
            self.reset_time_outs = self.termination_manager.time_outs
            self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
            # Compact terms record full states and task events directly. Their
            # pre-action obs comes from the previous boundary. Reuse read-only
            # world queries, while stepping each task state machine exactly once.
            if self.recorder_manager.active_terms:
                self.recorder_manager.record_post_step()
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids):
            self.recorder_manager.record_pre_reset(reset_ids)
            self._reset_idx(reset_ids)
            # Every terminal OMY env is frozen; no teleport or second target
            # submission is necessary on this command's finalization path.
            self.recorder_manager.record_post_reset(reset_ids)
        self.command_manager.compute(dt=self.step_dt)
        # Eligibility excludes interval events; no hook is omitted for admitted scenes.
        self.obs_buf = self.observation_manager.compute(update_history=True)
        result = self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
        return result

    def invalidate_omy_state(self):
        """Invalidate state-dependent caches even when restoring at the same step."""
        world = get_world(self)
        world.clear_contact_cache()
        world.reset_predicate_state(torch.arange(self.num_envs, device=self.device))
        self.scene.update(dt=0.0)

    def reset(self, *args, **kwargs):
        self._finish_pending()
        self.reset_eval_state()
        # Pending data was finalized above; isolate the next episode buffer.
        self.recorder_manager.clear()
        self.invalidate_omy_state()
        result = super().reset(*args, **kwargs)
        self.invalidate_omy_state()
        return result

    def reset_to(self, *args, **kwargs):
        self._finish_pending()
        self.reset_eval_state()
        self.recorder_manager.clear()
        super().reset_to(*args, **kwargs)
        self.invalidate_omy_state()
        # Upstream resets recorder terms before applying the restored state.
        # Replace that initial snapshot with the state actually restored.
        self.recorder_manager.clear()
        self.recorder_manager.reset()
        self.recorder_manager.record_post_reset(None)
        self.obs_buf = self.observation_manager.compute(update_history=False)
        return self.obs_buf, self.extras

    def _episode_succeeded(self, env_id):
        terms = self.termination_manager
        names = [name for name in terms.active_terms if name == "success" or name.startswith("success_")]
        if not names:
            return False
        return torch.stack([terms.get_term(name)[env_id] for name in names]).any()

    def _finish_pending(self):
        recorder = self.recorder_manager
        pending = [eid for eid, index in getattr(recorder, "_current_episode_index", {}).items() if index is not None]
        if pending:
            recorder.set_success_to_episodes(pending, torch.zeros(len(pending), dtype=torch.bool, device=self.device))
            recorder.export_episodes(pending)
