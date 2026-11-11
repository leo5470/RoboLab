# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client for the PPS steering server — `Pi0DroidJointposClient` plus five keys.

The server is `scripts/serve_pps.py` in the PPS repo: the same `pi05_droid_jointpos` base,
converted to PyTorch, with `gamma*(v_task - v_ref)` added inside the denoising loop. It speaks
openpi's wire protocol, so this client differs from the pi0-family one only in what it puts *into*
the request.

**Why the indices are sent rather than counted server-side.** The statistical design compares a
steered arm against the base on identical trials: the sampler's noise is a pure function of
`(task, episode, env_id, chunk)`, so two runs made days apart draw the same epsilon at the same
point of the same trial and exact McNemar applies across them. A counter on the server would look
identical across arms and silently not be — one websocket reconnect (which `_infer_with_retry`
performs transparently) and the two arms diverge with nothing raised. Sending the indices makes
the pairing a property of the request instead of of the connection's history. A request that
arrives without them is served with fresh entropy and marked UNPAIRED in the server's log, rather
than being given a fabricated pairing.

The counters live here and are reset by `begin_episode`, so a mid-episode reconnect resumes at the
chunk index the episode is actually on.
"""

import logging
from collections import defaultdict

from policies.pi0_family.client import Pi0DroidJointposClient

logger = logging.getLogger(__name__)

# Must match `pps.droid.wire` in the PPS repo. Kept as literals rather than imported because
# RoboLab's environment does not have the PPS package installed — the two processes share only
# the wire format.
MODE = "pps/mode"
GAMMA = "pps/gamma"
ARM = "pps/arm"
TASK = "pps/task"
EPISODE = "pps/episode"
ENV_ID = "pps/env_id"
CHUNK = "pps/chunk"

MODES = ("none", "pps", "naive")


class PpsClient(Pi0DroidJointposClient):
    """`Pi0DroidJointposClient` that also tells the server how to steer this chunk."""

    def __init__(self, *args, mode: str = "none", gamma: float = 0.0, arm: str | None = None,
                 task: str | None = None, **kwargs):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        if mode != "none" and not arm:
            raise ValueError(f"--mode {mode} needs --arm: the server holds several proxy pairs.")
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.gamma = float(gamma)
        self.arm = arm
        # None means "use the instruction", which is per-task and stable across runs. The label
        # only salts the epsilon, so all it has to be is the same for every arm of one experiment.
        self.task_label = task
        self._chunk_counters: dict[int, int] = defaultdict(int)
        self._env_id: int | None = None
        print(f"[{self.__class__.__name__}] steering: mode={mode} gamma={gamma} arm={arm}")

    # ---- episode bookkeeping ------------------------------------------------

    def begin_episode(self, episode_idx: int) -> None:
        super().begin_episode(episode_idx)
        self._chunk_counters.clear()

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._chunk_counters.clear()
        else:
            self._chunk_counters.pop(env_id, None)
        super().reset(env_id=env_id)

    # ---- request assembly ---------------------------------------------------

    def infer(self, obs, instruction: str, *, env_id: int = 0) -> dict:
        # `_pack_request` does not receive env_id (the base class splits the observation and the
        # wire format across two hooks), so it is stashed here, immediately before the call that
        # consumes it. The default `infer_batch` is a serial loop over this method, so there is no
        # interleaving to worry about.
        self._env_id = env_id
        return super().infer(obs, instruction, env_id=env_id)

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        if self._env_id is None:
            raise RuntimeError(
                "PpsClient._pack_request was reached without an env id. Something called the "
                "packing hook outside infer(); the request would be unpaired."
            )
        env_id = self._env_id
        chunk = self._chunk_counters[env_id]
        # Incremented once per chunk FETCH, not per env step, and not on retry:
        # `_infer_with_retry` resends this same dict, and a retry must reuse the same epsilon.
        self._chunk_counters[env_id] = chunk + 1

        request = super()._pack_request(extracted_obs, instruction)
        request.update({
            MODE: self.mode,
            GAMMA: self.gamma,
            TASK: self.task_label or instruction,
            EPISODE: int(self._eval_episode_idx),
            ENV_ID: int(env_id),
            CHUNK: int(chunk),
        })
        if self.arm is not None:
            request[ARM] = self.arm
        return request
