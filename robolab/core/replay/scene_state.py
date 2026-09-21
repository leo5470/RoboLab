# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene-state restore and validation against recorded episodes.

``restore_recorded_initial_state`` puts the scene in the exact state a
recording started from, so an open-loop action replay evolves the same way the
recording did. ``restore_scene_state`` is the lower-level building block: it
writes one recorded state row straight into the simulator without touching
episode counters, managers, or recorders — what offline rendering of a recorded
trajectory needs (see :mod:`robolab.core.replay.materialize`).
``StateValidator`` measures how closely a replay tracks the recorded per-step
states — a debug tool that turns "the replay diverged" from a guess into a
measurement.
"""

import numpy as np
import torch

from robolab.core.utils.file_utils import load_hdf5_initial_state, load_hdf5_states


def overlay_state_row(current: dict, recorded: dict, row: int = 0, *, num_envs: int = 1,
                      device: str = "cpu", _path: str = "") -> list[str]:
    """Write row ``row`` of a recorded state tree over ``current``, in place.

    Both trees follow the ``InteractiveScene.get_state()`` layout. Recorded
    leaves carry a leading axis (one row per recorded step, or a single row for
    ``initial_state``); the selected row is tiled across ``num_envs`` so every
    env is put in the state the recording was made against. Recorded keys that
    the current scene does not have (e.g. the ``cameras`` block the initial-state
    recorder adds, which is not a scene asset) are left out and reported.

    Args:
        current: The env's current state tree, mutated in place.
        recorded: The recorded state tree (numpy arrays or tensors).
        row: Index along each recorded leaf's leading axis.
        num_envs: Number of envs to tile the recorded row across.
        device: Device to place the resulting tensors on.

    Returns:
        Paths of recorded entries that were skipped because the scene has no
        such key (empty when the recording matches the scene exactly).
    """
    skipped: list[str] = []
    for key, value in recorded.items():
        path = f"{_path}/{key}" if _path else key
        if key not in current:
            skipped.append(path)
            continue
        if isinstance(value, dict):
            skipped.extend(
                overlay_state_row(current[key], value, row, num_envs=num_envs, device=device, _path=path)
            )
        else:
            selected = torch.as_tensor(np.asarray(value[row:row + 1]), device=device)
            current[key] = selected.repeat(num_envs, *([1] * (selected.ndim - 1)))
    return skipped


def restore_scene_state(env, recorded_state: dict, row: int = 0) -> list[str]:
    """Put the scene in a recorded state without disturbing episode state.

    This is the lowest-level restore RoboLab offers: it overlays the recorded
    row onto the env's current full state (``InteractiveScene.reset_to``
    requires an entry for every scene asset, while the recorder only saves
    dynamic ones) and writes it straight to the simulator. Unlike
    ``env.reset_to()`` it does not touch the episode length buffer, the
    managers, or the recorder terms, so it is safe to call once per frame while
    walking a recorded trajectory.

    Callers that need the change reflected in rendered images must follow this
    with a simulator sync and render (see
    :func:`robolab.core.replay.materialize.render_observations`).

    Args:
        env: The environment whose scene is restored.
        recorded_state: A recorded state tree in ``InteractiveScene.get_state()``
            layout with env-relative poses.
        row: Index along each recorded leaf's leading axis.

    Returns:
        Paths of recorded entries the scene has no key for.
    """
    state = env.scene.get_state(is_relative=True)
    skipped = overlay_state_row(state, recorded_state, row, num_envs=env.num_envs, device=env.device)
    env.scene.reset_to(state, env_ids=None, is_relative=True)
    return skipped


def restore_recorded_initial_state(env, hdf5_path: str, episode: int) -> None:
    """Reset ``env`` to the initial scene state recorded for ``episode``.

    The recorded ``initial_state`` follows the ``InteractiveScene.get_state()``
    layout with env-relative poses. Row 0 is tiled across ``env.num_envs`` so
    every env replays from the exact state the actions were recorded against.
    The recorded state is overlaid onto the env's current full state because
    ``InteractiveScene.reset_to`` requires an entry for every scene asset,
    while the recorder only saves dynamic assets (e.g. no static table).
    """
    try:
        recorded_state = load_hdf5_initial_state(hdf5_path, episode)
    except ValueError as err:
        print(f"WARNING: no recorded initial state to restore ({err}); "
              "replaying from default reset state, which may diverge from the recording.")
        return
    state = env.scene.get_state(is_relative=True)
    skipped = overlay_state_row(state, recorded_state, 0, num_envs=env.num_envs, device=env.device)
    for path in skipped:
        print(f"WARNING: recorded initial state has '{path}' which is not in the scene; skipping.")
    env.reset_to(state, env_ids=None, is_relative=True)


def _flatten_state_tree(tree: dict, prefix: str = "") -> dict:
    """Flatten a nested ``InteractiveScene.get_state()``-style dict into ``{"a/b/c": leaf}``."""
    flat = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_state_tree(value, path))
        else:
            flat[path] = value
    return flat


class StateValidator:
    """Compare per-step sim state against the recorded ``states`` group (debug tool).

    Recorded states are post-step snapshots in the ``InteractiveScene.get_state()``
    layout, one row per step. Comparison uses env 0 only, matching the
    single-env record/replay recipe. Quaternion sign flips (q vs -q) are not
    normalized, so a reported pose drift of ~2.0 on an otherwise tracking
    replay usually means a sign flip, not a real divergence.
    """

    def __init__(self, hdf5_path: str, episode: int, tolerance: float = 0.01):
        self.tolerance = tolerance
        self.recorded = _flatten_state_tree(load_hdf5_states(hdf5_path, episode))
        self.num_steps = min(leaf.shape[0] for leaf in self.recorded.values())
        self.max_drift = 0.0
        self.max_drift_step = None
        self.max_drift_path = None
        self.first_exceed_step = None
        self.paths_over_tolerance = set()

    def check_step(self, env, step: int) -> None:
        if step >= self.num_steps:
            return
        current = _flatten_state_tree(env.scene.get_state(is_relative=True))
        for path, series in self.recorded.items():
            if path not in current:
                continue
            simulated = current[path][0].detach().cpu().numpy()
            drift = float(np.max(np.abs(simulated - series[step])))
            if self.max_drift_path is None or drift > self.max_drift:
                self.max_drift, self.max_drift_step, self.max_drift_path = drift, step, path
            if drift > self.tolerance:
                self.paths_over_tolerance.add(path)
                if self.first_exceed_step is None:
                    self.first_exceed_step = step
                    print(f"\033[93mSTATE VALIDATION: drift first exceeded tolerance {self.tolerance} "
                          f"at step {step} ({path}: {drift:.4f}).\033[0m")

    def report(self) -> None:
        if self.first_exceed_step is None:
            print(f"STATE VALIDATION: replay tracked the recording over {self.num_steps} steps; "
                  f"max drift {self.max_drift:.4f} on {self.max_drift_path} (tolerance {self.tolerance}).")
        else:
            over = sorted(self.paths_over_tolerance)
            print(f"\033[93mSTATE VALIDATION: replay diverged from the recording. Max drift "
                  f"{self.max_drift:.4f} at step {self.max_drift_step} on {self.max_drift_path}; "
                  f"first exceeded tolerance {self.tolerance} at step {self.first_exceed_step}. "
                  f"Fields over tolerance ({len(over)}/{len(self.recorded)}): {', '.join(over)}\033[0m")
